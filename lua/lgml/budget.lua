--- Exact memory accounting of a model in its worker.
-- The parts are the ggml buffers of lgml (see `lgml.memory`), the KV
-- cache at full size, the worst-case compute buffers, and the driver
-- overhead: the DRM fdinfo less the buffers on Linux, else `OVERHEAD`.
-- An allocation that does not fit raises an error that starts with
-- `PREFIX` (see `lgml.memory_guard`); requests then get `STATUS`.
-- @module lgml.budget
local uv = require("luv")
local lgml = require("lgml")

local M = {}

local MiB = 1048576

--- The start of the message of an allocation that does not fit.
M.PREFIX = "memory budget:"

--- The HTTP status of a request or load that does not fit: 507.
M.STATUS = 507

--- The least driver overhead of a GPU worker: 256 MiB. The fdinfo at
-- load does not see the buffers that a backend makes at its first
-- compute, such as the Vulkan staging and dequant buffers.
M.OVERHEAD = 256 * MiB

--- The device bytes that stay free when a worker allocates: 256 MiB.
-- `lgml-worker --mem-margin` changes it.
M.MARGIN = 256 * MiB

--- True when err is an allocation that did not fit.
-- @param err an error value
-- @treturn boolean
function M.is_budget(err)
	return tostring(err):find(M.PREFIX, 1, true) ~= nil
end

--- The HTTP status of an error: `STATUS` for an allocation that did
-- not fit, else def.
-- @param err an error value
-- @tparam integer def the status of other errors
-- @treturn integer the status
function M.status(err, def)
	return M.is_budget(err) and M.STATUS or def
end

-- Bytes of an fdinfo value such as "12 KiB", "3 MiB" or "4096".
local function bytes(v)
	local n, unit = v:match("^%s*(%d+)%s*(%a*)")
	n = tonumber(n)
	if not n then
		return nil
	end
	local scale = { [""] = 1, B = 1, KiB = 1024, MiB = MiB, GiB = 1024 * MiB }
	return scale[unit] and n * scale[unit] or nil
end

--- The device memory of the DRM clients of a process, from fdinfo.
-- Each client counts once, when its `drm-pdev` is id or id is nil. The
-- keys are `drm-total-<region>`, else the amdgpu `drm-memory-<region>`.
-- With gtt true, the gtt region counts too: an integrated GPU uses it.
-- @tparam[opt] string id the bus id of the device (see `lgml.devices`)
-- @tparam[opt] boolean gtt true to add the gtt region
-- @tparam[opt] integer|string pid the process (default "self"), or a
-- directory of fdinfo files
-- @treturn ?integer the bytes, or nil when no DRM client has them
function M.drm(id, gtt, pid)
	local dir = tostring(pid or "self")
	if dir:sub(1, 1) ~= "/" then
		dir = ("/proc/%s/fdinfo"):format(dir)
	end
	local h = uv.fs_scandir(dir)
	if not h then
		return nil
	end
	local seen, total, found = {}, 0, false
	while true do
		local name = uv.fs_scandir_next(h)
		if not name then
			break
		end
		local f = io.open(dir .. "/" .. name)
		local s = f and f:read("a")
		if f then
			f:close()
		end
		local client = s and s:match("drm%-client%-id:%s*(%d+)")
		local pdev = s and s:match("drm%-pdev:%s*(%S+)")
		if client and not seen[client] and (not id or pdev == id) then
			seen[client] = true
			for _, region in ipairs(gtt and { "vram", "gtt" } or { "vram" }) do
				local v = s:match("drm%-total%-" .. region .. ":%s*([^\n]+)") or
				    s:match("drm%-memory%-" .. region .. ":%s*([^\n]+)")
				local n = v and bytes(v)
				if n then
					total, found = total + n, true
				end
			end
		end
	end
	return found and total or nil
end

-- The ggml device entry of a backend name.
local function device(name)
	for _, d in ipairs(lgml.devices()) do
		if d.name == name then
			return d
		end
	end
end

--- The free bytes of a device, as ggml reports them.
-- @tparam string name the backend name, such as "Vulkan0"
-- @treturn ?integer the bytes, or nil for an unknown device
function M.free(name)
	local d = device(name)
	return d and d.free
end

--- Wait until a device reports at least want free bytes, at most ms.
-- The kernel gives freed device memory back some time after the free,
-- so a check at once sees too little.
-- @tparam string name the backend name
-- @tparam integer want the free bytes to wait for
-- @tparam[opt=3000] integer ms the most time to wait
function M.settle(name, want, ms)
	local uv = require("luv")
	local t = uv.hrtime()
	while (M.free(name) or want) < want and uv.hrtime() - t < (ms or 3000) * 1e6 do
		uv.sleep(20)
	end
end

-- The device of the best backend: the first GPU, else the first
-- integrated GPU, else nil.
local function best()
	local igpu
	for _, d in ipairs(lgml.devices()) do
		if d.type == "gpu" then
			return d
		end
		igpu = igpu or (d.type == "igpu" and d) or nil
	end
	return igpu
end

--- The memory report of a loaded engine.
-- It calls `engine:memory()` when the engine has it; that call can
-- plan worst-case graphs. The parts are for the memory of the backend
-- of the engine: its device, or "CPU" for a CPU backend.
-- @param engine the engine
-- @tparam[opt] string backend the backend name, when the engine has no
-- `memory` method
-- @treturn table `{ memory, used, weights, cache, compute, overhead,
-- now, reserved, drm, host }`: memory is the name of the memory; used
-- is the sum of the four parts; now the bytes the process holds on it,
-- so used - now is what the model can still take; reserved is true when
-- the engine planned its worst-case graphs; drm the fdinfo bytes; host
-- is `{ weights, tensors, compute }` in host memory for a GPU backend
function M.report(engine, backend)
	local em = engine.memory and engine:memory() or {}
	local name = em.backend and em.backend:name() or backend
	local d = name and device(name) or not name and best()
	local gpu = d and d.type ~= "cpu"
	local key = gpu and d.name or "CPU"
	local all = lgml.memory()
	local k = all[key] or { weights = 0, tensors = 0, compute = 0, peak_compute = 0 }
	local cache_now = math.min(em.cache_now or 0, k.tensors)
	local r = {
		memory = key, reserved = em.compute ~= nil,
		weights = k.weights + k.tensors - cache_now,
		cache = math.max(em.cache or 0, cache_now),
		compute = math.max(k.peak_compute, em.compute or 0),
	}
	local held = k.weights + k.tensors + k.compute
	local drm = gpu and M.drm(d.id, d.type == "igpu")
	local seen = drm and math.max(0, drm - held) or 0
	r.overhead = gpu and math.max(seen, M.OVERHEAD) or 0
	r.used = r.weights + r.cache + r.compute + r.overhead
	-- The delta of the load holds the overhead that fdinfo sees.
	r.now = held + (drm and seen or r.overhead)
	r.drm = drm
	local h = gpu and all.CPU
	if h then
		r.host = { weights = h.weights, tensors = h.tensors, compute = h.peak_compute }
	end
	r.backend = em.backend
	return r
end

--- Check that the rest of a model fits: the bytes of a report that the
-- process does not hold yet, such as the KV cache growth and the
-- compute buffers, must fit the memory guard.
-- @tparam table r a report from `report`
-- @tparam string what the model name for the message
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string the error, which starts with `PREFIX`
function M.check(r, what)
	local rest = r.used - r.now
	if rest <= 0 or not r.backend then
		return true
	end
	return lgml.memory_check(r.backend, rest, what)
end

--- A report for a log line or `HELLO`: the parts without the backend.
-- @tparam table r a report from `report`
-- @treturn table the fields of r that are numbers, strings, booleans
-- or host
function M.plain(r)
	local t = {}
	for k, v in pairs(r) do
		if type(v) ~= "userdata" then
			t[k] = v
		end
	end
	return t
end

return M
