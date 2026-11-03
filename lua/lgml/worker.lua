--- Worker process: owns one device and runs engine jobs for the front.
-- Each job runs in its own coroutine. The engine yields between steps,
-- so `JOB` and `CANCEL` messages arrive while other jobs run. The
-- messages are in `lgml.ipc`. A chat job gets a sink that sends each
-- piece of output as a `DELTA` message.
-- Each job coroutine gets the request id of its message (see
-- `lgml.log.bind`), so the log lines of the job carry it.
-- @module lgml.worker
local uv = require("luv")
local ipc = require("lgml.ipc")
local log = require("lgml.log")

local L = log.new("worker")

local M = {}

-- A sink that sends each delta to the front. Tool call indexes are
-- local to the job; the front maps them to its own.
local function proxy(ch, id)
	local s = { ncalls = 0, stopped = false }

	local function delta(v)
		if s.stopped then
			return nil, "cancelled"
		end
		if not ch:send(ipc.DELTA, id, v) then
			s.stopped = true
			return nil, "cancelled"
		end
		return true
	end

	function s.reasoning(_, text)
		return delta({ k = "reasoning", text = text })
	end

	function s.content(_, text, lp)
		return delta({ k = "content", text = text, lp = lp })
	end

	function s.tool_call(_, name, cid)
		local index = s.ncalls
		s.ncalls = index + 1
		if not delta({ k = "tool_call", name = name, id = cid, index = index }) then
			return nil, "cancelled"
		end
		return index
	end

	function s.tool_arguments(_, index, text)
		return delta({ k = "tool_args", index = index, text = text })
	end

	function s.prompt(_, n, cached)
		return delta({ k = "prompt", n = n, cached = cached })
	end

	function s.progress(_, p)
		return delta({ k = "progress", p = p })
	end

	function s.finish(_, reason, usage, timings)
		s.done = { reason = reason, usage = usage, timings = timings }
	end

	function s.cancelled()
		return s.stopped
	end

	return s
end

-- Binds the request id of a job message to the running coroutine and
-- logs the job start. Returns the start time.
local function begin(kind, id, req)
	log.bind(req.rid)
	req.rid = nil
	L:debug("job", "kind", kind, "job", id)
	return uv.hrtime()
end

local function finish(kind, id, t0, ok, msg, status)
	if ok then
		L:debug("job done", "kind", kind, "job", id, "ms", (uv.hrtime() - t0) / 1e6)
	else
		L:debug("job failed", "kind", kind, "job", id, "ms", (uv.hrtime() - t0) / 1e6,
		    "status", status, "err", msg)
	end
end

-- The memory stats of the worker after a job: its resident and
-- swapped bytes, and the fill of the prompt cache RAM store.
local function stats(engine)
	local r = {}
	local f = io.open("/proc/self/status")
	if f then
		for line in f:lines() do
			local k, v = line:match("^(Vm%a+):%s+(%d+) kB")
			if k == "VmRSS" then
				r.rss = tonumber(v) * 1024
			elseif k == "VmSwap" then
				r.swap = tonumber(v) * 1024
			end
		end
		f:close()
	end
	local pc = engine.pc
	if pc and pc.bytes then
		local n = #(pc.store or {}) + #(pc.states or {})
		for _ in pairs(pc.blocks or {}) do
			n = n + 1
		end
		r.store_bytes, r.store_items = pc.bytes, n
	end
	local xc = engine.model and engine.model.xcache
	if xc then
		r.moe_cache_hits, r.moe_cache_misses = xc:stats()
		r.moe_cache_moved = xc.moved
	end
	return r
end

local function caps(engine)
	return {
		chat = engine.chat ~= nil, transcribe = engine.transcribe ~= nil,
		speak = engine.speak ~= nil, decide = engine.decide ~= nil,
		generate = engine.generate ~= nil, embed = engine.embed ~= nil,
	}
end

--- Tells the front that the engine did not load.
-- It sends `FAIL` with msg and waits until the message is out.
-- @tparam int fd the imsg socket to the front
-- @tparam string msg the error message
function M.fail(fd, msg)
	local ch = ipc.channel(fd, function() end)
	ch:send(ipc.FAIL, 0, { msg = msg })
	while ch.buf and ch.buf:queuelen() > 0 do
		uv.run("once")
	end
	ch:close()
end

--- The bytes in use on the device of a backend: total minus free.
-- Other processes change it too.
-- @tparam[opt] string backend the backend name (default: the first
-- GPU, else the first integrated GPU)
-- @treturn ?int the bytes, or nil for a CPU backend or no such device
function M.device_used(backend)
	local ok, lgml = pcall(require, "lgml")
	if not ok then
		return nil
	end
	local pick, gpu, igpu
	for _, d in ipairs(lgml.devices()) do
		if d.name == backend then
			pick = d
		end
		gpu = gpu or (d.type == "gpu" and d) or nil
		igpu = igpu or (d.type == "igpu" and d) or nil
	end
	if not backend then
		pick = gpu or igpu
	end
	if not pick or pick.type == "cpu" or pick.total == 0 then
		return nil
	end
	return pick.total - pick.free
end

--- Measures the memory of a loaded engine, for `HELLO`.
-- The parts come from `lgml.budget.report`, with `exact` true. With
-- before, the field delta is the device bytes in use now less before.
-- @param engine the engine
-- @tparam[opt] string backend the backend name
-- @tparam[opt] integer before `device_used` before the load
-- @treturn table the `mem` field of `HELLO`
-- @treturn table the report, for `fits`
function M.measure(engine, backend, before)
	local budget = require("lgml.budget")
	local r = budget.report(engine, backend)
	local mem = budget.plain(r)
	mem.exact = true
	local after = before and M.device_used(backend)
	if after then
		mem.delta = after - before
	end
	return mem, r
end

--- Checks that the rest of a model fits the memory guard.
-- See `lgml.budget.check`.
-- @tparam table r the report from `measure`
-- @tparam string name the model name for the message
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string the error
function M.fits(r, name)
	return require("lgml.budget").check(r, "model " .. tostring(name))
end

-- Options that do not change the KV cells of a prompt.
local NOT_STATE = { fd = true, ["state-fd"] = true, ["log-level"] = true, ["log-file"] = true,
	["dump-prompts"] = true, ["mem-margin"] = true, ["mem-limit"] = true }
-- Modules whose code computes the KV cells. A change in them, or in
-- the C module, makes a saved prompt store invalid.
local FORWARD = { "^lgml%.transformer$", "^lgml%.kv$", "^lgml%.kvpool$", "^lgml%.qwen4exp$",
	"^lgml%.replay$", "^lgml%.lora$", "^lgml%.spec$", "^lgml%.prompt_cache$", "^lgml%.xcache$",
	"^lgml%.vision" }

local function file_id(path)
	local st = path and uv.fs_stat(path)
	return st and ("%s %d %d.%09d"):format(path, st.size, st.mtime.sec, st.mtime.nsec) or tostring(path)
end

local function read_file(path)
	local f = path and io.open(path, "rb")
	if not f then
		return ""
	end
	local s = f:read("a")
	f:close()
	return s
end

--- The fingerprint of a saved prompt store (see `lgml.ipc.STATE`).
-- It covers the worker options, the size and time of the model files,
-- and the code that computes the KV cells. Call it after the engine
-- loads, so its modules are loaded.
-- @tparam table opts the options of `lgml-worker`, from `lgml.getopt`
-- @treturn string the fingerprint
function M.fingerprint(opts)
	local lgml = require("lgml")
	local parts, keys = {}, {}
	for k in pairs(opts) do
		if not NOT_STATE[k] then
			keys[#keys + 1] = k
		end
	end
	table.sort(keys)
	for _, k in ipairs(keys) do
		local v = opts[k]
		parts[#parts + 1] = k .. "=" .. (type(v) == "table" and table.concat(v, "\0") or tostring(v))
	end
	local files = { opts.m, opts.mmproj, opts["spec-mtp"] }
	for _, spec in ipairs(opts.lora or {}) do
		files[#files + 1] = spec:match("^(.*):[%d.eE+-]+$") or spec
	end
	for _, p in ipairs(files) do
		parts[#parts + 1] = file_id(p)
	end
	local mods = {}
	for name in pairs(package.loaded) do
		for _, pat in ipairs(FORWARD) do
			if type(name) == "string" and name:match(pat) then
				mods[#mods + 1] = name
				break
			end
		end
	end
	table.sort(mods)
	for _, name in ipairs(mods) do
		parts[#parts + 1] = name .. " " .. lgml.sha256(read_file(package.searchpath(name, package.path)))
	end
	parts[#parts + 1] = "lgml " .. lgml.sha256(read_file(package.searchpath("lgml", package.cpath)))
	return lgml.sha256(table.concat(parts, "\n"))
end

-- On SIGTERM: save the prompt store to a memory file, send it to the
-- front, and end by the signal once the message is out.
local function on_term(ch, engine, state, sig)
	local ok, lgml = pcall(require, "lgml")
	if state and engine.save_state and ok and lgml.memfd and ch.buf then
		local t0 = uv.hrtime()
		local fd, err = lgml.memfd("lgml-pc-" .. state.name)
		local bytes
		if fd then
			local good, n, e = pcall(engine.save_state, engine, fd, state.fp)
			if good and n then
				bytes = n
			else
				err = good and e or n
				uv.fs_close(fd)
				fd = nil
			end
		end
		if fd then
			ch:send(ipc.STATE, 0, { bytes = bytes }, fd)
			L:info("prompt store saved", "mib", bytes // 1048576, "ms", (uv.hrtime() - t0) // 1000000)
		elseif err ~= "no prompt cache" then
			L:warn("prompt store not saved", "err", err)
		end
	end
	local t, left = uv.new_timer(), 300
	t:start(0, 10, function()
		left = left - 1
		if not ch.buf or ch.buf:queuelen() == 0 or left <= 0 then
			t:stop()
			sig:stop()
			-- The end by the signal skips the exit hooks of luacov.
			local cov = package.loaded["luacov.runner"]
			if cov and cov.initialized then
				cov.shutdown()
			end
			uv.kill(uv.os_getpid(), "sigterm")
		end
	end)
end

--- Serves the engine on fd until the front closes it.
-- It sends `HELLO`, then runs the luv loop. It returns when the channel
-- closes. SIGHUP opens the log file again (see `lgml.log.reopen`). A job for a method that the engine does not have ends with
-- status 400.
-- @tparam int fd the imsg socket to the front
-- @tparam lgml.server.engine.Engine engine the engine
-- @tparam[opt] table mem the `mem` field of `HELLO` (see `lgml.ipc`)
-- @tparam[opt] table state `{ fp, name }`: on SIGTERM, the worker saves
-- the prompt store with fingerprint fp and sends it to the front (see
-- `lgml.ipc.STATE`), then ends
function M.serve(fd, engine, mem, state)
	local jobs, pending, deciding = {}, {}, {}
	local ch
	local function done(id, v)
		v.stats = stats(engine)
		ch:send(ipc.DONE, id, v)
	end
	local function chat(id, req)
		local sink = proxy(ch, id)
		jobs[id] = sink
		coroutine.wrap(function()
			local t0 = begin("chat", id, req)
			local ok, msg, status
			if engine.chat then
				ok, msg, status = engine:chat(req, sink)
			else
				msg, status = "model does not support chat", 400
			end
			jobs[id] = nil
			finish("chat", id, t0, ok, msg, status)
			done(id, {
				ok = ok == true, msg = msg, status = status, finish = sink.done,
			})
		end)()
	end
	-- A TRANSCRIBE request waits for its AUDIO message.
	local function transcribe(id, audio)
		local req = pending[id]
		pending[id] = nil
		if not req then
			return
		end
		local ctl = { stopped = false }
		function ctl.cancelled()
			return ctl.stopped
		end
		jobs[id] = ctl
		coroutine.wrap(function()
			local t0 = begin("transcribe", id, req)
			local r, msg, status
			if engine.transcribe then
				r, msg, status = engine:transcribe(req, audio, ctl)
			else
				msg, status = "model does not support transcription", 400
			end
			jobs[id] = nil
			finish("transcribe", id, t0, r ~= nil, msg, status)
			done(id, { ok = r ~= nil, msg = msg, status = status, result = r })
		end)()
	end
	-- r.audio is an lgml audio object, or r.wav holds WAV bytes.
	local function speak(id, req)
		local ctl = { stopped = false }
		function ctl.cancelled()
			return ctl.stopped
		end
		jobs[id] = ctl
		coroutine.wrap(function()
			local t0 = begin("speak", id, req)
			local r, msg, status
			if engine.speak then
				r, msg, status = engine:speak(req, ctl)
			else
				msg, status = "model does not support speech", 400
			end
			jobs[id] = nil
			local result
			if r then
				local wav = r.wav or r.audio:wav(1, "s16")
				result = { sample_rate = r.sample_rate or (r.audio and r.audio:rate()) }
				local ok, err = ch:send(ipc.AUDIO, id, wav)
				if not ok then
					r, msg, status = nil, "send audio: " .. tostring(err), 500
				end
			end
			finish("speak", id, t0, r ~= nil, msg, status)
			done(id, { ok = r ~= nil, msg = msg, status = status, result = result })
		end)()
	end
	local function decide(id, req)
		local ctl = { stopped = false }
		function ctl.cancelled()
			return ctl.stopped
		end
		jobs[id] = ctl
		coroutine.wrap(function()
			local t0 = begin("decide", id, req)
			local r, msg, status
			if engine.decide then
				r, msg, status = engine:decide(req, ctl)
			else
				msg, status = "model does not support decisions", 400
			end
			jobs[id] = nil
			finish("decide", id, t0, r ~= nil, msg, status)
			done(id, { ok = r ~= nil, msg = msg, status = status, result = r })
		end)()
	end
	-- The result images go out as IMAGE messages; DONE has the rest.
	local function generate(id, req)
		local ctl = { stopped = false }
		function ctl.cancelled()
			return ctl.stopped
		end
		jobs[id] = ctl
		coroutine.wrap(function()
			local t0 = begin("generate", id, req)
			local r, msg, status
			if engine.generate then
				r, msg, status = engine:generate(req, ctl)
			else
				msg, status = "model does not support images", 400
			end
			jobs[id] = nil
			if r then
				for _, png in ipairs(r.images) do
					local ok, err = ch:send(ipc.IMAGE, id, png)
					if not ok then
						r, msg, status = nil, "send image: " .. tostring(err), 500
						break
					end
				end
			end
			if r then
				r.images = nil
			end
			finish("generate", id, t0, r ~= nil, msg, status)
			done(id, { ok = r ~= nil, msg = msg, status = status, result = r })
		end)()
	end
	local function embed(id, req)
		local ctl = { stopped = false }
		function ctl.cancelled()
			return ctl.stopped
		end
		jobs[id] = ctl
		coroutine.wrap(function()
			local t0 = begin("embed", id, req)
			local r, msg, status
			if engine.embed then
				r, msg, status = engine:embed(req, ctl)
			else
				msg, status = "model does not support embeddings", 400
			end
			jobs[id] = nil
			finish("embed", id, t0, r ~= nil, msg, status)
			done(id, { ok = r ~= nil, msg = msg, status = status, result = r })
		end)()
	end
	ch = ipc.channel(fd, function(t, id, v)
		if t == ipc.JOB then
			chat(id, v)
		elseif t == ipc.TRANSCRIBE then
			pending[id] = v
		elseif t == ipc.AUDIO then
			transcribe(id, v)
		elseif t == ipc.SPEAK then
			speak(id, v)
		elseif t == ipc.DECIDE and type(v.image_types) == "table" and #v.image_types > 0 then
			v.images = {}
			deciding[id] = v
		elseif t == ipc.DECIDE then
			decide(id, v)
		elseif t == ipc.IMAGE and deciding[id] then
			local req = deciding[id]
			local k = #req.images + 1
			req.images[k] = { bytes = v, mime = req.image_types[k] }
			if k == #req.image_types then
				deciding[id], req.image_types = nil, nil
				decide(id, req)
			end
		elseif t == ipc.GENERATE and v.has_image then
			v.has_image = nil
			pending[id] = v
		elseif t == ipc.GENERATE then
			generate(id, v)
		elseif t == ipc.IMAGE and pending[id] then
			local req = pending[id]
			pending[id] = nil
			req.image = v
			generate(id, req)
		elseif t == ipc.EMBED then
			embed(id, v)
		elseif t == ipc.CANCEL and jobs[id] then
			jobs[id].stopped = true
		end
	end, function()
		uv.stop()
	end)
	-- SIGHUP opens the log file again, for log rotation.
	local hup = uv.new_signal()
	hup:start("sighup", function()
		log.reopen()
	end)
	hup:unref()
	local term, saving = uv.new_signal(), false
	term:start("sigterm", function()
		if not saving then
			saving = true
			on_term(ch, engine, state, term)
		end
	end)
	term:unref()
	ch:send(ipc.HELLO, 0, {
		models = engine:models(), props = engine:props(), caps = caps(engine), mem = mem,
	})
	uv.run()
end

return M
