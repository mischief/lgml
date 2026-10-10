-- Memory accounting and refusals: lgml.memory, the memory guard,
-- lgml.budget reports of a tiny chat engine, and a worker that does not
-- fit. CPU backend only. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local uv = require("luv")
local lgml = require("lgml")
local tiny = require("tiny")
local engine = require("lgml.engine")
local budget = require("lgml.budget")
local transformer = require("lgml.transformer")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function cpu()
	return lgml.memory().CPU or { weights = 0, tensors = 0, compute = 0, peak = 0, peak_compute = 0 }
end

-- lgml.memory counts tensors and compute buffers; graph:measure gives
-- the buffer that a compute then takes.
do
	local be <close> = assert(lgml.backend("CPU"))
	local m0 = cpu()
	local a = be:tensor("f32", { 256, 256 })
	ok(cpu().tensors - m0.tensors == 256 * 256 * 4, "a tensor counts its bytes")
	local g = be:graph()
	local y = g:add(g:mul_mat(a, a), a)
	local planned = g:measure(y)
	ok(planned > 0 and cpu().compute == m0.compute, "measure plans and allocates nothing")
	g:compute(y)
	local dev = g:buffer_size()
	ok(dev == planned and cpu().compute - m0.compute == dev, "compute takes the measured buffer")
	ok(cpu().peak_compute >= dev, "the compute peak holds the buffer")
	g:close()
	g, y = nil, nil
	collectgarbage()
	ok(cpu().compute == m0.compute, "a collected graph gives its buffer back")
	a:close()
	ok(cpu().tensors == m0.tensors, "a closed tensor gives its bytes back")
	lgml.memory_reset()
	ok(cpu().peak_compute == cpu().compute, "memory_reset sets the peaks to now")

	-- A limit refuses a tensor, a check and a graph plan past it.
	local old = lgml.memory_guard({ limit = cpu().weights + cpu().tensors + (1 << 20) })
	local good, err = pcall(be.tensor, be, "f32", { 1024, 1024 })
	ok(not good and budget.is_budget(err) and err:find("a tensor needs 4 MiB more on CPU", 1, true),
	    "a tensor past the limit raises the memory budget error")
	local fits, msg = lgml.memory_check(be, 8 << 20, "the test")
	ok(fits == nil and msg:find("^memory budget: the test needs 8 MiB"), "memory_check gives the error")
	ok(lgml.memory_check(be, 1024) == true, "memory_check passes what fits")
	local b = be:tensor("f32", { 512, 256 })
	local g2 = be:graph()
	local big = g2:mul_mat(b, b)
	good, err = pcall(g2.compute, g2, big)
	ok(not good and budget.is_budget(err) and err:find("a graph needs", 1, true),
	    "a graph plan past the limit raises the error before it allocates")
	lgml.memory_guard(old)
	ok(pcall(g2.compute, g2, big), "the same graph runs without the limit")
	ok(budget.status(err, 500) == 507 and budget.status("other", 500) == 500,
	    "budget.status maps the error to 507")
end

-- budget.drm reads fdinfo files: one count per client, by bus id.
do
	local dir = tiny.tmp("")
	assert(uv.fs_mkdir(dir, 448))
	local files = {
		["10"] = "drm-client-id:\t7\ndrm-pdev:\t0000:03:00.0\ndrm-total-vram:\t2 MiB\ndrm-total-gtt:\t4 KiB\n",
		["11"] = "drm-client-id:\t7\ndrm-pdev:\t0000:03:00.0\ndrm-total-vram:\t2 MiB\n",
		["12"] = "drm-client-id:\t8\ndrm-pdev:\t0000:6e:00.0\ndrm-memory-vram:\t1024 KiB\n",
		["13"] = "pos:\t0\nflags:\t02\n",
	}
	for name, text in pairs(files) do
		local f = assert(io.open(dir .. "/" .. name, "w"))
		f:write(text)
		f:close()
	end
	ok(budget.drm("0000:03:00.0", false, dir) == 2 << 20, "drm: a client counts once, by bus id")
	ok(budget.drm("0000:03:00.0", true, dir) == (2 << 20) + 4096, "drm: gtt counts on request")
	ok(budget.drm("0000:6e:00.0", false, dir) == 1 << 20, "drm: the amdgpu key works too")
	ok(budget.drm(nil, false, dir) == 3 << 20, "drm: no id counts every client")
	ok(budget.drm("0000:ff:00.0", false, dir) == nil, "drm: no client gives nil")
	for name in pairs(files) do
		os.remove(dir .. "/" .. name)
	end
	uv.fs_rmdir(dir)
end

local path = tiny.tmp(".gguf")
tiny.write(path, "qwen3", { n_ctx = 4096 })

local function sink()
	local r = { text = {} }
	r.sink = {
		content = function(_, t) r.text[#r.text + 1] = t return true end,
		reasoning = function() return true end,
		tool_call = function() return 0 end,
		tool_arguments = function() return true end,
		finish = function(_, reason, usage) r.reason, r.usage = reason, usage end,
		cancelled = function() return false end,
	}
	return r
end

-- Runs chats at once, one per word list; returns the replies.
local function chats(e, list)
	local out, left = {}, #list
	for i, words in ipairs(list) do
		coroutine.wrap(function()
			local r = sink()
			r.res, r.err, r.status = e:chat({ messages = { { role = "user", content = words } },
				max_tokens = 4, temperature = 0, chat_template_kwargs = { enable_thinking = false } },
				r.sink)
			out[i] = r
			left = left - 1
		end)()
	end
	while left > 0 do
		uv.run("once")
	end
	return out
end

-- Words that make about n tokens.
local function words(n, seed)
	local r, t = tiny.rng(seed), {}
	for i = 1, n do
		t[i] = string.char(97 + math.floor(r() * 26))
	end
	return table.concat(t, " ")
end

-- The report of an engine holds its peak: long prompts in two slots at
-- once stay within weights + cache + compute.
do
	lgml.memory_reset()
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 4096, slots = 2, kv_initial = 256,
		cache_ram = 0 })
	local per, fixed = e.model:cache_size(e.copts)
	local base = cpu()
	local r = budget.report(e, "CPU")
	ok(r.memory == "CPU" and r.reserved and r.overhead == 0, "the report is exact on the CPU")
	ok(r.cache == per * 4096 + fixed * 2 and r.cache > base.tensors,
	    "the cache part is the pool at n_ctx cells")
	ok(e.pool.cap == 256, "the reservation leaves the pool at its first size")
	ok(r.compute > 0 and r.used == r.weights + r.cache + r.compute + r.overhead,
	    "used is the sum of the parts")
	lgml.memory_reset()
	local out = chats(e, { words(1900, 1), words(1900, 2) })
	ok(out[1].res == true and out[2].res == true and e.pool.cap == 4096,
	    "two long prompts at once grow the pool to n_ctx")
	local peak = cpu().peak
	ok(peak <= r.used, ("the peak (%d) stays within the report (%d)"):format(peak, r.used))
	ok(cpu().peak_compute <= r.compute,
	    ("the compute peak (%d) stays within the reserved compute (%d)"):format(cpu().peak_compute, r.compute))
end

-- A pool at the limit shrinks: each old tensor closes before its new
-- one opens.
do
	collectgarbage()
	collectgarbage()
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 4096, slots = 1, kv_initial = 256,
		cache_ram = 0 })
	e.model:pool_resize(e.pool, 1024)
	collectgarbage()
	collectgarbage()
	local old = lgml.memory_guard({ limit = cpu().weights + cpu().tensors + cpu().compute })
	local good, err = pcall(e.model.pool_resize, e.model, e.pool, 256)
	lgml.memory_guard(old)
	ok(good and e.pool.cap == 256, "a full pool shrinks: " .. tostring(err))
	ok(chats(e, { words(100, 4) })[1].res == true, "the shrunk pool serves a prompt")
end

-- A prompt whose KV growth passes the guard fails with 507; the engine
-- then serves a short prompt. The limit must not count the tensors of
-- earlier engines that the collector has not freed yet.
do
	collectgarbage()
	collectgarbage()
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 4096, slots = 1, kv_initial = 256,
		cache_ram = 0 })
	local grow = transformer.block_bytes(e.pool) // e.pool.block * 256
	-- Graphs of the last block must go first: the limit counts compute
	-- memory that only this engine holds.
	collectgarbage()
	collectgarbage()
	local old = lgml.memory_guard({ limit = cpu().weights + cpu().tensors + cpu().compute + grow })
	local r = chats(e, { words(1500, 3) })[1]
	ok(r.res == nil and r.status == 507 and budget.is_budget(r.err) and
	    r.err:find("the KV pool from", 1, true), "a prompt past the KV budget gives 507")
	ok(e.pool.cap == 256, "the pool keeps its size")
	lgml.memory_guard(old)
	r = chats(e, { words(1500, 3) })[1]
	ok(r.res == true, "the same prompt runs without the limit")
end

-- A worker whose model does not fit refuses the load; the pool gives 507.
do
	local pool_mod = require("lgml.server.pool")
	local remote = require("lgml.server.remote_engine")
	local src = here .. ".."
	local p = pool_mod.new({ budget = { cpu = 1 << 30 } })
	local function add(id, args)
		return p:add({ id = id, modes = { "completion" }, kind = "cpu", size = 1 << 20,
			start = function(on_exit)
				return remote.new({ lua = "lua5.4", script = src .. "/bin/lgml-worker",
					args = args, restart = false, on_exit = on_exit })
			end })
	end
	local small = add("small", { "-m", path, "-b", "CPU", "-c", "4096", "--mem-limit", "2" })
	local fine = add("fine", { "-m", path, "-b", "CPU", "-c", "4096", "--kv-initial", "256" })
	local done, res = false, {}
	coroutine.wrap(function()
		res.a = table.pack(small:load())
		res.b = table.pack(fine:load())
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	ok(res.a[1] == nil and res.a[3] == 507 and res.a[2]:find("memory budget:", 1, true) and
	    small.state == "unloaded", "a worker past its limit fails the load with 507")
	local mem = fine:models()[1].memory
	ok(res.b[1] ~= nil and mem and mem.cache > 0 and mem.compute > 0 and
	    fine.size == mem.weights + mem.cache + mem.compute + mem.overhead,
	    "a loaded worker reports its parts, and the pool counts their sum")
	done = false
	coroutine.wrap(function()
		fine:unload()
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	p:stop()
end

os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
