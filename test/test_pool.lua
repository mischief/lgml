-- Tests for lgml.server.pool: on-demand loads and LRU unloads, with
-- fake engines in real worker processes. TAP output.
local uv = require("luv")
local json = require("lgml.json")
local pool_mod = require("lgml.server.pool")
local remote = require("lgml.server.remote_engine")
local router = require("lgml.server.router")
local stream = require("lgml.server.stream")
local metrics = require("lgml.server.metrics")
local server = require("lgml.server.httpd")
local openai = require("lgml.server.openai")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local MiB = 1048576
local src = debug.getinfo(1, "S").source:sub(2):match("^(.*)/test/") or "."
local ALL = { "completion", "decide", "transcription", "speech", "image_generation", "embedding" }
-- Pool log lines, for checks.
local log = require("lgml.log")
local lines = {}
log.set_output(function(l) lines[#lines + 1] = l end)

local function logged(pat)
	for _, l in ipairs(lines) do
		if l:find(pat) then
			return l
		end
	end
end

-- Adds a fake model to pool. o: size (MiB, the estimate), measured
-- (MiB, what the worker reports), extra (MiB of each other kind),
-- load_ms, delay, fail, idle.
local function add(pool, id, o)
	o = o or {}
	local args = { "--fake", "--alias", id, "--fake-size", tostring(o.measured or o.size or 40) }
	for flag, v in pairs({ ["--fake-load-ms"] = o.load_ms, ["--fake-delay"] = o.delay }) do
		table.move({ flag, tostring(v) }, 1, 2, #args + 1, args)
	end
	if o.fail then
		args[#args + 1] = "--fake-fail"
	end
	local extra = {}
	for kind, mb in pairs(o.extra or {}) do
		extra[kind] = mb * MiB
	end
	local m
	m = pool:add({
		id = id, modes = o.modes or ALL, kind = o.kind or "gpu", size = (o.size or 40) * MiB,
		idle = o.idle, extra = extra,
		start = function(on_exit)
			m.starts = (m.starts or 0) + 1
			return remote.new({
				lua = "lua5.4", script = src .. "/bin/lgml-worker", args = args,
				restart = false, on_exit = on_exit,
			})
		end,
	})
	m.starts = 0
	return m
end

local function sink()
	local r = { text = {} }
	function r.reasoning() return true end
	function r.content(_, t) r.text[#r.text + 1] = t return true end
	function r.tool_call() return 0 end
	function r.tool_arguments() return true end
	function r.finish(_, reason) r.reason = reason end
	function r.cancelled() return false end
	return r
end

local function chat(e)
	return e:chat({ messages = { { role = "user", content = "hi" } } }, sink())
end

-- Runs f in a new coroutine. wait(h) yields until it ends.
local function spawn(f)
	local h = { done = false }
	coroutine.wrap(function()
		h.res = table.pack(f())
		h.done = true
	end)()
	return h
end

local function wait(h)
	while not h.done do
		stream.sleep(5)
	end
	return table.unpack(h.res, 1, h.res.n)
end

local function cleanup(pool)
	for _, m in ipairs(pool.models) do
		while m.state == "loading" do
			stream.sleep(10)
		end
		while m.inflight > 0 do
			stream.sleep(10)
		end
		m:unload()
		m:stop()
	end
end

local function states(pool)
	local t = {}
	for _, m in ipairs(pool.models) do
		t[#t + 1] = m.id .. "=" .. m.state
	end
	return table.concat(t, ",")
end

local function test_lazy()
	local p = pool_mod.new()
	local a = add(p, "a")
	local r = router.new({ a })
	ok(a.state == "unloaded" and a.starts == 0 and r:models()[1].status == "unloaded",
	    "a model does not load before a request")
	ok(r:pick(nil, "speak") == a and a.starts == 0, "the router picks by mode before the load")
	ok(chat(r) == true and a.state == "loaded" and a.starts == 1, "the first request loads the model")
	ok(chat(r) == true and a.starts == 1, "the next request uses the loaded model")
	ok(a.size == 40 * MiB and p:used("gpu") == 40 * MiB, "the size counts against the budget")
	ok(metrics.render():find('lgml_model_loads_total{model="a"} 1', 1, true),
	    "metrics count the load")
	cleanup(p)
	ok(a.state == "unloaded" and a.engine == nil and p:used("gpu") == 0, "unload frees the room")
end

local function test_shared_load()
	local p = pool_mod.new()
	local b = add(p, "b", { load_ms = 150 })
	local hs = {}
	for i = 1, 3 do
		hs[i] = spawn(function() return chat(b) end)
	end
	stream.sleep(20)
	ok(b.state == "loading", "the model is loading")
	local good = true
	for i = 1, 3 do
		good = good and wait(hs[i]) == true
	end
	ok(good and b.starts == 1, "three requests share one load")
	cleanup(p)
end

local function test_lru()
	metrics.reset()
	local p = pool_mod.new({ budget = { gpu = 100 * MiB } })
	local a, b, c = add(p, "a"), add(p, "b"), add(p, "c")
	chat(a)
	chat(b)
	chat(a)
	ok(states(p) == "a=loaded,b=loaded,c=unloaded", "two models fit")
	ok(chat(c) == true and states(p) == "a=loaded,b=unloaded,c=loaded",
	    "the least recently used model unloads")
	ok(logged('msg=unloading model=b why="least recently used" for=c kind=gpu'),
	    "the unload logs its model and why")
	chat(b)
	ok(states(p) == "a=unloaded,b=loaded,c=loaded" and p:used("gpu") == 80 * MiB,
	    "then the next least recently used one")
	ok(metrics.render():find('lgml_model_unloads_total{model="a"} 1', 1, true),
	    "metrics count the unload")
	cleanup(p)
end

local function test_in_flight()
	local p = pool_mod.new({ budget = { gpu = 100 * MiB } })
	local a = add(p, "a", { delay = 80 })
	local b, d = add(p, "b"), add(p, "d", { size = 50 })
	local long = spawn(function() return chat(a) end)
	while a.inflight == 0 do
		stream.sleep(5)
	end
	chat(b)
	ok(a.last < b.last and a.inflight == 1, "a is the least recently used, with a request in flight")
	ok(chat(d) == true and states(p) == "a=loaded,b=unloaded,d=loaded",
	    "a model with a request in flight stays")
	ok(wait(long) == true, "the request in flight ends well")
	cleanup(p)
end

-- Two requests for two models that do not fit together. A model that
-- loads serves its request before the other load can unload it.
local function test_swap()
	local p = pool_mod.new({ budget = { gpu = 100 * MiB } })
	local a, b = add(p, "a", { size = 60, load_ms = 30 }), add(p, "b", { size = 60, load_ms = 30 })
	local ha = spawn(function() return chat(a) end)
	local hb = spawn(function() return chat(b) end)
	local t0 = uv.now()
	while not (ha.done and hb.done) and uv.now() - t0 < 5000 do
		stream.sleep(10)
	end
	ok(ha.done and hb.done and ha.res[1] == true and hb.res[1] == true,
	    "two requests for models that do not fit together both end")
	ok(a.starts + b.starts <= 3, ("the models load at most 3 times (%d)"):format(a.starts + b.starts))
	cleanup(p)
end

-- A model with parts on two kinds counts each part against its budget.
local function test_parts()
	local p = pool_mod.new({ budget = { gpu = 100 * MiB, cpu = 100 * MiB } })
	local g = add(p, "g", { size = 50 })
	local c = add(p, "c", { size = 60, kind = "cpu" })
	local s = add(p, "s", { size = 40, measured = 30, extra = { cpu = 50 } })
	chat(g)
	chat(c)
	ok(chat(s) == true and states(p) == "g=loaded,c=unloaded,s=loaded",
	    "the cpu part of a split model unloads a cpu model")
	ok(p:used("gpu") == 80 * MiB and p:used("cpu") == 50 * MiB and s.size == 30 * MiB,
	    "each part counts against its budget; the measure changes the main part")
	ok(chat(c) == true and states(p) == "g=loaded,c=loaded,s=unloaded" and p:used("gpu") == 50 * MiB,
	    "unloading the split model for its cpu part frees its gpu part too")
	local big = add(p, "big", { size = 10, extra = { cpu = 120 } })
	local r, msg, status = chat(big)
	ok(r == nil and status == 503 and msg:find("needs 120 MiB; the cpu budget is 100 MiB", 1, true),
	    "a part larger than its budget gives 503")
	cleanup(p)
end

local function test_no_room()
	local p = pool_mod.new({ budget = { gpu = 100 * MiB }, load_wait = 0.2 })
	local a = add(p, "a", { size = 60, delay = 150 })
	local b = add(p, "b", { size = 60 })
	local big = add(p, "big", { size = 200 })
	local res, msg, status = chat(big)
	ok(res == nil and status == 503 and msg:find("needs 200 MiB; the gpu budget is 100 MiB", 1, true),
	    "a model larger than the budget gives 503")
	local long = spawn(function() return chat(a) end)
	while a.inflight == 0 do
		stream.sleep(5)
	end
	local t0 = uv.hrtime()
	res, msg, status = chat(b)
	local waited = (uv.hrtime() - t0) / 1e9
	ok(res == nil and status == 503 and msg:find("requests in flight: a", 1, true) and waited >= 0.19,
	    "no room after load_wait gives 503 with the reason")
	ok(b.state == "unloaded" and not b.counted and a.state == "loaded", "the failed load is not counted")
	wait(long)
	cleanup(p)

	-- With a longer wait, the load waits for the request to end.
	p = pool_mod.new({ budget = { gpu = 100 * MiB }, load_wait = 10 })
	a = add(p, "a", { size = 60, delay = 60 })
	b = add(p, "b", { size = 60 })
	long = spawn(function() return chat(a) end)
	while a.inflight == 0 do
		stream.sleep(5)
	end
	ok(chat(b) == true and a.state == "unloaded" and b.state == "loaded" and wait(long) == true,
	    "a load waits for a request in flight, then unloads that model")
	cleanup(p)
end

local function test_idle()
	local p = pool_mod.new({ idle = 0.15 })
	local a = add(p, "a")
	local k = add(p, "keep", { idle = 0 })
	chat(a)
	chat(k)
	stream.sleep(80)
	ok(a.state == "loaded", "a model stays before the idle timeout")
	chat(a)
	stream.sleep(100)
	ok(a.state == "loaded", "a request restarts the idle time")
	stream.sleep(200)
	ok(a.state == "unloaded" and k.state == "loaded", "an idle model unloads; idle = 0 keeps one")
	ok(chat(a) == true and a.starts == 2, "an unloaded model loads again")
	cleanup(p)
end

local function test_failure()
	local p = pool_mod.new({ budget = { gpu = 100 * MiB } })
	local f = add(p, "f", { fail = true })
	local hs = { spawn(function() return chat(f) end), spawn(function() return chat(f) end) }
	local r1 = table.pack(wait(hs[1]))
	local r2 = table.pack(wait(hs[2]))
	ok(r1[1] == nil and r1[3] == 503 and r1[2]:find("model f did not load: fake load failure", 1, true),
	    "a failed load gives 503 with the reason")
	ok(r2[1] == nil and r2[3] == 503 and f.starts == 1, "a waiting request gets the same error")
	ok(f.state == "unloaded" and p:used("gpu") == 0, "a failed model goes back to unloaded")
	ok(metrics.render():find('lgml_model_load_failures_total{model="f"} 1', 1, true),
	    "metrics count the failure")
	cleanup(p)

	-- A worker that dies after its load: the model is unloaded, then loads again.
	p = pool_mod.new()
	local a = add(p, "a", { size = 40, measured = 30 })
	chat(a)
	ok(a.size == 30 * MiB, "the measure of the worker replaces the estimate")
	a.engine.proc:kill("sigkill")
	while a.state == "loaded" do
		stream.sleep(5)
	end
	ok(a.state == "unloaded" and not a.counted, "a worker crash unloads the model")
	ok(chat(a) == true and a.starts == 2, "the next request loads it again")
	cleanup(p)
end

-- HTTP client, as in test_http.
local function roundtrip(port, method, path, body)
	local tcp = uv.new_tcp()
	assert(not stream.await(function(done)
		tcp:connect("127.0.0.1", port, done)
	end))
	local s = stream.new(tcp)
	s:write(("%s %s HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Length: %d\r\n\r\n%s")
	    :format(method, path, #(body or ""), body or ""))
	local buf = ""
	while true do
		local d = s:read(3000)
		if not d then
			break
		end
		buf = buf .. d
	end
	s:close()
	local status = tonumber(buf:match("^HTTP/1%.%d (%d+)"))
	local head, rest = buf:match("^(.-)\r\n\r\n(.*)$")
	if head and head:lower():find("transfer%-encoding: chunked") then
		local parts = {}
		while true do
			local size, r = rest:match("^(%x+)\r\n(.*)$")
			size = tonumber(size or "0", 16)
			if size == 0 then
				break
			end
			parts[#parts + 1] = r:sub(1, size)
			rest = r:sub(size + 3)
		end
		rest = table.concat(parts)
	end
	return status, json.decode(rest or "") or rest
end

local function test_http()
	local p = pool_mod.new()
	local a = add(p, "a", { modes = { "completion", "multimodal", "decide" } })
	local w = add(p, "w", { modes = { "transcription" } })
	local srv = server.new()
	openai.install(srv, router.new({ a, w }))
	local port = assert(srv:listen("127.0.0.1", 0))
	local status, b = roundtrip(port, "GET", "/v1/models")
	ok(status == 200 and b.data[1].status.value == "unloaded" and b.data[1].lgml.status == "unloaded" and
	    table.concat(b.data[1].capabilities, ",") == "completion,multimodal,decide" and
	    table.concat(b.data[2].capabilities, ",") == "transcription", "models list modes and status")
	status, b = roundtrip(port, "POST", "/models/load", '{"model": "w"}')
	ok(status == 200 and b.success == true and w.state == "loaded", "POST /models/load loads")
	status, b = roundtrip(port, "GET", "/v1/models")
	ok(b.data[2].status.value == "loaded" and b.data[2].lgml.bytes == 40 * MiB, "models show the load")
	status, b = roundtrip(port, "POST", "/v1/models/unload", '{"model": "w"}')
	ok(status == 200 and w.state == "unloaded", "POST /v1/models/unload unloads")
	status, b = roundtrip(port, "POST", "/models/load", '{"model": "nope"}')
	ok(status == 404 and b.error.message:find("unknown model"), "unknown model gives 404")
	status = roundtrip(port, "POST", "/v1/chat/completions",
	    '{"messages": [{"role": "user", "content": "hi"}]}')
	ok(status == 200 and a.state == "loaded" and w.state == "unloaded",
	    "a chat request loads the chat model")
	status, b = roundtrip(port, "GET", "/metrics")
	ok(type(b) == "string" and b:find('lgml_model_loaded_bytes{model="a"} 41943040', 1, true) and
	    b:find('lgml_model_load_seconds_total{model="a"}', 1, true), "metrics show loaded bytes and load time")
	srv:close()
	cleanup(p)
end

-- Global LoRA scales stay in the front process, across loads, and go
-- with each chat and decide request that has no lora field.
local function test_lora()
	local p = pool_mod.new()
	local m = p:add({ id = "l", modes = ALL, size = MiB,
	    props = { n_ctx = 8, total_slots = 1, lora = { { id = 0, path = "a.gguf", scale = 1 },
	        { id = 1, path = "b.gguf", scale = 1 } } },
	    start = function() error("no start") end })
	local seen = {}
	local stub = {}
	function stub.chat(_, req) seen.chat = req.lora return true end
	function stub.decide(_, req) seen.decide = req.lora return {} end
	m.load = function() return stub end
	m.release = function() end
	ok(m:lora_adapters({ { id = 1, scale = 0.5 } }), "a pool model takes global scales")
	local l = m:props().lora
	ok(l[1].scale == 0 and l[2].scale == 0.5 and l[2].path == "b.gguf" and m.static_props.lora[2].scale == 1,
	    "props show the global scales")
	m:chat({ messages = {} }, sink())
	m:decide({ questions = {} })
	ok(seen.chat and seen.chat[2].scale == 0.5 and seen.chat[1].scale == 0 and
	    seen.decide and seen.decide[2].scale == 0.5, "chat and decide send the global scales")
	m:chat({ messages = {}, lora = { { id = 0, scale = 2 } } }, sink())
	ok(#seen.chat == 1 and seen.chat[1].scale == 2, "a request lora goes as it is")
	local _, msg, status = m:lora_adapters({ { id = 2, scale = 1 } })
	ok(status == 400 and msg:find("id from 0 to 1", 1, true), "a bad id gives 400")
end

-- KV cache bytes of the tiny model at 256 positions, f16.
local function per256(model)
	local per, fixed = model:cache_size({ type = "f16" })
	return per * 256 + fixed
end

-- Estimates from files and the KV cache, checked against a loaded model.
local function test_estimate()
	local estimate = require("lgml.estimate")
	local lgml = require("lgml")
	package.path = src .. "/test/?.lua;" .. package.path
	local tiny = require("tiny")
	local path = tiny.tmp(".gguf")
	tiny.write(path, "qwen3")
	local model = require("lgml.transformer").load(lgml.backend("CPU"), path)
	-- Head size 16 does not fit q8_0, so the cache is f16, as in lgml.engine.
	local per, fixed = model:cache_size({ type = "f16" })
	local size = estimate.files(path)
	local e = estimate.model({ model = path })
	ok(e.n_ctx == 256 and e.files == size and e.kv == per * 256 + fixed and
	    e.bytes == size + e.kv + estimate.MARGIN, "estimate: files, cache and margin")
	per = model:cache_size({ type_k = "f32", type_v = "f16" })
	e = estimate.model({ model = path, ctx = 128, cache_type_k = "f32" })
	ok(e.n_ctx == 128 and e.kv == per * 128, "estimate: ctx and cache types")
	e = estimate.model({ model = path }, size + estimate.MARGIN + 130 * model:cache_size({ type = "f16" }))
	ok(e.n_ctx == 256, "estimate: the context shrinks to fit the budget, to at least 256")
	ok(estimate.id({ model = path }) == "tiny qwen3" and estimate.id({ model = "/m/x.gguf", id = "y" }) == "y" and
	    estimate.id({ model = "/m/w/", engine = "lgml.stt_engine" }) == "w" and
	    estimate.id({ model = "/m/o", engine = "lgml.tts.orpheus" }) == "orpheus", "estimate: model ids")
	-- Weights on the CPU count against cpu, with the cache of their layers.
	local info = lgml.gguf(path)
	local ffn, first = 0, 0
	for _, t in ipairs(info.tensors) do
		if t.name:find("ffn_") then
			ffn = ffn + t.size
		end
		if t.name:match("^blk%.0%.") or t.name:match("^token_embd") or t.name:match("^output") then
			first = first + t.size
		end
	end
	local gspec = { model = path, backend = "Vulkan0", override = "ffn_=CPU" }
	e = estimate.model(gspec)
	ok(e.files == size - ffn and e.extra.cpu == ffn + estimate.CPU_MARGIN and
	    e.kv == per256(model) and e.bytes == e.files + e.kv + estimate.MARGIN,
	    "estimate: weights on the CPU count against cpu")
	gspec = { model = path, backend = "Vulkan0", n_gpu_layers = 1 }
	e = estimate.model(gspec)
	local lper = model:cache_size({ type = "f16" }) // 2
	ok(e.files == size - first and e.kv == lper * 256 and
	    e.extra.cpu == first + lper * 256 + estimate.CPU_MARGIN,
	    "estimate: a layer on the CPU takes its cache along")
	e = estimate.model({ model = path, backend = "CPU", n_gpu_layers = 1 })
	ok(e.files == size and next(e.extra) == nil, "estimate: placement on a CPU backend moves nothing")
	e = estimate.model({ model = path, backend = "Vulkan0", n_cpu_moe = "auto" }, 1)
	ok(e.n_cpu_moe == 0 and next(e.extra) == nil, "estimate: auto without experts moves nothing")
	ok(not pcall(estimate.model, { model = path, backend = "Vulkan0", override = "x" }),
	    "estimate: a bad override is an error")
	ok(estimate.kind("CPU") == "cpu" and estimate.kind("Vulkan0") == "gpu" and
	    estimate.kind(nil) == "gpu" and estimate.kind(nil, false) == "cpu", "estimate: backend kinds")
	-- An image model with its text encoder on the CPU has a cpu part.
	local idir = tiny.tmp()
	os.execute("mkdir -p " .. idir)
	for name, len in pairs({ ["z_image_turbo.gguf"] = 100, ["Qwen3-4B.gguf"] = 40, ["ae.safetensors"] = 7 }) do
		local f = assert(io.open(idir .. "/" .. name, "wb"))
		f:write(string.rep("x", len))
		f:close()
	end
	local ispec = { model = idir, engine = "lgml.image.engine", backend = "Vulkan0" }
	e = estimate.model(ispec)
	ok(e.files == 147 and next(e.extra) == nil, "estimate: an image model on one backend")
	ispec.text_backend = "CPU"
	e = estimate.model(ispec)
	ok(e.files == 107 and e.bytes == 107 + estimate.MARGIN and e.kind == "gpu" and
	    e.extra.cpu == 40 + estimate.TEXT_MARGIN, "estimate: the text encoder counts against cpu")
	for _, name in ipairs({ "z_image_turbo.gguf", "Qwen3-4B.gguf", "ae.safetensors" }) do
		os.remove(idir .. "/" .. name)
	end
	os.remove(idir)
	local dir = tiny.tmp()
	os.execute("mkdir -p " .. dir .. "/sub")
	for name, len in pairs({ a = 10, b = 5, ["sub/c"] = 100 }) do
		local f = assert(io.open(dir .. "/" .. name, "wb"))
		f:write(string.rep("x", len))
		f:close()
	end
	ok(estimate.files(dir) == 15 and estimate.files(dir .. "/none") == 0,
	    "estimate: a directory counts its own files")
	os.remove(dir .. "/sub/c")
	os.remove(dir .. "/sub")
	os.remove(dir .. "/a")
	os.remove(dir .. "/b")
	os.remove(dir)
	os.remove(path)
end

-- A Lua error after main yields ends the coroutine without a report.
local finished = false

local function main()
	test_estimate()
	test_lazy()
	test_shared_load()
	test_lru()
	test_in_flight()
	test_swap()
	test_parts()
	test_no_room()
	test_idle()
	test_failure()
	test_http()
	test_lora()
	print("1.." .. n)
	finished = true
	uv.stop()
end

local co = coroutine.create(main)
local good, err = coroutine.resume(co)
if not good then
	error(debug.traceback(co, err))
end
uv.run()
if not finished then
	print("not ok - main did not finish")
	failed = failed + 1
end
os.exit(failed == 0 and 0 or 1)
