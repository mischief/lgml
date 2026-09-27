-- Tests for the worker process and remote engine, with the fake engine
-- in the worker. TAP output.
local uv = require("luv")
local stream = require("lgml.server.stream")
local remote = require("lgml.server.remote_engine")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local src = debug.getinfo(1, "S").source:sub(2):match("^(.*)/test/") or "."
local eng = remote.new({
	lua = "lua5.4", script = src .. "/bin/lgml-worker", args = { "--fake", "-t", "2" },
})

-- A sink that records what it gets.
local function recorder(stop_after)
	local r = { text = {}, calls = {}, args = {}, n = 0 }
	local function count()
		r.n = r.n + 1
		if stop_after and r.n > stop_after then
			r.gone = true
			return nil, "cancelled"
		end
		return true
	end
	function r.reasoning(_, t) r.text[#r.text + 1] = "R:" .. t return count() end
	function r.content(_, t, lp)
		r.text[#r.text + 1] = "C:" .. t
		for _, e in ipairs(lp or {}) do
			r.lp = r.lp or {}
			r.lp[#r.lp + 1] = e
		end
		return count()
	end
	function r.tool_call(_, name)
		r.calls[#r.calls + 1] = name
		if not count() then return nil end
		return #r.calls - 1
	end
	function r.tool_arguments(_, i, t)
		r.args[i] = (r.args[i] or "") .. t
		return count()
	end
	function r.finish(_, reason, usage) r.reason, r.usage = reason, usage end
	function r.cancelled() return r.gone == true end
	return r
end

local function in_co(f)
	local done = false
	coroutine.wrap(function()
		f()
		done = true
	end)()
	while not done do
		uv.run("once")
	end
end

ok(eng:models()[1].id == "fake", "hello gives models")
ok(eng:props().total_slots == 2, "hello gives props")

local tools = { { type = "function", ["function"] = { name = "get_weather" } } }
in_co(function()
	local r = recorder()
	local res = eng:chat({ messages = { { role = "user", content = "hi" } }, tools = tools }, r)
	ok(res == true, "chat returns true")
	ok(table.concat(r.text, "|") == "R:The user asks |R:about Paris.|C:Let me |C:check the weather.",
	    "deltas arrive in order")
	ok(r.calls[1] == "get_weather" and r.args[0] == '{"city": "Paris"}', "tool call forwarded")
	ok(r.reason == "stop" and r.usage.completion_tokens == 7, "finish forwarded")
end)

in_co(function()
	local r = recorder()
	local res = eng:chat({ messages = { { role = "user", content = "hi" } }, logprobs = true,
		top_logprobs = 1, stop = { "weather" } }, r)
	ok(res == true and r.lp and #r.lp == 2 and r.lp[2].token == "check the weather." and
	    r.lp[2].top_logprobs[1].bytes[1] == 99, "logprobs are forwarded")
	ok(table.concat(r.text, "|"):find("C:check the |", 1, true) == nil and
	    r.text[#r.text] == "C:check the " and r.reason == "stop", "stop runs in the worker")
end)

in_co(function()
	local r = recorder(2)
	local res = eng:chat({ messages = { { role = "user", content = "hi" } } }, r)
	ok(res == true and r.reason == nil, "client cancel stops the job")
end)

in_co(function()
	local r = recorder()
	local res, msg, status = eng:chat({ messages = { { role = "user", content = "fail" } } }, r)
	ok(res == nil and msg == "fake failure" and status == 500, "engine error forwarded")
end)

-- Kill the worker during a job: the job fails with 503, a new worker starts.
in_co(function()
	local r = recorder()
	local t = uv.new_timer()
	t:start(8, 0, function()
		t:close()
		eng.proc:kill("sigkill")
	end)
	local res, msg, status = eng:chat({ messages = { { role = "user", content = "hi" } } }, r)
	ok(res == nil and status == 503 and msg:find("worker died"), "crash gives 503")
end)
eng.hello = nil
ok(eng:models()[1].id == "fake", "new worker says hello")
in_co(function()
	local r = recorder()
	ok(eng:chat({ messages = { { role = "user", content = "hi" } } }, r) == true,
	    "new worker serves")
end)

-- Transcription: the request as JSON, the audio as raw bytes.
in_co(function()
	local audio = string.rep("\0\1\255", 7 << 20)
	local r = eng:transcribe({ language = "en" }, audio)
	ok(r and r.text == ("fake transcript of %d bytes"):format(#audio) and r.language == "en",
	    "21 MB audio reaches the worker")
	local res, msg, status = eng:transcribe({}, "fail")
	ok(res == nil and status == 400 and msg:find("RIFF"), "transcription error forwarded")
end)

in_co(function()
	local r = eng:speak({ input = "hello", response_format = "wav" })
	ok(r and r.wav:sub(1, 4) == "RIFF" and r.sample_rate == 24000 and
	    #r.wav == 44 + 5 * 240 * 2, "speech returns wav bytes")
end)

in_co(function()
	local req = { state = "Paris is the capital of France.", questions = { {
		id = "q", type = "choice", instructions = "",
		options = { { key = "Berlin", text = "" }, { key = "Paris", text = "" } },
	} } }
	local r = eng:decide(req)
	local p = r and r.answers.q.probabilities
	ok(p and p[2] > 0.9 and r.model == "fake" and r.usage.output_tokens == 2,
	    "decide runs on the worker")
	local png = require("lgml.base64").decode(
	    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
	local big = png .. string.rep("\0\255", 1 << 20)
	req.images = { { bytes = big, mime = "image/png" }, { bytes = "\255\216\255\224", mime = "image/jpeg" } }
	r = eng:decide(req)
	ok(r and r.timings.images == 2 and r.timings.image_bytes == #big + 4 and #req.images == 2,
	    "decide images go to the worker as raw bytes")
	req.images = { { bytes = "GIF89a", mime = "image/gif" } }
	local res, msg, status = eng:decide(req)
	ok(res == nil and status == 400 and msg:find("image 1", 1, true), "a bad decide image gives 400")
	req.images = nil
	req.state = "fail"
	res, msg, status = eng:decide(req)
	ok(res == nil and msg == "fake failure" and status == 500, "decide error forwarded")
end)

-- Images: the input and each result PNG go as raw IMAGE messages.
in_co(function()
	local r = eng:generate({ prompt = "a fox", n = 3, seed = 5, width = 64, height = 32 })
	ok(r and #r.images == 3 and r.images[3]:sub(2, 4) == "PNG" and r.seeds[3] == 7 and
	    r.width == 64 and r.steps == 24, "generate returns PNGs and seeds")
	local png = r.images[1]
	r = eng:generate({ prompt = "a wolf", image = png .. string.rep("\0", 5 << 20), strength = 0.5 })
	ok(r and #r.images == 1, "edit sends a 5 MB input image")
	local res, msg, status = eng:generate({ prompt = "a wolf", image = "GIF89a" })
	ok(res == nil and status == 400 and msg:find("PNG"), "edit error forwarded")
	res, msg, status = eng:generate({ prompt = "fail" })
	ok(res == nil and msg == "fake failure" and status == 500, "generate error forwarded")
end)

eng:stop()

-- A worker whose engine does not load sends FAIL and exits.
in_co(function()
	local exits = 0
	local bad = remote.new({
		lua = "lua5.4", script = src .. "/bin/lgml-worker", args = { "--fake", "--fake-fail" },
		restart = false, on_exit = function() exits = exits + 1 end,
	})
	local h, msg = bad:ready()
	ok(h == nil and msg == "fake load failure" and exits == 1, "a failed load gives its message")
	local res, m2, status = bad:chat({ messages = {} }, recorder())
	ok(res == nil and status == 503 and m2 == "fake load failure", "a job on a failed worker gives 503")
	local good = remote.new({
		lua = "lua5.4", script = src .. "/bin/lgml-worker",
		args = { "--fake", "--alias", "sized", "--fake-size", "12" }, restart = false,
	})
	h = good:ready()
	ok(h and h.models[1].id == "sized" and h.mem.used == 12 * 1048576, "hello has the measured size")
	good:shutdown()
	ok(good.proc == nil, "shutdown waits for the exit")
end)
-- The worker gets the log level and file of the front, and the lines
-- of a job carry the request id of the caller.
in_co(function()
	local log = require("lgml.log")
	local path = os.tmpname()
	log.set_level("debug")
	assert(log.open(path))
	local w = remote.new({
		lua = "lua5.4", script = src .. "/bin/lgml-worker", args = { "--fake", "--alias", "logged" },
		restart = false,
	})
	log.bind("feedc0de")
	local r = recorder()
	local res = w:chat({ messages = { { role = "user", content = "hi" } } }, r)
	log.bind(nil)
	w:shutdown()
	log.open(nil)
	log.set_level("info")
	local f = assert(io.open(path))
	local text = f:read("a")
	f:close()
	os.remove(path)
	ok(res == true and text:find("--log-level debug --log-file " .. path, 1, true),
	    "the worker gets --log-level and --log-file")
	ok(text:find('level=debug comp=worker:logged/engine rid=feedc0de msg="chat prompt"', 1, true),
	    "a worker debug line has the model id and the request id")
	ok(text:find("comp=worker:logged rid=feedc0de msg=\"job done\" kind=chat", 1, true),
	    "the worker logs the job with the request id")
end)

-- A worker that crashes twice within the window does not start again.
in_co(function()
	local w = remote.new({
		lua = "lua5.4", script = src .. "/bin/lgml-worker", args = { "--fake", "--alias", "twice" },
	})
	local crash = { messages = { { role = "user", content = "crash" } } }
	local res, _, status = w:chat(crash, recorder())
	ok(res == nil and status == 503, "the first crash fails its request")
	w.hello = nil
	ok(w:ready() and w:chat({ messages = { { role = "user", content = "hi" } } }, recorder()) == true,
	    "a new worker starts after the first crash")
	w:chat(crash, recorder())
	stream.sleep(1500)
	local r2, msg2, status2 = w:chat({ messages = { { role = "user", content = "hi" } } }, recorder())
	ok(r2 == nil and status2 == 503 and msg2:find("crashed twice", 1, true) and w.proc == nil,
	    "after a second crash no worker starts and requests say why")
	w:shutdown()
end)

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
