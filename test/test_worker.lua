-- Tests for the worker process and remote engine, with the fake engine
-- in the worker. TAP output.
local uv = require("luv")
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
local logs = {}
local eng = remote.new({
	lua = "lua5.4", script = src .. "/bin/lgml-worker", args = { "--fake" },
	log = function(m) logs[#logs + 1] = m end,
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
	function r.content(_, t) r.text[#r.text + 1] = "C:" .. t return count() end
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
	req.state = "fail"
	local res, msg, status = eng:decide(req)
	ok(res == nil and msg == "fake failure" and status == 500, "decide error forwarded")
end)

eng:stop()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
