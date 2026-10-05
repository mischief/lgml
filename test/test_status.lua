-- Keep-alive comments in the three chat streams while a slow prompt
-- reads, and GET /status with the progress of each slot: with the fake
-- engine in this process, and with the fake engine in a worker, whose
-- progress comes over IPC. TAP output.
local uv = require("luv")
local json = require("lgml.json")
local http = require("lgml.server.http")
local httpd = require("lgml.server.httpd")
local openai = require("lgml.server.openai")
local responses = require("lgml.server.responses")
local anthropic = require("lgml.server.anthropic")
local fake = require("lgml.server.fake_engine")
local remote = require("lgml.server.remote_engine")
local router = require("lgml.server.router")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local inproc = require("inproc")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local SLOW = { { role = "user", content = "slow" } }

-- Starts a request on srv and returns at once. The table gets done and
-- the response text when the request ends.
local function start(srv, method, target, body, headers)
	local h = { out = {} }
	local s = {
		write = function(_, data)
			h.out[#h.out + 1] = data
			return true
		end,
		gone = function()
			return false
		end,
	}
	local hs = {}
	for k, v in pairs(headers or {}) do
		hs[k:lower()] = v
	end
	local req = { method = method, target = target, path = target, query = "", version = "1.0",
		body = assert(json.encode(body)), headers = hs }
	local res = http.response({ s = s }, req)
	res.rid = "0000beef"
	coroutine.wrap(function()
		require("lgml.log").bind("0000beef")
		srv:dispatch(req, res)
		h.done = true
	end)()
	return h
end

local function run_until(f, ms)
	local t0 = uv.now()
	while not f() and uv.now() - t0 < (ms or 5000) do
		uv.run("once")
	end
	return f()
end

-- The count of comment lines before the first data line of a stream.
local function comments_before_data(text)
	local body = text:match("\r\n\r\n(.*)$") or ""
	local k = 0
	for line in body:gmatch("[^\n]+") do
		if line:match("^data:") or line:match("^event:") then
			break
		elseif line == ": keep-alive" then
			k = k + 1
		end
	end
	return k
end

local function test_keepalive()
	local eng = fake.new({ delay = 10 })
	local srv = httpd.new()
	openai.install(srv, eng, { key = ("k"):rep(32) })
	local keep = { openai.KEEPALIVE_MS, responses.KEEPALIVE_MS, anthropic.PING_MS }
	openai.KEEPALIVE_MS, responses.KEEPALIVE_MS, anthropic.PING_MS = 20, 20, 20
	local _, _, text = inproc.request(srv, "POST", "/v1/chat/completions",
	    { stream = true, messages = SLOW })
	local k = comments_before_data("\r\n\r\n" .. text)
	ok(k >= 3 and text:find("data: [DONE]", 1, true),
	    ("chat: keep-alive comments while the prompt reads (%d)"):format(k))
	_, _, text = inproc.request(srv, "POST", "/v1/responses", { stream = true, input = "slow" })
	local _, framed, c = inproc.events(text)
	ok(framed and c >= 3 and text:find(": keep-alive\n\n", 1, true),
	    ("responses: keep-alive comments while the prompt reads (%d)"):format(c))
	_, _, text = inproc.request(srv, "POST", "/v1/messages", { model = "fake", stream = true, max_tokens = 64, messages = SLOW },
	    { ["anthropic-version"] = "2023-06-01" })
	local pings = 0
	for _ in text:gmatch("event: ping\n") do
		pings = pings + 1
	end
	ok(pings >= 3, ("messages: ping events while the prompt reads (%d)"):format(pings))
	openai.KEEPALIVE_MS, responses.KEEPALIVE_MS, anthropic.PING_MS = table.unpack(keep)
	_, _, text = inproc.request(srv, "POST", "/v1/chat/completions", { stream = true, messages = SLOW })
	ok(not text:find(": keep-alive", 1, true), "chat: no comment before the interval")
end

-- The slot of /status that is not idle, or nil.
local function busy(status)
	for _, m in ipairs(status.models or {}) do
		for _, s in ipairs(m.slots) do
			if s.state ~= "idle" then
				return s, m
			end
		end
	end
end

local function test_status_inproc()
	local eng = fake.new({ delay = 10, total_slots = 2 })
	local srv = httpd.new()
	openai.install(srv, eng, { key = ("k"):rep(32) })
	local _, st = inproc.json(srv, "GET", "/status")
	ok(#st.models == 1 and st.models[1].id == "fake" and #st.models[1].slots == 2 and
	    st.models[1].slots[1].state == "idle" and st.models[1].queued == 0, "status: idle slots")
	local h = start(srv, "POST", "/v1/chat/completions", { messages = SLOW })
	local s
	run_until(function()
		s = busy(select(2, inproc.json(srv, "GET", "/v1/status")))
		return s and s.state == "prompt" and s.prompt_done > 0
	end)
	ok(s and s.state == "prompt" and s.id == 0 and s.request_id == "0000beef" and s.prompt_tokens == 5 and
	    s.prompt_done > 0, "status: the slot reads the prompt, n of m tokens, with the request id")
	run_until(function()
		s = busy(select(2, inproc.json(srv, "GET", "/status")))
		return s and s.state == "answer" and s.tokens > 0
	end)
	ok(s and s.state == "answer" and s.tokens > 0, "status: the slot answers, with the tokens so far")
	run_until(function()
		return h.done
	end)
	_, st = inproc.json(srv, "GET", "/status")
	ok(h.done and not busy(st), "status: idle again after the reply")
	-- A router lists each model.
	local r = router.new({ fake.new({ model = "a" }), fake.new({ model = "b" }) })
	local s2 = httpd.new()
	openai.install(s2, r, { key = ("k"):rep(32) })
	_, st = inproc.json(s2, "GET", "/status")
	ok(#st.models == 2 and st.models[1].id == "a" and st.models[2].id == "b", "status: a router lists each model")
end

local function test_status_worker()
	local src = here:match("^(.*)/test/$") or ".."
	local eng = remote.new({
		lua = "lua5.4", script = src .. "/bin/lgml-worker", args = { "--fake", "--fake-delay", "10" },
	})
	local ready
	coroutine.wrap(function()
		ready = eng:ready()
	end)()
	run_until(function()
		return ready
	end, 10000)
	local srv = httpd.new()
	openai.install(srv, eng, { key = ("k"):rep(32) })
	local h = start(srv, "POST", "/v1/chat/completions", { messages = SLOW })
	local s
	run_until(function()
		s = busy(select(2, inproc.json(srv, "GET", "/status")))
		return s and s.state == "prompt" and s.prompt_done > 0
	end)
	ok(s and s.request_id == "0000beef" and s.prompt_tokens == 5 and s.prompt_done > 0,
	    "worker: progress over IPC shows the prompt read")
	run_until(function()
		s = busy(select(2, inproc.json(srv, "GET", "/status")))
		return s and s.state == "answer"
	end)
	ok(s and s.state == "answer", "worker: the slot answers")
	run_until(function()
		return h.done
	end)
	local _, st = inproc.json(srv, "GET", "/status")
	ok(not busy(st) and st.models[1].slots[1].state == "idle", "worker: idle after the reply")
	local d
	coroutine.wrap(function()
		eng:shutdown()
		d = true
	end)()
	run_until(function()
		return d
	end)
end

test_keepalive()
test_status_inproc()
test_status_worker()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
