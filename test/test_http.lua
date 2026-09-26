-- Tests for lgml.server with the fake engine. TAP output.
-- The server and a luv TCP client share one loop.

local uv = require("luv")
local json = require("lgml.json")
local server = require("lgml.server.httpd")
local http = require("lgml.server.http")
local openai = require("lgml.server.openai")
local fake = require("lgml.server.fake_engine")
local stream = require("lgml.server.stream")
local multipart = require("lgml.server.multipart")
local base64 = require("lgml.base64")
local log = require("lgml.log")

-- Log lines go to a table.
local lines = {}
log.set_output(function(l) lines[#lines + 1] = l end)

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

-- client

local function connect(port)
	local tcp = uv.new_tcp()
	local err = stream.await(function(done)
		tcp:connect("127.0.0.1", port, done)
	end)
	assert(not err, err)
	return { s = stream.new(tcp), buf = "" }
end

local function fill(c)
	local data, err = c.s:read(3000)
	if not data then
		return nil, err
	end
	c.buf = c.buf .. data
	return true
end

-- Returns the next n bytes, or nil at EOF.
local function take(c, len)
	while #c.buf < len do
		if not fill(c) then
			return nil
		end
	end
	local out = c.buf:sub(1, len)
	c.buf = c.buf:sub(len + 1)
	return out
end

local function line(c)
	while not c.buf:find("\r\n", 1, true) do
		if not fill(c) then
			return nil
		end
	end
	local l, rest = c.buf:match("^(.-)\r\n(.*)$")
	c.buf = rest
	return l
end

local function read_chunked(c, on_chunk)
	local parts = {}
	while true do
		local size = tonumber(line(c) or "", 16)
		if not size then
			return nil
		end
		if size == 0 then
			line(c)
			return table.concat(parts)
		end
		local data = take(c, size)
		take(c, 2)
		parts[#parts + 1] = data
		if on_chunk and on_chunk(data) == false then
			return nil, "stopped"
		end
	end
end

-- Reads one response. on_chunk sees each chunk and can stop the read.
local function read_response(c, on_chunk, head)
	local status_line = line(c)
	if not status_line then
		return nil
	end
	local r = { status = tonumber(status_line:match("^HTTP/1%.%d (%d+)")),
	    reason = status_line:match("^HTTP/1%.%d %d+ (.*)$"), headers = {} }
	for l in function() return line(c) end do
		if l == "" then
			break
		end
		local k, v = l:match("^([^:]+):%s*(.*)$")
		r.headers[k:lower()] = v
	end
	if r.status == 100 then
		return read_response(c, on_chunk, head)
	end
	local h = r.headers
	if head then
		r.body = ""
	elseif h["transfer-encoding"] == "chunked" then
		r.body = read_chunked(c, on_chunk)
	elseif h["content-length"] then
		r.body = take(c, tonumber(h["content-length"]))
	else
		while fill(c) do
		end
		r.body, c.buf = c.buf, ""
	end
	return r
end

local function request(method, path, body, extra)
	local t = { method .. " " .. path .. " HTTP/1.1", "Host: test" }
	for k, v in pairs(extra or {}) do
		t[#t + 1] = k .. ": " .. v
	end
	if body then
		t[#t + 1] = "Content-Length: " .. #body
	end
	return table.concat(t, "\r\n") .. "\r\n\r\n" .. (body or "")
end

local function roundtrip(port, raw)
	local c = connect(port)
	c.s:write(raw)
	local r = read_response(c, nil, raw:find("^HEAD "))
	c.s:close()
	return r
end

local function body_json(r)
	return r and r.body and json.decode(r.body)
end

local function sse_events(body)
	local ev = {}
	for data in (body or ""):gmatch("data: ([^\n]*)\n\n") do
		ev[#ev + 1] = data
	end
	return ev
end

local TOOLS = { { type = "function", ["function"] = {
	name = "get_weather", parameters = { type = "object" } } } }

local function chat_body(extra)
	local b = { messages = { { role = "user", content = "weather?" } } }
	for k, v in pairs(extra or {}) do
		b[k] = v
	end
	return assert(json.encode(b))
end

-- tests

local function test_basic(port)
	local r = roundtrip(port, request("GET", "/v1/models"))
	local b = body_json(r)
	ok(r.status == 200 and b.object == "list", "models status and object")
	ok(b.data[1].id == "fake" and b.data[1].context_length == 4096,
	    "models context_length")
	ok(table.concat(b.data[1].capabilities, ",") == "completion,decide,transcription,speech,image_generation",
	    "models capabilities from the engine methods")

	b = body_json(roundtrip(port, request("GET", "/props")))
	ok(b.default_generation_settings.n_ctx == 4096 and b.total_slots == 2,
	    "props n_ctx and total_slots")

	r = roundtrip(port, request("GET", "/health"))
	ok(r.status == 200 and body_json(r).status == "ok", "health")

	r = roundtrip(port, request("HEAD", "/health"))
	ok(r.status == 200 and r.body == "" and
	    tonumber(r.headers["content-length"]) > 0, "HEAD has no body")
end

local function test_errors(port)
	local r = roundtrip(port, request("POST", "/v1/chat/completions", "{bad"))
	local b = body_json(r)
	ok(r.status == 400 and b.error.type == "invalid_request_error",
	    "bad JSON gives 400")

	r = roundtrip(port, request("POST", "/v1/chat/completions",
	    '{"messages":[]}'))
	b = body_json(r)
	ok(r.status == 400 and b.error.param == "messages", "empty messages")

	r = roundtrip(port, request("POST", "/v1/chat/completions",
	    chat_body({ max_tokens = "x" })))
	ok(r.status == 400 and body_json(r).error.param == "max_tokens",
	    "bad max_tokens")

	r = roundtrip(port, request("POST", "/v1/chat/completions", json.encode({ messages = {
		{ role = "user", content = { { type = "image_url", image_url = { detail = "low" } } } } } })))
	ok(r.status == 400 and body_json(r).error.param == "messages", "image part without a URL")

	r = roundtrip(port, request("GET", "/nope"))
	ok(r.status == 404 and body_json(r).error.type == "not_found_error",
	    "unknown path gives 404")
	ok(r.reason == "Not Found", "status line has the reason phrase")

	r = roundtrip(port, request("POST", "/v1/systemone", "{bad"))
	ok(r.status == 422 and r.reason == "Unprocessable Content", "422 has its reason phrase")
	ok(http.reasons[409] == "Conflict" and http.reasons[499] == "Client Closed Request",
	    "409 and 499 have reason phrases")

	r = roundtrip(port, request("DELETE", "/v1/models"))
	ok(r.status == 405 and r.headers.allow == "GET", "wrong method gives 405")

	r = roundtrip(port, "POST /v1/chat/completions HTTP/1.1\r\n" ..
	    "Transfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n")
	ok(r.status == 411 and r.headers.connection == "close",
	    "chunked request body gives 411")

	r = roundtrip(port, "POST /x HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n")
	ok(r.status == 413, "large body gives 413")

	r = roundtrip(port, "GET / HTTP/1.1\r\nX: " .. string.rep("a", 70000) ..
	    "\r\n\r\n")
	ok(r.status == 431, "large header gives 431")

	r = roundtrip(port, "BROKEN\r\n\r\n")
	ok(r.status == 400, "bad request line gives 400")

	r = roundtrip(port, "GET / HTTP/2.0\r\n\r\n")
	ok(r.status == 505, "HTTP/2 request line gives 505")

	r = roundtrip(port, request("POST", "/v1/chat/completions",
	    json.encode({ messages = { { role = "user", content = "fail" } } })))
	b = body_json(r)
	ok(r.status == 500 and b.error.message == "fake failure" and
	    b.error.type == "server_error", "engine failure gives 500")
end

local function test_chat(port)
	local r = roundtrip(port, request("POST", "/v1/chat/completions",
	    chat_body({ tools = TOOLS })))
	local b = body_json(r)
	local ch = b and b.choices[1]
	ok(r.status == 200 and b.object == "chat.completion", "non-stream object")
	ok(ch.message.content == "Let me check the weather." and
	    ch.message.reasoning_content == "The user asks about Paris.",
	    "non-stream content and reasoning")
	local tc = ch.message.tool_calls[1]
	local args = tc and json.decode(tc["function"].arguments)
	ok(tc and tc.type == "function" and tc["function"].name == "get_weather"
	    and args and args.city == "Paris" and tc.id:find("^call_"),
	    "non-stream tool call")
	ok(ch.finish_reason == "tool_calls", "non-stream finish_reason")
	ok(r.headers["x-lgml-predicted-n"] == "7" and r.headers["x-lgml-prompt-n"] ~= nil,
	    "non-stream x-lgml timing headers")
	ok(b.usage.completion_tokens == 7 and b.usage.total_tokens ==
	    b.usage.prompt_tokens + 7 and
	    b.usage.prompt_tokens_details.cached_tokens == 0, "non-stream usage")

	b = body_json(roundtrip(port, request("POST", "/v1/chat/completions",
	    chat_body({ max_tokens = 2 }))))
	ok(b.choices[1].finish_reason == "length" and
	    b.choices[1].message.content == "" and
	    b.choices[1].message.reasoning_content ==
	    "The user asks about Paris.", "max_tokens gives length")

	b = body_json(roundtrip(port, request("POST", "/v1/chat/completions",
	    chat_body())))
	ok(b.choices[1].finish_reason == "stop" and
	    b.choices[1].message.tool_calls == nil, "no tools gives stop")
end

-- GET /metrics counts the earlier chat requests in Prometheus text.
local function test_metrics(port)
	local r = roundtrip(port, request("GET", "/metrics"))
	local n = r.body:match('lgml_chat_requests_total{model="[^"]*"} (%d+)')
	ok(r.status == 200 and r.headers["content-type"]:find("^text/plain") and
	    n and tonumber(n) >= 3, "metrics count chat requests")
	ok(r.body:find("# TYPE lgml_predicted_tokens_total counter", 1, true) ~= nil,
	    "metrics have type lines")
end

-- Assembles stream events the way a client does.
local function assemble(events)
	local a = { reasoning = "", content = "", tools = {}, n = #events }
	for i, data in ipairs(events) do
		if data == "[DONE]" then
			a.done_last = i == #events
		else
			local c = assert(json.decode(data))
			a.object = a.object or c.object
			if c.usage then
				a.usage = c.usage
			end
			local ch = c.choices[1]
			local d = ch and ch.delta or {}
			if i == 1 then
				a.role = d.role
			end
			a.reasoning = a.reasoning .. (d.reasoning_content or "")
			a.content = a.content .. (d.content or "")
			for _, t in ipairs(d.tool_calls or {}) do
				local cur = a.tools[t.index + 1] or { args = "" }
				a.tools[t.index + 1] = cur
				cur.id = t.id or cur.id
				cur.name = t["function"].name or cur.name
				cur.args = cur.args .. (t["function"].arguments or "")
			end
			if ch and ch.finish_reason ~= json.null then
				a.finish = ch.finish_reason
				a.timings = c.timings
			end
		end
	end
	return a
end

local function test_stream(port)
	local r = roundtrip(port, request("POST", "/v1/chat/completions",
	    chat_body({ stream = true, tools = TOOLS,
	    stream_options = { include_usage = true } })))
	ok(r.status == 200 and r.headers["content-type"] == "text/event-stream"
	    and r.headers["transfer-encoding"] == "chunked", "stream headers")
	local a = assemble(sse_events(r.body))
	ok(a.object == "chat.completion.chunk" and a.role == "assistant",
	    "stream object and first role")
	ok(a.reasoning == "The user asks about Paris." and
	    a.content == "Let me check the weather.", "stream text deltas")
	local t = a.tools[1]
	ok(t and t.name == "get_weather" and t.id and
	    json.decode(t.args).city == "Paris", "stream tool_calls assembly")
	ok(a.finish == "tool_calls" and a.timings, "stream finish and timings")
	ok(a.usage and a.usage.completion_tokens == 7 and
	    a.usage.prompt_tokens_details.cached_tokens == 0, "stream usage chunk")
	ok(a.done_last, "stream ends with [DONE]")

	r = roundtrip(port, request("POST", "/v1/chat/completions",
	    chat_body({ stream = true })))
	a = assemble(sse_events(r.body))
	ok(a.finish == "stop" and a.usage == nil, "stream without usage")
end

local function test_keepalive(port)
	local c = connect(port)
	c.s:write(request("GET", "/health"))
	local r1 = read_response(c)
	c.s:write(request("POST", "/v1/chat/completions",
	    chat_body({ stream = true })))
	local r2 = read_response(c)
	c.s:write(request("GET", "/props"))
	local r3 = read_response(c)
	ok(r1.status == 200 and r2.status == 200 and r3.status == 200 and
	    assemble(sse_events(r2.body)).done_last, "keep-alive across stream")

	c.s:write(request("GET", "/health") .. request("POST",
	    "/v1/chat/completions", chat_body()) .. request("GET", "/v1/models"))
	local p1, p2, p3 = read_response(c), read_response(c), read_response(c)
	ok(p1.status == 200 and body_json(p2).object == "chat.completion" and
	    body_json(p3).object == "list", "pipelined requests")
	c.s:close()

	c = connect(port)
	local raw = request("POST", "/v1/chat/completions", chat_body())
	for i = 1, #raw, 7 do
		c.s:write(raw:sub(i, i + 6))
		server.sleep(1)
	end
	r1 = read_response(c)
	ok(r1.status == 200 and body_json(r1).object == "chat.completion",
	    "request split into small writes")
	c.s:close()

	c = connect(port)
	c.s:write(request("GET", "/health", nil, { Connection = "close" }))
	r1 = read_response(c)
	local data = c.s:read(1000)
	ok(r1.status == 200 and r1.headers.connection == "close" and not data,
	    "Connection: close closes")
	c.s:close()

	c = connect(port)
	c.s:write("GET /health HTTP/1.0\r\n\r\n")
	r1 = read_response(c)
	ok(r1.status == 200 and body_json(r1).status == "ok",
	    "HTTP/1.0 without keep-alive")
	c.s:close()

	c = connect(port)
	local t0 = uv.now()
	data = c.s:read(3000)
	ok(not data and uv.now() - t0 < 2000, "idle connection closes")
	c.s:close()
end

local function test_cancel(port, eng)
	local before = eng.cancels
	local c = connect(port)
	c.s:write(request("POST", "/v1/chat/completions",
	    chat_body({ stream = true, tools = TOOLS })))
	local got = 0
	read_response(c, function(data)
		got = got + 1
		return got < 2
	end)
	c.s:abort()
	for _ = 1, 100 do
		if eng.cancels > before then
			break
		end
		server.sleep(10)
	end
	ok(got == 2 and eng.cancels == before + 1,
	    "client disconnect mid-stream cancels the engine")

	before = eng.cancels
	eng.delay = 20
	c = connect(port)
	c.s:write(request("POST", "/v1/chat/completions", chat_body()))
	server.sleep(30)
	c.s:abort()
	for _ = 1, 100 do
		if eng.cancels > before then
			break
		end
		server.sleep(10)
	end
	ok(eng.cancels == before + 1, "client disconnect cancels non-stream")
	eng.delay = 2
end

local function test_multipart()
	local body = "--XyZ\r\n" ..
	    'Content-Disposition: form-data; name="model"\r\n\r\n' ..
	    "whisper-1\r\n--XyZ\r\n" ..
	    'Content-Disposition: form-data; name="file"; filename="a b.wav"\r\n' ..
	    "Content-Type: audio/wav\r\n\r\nRIFF\0\r\n--X\r\n--XyZ--\r\n"
	local req = { headers = {
	    ["content-type"] = 'multipart/form-data; boundary="XyZ"' },
	    body = body }
	local m = multipart.from_request(req)
	ok(m and m.fields.model == "whisper-1" and m.files.file.filename ==
	    "a b.wav" and m.files.file.type == "audio/wav" and
	    m.files.file.data == "RIFF\0\r\n--X", "multipart fields and files")
	ok(multipart.parse("--XyZ\r\nbroken", "XyZ") == nil,
	    "multipart rejects a broken body")
end

local function form(parts)
	local t = {}
	for _, p in ipairs(parts) do
		local fn = p[3] and ('; filename="%s"'):format(p[3]) or ""
		t[#t + 1] = ('--B0\r\nContent-Disposition: form-data; name="%s"%s\r\n\r\n%s\r\n')
		    :format(p[1], fn, p[2])
	end
	return table.concat(t) .. "--B0--\r\n"
end

local function transcribe(port, parts)
	return roundtrip(port, request("POST", "/v1/audio/transcriptions", form(parts),
	    { ["Content-Type"] = "multipart/form-data; boundary=B0" }))
end

local function test_transcribe(port, bare_port)
	local wav = "RIFF\0\1\2\r\n"
	local r = transcribe(port, { { "file", wav, "a.wav" }, { "model", "fake" } })
	local b = body_json(r)
	ok(r.status == 200 and b.text == "fake transcript of 9 bytes", "transcription json")
	r = transcribe(port, { { "file", wav, "a.wav" }, { "response_format", "text" } })
	ok(r.status == 200 and r.body == "fake transcript of 9 bytes\n" and
	    r.headers["content-type"]:find("^text/plain"), "transcription text")
	b = body_json(transcribe(port, { { "file", wav, "a.wav" },
	    { "response_format", "verbose_json" }, { "language", "de" } }))
	ok(b.task == "transcribe" and b.language == "english" and b.duration == 1.5 and
	    b.segments[1]["end"] == 1.5, "transcription verbose_json")
	r = transcribe(port, { { "model", "fake" } })
	ok(r.status == 400 and body_json(r).error.param == "file", "transcription needs a file")
	r = transcribe(port, { { "file", wav, "a.wav" }, { "response_format", "srt" } })
	ok(r.status == 400 and body_json(r).error.param == "response_format",
	    "transcription rejects srt")
	r = transcribe(port, { { "file", "fail", "a.wav" } })
	ok(r.status == 400 and body_json(r).error.message:find("RIFF"),
	    "transcription engine error forwarded")
	r = roundtrip(port, request("POST", "/v1/audio/transcriptions", "{}",
	    { ["Content-Type"] = "application/json" }))
	ok(r.status == 400, "transcription needs multipart")
	r = transcribe(bare_port, { { "file", wav, "a.wav" } })
	ok(r.status == 404 and body_json(r).error.type == "not_found_error",
	    "engine without transcribe gives 404")
end

local PNG_SIG = "\137PNG\r\n\26\n"

local function gen_json(port, body)
	return roundtrip(port, request("POST", "/v1/images/generations", json.encode(body),
	    { ["Content-Type"] = "application/json" }))
end

local function edit(port, parts)
	return roundtrip(port, request("POST", "/v1/images/edits", form(parts),
	    { ["Content-Type"] = "multipart/form-data; boundary=B0" }))
end

local function test_images(port, bare_port)
	local r = gen_json(port, { prompt = "a fox", n = 2, size = "512x256", seed = 7, steps = 4 })
	local b = body_json(r)
	local png = b and b.data and b.data[1] and base64.decode(b.data[1].b64_json)
	ok(r.status == 200 and #b.data == 2 and png and png:sub(1, 8) == PNG_SIG,
	    "generations gives two base64 PNGs")
	ok(b.data[1].seed == 7 and b.data[2].seed == 8 and b.size == "512x256" and
	    b.output_format == "png" and type(b.created) == "number",
	    "generations gives seeds, size and format")
	ok(r.headers["x-lgml-steps"] == "8" and r.headers["x-lgml-images"] == "2" and
	    r.headers["x-lgml-seed"] == "7" and tonumber(r.headers["x-lgml-elapsed-ms"]) and
	    r.headers["x-lgml-ms-per-step"], "generations sets x-lgml headers")
	b = body_json(gen_json(port, { prompt = "a fox" }))
	ok(b.size == "1024x1024" and b.data[1].seed == 42, "generations defaults")
	local bad = {
		{ { n = 2 }, "prompt" }, { { prompt = "x", size = "1000x1000" }, "size" },
		{ { prompt = "x", size = "4096x16" }, "size" },
		{ { prompt = "x", response_format = "url" }, "response_format" },
		{ { prompt = "x", n = 11 }, "n" }, { { prompt = "x", steps = 0 }, "steps" },
		{ { prompt = "x", strength = 0.5 }, "strength" },
		{ { prompt = "x", output_format = "jpeg" }, "output_format" },
	}
	for _, c in ipairs(bad) do
		r = gen_json(port, c[1])
		ok(r.status == 400 and body_json(r).error.param == c[2], "generations rejects bad " .. c[2])
	end
	r = gen_json(port, { prompt = "fail" })
	ok(r.status == 500 and body_json(r).error.message == "fake failure",
	    "generations engine error forwarded")
	local pngin = base64.decode(b.data[1].b64_json)
	r = edit(port, { { "image", pngin, "in.png" }, { "prompt", "a wolf" }, { "strength", "0.6" },
	    { "seed", "3" }, { "n", "1" } })
	b = body_json(r)
	ok(r.status == 200 and #b.data == 1 and b.data[1].seed == 3 and
	    base64.decode(b.data[1].b64_json):sub(1, 8) == PNG_SIG, "edits gives a PNG")
	r = edit(port, { { "image[]", pngin, "in.png" }, { "prompt", "a wolf" }, { "size", "256x256" } })
	ok(r.status == 200 and body_json(r).size == "256x256", "edits takes image[] and size")
	r = edit(port, { { "prompt", "a wolf" } })
	ok(r.status == 400 and body_json(r).error.param == "image", "edits needs an image")
	r = edit(port, { { "image", pngin, "in.png" }, { "prompt", "a wolf" }, { "mask", pngin, "m.png" } })
	ok(r.status == 400 and body_json(r).error.param == "mask", "edits rejects a mask")
	r = edit(port, { { "image", pngin, "in.png" }, { "prompt", "a wolf" }, { "strength", "2" } })
	ok(r.status == 400 and body_json(r).error.param == "strength", "edits checks strength")
	r = edit(port, { { "image", "GIF89a", "in.gif" }, { "prompt", "a wolf" } })
	ok(r.status == 400 and body_json(r).error.message:find("PNG"), "edits forwards image errors")
	r = gen_json(bare_port, { prompt = "a fox" })
	ok(r.status == 404, "engine without generate gives 404")
	r = roundtrip(port, request("GET", "/metrics"))
	local n = r.body:match('lgml_image_requests_total{model="fake"} (%d+)')
	local imgs = r.body:match('lgml_images_total{model="fake"} (%d+)')
	local errs = r.body:match('lgml_image_errors_total{model="fake"} (%d+)')
	ok(tonumber(n) and tonumber(n) >= 5 and tonumber(imgs) >= 5 and tonumber(errs) == 2,
	    "metrics count image requests")
end

-- CORS: headers on every response and preflight answers, only when the
-- server has an origin.
local function test_cors(eng, plain_port)
	local srv = server.new({ cors = "*" })
	openai.install(srv, eng)
	local port = assert(srv:listen("127.0.0.1", 0))
	local r = roundtrip(port, request("OPTIONS", "/v1/chat/completions", nil, {
		Origin = "http://ui.test", ["Access-Control-Request-Method"] = "POST",
		["Access-Control-Request-Headers"] = "content-type",
	}))
	ok(r.status == 204 and r.headers["access-control-allow-origin"] == "*" and
	    r.headers["access-control-allow-methods"]:find("POST") and
	    r.headers["access-control-allow-headers"] == "content-type", "preflight answered")
	r = roundtrip(port, request("OPTIONS", "/nope", nil, { Origin = "http://ui.test" }))
	ok(r.status == 404, "preflight on an unknown path gives 404")
	r = roundtrip(port, request("POST", "/v1/chat/completions", chat_body(), { Origin = "http://ui.test" }))
	ok(r.status == 200 and r.headers["access-control-allow-origin"] == "*" and
	    r.headers["access-control-expose-headers"] == "*", "response has CORS headers")
	r = roundtrip(port, request("POST", "/v1/chat/completions", chat_body({ stream = true }),
	    { Origin = "http://ui.test" }))
	ok(r.headers["access-control-allow-origin"] == "*", "stream has CORS headers")
	r = roundtrip(port, request("GET", "/nope"))
	ok(r.status == 404 and r.headers["access-control-allow-origin"] == "*", "error has CORS headers")
	srv:close()
	r = roundtrip(plain_port, request("OPTIONS", "/v1/models", nil, { Origin = "http://ui.test" }))
	ok(r.status == 405 and r.headers["access-control-allow-origin"] == nil, "no CORS by default")
end

-- Log lines of a request id.
local function lines_of(rid)
	local out = {}
	for _, l in ipairs(lines) do
		if l:find(" rid=" .. rid .. " ", 1, true) then
			out[#out + 1] = l
		end
	end
	return out
end

local function find_line(list, pat)
	for _, l in ipairs(list) do
		if l:find(pat, 1, true) then
			return l
		end
	end
end

-- Request ids, access lines and debug lines.
local function test_logging(port)
	local r = roundtrip(port, request("POST", "/v1/chat/completions", chat_body({ max_tokens = 3 })))
	local rid = r.headers["x-lgml-request-id"]
	ok(r.status == 200 and rid and rid:match("^%x+$"), "the reply has x-lgml-request-id")
	local got = lines_of(rid)
	local access = find_line(got, "msg=access ")
	ok(access and access:find("level=info comp=http", 1, true) and
	    access:find(" method=POST path=/v1/chat/completions status=200 ", 1, true) and
	    access:find(" model=fake prompt_tokens=", 1, true) and access:find(" finish=length", 1, true),
	    "info gives an access line with the model, tokens and finish reason")
	ok(not find_line(got, "level=debug"), "info gives no debug lines")
	r = roundtrip(port, request("POST", "/v1/chat/completions", "{}"))
	access = find_line(lines_of(r.headers["x-lgml-request-id"]), "msg=access ")
	ok(access and access:find(' status=400 ', 1, true) and
	    access:find('err="messages must be a non-empty array"', 1, true), "an error reply logs its message")

	log.set_level("debug")
	local png = base64.decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
	local url = "data:image/png;base64," .. base64.encode(png .. string.rep("\0", 300))
	local body = assert(json.encode({ messages = {
		{ role = "system", content = "Be brief." },
		{ role = "user", content = { { type = "text", text = "what is this?" },
		    { type = "image_url", image_url = { url = url } } } },
		{ role = "user", content = "weather in Paris?" },
	} }))
	r = roundtrip(port, request("POST", "/v1/chat/completions", body, { ["Content-Type"] = "application/json" }))
	rid = r.headers["x-lgml-request-id"]
	got = lines_of(rid)
	local req_line = find_line(got, "msg=request ")
	ok(req_line and req_line:find("level=debug comp=http", 1, true) and
	    req_line:find("<data image/png 370B sha256:", 1, true) and not req_line:find("iVBOR", 1, true),
	    "debug logs the body with the data URL elided")
	local prompt = find_line(got, 'msg="chat prompt"')
	ok(prompt and prompt:find('text="<|system|>\\nBe brief.\\n<|user|>\\n\\n<|user|>\\nweather in Paris?\\n' ..
	    '<|assistant|>\\n"', 1, true), "debug logs the rendered prompt")
	ok(find_line(got, "msg=finish reason=stop") and find_line(got, 'msg=reply ') and
	    find_line(got, "msg=access "), "debug logs the finish reason and the reply")
	log.set_level("info")
end

local function main()
	local eng = fake.new({ delay = 2 })
	local srv = server.new({ limits = { max_body = 1 << 20, idle_ms = 500 } })
	openai.install(srv, eng)
	local port = assert(srv:listen("127.0.0.1", 0))
	ok(port > 0, "listen on an ephemeral port")
	test_basic(port)
	test_errors(port)
	test_chat(port)
	test_stream(port)
	test_metrics(port)
	test_keepalive(port)
	test_cancel(port, eng)
	test_multipart()
	local bare = setmetatable({ transcribe = false, generate = false }, { __index = eng })
	local srv2 = server.new()
	openai.install(srv2, bare)
	local bare_port = assert(srv2:listen("127.0.0.1", 0))
	test_transcribe(port, bare_port)
	test_images(port, bare_port)
	test_cors(eng, port)
	test_logging(port)
	srv2:close()
	srv:close()
	print("1.." .. n)
	uv.stop()
end

local co = coroutine.create(main)
local good, err = coroutine.resume(co)
if not good then
	error(debug.traceback(co, err))
end
uv.run()
if coroutine.status(co) ~= "dead" then
	print("not ok - main did not finish")
	failed = failed + 1
end
os.exit(failed == 0 and 0 or 1)
