-- Anthropic Messages compatibility: the field table of docs/api.md
-- against the request checks, and the Anthropic reference examples
-- (test/fixtures/anthropic) against the server with the fake engine, in
-- process. TAP output.
local json = require("lgml.json")
local httpd = require("lgml.server.httpd")
local openai = require("lgml.server.openai")
local anthropic = require("lgml.server.anthropic")
local translate = require("lgml.server.translate")
local fake = require("lgml.server.fake_engine")

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

local function read(path)
	local f = assert(io.open(here .. "../" .. path, "rb"))
	local s = f:read("a")
	f:close()
	return s
end

local EX = assert(json.decode((read("test/fixtures/anthropic/examples.json"))))
-- Fixture requests with the real PNG in place of "$PNG".
local function example(name)
	local text = assert(json.encode(EX.requests[name])):gsub("%$PNG", EX.png)
	return assert(json.decode(text))
end

-- The rows of the markdown table after heading, as lists of cells.
local function table_rows(md, heading)
	local at = assert(md:find("\n" .. heading .. "\n", 1, true), "no heading " .. heading)
	local rows, started = {}, false
	for line in md:sub(at + #heading + 2):gmatch("([^\n]*)\n") do
		if line:match("^|") then
			started = true
			local cells = {}
			for cell in line:gmatch("|%s*([^|]-)%s*%f[|]") do
				cells[#cells + 1] = cell
			end
			if not cells[1]:match("^%-%-%-") and cells[1] ~= "Field" then
				rows[#rows + 1] = cells
			end
		elseif started then
			break
		end
	end
	return rows
end

-- The server, with an engine that records each chat request.
local engine = fake.new({ delay = 0 })
local last
local rec = setmetatable({
	chat = function(_, req, sink)
		last = req
		return engine:chat(req, sink)
	end,
}, { __index = engine })
local srv = httpd.new()
local KEY = ("k"):rep(32)
openai.install(srv, rec, { key = KEY })

local HEADERS = { ["anthropic-version"] = "2023-06-01", ["x-api-key"] = "anything" }

local function post(body, headers, path)
	return inproc.json(srv, "POST", path or "/v1/messages", body, headers or HEADERS)
end

local function jtype(v)
	if v == json.null or v == nil then
		return "null"
	elseif type(v) == "table" then
		return json.isarray(v) and "array" or "object"
	end
	return type(v)
end

-- Checks that got has each field of want with the same JSON type.
local function shape(want, got, path, bad)
	local tw, tg = jtype(want), jtype(got)
	if tw ~= tg and not (tw == "null" or tg == "null") then
		bad[#bad + 1] = ("%s: want %s, got %s"):format(path, tw, tg)
		return
	end
	if tw == "object" then
		for k, v in pairs(want) do
			local p = path == "" and k or path .. "." .. k
			if got[k] == nil then
				bad[#bad + 1] = p .. ": missing"
			else
				shape(v, got[k], p, bad)
			end
		end
	end
end

local function same_shape(want, got, name)
	local bad = {}
	shape(want, got, "", bad)
	ok(#bad == 0, name .. (#bad > 0 and ": " .. table.concat(bad, "; ") or ""))
end

local STOP = { end_turn = true, max_tokens = true, stop_sequence = true, tool_use = true,
	model_context_window_exceeded = true }

-- The values that Anthropic fixes: ids, type, role, stop reason, usage.
local function check_values(b, name)
	local good = type(b.id) == "string" and b.id:match("^msg_%w+$") ~= nil and #b.id == 28 and
	    b.type == "message" and b.role == "assistant" and b.model == "fake" and STOP[b.stop_reason] and
	    json.isarray(b.content) and b.stop_details == json.null
	for _, c in ipairs(b.content) do
		if c.type == "tool_use" then
			good = good and c.id:match("^toolu_%w+$") and type(c.input) == "table" and not json.isarray(c.input)
		elseif c.type == "thinking" then
			good = good and translate.ours(c.signature)
		end
	end
	local u = b.usage
	good = good and math.type(u.input_tokens) == "integer" and math.type(u.output_tokens) == "integer" and
	    u.cache_creation_input_tokens == 0 and math.type(u.cache_read_input_tokens) == "integer"
	ok(good, name .. ": ids, type, role, stop reason and usage")
end

local function test_field_table()
	local rows = table_rows(read("docs/api.md"), "### Messages request fields")
	local docs = {}
	for _, r in ipairs(rows) do
		local name = r[1]:match("^`([%w_]+)`$")
		ok(name ~= nil, "request row names a field: " .. r[1])
		if name then
			docs[name] = { label = r[2], status = r[3] }
		end
	end
	local fields = {}
	for _, c in ipairs(anthropic.FIELDS) do
		fields[c[1]] = true
		local d = docs[c[1]]
		ok(d and d.status == c.status and d.label == c.label,
		    ("%s: docs say %s %s, the checks %s %s"):format(c[1], d and d.label, d and d.status, c.label, c.status))
	end
	for name in pairs(docs) do
		ok(fields[name], name .. " of the docs table is in FIELDS")
	end
end

local function test_examples()
	local status, b, h = post(example("plain_text"))
	ok(status == 200 and h["content-type"] == "application/json", "plain text: 200 with JSON")
	same_shape(EX.replies.plain_text, b, "plain text: the shape of the reference reply")
	check_values(b, "plain text")
	ok(h["request-id"] == "req_0000abcd", "the request-id header")
	ok(last.messages[1].role == "user" and last.messages[1].content == "Hello, Claude", "the chat message")
	ok(b.content[1].type == "thinking" and b.content[2].type == "text" and b.stop_reason == "end_turn",
	    "thinking comes before the text")
	ok(b.usage.input_tokens + b.usage.cache_read_input_tokens == 7, "usage counts the prompt")

	status, b = post(example("image_base64"))
	local parts = last.messages[1].content
	ok(status == 200 and parts[1].type == "image_url" and
	    parts[1].image_url.url == "data:image/png;base64," .. EX.png and parts[2].text == "Describe this image.",
	    "a base64 image becomes a data URL part")

	status, b = post(example("tool_request_1"))
	same_shape(EX.replies.tool_use, b, "tool call: the shape of the reference reply")
	check_values(b, "tool call")
	local call = b.content[#b.content]
	ok(b.stop_reason == "tool_use" and call.name == "get_weather" and call.input.city == "Paris",
	    "a tool_use block with an object input")
	local tool = last.tools[1]["function"]
	ok(tool.name == "get_weather" and tool.parameters.required[1] == "location" and
	    tool.description == "Get the current weather", "the tool becomes a chat function tool")

	status, b = post(example("tool_request_2"))
	local m = last.messages
	ok(status == 200 and m[2].role == "assistant" and m[2].content == "I'll check the weather in Paris." and
	    m[2].tool_calls[1].id == "toolu_01A09q90qw90lq917835lq9" and
	    json.decode(m[2].tool_calls[1]["function"].arguments).location == "Paris",
	    "an assistant tool_use becomes a chat tool call")
	ok(m[3].role == "tool" and m[3].tool_call_id == "toolu_01A09q90qw90lq917835lq9" and m[3].content == "20°C, sunny",
	    "a tool_result becomes a tool message")

	status = post(example("tool_result_image"))
	m = last.messages
	ok(status == 200 and m[3].role == "tool" and m[3].content[1].text == "Screenshot of the page:" and
	    m[3].content[2].type == "image_url" and m[4].role == "user" and
	    m[4].content == "Is the login button visible?", "a tool result with an image, then text")

	status, b = post(example("thinking"))
	same_shape(EX.replies.thinking, b, "thinking: the shape of the reference reply")
	ok(last.chat_template_kwargs.enable_thinking == true and last.reasoning_budget == nil, "adaptive: thinking on")
	status = post(example("thinking_enabled"))
	ok(status == 200 and last.reasoning_budget == 10000, "enabled: the budget caps the reasoning")
end

local function body(extra)
	local b = { model = "m", max_tokens = 100, messages = { { role = "user", content = "Hi" } } }
	for k, v in pairs(extra or {}) do
		b[k] = v
	end
	return b
end

local TOOLS = { { name = "get_weather", input_schema = { type = "object" } },
	{ name = "get_time", input_schema = { type = "object" } } }

local function test_requests()
	local status, b = post(body({ stop_sequences = { "zzz", "weather" } }))
	ok(status == 200 and b.stop_reason == "stop_sequence" and b.stop_sequence == "weather" and
	    b.content[2].text == "Let me check the ", "a stop sequence: " .. tostring(b.stop_reason))
	status, b = post(body({ max_tokens = 0 }))
	ok(status == 200 and #b.content == 0 and b.stop_reason == "max_tokens", "max_tokens 0 gives no content")
	status, b = post(body({ max_tokens = 3 }))
	ok(b.stop_reason == "max_tokens" and b.usage.output_tokens == 3, "max_tokens stops the reply")
	status, b = post(body({ thinking = { type = "disabled" } }))
	ok(status == 200 and b.content[1].type == "text" and last.chat_template_kwargs.enable_thinking == false,
	    "thinking disabled")
	status, b = post(body({ output_config = { effort = "none" } }))
	ok(b.content[1].type == "text" and last.reasoning_effort == "none", "effort none turns thinking off")
	local h
	status, b, h = post(body({ output_config = { effort = "high" } }))
	ok(status == 200 and h["x-lgml-ignored"] == "output_config.effort", "other efforts are named as ignored")
	status, b, h = post(body({ system = { { type = "text", text = "A", cache_control = { type = "ephemeral" } },
		{ type = "text", text = "B" } }, metadata = { user_id = "u" }, cache_control = { type = "ephemeral" },
		service_tier = "auto", temperature = 0.5, top_p = 0.9, top_k = 5 }))
	ok(status == 200 and last.messages[1].role == "system" and last.messages[1].content[2].text == "B" and
	    last.temperature == 0.5 and last.top_k == 5 and h["x-lgml-ignored"] == nil,
	    "system blocks, sampling and no-op fields")
	status, b = post(body({ output_config = { format = { type = "json_schema",
		schema = { type = "object", properties = { a = { type = "string" } } } } } }))
	ok(status == 200 and type(last.grammar) == "string", "output_config.format becomes a grammar")
	status, b, h = post(body({ context_management = { edits = { { type = "clear_tool_uses_20250919" } } } }))
	ok(status == 200 and h["x-lgml-ignored"] == "context_management", "context_management is accepted")
	status, b = post(body({ messages = { { role = "user", content = "A" }, { role = "user", content = "B" },
		{ role = "assistant", content = "C" }, { role = "system", content = "Be brief." },
		{ role = "user", content = "D" } } }))
	local m = last.messages
	ok(status == 200 and #m == 4 and m[1].content[2].text == "B" and m[3].role == "system" and m[4].content == "D",
	    "consecutive user messages merge; a system message stays in place")
end

local function test_tool_choice()
	local status, b = post(body({ tools = TOOLS, tool_choice = { type = "any" } }))
	ok(status == 200 and last.tool_choice == "required" and b.stop_reason == "tool_use", "any is required")
	status, b = post(body({ tools = TOOLS, tool_choice = { type = "tool", name = "get_time" } }))
	ok(b.content[#b.content].name == "get_time", "tool names the function")
	status, b = post(body({ tools = TOOLS, tool_choice = { type = "none" } }))
	ok(status == 200 and last.tools == nil and b.stop_reason == "end_turn", "none drops the tools")
	status, b = post(body({ tools = TOOLS, tool_choice = { type = "auto", disable_parallel_tool_use = true } }))
	ok(status == 200 and last.parallel_tool_calls == false, "disable_parallel_tool_use")
	status, b = post(body({ tools = TOOLS, tool_choice = { type = "tool", name = "nope" } }))
	ok(status == 400 and b.error.message:find("not in tools", 1, true), "an unknown tool name gives 400")
end

local function test_thinking_blocks()
	local _, b = post(body())
	local th = b.content[1]
	local function turn(block)
		return { messages = { { role = "user", content = "Hi" },
			{ role = "assistant", content = { block, { type = "text", text = "Hello." } } },
			{ role = "user", content = "Go on." } } }
	end
	local status, _, h = post(body(turn(th)))
	ok(status == 200 and last.messages[2].reasoning_content == "The user asks about Paris." and
	    h["x-lgml-thinking-dropped"] == nil, "a signed thinking block gives its reasoning to the template")
	local bad = { type = "thinking", thinking = "Something else.", signature = th.signature }
	local eb
	status, eb = post(body(turn(bad)))
	ok(status == 400 and eb.error.message == "messages.1.content.0: Invalid `signature` in `thinking` block",
	    "a changed thinking block gives 400")
	status, _, h = post(body(turn({ type = "thinking", thinking = "x", signature = "EqQBCgIYAhIM" })))
	ok(status == 200 and h["x-lgml-thinking-dropped"] == "1" and last.messages[2].reasoning_content == nil,
	    "a foreign signature is dropped and counted")
	status, _, h = post(body(turn({ type = "redacted_thinking", data = "EmwKAhgBEgy3va3pzix" })))
	ok(status == 200 and h["x-lgml-thinking-dropped"] == "1", "redacted_thinking is dropped and counted")
	_, b = post(body({ thinking = { type = "adaptive", display = "omitted" } }))
	th = b.content[1]
	ok(th.thinking == "" and translate.verify(KEY, "fake", th.signature, "") == "The user asks about Paris.",
	    "omitted: empty thinking, the text in the signature")
	post(body(turn(th)))
	ok(last.messages[2].reasoning_content == "The user asks about Paris.", "an omitted block round trips")
end

local function test_headers()
	local status, b = post(body(), { ["anthropic-version"] = "2020-01-01" })
	ok(status == 400 and b.type == "error" and b.error.message:find("2023-06-01", 1, true),
	    "an unknown anthropic-version gives 400")
	status = post(body(), {})
	ok(status == 200, "a request without anthropic-version works")
	local h
	status, b, h = post(body(), { ["anthropic-version"] = "2023-06-01", authorization = "Bearer x",
		["anthropic-beta"] = "context-1m-2025-08-07, oauth-2025-04-20", ["x-stainless-os"] = "Linux" })
	ok(status == 200 and h["x-lgml-ignored-betas"] == "context-1m-2025-08-07,oauth-2025-04-20",
	    "any beta value works and is listed")
	status = post(body(), HEADERS, "/v1/messages?beta=true")
	ok(status == 200, "the query string is ignored")
end

local function test_errors()
	for _, e in ipairs(EX.errors) do
		local status, b = post(e.body)
		ok(status == 400 and b.type == "error" and b.error.type == "invalid_request_error" and
		    b.request_id == "req_0000abcd" and b.error.message:find(e.message, 1, true),
		    ("%s: 400 %s"):format(e.name, tostring(b.error and b.error.message)))
	end
	local status, b = post(body({ messages = { { role = "user", content = "fail" } } }))
	ok(status == 500 and b.error.type == "api_error" and b.error.message == "fake failure", "an engine error")
	status, b = post("{")
	ok(status == 400 and b.error.type == "invalid_request_error", "bad JSON")
	translate.set_unknown_fields("ignore")
	local h
	status, b, h = post(body({ nope = 1, messages = { { role = "user", content = { { type = "text", text = "x",
		zap = 2 } } } } }))
	translate.set_unknown_fields("reject")
	ok(status == 200 and (h["x-lgml-ignored"] == "nope,messages.0.content.0.zap" or
	    h["x-lgml-ignored"] == "messages.0.content.0.zap,nope"), "ignore: unknown fields are named")
end

-- Checks the event rules of a Messages stream: message_start first,
-- then blocks with dense indexes, each started, grown and stopped in
-- turn, then message_delta and message_stop last. Pings can come
-- anywhere. Returns the message that the events build.
local function check_stream(text, name)
	local list, framed = inproc.events(text)
	ok(framed and #list > 0, name .. ": each event is an event line, a data line and a blank line")
	local msg, open, good, why, pings = nil, nil, true, "", 0
	local function bad(s)
		if good then
			good, why = false, s
		end
	end
	local stopped = false
	for i, ev in ipairs(list) do
		local d = ev.data
		if d.type ~= ev.name then
			bad("the data type is not the event name at " .. i)
		end
		if stopped then
			bad("an event after message_stop")
		end
		if ev.name == "ping" then
			pings = pings + 1
		elseif ev.name == "message_start" then
			if i ~= 1 and not (i == pings + 1) then
				bad("message_start is not first")
			end
			msg = d.message
			msg.content, msg.start_reason = {}, msg.stop_reason
		elseif not msg then
			bad(ev.name .. " before message_start")
		elseif ev.name == "content_block_start" then
			if open or d.index ~= #msg.content then
				bad("a block starts out of order at " .. i)
			end
			local cb = d.content_block
			msg.content[#msg.content + 1] = cb
			cb.json = {}
			open = d.index
		elseif ev.name == "content_block_delta" then
			local cb = msg.content[d.index + 1]
			if d.index ~= open then
				bad("a delta for a block that is not open at " .. i)
			elseif d.delta.type == "text_delta" then
				cb.text = cb.text .. d.delta.text
			elseif d.delta.type == "thinking_delta" then
				cb.thinking = cb.thinking .. d.delta.thinking
			elseif d.delta.type == "signature_delta" then
				cb.signature = cb.signature .. d.delta.signature
			elseif d.delta.type == "input_json_delta" then
				cb.json[#cb.json + 1] = d.delta.partial_json
			end
		elseif ev.name == "content_block_stop" then
			if d.index ~= open then
				bad("a stop for a block that is not open at " .. i)
			end
			local cb = msg.content[d.index + 1]
			if cb.type == "tool_use" then
				cb.input = json.decode(table.concat(cb.json))
			end
			cb.json = nil
			open = nil
		elseif ev.name == "message_delta" then
			if open then
				bad("message_delta while a block is open")
			end
			msg.stop_reason, msg.stop_sequence = d.delta.stop_reason, d.delta.stop_sequence
			msg.usage.output_tokens = d.usage.output_tokens
			msg.final_usage = d.usage
		elseif ev.name == "message_stop" then
			stopped = true
		elseif ev.name == "error" then
			msg.error = d.error
			stopped = true
		end
	end
	ok(good and (stopped or msg.error), name .. ": the event order" .. (good and "" or ": " .. why))
	return msg, list, pings
end

local function stream_post(b, headers)
	b.stream = true
	return inproc.request(srv, "POST", "/v1/messages", b, headers or HEADERS)
end

local function test_stream()
	local status, h, text = stream_post(example("plain_text"))
	ok(status == 200 and h["content-type"] == "text/event-stream" and h["request-id"] == "req_0000abcd",
	    "stream: 200 with an event stream and request-id")
	ok(not text:find("[DONE]", 1, true), "stream: no [DONE]")
	local msg = check_stream(text, "plain text stream")
	ok(msg.id:match("^msg_%w+$") and msg.start_reason == json.null and
	    msg.usage.input_tokens == 7 and msg.usage.output_tokens == 4,
	    "message_start has the id and the prompt size; message_delta the output")
	ok(msg.content[1].type == "thinking" and msg.content[1].thinking == "The user asks about Paris." and
	    translate.ours(msg.content[1].signature) and msg.content[2].text == "Let me check the weather." and
	    msg.stop_reason == "end_turn", "thinking and text blocks with their deltas")
	local ref = inproc.events(text)
	ok(ref[1].name == "message_start" and ref[#ref].name == "message_stop" and
	    ref[#ref - 1].name == "message_delta", "message_start first; message_delta and message_stop last")

	local _, _, ttext = stream_post(example("tool_request_1"))
	msg = check_stream(ttext, "tool stream")
	local call = msg.content[3]
	ok(call.type == "tool_use" and call.id:match("^toolu_%w+$") and call.name == "get_weather" and
	    call.input.city == "Paris" and msg.stop_reason == "tool_use",
	    "a tool_use block starts with its id and name; input_json_delta joins to the input")
	local starts = 0
	for _, ev in ipairs(inproc.events(ttext)) do
		if ev.name == "content_block_start" and ev.data.content_block.type == "tool_use" then
			starts = starts + 1
			ok(next(ev.data.content_block.input) == nil, "content_block_start has an empty input")
		end
	end
	ok(starts == 1, "one tool_use block")

	_, _, text = stream_post(body({ thinking = { type = "adaptive", display = "omitted" } }))
	msg = check_stream(text, "omitted thinking stream")
	ok(msg.content[1].thinking == "" and
	    translate.verify(KEY, "fake", msg.content[1].signature, "") == "The user asks about Paris.",
	    "omitted: one empty thinking_delta and the signature")

	_, _, text = stream_post(body({ stop_sequences = { "weather" } }))
	msg = check_stream(text, "stop sequence stream")
	ok(msg.stop_reason == "stop_sequence" and msg.stop_sequence == "weather", "stop_sequence in message_delta")
	local st, _, etext = stream_post(body({ messages = { { role = "user", content = "fail" } } }))
	ok(st == 500 and json.decode(etext).error.type == "api_error", "a failure before the stream is plain JSON")
end

-- A server whose engine waits, fails or works as the test says.
local function scripted(fn)
	local s = httpd.new()
	openai.install(s, {
		models = function() return { { id = "script" } } end,
		props = function() return {} end,
		chat = function(_, req, sink)
			return fn(req, sink)
		end,
	}, { key = KEY })
	return s
end

local function test_stream_events()
	local sleep = require("lgml.server.stream").sleep
	local s = scripted(function(_, sink)
		sink:prompt(50, 10)
		sink:content("Hi")
		return nil, "boom", 500
	end)
	local _, _, text = inproc.request(s, "POST", "/v1/messages", body({ stream = true }), HEADERS)
	local msg = check_stream(text, "error stream")
	ok(msg.error and msg.error.type == "api_error" and msg.error.message == "boom" and
	    msg.usage.input_tokens == 40 and msg.usage.cache_read_input_tokens == 10,
	    "an error after the start is an error event")
	local keep = anthropic.PING_MS
	anthropic.PING_MS = 20
	s = scripted(function(_, sink)
		sleep(120)
		sink:prompt(50, 0)
		sleep(120)
		sink:content("Hi")
		sink:finish("stop", { prompt_tokens = 50, completion_tokens = 1 })
		return true
	end)
	local pings
	_, _, text = inproc.request(s, "POST", "/v1/messages", body({ stream = true }), HEADERS)
	msg, _, pings = check_stream(text, "ping stream")
	anthropic.PING_MS = keep
	ok(pings >= 2 and msg.content[1].text == "Hi" and msg.final_usage.input_tokens == 50,
	    ("pings keep a slow stream alive (%d pings); message_delta has the input tokens"):format(pings))
end

test_field_table()
test_examples()
test_stream()
test_stream_events()
test_requests()
test_tool_choice()
test_thinking_blocks()
test_headers()
test_errors()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
