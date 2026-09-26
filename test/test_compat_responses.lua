-- OpenAI Responses compatibility: the field table of docs/api.md
-- against the request checks, and the Responses reference examples
-- (test/fixtures/responses) against the server with the fake engine, in
-- process. TAP output.
local json = require("lgml.json")
local httpd = require("lgml.server.httpd")
local openai = require("lgml.server.openai")
local responses = require("lgml.server.responses")
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

local EX = assert(json.decode((read("test/fixtures/responses/examples.json"))))
local function example(name)
	local text = assert(json.encode(EX.requests[name])):gsub("%$PNG", EX.png)
	return assert(json.decode(text))
end

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

local engine = fake.new({ delay = 0 })
local last
local rec = setmetatable({
	chat = function(_, req, sink)
		last = req
		return engine:chat(req, sink)
	end,
}, { __index = engine })
local srv = httpd.new()
local KEY = ("r"):rep(32)
openai.install(srv, rec, { key = KEY })

local HEADERS = { authorization = "Bearer sk-local" }

local function post(body, path)
	return inproc.json(srv, "POST", path or "/v1/responses", body, HEADERS)
end

local function jtype(v)
	if v == json.null or v == nil then
		return "null"
	elseif type(v) == "table" then
		return json.isarray(v) and "array" or "object"
	end
	return type(v)
end

local function shape(want, got, path, bad)
	local tw, tg = jtype(want), jtype(got)
	if tw ~= tg and not (tw == "null" or tg == "null") and not (tw == "number" and tg == "number") then
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
	elseif tw == "array" and #want > 0 and #got > 0 then
		shape(want[1], got[#got], path .. "[]", bad)
	end
end

local function same_shape(want, got, name)
	local bad = {}
	shape(want, got, "", bad)
	ok(#bad == 0, name .. (#bad > 0 and ": " .. table.concat(bad, "; ") or ""))
end

-- Keys that Open Responses marks required on a Response.
local REQUIRED = { "id", "object", "created_at", "completed_at", "status", "incomplete_details", "model",
	"previous_response_id", "instructions", "output", "error", "tools", "tool_choice", "truncation",
	"parallel_tool_calls", "text", "top_p", "presence_penalty", "frequency_penalty", "top_logprobs",
	"temperature", "reasoning", "user", "usage", "max_output_tokens", "max_tool_calls", "store", "background",
	"service_tier", "metadata", "safety_identifier", "prompt_cache_key" }

local function check_values(b, name)
	local good = b.id:match("^resp_%w+$") and #b.id == 29 and b.object == "response" and
	    math.type(b.created_at) == "integer" and b.model == "fake" and json.isarray(b.output) and
	    b.output_text == nil
	for _, k in ipairs(REQUIRED) do
		good = good and b[k] ~= nil
	end
	for _, it in ipairs(b.output) do
		if it.type == "message" then
			good = good and it.id:match("^msg_%w+$") and it.role == "assistant" and it.content[1].type == "output_text"
		elseif it.type == "function_call" then
			good = good and it.id:match("^fc_%w+$") and it.call_id:match("^call_%w+$") and
			    type(it.arguments) == "string" and json.decode(it.arguments) ~= nil
		elseif it.type == "reasoning" then
			good = good and it.id:match("^rs_%w+$") and json.isarray(it.summary) and
			    translate.ours(it.encrypted_content)
		end
	end
	local u = b.usage
	good = good and u.total_tokens == u.input_tokens + u.output_tokens and
	    math.type(u.input_tokens_details.cache_write_tokens) == "integer" and
	    math.type(u.output_tokens_details.reasoning_tokens) == "integer"
	ok(good, name .. ": ids, required keys, items and usage")
end

local function test_field_table()
	local rows = table_rows(read("docs/api.md"), "### Responses request fields")
	local docs = {}
	for _, r in ipairs(rows) do
		local name = r[1]:match("^`([%w_]+)`$")
		ok(name ~= nil, "request row names a field: " .. r[1])
		if name then
			docs[name] = { label = r[2], status = r[3] }
		end
	end
	local fields = {}
	for _, c in ipairs(responses.FIELDS) do
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
	ok(last.messages[1].role == "system" and last.messages[1].content == "Be brief." and
	    last.messages[2].role == "user" and last.messages[2].content == "Say hi.", "instructions and a string input")
	ok(b.status == "completed" and b.instructions == "Be brief." and b.output[2].content[1].text ==
	    "Let me check the weather.", "a completed response echoes the instructions")

	status = post(example("image_data_url"))
	local parts = last.messages[1].content
	ok(status == 200 and parts[1].text == "What is in this image?" and parts[2].image_url.url == EX.png,
	    "an input_image data URL")

	status, b = post(example("tool_turn_1"))
	check_values(b, "tool call")
	local call = b.output[#b.output]
	same_shape(EX.replies.function_call, call, "the function_call item has the reference shape")
	ok(status == 200 and b.status == "completed" and call.name == "get_weather" and
	    json.decode(call.arguments).city == "Paris", "a tool call is a function_call item and status completed")
	ok(last.tools[1]["function"].name == "get_weather" and last.tools[1]["function"].strict == true and
	    b.tools[1].name == "get_weather", "the flat tool becomes a chat tool and is echoed")

	status = post(example("tool_turn_2"))
	local m = last.messages
	ok(status == 200 and m[2].role == "assistant" and m[2].tool_calls[1].id == "call_02" and
	    m[3].role == "tool" and m[3].tool_call_id == "call_02" and m[3].content == '{"temp_c":18,"sky":"clear"}',
	    "function_call and function_call_output items")

	status = post(example("tool_result_image"))
	m = last.messages
	ok(status == 200 and m[3].role == "tool" and m[3].content[1].text == "Screenshot of display :0" and
	    m[3].content[2].type == "image_url", "an output list with an image")

	status, b = post(example("reasoning"))
	local rs = b.output[1]
	same_shape(EX.replies.reasoning, rs, "the reasoning item has the reference shape")
	ok(rs.summary[1].text == "The user asks about Paris." and rs.content[1].type == "reasoning_text" and
	    translate.verify(KEY, "fake", rs.encrypted_content) == "The user asks about Paris.",
	    "the reasoning item: summary, content and encrypted_content")
	local again = example("reasoning")
	again.input[2] = rs
	again.input[3] = b.output[2]
	again.input[4] = { role = "user", content = "And 1009?" }
	status = post(again)
	m = last.messages
	ok(status == 200 and m[2].role == "assistant" and m[2].reasoning_content == "The user asks about Paris." and
	    m[2].content == "Let me check the weather." and m[3].content == "And 1009?",
	    "the output items echo back as one assistant turn with its reasoning")
	status, b = post(example("codex"))
	ok(status == 200 and last.messages[1].content == "You are Codex." and last.messages[2].role == "system" and
	    last.parallel_tool_calls == false and last.reasoning_effort == "high", "a Codex request")
	status, b = post(example("clm"))
	ok(status == 200 and last.max_completion_tokens == 64000 and last.messages[1].role == "system" and
	    b.output[1].summary[1].text ~= nil, "a clm request")
end

local function body(extra)
	local b = { model = "m", input = "Hi" }
	for k, v in pairs(extra or {}) do
		b[k] = v
	end
	return b
end

local TOOLS = { { type = "function", name = "get_weather", parameters = { type = "object" } },
	{ type = "function", name = "get_time" } }

local function test_requests()
	local status, b = post(body({ max_output_tokens = 3 }))
	ok(status == 200 and b.status == "incomplete" and b.incomplete_details.reason == "max_output_tokens" and
	    b.output[#b.output].status == "incomplete" and b.completed_at == json.null, "the token limit: incomplete")
	status, b = post(body({ reasoning = { effort = "none" } }))
	ok(status == 200 and b.output[1].type == "message" and last.reasoning_effort == "none", "effort none")
	status, b = post(body({ tools = TOOLS, tool_choice = "required" }))
	ok(last.tool_choice == "required" and b.output[#b.output].type == "function_call", "tool_choice required")
	status, b = post(body({ tools = TOOLS, tool_choice = { type = "function", name = "get_time" } }))
	ok(b.output[#b.output].name == "get_time", "a named function")
	status, b = post(body({ tools = TOOLS, tool_choice = { type = "allowed_tools", mode = "required",
		tools = { { type = "function", name = "get_time" } } } }))
	ok(status == 200 and #last.tools == 1 and b.output[#b.output].name == "get_time", "allowed_tools")
	status, b = post(body({ tools = TOOLS, tool_choice = "none" }))
	ok(last.tools == nil and b.output[#b.output].type == "message", "tool_choice none")
	status, b = post(body({ text = { format = { type = "json_schema", name = "p",
		schema = { type = "object", properties = { a = { type = "string" } } }, strict = true } } }))
	ok(status == 200 and type(last.grammar) == "string" and b.text.format.name == "p",
	    "a flat json_schema format becomes a grammar")
	status = post(body({ text = { format = { type = "json_object" } } }))
	ok(status == 200 and type(last.grammar) == "string", "json_object")
	local h
	status, b, h = post(body({ tools = { { type = "function", name = "f", defer_loading = false } } }))
	ok(status == 200 and h["x-lgml-ignored"] == "tools.defer_loading", "a no-effect tool field is named")
	status, b = post(body({ metadata = { a = "b" }, user = "u", safety_identifier = "s", service_tier = "auto",
		prompt_cache_key = "k", max_tool_calls = 3, client_metadata = { x = "y" }, top_logprobs = 0,
		truncation = "disabled", background = false, stream_options = json.null }))
	ok(status == 200 and b.metadata.a == "b" and b.max_tool_calls == 3, "no-op fields are accepted and echoed")
	status, b = post(body({ input = { { role = "developer", content = "Rules." },
		{ type = "message", role = "assistant", id = "msg_1", status = "completed", phase = "final_answer",
			content = { { type = "output_text", text = "Earlier.", annotations = {} } } },
		{ role = "user", content = { { type = "input_file", filename = "a.txt",
			file_data = "data:text/plain;base64,aGVsbG8=" } } } } }))
	local m = last.messages
	ok(status == 200 and m[1].role == "system" and m[2].content == "Earlier." and m[3].content == "a.txt:\nhello",
	    "developer, a replayed assistant message and a text file")
	status, b = post(body({ input = { { role = "user", content = "fail" } } }))
	ok(status == 500 and b.error.type == "server_error" and b.error.message == "fake failure", "an engine error")
end

local function test_errors()
	for _, e in ipairs(EX.errors) do
		local status, b = post(e.body)
		ok(status == 400 and b.error and b.error.type == "invalid_request_error" and b.error.param == e.param and
		    (e.code == nil or b.error.code == e.code),
		    ("%s: 400 param %s: %s"):format(e.name, tostring(b.error and b.error.param),
		    tostring(b.error and b.error.message)))
	end
end

-- The event patterns of one item, after output_item.added.
local ITEM_EVENTS = {
	message = { "response.content_part.added", "response.output_text.delta*", "response.output_text.done",
		"response.content_part.done", "response.output_item.done" },
	reasoning = { "response.reasoning_summary_part.added", "response.reasoning_summary_text.delta*",
		"response.reasoning_summary_text.done", "response.reasoning_summary_part.done",
		"response.output_item.done" },
	function_call = { "response.function_call_arguments.delta*", "response.function_call_arguments.done",
		"response.output_item.done" },
}

-- Checks a Responses stream: framing, sequence numbers from 0, the
-- start and end events, and the event order of each item. Returns the
-- events, the output built from output_item.done, and the comment count.
local function check_stream(text, name)
	local list, framed, comments = inproc.events(text)
	ok(framed and #list >= 3, name .. ": each event is an event line, a data line and a blank line")
	local good, why = true, ""
	local function bad(s)
		if good then
			good, why = false, s
		end
	end
	for i, ev in ipairs(list) do
		if ev.data.type ~= ev.name or ev.data.sequence_number ~= i - 1 then
			bad(("event %d: type %s, sequence %s"):format(i, tostring(ev.data.type),
			    tostring(ev.data.sequence_number)))
		end
		if ev.name ~= "error" and ev.data.error ~= nil then
			bad("a top-level error key on " .. ev.name)
		end
	end
	if not (list[1] and list[1].name == "response.created" and list[2].name == "response.in_progress") then
		bad("the stream does not start with response.created and response.in_progress")
	end
	local output, i = {}, 3
	while list[i] and list[i].name == "response.output_item.added" do
		local d = list[i].data
		local item = d.item
		if d.output_index ~= #output then
			bad("output_index out of order")
		end
		if item.type == "function_call" and not (item.name and item.call_id and item.arguments == "") then
			bad("output_item.added of a call lacks name, call_id or empty arguments")
		end
		i = i + 1
		for _, want in ipairs(ITEM_EVENTS[item.type] or {}) do
			local many = want:sub(-1) == "*"
			want = many and want:sub(1, -2) or want
			if many then
				while list[i] and list[i].name == want do
					if list[i].data.output_index ~= d.output_index or list[i].data.item_id ~= item.id then
						bad("a delta for another item")
					end
					i = i + 1
				end
			elseif not list[i] or list[i].name ~= want then
				bad(("want %s, got %s"):format(want, tostring(list[i] and list[i].name)))
				break
			else
				i = i + 1
			end
		end
		output[#output + 1] = list[i - 1].data.item
	end
	local last = list[i]
	if not (last and (last.name == "response.completed" or last.name == "response.incomplete" or
	    last.name == "error" or last.name == "response.failed") ) then
		bad("an unexpected event: " .. tostring(last and last.name))
	elseif last.name == "error" and not (list[i + 1] and list[i + 1].name == "response.failed") then
		bad("error without response.failed")
	end
	if last and last.name ~= "error" and list[i + 1] then
		bad("events after the end")
	end
	ok(good, name .. ": the event order" .. (good and "" or ": " .. why))
	return list, output, comments
end

local function stream_post(b)
	b.stream = true
	return inproc.request(srv, "POST", "/v1/responses", b, HEADERS)
end

local function test_stream()
	local status, h, text = stream_post(example("plain_text"))
	ok(status == 200 and h["content-type"] == "text/event-stream" and not text:find("[DONE]", 1, true),
	    "stream: an event stream with no [DONE]")
	local list, output = check_stream(text, "text stream")
	local done = list[#list].data.response
	ok(list[#list].name == "response.completed" and done.status == "completed" and done.usage.total_tokens > 0,
	    "response.completed has the usage")
	ok(#output == 2 and output[1].type == "reasoning" and output[2].content[1].text == "Let me check the weather." and
	    json.encode(output[2]) == json.encode(done.output[2]), "output_item.done items are those of the response")
	ok(list[1].data.response.status == "in_progress" and list[1].data.response.usage == json.null,
	    "response.created is in progress with no usage")

	local _, _, ttext = stream_post(example("tool_turn_1"))
	local tl, tout = check_stream(ttext, "tool stream")
	local names = {}
	for _, ev in ipairs(tl) do
		if ev.data.output_index == 2 or ev.name:find("^response%.[ci][on]") then
			names[#names + 1] = ev.name
		end
	end
	local want = EX.stream_tool
	ok(table.concat(names, ","):find(table.concat({ want[3], want[4], want[5], want[6], want[7] }, ","), 1, true),
	    "a function call streams as the reference transcript: " .. table.concat(names, ","))
	ok(tout[3].type == "function_call" and json.decode(tout[3].arguments).city == "Paris",
	    "the call item has the arguments")

	_, _, text = stream_post(body({ max_output_tokens = 3 }))
	list = check_stream(text, "incomplete stream")
	ok(list[#list].name == "response.incomplete" and
	    list[#list].data.response.incomplete_details.reason == "max_output_tokens", "response.incomplete")

	local st, _, etext = stream_post(body({ input = { { role = "user", content = "fail" } } }))
	ok(st == 500 and json.decode(etext).error.message == "fake failure", "a failure before the stream is JSON")
end

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
		sink:prompt(10, 0)
		sink:content("Hi")
		return nil, "boom", 500
	end)
	local _, _, text = inproc.request(s, "POST", "/v1/responses", body({ stream = true }), HEADERS)
	local list = inproc.events(text)
	local e, f = list[#list - 1], list[#list]
	ok(e.name == "error" and e.data.code == "server_error" and e.data.message == "boom" and
	    e.data.error.message == "boom" and f.name == "response.failed" and f.data.response.status == "failed" and
	    f.data.response.error.message == "boom", "an error after the start: error, then response.failed")
	local keep = responses.KEEPALIVE_MS
	responses.KEEPALIVE_MS = 20
	s = scripted(function(_, sink)
		sleep(120)
		sink:content("Hi")
		sink:finish("stop", { prompt_tokens = 5, completion_tokens = 1 })
		return true
	end)
	_, _, text = inproc.request(s, "POST", "/v1/responses", body({ stream = true }), HEADERS)
	responses.KEEPALIVE_MS = keep
	local _, output, comments = check_stream(text, "keepalive stream")
	ok(comments >= 2 and output[1].content[1].text == "Hi", ("keepalive comments (%d)"):format(comments))
end

local function get(path)
	return inproc.json(srv, "GET", path, nil, HEADERS)
end

local function test_store()
	local _, r1 = post(body({ instructions = "Be brief.", input = "Hi" }))
	local status, r2 = post(body({ previous_response_id = r1.id, input = "And then?" }))
	local m = last.messages
	ok(status == 200 and #m == 3 and m[1].role == "user" and m[1].content == "Hi" and m[2].role == "assistant" and
	    m[2].content == "Let me check the weather." and m[2].reasoning_content == "The user asks about Paris." and
	    m[3].content == "And then?" and r2.previous_response_id == r1.id,
	    "previous_response_id puts the stored input and output first, without the old instructions")
	status = post(body({ previous_response_id = r2.id, input = "Last." }))
	ok(status == 200 and #last.messages == 5, "a chain of three responses")

	local _, t1 = post(body({ tools = TOOLS, input = "Weather?" }))
	local call = t1.output[#t1.output]
	status = post(body({ tools = TOOLS, previous_response_id = t1.id,
		input = { { type = "function_call_output", call_id = call.call_id, output = "sunny" } } }))
	m = last.messages
	ok(status == 200 and m[2].tool_calls[1].id == call.call_id and m[3].role == "tool" and m[3].content == "sunny",
	    "a function_call_output after a stored call")
	local again
	status, again = post(body({ tools = TOOLS, previous_response_id = t1.id,
		input = { { type = "function_call", call_id = call.call_id, name = call.name, arguments = call.arguments },
			{ type = "function_call_output", call_id = call.call_id, output = "sunny" } } }))
	m = last.messages
	ok(status == 200 and #m == 3 and #m[2].tool_calls == 1 and m[3].role == "tool",
	    "a stored call sent again with its output, as the clm client sends it, is not a second call")
	local _, items = get("/v1/responses/" .. again.id .. "/input_items")
	ok(#items.data == 1 and items.data[1].type == "function_call_output", "the input items skip the known call")

	local _, nostore = post(body({ store = false }))
	local e
	status, e = post(body({ previous_response_id = nostore.id }))
	ok(status == 400 and e.error.code == "previous_response_not_found" and e.error.param == "previous_response_id" and
	    e.error.message:find("previous_response", 1, true) == nil and e.error.message:find(nostore.id, 1, true),
	    "store false keeps nothing; its id gives previous_response_not_found")
	ok(nostore.store == false and r1.store == true, "the response echoes store")

	local got
	status, got = get("/v1/responses/" .. r1.id)
	ok(status == 200 and got.id == r1.id and got.output[2].content[1].text == r1.output[2].content[1].text,
	    "GET gives the stored response")
	status, got = get("/v1/responses/" .. r2.id .. "/input_items")
	ok(status == 200 and got.object == "list" and #got.data == 1 and got.data[1].id:match("^msg_") and
	    got.data[1].content == "And then?" and got.has_more == false and got.first_id == got.data[1].id,
	    "input_items lists the input of that response only")
	local _, many = post(body({ input = { { role = "user", content = "a" }, { role = "user", content = "b" },
		{ role = "user", content = "c" } } }))
	status, got = get("/v1/responses/" .. many.id .. "/input_items?limit=2&order=asc")
	ok(#got.data == 2 and got.data[1].content == "a" and got.has_more == true, "input_items: limit and order")
	status, got = get(("/v1/responses/%s/input_items?after=%s"):format(many.id, got.data[2].id))
	ok(#got.data == 1 and got.data[1].content == "a", "input_items: after, newest first by default")

	local ref = r1.output[2].id
	status = post(body({ input = { { type = "item_reference", id = ref }, { role = "user", content = "Go on." } } }))
	ok(status == 200 and last.messages[1].role == "assistant" and last.messages[1].content ==
	    "Let me check the weather.", "item_reference takes a stored item")

	status, got = inproc.json(srv, "POST", "/v1/responses/" .. r1.id .. "/cancel", {}, HEADERS)
	ok(status == 400 and got.error.message:find("background", 1, true), "cancel: no background responses")
	status, got = inproc.json(srv, "DELETE", "/v1/responses/" .. r1.id, nil, HEADERS)
	ok(status == 200 and got.deleted == true and got.object == "response.deleted" and got.id == r1.id, "DELETE")
	status, got = get("/v1/responses/" .. r1.id)
	ok(status == 404 and got.error.type == "not_found_error", "a deleted response is gone")
	status, e = post(body({ previous_response_id = r2.id }))
	ok(status == 400 and e.error.code == "previous_response_not_found", "a chain with a deleted link fails")

	local _, _, text = inproc.request(srv, "POST", "/v1/responses", body({ stream = true, input = "S" }), HEADERS)
	local list = inproc.events(text)
	local sid = list[#list].data.response.id
	status, got = get("/v1/responses/" .. sid)
	ok(status == 200 and got.status == "completed", "a streamed response is stored")

	local _, made = post(body({ input = example("image_data_url").input }))
	local _, count = post(body({ input = example("image_data_url").input }), "/v1/responses/input_tokens")
	ok(count.object == "response.input_tokens" and count.input_tokens == made.usage.input_tokens,
	    ("input_tokens counts the prompt: %s"):format(tostring(count.input_tokens)))
end

local function test_store_limits()
	local store = require("lgml.server.response_store")
	local now = 1000
	local s = store.new({ entries = 2, age = 60 }, function() return now end)
	local function put(id, prev)
		s:put({ id = id, prev = prev, input = { { type = "message", id = "i" .. id, role = "user", content = id } },
			output = {}, response = { id = id } })
	end
	put("a")
	put("b", "a")
	s:get("a")
	put("c", "b")
	ok(s:get("b") == nil and s:get("a") and s:get("c"), "the least recently used entry goes first")
	local chain, missing = s:chain("c")
	ok(chain == nil and missing == "b", "a chain with a gap names the missing id")
	ok(s:item("ia") and s:item("ic") and not s:item("ib"), "stored items by id")
	now = now + 61
	ok(s:get("a") == nil and s:get("c") == nil and s.count == 0, "entries past the age limit are gone")
	local b = store.new({ bytes = 300 })
	for i = 1, 5 do
		b:put({ id = "r" .. i, input = { { content = ("x"):rep(100) } }, output = {}, response = {} })
	end
	ok(b.bytes <= 300 and b:get("r5") and not b:get("r1"), "the byte limit drops old entries")
end

test_field_table()
test_examples()
test_requests()
test_errors()
test_stream()
test_stream_events()
test_store()
test_store_limits()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
