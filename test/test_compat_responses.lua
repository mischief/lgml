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

test_field_table()
test_examples()
test_requests()
test_errors()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
