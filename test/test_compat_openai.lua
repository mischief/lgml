-- OpenAI Chat Completions compatibility: the field table of docs/api.md
-- against the request checks and the openai-python request fields.
-- TAP output.

local uv = require("luv")
local json = require("lgml.json")
local chat_request = require("lgml.server.chat_request")
local http = require("lgml.server.http")
local httpd = require("lgml.server.httpd")
local openai = require("lgml.server.openai")
local fake = require("lgml.server.fake_engine")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local root = (arg[0]:match("^(.*)/[^/]*$") or ".") .. "/.."

local function read(path)
	local f = assert(io.open(root .. "/" .. path, "rb"))
	local s = f:read("a")
	f:close()
	return s
end

local function fixture(name)
	return assert(json.decode(read("test/fixtures/openai/" .. name)))
end

-- The rows of the markdown table after heading, as lists of cells.
local function table_rows(md, heading)
	local at = md:find("\n" .. heading .. "\n", 1, true)
	assert(at, "no heading " .. heading)
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

-- A deep copy, so each check sees fresh tables.
local function copy(v)
	if type(v) ~= "table" then
		return v
	end
	local t = {}
	for k, x in pairs(v) do
		t[k] = copy(x)
	end
	return setmetatable(t, getmetatable(v))
end

local function body(extra)
	local b = { messages = json.array({ { role = "user", content = "Hi" } }) }
	for k, v in pairs(extra or {}) do
		b[k] = copy(v)
	end
	return b
end

local ACCEPTED = { supported = true, limited = true, ["no-op"] = true }

-- Fields that the good value of a field needs, when lgml accepts it.
local COMPANIONS = {
	tool_choice = { tools = json.array({ { type = "function", ["function"] = { name = "f" } } }) },
	top_logprobs = { logprobs = true },
	stream_options = { stream = true },
}

local function test_field_table()
	local md = read("docs/api.md")
	local rows = table_rows(md, "### Request fields")
	local docs = {}
	for _, r in ipairs(rows) do
		local name = r[1]:match("^`([%w_]+)`$")
		ok(name ~= nil, "request row names a field: " .. r[1])
		if name then
			docs[name] = { label = r[2], status = r[3] }
		end
	end
	local fields = {}
	for _, c in ipairs(chat_request.FIELDS) do
		fields[c[1]] = c
		local d = docs[c[1]]
		ok(d and d.status == c.status and d.label == c.label,
		    ("%s: docs say %s %s, the checks %s %s"):format(c[1], d and d.label, d and d.status,
		    c.label, c.status))
		ok(chat_request.STATUS[c.status] ~= nil, c[1] .. " has a known status")
	end
	for name in pairs(docs) do
		ok(fields[name] ~= nil, name .. " of the docs table is in FIELDS")
	end

	local sdk = fixture("request_fields.json").fields
	ok(#sdk >= 36, "the SDK list has the request fields")
	for _, f in ipairs(sdk) do
		local c = fields[f.name]
		ok(c ~= nil and c.label == "OpenAI", f.name .. ": OpenAI field in FIELDS")
		if c then
			local extra = {}
			if ACCEPTED[c.status] then
				for k, v in pairs(COMPANIONS[f.name] or {}) do
					extra[k] = v
				end
			end
			extra[f.name] = f.good
			local b = f.name == "messages" and { messages = copy(f.good) } or body(extra)
			local good, err, param = chat_request.check(b)
			if ACCEPTED[c.status] then
				ok(good == true, ("%s (%s) accepts %s: %s"):format(f.name, c.status, json.encode(f.good),
				    tostring(err)))
			else
				ok(not good and param == f.name, ("%s (%s) gives 400 for %s: %s"):format(f.name,
				    c.status, json.encode(f.good), tostring(err)))
			end
			if f.limit then
				good, err, param = chat_request.check(body({ [f.name] = f.limit }))
				ok(c.status == "limited" and not good and param == f.name,
				    ("%s: the limit value gives 400: %s"):format(f.name, tostring(err)))
			end
		end
	end
	local _, err, param, code = chat_request.check(body({ nope = 1 }))
	ok(param == "nope" and err:find("unknown field") and code == "unknown_parameter",
	    "an unknown field gives 400 with code unknown_parameter")
	_, _, param, code = chat_request.check(body({ verbosity = "low" }))
	ok(param == "verbosity" and code == "unsupported_value", "a rejected value has code unsupported_value")
	_, _, param, code = chat_request.check(body({ temperature = -1 }))
	ok(param == "temperature" and code == "invalid_value", "a bad value has code invalid_value")
	_, _, param = chat_request.check(body({ stream_options = { include_usage = true } }))
	ok(param == "stream_options", "stream_options needs stream")

end

local function test_messages()
	local function msg_check(m)
		return chat_request.check({ messages = json.array({ m }) })
	end
	local good = {
		{ role = "user", content = "hi", name = "ann" },
		{ role = "user", content = json.array({ { type = "text", text = "hi" },
			{ type = "image_url", image_url = { url = "data:image/png;base64,AA", detail = "low" } } }) },
		{ role = "assistant", content = json.null, refusal = json.null, annotations = json.array(),
			tool_calls = json.array({ { id = "call_1", type = "function",
				["function"] = { name = "f", arguments = "{}" } } }) },
		{ role = "tool", tool_call_id = "call_1", content = json.array({ { type = "text", text = "ok" } }) },
		{ role = "developer", content = "Be brief." },
		{ role = "user", content = json.array({ { type = "image_url",
			image_url = { url = "data:image/png;base64,AA", detail = "original" } } }) },
	}
	for i, m in ipairs(good) do
		local r, err = msg_check(m)
		ok(r == true, ("message %d is accepted: %s"):format(i, tostring(err)))
	end
	local m = { role = "assistant", content = "x", reasoning = "because" }
	ok(msg_check(m) and m.reasoning_content == "because" and m.reasoning == nil,
	    "assistant reasoning becomes reasoning_content")
	local bad = {
		{ { role = "function", name = "f", content = "x" }, "role" },
		{ { role = "user", content = json.array({ { type = "input_audio",
			input_audio = { data = "AA", format = "wav" } } }) }, "input_audio" },
		{ { role = "user", content = json.array({ { type = "file", file = { file_id = "f" } } }) }, "file" },
		{ { role = "assistant", content = "x", audio = { id = "a" } }, "audio" },
		{ { role = "assistant", content = "x", refusal = "no" }, "refusal" },
		{ { role = "user", content = "x", extra = 1 }, "extra" },
	}
	for _, c in ipairs(bad) do
		local r, err, param = msg_check(c[1])
		ok(not r and param == "messages" and err:find(c[2], 1, true),
		    ("message with %s gives 400: %s"):format(c[2], tostring(err)))
	end
end

-- in-process requests

local srv = httpd.new()
local engine = fake.new({ delay = 0 })
openai.install(srv, engine)

-- Runs POST /v1/chat/completions with body (a table or JSON text) on
-- the fake engine, without a socket. Returns the status, the headers
-- and the body text.
local function post(body)
	local out = {}
	local s = {
		write = function(_, data)
			out[#out + 1] = data
			return true
		end,
		gone = function()
			return false
		end,
	}
	local text = type(body) == "string" and body or assert(json.encode(body))
	local req = { method = "POST", path = "/v1/chat/completions", version = "1.0", body = text,
		headers = {} }
	local res = http.response({ s = s }, req)
	local done
	coroutine.wrap(function()
		srv.routes[req.path].POST(req, res)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	local raw = table.concat(out)
	local head, rest = raw:match("^(.-)\r\n\r\n(.*)$")
	local headers = {}
	for k, v in head:gmatch("\r\n([^:\r\n]+): ([^\r\n]*)") do
		headers[k:lower()] = v
	end
	return tonumber(head:match("^HTTP/1%.1 (%d+)")), headers, rest
end

local function post_json(body)
	local status, headers, text = post(body)
	return status, json.decode(text), headers
end

-- The events of a streamed reply: each is decoded JSON, or "[DONE]".
-- Returns the events and true when each is a "data: ...\n\n" line.
local function events(text)
	local list, framed, pos = {}, true, 1
	while pos <= #text do
		local line, next_pos = text:match("^data: ([^\n]*)\n\n()", pos)
		if not line then
			framed = false
			break
		end
		list[#list + 1] = line == "[DONE]" and line or json.decode(line)
		pos = next_pos
	end
	return list, framed
end

-- The content, finish reasons and chunks of a stream, by choice index.
local function collect(list)
	local c = { content = {}, finish = {}, chunks = 0 }
	for _, ev in ipairs(list) do
		if type(ev) == "table" and ev.choices then
			c.chunks = c.chunks + 1
			for _, ch in ipairs(ev.choices) do
				local i = ch.index
				if type(ch.delta.content) == "string" then
					c.content[i] = (c.content[i] or "") .. ch.delta.content
				end
				if ch.finish_reason ~= json.null then
					c.finish[i] = ch.finish_reason
				end
			end
			if ev.usage and ev.usage ~= json.null then
				c.usage = ev.usage
			end
		end
	end
	return c
end

local TOOLS = json.array({ { type = "function", ["function"] = {
	name = "get_weather", parameters = { type = "object",
	properties = { city = { type = "string" } }, required = json.array({ "city" }) } } } })

-- The value at a path such as "choices[].message.content"; [] takes
-- the first entry. nil when a step is missing; json.null stays.
local function at(v, path)
	for step in path:gmatch("[^.]+") do
		local name, arr = step:match("^([%w_]+)(.*)$")
		if type(v) ~= "table" then
			return nil
		end
		v = v[name]
		if arr == "[]" then
			v = type(v) == "table" and v[1] or nil
		end
	end
	return v
end

local function test_reply_table()
	local rows = table_rows(read("docs/api.md"), "### Reply fields")
	local status, b = post_json(body({ tools = TOOLS }))
	ok(status == 200, "a reply with a tool call")
	ok(#rows >= 20, "the reply table has rows")
	for _, r in ipairs(rows) do
		local path = r[1]:match("^`(.+)`$")
		ok(path and at(b, path) ~= nil, ("reply has %s"):format(tostring(path)))
	end
end

local function test_stop()
	local es, eb = post_json(body({ nope = 1 }))
	ok(es == 400 and eb.error.code == "unknown_parameter" and eb.error.param == "nope" and
	    eb.error.type == "invalid_request_error", "the error object has the code")
	local status, b = post_json(body({ stop = "weather" }))
	local ch = b.choices[1]
	ok(status == 200 and ch.message.content == "Let me check the " and ch.finish_reason == "stop",
	    "stop cuts the content: " .. tostring(ch.message.content))
	local _, _, text = post(body({ stream = true, stop = json.array({ "zzz", "check" }) }))
	local c = collect(events(text))
	ok(c.content[0] == "Let me " and c.finish[0] == "stop", "stop in a stream")
end

local function test_logprobs()
	local status, b = post_json(body({ logprobs = true, top_logprobs = 2 }))
	local lp = b.choices[1].logprobs
	ok(status == 200 and lp.refusal == json.null and #lp.content == 2, "logprobs lists the content tokens")
	local e = lp.content[1]
	ok(e.token == "Let me " and e.logprob == -0.5 and #e.bytes == 7 and e.bytes[1] == 76 and
	    #e.top_logprobs == 2 and e.top_logprobs[2].token == "x", "a logprobs entry has the OpenAI shape")
	local _, _, text = post(body({ stream = true, logprobs = true }))
	local n, empty_top = 0, true
	for _, ev in ipairs(events(text)) do
		if type(ev) == "table" and ev.choices and ev.choices[1] then
			local l = ev.choices[1].logprobs
			if l ~= json.null then
				n = n + #l.content
				empty_top = empty_top and #l.content[1].top_logprobs == 0
			end
		end
	end
	ok(n == 2 and empty_top, "stream chunks carry logprobs; top_logprobs defaults to none")
	status, b = post_json(body({ top_logprobs = 2 }))
	ok(status == 400 and b.error.param == "top_logprobs", "top_logprobs needs logprobs")
	status, b = post_json(body())
	ok(b.choices[1].logprobs == json.null, "logprobs are null when not asked")
end

local function test_tool_choice()
	local two = json.array({ TOOLS[1], { type = "function", ["function"] = { name = "get_time" } } })
	local status, b = post_json(body({ tools = two, tool_choice = "none" }))
	ok(status == 200 and b.choices[1].message.tool_calls == nil and b.choices[1].finish_reason == "stop",
	    "tool_choice none gives no calls")
	status, b = post_json(body({ tools = two, tool_choice = { type = "function",
		["function"] = { name = "get_time" } } }))
	ok(b.choices[1].message.tool_calls[1]["function"].name == "get_time", "a named tool_choice")
	status, b = post_json(body({ tools = two, tool_choice = { type = "allowed_tools", allowed_tools = {
		mode = "required", tools = json.array({ { type = "function", ["function"] = { name = "get_time" } } }) } } }))
	ok(status == 200 and b.choices[1].message.tool_calls[1]["function"].name == "get_time",
	    "allowed_tools keeps the named tools")
	local bad = {
		{ { tool_choice = "required" }, "needs tools" },
		{ { tools = two, tool_choice = { type = "function", ["function"] = { name = "nope" } } }, "not in tools" },
		{ { tools = two, tool_choice = { type = "custom", custom = { name = "x" } } }, "custom" },
		{ { tools = two, tool_choice = "sometimes" }, "tool_choice" },
		{ { tools = two, tool_choice = { type = "allowed_tools", allowed_tools = { mode = "auto",
			tools = json.array({ { type = "function", ["function"] = { name = "nope" } } }) } } }, "not in tools" },
	}
	for _, c in ipairs(bad) do
		status, b = post_json(body(c[1]))
		ok(status == 400 and b.error.param == "tool_choice" and b.error.message:find(c[2], 1, true),
		    "bad tool_choice gives 400: " .. tostring(b.error and b.error.message))
	end
	status, b = post_json(body({ tools = two, parallel_tool_calls = false }))
	ok(status == 200 and #b.choices[1].message.tool_calls == 1, "parallel_tool_calls false")
end

local function test_n()
	local status, b = post_json(body({ n = 3, tools = TOOLS }))
	local idx = {}
	for i, ch in ipairs(b.choices) do
		idx[i] = ch.index .. ":" .. ch.finish_reason
	end
	ok(status == 200 and table.concat(idx, ",") == "0:tool_calls,1:tool_calls,2:tool_calls",
	    "n gives that many choices")
	ok(b.usage.completion_tokens == 21 and b.usage.completion_tokens_details.reasoning_tokens == 6,
	    "usage counts the tokens of each choice")
	local _, _, text = post(body({ n = 2, stream = true, stream_options = { include_usage = true } }))
	local list = events(text)
	local c = collect(list)
	local roles = 0
	for _, ev in ipairs(list) do
		if type(ev) == "table" and ev.choices[1] and ev.choices[1].delta.role then
			roles = roles + 1
		end
	end
	ok(c.finish[0] == "stop" and c.finish[1] == "stop" and roles == 2 and c.usage.completion_tokens == 8,
	    "a stream with n = 2 has a role and a finish for each choice, and one usage")
	-- The seed of each choice, from an engine that records them.
	local seeds = {}
	local rec = httpd.new()
	openai.install(rec, {
		models = function() return { { id = "rec" } } end,
		props = function() return {} end,
		chat = function(_, req, sink)
			seeds[#seeds + 1] = req.seed
			sink:content("x")
			sink:finish("stop", { prompt_tokens = 1, completion_tokens = 1 })
			return true
		end,
	})
	local route = srv.routes["/v1/chat/completions"].POST
	srv.routes["/v1/chat/completions"].POST = rec.routes["/v1/chat/completions"].POST
	post_json(body({ n = 3, seed = 7 }))
	srv.routes["/v1/chat/completions"].POST = route
	ok(table.concat(seeds, ",") == "7,8,9", "choice i takes seed + i")
	status, b = post_json(body({ n = 0 }))
	ok(status == 400 and b.error.param == "n", "n 0 gives 400")
end

-- Images after tool results: the OpenAI form (a user message after the
-- tool messages) and the lgml extension (image parts in a tool message).
local function test_tool_images()
	local gemma4 = require("lgml.chat.gemma4")
	local media = require("lgml.chat.media")
	local png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
	local function run(msgs)
		local b = { messages = json.array(msgs) }
		local good, err = chat_request.check(b)
		assert(good, err)
		local m, images = media.extract(b.messages, gemma4.image.token)
		return gemma4.render(m, { add_generation_prompt = true }), images
	end
	local calls = json.array({ { id = "call_1", type = "function",
		["function"] = { name = "shot", arguments = "{}" } },
		{ id = "call_2", type = "function", ["function"] = { name = "shot", arguments = "{}" } } })
	local head = {
		{ role = "user", content = "Take two screenshots." },
		{ role = "assistant", content = json.null, tool_calls = calls },
		{ role = "tool", tool_call_id = "call_1", content = "saved 1" },
		{ role = "tool", tool_call_id = "call_2", content = "saved 2" },
	}
	local msgs = table.move(head, 1, #head, 1, {})
	msgs[#msgs + 1] = { role = "user", content = json.array({ { type = "text", text = "The screenshots:" },
		{ type = "image_url", image_url = { url = png } }, { type = "image_url", image_url = { url = png } } }) }
	local text, images = run(msgs)
	local a, b, img = text:find("saved 1", 1, true), text:find("saved 2", 1, true), text:find(gemma4.image.token, 1, true)
	ok(#images == 2 and a and b and img and img > b, "images in a user message after the tool results")
	msgs = table.move(head, 1, 3, 1, {})
	msgs[4] = { role = "tool", tool_call_id = "call_2", content = json.array({ { type = "text", text = "saved 2" },
		{ type = "image_url", image_url = { url = png } } }) }
	text, images = run(msgs)
	img = text:find(gemma4.image.token, 1, true)
	ok(#images == 1 and img and img > text:find("saved 2", 1, true), "an image part in a tool message (lgml)")
end

test_field_table()
test_messages()
test_tool_images()
test_n()
test_tool_choice()
test_reply_table()
test_stop()
test_logprobs()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
