--- Checks for chat completion requests.
-- `FIELDS` lists each top-level field that a chat request can have,
-- with its check and its compatibility status. A field that is not in
-- the list gives an error, as the OpenAI API gives. `check` runs the
-- checks and puts some fields in a normal form for the engines.
-- `docs/api.md` has the table of the fields and their status.
-- @module lgml.server.chat_request
local json = require("lgml.json")
local grammar = require("lgml.grammar")

local M = {}

--- The compatibility status values of `FIELDS`, with their meaning.
-- Read only.
-- @table STATUS
M.STATUS = {
	supported = "the server does what the OpenAI reference says",
	limited = "supported with limits; docs/api.md gives them",
	["no-op"] = "accepted; it has no effect, as for OpenAI it only labels or routes the request",
	rejected = "a value other than the default gives 400",
	planned = "a value other than the default gives 400 until lgml supports it",
}

local ROLES = {
	system = true, developer = true, user = true, assistant = true,
	tool = true,
}
--- The message roles that a chat request can use, as a set. Read only.
M.MESSAGE_ROLES = ROLES

local function is_object(v)
	return type(v) == "table" and not json.isarray(v)
end

local function is_list(v)
	return json.isarray(v)
end

-- Removes the nulls of t, so "max_tokens": null reads as absent.
local function strip_nulls(t)
	for k, v in pairs(t) do
		if v == json.null then
			t[k] = nil
		end
	end
end

-- The first key of t that is not in set, or nil.
local function extra_key(t, set)
	for k in pairs(t) do
		if not set[k] then
			return tostring(k)
		end
	end
	return nil
end

local function set_of(...)
	local s = {}
	for _, k in ipairs({ ... }) do
		s[k] = true
	end
	return s
end

-- content parts

local PART_KEYS = {
	text = set_of("type", "text", "prompt_cache_breakpoint"),
	image_url = set_of("type", "image_url"),
}
local IMAGE_URL_KEYS = set_of("url", "detail")
local DETAIL = set_of("auto", "low", "high", "original")

-- Checks one content part. Returns true, or nil and a message.
local function check_part(part, at)
	if not is_object(part) or type(part.type) ~= "string" then
		return nil, at .. " must be an object with a type"
	end
	strip_nulls(part)
	local keys = PART_KEYS[part.type]
	if not keys then
		return nil, ("%s.type %q is not supported; use text or image_url"):format(at, part.type)
	end
	local k = extra_key(part, keys)
	if k then
		return nil, ("%s.%s is not supported"):format(at, k)
	end
	if part.type == "text" then
		if type(part.text) ~= "string" then
			return nil, at .. ".text must be a string"
		end
		return true
	end
	local u = part.image_url
	if is_object(u) then
		strip_nulls(u)
		k = extra_key(u, IMAGE_URL_KEYS)
		if k then
			return nil, ("%s.image_url.%s is not supported"):format(at, k)
		elseif u.detail ~= nil and not DETAIL[u.detail] then
			return nil, at .. ".image_url.detail must be auto, low, high or original"
		end
		u = u.url
	end
	if type(u) ~= "string" then
		return nil, at .. ".image_url must be a URL string or an object with url"
	end
	return true
end

-- Content is a string, null or an array of parts.
local function check_content(c, at)
	if c == nil or type(c) == "string" then
		return true
	elseif not json.isarray(c) then
		return nil, at .. " must be a string, null or an array of parts"
	end
	for i, part in ipairs(c) do
		local ok, err = check_part(part, ("%s[%d]"):format(at, i - 1))
		if not ok then
			return nil, err
		end
	end
	return true
end

-- Each call names a function; arguments is a string or an object.
local function check_tool_calls(calls, at)
	if not json.isarray(calls) then
		return nil, at .. " must be an array"
	end
	for i, tc in ipairs(calls) do
		local f = is_object(tc) and (tc["function"] or tc)
		if not is_object(f) or type(f.name) ~= "string" or
		    (type(f.arguments) ~= "string" and not is_object(f.arguments)) then
			return nil, ("%s[%d] must name a function and have arguments"):format(at, i - 1)
		elseif tc.type ~= nil and tc.type ~= "function" then
			return nil, ("%s[%d].type must be function"):format(at, i - 1)
		end
	end
	return true
end

-- The fields of a message by role. true: checked by check_message.
local COMMON_KEYS = { "role", "content", "name" }
local MESSAGE_KEYS = {
	system = set_of(table.unpack(COMMON_KEYS)),
	developer = set_of(table.unpack(COMMON_KEYS)),
	user = set_of(table.unpack(COMMON_KEYS)),
	assistant = set_of("role", "content", "name", "tool_calls", "reasoning_content",
	    "reasoning", "refusal", "annotations"),
	tool = set_of("role", "content", "tool_call_id", "name"),
}

local function check_message(m, at)
	if not is_object(m) or not ROLES[m.role] then
		return nil, at .. ".role must be system, developer, user, assistant or tool"
	end
	strip_nulls(m)
	local k = extra_key(m, MESSAGE_KEYS[m.role])
	if k == "audio" or k == "function_call" then
		return nil, ("%s.%s is not supported"):format(at, k)
	elseif k then
		return nil, ("%s.%s is not a field of a %s message"):format(at, k, m.role)
	end
	local ok, err = check_content(m.content, at .. ".content")
	if not ok then
		return nil, err
	elseif m.name ~= nil and type(m.name) ~= "string" then
		return nil, at .. ".name must be a string"
	elseif m.reasoning_content ~= nil and type(m.reasoning_content) ~= "string" then
		return nil, at .. ".reasoning_content must be a string"
	elseif m.reasoning ~= nil and type(m.reasoning) ~= "string" then
		return nil, at .. ".reasoning must be a string"
	elseif m.tool_call_id ~= nil and type(m.tool_call_id) ~= "string" then
		return nil, at .. ".tool_call_id must be a string"
	elseif m.refusal ~= nil and m.refusal ~= "" then
		return nil, at .. ".refusal is not supported"
	elseif m.annotations ~= nil and not json.isarray(m.annotations) then
		return nil, at .. ".annotations must be an array"
	elseif m.tool_calls ~= nil then
		return check_tool_calls(m.tool_calls, at .. ".tool_calls")
	end
	-- vLLM names the field reasoning.
	if m.reasoning ~= nil then
		m.reasoning_content = m.reasoning_content or m.reasoning
		m.reasoning = nil
	end
	return true
end

-- Checks the messages. A null field of a message reads as absent.
local function check_messages(msgs)
	if type(msgs) ~= "table" or not json.isarray(msgs) or #msgs == 0 then
		return nil, "messages must be a non-empty array"
	end
	for i, m in ipairs(msgs) do
		local ok, err = check_message(m, ("messages[%d]"):format(i - 1))
		if not ok then
			return nil, err
		end
	end
	return true
end

-- field checks

-- lgml.json decodes whole numbers as integers, so 5 and 5.0 both pass.
local function int_in(lo, hi)
	return function(v)
		local i = math.tointeger(v)
		return i ~= nil and i >= lo and (hi == nil or i <= hi)
	end
end

local function num_in(lo, hi)
	return function(v)
		return v >= lo and (hi == nil or v <= hi)
	end
end

local function one_of(...)
	local s = set_of(...)
	return function(v)
		return s[v] == true
	end
end

local function never()
	return false
end

-- metadata: at most 16 string values; keys of at most 64 bytes, values
-- of at most 512 bytes.
local function metadata(v)
	if not is_object(v) then
		return false
	end
	local n = 0
	for k, x in pairs(v) do
		n = n + 1
		if type(k) ~= "string" or #k > 64 or type(x) ~= "string" or #x > 512 then
			return false
		end
	end
	return n <= 16
end

local function is_text_content(c)
	if type(c) == "string" then
		return true
	elseif not json.isarray(c) then
		return false
	end
	for _, p in ipairs(c) do
		if not is_object(p) or p.type ~= "text" or type(p.text) ~= "string" then
			return false
		end
	end
	return true
end

local function prediction(v)
	return is_object(v) and v.type == "content" and is_text_content(v.content)
end

local function stream_options(v)
	if not is_object(v) then
		return false
	end
	strip_nulls(v)
	for k, x in pairs(v) do
		if (k ~= "include_usage" and k ~= "include_obfuscation") or type(x) ~= "boolean" then
			return false
		end
	end
	return true
end

-- A string, or an array of at most 4 strings.
local function stop(v)
	if type(v) == "string" then
		return true
	elseif not json.isarray(v) or #v > 4 then
		return false
	end
	for _, x in ipairs(v) do
		if type(x) ~= "string" then
			return false
		end
	end
	return true
end

-- Token ids as decimal strings, biases from -100 to 100.
local function logit_bias(v)
	if not is_object(v) then
		return false
	end
	for k, x in pairs(v) do
		if type(k) ~= "string" or not k:match("^%d+$") or type(x) ~= "number" or x < -100 or x > 100 then
			return false
		end
	end
	return true
end

local function prompt_cache_options(v)
	if not is_object(v) then
		return false
	end
	strip_nulls(v)
	return extra_key(v, set_of("mode", "ttl")) == nil and
	    (v.mode == nil or v.mode == "implicit" or v.mode == "explicit") and
	    (v.ttl == nil or v.ttl == "30m")
end

local function modalities(v)
	if not json.isarray(v) then
		return false
	end
	for _, x in ipairs(v) do
		if x ~= "text" then
			return false
		end
	end
	return true
end

-- A function tool: { type = "function", function = { name, description,
-- parameters, strict } }.
local function tools(v)
	if not json.isarray(v) then
		return false
	end
	for i, t in ipairs(v) do
		if not is_object(t) or t.type ~= "function" then
			return false, ("tools[%d]: only function tools are supported"):format(i - 1)
		end
		strip_nulls(t)
		local f = t["function"]
		if not is_object(f) or type(f.name) ~= "string" or f.name == "" then
			return false, ("tools[%d].function must have a name"):format(i - 1)
		end
		strip_nulls(f)
		local k = extra_key(f, set_of("name", "description", "parameters", "strict"))
		if k or extra_key(t, set_of("type", "function")) then
			return false, ("tools[%d]: %s is not a field of a function tool"):format(i - 1,
			    k or extra_key(t, set_of("type", "function")))
		elseif (f.description ~= nil and type(f.description) ~= "string") or
		    (f.parameters ~= nil and not is_object(f.parameters)) or
		    (f.strict ~= nil and type(f.strict) ~= "boolean") then
			return false, ("tools[%d].function has a field of the wrong type"):format(i - 1)
		end
	end
	return true
end

-- The name of a { type = "function", function = { name } } choice.
local function choice_name(v)
	local f = is_object(v) and v.type == "function" and v["function"]
	return is_object(f) and type(f.name) == "string" and f.name or nil
end

-- "none", "auto", "required", a named function or allowed_tools.
local function tool_choice(v)
	if v == "none" or v == "auto" or v == "required" then
		return true
	elseif not is_object(v) then
		return false
	elseif v.type == "custom" then
		return false, "tool_choice: custom tools are not supported"
	elseif v.type == "function" then
		return choice_name(v) ~= nil
	elseif v.type ~= "allowed_tools" then
		return false
	end
	local a = v.allowed_tools
	if not is_object(a) or (a.mode ~= "auto" and a.mode ~= "required") or not json.isarray(a.tools) then
		return false
	end
	for _, t in ipairs(a.tools) do
		if not choice_name(t) then
			return false, "tool_choice: allowed_tools takes function tools only"
		end
	end
	return true
end

-- The tool of tools with name, or nil.
local function tool_named(tools, name)
	for _, t in ipairs(tools or {}) do
		if t["function"].name == name then
			return t
		end
	end
	return nil
end

-- Checks tool_choice against the tools and puts it in a normal form:
-- "none" drops the tools, and allowed_tools keeps the tools that it
-- names, with its mode as tool_choice.
local function normal_tool_choice(b)
	local tc, tools = b.tool_choice, b.tools
	if type(tc) == "table" and tc.type == "allowed_tools" then
		local keep = json.array()
		for _, t in ipairs(tc.allowed_tools.tools) do
			local name = choice_name(t)
			local tool = tool_named(tools, name)
			if not tool then
				return nil, ("tool_choice: %s is not in tools"):format(name)
			end
			keep[#keep + 1] = tool
		end
		b.tools, b.tool_choice, tools, tc = keep, tc.allowed_tools.mode, keep, tc.allowed_tools.mode
	end
	if tc == "none" then
		b.tools = nil
	elseif tc == "required" and not (tools and #tools > 0) then
		return nil, 'tool_choice "required" needs tools'
	elseif type(tc) == "table" and not tool_named(tools, choice_name(tc)) then
		return nil, ("tool_choice: %s is not in tools"):format(choice_name(tc))
	end
	return true
end

--- The top-level fields of a chat request, in the order of the checks.
-- Each entry is `{ name, types, check, text, status = s, label = l }`.
-- types is a Lua type name, or names joined by `|`. check is optional:
-- a function that takes the value and returns true when it is good,
-- or false and a message. text tells what check accepts. status is a
-- key of `STATUS`. label is `"OpenAI"`, `"llama.cpp"` or `"lgml"`.
-- Read only, except in tests.
-- @table FIELDS
M.FIELDS = {
	{ "messages", "table", is_list, "a non-empty array", status = "supported", label = "OpenAI" },
	{ "model", "string", status = "supported", label = "OpenAI" },
	{ "stream", "boolean", status = "supported", label = "OpenAI" },
	{ "stream_options", "table", stream_options,
	    "an object with boolean include_usage and include_obfuscation",
	    status = "supported", label = "OpenAI" },
	{ "max_tokens", "number", int_in(1), "a positive integer", status = "supported", label = "OpenAI" },
	{ "max_completion_tokens", "number", int_in(1), "a positive integer", status = "supported",
	    label = "OpenAI" },
	{ "temperature", "number", num_in(0), "a number from 0", status = "supported", label = "OpenAI" },
	{ "top_p", "number", num_in(0, 1), "a number from 0 to 1", status = "supported", label = "OpenAI" },
	{ "presence_penalty", "number", num_in(-2, 2), "a number from -2 to 2", status = "supported",
	    label = "OpenAI" },
	{ "frequency_penalty", "number", num_in(-2, 2), "a number from -2 to 2", status = "supported",
	    label = "OpenAI" },
	{ "seed", "number", int_in(math.mininteger), "an integer", status = "supported", label = "OpenAI" },
	{ "stop", "string|table", stop, "a string or an array of at most 4 strings", status = "supported",
	    label = "OpenAI" },
	{ "n", "number", int_in(1, 128), "an integer from 1 to 128", status = "supported", label = "OpenAI" },
	{ "logprobs", "boolean", status = "supported", label = "OpenAI" },
	{ "top_logprobs", "number", int_in(0, 20), "an integer from 0 to 20", status = "supported",
	    label = "OpenAI" },
	{ "logit_bias", "table", logit_bias, "an object of token ids to numbers from -100 to 100",
	    status = "supported", label = "OpenAI" },
	{ "tools", "table", tools, "an array of function tools", status = "limited", label = "OpenAI" },
	{ "tool_choice", "string|table", tool_choice,
	    "none, auto, required, a function or allowed_tools", status = "supported", label = "OpenAI" },
	{ "parallel_tool_calls", "boolean", status = "supported", label = "OpenAI" },
	{ "response_format", "table", is_object, "an object", status = "supported", label = "OpenAI" },
	{ "reasoning_effort", "string", one_of("none", "minimal", "low", "medium", "high", "xhigh", "max"),
	    "none, minimal, low, medium, high, xhigh or max", status = "limited", label = "OpenAI" },
	{ "modalities", "table", modalities, '["text"]', status = "limited", label = "OpenAI" },
	{ "user", "string", status = "no-op", label = "OpenAI" },
	{ "safety_identifier", "string", status = "no-op", label = "OpenAI" },
	{ "metadata", "table", metadata, "an object of at most 16 strings", status = "no-op",
	    label = "OpenAI" },
	{ "store", "boolean", status = "no-op", label = "OpenAI" },
	{ "service_tier", "string", one_of("auto", "default", "flex", "scale", "priority", "fast"),
	    "auto, default, flex, scale, priority or fast", status = "no-op", label = "OpenAI" },
	{ "prompt_cache_key", "string", status = "no-op", label = "OpenAI" },
	{ "prompt_cache_retention", "string", one_of("in_memory", "24h"), "in_memory or 24h",
	    status = "no-op", label = "OpenAI" },
	{ "prompt_cache_options", "table", prompt_cache_options,
	    "an object with mode (implicit or explicit) and ttl (30m)", status = "no-op", label = "OpenAI" },
	{ "prediction", "table", prediction, "an object with type content and text content",
	    status = "no-op", label = "OpenAI" },
	{ "verbosity", "string", function(v) return v == "medium" end, "medium", status = "rejected",
	    label = "OpenAI" },
	{ "audio", "table", never, "null (no audio output)", status = "rejected", label = "OpenAI" },
	{ "moderation", "table", never, "null (no moderation)", status = "rejected", label = "OpenAI" },
	{ "web_search_options", "table", never, "null (no web search)", status = "rejected",
	    label = "OpenAI" },
	{ "functions", "table", never, "null (use tools)", status = "rejected", label = "OpenAI" },
	{ "function_call", "string|table", never, "null (use tool_choice)", status = "rejected",
	    label = "OpenAI" },
	{ "top_k", "number", int_in(0), "an integer from 0", status = "supported", label = "llama.cpp" },
	{ "min_p", "number", num_in(0, 1), "a number from 0 to 1", status = "supported", label = "llama.cpp" },
	{ "repeat_penalty", "number", num_in(0), "a number above 0", status = "supported",
	    label = "llama.cpp" },
	{ "grammar", "string", status = "supported", label = "llama.cpp" },
	{ "json_schema", "table", is_object, "an object", status = "supported", label = "llama.cpp" },
	{ "chat_template_kwargs", "table", is_object, "an object", status = "supported", label = "llama.cpp" },
	{ "lora", "table", is_list, "an array of { id, scale }", status = "supported", label = "llama.cpp" },
}

-- Checks the type and the value of field c in b.
local function check_field(b, c)
	local key, v = c[1], b[c[1]]
	if v == nil then
		return true
	end
	local good = false
	for t in c[2]:gmatch("[^|]+") do
		good = good or type(v) == t
	end
	local msg
	if good and c[3] then
		good, msg = c[3](v)
	end
	if not good then
		return nil, msg or (c[4] and key .. " must be " .. c[4] or key .. " is not valid")
	end
	return true
end

local by_name
local function field(name)
	if not by_name or not by_name[name] then
		by_name = {}
		for _, c in ipairs(M.FIELDS) do
			by_name[c[1]] = c
		end
	end
	return by_name[name]
end

-- Checks b. On error it returns nil, a message, the parameter and
-- maybe an error code.
local function check_body(b, raw)
	if type(b) ~= "table" or json.isarray(b) then
		return nil, "request body must be a JSON object"
	end
	strip_nulls(b)
	for k in pairs(b) do
		if type(k) ~= "string" or not field(k) then
			return nil, ("unknown field %s: see the Compatibility section of docs/api.md"):format(
			    tostring(k)), tostring(k), "unknown_parameter"
		end
	end
	local ok, err = check_messages(b.messages)
	if not ok then
		return nil, err, "messages"
	end
	for _, c in ipairs(M.FIELDS) do
		ok, err = check_field(b, c)
		if not ok then
			local code = (c.status == "rejected" or c.status == "planned") and "unsupported_value" or nil
			return nil, err, c[1], code
		end
	end
	if b.top_logprobs ~= nil and b.logprobs ~= true then
		return nil, "top_logprobs needs logprobs: true", "top_logprobs"
	elseif b.stream_options ~= nil and b.stream ~= true then
		return nil, "stream_options needs stream: true", "stream_options"
	end
	ok, err = normal_tool_choice(b)
	if not ok then
		return nil, err, "tool_choice"
	end
	if type(b.stop) == "string" then
		b.stop = json.array({ b.stop })
	end
	if raw and type(b.response_format) == "table" then
		local o = require("lgml.json_schema").decode(raw)
		if type(o) == "table" and type(o.response_format) == "table" then
			b.response_format = o.response_format
		end
	end
	local text, param
	text, err, param = grammar.from_request(b)
	if err then
		return nil, err, param
	elseif text then
		ok, err = grammar.check(text)
		if not ok then
			return nil, err, b.grammar and "grammar" or "response_format"
		end
		b.grammar, b.response_format, b.json_schema = text, nil, nil
	end
	return true
end

--- Checks a chat request body.
-- It removes top-level nulls and nulls in each message. It gives an
-- error for a field that is not in `FIELDS`, for a value that fails
-- the check of its field, and for a bad message. A string `stop`
-- becomes an array of one string. `tool_choice` `"none"` removes
-- `tools`; `allowed_tools` keeps the tools that it names and becomes
-- its mode. It turns `grammar`, `response_format` or `json_schema` into
-- GBNF text in `grammar` and checks it (see `lgml.grammar`); with raw,
-- schema properties keep their order. `stream_options` needs `stream`.
-- An assistant `reasoning` field becomes `reasoning_content`.
-- @tparam any b the decoded body
-- @tparam[opt] string raw the body text
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] ?string the bad parameter name
-- @treturn[2] ?string the OpenAI error code: `"unknown_parameter"`,
-- `"unsupported_value"` or `"invalid_value"`
function M.check(b, raw)
	local ok, err, param, code = check_body(b, raw)
	if ok or not param then
		return ok, err, param
	end
	if not code then
		code = err:find("not supported", 1, true) and "unsupported_value" or "invalid_value"
	end
	return nil, err, param, code
end

return M
