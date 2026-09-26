--- The OpenAI Responses API on top of the chat engine.
-- `POST /v1/responses` translates a Responses request into the chat
-- request of `Engine:chat` (see `lgml.server.engine`) and the engine
-- output into a Response object. `docs/api.md` has the reference and
-- the compatibility table; `FIELDS` holds the request fields with
-- their status.
-- @module lgml.server.responses
local json = require("lgml.json")
local chat_request = require("lgml.server.chat_request")
local translate = require("lgml.server.translate")
local media = require("lgml.chat.media")
local log = require("lgml.log")

local L = log.new("responses")

local M = {}

local function is_object(v)
	return type(v) == "table" and not json.isarray(v)
end

local function set_of(...)
	local s = {}
	for _, k in ipairs({ ... }) do
		s[k] = true
	end
	return s
end

-- An error: a message, the parameter and the OpenAI error code.
local function fail(msg, param, code)
	return nil, { msg = msg, param = param, code = code }
end

-- field checks

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

local function never(name, what)
	return function()
		return false, ("%s: %s is not supported by this server"):format(name, what)
	end
end

local function metadata(v)
	if not is_object(v) then
		return false
	end
	local count = 0
	for k, x in pairs(v) do
		count = count + 1
		if type(k) ~= "string" or #k > 64 or type(x) ~= "string" or #x > 512 then
			return false
		end
	end
	return count <= 16
end

local function string_map(v)
	if not is_object(v) then
		return false
	end
	for _, x in pairs(v) do
		if type(x) ~= "string" then
			return false
		end
	end
	return true
end

local INCLUDE = set_of("reasoning.encrypted_content")

local function include(v)
	if not json.isarray(v) then
		return false
	end
	for _, x in ipairs(v) do
		if not INCLUDE[x] then
			return false, ("include: %s is not supported by this server; only reasoning.encrypted_content")
			    :format(tostring(x))
		end
	end
	return true
end

local function stream_options(v)
	if not is_object(v) then
		return false
	end
	for k, x in pairs(v) do
		if k ~= "include_obfuscation" or type(x) ~= "boolean" then
			return false
		end
	end
	return true
end

local function cache_options(v)
	if not is_object(v) then
		return false
	elseif v.prewarm == true then
		return false, "prompt_cache_options.prewarm: not supported by this server"
	end
	return true
end

--- The top-level fields of a Responses request, in check order.
-- Each entry is `{ name, types, check, text, status = s, label = l }`,
-- as `lgml.server.chat_request.FIELDS`. A field that is not in the list
-- follows `lgml.server.translate.unknown_fields`. Read only.
-- @table FIELDS
M.FIELDS = {
	{ "model", "string", status = "supported", label = "OpenAI" },
	{ "input", "string|table", nil, "a string or an array of items", status = "supported", label = "OpenAI" },
	{ "instructions", "string", status = "supported", label = "OpenAI" },
	{ "stream", "boolean", status = "supported", label = "OpenAI" },
	{ "stream_options", "table", stream_options, "an object with boolean include_obfuscation",
	    status = "supported", label = "OpenAI" },
	{ "max_output_tokens", "number", int_in(1), "a positive integer", status = "supported", label = "OpenAI" },
	{ "temperature", "number", num_in(0, 2), "a number from 0 to 2", status = "supported", label = "OpenAI" },
	{ "top_p", "number", num_in(0, 1), "a number from 0 to 1", status = "supported", label = "OpenAI" },
	{ "tools", "table", json.isarray, "an array of function tools", status = "limited", label = "OpenAI" },
	{ "tool_choice", "string|table", nil, "none, auto, required, a function or allowed_tools",
	    status = "supported", label = "OpenAI" },
	{ "parallel_tool_calls", "boolean", status = "supported", label = "OpenAI" },
	{ "reasoning", "table", is_object, "an object", status = "limited", label = "OpenAI" },
	{ "text", "table", is_object, "an object", status = "limited", label = "OpenAI" },
	{ "store", "boolean", status = "supported", label = "OpenAI" },
	{ "previous_response_id", "string", status = "supported", label = "OpenAI" },
	{ "include", "table", include, "an array of include names", status = "limited", label = "OpenAI" },
	{ "metadata", "table", metadata, "an object of at most 16 strings", status = "no-op", label = "OpenAI" },
	{ "max_tool_calls", "number", int_in(1), "a positive integer", status = "no-op", label = "OpenAI" },
	{ "prompt_cache_key", "string", status = "no-op", label = "OpenAI" },
	{ "prompt_cache_options", "table", cache_options, "an object", status = "no-op", label = "OpenAI" },
	{ "prompt_cache_retention", "string", one_of("in_memory", "24h"), "in_memory or 24h", status = "no-op",
	    label = "OpenAI" },
	{ "safety_identifier", "string", status = "no-op", label = "OpenAI" },
	{ "user", "string", status = "no-op", label = "OpenAI" },
	{ "service_tier", "string", one_of("auto", "default", "flex", "scale", "priority", "fast", "ultrafast"),
	    "auto, default, flex, scale, priority, fast or ultrafast", status = "no-op", label = "OpenAI" },
	{ "client_metadata", "table", string_map, "an object of strings", status = "no-op", label = "OpenAI" },
	{ "top_logprobs", "number", int_in(0, 0), "0 (no log probabilities)", status = "rejected", label = "OpenAI" },
	{ "truncation", "string", one_of("disabled"), "disabled", status = "rejected", label = "OpenAI" },
	{ "background", "boolean", function(v) return v == false end, "false", status = "rejected",
	    label = "OpenAI" },
	{ "conversation", "string|table", never("conversation", "a conversation"), "absent", status = "rejected",
	    label = "OpenAI" },
	{ "prompt", "table", never("prompt", "a stored prompt"), "absent", status = "rejected", label = "OpenAI" },
	{ "moderation", "table", never("moderation", "moderation"), "absent", status = "rejected", label = "OpenAI" },
	{ "context_management", "table", never("context_management", "context management"), "absent",
	    status = "rejected", label = "OpenAI" },
}

local function unknown(path)
	return ("Unknown parameter: '%s'."):format(path)
end

-- Checks the keys of an object. Unknown keys follow the unknown field
-- rule.
local function keys(ctx, obj, set, at)
	for k, v in pairs(obj) do
		if v == json.null then
			obj[k] = nil
		elseif not set[k] then
			local path = at .. "." .. tostring(k)
			local msg = ctx:unknown(path, unknown(path))
			if msg then
				return fail(msg, path, "unknown_parameter")
			end
			obj[k] = nil
		end
	end
	return true
end

-- input items

local PART_KEYS = {
	input_text = set_of("type", "text", "prompt_cache_breakpoint"),
	input_image = set_of("type", "image_url", "file_id", "detail", "prompt_cache_breakpoint"),
	input_file = set_of("type", "file_id", "file_url", "file_data", "filename", "detail",
		"prompt_cache_breakpoint"),
	output_text = set_of("type", "text", "annotations", "logprobs"),
	refusal = set_of("type", "refusal"),
}
local DETAIL = set_of("auto", "low", "high", "original")

-- A text file part as text, from a data URL or base64 data.
local function file_part(p, at)
	local data = p.file_data
	if p.file_id ~= nil or p.file_url ~= nil or type(data) ~= "string" then
		return fail(at .. ": input_file needs file_data; file ids and URLs are not supported by this server", at)
	end
	local mime, b64 = data:match("^data:([^;,]*);base64,(.*)$")
	b64 = b64 or data
	local bytes = require("lgml.base64").decode(b64)
	if not bytes or not ((mime or ""):find("^text/") or (not mime and utf8.len(bytes))) or
	    not utf8.len(bytes) then
		return fail(at .. ": only text files (text/*) are supported by this server", at)
	end
	local name = type(p.filename) == "string" and p.filename .. ":\n" or ""
	return { type = "text", text = name .. bytes }
end

-- Adds the chat part of one content part to parts. roles: the part
-- types that the item takes.
local function content_part(ctx, p, at, parts, allowed)
	if not is_object(p) or type(p.type) ~= "string" then
		return fail(at .. ": must be an object with a type", at)
	elseif not allowed[p.type] then
		return fail(("%s.type: %s is not supported here"):format(at, p.type), at .. ".type", "unsupported_value")
	end
	local ok, err = keys(ctx, p, PART_KEYS[p.type], at)
	if not ok then
		return nil, err
	end
	if p.type == "input_text" or p.type == "output_text" then
		if type(p.text) ~= "string" then
			return fail(at .. ".text: must be a string", at .. ".text")
		end
		parts[#parts + 1] = { type = "text", text = p.text }
	elseif p.type == "refusal" then
		parts[#parts + 1] = { type = "text", text = tostring(p.refusal) }
	elseif p.type == "input_image" then
		if p.detail ~= nil and not DETAIL[p.detail] then
			return fail(at .. ".detail: must be auto, low, high or original", at .. ".detail")
		elseif p.file_id ~= nil then
			return fail(at .. ".file_id: file ids are not supported by this server; send a data URL",
			    at .. ".file_id", "unsupported_value")
		end
		local part, perr = translate.url_image(p.image_url)
		if not part then
			return fail(at .. ".image_url: " .. perr, at .. ".image_url", "invalid_image_url")
		end
		parts[#parts + 1] = part
	else
		local part, perr = file_part(p, at)
		if not part then
			return nil, perr
		end
		parts[#parts + 1] = part
	end
	return true
end

local USER_PARTS = set_of("input_text", "input_image", "input_file")
local ASSISTANT_PARTS = set_of("output_text", "refusal", "input_text")
local OUTPUT_PARTS = set_of("input_text", "input_image", "input_file")

-- Chat content from an item content: a string, or parts.
local function item_content(ctx, c, at, allowed)
	if type(c) == "string" then
		return c
	elseif not json.isarray(c) then
		return fail(at .. ": must be a string or an array of parts", at)
	end
	local parts = json.array()
	for j, p in ipairs(c) do
		local ok, err = content_part(ctx, p, ("%s[%d]"):format(at, j - 1), parts, allowed)
		if not ok then
			return nil, err
		end
	end
	if #parts == 1 and parts[1].type == "text" then
		return parts[1].text
	end
	return parts
end

-- The text of chat content: a string, or its text parts joined.
local function joined(c)
	if type(c) == "string" then
		return c
	end
	local t = {}
	for _, x in ipairs(c) do
		t[#t + 1] = x.text or ""
	end
	return table.concat(t)
end

local ITEM_KEYS = {
	message = set_of("type", "role", "content", "id", "status", "phase"),
	function_call = set_of("type", "id", "call_id", "name", "arguments", "status", "namespace", "caller"),
	function_call_output = set_of("type", "id", "call_id", "output", "status", "name", "namespace", "caller"),
	reasoning = set_of("type", "id", "summary", "content", "encrypted_content", "status"),
	item_reference = set_of("type", "id"),
}
local ROLES = { user = "user", system = "system", developer = "system", assistant = "assistant" }

-- The reasoning text of a reasoning item.
local function reasoning_text(ctx, it, at)
	local enc = it.encrypted_content
	if enc ~= nil then
		if type(enc) ~= "string" or not translate.ours(enc) then
			return fail(at .. ".encrypted_content: not made by this server", at .. ".encrypted_content",
			    "invalid_encrypted_content")
		end
		local text = translate.verify(ctx.key, ctx.model, enc)
		if not text then
			return fail(at .. ".encrypted_content: the signature does not match", at .. ".encrypted_content",
			    "invalid_encrypted_content")
		end
		return text
	end
	for _, field in ipairs({ "content", "summary" }) do
		local list = it[field]
		if list ~= nil and not json.isarray(list) then
			return fail(("%s.%s: must be an array"):format(at, field), at .. "." .. field)
		end
		local t = {}
		for _, p in ipairs(list or {}) do
			if is_object(p) and type(p.text) == "string" then
				t[#t + 1] = p.text
			end
		end
		if #t > 0 then
			return table.concat(t, "\n\n")
		end
	end
	return ""
end

--- Translates input items into chat messages.
-- Consecutive assistant messages, function calls and reasoning items
-- make one assistant message. Each function_call_output needs a
-- function_call with its call_id before it. ctx gets `calls`, the
-- call ids, and `items`, the items in order (for the store).
-- @tparam lgml.server.translate.Context ctx the request context, with
-- `key` and `model`
-- @tparam table items the items
-- @tparam table out the chat messages to add to
-- @tparam string at the path of the items
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] table `{ msg, param, code }`
function M.convert_items(ctx, items, out, at)
	local cur
	local function flush()
		if cur then
			local m = { role = "assistant", content = table.concat(cur.texts) }
			if #cur.calls > 0 then
				m.tool_calls = cur.calls
				if m.content == "" then
					m.content = json.null
				end
			end
			if #cur.reasoning > 0 then
				m.reasoning_content = table.concat(cur.reasoning, "\n\n")
			end
			out[#out + 1] = m
			cur = nil
		end
	end
	local function assistant()
		cur = cur or { texts = {}, calls = json.array(), reasoning = {} }
		return cur
	end
	for i, it in ipairs(items) do
		local p = ("%s[%d]"):format(at, i - 1)
		if not is_object(it) then
			return fail(p .. ": must be an object", p)
		end
		local kind = it.type or (it.role and "message")
		if kind == "item_reference" and ctx.resolve then
			local ref, rerr = ctx.resolve(it.id, p)
			if not ref then
				return nil, rerr
			end
			it, kind = ref, ref.type or "message"
		end
		if not ITEM_KEYS[kind] or kind == "item_reference" then
			return fail(("%s.type: item type %s is not supported by this server"):format(p, tostring(kind)),
			    p .. ".type", "unsupported_value")
		end
		local ok, err = keys(ctx, it, ITEM_KEYS[kind], p)
		if not ok then
			return nil, err
		end
		ctx.items[#ctx.items + 1] = it
		if kind == "message" then
			local role = ROLES[it.role]
			if not role then
				return fail(p .. ".role: must be user, assistant, system or developer", p .. ".role")
			end
			local c
			c, err = item_content(ctx, it.content, p .. ".content",
			    role == "assistant" and ASSISTANT_PARTS or USER_PARTS)
			if c == nil then
				return nil, err
			end
			if role == "assistant" then
				local a = assistant()
				if #a.calls > 0 then
					flush()
					a = assistant()
				end
				a.texts[#a.texts + 1] = joined(c)
			else
				flush()
				out[#out + 1] = { role = role, content = c }
			end
		elseif kind == "function_call" then
			if type(it.call_id) ~= "string" or it.call_id == "" then
				return fail(p .. ".call_id: must be a non-empty string", p .. ".call_id")
			elseif type(it.name) ~= "string" or it.name == "" then
				return fail(p .. ".name: must be a non-empty string", p .. ".name")
			elseif type(it.arguments) ~= "string" then
				return fail(p .. ".arguments: must be a JSON string", p .. ".arguments")
			end
			ctx.calls[it.call_id] = true
			local a = assistant()
			a.calls[#a.calls + 1] = { id = it.call_id, type = "function",
				["function"] = { name = it.name, arguments = it.arguments } }
		elseif kind == "function_call_output" then
			if type(it.call_id) ~= "string" or not ctx.calls[it.call_id] then
				return fail(("%s.call_id: no function call found with call_id %s"):format(p, tostring(it.call_id)),
				    p .. ".call_id")
			end
			local c
			c, err = item_content(ctx, it.output, p .. ".output", OUTPUT_PARTS)
			if c == nil then
				return nil, err
			end
			flush()
			out[#out + 1] = { role = "tool", tool_call_id = it.call_id, content = c }
		elseif kind == "reasoning" then
			local text
			text, err = reasoning_text(ctx, it, p)
			if not text then
				return nil, err
			end
			if cur and (#cur.texts > 0 or #cur.calls > 0) then
				flush()
			end
			if text ~= "" then
				local a = assistant()
				a.reasoning[#a.reasoning + 1] = text
			end
		end
	end
	flush()
	return true
end

-- tools, choices and formats

local TOOL_KEYS = set_of("type", "name", "description", "parameters", "strict", "defer_loading",
	"allowed_callers", "output_schema", "async")
local TOOL_IGNORED = set_of("defer_loading", "allowed_callers", "output_schema", "async")

local function convert_tools(ctx, tools)
	local out = json.array()
	for i, t in ipairs(tools) do
		local at = ("tools[%d]"):format(i - 1)
		if not is_object(t) then
			return fail(at .. ": must be an object", "tools")
		elseif t.type == "function" and is_object(t["function"]) then
			return fail(at .. ': use the flat Responses shape {"type": "function", "name": ..., ' ..
			    '"parameters": ...}, not the nested Chat Completions shape', "tools", "invalid_value")
		elseif t.type ~= "function" then
			return fail(("%s.type: %s tools are not supported by this server; only function tools")
			    :format(at, tostring(t.type)), "tools", "unsupported_value")
		end
		local ok, err = keys(ctx, t, TOOL_KEYS, at)
		if not ok then
			return nil, err
		end
		if type(t.name) ~= "string" or t.name == "" then
			return fail(at .. ".name: must be a non-empty string", "tools")
		elseif t.parameters ~= nil and not is_object(t.parameters) then
			return fail(at .. ".parameters: must be an object", "tools")
		elseif t.description ~= nil and type(t.description) ~= "string" then
			return fail(at .. ".description: must be a string", "tools")
		elseif t.strict ~= nil and type(t.strict) ~= "boolean" then
			return fail(at .. ".strict: must be a boolean", "tools")
		end
		for k in pairs(t) do
			if TOOL_IGNORED[k] then
				ctx:ignore("tools." .. k)
			end
		end
		out[i] = translate.function_tool(t.name, t.description, t.parameters, t.strict)
	end
	return out
end

local function function_choice(t)
	if is_object(t) and t.type == "function" and type(t.name) == "string" then
		return { type = "function", ["function"] = { name = t.name } }
	end
	return nil
end

local function convert_tool_choice(tc)
	if tc == "none" or tc == "auto" or tc == "required" then
		return tc
	elseif function_choice(tc) then
		return function_choice(tc)
	elseif is_object(tc) and tc.type == "allowed_tools" and (tc.mode == "auto" or tc.mode == "required") and
	    json.isarray(tc.tools) then
		local list = json.array()
		for i, t in ipairs(tc.tools) do
			list[i] = function_choice(t)
			if not list[i] then
				return fail("tool_choice: allowed_tools takes function tools only", "tool_choice",
				    "unsupported_value")
			end
		end
		return { type = "allowed_tools", allowed_tools = { mode = tc.mode, tools = list } }
	end
	return fail('tool_choice: must be "none", "auto", "required", {"type": "function", "name": ...} or ' ..
	    "allowed_tools", "tool_choice", "unsupported_value")
end

local EFFORT = set_of("none", "minimal", "low", "medium", "high", "xhigh", "max")

local function convert_reasoning(ctx, r, chat)
	local ok, err = keys(ctx, r, set_of("effort", "summary", "generate_summary", "context", "mode"), "reasoning")
	if not ok then
		return nil, err
	end
	if r.effort ~= nil and not EFFORT[r.effort] then
		return fail("reasoning.effort: must be none, minimal, low, medium, high, xhigh or max", "reasoning.effort")
	end
	for _, k in ipairs({ "summary", "generate_summary" }) do
		if r[k] ~= nil and not set_of("auto", "concise", "detailed")[r[k]] then
			return fail(("reasoning.%s: must be auto, concise or detailed"):format(k), "reasoning." .. k)
		end
	end
	if r.context ~= nil and r.context ~= "auto" then
		return fail("reasoning.context: only auto is supported by this server", "reasoning.context",
		    "unsupported_value")
	elseif r.mode ~= nil and r.mode ~= "standard" then
		return fail("reasoning.mode: only standard is supported by this server", "reasoning.mode",
		    "unsupported_value")
	end
	chat.reasoning_effort = r.effort
	return true
end

local function convert_text(ctx, t, chat)
	local ok, err = keys(ctx, t, set_of("format", "verbosity"), "text")
	if not ok then
		return nil, err
	end
	if t.verbosity ~= nil and t.verbosity ~= "medium" then
		return fail("text.verbosity: only medium is supported by this server", "text.verbosity",
		    "unsupported_value")
	end
	local f = t.format
	if f == nil or (is_object(f) and f.type == "text") then
		return true
	elseif not is_object(f) then
		return fail("text.format: must be an object", "text.format")
	elseif f.type == "json_object" then
		chat.response_format = { type = "json_object" }
	elseif f.type == "json_schema" then
		if not is_object(f.schema) then
			return fail("text.format.schema: must be an object", "text.format.schema")
		elseif f.json_schema ~= nil then
			return fail('text.format: use the flat Responses shape {"type": "json_schema", "name": ..., ' ..
			    '"schema": ...}', "text.format")
		end
		chat.response_format = { type = "json_schema", json_schema = { name = f.name, schema = f.schema,
			strict = f.strict, description = f.description } }
	else
		return fail("text.format.type: must be text, json_object or json_schema", "text.format.type")
	end
	return true
end

-- Chat check parameters to Responses field names.
local PARAMS = { response_format = "text.format", tools = "tools", tool_choice = "tool_choice",
	messages = "input", reasoning_effort = "reasoning.effort" }

--- Translates a Responses request body into a chat request.
-- The chat request is checked with `lgml.server.chat_request.check`.
-- The context needs `key` and `model` (for reasoning items) before the
-- call; it gets `items`, the input items in order.
-- @tparam table body the decoded body
-- @tparam lgml.server.translate.Context ctx the request context
-- @tparam[opt] table history chat messages of earlier responses, put
-- after the instructions and before the input
-- @treturn[1] table the chat request
-- @treturn[2] nil
-- @treturn[2] table `{ msg, param, code }`
function M.translate(body, ctx, history)
	local out = json.array()
	if type(body.instructions) == "string" and body.instructions ~= "" then
		out[1] = { role = "system", content = body.instructions }
	end
	for _, m in ipairs(history or {}) do
		out[#out + 1] = m
	end
	local input = body.input
	if type(input) == "string" then
		input = json.array({ { role = "user", content = input } })
	elseif input == nil then
		return fail("input: Missing required parameter: 'input'.", "input", "missing_required_parameter")
	elseif not json.isarray(input) then
		return fail("input: must be a string or an array of items", "input")
	end
	local ok, err = M.convert_items(ctx, input, out, "input")
	if not ok then
		return nil, err
	end
	if #out == 0 then
		return fail("input: must not be empty", "input")
	end
	local chat = { model = body.model, messages = out }
	if body.tools ~= nil and #body.tools > 0 then
		chat.tools, err = convert_tools(ctx, body.tools)
		if not chat.tools then
			return nil, err
		end
	end
	if body.tool_choice ~= nil then
		chat.tool_choice, err = convert_tool_choice(body.tool_choice)
		if not chat.tool_choice then
			return nil, err
		end
		if chat.tool_choice ~= "none" and chat.tool_choice ~= "auto" and not chat.tools then
			return fail("tool_choice: needs tools", "tool_choice")
		end
	end
	if body.reasoning ~= nil then
		ok, err = convert_reasoning(ctx, body.reasoning, chat)
		if not ok then
			return nil, err
		end
	end
	if body.text ~= nil then
		ok, err = convert_text(ctx, body.text, chat)
		if not ok then
			return nil, err
		end
	end
	chat.parallel_tool_calls = body.parallel_tool_calls
	chat.temperature, chat.top_p = body.temperature, body.top_p
	chat.max_completion_tokens = body.max_output_tokens
	local param, code
	ok, err, param, code = chat_request.check(chat)
	if not ok then
		local p = PARAMS[param] or param
		return fail(err, p, code)
	end
	return chat
end

-- replies

-- The Response object with the echo of the request settings.
local function response_object(r, status)
	local b = r.body
	local reasoning = is_object(b.reasoning) and b.reasoning or {}
	local text = is_object(b.text) and b.text or {}
	return {
		id = r.id, object = "response", created_at = r.created, completed_at = json.null,
		status = status, error = json.null, incomplete_details = json.null, model = r.model,
		instructions = b.instructions or json.null, output = json.array(), usage = json.null,
		parallel_tool_calls = b.parallel_tool_calls ~= false,
		tool_choice = b.tool_choice or "auto", tools = b.tools or json.array(),
		temperature = b.temperature or 1, top_p = b.top_p or 1, top_logprobs = 0,
		presence_penalty = 0, frequency_penalty = 0,
		max_output_tokens = b.max_output_tokens or json.null, max_tool_calls = b.max_tool_calls or json.null,
		previous_response_id = b.previous_response_id or json.null,
		reasoning = { effort = reasoning.effort or json.null,
			summary = reasoning.summary or reasoning.generate_summary or json.null },
		text = { format = text.format or { type = "text" }, verbosity = text.verbosity or "medium" },
		truncation = "disabled", metadata = b.metadata or json.object(), store = r.store,
		background = false, service_tier = "default", prompt_cache_key = b.prompt_cache_key or json.null,
		safety_identifier = b.safety_identifier or json.null, user = b.user or json.null,
		conversation = json.null,
	}
end

--- The Responses `usage` object.
-- `input_tokens` counts the whole prompt; `cached_tokens` the part
-- reused from the cache.
-- @tparam table c the counts of `lgml.server.translate.counts`
-- @treturn table the usage
function M.usage(c)
	return {
		input_tokens = c.prompt, input_tokens_details = { cached_tokens = c.cached, cache_write_tokens = 0 },
		output_tokens = c.output, output_tokens_details = { reasoning_tokens = c.reasoning },
		total_tokens = c.prompt + c.output,
	}
end

-- The item id of a block, made on first use.
local function item_id(b)
	if not b.item_id then
		local prefix = b.kind == "reasoning" and "rs_" or b.kind == "tool" and "fc_" or "msg_"
		b.item_id = translate.id(prefix)
	end
	return b.item_id
end

local function text_part(text)
	return { type = "output_text", text = text, annotations = json.array(), logprobs = json.array() }
end

-- The output item of a block: its full form, or with done false its
-- form when the item opens.
local function item_of(r, b, done, status)
	local text = done and translate.text(b) or ""
	local st = done and (status or "completed") or "in_progress"
	if b.kind == "reasoning" then
		local it = { type = "reasoning", id = item_id(b), summary = json.array(), status = st }
		if done then
			it.summary[1] = { type = "summary_text", text = text }
			it.content = json.array({ { type = "reasoning_text", text = text } })
			it.encrypted_content = translate.sign(r.key, r.model, text, true)
		end
		return it
	elseif b.kind == "tool" then
		return { type = "function_call", id = item_id(b), call_id = b.id, name = b.name, arguments = text,
			status = st }
	end
	return { type = "message", id = item_id(b), status = st, role = "assistant",
		content = done and json.array({ text_part(text) }) or json.array() }
end

-- The status of a finished reply: completed or incomplete.
local function end_status(r, f, ncalls)
	local e = translate.ending(f, ncalls, r.max)
	if e == "length" or e == "context" then
		return "incomplete"
	end
	return "completed"
end

-- The finished Response object.
local function final_object(r, sink)
	local f = sink.finished
	local status = end_status(r, f, #sink.calls)
	local o = response_object(r, status)
	for i, b in ipairs(sink.list) do
		local last = i == #sink.list
		o.output[i] = item_of(r, b, true, last and status == "incomplete" and b.kind ~= "reasoning" and
		    "incomplete" or nil)
	end
	if status == "incomplete" then
		o.incomplete_details = { reason = "max_output_tokens" }
	else
		o.completed_at = os.time()
	end
	o.usage = M.usage(translate.counts(f.usage))
	return o
end

local function send_error(r, status, e)
	return r.res:json(status, require("lgml.server.http").error_object(status, e.msg, nil, e.param, e.code),
	    r.ctx:headers({}))
end

-- Reads and translates the request. Returns the reply state and the
-- chat request, or sends an error and returns nil.
local function prepare(eng, opts, req, res)
	local ctx = translate.context()
	local r = { res = res, ctx = ctx, key = opts.key, id = translate.id("resp_"), created = os.time() }
	if type(eng.chat) ~= "function" then
		send_error(r, 404, { msg = "this server does not serve chat" })
		return nil
	end
	local body, err = json.decode(req.body)
	if not is_object(body) then
		send_error(r, 400, { msg = body == nil and "bad JSON: " .. tostring(err) or
		    "the request body must be a JSON object" })
		return nil
	end
	local ok, msg, path = translate.check_fields(body, M.FIELDS, "", ctx, unknown)
	if not ok then
		local status
		for _, c in ipairs(M.FIELDS) do
			if c[1] == path then
				status = c.status
			end
		end
		local code = not status and "unknown_parameter" or
		    (status == "rejected" or msg:find("not supported")) and "unsupported_value" or "invalid_value"
		send_error(r, 400, { msg = msg, param = path, code = code })
		return nil
	elseif body.stream_options ~= nil and body.stream ~= true then
		send_error(r, 400, { msg = "stream_options: only with stream: true", param = "stream_options" })
		return nil
	end
	r.body, r.store = body, body.store ~= false
	ctx.key, ctx.calls, ctx.items = opts.key, {}, {}
	if body.previous_response_id ~= nil then
		send_error(r, 400, { msg = ("Previous response with id '%s' not found."):format(body.previous_response_id),
			param = "previous_response_id", code = "previous_response_not_found" })
		return nil
	end
	-- Reasoning items need the model to check their signatures; the
	-- model needs to know of images. Images come from the input only.
	ctx.model = translate.served_model(eng, body.model, "chat", false)
	local chat
	chat, err = M.translate(body, ctx)
	if chat and media.has_images(chat.messages) then
		local model = translate.served_model(eng, body.model, "chat", true)
		if model ~= ctx.model then
			ctx.model, ctx.calls, ctx.items = model, {}, {}
			chat, err = M.translate(body, ctx)
		end
	end
	if not chat then
		L:debug("bad request", "param", err.param, "err", err.msg)
		send_error(r, 400, err)
		return nil
	end
	r.model, r.max = ctx.model, chat.max_completion_tokens
	return r, chat
end

local function create(eng, opts, req, res)
	local r, chat = prepare(eng, opts, req, res)
	if not r then
		return
	end
	if r.body.stream then
		return send_error(r, 400, { msg = "stream: not supported by this server yet", param = "stream" })
	end
	local sink = translate.blocks(res)
	local ok, msg, status = translate.run(eng, chat, sink, r.model)
	if sink:cancelled() then
		res.close = true
		return
	elseif not ok then
		return send_error(r, status or 500, { msg = msg or "engine failed" })
	end
	return res:json(200, final_object(r, sink), r.ctx:headers({}))
end

--- Adds the Responses routes for eng to srv: `POST /v1/responses`.
-- @tparam lgml.server.httpd.Server srv the server
-- @tparam lgml.server.engine.Engine eng the engine
-- @tparam table opts the options: `key`, the server key for reasoning
-- signatures (see `lgml.server.translate.load_key`)
function M.install(srv, eng, opts)
	srv:route("POST", "/v1/responses", function(req, res)
		return create(eng, opts, req, res)
	end)
end

return M
