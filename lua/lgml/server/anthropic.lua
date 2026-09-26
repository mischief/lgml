--- The Anthropic Messages API on top of the chat engine.
-- `POST /v1/messages` translates a Messages request into the chat
-- request of `Engine:chat` (see `lgml.server.engine`) and the engine
-- output into a Messages reply. `docs/api.md` has the reference and the
-- compatibility table; `FIELDS` holds the request fields with their
-- status.
-- @module lgml.server.anthropic
local json = require("lgml.json")
local chat_request = require("lgml.server.chat_request")
local translate = require("lgml.server.translate")
local media = require("lgml.chat.media")
local sse = require("lgml.server.sse")
local log = require("lgml.log")

local L = log.new("anthropic")

local M = {}

--- The API version that the server speaks.
M.VERSION = "2023-06-01"

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

-- errors

local ERROR_TYPES = {
	[400] = "invalid_request_error", [401] = "authentication_error", [402] = "billing_error",
	[403] = "permission_error", [404] = "not_found_error", [409] = "conflict_error",
	[413] = "request_too_large", [429] = "rate_limit_error", [500] = "api_error",
	[503] = "overloaded_error", [504] = "timeout_error", [529] = "overloaded_error",
}

--- The Anthropic error type of an HTTP status.
-- @tparam int status the status
-- @treturn string the type, such as `"invalid_request_error"`
function M.error_type(status)
	return ERROR_TYPES[status] or (status >= 500 and "api_error" or "invalid_request_error")
end

--- The Anthropic error body.
-- @tparam int status the HTTP status
-- @tparam string message the message
-- @tparam ?string rid the request id
-- @treturn table `{ type = "error", error = { type, message }, request_id }`
function M.error_body(status, message, rid)
	return { type = "error", error = { type = M.error_type(status), message = message },
		request_id = rid or json.null }
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

local function string_list(v)
	if not json.isarray(v) then
		return false
	end
	for _, x in ipairs(v) do
		if type(x) ~= "string" then
			return false
		end
	end
	return true
end

local function never(name)
	return function()
		return false, name .. ": not supported by this server"
	end
end

--- The top-level fields of a Messages request, in check order.
-- Each entry is `{ name, types, check, text, status = s, label = l }`,
-- as `lgml.server.chat_request.FIELDS`. A field that is not in the list
-- follows `lgml.server.translate.unknown_fields`. Read only.
-- @table FIELDS
M.FIELDS = {
	{ "model", "string", status = "supported", label = "Anthropic" },
	{ "messages", "table", json.isarray, "an array of messages", status = "supported", label = "Anthropic" },
	{ "max_tokens", "number", int_in(0), "an integer from 0", status = "supported", label = "Anthropic" },
	{ "system", "string|table", nil, "a string or an array of text blocks", status = "supported",
	    label = "Anthropic" },
	{ "stream", "boolean", status = "supported", label = "Anthropic" },
	{ "stop_sequences", "table", string_list, "an array of strings", status = "supported", label = "Anthropic" },
	{ "temperature", "number", num_in(0), "a number from 0", status = "supported", label = "Anthropic" },
	{ "top_p", "number", num_in(0, 1), "a number from 0 to 1", status = "supported", label = "Anthropic" },
	{ "top_k", "number", int_in(0), "an integer from 0", status = "supported", label = "Anthropic" },
	{ "tools", "table", json.isarray, "an array of tools", status = "limited", label = "Anthropic" },
	{ "tool_choice", "table", is_object, "an object", status = "supported", label = "Anthropic" },
	{ "thinking", "table", is_object, "an object", status = "limited", label = "Anthropic" },
	{ "output_config", "table", is_object, "an object", status = "limited", label = "Anthropic" },
	{ "metadata", "table", is_object, "an object", status = "no-op", label = "Anthropic" },
	{ "service_tier", "string", one_of("auto", "standard_only"), "auto or standard_only", status = "no-op",
	    label = "Anthropic" },
	{ "cache_control", "table", is_object, "an object", status = "no-op", label = "Anthropic" },
	{ "context_management", "table", is_object, "an object", status = "no-op", label = "Anthropic" },
	{ "container", "string|table", never("container"), "absent", status = "rejected", label = "Anthropic" },
	{ "inference_geo", "string", never("inference_geo"), "absent", status = "rejected", label = "Anthropic" },
	{ "mcp_servers", "table", never("mcp_servers"), "absent", status = "rejected", label = "Anthropic" },
	{ "speed", "string", never("speed"), "absent", status = "rejected", label = "Anthropic" },
}

--- The fields of a count_tokens request: those of `FIELDS` that change
-- the prompt, and the generation fields, which it accepts and ignores.
-- Read only.
M.COUNT_IGNORED = set_of("max_tokens", "stream", "stop_sequences", "temperature", "top_p", "top_k",
	"metadata", "service_tier")

local function extra(path)
	return path .. ": Extra inputs are not permitted"
end

-- Checks the keys of obj against set. An unknown key follows the
-- unknown field rule. Returns true, or nil, a message and the path.
local function keys(ctx, obj, set, at)
	for k, v in pairs(obj) do
		if v == json.null then
			obj[k] = nil
		elseif not set[k] then
			local path = at .. "." .. tostring(k)
			local msg = ctx:unknown(path, extra(path))
			if msg then
				return nil, msg, path
			end
			obj[k] = nil
		end
	end
	return true
end

local function fail(msg, path)
	return nil, msg, path
end

-- request translation

-- Block fields by block type.
local BLOCK_KEYS = {
	text = set_of("type", "text", "cache_control", "citations"),
	image = set_of("type", "source", "cache_control", "transformations"),
	document = set_of("type", "source", "title", "context", "citations", "cache_control"),
	search_result = set_of("type", "source", "title", "content", "citations", "cache_control"),
	tool_result = set_of("type", "tool_use_id", "content", "is_error", "cache_control", "toolset_name"),
	tool_use = set_of("type", "id", "name", "input", "cache_control", "caller", "toolset_name"),
	thinking = set_of("type", "thinking", "signature"),
	redacted_thinking = set_of("type", "data"),
}

-- The block types that each role takes.
local ROLE_BLOCKS = {
	user = set_of("text", "image", "document", "search_result", "tool_result"),
	assistant = set_of("text", "tool_use", "thinking", "redacted_thinking"),
	system = set_of("text"),
}

-- Block types inside a tool_result and a document content source.
local INNER_BLOCKS = set_of("text", "image", "document", "search_result")

local ID_PATTERN = "^[%w_-]+$"

-- Checks one block: an object with a known type for the role, and known
-- keys. Returns true, or nil, a message and the path.
local function check_block(ctx, b, at, allowed)
	if not is_object(b) or type(b.type) ~= "string" then
		return fail(at .. ": must be an object with a type", at)
	end
	if not BLOCK_KEYS[b.type] then
		return fail(("%s: block type %q is not supported by this server"):format(at, b.type), at .. ".type")
	elseif not allowed[b.type] then
		return fail(("%s: a %s block is not allowed here"):format(at, b.type), at .. ".type")
	end
	return keys(ctx, b, BLOCK_KEYS[b.type], at)
end

local function image_block(ctx, b, at)
	local src = b.source
	if not is_object(src) then
		return fail(at .. ".source: must be an object", at .. ".source")
	end
	local tr = b.transformations
	if tr ~= nil and not (is_object(tr) and (tr.oversized_image == nil or tr.oversized_image == "downsize")) then
		return fail(at .. '.transformations: only oversized_image "downsize" is supported by this server',
		    at .. ".transformations")
	end
	local part, err
	if src.type == "base64" then
		local ok, kerr, kpath = keys(ctx, src, set_of("type", "media_type", "data"), at .. ".source")
		if not ok then
			return nil, kerr, kpath
		end
		part, err = translate.image_part(src.media_type, src.data)
	elseif src.type == "url" then
		part, err = translate.url_image(src.url)
	else
		return fail(('%s.source: type "%s" is not supported by this server; send base64'):format(at,
		    tostring(src.type)), at .. ".source")
	end
	if not part then
		return fail(at .. ".source: " .. err, at .. ".source")
	end
	return part
end

local convert_inner

-- A text part for a document block. Only text and content sources work.
local function document_parts(ctx, b, at, parts)
	if b.citations ~= nil and not (is_object(b.citations) and b.citations.enabled ~= true) then
		return fail(at .. ".citations: not supported by this server", at .. ".citations")
	end
	local src = b.source
	if is_object(src) and src.type == "text" and type(src.data) == "string" then
		local head = type(b.title) == "string" and b.title .. "\n" or ""
		parts[#parts + 1] = { type = "text", text = head .. src.data }
		return true
	elseif is_object(src) and src.type == "content" then
		if type(src.content) == "string" then
			parts[#parts + 1] = { type = "text", text = src.content }
			return true
		end
		return convert_inner(ctx, src.content, at .. ".source.content", parts)
	elseif is_object(src) and (src.type == "base64" or src.type == "url") and
	    (src.media_type == nil or src.media_type == "application/pdf") then
		return fail(at .. ": PDF documents are not supported by this server", at .. ".source")
	end
	return fail(('%s.source: type "%s" is not supported by this server; send text'):format(at,
	    tostring(is_object(src) and src.type)), at .. ".source")
end

local function search_parts(b, parts)
	local t = {}
	if type(b.title) == "string" then
		t[#t + 1] = b.title
	end
	if type(b.source) == "string" then
		t[#t + 1] = b.source
	end
	for _, c in ipairs(json.isarray(b.content) and b.content or {}) do
		if is_object(c) and type(c.text) == "string" then
			t[#t + 1] = c.text
		end
	end
	parts[#parts + 1] = { type = "text", text = table.concat(t, "\n") }
end

-- Adds the chat parts of one text, image, document or search_result
-- block to parts. The block is checked.
local function inner_block(ctx, b, p, parts)
	if b.type == "text" then
		if type(b.text) ~= "string" then
			return fail(p .. ".text: must be a string", p .. ".text")
		end
		parts[#parts + 1] = { type = "text", text = b.text }
	elseif b.type == "image" then
		local part, err, path = image_block(ctx, b, p)
		if not part then
			return nil, err, path
		end
		parts[#parts + 1] = part
	elseif b.type == "document" then
		return document_parts(ctx, b, p, parts)
	else
		search_parts(b, parts)
	end
	return true
end

-- Adds the chat parts of a list of text, image, document and
-- search_result blocks to parts.
convert_inner = function(ctx, blocks, at, parts)
	if not json.isarray(blocks) then
		return fail(at .. ": must be a string or an array of blocks", at)
	end
	for j, b in ipairs(blocks) do
		local p = ("%s.%d"):format(at, j - 1)
		local ok, err, path = check_block(ctx, b, p, INNER_BLOCKS)
		if ok then
			ok, err, path = inner_block(ctx, b, p, parts)
		end
		if not ok then
			return nil, err, path
		end
	end
	return true
end

-- The chat content of a tool_result block.
local function result_content(ctx, b, at)
	local c = b.content
	if b.is_error ~= nil and type(b.is_error) ~= "boolean" then
		return fail(at .. ".is_error: must be a boolean", at .. ".is_error")
	end
	local err_note = b.is_error and "Error: " or ""
	if c == nil or type(c) == "string" then
		return err_note .. (c or "")
	end
	local parts = json.array()
	if b.is_error then
		parts[1] = { type = "text", text = "Error:" }
	end
	local ok, err, path = convert_inner(ctx, c, at .. ".content", parts)
	if not ok then
		return nil, err, path
	end
	return parts
end

-- Merges consecutive messages of one role. Each entry is { role,
-- blocks = { { block, path }, ... } }; string content is one text block.
local function merge(ctx, list)
	local out = {}
	for i, m in ipairs(list) do
		local at = ("messages.%d"):format(i - 1)
		if not is_object(m) then
			return fail(at .. ": must be an object", at)
		end
		local ok, err, path = keys(ctx, m, set_of("role", "content"), at)
		if not ok then
			return nil, err, path
		end
		if not ROLE_BLOCKS[m.role] then
			return fail(at .. ".role: must be user, assistant or system", at .. ".role")
		end
		local blocks = {}
		if type(m.content) == "string" then
			blocks[1] = { { type = "text", text = m.content }, at .. ".content" }
		elseif json.isarray(m.content) then
			for j, b in ipairs(m.content) do
				blocks[j] = { b, ("%s.content.%d"):format(at, j - 1) }
			end
		else
			return fail(at .. ".content: must be a string or an array of blocks", at .. ".content")
		end
		local prev = out[#out]
		if prev and prev.role == m.role and m.role ~= "system" then
			table.move(blocks, 1, #blocks, #prev.blocks + 1, prev.blocks)
		else
			out[#out + 1] = { role = m.role, blocks = blocks, at = at }
		end
	end
	return out
end

-- Translates the blocks of a user turn. Adds tool messages and a user
-- message to out. ids is the set of tool_use ids of the turn before;
-- each needs a result here.
local function user_turn(ctx, turn, ids, out)
	local parts, seen, other = json.array(), {}, false
	for _, e in ipairs(turn.blocks) do
		local b, at = e[1], e[2]
		local ok, err, path = check_block(ctx, b, at, ROLE_BLOCKS.user)
		if not ok then
			return nil, err, path
		end
		if b.type == "tool_result" then
			if other then
				return fail(at .. ": tool_result blocks must come before the other blocks of a user message", at)
			elseif type(b.tool_use_id) ~= "string" or not b.tool_use_id:find(ID_PATTERN) then
				return fail(at .. ".tool_use_id: must match ^[a-zA-Z0-9_-]+$", at .. ".tool_use_id")
			elseif not ids[b.tool_use_id] then
				return fail(("%s: unexpected tool_use_id found in tool_result blocks: %s. Each tool_result " ..
				    "block must have a corresponding tool_use block in the previous message."):format(at,
				    b.tool_use_id), at .. ".tool_use_id")
			end
			seen[b.tool_use_id] = true
			local content
			content, err, path = result_content(ctx, b, at)
			if not content then
				return nil, err, path
			end
			out[#out + 1] = { role = "tool", tool_call_id = b.tool_use_id, content = content }
		else
			other = true
			ok, err, path = inner_block(ctx, b, at, parts)
			if not ok then
				return nil, err, path
			end
		end
	end
	local missing = {}
	for id in pairs(ids) do
		if not seen[id] then
			missing[#missing + 1] = id
		end
	end
	if #missing > 0 then
		table.sort(missing)
		return fail(("%s: tool_use ids were found without tool_result blocks immediately after: %s. " ..
		    "Each tool_use block must have a corresponding tool_result block in the next message.")
		    :format(turn.at, table.concat(missing, ", ")), turn.at)
	end
	if #parts == 1 and parts[1].type == "text" then
		out[#out + 1] = { role = "user", content = parts[1].text }
	elseif #parts > 0 then
		out[#out + 1] = { role = "user", content = parts }
	elseif not next(seen) then
		out[#out + 1] = { role = "user", content = "" }
	end
	return true
end

-- Translates an assistant turn. Returns the set of its tool_use ids.
-- Thinking blocks go to ctx.thinking for a check after the model is known.
local function assistant_turn(ctx, turn, out)
	local texts, calls, ids = {}, json.array(), {}
	local msg = { role = "assistant" }
	for _, e in ipairs(turn.blocks) do
		local b, at = e[1], e[2]
		local ok, err, path = check_block(ctx, b, at, ROLE_BLOCKS.assistant)
		if not ok then
			return nil, err, path
		end
		if b.type == "text" then
			if type(b.text) ~= "string" then
				return fail(at .. ".text: must be a string", at .. ".text")
			end
			texts[#texts + 1] = b.text
		elseif b.type == "tool_use" then
			if type(b.id) ~= "string" or not b.id:find(ID_PATTERN) then
				return fail(at .. ".id: must match ^[a-zA-Z0-9_-]+$", at .. ".id")
			elseif type(b.name) ~= "string" or b.name == "" then
				return fail(at .. ".name: must be a non-empty string", at .. ".name")
			elseif not is_object(b.input) then
				return fail(at .. ".input: must be an object", at .. ".input")
			end
			ids[b.id] = true
			calls[#calls + 1] = { id = b.id, type = "function",
				["function"] = { name = b.name, arguments = assert(json.encode(json.object(b.input))) } }
		elseif b.type == "thinking" then
			if type(b.thinking) ~= "string" or type(b.signature) ~= "string" then
				return fail(at .. ": thinking and signature must be strings", at)
			end
			ctx.thinking[#ctx.thinking + 1] = { msg = msg, text = b.thinking, sig = b.signature, at = at }
		else
			ctx.dropped = ctx.dropped + 1
		end
	end
	msg.content = table.concat(texts)
	if #calls > 0 then
		msg.tool_calls = calls
		if msg.content == "" then
			msg.content = json.null
		end
	end
	out[#out + 1] = msg
	return ids
end

-- The chat content of a system prompt: a string, or text parts from a
-- list of { block, path }.
local function system_parts(ctx, sys, at)
	if type(sys) == "string" then
		return sys
	end
	local parts = json.array()
	for j, b in ipairs(sys) do
		local p = ("%s.%d"):format(at, j - 1)
		if type(b) == "table" and b[2] then
			b, p = b[1], b[2]
		end
		local ok, err, path = check_block(ctx, b, p, ROLE_BLOCKS.system)
		if not ok then
			return nil, err, path
		elseif type(b.text) ~= "string" then
			return fail(p .. ".text: must be a string", p .. ".text")
		end
		parts[#parts + 1] = { type = "text", text = b.text }
	end
	return parts
end

-- Translates the system prompt and the messages into chat messages.
local function convert_messages(ctx, body)
	local out = json.array()
	if body.system ~= nil then
		if type(body.system) ~= "string" and not json.isarray(body.system) then
			return fail("system: must be a string or an array of text blocks", "system")
		end
		local c, err, path = system_parts(ctx, body.system, "system")
		if not c then
			return nil, err, path
		end
		if (type(c) == "string" and c ~= "") or (type(c) == "table" and #c > 0) then
			out[1] = { role = "system", content = c }
		end
	end
	local turns, err, path = merge(ctx, body.messages)
	if not turns then
		return nil, err, path
	elseif #turns == 0 then
		return fail("messages: at least one message is required", "messages")
	elseif turns[#turns].role == "assistant" then
		return fail("messages: a final assistant message (prefill) is not supported by this server", "messages")
	end
	local ids = {}
	for _, turn in ipairs(turns) do
		if turn.role == "user" then
			local ok
			ok, err, path = user_turn(ctx, turn, ids, out)
			if not ok then
				return nil, err, path
			end
			ids = {}
		elseif turn.role == "assistant" then
			if next(ids) then
				return fail(turn.at .. ": tool_use ids were found without tool_result blocks immediately after",
				    turn.at)
			end
			ids, err, path = assistant_turn(ctx, turn, out)
			if not ids then
				return nil, err, path
			end
		else
			local c
			c, err, path = system_parts(ctx, turn.blocks, turn.at .. ".content")
			if not c then
				return nil, err, path
			end
			out[#out + 1] = { role = "system", content = c }
		end
	end
	return out
end

local TOOL_KEYS = set_of("name", "description", "input_schema", "type", "cache_control", "strict",
	"eager_input_streaming", "input_examples", "defer_loading", "allowed_callers")
-- Tool fields that change nothing here.
local TOOL_IGNORED = set_of("eager_input_streaming", "input_examples", "defer_loading", "allowed_callers")

local function convert_tools(ctx, tools)
	local out = json.array()
	for i, t in ipairs(tools) do
		local at = ("tools.%d"):format(i - 1)
		if not is_object(t) then
			return fail(at .. ": must be an object", at)
		elseif t.type ~= nil and t.type ~= "custom" then
			return fail(("%s: Input tag '%s' found using 'type' does not match any of the expected tags: " ..
			    "'custom'"):format(at, tostring(t.type)), at .. ".type")
		end
		local ok, err, path = keys(ctx, t, TOOL_KEYS, at)
		if not ok then
			return nil, err, path
		end
		if type(t.name) ~= "string" or not t.name:find("^[%w_-]+$") or #t.name > 128 then
			return fail(at .. ".name: must match ^[a-zA-Z0-9_-]{1,128}$", at .. ".name")
		elseif not is_object(t.input_schema) then
			return fail(at .. ".input_schema: must be an object", at .. ".input_schema")
		elseif t.description ~= nil and type(t.description) ~= "string" then
			return fail(at .. ".description: must be a string", at .. ".description")
		elseif t.strict ~= nil and type(t.strict) ~= "boolean" then
			return fail(at .. ".strict: must be a boolean", at .. ".strict")
		end
		for k in pairs(t) do
			if TOOL_IGNORED[k] then
				ctx:ignore("tools." .. k)
			end
		end
		out[i] = translate.function_tool(t.name, t.description, t.input_schema, t.strict)
	end
	return out
end

local function convert_tool_choice(ctx, tc, chat)
	local ok, err, path = keys(ctx, tc, set_of("type", "name", "disable_parallel_tool_use"), "tool_choice")
	if not ok then
		return nil, err, path
	end
	local dp = tc.disable_parallel_tool_use
	if dp ~= nil and type(dp) ~= "boolean" then
		return fail("tool_choice.disable_parallel_tool_use: must be a boolean", "tool_choice")
	elseif dp and tc.type == "none" then
		return fail("tool_choice.disable_parallel_tool_use: not allowed with type none", "tool_choice")
	end
	if tc.type == "auto" then
		chat.tool_choice = "auto"
	elseif tc.type == "any" then
		chat.tool_choice = "required"
	elseif tc.type == "none" then
		chat.tool_choice = "none"
	elseif tc.type == "tool" then
		if type(tc.name) ~= "string" then
			return fail("tool_choice.name: must be a string", "tool_choice")
		end
		chat.tool_choice = { type = "function", ["function"] = { name = tc.name } }
	else
		return fail("tool_choice.type: must be auto, any, tool or none", "tool_choice")
	end
	if dp then
		chat.parallel_tool_calls = false
	end
	return true
end

local function convert_thinking(ctx, th, chat)
	local ok, err, path = keys(ctx, th, set_of("type", "budget_tokens", "display", "block_binding"), "thinking")
	if not ok then
		return nil, err, path
	end
	if th.block_binding ~= nil then
		return fail("thinking.block_binding: not supported by this server", "thinking.block_binding")
	elseif th.display ~= nil and th.display ~= "summarized" and th.display ~= "omitted" then
		if th.display == "updates" then
			return fail('thinking.display: "updates" is not supported by this server', "thinking.display")
		end
		return fail("thinking.display: must be summarized or omitted", "thinking.display")
	end
	if th.type == "enabled" then
		local b = math.tointeger(th.budget_tokens)
		if not b or b < 1024 then
			return fail("thinking.budget_tokens: must be an integer of at least 1024", "thinking.budget_tokens")
		end
		ctx.budget = b
	elseif th.type == "disabled" then
		if th.display ~= nil then
			return fail("thinking.display: not allowed with type disabled", "thinking.display")
		end
	elseif th.type ~= "adaptive" then
		return fail("thinking.type: must be enabled, adaptive or disabled", "thinking.type")
	end
	if th.budget_tokens ~= nil and th.type ~= "enabled" then
		return fail("thinking.budget_tokens: only with type enabled", "thinking.budget_tokens")
	end
	ctx.display = th.display or "summarized"
	translate.set_thinking(chat, th.type ~= "disabled")
	return true
end

local EFFORTS = set_of("none", "minimal", "low", "medium", "high", "xhigh", "max")

local function convert_output_config(ctx, oc, chat)
	local ok, err, path = keys(ctx, oc, set_of("effort", "format", "task_budget"), "output_config")
	if not ok then
		return nil, err, path
	end
	if oc.task_budget ~= nil then
		return fail("output_config.task_budget: not supported by this server", "output_config.task_budget")
	end
	if oc.effort ~= nil then
		if not EFFORTS[oc.effort] then
			return fail("output_config.effort: must be low, medium, high, xhigh or max", "output_config.effort")
		end
		-- "none" turns thinking off, as the chat reasoning_effort; the
		-- others do not change the reasoning length.
		chat.reasoning_effort = oc.effort
		if oc.effort ~= "none" then
			ctx:ignore("output_config.effort")
		end
	end
	local f = oc.format
	if f ~= nil then
		if not is_object(f) or f.type ~= "json_schema" or not is_object(f.schema) then
			return fail('output_config.format: must be {"type": "json_schema", "schema": {...}}',
			    "output_config.format")
		end
		chat.response_format = { type = "json_schema", json_schema = { schema = f.schema } }
	end
	return true
end

-- Chat check parameters to Messages field names.
local PARAMS = { response_format = "output_config.format", tools = "tools", tool_choice = "tool_choice",
	messages = "messages", reasoning_effort = "output_config.effort" }

--- Translates a Messages request body into a chat request.
-- The chat request is checked with `lgml.server.chat_request.check`.
-- Thinking blocks are checked later, with `attach_thinking`, since
-- their signatures depend on the model. For count_tokens, count is
-- true: the generation fields are accepted and ignored, and
-- `max_tokens` is optional.
-- @tparam table body the decoded body
-- @tparam lgml.server.translate.Context ctx the request context; it
-- also gets `display`, `budget`, `thinking` and `dropped`
-- @tparam[opt] boolean count true for count_tokens
-- @treturn[1] table the chat request
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] string the path of the bad field
function M.translate(body, ctx, count)
	if not is_object(body) then
		return fail("the request body must be a JSON object", nil)
	end
	ctx.display, ctx.thinking, ctx.dropped = "summarized", {}, 0
	local ok, err, path = translate.check_fields(body, M.FIELDS, "", ctx, extra)
	if not ok then
		return nil, err, path
	end
	if body.model == nil then
		return fail("model: Field required", "model")
	elseif body.messages == nil then
		return fail("messages: Field required", "messages")
	elseif body.max_tokens == nil and not count then
		return fail("max_tokens: Field required", "max_tokens")
	end
	if body.context_management ~= nil then
		ctx:ignore("context_management")
	end
	local chat = { model = body.model }
	local msgs
	msgs, err, path = convert_messages(ctx, body)
	if not msgs then
		return nil, err, path
	end
	chat.messages = msgs
	if body.tools ~= nil then
		chat.tools, err, path = convert_tools(ctx, body.tools)
		if not chat.tools then
			return nil, err, path
		end
	end
	if body.tool_choice ~= nil then
		ok, err, path = convert_tool_choice(ctx, body.tool_choice, chat)
		if not ok then
			return nil, err, path
		end
	end
	if body.thinking ~= nil then
		ok, err, path = convert_thinking(ctx, body.thinking, chat)
		if not ok then
			return nil, err, path
		end
	end
	if body.output_config ~= nil then
		ok, err, path = convert_output_config(ctx, body.output_config, chat)
		if not ok then
			return nil, err, path
		end
	end
	if not count then
		chat.temperature, chat.top_p, chat.top_k = body.temperature, body.top_p, body.top_k
	end
	local param
	ok, err, param = chat_request.check(chat)
	if not ok then
		local p = PARAMS[param] or param
		return fail(p and p .. ": " .. err or err, p)
	end
	if count then
		chat.count_only = true
	else
		chat.max_tokens, chat.stop = body.max_tokens, body.stop_sequences
		chat.reasoning_budget = ctx.budget
	end
	return chat
end

--- Checks the thinking blocks of a translated request and puts their
-- reasoning into the chat messages.
-- A block with an lgml signature that verifies gives its reasoning
-- (see `lgml.server.translate.verify`); one that fails gives an error.
-- A block with another signature, as an Anthropic one, is dropped and
-- counted in `ctx.dropped`.
-- @tparam lgml.server.translate.Context ctx the request context
-- @tparam string key the server key
-- @tparam string model the served model id
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] string the path of the block
function M.attach_thinking(ctx, key, model)
	for _, t in ipairs(ctx.thinking) do
		if translate.ours(t.sig) then
			local text = translate.verify(key, model, t.sig, t.text)
			if not text then
				return fail(t.at .. ": Invalid `signature` in `thinking` block", t.at)
			end
			local m = t.msg
			m.reasoning_content = m.reasoning_content and m.reasoning_content .. "\n\n" .. text or text
		else
			ctx.dropped = ctx.dropped + 1
		end
	end
	return true
end

-- replies

local STOP_REASONS = {
	tool_calls = "tool_use", length = "max_tokens", context = "model_context_window_exceeded",
	stop_sequence = "stop_sequence", stop = "end_turn",
}

--- The Messages `usage` object of a reply.
-- `input_tokens` counts the prompt tokens that were not in the cache,
-- `cache_read_input_tokens` those that were; their sum is the prompt.
-- @tparam table c the counts of `lgml.server.translate.counts`
-- @treturn table the usage
function M.usage(c)
	return {
		input_tokens = c.prompt - c.cached, output_tokens = c.output,
		cache_creation_input_tokens = 0, cache_read_input_tokens = c.cached,
		cache_creation = { ephemeral_5m_input_tokens = 0, ephemeral_1h_input_tokens = 0 },
		output_tokens_details = { thinking_tokens = c.reasoning },
		service_tier = "standard",
	}
end

-- The input object of a tool call from its argument text.
local function tool_input(b)
	local v = json.decode(translate.text(b))
	if not is_object(v) then
		L:warn("tool arguments are not a JSON object", "name", b.name, "arguments", translate.text(b))
		return json.object()
	end
	return json.object(v)
end

-- A reply: the request state that the reply needs.
local function new_reply(res, ctx, opts, model, max)
	return {
		res = res, ctx = ctx, key = opts.key, model = model, max = max,
		id = translate.id("msg_"), rid = "req_" .. (res.rid or translate.hex_id("", 8)),
	}
end

-- The content block of a finished block.
local function content_block(r, b)
	local text = translate.text(b)
	if b.kind == "reasoning" then
		local omitted = r.ctx.display == "omitted"
		return { type = "thinking", thinking = omitted and "" or text,
			signature = translate.sign(r.key, r.model, text, omitted) }
	elseif b.kind == "tool" then
		return { type = "tool_use", id = b.id, name = b.name, input = tool_input(b) }
	end
	return { type = "text", text = text }
end

local function stop_of(r, sink)
	local f = sink.finished
	local reason = STOP_REASONS[translate.ending(f, #sink.calls, r.max)]
	local seq = reason == "stop_sequence" and f.usage.stop_sequence or json.null
	return reason, seq
end

local function reply_whole(r, sink)
	local content = json.array()
	for i, b in ipairs(sink.list) do
		content[i] = content_block(r, b)
	end
	local reason, seq = stop_of(r, sink)
	return {
		id = r.id, type = "message", role = "assistant", model = r.model, content = content,
		stop_reason = reason, stop_sequence = seq, stop_details = json.null,
		usage = M.usage(translate.counts(sink.finished.usage)),
	}
end

-- The headers of each reply: the request id, and the notes of the
-- request context.
local function headers_of(r, extra_headers)
	local h = extra_headers or {}
	h["request-id"] = r.rid
	r.ctx:headers(h)
	if r.ctx.dropped and r.ctx.dropped > 0 then
		h["x-lgml-thinking-dropped"] = r.ctx.dropped
	end
	if r.ctx.betas then
		h["x-lgml-ignored-betas"] = r.ctx.betas
	end
	return h
end

local function send_error(r, status, msg)
	return r.res:json(status, M.error_body(status, msg, r.rid), headers_of(r))
end

-- The beta names of the anthropic-beta header, comma separated with
-- optional spaces; repeats are joined.
local function betas(req)
	local v = req.headers["anthropic-beta"]
	if not v then
		return nil
	end
	local t = {}
	for name in v:gmatch("[^,%s]+") do
		t[#t + 1] = name
	end
	return #t > 0 and table.concat(t, ",") or nil
end

-- Reads and translates the request. Returns the reply state and the
-- chat request, or sends an error and returns nil.
local function prepare(eng, opts, req, res, count)
	local ctx = translate.context()
	local r = new_reply(res, ctx, opts, nil, nil)
	ctx.betas = betas(req)
	local version = req.headers["anthropic-version"]
	if version and version ~= M.VERSION then
		send_error(r, 400, ("anthropic-version: %q is not supported by this server; use %s"):format(version,
		    M.VERSION))
		return nil
	end
	if type(eng.chat) ~= "function" then
		send_error(r, 404, "this server does not serve chat")
		return nil
	end
	local body, err = json.decode(req.body)
	if body == nil then
		send_error(r, 400, "the request body is not valid JSON: " .. tostring(err))
		return nil
	end
	if count and type(body) == "table" then
		for k in pairs(M.COUNT_IGNORED) do
			body[k] = nil
		end
	end
	local chat, path
	chat, err, path = M.translate(body, ctx, count)
	if not chat then
		L:debug("bad request", "path", path, "err", err)
		send_error(r, 400, err)
		return nil
	end
	r.model = translate.served_model(eng, body.model, "chat", media.has_images(chat.messages))
	r.max = chat.max_tokens
	local ok
	ok, err = M.attach_thinking(ctx, opts.key, r.model)
	if not ok then
		send_error(r, 400, err)
		return nil
	end
	r.stream = body.stream == true
	return r, chat
end

-- streaming

local function event(name, value)
	value.type = name
	return sse.event(name, value)
end

-- The message_start event, with the prompt size when the engine gave it.
local function start_event(r, sink)
	local n, cached = sink.prompt_n or 0, sink.cached_n or 0
	local u = M.usage({ prompt = n, cached = cached, output = 0, reasoning = 0 })
	u.output_tokens_details = nil
	return event("message_start", { message = {
		id = r.id, type = "message", role = "assistant", model = r.model, content = json.array(),
		stop_reason = json.null, stop_sequence = json.null, stop_details = json.null, usage = u,
	} })
end

-- The handler of the block sink for a stream: each block goes out as
-- content_block_start, deltas and content_block_stop.
local function stream_handler(r, w)
	local h, sink = {}, nil
	local function first()
		return start_event(r, sink)
	end
	local function send(text)
		return w:send(text, first, headers_of(r))
	end
	function h.bind(_, s)
		sink = s
	end
	function h.prompt()
		return send("")
	end
	function h.open(_, b)
		local cb
		if b.kind == "reasoning" then
			cb = { type = "thinking", thinking = "", signature = "" }
		elseif b.kind == "tool" then
			cb = { type = "tool_use", id = b.id, name = b.name, input = json.object() }
		else
			cb = { type = "text", text = "" }
		end
		return send(event("content_block_start", { index = b.index, content_block = cb }))
	end
	function h.delta(_, b, text)
		local d
		if b.kind == "reasoning" then
			if r.ctx.display == "omitted" then
				return true
			end
			d = { type = "thinking_delta", thinking = text }
		elseif b.kind == "tool" then
			d = { type = "input_json_delta", partial_json = text }
		else
			d = { type = "text_delta", text = text }
		end
		return send(event("content_block_delta", { index = b.index, delta = d }))
	end
	function h.close(_, b)
		local t = {}
		if b.kind == "reasoning" then
			local omitted = r.ctx.display == "omitted"
			if omitted then
				t[#t + 1] = event("content_block_delta", { index = b.index,
					delta = { type = "thinking_delta", thinking = "" } })
			end
			t[#t + 1] = event("content_block_delta", { index = b.index, delta = { type = "signature_delta",
				signature = translate.sign(r.key, r.model, translate.text(b), omitted) } })
		end
		t[#t + 1] = event("content_block_stop", { index = b.index })
		return send(table.concat(t))
	end
	function h.ping()
		return send(event("ping", {}))
	end
	function h.finish()
		local reason, seq = stop_of(r, sink)
		local u = M.usage(translate.counts(sink.finished.usage))
		u.cache_creation, u.service_tier = nil, nil
		return send(event("message_delta", { delta = { stop_reason = reason, stop_sequence = seq }, usage = u }) ..
		    event("message_stop", {}))
	end
	function h.fail(_, status, msg)
		local e = M.error_body(status, msg, r.rid)
		return send(event("error", { error = e.error, request_id = e.request_id }))
	end
	return h
end

-- Runs a streamed request. The stream starts when the engine tells the
-- prompt size, at the first block, or at the first keepalive ping.
local function run_stream(eng, r, chat)
	local w = translate.writer(r.res)
	local h = stream_handler(r, w)
	local sink = translate.blocks(r.res, h, { tool_id = "toolu_" })
	h:bind(sink)
	w:keepalive(function()
		h:ping()
	end, M.PING_MS)
	local ok, msg, status = translate.run(eng, chat, sink, r.model)
	w:stop()
	if sink:cancelled() then
		r.res.close = true
		return
	elseif not ok and not w.started then
		return send_error(r, status or 500, msg or "engine failed")
	end
	r.res.err = not ok and (msg or "engine failed") or nil
	local sent
	if ok then
		sent = h:finish()
	else
		sent = h:fail(status or 500, msg or "engine failed")
	end
	if not sent or not w:finish() then
		r.res.close = true
	end
end

--- Milliseconds without bytes out before a `ping` event in a stream.
M.PING_MS = 10000

-- POST /v1/messages/count_tokens: the prompt size from the template
-- and the tokenizer of the model, images too.
local function count_tokens(eng, opts, req, res)
	local r, chat = prepare(eng, opts, req, res, true)
	if not r then
		return
	end
	log.note("model", r.model)
	local sink = translate.blocks(res)
	local ok, msg, status = eng:chat(chat, sink)
	if sink:cancelled() then
		res.close = true
		return
	elseif not ok or not sink.finished then
		return send_error(r, status or 500, msg or "engine failed")
	end
	local n = translate.counts(sink.finished.usage).prompt
	log.note("prompt_tokens", n)
	return res:json(200, { input_tokens = n }, headers_of(r))
end

--- The Anthropic model list for `GET /v1/models` with an
-- `anthropic-version` header.
-- @tparam lgml.server.engine.Engine eng the engine
-- @treturn table `{ data, has_more, first_id, last_id }`; each entry is
-- `{ type = "model", id, display_name, created_at }`
function M.models(eng)
	local data = json.array()
	for i, m in ipairs(eng:models()) do
		data[i] = { type = "model", id = m.id, display_name = m.id,
			created_at = os.date("!%Y-%m-%dT%H:%M:%SZ", m.created or 0) }
	end
	return { data = data, has_more = false, first_id = data[1] and data[1].id or json.null,
		last_id = data[#data] and data[#data].id or json.null }
end

local function messages(eng, opts, req, res)
	local r, chat = prepare(eng, opts, req, res, false)
	if not r then
		return
	end
	if r.stream then
		return run_stream(eng, r, chat)
	end
	local sink = translate.blocks(res, nil, { tool_id = "toolu_" })
	local ok, msg, status = translate.run(eng, chat, sink, r.model)
	if sink:cancelled() then
		res.close = true
		return
	elseif not ok then
		return send_error(r, status or 500, msg or "engine failed")
	end
	return res:json(200, reply_whole(r, sink), headers_of(r))
end

--- Adds the Messages routes for eng to srv.
-- `POST /v1/messages` and `POST /v1/messages/count_tokens`. The paths
-- take a query string, such as `?beta=true`, which they ignore.
-- @tparam lgml.server.httpd.Server srv the server
-- @tparam lgml.server.engine.Engine eng the engine
-- @tparam table opts the options: `key`, the server key for reasoning
-- signatures (see `lgml.server.translate.load_key`)
function M.install(srv, eng, opts)
	srv:route("POST", "/v1/messages", function(req, res)
		return messages(eng, opts, req, res)
	end)
	srv:route("POST", "/v1/messages/count_tokens", function(req, res)
		return count_tokens(eng, opts, req, res)
	end)
end

return M
