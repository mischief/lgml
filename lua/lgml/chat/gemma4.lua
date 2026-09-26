--- Gemma-4 chat template and tool-call parser.
-- `render` gives the same bytes as the gguf Jinja template run by
-- llama.cpp, with its message fixes: no `<bos>` (the tokenizer adds
-- it), tool call arguments parsed as JSON, developer role as system,
-- and the gemma4 continuation and generation prompt rules.
-- @module lgml.chat.gemma4
local tojson = require("lgml.chat.tojson")

local M = {}

--- The image tokens, as llama.cpp mtmd for gemma4v: `begin` and
-- `finish` go around the soft `token`, one for each image embedding.
M.image = { begin = "<|image>", token = "<|image|>", finish = "<image|>" }

local Q = '<|"|>'

local function isnull(v)
	return v == nil or v == tojson.null or type(v) == "userdata"
end

local function kind(v)
	if type(v) ~= "table" then
		return type(v)
	end
	return tojson.kind(v)
end

-- Jinja truth: empty strings, lists and maps are false.
local function truthy(v)
	if isnull(v) or v == false or v == 0 or v == "" then
		return false
	elseif type(v) == "table" then
		return next(v) ~= nil
	end
	return true
end

-- v[k], or nil when v is not a map.
local function get(v, k)
	if kind(v) == "object" then
		local r = v[k]
		if not isnull(r) then
			return r
		end
	end
	return nil
end

local function trim(s)
	return (s:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function upper(v)
	return type(v) == "string" and v:upper() or ""
end

-- Keys in byte order, as llama.cpp dictsort. Not a map: no keys.
local function sorted_keys(t)
	local keys = {}
	if kind(t) ~= "object" then
		return keys
	end
	for k in pairs(t) do
		keys[#keys + 1] = k
	end
	table.sort(keys)
	return keys
end

-- A Jinja value as text: floats as C++ std::to_string without the
-- trailing zeros, None as nothing.
local function scalar(v)
	if isnull(v) then
		return ""
	elseif math.type(v) == "float" then
		local s = string.format("%f", v):gsub("0+$", "")
		return s:sub(-1) == "." and s .. "0" or s
	end
	return tostring(v)
end

local format_argument

function format_argument(v, escape_keys)
	local k = kind(v)
	if k == "string" then
		return Q .. v .. Q
	elseif k == "boolean" then
		return tostring(v)
	elseif k == "object" then
		local parts = {}
		for _, key in ipairs(sorted_keys(v)) do
			local name = escape_keys and Q .. key .. Q or key
			parts[#parts + 1] = name .. ":" .. format_argument(v[key], escape_keys)
		end
		return "{" .. table.concat(parts, ",") .. "}"
	elseif k == "array" then
		local parts = {}
		for i = 1, #v do
			parts[i] = format_argument(v[i], escape_keys)
		end
		return "[" .. table.concat(parts, ",") .. "]"
	end
	return scalar(v)
end

local function quoted_list(list)
	local parts = {}
	if kind(list) ~= "array" then
		return ""
	end
	for i, x in ipairs(list) do
		parts[i] = Q .. scalar(x) .. Q
	end
	return table.concat(parts, ",")
end

local STANDARD = { description = true, type = true, properties = true, required = true,
	nullable = true }

local format_parameters

local function format_items(items)
	local out, first = {}, true
	for _, ik in ipairs(sorted_keys(items)) do
		local iv = items[ik]
		if not isnull(iv) then
			if not first then
				out[#out + 1] = ","
			end
			first = false
			if ik == "properties" then
				out[#out + 1] = "properties:{"
				if kind(iv) == "object" then
					out[#out + 1] = format_parameters(iv, get(items, "required") or {})
				end
				out[#out + 1] = "}"
			elseif ik == "required" then
				out[#out + 1] = "required:[" .. quoted_list(iv) .. "]"
			elseif ik == "type" then
				out[#out + 1] = "type:" .. format_argument(upper(iv), true)
			else
				out[#out + 1] = ik .. ":" .. format_argument(iv, true)
			end
		end
	end
	return table.concat(out)
end

-- One property of a tool schema: name:{description,...,type}.
local function format_property(key, value, out)
	local comma = false
	local function sep()
		if comma then
			out[#out + 1] = ","
		end
		comma = true
	end
	out[#out + 1] = key .. ":{"
	local desc = get(value, "description")
	if truthy(desc) then
		out[#out + 1] = "description:" .. Q .. scalar(desc) .. Q
		comma = true
	end
	local ty = upper(get(value, "type"))
	if ty == "STRING" then
		if truthy(get(value, "enum")) then
			sep()
			out[#out + 1] = "enum:" .. format_argument(value.enum, true)
		end
	elseif ty == "ARRAY" then
		local items = get(value, "items")
		if kind(items) == "object" and truthy(items) then
			sep()
			out[#out + 1] = "items:{" .. format_items(items) .. "}"
		end
	end
	if truthy(get(value, "nullable")) then
		sep()
		out[#out + 1] = "nullable:true"
	end
	if ty == "OBJECT" then
		local props = get(value, "properties")
		if kind(props) == "object" then
			sep()
			out[#out + 1] = "properties:{" ..
				format_parameters(props, get(value, "required") or {}) .. "}"
		elseif kind(value) == "object" then
			sep()
			out[#out + 1] = "properties:{" ..
				format_parameters(value, get(value, "required") or {}, true) .. "}"
		end
		local req = get(value, "required")
		if truthy(req) then
			sep()
			out[#out + 1] = "required:[" .. quoted_list(req) .. "]"
		end
	end
	sep()
	out[#out + 1] = "type:" .. Q .. ty .. Q .. "}"
end

function format_parameters(props, _, filter_keys)
	local out, first = {}, true
	for _, key in ipairs(sorted_keys(props)) do
		if not filter_keys or not STANDARD[key] then
			if not first then
				out[#out + 1] = ","
			end
			first = false
			format_property(key, props[key], out)
		end
	end
	return table.concat(out)
end

-- llama.cpp passes only name, description and parameters of a tool.
local function format_declaration(tool)
	local f = get(tool, "function") or tool
	local out = { "declaration:", scalar(get(f, "name")), "{description:", Q,
		scalar(get(f, "description")), Q }
	local params = get(f, "parameters")
	if truthy(params) then
		out[#out + 1] = ",parameters:{"
		local props = get(params, "properties")
		if truthy(props) then
			out[#out + 1] = "properties:{" .. format_parameters(props, get(params, "required")) .. "},"
		end
		local req = get(params, "required")
		if truthy(req) then
			out[#out + 1] = "required:[" .. quoted_list(req) .. "],"
		end
		if truthy(get(params, "type")) then
			out[#out + 1] = "type:" .. Q .. upper(params.type) .. Q .. "}"
		end
	end
	out[#out + 1] = "}"
	return table.concat(out)
end

-- JSON decoder for tool call arguments. Keeps floats apart from
-- integers, as nlohmann json does. Returns nil on bad input.
local decode_json
do
	local s, pos
	local ESC = { b = "\b", f = "\f", n = "\n", r = "\r", t = "\t" }
	local function ws()
		pos = s:find("[^ \t\r\n]", pos) or #s + 1
	end
	local function str()
		local out = {}
		pos = pos + 1
		while true do
			local i = s:find('["\\]', pos)
			if not i then
				error("string")
			end
			out[#out + 1] = s:sub(pos, i - 1)
			if s:sub(i, i) == '"' then
				pos = i + 1
				return table.concat(out)
			end
			local e = s:sub(i + 1, i + 1)
			if e == "u" then
				local h = s:match("^%x%x%x%x", i + 2) or error("escape")
				local c = tonumber(h, 16)
				pos = i + 6
				if c >= 0xd800 and c < 0xdc00 and s:match("^\\u[dD][c-fC-F]%x%x", pos) then
					local lo = tonumber(s:sub(pos + 2, pos + 5), 16)
					c = 0x10000 + (c - 0xd800) * 0x400 + (lo - 0xdc00)
					pos = pos + 6
				end
				out[#out + 1] = utf8.char(c)
			else
				out[#out + 1] = ESC[e] or (e:match('["\\/]') and e) or error("escape")
				pos = i + 2
			end
		end
	end
	local value
	local function list(close, item)
		pos = pos + 1
		ws()
		if s:sub(pos, pos) == close then
			pos = pos + 1
			return
		end
		while true do
			ws()
			item()
			ws()
			local c = s:sub(pos, pos)
			pos = pos + 1
			if c == close then
				return
			elseif c ~= "," then
				error("list")
			end
		end
	end
	function value()
		ws()
		local c = s:sub(pos, pos)
		if c == "{" then
			local t = setmetatable({}, { __jsontype = "object" })
			list("}", function()
				if s:sub(pos, pos) ~= '"' then
					error("key")
				end
				local k = str()
				ws()
				if s:sub(pos, pos) ~= ":" then
					error("colon")
				end
				pos = pos + 1
				t[k] = value()
			end)
			return t
		elseif c == "[" then
			local t = setmetatable({}, { __jsontype = "array" })
			list("]", function()
				t[#t + 1] = value()
			end)
			return t
		elseif c == '"' then
			return str()
		end
		for lit, v in pairs({ ["true"] = true, ["false"] = false, null = tojson.null }) do
			if s:sub(pos, pos + #lit - 1) == lit then
				pos = pos + #lit
				return v
			end
		end
		local num = s:match("^-?%d+%.?%d*[eE]?[-+]?%d*", pos) or error("value")
		pos = pos + #num
		if num:find("[.eE]") then
			local v = tonumber(num) or error("number")
			return math.type(v) == "float" and v or v * 1.0
		end
		return math.tointeger(tonumber(num)) or tonumber(num)
	end
	function decode_json(text)
		s, pos = text, 1
		local good, v = pcall(value)
		ws()
		if good and pos > #s then
			return v
		end
		return nil
	end
end

local function render_call(tc)
	local f = get(tc, "function") or tc
	local args = f.arguments
	if type(args) == "string" then
		args = decode_json(args) or args
	end
	local body = ""
	if kind(args) == "object" then
		local parts = {}
		for _, k in ipairs(sorted_keys(args)) do
			parts[#parts + 1] = k .. ":" .. format_argument(args[k], false)
		end
		body = table.concat(parts, ",")
	elseif type(args) == "string" then
		body = args
	end
	return "<|tool_call>call:" .. scalar(f.name) .. "{" .. body .. "}<tool_call|>"
end

local function response_block(name, response)
	local body
	if kind(response) == "object" then
		local parts = {}
		for _, k in ipairs(sorted_keys(response)) do
			parts[#parts + 1] = k .. ":" .. format_argument(response[k], false)
		end
		body = table.concat(parts, ",")
	else
		body = "value:" .. format_argument(response, false)
	end
	return "<|tool_response>response:" .. name .. "{" .. body .. "}<tool_response|>"
end

local function strip_thinking(text)
	local out, pos = {}, 1
	while true do
		local i = text:find("<channel|>", pos, true)
		local part = text:sub(pos, (i or 0) - 1)
		local c = part:find("<|channel>", 1, true)
		out[#out + 1] = c and part:sub(1, c - 1) or part
		if not i then
			break
		end
		pos = i + 10
	end
	return trim(table.concat(out))
end

-- Content of a message as the template renders it.
local function render_content(m, model)
	local c = m.content
	if type(c) == "string" then
		return model and strip_thinking(c) or trim(c)
	elseif kind(c) ~= "array" then
		return ""
	end
	local out = {}
	for _, item in ipairs(c) do
		if get(item, "type") == "text" then
			local t = scalar(get(item, "text"))
			out[#out + 1] = model and strip_thinking(t) or trim(t)
		end
	end
	return table.concat(out)
end

local function system_text(m)
	local c = m.content
	if type(c) == "string" then
		return trim(c)
	elseif kind(c) ~= "array" then
		return ""
	end
	local out = {}
	for _, item in ipairs(c) do
		out[#out + 1] = trim(scalar(get(item, "text"))) .. " "
	end
	return table.concat(out)
end

-- llama.cpp drops empty names and ids.
local function nonempty(v)
	if isnull(v) or v == "" then
		return nil
	end
	return v
end

-- Tool responses that follow message i, and the name of each.
local function render_responses(msgs, i, calls, out)
	local any = false
	for k = i + 1, #msgs do
		local f = msgs[k]
		if f.role ~= "tool" then
			break
		end
		local name = nonempty(f.name) or "unknown"
		for _, tc in ipairs(calls) do
			if nonempty(get(tc, "id")) == nonempty(f.tool_call_id) then
				name = (get(tc, "function") or tc).name
			end
		end
		local body = f.content
		if kind(body) == "array" then
			local t = {}
			for _, p in ipairs(body) do
				if get(p, "type") == "text" then
					t[#t + 1] = scalar(get(p, "text"))
				end
			end
			body = table.concat(t)
		elseif isnull(body) then
			body = ""
		end
		out[#out + 1] = response_block(name, body)
		any = true
	end
	return any
end

local function slice(t, a, b)
	local out = {}
	for i = a, b do
		out[#out + 1] = t[i]
	end
	return out
end

local function role_of(m)
	return m.role == "developer" and "system" or m.role
end

local function render_messages(msgs, ns, out)
	local last_user = 0
	for i, m in ipairs(msgs) do
		if role_of(m) == "user" then
			last_user = i
		end
	end
	local prev_role
	for i, m in ipairs(msgs) do
		local mrole = role_of(m)
		if mrole ~= "tool" then
			ns.prev = nil
			local role = mrole == "assistant" and "model" or mrole
			if not (role == "model" and prev_role == "assistant") then
				out[#out + 1] = "<|turn>" .. role .. "\n"
			end
			prev_role = mrole
			local calls = kind(m.tool_calls) == "array" and truthy(m.tool_calls) and m.tool_calls or nil
			local thought = m.reasoning_content
			if truthy(thought) and i > last_user and calls then
				out[#out + 1] = "<|channel>thought\n" .. thought .. "\n<channel|>"
			end
			local responded = false
			if calls then
				for _, tc in ipairs(calls) do
					out[#out + 1] = render_call(tc)
				end
				ns.prev = "tool_call"
				responded = render_responses(msgs, i, calls, out)
				if responded then
					ns.prev = "tool_response"
				end
			end
			local content = render_content(m, role == "model")
			out[#out + 1] = content
			if ns.prev == "tool_call" and not responded then
				out[#out + 1] = "<|tool_response>"
			elseif not (responded and trim(content) == "") then
				out[#out + 1] = "<turn|>\n"
			end
		end
	end
end

local function render_template(msgs, opts, gen_prompt)
	local out, ns = {}, {}
	local think = opts.enable_thinking ~= false
	local tools = kind(opts.tools) == "array" and truthy(opts.tools) and opts.tools or nil
	local first = msgs[1]
	local sys = first and role_of(first) == "system"
	if think or tools or sys then
		out[#out + 1] = "<|turn>system\n"
		if think then
			out[#out + 1] = "<|think|>\n"
		end
		if sys then
			out[#out + 1] = system_text(first)
		end
		for _, t in ipairs(tools or {}) do
			out[#out + 1] = "<|tool>" .. trim(format_declaration(t)) .. "<tool|>"
		end
		out[#out + 1] = "<turn|>\n"
	end
	render_messages(sys and slice(msgs, 2, #msgs) or msgs, ns, out)
	if gen_prompt and ns.prev ~= "tool_response" and ns.prev ~= "tool_call" then
		out[#out + 1] = "<|turn>model\n"
	end
	return table.concat(out)
end

local function continuation(m)
	local content = m.content
	if kind(content) == "array" then
		local t = {}
		for _, p in ipairs(content) do
			if get(p, "type") == "text" then
				t[#t + 1] = scalar(get(p, "text"))
			end
		end
		content = table.concat(t, "\n\n")
	elseif type(content) ~= "string" then
		content = ""
	end
	local reasoning = type(m.reasoning_content) == "string" and m.reasoning_content or ""
	return content, reasoning
end

--- Renders messages as prompt text.
-- @tparam {table,...} messages OpenAI chat messages
-- @tparam[opt] table opts the options:
--
-- - `tools`: OpenAI tools list
-- - `add_generation_prompt`: add the assistant turn header
-- - `enable_thinking`: default true, as in llama-server
-- - `continue_final_message`: continue a final assistant turn: its
--   reasoning, then its content
-- @treturn string the prompt text
function M.render(messages, opts)
	opts = opts or {}
	local last = messages[#messages]
	if opts.continue_final_message and last and last.role == "assistant" then
		local text = render_template(slice(messages, 1, #messages - 1), opts, false)
		local content, reasoning = continuation(last)
		if content == "" and reasoning == "" and not truthy(last.tool_calls) then
			return text
		end
		if text:sub(-8) == "<turn|>\n" then
			text = text .. "<|turn>model\n"
		end
		text = text .. "<|channel>thought\n" .. reasoning
		if content ~= "" or reasoning == "" then
			text = text .. "<channel|>" .. content
		end
		return text
	end
	local text = render_template(messages, opts, opts.add_generation_prompt)
	if opts.add_generation_prompt and text:sub(-8) == "<turn|>\n" then
		text = text .. "<|turn>model\n"
	end
	return text
end

--- Makes an output parser. This is `lgml.chat.gemma4_parser.new_parser`.
-- @function new_parser
-- @tparam[opt] table opts the parser options
-- @treturn lgml.chat.hermes.Parser the parser
M.new_parser = require("lgml.chat.gemma4_parser").new_parser

return M
