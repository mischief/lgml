--- Qwen3.5 chat template with XML tool calls.
-- A call is `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`.
-- `render` gives the same bytes as the gguf Jinja template run by
-- llama.cpp, with its message fixes: developer role as system, text
-- parts joined, tool call arguments parsed as JSON, and the llama.cpp
-- continuation rule. Two template variants exist; see `variant`.
-- @module lgml.chat.qwen35
local tojson = require("lgml.chat.tojson")

local M = {}

local TOOLS_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
local TOOLS_TAIL = "\n</tools>\n\nIf you choose to call a function ONLY reply in the "
	.. "following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
	.. "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n"
	.. "This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
	.. "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
	.. "specified format: an inner <function=...></function> block must be nested within "
	.. "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may "
	.. "provide optional reasoning for your function call in natural language BEFORE the "
	.. "function call, but NOT after\n- If there is no function call available, answer the "
	.. "question like normal with your current knowledge and do not tell the user about "
	.. "function calls\n</IMPORTANT>"

local EFFORT = {
	xhigh = "Reasoning effort is set to xhigh. Please think carefully through the task, "
		.. "validate key assumptions, consider plausible alternatives, and prioritize "
		.. "correctness, consistency, and clarity in the final answer.",
	low = "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
		.. "directly to the conclusion without unnecessary elaboration.",
	medium = "",
}

local function isnull(v)
	return v == nil or v == tojson.null or type(v) == "userdata"
end

local function kind(v)
	if type(v) ~= "table" then
		return type(v)
	end
	return tojson.kind(v)
end

-- llama.cpp Jinja prints floats as C++ streams do: %g.
local function float(x)
	return string.format("%g", x)
end

local function json(v)
	return tojson.encode(v, float)
end

-- Jinja trim strips ASCII white space only.
local function trim(s)
	return (s:gsub("^[ \t\n\r\v\f]+", ""):gsub("[ \t\n\r\v\f]+$", ""))
end

local function lstrip_nl(s)
	return (s:gsub("^\n+", ""))
end

local function rstrip_nl(s)
	return (s:gsub("\n+$", ""))
end

local function split_first(s, sep)
	local i = s:find(sep, 1, true)
	return i and s:sub(1, i - 1) or s
end

local function split_last(s, sep)
	local last
	local i = s:find(sep, 1, true)
	while i do
		last = i
		i = s:find(sep, i + 1, true)
	end
	return last and s:sub(last + #sep) or s
end

-- Message content as a string. Text parts join with sep, as llama.cpp:
-- no sep while the text is empty.
local function text(content, sep)
	if type(content) == "string" then
		return content
	elseif kind(content) ~= "array" then
		return ""
	end
	local s = ""
	for _, p in ipairs(content) do
		if type(p) == "table" and p.type == "text" and type(p.text) == "string" then
			s = s .. (s ~= "" and sep or "") .. p.text
		end
	end
	return s
end

-- Ordered JSON decoder for tool call arguments. Floats stay apart
-- from integers, as in nlohmann json. Returns nil on bad input.
local decode_json
do
	local s, pos
	local ESC = { b = "\b", f = "\f", n = "\n", r = "\r", t = "\t", ['"'] = '"', ["\\"] = "\\", ["/"] = "/" }
	local function ws()
		pos = s:find("[^ \t\r\n]", pos) or #s + 1
	end
	local function str()
		local out = {}
		pos = pos + 1
		while true do
			local i = s:find('["\\%c]', pos)
			if not i or s:byte(i) < 32 then
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
					c = 0x10000 + (c - 0xd800) * 0x400 + (tonumber(s:sub(pos + 2, pos + 5), 16) - 0xdc00)
					pos = pos + 6
				end
				out[#out + 1] = utf8.char(c)
			else
				out[#out + 1] = ESC[e] or error("escape")
				pos = i + 2
			end
		end
	end
	-- Integers wrap to int64 up to 2^64 - 1, as nlohmann uint64 in Jinja.
	local function int(num)
		local digits = num:gsub("^-", "")
		if #digits > 20 or (#digits == 20 and digits > "18446744073709551615") then
			return tonumber(num) + 0.0
		end
		local v = 0
		for d in digits:gmatch("%d") do
			v = v * 10 + tonumber(d)
		end
		if num:sub(1, 1) == "-" then
			return math.tointeger(tonumber(num)) or tonumber(num) + 0.0
		end
		return v
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
	function value(depth)
		assert(depth < 200, "depth")
		ws()
		local c = s:sub(pos, pos)
		if c == "{" then
			local order = {}
			local t = setmetatable({}, { __jsontype = "object", __order = order })
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
				if t[k] == nil then
					order[#order + 1] = k
				end
				t[k] = value(depth + 1)
			end)
			return t
		elseif c == "[" then
			local t = setmetatable({}, { __jsontype = "array" })
			local n = 0
			list("]", function()
				n = n + 1
				t[n] = value(depth + 1)
			end)
			return t
		elseif c == '"' then
			return str()
		end
		for _, lit in ipairs({ "true", "false", "null" }) do
			if s:sub(pos, pos + #lit - 1) == lit then
				pos = pos + #lit
				return ({ ["true"] = true, ["false"] = false, null = tojson.null })[lit]
			end
		end
		local num = s:match("^%-?%d[%d%.eE%+%-]*", pos) or error("value")
		local rest = num:gsub("^%-?", ""):gsub("^0", "", 1)
		if rest == num:gsub("^%-?", "") then
			rest = rest:gsub("^[1-9]%d*", "", 1)
		end
		rest = rest:gsub("^%.%d+", "", 1):gsub("^[eE][-+]?%d+", "", 1)
		if rest ~= "" then
			error("number")
		end
		pos = pos + #num
		if num:find("[.eE]") then
			return tonumber(num) + 0.0
		end
		return int(num)
	end
	function decode_json(text)
		s, pos = text, 1
		local good, v = pcall(value, 0)
		ws()
		if good and pos > #s then
			return v
		end
		return nil
	end
end

-- The tool list as llama.cpp rebuilds it.
local function tools_list(tools)
	if kind(tools) ~= "array" or #tools == 0 then
		return nil
	end
	local out = {}
	for i, t in ipairs(tools) do
		local f = type(t) == "table" and t["function"]
		f = type(f) == "table" and f or {}
		local params = f.parameters
		if params == nil then
			params = setmetatable({}, { __jsontype = "object" })
		end
		local fn = setmetatable({ name = f.name, description = f.description or "", parameters = params },
			{ __order = { "name", "description", "parameters" } })
		out[i] = setmetatable({ type = "function", ["function"] = fn }, { __order = { "type", "function" } })
	end
	return out
end

-- Messages as llama.cpp gives them to the template.
local function fix_messages(messages)
	local out = {}
	for i, m in ipairs(messages) do
		local r = { role = m.role == "developer" and "system" or m.role, content = text(m.content, "\n") }
		if type(m.reasoning_content) == "string" and m.reasoning_content ~= "" then
			r.reasoning_content = m.reasoning_content
		end
		if kind(m.tool_calls) == "array" and #m.tool_calls > 0 then
			r.tool_calls = {}
			for k, tc in ipairs(m.tool_calls) do
				local f = type(tc) == "table" and tc["function"]
				f = type(f) == "table" and f or {}
				local args = f.arguments
				if type(args) == "string" then
					args = decode_json(args)
					if args == nil then
						error("chat: tool call arguments are not JSON", 0)
					end
				end
				r.tool_calls[k] = { name = f.name, arguments = args }
			end
		end
		out[i] = r
	end
	return out
end

-- Argument value text. The effort variant uses tojson for all but
-- strings; the other uses Jinja string for scalars.
local function arg_value(v, effort)
	local k = kind(v)
	if k == "string" then
		return v
	elseif effort or k == "array" or k == "object" then
		return json(v)
	elseif isnull(v) then
		return "None"
	elseif k == "boolean" then
		return v and "True" or "False"
	elseif math.type(v) == "float" then
		return float(v)
	end
	return tostring(v)
end

local function render_args(name, args, effort, out)
	if kind(args) == "object" then
		for _, key in ipairs(tojson.keys(args)) do
			out[#out + 1] = "<parameter=" .. tostring(key) .. ">\n" .. arg_value(args[key], effort)
				.. "\n</parameter>\n"
		end
	elseif not effort then
		return
	elseif type(args) == "string" then
		if trim(args) ~= "" then
			error(('chat: tool call arguments for function "%s" were passed as a JSON string'):format(name), 0)
		end
	elseif not isnull(args) then
		error(('chat: tool call arguments for function "%s" must be an object'):format(name), 0)
	end
end

local function render_calls(m, content, effort, out)
	for k, tc in ipairs(m.tool_calls or {}) do
		local name = tc.name
		if type(name) ~= "string" then
			if effort then
				error("chat: tool call is missing a function name", 0)
			end
			name = tostring(name)
		end
		if k > 1 then
			out[#out + 1] = "\n"
		elseif trim(content) ~= "" then
			out[#out + 1] = "\n\n"
		end
		out[#out + 1] = "<tool_call>\n<function=" .. name .. ">\n"
		render_args(name, tc.arguments, effort, out)
		out[#out + 1] = "</function>\n</tool_call>"
	end
end

-- Split assistant content into (content, reasoning) like the template.
local function split_reasoning(m, content, effort)
	if m.reasoning_content then
		return content, m.reasoning_content
	elseif effort or not content:find("</think>", 1, true) then
		return content, ""
	end
	local reasoning = lstrip_nl(split_last(rstrip_nl(split_first(content, "</think>")), "<think>"))
	return lstrip_nl(split_last(content, "</think>")), reasoning
end

local function render_assistant(m, content, think, effort, out)
	local reasoning
	content, reasoning = split_reasoning(m, content, effort)
	out[#out + 1] = "<|im_start|>assistant\n"
	if think then
		out[#out + 1] = "<think>\n" .. trim(reasoning) .. "\n</think>\n\n"
	end
	out[#out + 1] = content
	render_calls(m, content, effort, out)
	out[#out + 1] = "<|im_end|>\n"
end

local function render_tool(msgs, i, content, out)
	local prev, nxt = msgs[i - 1], msgs[i + 1]
	if prev and prev.role ~= "tool" then
		out[#out + 1] = "<|im_start|>user"
	end
	out[#out + 1] = "\n<tool_response>\n" .. content .. "\n</tool_response>"
	if not nxt or nxt.role ~= "tool" then
		out[#out + 1] = "<|im_end|>\n"
	end
end

-- System text and the count of leading system messages.
local function system_prefix(msgs, effort)
	if effort then
		local count, merged = 0, ""
		for i, m in ipairs(msgs) do
			if count ~= i - 1 or m.role ~= "system" then
				break
			end
			local c = trim(m.content)
			if c ~= "" then
				merged = merged .. (merged ~= "" and "\n" or "") .. c
			end
			count = count + 1
		end
		return merged, count
	end
	if not msgs[1] or msgs[1].role ~= "system" then
		return "", 0
	end
	local first = trim(msgs[1].content)
	if msgs[2] and msgs[2].role == "system" then
		return first .. "\n" .. trim(msgs[2].content), 2
	end
	return first, 1
end

local function reasoning_instructions(opts)
	if opts.enable_thinking == false then
		return ""
	end
	local effort = opts.reasoning_effort or "xhigh"
	if effort == "high" then
		effort = "xhigh"
	end
	local s = EFFORT[effort]
	if not s then
		error("chat: unexpected reasoning effort " .. tostring(opts.reasoning_effort)
			.. ". Supported types are xhigh (default), medium, and low.", 0)
	end
	return s
end

local function system_block(msgs, opts, effort, out)
	local merged, nsys = system_prefix(msgs, effort)
	local instr = effort and reasoning_instructions(opts) or ""
	local tools = tools_list(opts.tools)
	if tools then
		out[#out + 1] = "<|im_start|>system\n"
		if instr ~= "" then
			out[#out + 1] = instr .. "\n\n"
		end
		out[#out + 1] = TOOLS_HEAD
		for _, t in ipairs(tools) do
			out[#out + 1] = "\n" .. json(t)
		end
		out[#out + 1] = TOOLS_TAIL
		if merged ~= "" then
			out[#out + 1] = "\n\n" .. merged
		end
		out[#out + 1] = "<|im_end|>\n"
	elseif merged ~= "" then
		out[#out + 1] = "<|im_start|>system\n" .. (instr ~= "" and instr .. "\n\n" or "") .. merged .. "<|im_end|>\n"
	elseif instr ~= "" then
		out[#out + 1] = "<|im_start|>system\n" .. instr .. "<|im_end|>\n"
	end
	return nsys
end

-- 1-based index of the last real user query; #msgs if none.
local function last_query(msgs)
	for i = #msgs, 1, -1 do
		if msgs[i].role == "user" then
			local c = trim(msgs[i].content)
			if not (c:sub(1, 15) == "<tool_response>" and c:sub(-16) == "</tool_response>") then
				return i
			end
		end
	end
	return #msgs
end

local function render_template(msgs, opts, gen_prompt)
	if #msgs == 0 then
		error("chat: no messages provided", 0)
	end
	local effort = opts.variant == "effort"
	local out = {}
	local nsys = system_block(msgs, opts, effort, out)
	local lq = last_query(msgs)
	local keep = opts.preserve_thinking ~= false
	for i, m in ipairs(msgs) do
		local role = m.role
		if i > nsys and (effort or role ~= "system") then
			local content = trim(m.content)
			if role == "system" then
				error("chat: system message must be at the beginning", 0)
			elseif role == "user" then
				out[#out + 1] = "<|im_start|>user\n" .. content .. "<|im_end|>\n"
			elseif role == "assistant" then
				render_assistant(m, content, not effort or keep or i > lq, effort, out)
			elseif role == "tool" then
				render_tool(msgs, i, content, out)
			elseif effort then
				error("chat: unexpected message role", 0)
			end
		end
	end
	if gen_prompt then
		out[#out + 1] = "<|im_start|>assistant\n"
		out[#out + 1] = opts.enable_thinking == false and "<think>\n\n</think>\n\n" or "<think>\n"
	end
	return table.concat(out)
end

local function nonempty(v)
	return (type(v) == "string" and v ~= "") or (kind(v) == "array" and #v > 0)
end

-- The continued turn: (reasoning, content, mode), as llama.cpp gives
-- it. The mode is "reasoning_content" or "content". An empty turn
-- gives nil.
local function continuation(m, mode)
	local reasoning = type(m.reasoning_content) == "string" and m.reasoning_content or ""
	local content, parts = text(m.content, "\n\n"), kind(m.content) == "array" and #m.content > 0
	if content == "" and not parts and reasoning == "" and not nonempty(m.tool_calls)
		and not nonempty(m.name) and not nonempty(m.tool_call_id) then
		return nil
	end
	if mode ~= "content" and mode ~= "reasoning_content" then
		mode = (reasoning ~= "" and content == "" and not parts) and "reasoning_content" or "content"
	end
	return reasoning, content, mode
end

--- Finds the template variant.
-- The `"effort"` variant merges all leading system messages, adds
-- reasoning effort text and can drop old reasoning. The default
-- variant merges up to two system messages and takes reasoning from
-- `<think>` in content.
-- @tparam string tpl the Jinja template text
-- @treturn ?string `"effort"` or nil
function M.variant(tpl)
	if type(tpl) == "string" and tpl:find("reasoning_effort", 1, true) and tpl:find("sysns", 1, true) then
		return "effort"
	end
	return nil
end

--- Renders messages as prompt text.
-- @tparam {table,...} messages OpenAI chat messages
-- @tparam[opt] table opts the options:
--
-- - `tools`: OpenAI tools list
-- - `add_generation_prompt`: add the assistant turn header
-- - `enable_thinking`: default true, as in llama-server
-- - `continue_final_message`: continue a final assistant turn; true
--   picks the mode, or `"content"` or `"reasoning_content"`
-- - `variant`: `"effort"` for the reasoning effort template
-- - `reasoning_effort`: `"xhigh"` (default), `"high"`, `"medium"` or
--   `"low"`; effort variant only
-- - `preserve_thinking`: false drops reasoning before the last user
--   query; effort variant only
-- @treturn string the prompt text
-- @raise for arguments that are not JSON, and where the template raises
function M.render(messages, opts)
	opts = opts or {}
	local msgs = fix_messages(messages)
	local last = messages[#messages]
	if opts.continue_final_message and last and last.role == "assistant" then
		local prompt = render_template(table.move(msgs, 1, #msgs - 1, 1, {}), opts, false)
		local reasoning, content, mode = continuation(last, opts.continue_final_message)
		if not reasoning then
			return prompt
		end
		prompt = prompt .. "<|im_start|>assistant\n<think>\n" .. reasoning
		if mode == "content" then
			prompt = prompt .. "\n</think>\n\n" .. content
		end
		return prompt
	end
	return render_template(msgs, opts, opts.add_generation_prompt)
end

--- Makes a family module bound to a template variant.
-- @tparam string tpl the Jinja template text
-- @treturn table a module with `render` and `new_parser`
function M.bind(tpl)
	local variant = M.variant(tpl)
	return {
		variant = variant,
		render = function(messages, opts)
			local o = {}
			for k, v in pairs(opts or {}) do
				o[k] = v
			end
			o.variant = o.variant or variant
			return M.render(messages, o)
		end,
		new_parser = M.new_parser,
	}
end

--- Makes an output parser. This is `lgml.chat.qwen35_parser.new_parser`.
-- @function new_parser
-- @tparam[opt] table opts the parser options
-- @treturn lgml.chat.hermes.Parser the parser
M.new_parser = require("lgml.chat.qwen35_parser").new_parser

return M
