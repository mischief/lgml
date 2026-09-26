--- Qwen3 chat template and hermes-style tool-call parser.
-- `render` gives the same bytes as the gguf Jinja template.
-- @module lgml.chat.qwen3
local tojson = require("lgml.chat.tojson")

local M = {}

local TOOLS_HEAD = "# Tools\n\nYou may call one or more functions to assist with "
	.. "the user query.\n\nYou are provided with function signatures within "
	.. "<tools></tools> XML tags:\n<tools>"
local TOOLS_TAIL = "\n</tools>\n\nFor each function call, return a json object "
	.. "with function name and arguments within <tool_call></tool_call> XML "
	.. 'tags:\n<tool_call>\n{"name": <function-name>, "arguments": '
	.. "<args-json-object>}\n</tool_call><|im_end|>\n"

-- Python str.strip family. chars is a Lua pattern class body.
local function lstrip(s, chars)
	return (s:gsub("^[" .. chars .. "]+", ""))
end

local function rstrip(s, chars)
	local i = #s
	while i > 0 and s:sub(i, i):find("[" .. chars .. "]") do
		i = i - 1
	end
	return s:sub(1, i)
end

local function strip(s, chars)
	return rstrip(lstrip(s, chars), chars)
end

-- Message content as a string. Text parts join with "\n" (llama.cpp).
local function text(content)
	if content == nil or content == tojson.null then
		return ""
	elseif type(content) == "string" then
		return content
	end
	local parts = {}
	for _, p in ipairs(content) do
		if p.type == "text" then
			parts[#parts + 1] = p.text
		end
	end
	return table.concat(parts, "\n")
end

local function has(s, sub)
	return s:find(sub, 1, true) ~= nil
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

local function is_tool_response(c)
	return c:sub(1, 15) == "<tool_response>" and c:sub(-16) == "</tool_response>"
end

-- 1-based index of the last real user query; #messages if none.
local function last_query(messages)
	for i = #messages, 1, -1 do
		local m = messages[i]
		if m.role == "user" and not is_tool_response(text(m.content)) then
			return i
		end
	end
	return #messages
end

local function system_block(messages, tools, out)
	local first = messages[1]
	local sys = first and first.role == "system"
	if tools and #tools > 0 then
		out[#out + 1] = "<|im_start|>system\n"
		if sys then
			out[#out + 1] = text(first.content) .. "\n\n"
		end
		out[#out + 1] = TOOLS_HEAD
		for _, t in ipairs(tools) do
			out[#out + 1] = "\n" .. tojson.encode(t)
		end
		out[#out + 1] = TOOLS_TAIL
	elseif sys then
		out[#out + 1] = "<|im_start|>system\n" .. text(first.content) .. "<|im_end|>\n"
	end
end

-- Split assistant content into (content, reasoning) like the template.
local function split_reasoning(m)
	local content = text(m.content)
	local rc = m.reasoning_content
	if rc ~= nil and rc ~= tojson.null then
		return content, rc
	end
	if not has(content, "</think>") then
		return content, ""
	end
	local reasoning = rstrip(split_first(content, "</think>"), "\n")
	reasoning = lstrip(split_last(reasoning, "<think>"), "\n")
	return lstrip(split_last(content, "</think>"), "\n"), reasoning
end

local function render_tool_call(tc, first, content)
	local s = ""
	if not first or content ~= "" then
		s = "\n"
	end
	local f = tc["function"] or tc
	local args = f.arguments
	if type(args) ~= "string" then
		args = tojson.encode(args)
	end
	return s .. '<tool_call>\n{"name": "' .. f.name .. '", "arguments": ' .. args .. "}\n</tool_call>"
end

local function render_assistant(m, i, n, lq, out)
	local content, reasoning = split_reasoning(m)
	local head, body = "<|im_start|>assistant\n", content
	if i > lq and (i == n or strip(reasoning, "%s") ~= "") then
		head = head .. "<think>\n" .. strip(reasoning, "\n") .. "\n</think>\n\n"
		body = lstrip(content, "\n")
	end
	out[#out + 1] = head .. body
	for k, tc in ipairs(m.tool_calls or {}) do
		out[#out + 1] = render_tool_call(tc, k == 1, content)
	end
	out[#out + 1] = "<|im_end|>\n"
end

local function render_tool(messages, i, out)
	local prev, nxt = messages[i - 1], messages[i + 1]
	if not prev or prev.role ~= "tool" then
		out[#out + 1] = "<|im_start|>user"
	end
	out[#out + 1] = "\n<tool_response>\n" .. text(messages[i].content) .. "\n</tool_response>"
	if not nxt or nxt.role ~= "tool" then
		out[#out + 1] = "<|im_end|>\n"
	end
end

local function render_messages(messages, out)
	local lq, n = last_query(messages), #messages
	for i, m in ipairs(messages) do
		if m.role == "user" or (m.role == "system" and i > 1) then
			out[#out + 1] = "<|im_start|>" .. m.role .. "\n" .. text(m.content) .. "<|im_end|>\n"
		elseif m.role == "assistant" then
			render_assistant(m, i, n, lq, out)
		elseif m.role == "tool" then
			render_tool(messages, i, out)
		end
	end
end

--- Renders messages as prompt text.
-- @tparam {table,...} messages OpenAI chat messages
-- @tparam[opt] table opts the options:
--
-- - `tools`: OpenAI tools list
-- - `add_generation_prompt`: add the assistant turn header
-- - `enable_thinking`: false adds an empty think block
-- - `continue_final_message`: leave a final assistant turn open
-- @treturn string the prompt text
function M.render(messages, opts)
	opts = opts or {}
	local out = {}
	system_block(messages, opts.tools, out)
	render_messages(messages, out)
	local last = messages[#messages]
	if opts.continue_final_message and last and last.role == "assistant" then
		out[#out] = nil
		return table.concat(out)
	end
	if opts.add_generation_prompt then
		out[#out + 1] = "<|im_start|>assistant\n"
		if opts.enable_thinking == false then
			out[#out + 1] = "<think>\n\n</think>\n\n"
		end
	end
	return table.concat(out)
end

--- Makes an output parser. This is `lgml.chat.hermes.new_parser`.
-- @function new_parser
-- @tparam[opt] table opts the parser options
-- @treturn lgml.chat.hermes.Parser the parser
M.new_parser = require("lgml.chat.hermes").new_parser

return M
