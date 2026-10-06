--- LLaDA-MoE-v2 chat template and hermes-style tool-call parser.
-- Turns are `<role>NAME</role>text<|role_end|>`. `render` gives the
-- same bytes as the Jinja template of LLaDA-MoE-v2-Instruct, which
-- always writes "detailed thinking off".
-- @module lgml.chat.llada
local tojson = require("lgml.chat.tojson")

local M = {}

local TOOLS_HEAD = "# Tools\n\nYou may call one or more functions to assist with "
	.. "the user query.\n\nYou are provided with function signatures within "
	.. "<tools></tools> XML tags:\n<tools>"
local TOOLS_TAIL = "\n</tools>\n\nFor each function call, return a json object "
	.. "with function name and arguments within <tool_call></tool_call> XML "
	.. 'tags:\n<tool_call>\n{"name": <function-name>, "arguments": '
	.. "<args-json-object>}\n</tool_call>\n"
local END = "<|role_end|>"

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

-- Split assistant content into (content, reasoning) like the template.
local function split_reasoning(m)
	local content = text(m.content)
	local rc = m.reasoning_content
	if type(rc) == "string" then
		return content, rc
	end
	if not content:find("</think>", 1, true) then
		return content, ""
	end
	local reasoning = rstrip(split_first(content, "</think>"), "\n")
	reasoning = lstrip(split_last(reasoning, "<think>"), "\n")
	return lstrip(split_last(content, "</think>"), "\n"), reasoning
end

local function render_assistant(m, i, lq, out)
	local content, reasoning = split_reasoning(m)
	if i > lq and reasoning ~= "" then
		out[#out + 1] = "<role>ASSISTANT</role>\n<think>\n" .. lstrip(rstrip(reasoning, "\n"), "\n")
			.. "\n</think>\n\n" .. lstrip(content, "\n")
	else
		out[#out + 1] = "<role>ASSISTANT</role>" .. content
	end
	for k, tc in ipairs(m.tool_calls or {}) do
		if k > 1 or content ~= "" then
			out[#out + 1] = "\n"
		end
		local f = tc["function"] or tc
		local args = f.arguments
		if type(args) ~= "string" then
			args = tojson.encode(args)
		end
		out[#out + 1] = '<tool_call>\n{"name": "' .. f.name .. '", "arguments": ' .. args .. "}\n</tool_call>"
	end
	out[#out + 1] = END
end

local function render_tool(messages, i, out)
	local prev, nxt = messages[i - 1], messages[i + 1]
	if not prev or prev.role ~= "tool" then
		out[#out + 1] = "<role>OBSERVATION</role>"
	end
	out[#out + 1] = "\n<tool_response>\n" .. text(messages[i].content) .. "\n</tool_response>"
	if not nxt or nxt.role ~= "tool" then
		out[#out + 1] = END
	end
end

--- Renders messages as prompt text.
-- @tparam {table,...} messages OpenAI chat messages
-- @tparam[opt] table opts the options:
--
-- - `tools`: OpenAI tools list
-- - `add_generation_prompt`: add the assistant turn header
-- - `continue_final_message`: leave a final assistant turn open
-- @treturn string the prompt text
function M.render(messages, opts)
	opts = opts or {}
	local out = { "<role>SYSTEM</role>" }
	local first = messages[1]
	if first and first.role == "system" then
		out[#out + 1] = text(first.content) .. "\n"
	end
	if opts.tools and #opts.tools > 0 then
		out[#out + 1] = TOOLS_HEAD
		for _, t in ipairs(opts.tools) do
			out[#out + 1] = "\n" .. tojson.encode(t)
		end
		out[#out + 1] = TOOLS_TAIL
	end
	out[#out + 1] = "detailed thinking off" .. END
	local lq = last_query(messages)
	for i, m in ipairs(messages) do
		if m.role == "user" then
			out[#out + 1] = "<role>HUMAN</role>" .. text(m.content) .. END
		elseif m.role == "system" and i > 1 then
			out[#out + 1] = "<role>SYSTEM</role>" .. text(m.content) .. END
		elseif m.role == "assistant" then
			render_assistant(m, i, lq, out)
		elseif m.role == "tool" then
			render_tool(messages, i, out)
		end
	end
	local last = messages[#messages]
	if opts.continue_final_message and last and last.role == "assistant" then
		out[#out] = nil
		return table.concat(out)
	end
	if opts.add_generation_prompt then
		out[#out + 1] = "<role>ASSISTANT</role>"
	end
	return table.concat(out)
end

--- Makes an output parser. This is `lgml.chat.hermes.new_parser`.
-- @function new_parser
-- @tparam[opt] table opts the parser options
-- @treturn lgml.chat.hermes.Parser the parser
M.new_parser = require("lgml.chat.hermes").new_parser

--- Makes a grammar that forces tool calls. This is
-- `lgml.chat.tool_grammar.hermes`.
-- @function tool_grammar
-- @tparam {table,...} tools OpenAI function tools
-- @tparam[opt] table opts the options
-- @treturn string the GBNF text
-- @treturn string the start rule
M.tool_grammar = require("lgml.chat.tool_grammar").hermes

return M
