--- Granite 4 chat template: ChatML with `<think>` and the XML tool calls
-- of Qwen3.5.
-- `render` gives the same bytes as the gguf Jinja template run by
-- llama.cpp, with its message fixes and continuation rule. The parser
-- and tool grammar are the Qwen3.5 ones.
-- @module lgml.chat.granite
local qwen35 = require("lgml.chat.qwen35")

local S = qwen35.shared
local trim, json, split_first, split_last = S.trim, S.json, S.split_first, S.split_last

local M = {}

-- Content after the last `</think>`, or before an unclosed `<think>`.
local function after_think(c)
	if c:find("</think>", 1, true) then
		return split_last(c, "</think>")
	elseif c:find("<think>", 1, true) then
		return split_first(c, "<think>")
	end
	return c
end

local function render_calls(m, out)
	for _, tc in ipairs(m.tool_calls) do
		out[#out + 1] = "<tool_call>\n<function=" .. tostring(tc.name) .. ">\n"
		if S.kind(tc.arguments) == "object" then
			for _, key in ipairs(require("lgml.chat.tojson").keys(tc.arguments)) do
				out[#out + 1] = "<parameter=" .. tostring(key) .. ">\n" .. S.arg_value(tc.arguments[key], false)
					.. "\n</parameter>\n"
			end
		end
		out[#out + 1] = "</function>\n</tool_call>\n"
	end
end

-- An assistant turn. old is true for a turn before the last user
-- message whose thinking the template drops.
local function render_assistant(m, old, out)
	local content
	if type(m.reasoning_content) == "string" and trim(m.reasoning_content) ~= "" then
		content = "<think>\n" .. m.reasoning_content .. "\n</think>\n" .. m.content
	else
		content = m.content
		if not content:find("<think>", 1, true) and not content:find("</think>", 1, true) then
			content = "<think></think>" .. content
		end
	end
	if m.tool_calls then
		out[#out + 1] = "<|im_start|>assistant\n"
		if trim(content) == "" then
			out[#out + 1] = "<think></think>"
		elseif not old then
			out[#out + 1] = trim(content) .. "\n"
		else
			local c = "<think></think>" .. trim(after_think(content))
			out[#out + 1] = c .. "\n"
		end
		render_calls(m, out)
		out[#out + 1] = "<|im_end|>\n"
		return
	end
	local c = content
	if old then
		if c:find("<think>", 1, true) and c:find("</think>", 1, true) then
			c = "<think></think>" .. split_last(c, "</think>")
		end
	end
	out[#out + 1] = "<|im_start|>assistant\n" .. trim(c) .. "<|im_end|>\n"
end

local function render_template(msgs, opts, gen_prompt)
	local tools = S.tools_list(opts.tools)
	local low = opts.reasoning_effort == "low"
	local truncate = opts.truncate_history_thinking ~= false
	local sys, first = "", 1
	if msgs[1] and msgs[1].role == "system" then
		sys, first = msgs[1].content, 2
	end
	local last_user = -1
	for i = first, #msgs do
		if msgs[i].role == "user" then
			last_user = i
		end
	end
	local out = { "<|im_start|>system\n" .. sys }
	if tools then
		if sys ~= "" then
			out[#out + 1] = "\n\n"
		end
		out[#out + 1] = S.TOOLS_HEAD
		for _, t in ipairs(tools) do
			out[#out + 1] = "\n" .. json(t["function"])
		end
		out[#out + 1] = S.TOOLS_TAIL
	end
	out[#out + 1] = "<|im_end|>\n"
	for i = first, #msgs do
		local m = msgs[i]
		local role = m.role
		if role == "assistant" then
			render_assistant(m, truncate and i < last_user, out)
		elseif role == "user" or role == "system" then
			local effort = role == "user" and i == last_user and low and "\n\n{reasoning effort: low}" or ""
			out[#out + 1] = "<|im_start|>" .. role .. "\n" .. m.content .. effort .. "<|im_end|>\n"
		elseif role == "tool" then
			local prev, nxt = msgs[i - 1], msgs[i + 1]
			if i > first and prev.role ~= "tool" then
				out[#out + 1] = "<|im_start|>user\n"
			end
			out[#out + 1] = "<tool_response>\n" .. m.content .. "\n</tool_response>\n"
			if not nxt or nxt.role ~= "tool" then
				out[#out + 1] = "<|im_end|>\n"
			end
		else
			out[#out + 1] = "<|im_start|>" .. tostring(role) .. "\n" .. m.content .. "<|im_end|>\n"
		end
	end
	if gen_prompt then
		out[#out + 1] = opts.enable_thinking == false and "<|im_start|>assistant\n<think></think>"
			or "<|im_start|>assistant\n<think>\n"
	end
	return table.concat(out)
end

--- Renders messages as prompt text.
-- @tparam {table,...} messages OpenAI chat messages
-- @tparam[opt] table opts the options:
--
-- - `tools`: OpenAI tools list
-- - `add_generation_prompt`: add the assistant turn header
-- - `enable_thinking`: default true
-- - `reasoning_effort`: `"low"` asks for brief thinking
-- - `truncate_history_thinking`: false keeps the thinking of turns
--   before the last user message (default true)
-- - `continue_final_message`: continue a final assistant turn; true
--   picks the mode, or `"content"` or `"reasoning_content"`
-- @treturn string the prompt text
-- @raise for tool call arguments that are not JSON
function M.render(messages, opts)
	opts = opts or {}
	local msgs = S.fix_messages(messages)
	local last = messages[#messages]
	if opts.continue_final_message and last and last.role == "assistant" then
		local head = table.move(msgs, 1, #msgs - 1, 1, {})
		local reasoning, content, mode = S.continuation(last, opts.continue_final_message)
		if not reasoning then
			return render_template(head, opts, false)
		end
		local prompt = render_template(head, opts, true)
		if prompt:sub(-8) ~= "<think>\n" then
			prompt = prompt .. "<think>\n"
		end
		prompt = prompt .. reasoning
		if mode == "content" then
			prompt = prompt .. "\n</think>\n" .. content
		end
		return prompt
	end
	return render_template(msgs, opts, opts.add_generation_prompt)
end

--- Makes an output parser. This is `lgml.chat.qwen35_parser.new_parser`.
-- @function new_parser
-- @tparam[opt] table opts the parser options
-- @treturn lgml.chat.hermes.Parser the parser
M.new_parser = qwen35.new_parser

--- Makes a grammar that forces tool calls. This is
-- `lgml.chat.tool_grammar.qwen35`.
-- @function tool_grammar
M.tool_grammar = qwen35.tool_grammar

return M
