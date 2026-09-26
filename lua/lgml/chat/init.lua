--- Chat template family selection from gguf metadata.
-- A family module has `render(messages, opts)`, which returns the
-- prompt text, and `new_parser(opts)`, which returns a streaming output
-- parser. See `lgml.chat.qwen3`, `lgml.chat.qwen35` and
-- `lgml.chat.gemma4`.
-- @module lgml.chat
local M = {}

--- Family module names by family name.
M.families = { qwen3 = "lgml.chat.qwen3", qwen35 = "lgml.chat.qwen35", gemma4 = "lgml.chat.gemma4" }

local GEMMA4_ARCH = { gemma4 = true, ["diffusion-gemma"] = true }
local QWEN3_ARCH = { qwen3 = true, qwen3moe = true, qwen3next = true, qwen35moe = true }

local function has(s, sub)
	return s:find(sub, 1, true) ~= nil
end

--- Finds the chat family for gguf metadata.
-- @tparam table meta the gguf metadata
-- @treturn[1] string the family name: `"qwen3"`, `"qwen35"` or
-- `"gemma4"`
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.detect(meta)
	local arch = meta["general.architecture"]
	local tpl = meta["tokenizer.chat_template"] or ""
	local chatml = has(tpl, "<tool_call>") and has(tpl, "<|im_start|>")
	-- Qwen3.5 templates write XML <function=...> calls and think blocks.
	-- Qwen3-Coder templates also use <function=...> but differ.
	if chatml and has(tpl, "<function=") and has(tpl, "<think>") and has(tpl, "tool | tojson") then
		return "qwen35"
	end
	if QWEN3_ARCH[arch] and (tpl == "" or (chatml and not has(tpl, "<function="))) then
		return "qwen3"
	end
	-- llama.cpp picks gemma4 by the template's tool call syntax.
	if has(tpl, "'<|tool_call>call:'") or (tpl == "" and GEMMA4_ARCH[arch]) then
		return "gemma4"
	end
	return nil, string.format("chat: no template for architecture %q", tostring(arch))
end

--- Loads the chat family module for gguf metadata.
-- A module with `bind` is bound to the template text.
-- @tparam table meta the gguf metadata
-- @treturn table the family module, with `render` and `new_parser`
-- @treturn string the family name
-- @raise when no family fits
function M.load(meta)
	local name = assert(M.detect(meta))
	local mod = require(M.families[name])
	if mod.bind then
		mod = mod.bind(meta["tokenizer.chat_template"] or "")
	end
	return mod, name
end

return M
