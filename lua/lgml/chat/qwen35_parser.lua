--- Streaming parser for Qwen3.5 output.
-- It reads `<think>` reasoning and XML tool calls (see
-- `lgml.chat.qwen35_call`). It is a subclass of
-- `lgml.chat.hermes.Parser`, and the events are the same as the hermes
-- parser gives. The Qwen3.5 prompt opens `<think>` itself, so the
-- parser starts in reasoning unless thinking is off.
-- @module lgml.chat.qwen35_parser
local hermes = require("lgml.chat.hermes")
local call = require("lgml.chat.qwen35_call")

local M = {}

local P = setmetatable({}, { __index = hermes.P })
P.__index = P
P.call_tag = "<tool_call>\n"

--- Makes a parser.
-- @tparam[opt] table opts the options:
--
-- - `in_think`: the prompt already opened `<think>`; default true
--   unless `enable_thinking` is false
-- - `enable_thinking`: false when the prompt closed the think block
-- - `tools`: tool names or OpenAI tools. Calls to other names become
--   content. Parameter types come from the tool schemas.
-- @treturn lgml.chat.hermes.Parser the parser
function M.new_parser(opts)
	opts = opts or {}
	local in_think = opts.in_think
	if in_think == nil then
		in_think = opts.enable_thinking ~= false
	end
	return setmetatable({
		state = in_think and "reasoning" or "start",
		buf = "",
		hold = "",
		hold_kind = "content",
		lead = true,
		ncalls = 0,
		tools = call.tools(opts.tools),
	}, P)
end

function P:step_content(final)
	if not self:scan("content", P.call_tag, final) then
		return false
	end
	self.pre_ws, self.hold = self.hold, ""
	self.state = "call"
	self.call = call.new(self.tools)
	return true
end

return M
