--- Streaming parser for Qwen3.5 output.
-- It reads `<think>` reasoning and XML tool calls (see
-- `lgml.chat.qwen35_call`). It is a subclass of
-- `lgml.chat.hermes.Parser`, and the events are the same as the hermes
-- parser gives. The Qwen3.5 prompt opens `<think>` itself, so the
-- parser starts in reasoning unless thinking is off. As in llama.cpp,
-- `<function=NAME>` for a declared NAME also opens a call without
-- `<tool_call>` (Qwen3-Coder).
-- @module lgml.chat.qwen35_parser
local hermes = require("lgml.chat.hermes")
local call = require("lgml.chat.qwen35_call")

local M = {}

local P = setmetatable({}, { __index = hermes.P })
P.__index = P
P.call_tag = "<tool_call>\n"

local FOPEN = "<function="
local MAX_NAME = 512

-- Length of the longest suffix of s that is a proper prefix of tag.
local function partial(s, tag)
	for k = math.min(#s, #tag - 1), 1, -1 do
		if s:sub(-k) == tag:sub(1, k) then
			return k
		end
	end
	return 0
end

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

-- Where a bare call starts in buf from i: "call" when the name is a
-- declared tool, "wait" when more text must decide, else nil.
function P:bare_at(i, final)
	local from = i + #FOPEN
	local gt = self.buf:find("[>\n<]", from)
	if not gt then
		return not final and #self.buf - from < MAX_NAME and "wait" or nil
	end
	local name = self.buf:sub(from, gt - 1)
	if self.buf:sub(gt, gt) == ">" and self.tools and self.tools[name] then
		return "call"
	end
	return nil
end

function P:start_call(at, skip, bare)
	self:emit("content", self.buf:sub(1, at - 1))
	self.buf = self.buf:sub(at + skip)
	self.pre_ws, self.hold = self.hold, ""
	self.state = "call"
	self.call = call.new(self.tools, { bare = bare })
	return true
end

function P:step_content(final)
	local buf, pos = self.buf, 1
	while true do
		local a = buf:find(P.call_tag, pos, true)
		local b = buf:find(FOPEN, pos, true)
		if a and (not b or a < b) then
			return self:start_call(a, #P.call_tag, false)
		elseif not b then
			break
		end
		local r = self:bare_at(b, final)
		if r == "call" then
			return self:start_call(b, 0, true)
		elseif r == "wait" then
			self:emit("content", buf:sub(1, b - 1))
			self.buf = buf:sub(b)
			return false
		end
		pos = b + 1
	end
	local k = final and 0 or math.max(partial(buf, P.call_tag), partial(buf, FOPEN))
	self:emit("content", buf:sub(1, #buf - k))
	self.buf = buf:sub(#buf - k + 1)
	return false
end

local TOPEN = "<tool_call>\n"
local TCALL = TOPEN .. FOPEN
local TEND = "</think>"

-- Qwen3.5 can open a call inside its reasoning, with no `</think>`.
-- A call to a declared tool there ends the reasoning; other text,
-- such as a call to an unknown name, stays reasoning.
function P:step_reasoning(final)
	if not self.tools then
		return hermes.P.step_reasoning(self, final)
	end
	local buf, pos = self.buf, 1
	local t = buf:find(TEND, 1, true)
	while true do
		local c = buf:find(TCALL, pos, true)
		if not c or (t and t < c) then
			break
		end
		local r = self:bare_at(c + #TOPEN, final)
		if r == "call" or r == "wait" then
			self:emit("reasoning", buf:sub(1, c - 1))
			self.buf = buf:sub(c)
			if r == "wait" then
				return false
			end
			self:switch("content")
			return true
		end
		pos = c + 1
	end
	if t then
		return hermes.P.step_reasoning(self, final)
	end
	local k = final and 0 or math.max(partial(buf, TEND), partial(buf, TCALL))
	self:emit("reasoning", buf:sub(1, #buf - k))
	self.buf = buf:sub(#buf - k + 1)
	return false
end

-- A bare call that is not a call: its "<" is content, and the scan
-- goes on after it.
function P:fallback(raw)
	if not self.call.bare then
		return hermes.P.fallback(self, raw)
	end
	self.state, self.hold, self.lead = "content", self.pre_ws, false
	self:emit("content", "<")
	self.buf = raw:sub(2) .. self.buf
end

-- Bytes that a bare call took after its end go back to the text.
function P:on_call_event(ev, rest)
	local back = self.call and self.call.back
	if back and (ev == "end" or ev == "error") then
		rest = back .. rest
	end
	return hermes.P.on_call_event(self, ev, rest)
end

return M
