--- Streaming parser for Gemma-4 output.
-- It reads `<|channel>thought ...<channel|>` reasoning blocks and
-- `<|tool_call>call:f{...}<tool_call|>` calls. It is a subclass of
-- `lgml.chat.hermes.Parser`, and the events are the same as the hermes
-- parser gives.
-- @module lgml.chat.gemma4_parser
local hermes = require("lgml.chat.hermes")

local M = {}

local P = setmetatable({}, { __index = hermes.P })
P.__index = P
P.call_tag = "<|tool_call>"

local TURN = "<|turn>model\n"

--- Makes a parser.
-- @tparam[opt] table opts the options:
--
-- - `in_think`: the prompt already opened `<|channel>thought`
-- - `tools`: tool names or OpenAI tools; calls to other names become
--   content
-- @treturn lgml.chat.hermes.Parser the parser
function M.new_parser(opts)
	opts = opts or {}
	return setmetatable({
		state = opts.in_think and "reasoning" or "start",
		buf = "",
		hold = "",
		hold_kind = "content",
		lead = true,
		ncalls = 0,
		names = hermes.name_set(opts.tools),
	}, P)
end

-- Length of the longest suffix of s that is a proper prefix of a tag.
local function partial(s, tags)
	for k = math.min(#s, 11), 1, -1 do
		local tail = s:sub(-k)
		for _, tag in ipairs(tags) do
			if k < #tag and tag:sub(1, k) == tail then
				return k
			end
		end
	end
	return 0
end

local TAGS = { "<|channel>", "<channel|>", "<|tool_call>" }

-- A model turn header at the start is not output.
function P:step_start(final)
	local buf = self.buf
	if buf:sub(1, #TURN) == TURN then
		self.buf = buf:sub(#TURN + 1)
	elseif not final and TURN:sub(1, #buf) == buf then
		return false
	end
	self.state = "content"
	return true
end

function P:step_content(final)
	local buf = self.buf
	local at, tag
	for _, t in ipairs(TAGS) do
		local i = buf:find(t, 1, true)
		if i and (not at or i < at) then
			at, tag = i, t
		end
	end
	if not at then
		local k = final and 0 or partial(buf, TAGS)
		self:emit("content", buf:sub(1, #buf - k))
		self.buf = buf:sub(#buf - k + 1)
		return false
	end
	self:emit("content", buf:sub(1, at - 1))
	self.buf = buf:sub(at + #tag)
	if tag == "<|channel>" then
		self.state = "channel"
	elseif tag == "<channel|>" then
		-- A stray end of thought is dropped.
		self:switch("content")
	else
		self.pre_ws, self.hold = self.hold, ""
		self.state = "call"
		self.call = require("lgml.chat.gemma4_call").new(self.names)
	end
	return true
end

-- After <|channel>: "thought" starts reasoning. An empty channel is
-- dropped.
function P:step_channel(final)
	local buf = self.buf
	if buf:sub(1, 7) == "thought" then
		self.buf = buf:sub(8)
		self:switch("reasoning")
		return true
	elseif not final and ("thought"):sub(1, #buf) == buf then
		return false
	end
	self.state = "content"
	return true
end

function P:step_reasoning(final)
	if not self:scan("reasoning", "<channel|>", final) then
		return false
	end
	self:switch("content")
	return true
end

return M
