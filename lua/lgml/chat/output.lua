--- Sends the output of a chat parser to a sink.
-- Chat engines feed the text of each token to an `Output`. It runs the
-- parser of the chat family (see `lgml.chat`), sends the events to a
-- `lgml.server.engine.Sink` and maps the tool call indexes of the
-- parser to those of the sink. It counts the reasoning tokens.
-- @module lgml.chat.output
local M = {}

local O = {}
O.__index = O

-- Parser states that give content.
local CONTENT = { start = true, content = true }

--- Makes an output.
-- @tparam lgml.server.engine.Sink sink the reply
-- @tparam table parser a parser from the `new_parser` of a chat family
-- @treturn Output the output
function M.new(sink, parser)
	return setmetatable({
		sink = sink, parser = parser, map = {},
		reasoning_tokens = 0,
	}, O)
end

--- An output of one chat reply.
-- The field `reasoning_tokens` counts the tokens that the parser read
-- in a reasoning block, and the tags of the block.
-- @type Output

-- Sends one event. Returns true, or nil when the client is gone.
function O:event(ev)
	local sink, t = self.sink, ev.type
	if t == "reasoning" then
		return sink:reasoning(ev.text)
	elseif t == "content" then
		return sink:content(ev.text)
	elseif t == "tool_call_begin" then
		local idx = sink:tool_call(ev.name)
		self.map[ev.index] = idx
		return idx
	elseif t == "tool_call_args" then
		return sink:tool_arguments(self.map[ev.index], ev.text)
	end
	return true
end

--- Sends parser events.
-- @tparam {table,...} events the events
-- @treturn ?boolean true, or nil when the client is gone
function O:deliver(events)
	for _, ev in ipairs(events) do
		if not self:event(ev) then
			return nil
		end
	end
	return true
end

--- Feeds the text of one token.
-- @tparam string text the text
-- @treturn ?boolean true, or nil when the client is gone
function O:token(text)
	local p = self.parser
	local before = p.state
	local events = p:feed(text)
	if before == "reasoning" or p.state == "reasoning" then
		self.reasoning_tokens = self.reasoning_tokens + 1
	end
	return self:deliver(events)
end

--- True when the parser reads content, not reasoning or a tool call.
-- @treturn boolean
function O:in_content()
	return CONTENT[self.parser.state] == true
end

--- Ends the output: feeds the rest of the text and ends the parser.
-- @tparam[opt] string rest text that the decoder held back
-- @treturn ?boolean true, or nil when the client is gone
function O:finish(rest)
	local p = self.parser
	local events = p:feed(rest or "")
	for _, ev in ipairs(p:finish()) do
		events[#events + 1] = ev
	end
	return self:deliver(events)
end

return M
