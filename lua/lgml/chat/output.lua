--- Sends the output of a chat parser to a sink.
-- Chat engines feed the text of each token to an `Output`. It runs the
-- parser of the chat family (see `lgml.chat`), sends the events to a
-- `lgml.server.engine.Sink` and maps the tool call indexes of the
-- parser to those of the sink. It counts the reasoning tokens.
--
-- Stop strings apply to the content only. The output holds back
-- content that can be the start of a stop string. When the content
-- holds a stop string, the output sends the text before it and ends
-- the reply; the stop string does not go out.
--
-- A token can carry a logprobs entry (see `lgml.server.engine.Sink`).
-- The output keeps the entries of content tokens, and sends them with
-- the next content text that goes out.
-- @module lgml.chat.output
local M = {}

local O = {}
O.__index = O

-- Parser states that give content.
local CONTENT = { start = true, content = true }

--- Makes an output.
-- @tparam lgml.server.engine.Sink sink the reply
-- @tparam ?table parser a parser from the `new_parser` of a chat
-- family; nil when the caller sends events with `deliver`
-- @tparam[opt] table opts the options:
--
-- - `stop`: a list of stop strings; empty strings are skipped
-- @treturn Output the output
function M.new(sink, parser, opts)
	opts = opts or {}
	local stops = {}
	for _, st in ipairs(type(opts.stop) == "table" and opts.stop or {}) do
		if type(st) == "string" and st ~= "" then
			stops[#stops + 1] = st
		end
	end
	return setmetatable({
		sink = sink, parser = parser, map = {},
		reasoning_tokens = 0, stops = stops, hold = "", lp = {},
	}, O)
end

--- An output of one chat reply.
-- The field `reasoning_tokens` counts the tokens that the parser read
-- in a reasoning block, and the tags of the block.
-- @type Output

-- The length of the longest end of s that starts a stop string.
local function partial(s, stops)
	local k = 0
	for _, st in ipairs(stops) do
		for n = math.min(#s, #st - 1), k + 1, -1 do
			if s:sub(-n) == st:sub(1, n) then
				k = n
				break
			end
		end
	end
	return k
end

-- Sends content text with the kept logprobs entries. Empty text
-- keeps them, unless all is true.
function O:send(text, all)
	local lp = self.lp
	if text == "" and not (all and #lp > 0) then
		return true
	end
	self.lp = {}
	return self.sink:content(text, #lp > 0 and lp or nil)
end

-- Sends content through the stop strings. Returns true, "stop" or nil.
function O:content(text)
	if #self.stops == 0 then
		return self:send(text)
	end
	local s = self.hold .. text
	local cut
	for _, st in ipairs(self.stops) do
		local i = s:find(st, 1, true)
		if i and (not cut or i < cut) then
			cut = i
		end
	end
	if cut then
		self.hold = ""
		return self:send(s:sub(1, cut - 1), true) and "stop" or nil
	end
	local k = partial(s, self.stops)
	self.hold = s:sub(#s - k + 1)
	return self:send(s:sub(1, #s - k))
end

-- Sends the held content. Returns true, or nil when the client is gone.
function O:flush(all)
	local h = self.hold
	self.hold = ""
	return self:send(h, all)
end

-- Sends one event. Returns true, "stop", or nil when the client is gone.
function O:event(ev)
	local sink, t = self.sink, ev.type
	if t == "content" then
		return self:content(ev.text)
	elseif self.hold ~= "" and not self:flush() then
		return nil
	end
	if t == "reasoning" then
		return sink:reasoning(ev.text)
	elseif t == "tool_call_begin" then
		local idx = sink:tool_call(ev.name)
		self.map[ev.index] = idx
		return idx ~= nil or nil
	elseif t == "tool_call_args" then
		return sink:tool_arguments(self.map[ev.index], ev.text)
	end
	return true
end

--- Sends parser events.
-- @tparam {table,...} events the events
-- @treturn ?boolean|string true; `"stop"` when a stop string ends the
-- reply; nil when the client is gone. After `"stop"`, end the reply
-- with reason `stop` and do not call the output again.
function O:deliver(events)
	for _, ev in ipairs(events) do
		local r = self:event(ev)
		if r ~= true then
			return r
		end
	end
	return true
end

--- Feeds the text of one token.
-- A token counts as content when the parser reads content before and
-- after it; then lp, when given, goes out with the content.
-- @tparam string text the text
-- @tparam[opt] table lp the logprobs entry of the token
-- @treturn ?boolean|string true, `"stop"` or nil, as `deliver`
function O:token(text, lp)
	local p = self.parser
	local before = p.state
	local events = p:feed(text)
	if before == "reasoning" or p.state == "reasoning" then
		self.reasoning_tokens = self.reasoning_tokens + 1
	end
	if lp and CONTENT[before] and CONTENT[p.state] then
		self.lp[#self.lp + 1] = lp
	end
	return self:deliver(events)
end

--- Keeps the logprobs entry of a content token that the caller sends
-- with `deliver`.
-- @tparam table lp the entry
function O:logprob(lp)
	self.lp[#self.lp + 1] = lp
end

--- True when the parser reads content, not reasoning or a tool call.
-- @treturn boolean
function O:in_content()
	return CONTENT[self.parser.state] == true
end

--- Ends the output: feeds the rest of the text, ends the parser and
-- sends the held content.
-- @tparam[opt] string rest text that the decoder held back
-- @treturn ?boolean|string true, `"stop"` or nil, as `deliver`
function O:finish(rest)
	local p = self.parser
	local events = {}
	if p then
		events = p:feed(rest or "")
		for _, ev in ipairs(p:finish()) do
			events[#events + 1] = ev
		end
	end
	local r = self:deliver(events)
	if r == true and (self.hold ~= "" or #self.lp > 0) then
		return self:flush(true)
	end
	return r
end

return M
