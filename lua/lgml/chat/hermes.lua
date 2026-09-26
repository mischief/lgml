--- Streaming parser for `<think>` blocks and hermes-style tool calls.
-- A call is `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.
-- `Parser:feed` and `Parser:finish` return lists of events:
--
-- - `{ type = "reasoning" or "content", text }`
-- - `{ type = "tool_call_begin", index, name }`
-- - `{ type = "tool_call_args", index, text }`
-- - `{ type = "tool_call_end", index, error }`
--
-- index is 0-based. error is nil for a good call.
-- @module lgml.chat.hermes
local M = {}

local P = {}
P.__index = P
P.call_tag = "<tool_call>"

--- The parser class, for formats that differ only in their tags.
-- A subclass sets `call_tag` and overrides `step_<state>` methods.
M.P = P

-- Bytes at the end of s that start an incomplete UTF-8 sequence.
local function utf8_tail(s)
	for k = 1, math.min(3, #s) do
		local b = s:byte(#s - k + 1)
		if b < 0x80 then
			return 0
		elseif b >= 0xC0 then
			local need = b >= 0xF0 and 4 or b >= 0xE0 and 3 or 2
			return need > k and k or 0
		end
	end
	return 0
end

-- Length of the longest suffix of s that is a proper prefix of tag.
local function partial(s, tag)
	for k = math.min(#s, #tag - 1), 1, -1 do
		if s:sub(-k) == tag:sub(1, k) then
			return k
		end
	end
	return 0
end

--- Makes a set of tool names.
-- @tparam[opt] {string|table,...} tools tool names or OpenAI tools
-- @treturn ?table a set of names, or nil when tools is nil
function M.name_set(tools)
	if not tools then
		return nil
	end
	local set = {}
	for _, t in ipairs(tools) do
		local f = type(t) == "table" and (t["function"] or t) or nil
		set[f and f.name or t] = true
	end
	return set
end

--- Makes a parser.
-- @tparam[opt] table opts the options:
--
-- - `in_think`: the prompt already opened `<think>`
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
		names = M.name_set(opts.tools),
	}, P)
end

--- A streaming parser. Its state names a `step_<state>` method.
-- Each step method takes final and returns true to run again. The
-- states are `start`, `content`, `reasoning` and `call`.
-- @type Parser

function P:push(ev)
	local out, last = self.out, self.out[#self.out]
	if ev.text and last and last.type == ev.type and last.index == ev.index then
		last.text = last.text .. ev.text
	else
		out[#out + 1] = ev
	end
end

--- Adds text output.
-- Leading space after a tag is dropped; trailing space and partial
-- UTF-8 wait for more text.
-- @tparam string kind `"content"` or `"reasoning"`
-- @tparam string s the text
function P:emit(kind, s)
	if self.lead then
		s = s:gsub("^%s+", "")
		if s == "" then
			return
		end
		self.lead = false
	end
	s = self.hold .. s
	local body, ws = s:match("^(.-)(%s*)$")
	local k = utf8_tail(body)
	self.hold, self.hold_kind = body:sub(#body - k + 1) .. ws, kind
	body = body:sub(1, #body - k)
	if body ~= "" then
		self:push({ type = kind, text = body })
	end
end

--- Flushes the held text except trailing space.
function P:drop_hold()
	local body = self.hold:match("^(.-)%s*$")
	if body ~= "" then
		self:push({ type = self.hold_kind, text = body })
	end
	self.hold = ""
end

--- Flushes the held text and goes to state.
-- Leading space in the new state is dropped.
-- @tparam string state the new state
function P:switch(state)
	self:drop_hold()
	self.state, self.lead = state, true
end

--- Emits buffered text in kind up to tag.
-- The tag is consumed. When no tag is found, a partial tag at the end
-- waits for more text, unless final is true.
-- @tparam string kind `"content"` or `"reasoning"`
-- @tparam string tag the tag to find
-- @tparam boolean final true at the end of output
-- @treturn boolean true when tag was found
function P:scan(kind, tag, final)
	local buf = self.buf
	local i = buf:find(tag, 1, true)
	if i then
		self:emit(kind, buf:sub(1, i - 1))
		self.buf = buf:sub(i + #tag)
		return true
	end
	local k = final and 0 or partial(buf, tag)
	self:emit(kind, buf:sub(1, #buf - k))
	self.buf = buf:sub(#buf - k + 1)
	return false
end

function P:step_start(final)
	self.buf = self.buf:gsub("^%s+", "")
	local buf = self.buf
	if buf:sub(1, 7) == "<think>" then
		self.buf = buf:sub(8)
		self:switch("reasoning")
		return true
	elseif buf == "" or (not final and ("<think>"):sub(1, #buf) == buf) then
		return false
	end
	self.state = "content"
	return true
end

function P:step_content(final)
	if not self:scan("content", "<tool_call>", final) then
		return false
	end
	self.pre_ws, self.hold = self.hold, ""
	self.state = "call"
	self.call = require("lgml.chat.hermes_call").new(self.names)
	return true
end

function P:step_reasoning(final)
	if not self:scan("reasoning", "</think>", final) then
		return false
	end
	self:switch("content")
	return true
end

function P:args(text)
	self.args_buf = (self.args_buf or "") .. text
end

function P:flush_args(all)
	local s = self.args_buf or ""
	local k = all and 0 or utf8_tail(s)
	if #s > k then
		self:push({ type = "tool_call_args", index = self.ncalls - 1, text = s:sub(1, #s - k) })
	end
	self.args_buf = s:sub(#s - k + 1)
end

-- The call text is not a tool call: emit it as content.
function P:fallback(raw)
	self.state, self.hold, self.lead = "content", self.pre_ws, false
	self:emit("content", self.call_tag)
	self.buf = raw .. self.buf
end

function P:end_call(err)
	self:flush_args(true)
	self:push({ type = "tool_call_end", index = self.ncalls - 1, error = err })
	self.call = nil
	self:switch("content")
end

function P:on_call_event(ev, rest)
	if ev == "begin" then
		self.ncalls = self.ncalls + 1
		self:push({ type = "tool_call_begin", index = self.ncalls - 1, name = self.call.name })
	elseif ev == "fail" then
		self.buf = rest
		self:fallback(self.call.raw)
		self.call = nil
		return true
	elseif ev == "error" then
		self.buf = rest
		self:end_call("invalid tool call")
		self.lead = false
		return true
	elseif ev == "end" then
		self.buf = rest
		self:end_call(nil)
		return true
	end
	return false
end

function P:step_call(final)
	local call, buf = self.call, self.buf
	local i = 1
	while i <= #buf do
		local ev, consumed, args = call:step(buf:sub(i, i))
		if args then
			self:args(args)
		end
		if consumed then
			i = i + 1
		end
		if ev and self:on_call_event(ev, buf:sub(i)) then
			return true
		end
	end
	self.buf = ""
	self:flush_args(false)
	if final then
		self:on_call_event(call:finish(), "")
		return true
	end
	return false
end

function P:run(final)
	while true do
		local fn = self["step_" .. self.state]
		if not fn(self, final) then
			break
		end
	end
end

--- Parses more output text.
-- @tparam string s the text
-- @treturn {table,...} the events
function P:feed(s)
	self.out = {}
	self.buf = self.buf .. s
	self:run(false)
	return self.out
end

--- Ends the output. An open tool call ends or fails.
-- @treturn {table,...} the last events
function P:finish()
	self.out = {}
	self:run(true)
	self:drop_hold()
	return self.out
end

return M
