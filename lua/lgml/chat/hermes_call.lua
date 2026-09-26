--- Byte-at-a-time matcher for the body of one hermes tool call.
-- The body is `{"name": "f", "arguments": <json>}</tool_call>`.
-- `Call:step` returns event, consumed, args_text. Events: `"begin"`
-- (name known), `"fail"` (not a tool call), `"error"` (bad call after
-- begin), `"end"`. A byte that is not consumed must be given again or
-- re-read.
-- @module lgml.chat.hermes_call
local M = {}

local C = {}
C.__index = C

local WS = { [" "] = true, ["\t"] = true, ["\n"] = true, ["\r"] = true }

local HEAD = { "ws", "{", "ws", '"name"', "ws", ":", "ws", "str", "ws", ",", "ws",
	'"arguments"', "ws", ":", "ws" }

local MAX_HEAD = 512

--- Makes a matcher.
-- @tparam[opt] table names a set of allowed tool names; nil allows all
-- @treturn Call the matcher
function M.new(names)
	return setmetatable({
		phase = "head",
		hp = 1,
		lp = 1,
		raw = "",
		names = names,
		v = require("lgml.chat.jsonscan").new(),
	}, C)
end

local UNESC = { b = "\b", f = "\f", n = "\n", r = "\r", t = "\t" }

local function unescape(s)
	s = s:gsub("\\u(%x%x%x%x)", function(h)
		return utf8.char(tonumber(h, 16))
	end)
	return (s:gsub("\\(.)", function(c)
		return UNESC[c] or c
	end))
end

--- A tool call matcher.
-- @type Call

-- Name string. Returns true when the byte is consumed, nil to fail.
function C:head_str(c)
	local s = self.str
	if not s then
		if c ~= '"' then
			return nil
		end
		self.str, self.esc = "", false
		return true
	elseif self.esc then
		self.str, self.esc = s .. c, false
	elseif c == "\\" then
		self.str, self.esc = s .. c, true
	elseif c == '"' then
		self.name = unescape(s)
		self.hp = self.hp + 1
		if self.names and not self.names[self.name] then
			return nil
		end
	elseif c:byte() < 32 then
		return nil
	else
		self.str = s .. c
	end
	return true
end

-- Returns true when consumed, false to retry at the next item, nil to fail.
function C:head_item(c)
	local item = HEAD[self.hp]
	if item == "ws" then
		if WS[c] then
			return true
		end
		self.hp = self.hp + 1
		return false
	elseif item == "str" then
		return self:head_str(c)
	elseif c ~= item:sub(self.lp, self.lp) then
		return nil
	end
	self.lp = self.lp + 1
	if self.lp > #item then
		self.hp, self.lp = self.hp + 1, 1
	end
	return true
end

function C:step_head(c)
	while self.hp <= #HEAD do
		local r = self:head_item(c)
		if r == nil or #self.raw >= MAX_HEAD then
			return "fail", false
		elseif r then
			self.raw = self.raw .. c
			return nil, true
		end
	end
	self.phase = "args"
	return "begin", false
end

function C:step_args(c)
	local r = self.v:step(c)
	if r == "err" then
		return "error", false
	elseif r == "ok" then
		return nil, true, c
	elseif r == "done" then
		self.phase = "tail"
		return nil, true, c
	end
	self.phase = "tail"
	return self:step_tail(c)
end

local CLOSE = "</tool_call>"

-- ws "}" ws "</tool_call>"
function C:step_tail(c)
	local t = self.tail or 0
	if t == 0 or t == 1 then
		if WS[c] then
			return nil, true
		elseif t == 0 and c == "}" then
			self.tail = 1
			return nil, true
		elseif t == 1 and c == "<" then
			self.tail = 2
			return nil, true
		end
		return "error", false
	end
	if c ~= CLOSE:sub(t, t) then
		return "error", false
	end
	self.tail = t + 1
	return self.tail > #CLOSE and "end" or nil, true
end

--- Matches one byte.
-- @tparam string c the byte
-- @treturn ?string the event, or nil
-- @treturn boolean true when the byte is consumed
-- @treturn ?string argument text to add, or nil
function C:step(c)
	return self["step_" .. self.phase](self, c)
end

--- Ends the output. A call with its closing `}` counts as complete.
-- @treturn string `"end"`, `"fail"` or `"error"`
function C:finish()
	if self.phase == "head" then
		return "fail"
	elseif self.phase == "tail" and (self.tail or 0) >= 1 then
		return "end"
	end
	return "error"
end

return M
