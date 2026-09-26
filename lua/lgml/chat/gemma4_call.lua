--- Byte-at-a-time matcher for one Gemma-4 tool call body.
-- The body is `call:NAME{key:value,...}<tool_call|>`. Strings are
-- `<|"|>...<|"|>`; keys are bare. Arguments come out as JSON, as
-- llama.cpp `gemma4_to_json`. `Call:step` returns event, consumed,
-- args_text. Events: `"begin"`, `"fail"` (not a call), `"error"` (bad
-- call after begin), `"end"`. A byte that is not consumed must be
-- given again.
-- @module lgml.chat.gemma4_call
local M = {}

local C = {}
C.__index = C

local WS = { [" "] = true, ["\t"] = true, ["\n"] = true, ["\r"] = true }

local QUOTE = '<|"|>'
local CLOSE = "<tool_call|>"
local MAX_HEAD = 512

--- Makes a matcher.
-- @tparam[opt] table names a set of allowed tool names; nil allows all
-- @treturn Call the matcher
function M.new(names)
	return setmetatable({
		phase = "head",
		raw = "",
		name = "",
		names = names,
		stack = {},
		st = "value",
		held = "",
		lit = "",
	}, C)
end

local ESC = { ['"'] = '\\"', ["\\"] = "\\\\", ["\n"] = "\\n", ["\r"] = "\\r",
	["\t"] = "\\t", ["\b"] = "\\b", ["\f"] = "\\f" }

-- JSON string body escape for one byte.
local function esc(c)
	if ESC[c] then
		return ESC[c]
	elseif c:byte() < 0x20 then
		return string.format("\\u%04x", c:byte())
	end
	return c
end

--- A tool call matcher.
-- @type Call

-- "call:" then the name up to "{".
function C:step_head(c)
	if #self.raw >= MAX_HEAD then
		return "fail", false
	end
	local n = #self.raw
	if n < 5 then
		if c ~= ("call:"):sub(n + 1, n + 1) then
			return "fail", false
		end
	elseif c == "{" then
		if self.name == "" or (self.names and not self.names[self.name]) then
			return "fail", false
		end
		self.phase = "args"
		return "begin", false
	elseif WS[c] or c == "<" or c == "}" then
		return "fail", false
	else
		self.name = self.name .. c
	end
	self.raw = self.raw .. c
	return nil, true
end

-- JSON number, true, false or null.
local function valid_lit(s)
	if s == "true" or s == "false" or s == "null" then
		return true
	end
	local i = s:match("^%-?()")
	local int = s:match("^0", i) or s:match("^[1-9]%d*", i)
	if not int then
		return false
	end
	i = i + #int
	local frac = s:match("^%.%d+", i)
	i = i + (frac and #frac or 0)
	local exp = s:match("^[eE][-+]?%d+", i)
	i = i + (exp and #exp or 0)
	return i == #s + 1
end

-- After a value: close the container or read the next item.
function C:after_value(c)
	local top = self.stack[#self.stack]
	if not top then
		self.phase = "tail"
		return self:step_tail(c)
	elseif WS[c] then
		return nil, true
	elseif c == "," then
		self.st = top == "{" and "key_start" or "value"
		return nil, true, ","
	elseif (top == "{" and c == "}") or (top == "[" and c == "]") then
		self.stack[#self.stack] = nil
		return nil, true, c
	end
	return "error", false
end

function C:start_value(c)
	if WS[c] then
		return nil, true
	elseif c == "{" or c == "[" then
		self.stack[#self.stack + 1] = c
		self.st = c == "{" and "map_open" or "list_open"
		return nil, true, c
	elseif c == "<" then
		self.st, self.held = "open", "<"
		return nil, true
	elseif c:match("[-%dtfn]") then
		self.st, self.lit = "lit", c
		return nil, true
	end
	return "error", false
end

-- The <|"|> that opens a string.
function C:step_open(c)
	local held = self.held .. c
	if held ~= QUOTE:sub(1, #held) then
		return "error", false
	elseif #held < #QUOTE then
		self.held = held
		return nil, true
	end
	self.st, self.held = "str", ""
	return nil, true, '"'
end

-- String body up to <|"|>. Bytes that may start the close wait.
function C:step_str(c)
	local held = self.held .. c
	if held == QUOTE:sub(1, #held) then
		if #held < #QUOTE then
			self.held = held
			return nil, true
		end
		self.held, self.st = "", "after"
		return nil, true, '"'
	end
	self.held = ""
	if #held > 1 then
		-- Not the close: flush all but c, then look at c again.
		return nil, false, (held:sub(1, -2):gsub(".", esc))
	end
	return nil, true, esc(c)
end

function C:step_lit(c)
	if c:match("[%w%.%+%-]") then
		if #self.lit >= 64 then
			return "error", false
		end
		self.lit = self.lit .. c
		return nil, true
	end
	if not valid_lit(self.lit) then
		return "error", false
	end
	local s = self.lit
	self.st, self.lit = "after", ""
	return nil, false, s
end

function C:step_args(c)
	local st = self.st
	if st == "value" then
		return self:start_value(c)
	elseif st == "map_open" then
		if WS[c] then
			return nil, true
		elseif c == "}" then
			self.stack[#self.stack] = nil
			self.st = "after"
			return nil, true, "}"
		end
		self.st = "key_start"
		return self:step_args(c)
	elseif st == "key_start" then
		if WS[c] then
			return nil, true
		elseif c == ":" or c == "}" then
			return "error", false
		end
		self.st = "key"
		return nil, true, '"' .. esc(c)
	elseif st == "key" then
		if c == ":" then
			self.st = "value"
			return nil, true, '":'
		elseif c == "}" then
			return "error", false
		end
		return nil, true, esc(c)
	elseif st == "list_open" then
		if WS[c] then
			return nil, true
		elseif c == "]" then
			self.stack[#self.stack] = nil
			self.st = "after"
			return nil, true, "]"
		end
		self.st = "value"
		return self:start_value(c)
	elseif st == "open" then
		return self:step_open(c)
	elseif st == "str" then
		return self:step_str(c)
	elseif st == "lit" then
		return self:step_lit(c)
	end
	return self:after_value(c)
end

-- ws "<tool_call|>"
function C:step_tail(c)
	local t = self.tail or 0
	if t == 0 and WS[c] then
		return nil, true
	elseif c ~= CLOSE:sub(t + 1, t + 1) then
		return "error", false
	end
	self.tail = t + 1
	return self.tail == #CLOSE and "end" or nil, true
end

--- Matches one byte.
-- @tparam string c the byte
-- @treturn ?string the event, or nil
-- @treturn boolean true when the byte is consumed
-- @treturn ?string argument text to add, or nil
function C:step(c)
	if self.phase == "head" then
		return self:step_head(c)
	elseif self.phase == "tail" then
		return self:step_tail(c)
	end
	return self:step_args(c)
end

--- Ends the output. A call whose arguments closed counts as complete.
-- @treturn string `"end"`, `"fail"` or `"error"`
function C:finish()
	if self.phase == "head" then
		return "fail"
	elseif self.phase == "tail" or (#self.stack == 0 and self.st == "after") then
		return "end"
	end
	return "error"
end

return M
