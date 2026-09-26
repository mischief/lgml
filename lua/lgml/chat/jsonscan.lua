--- Strict byte-at-a-time JSON validator for one value.
-- `Scanner:step` returns `"ok"` (consumed), `"done"` (consumed, value
-- complete), `"done_back"` (value complete, byte not consumed) or
-- `"err"`.
-- @module lgml.chat.jsonscan
local M = {}

local S = {}
S.__index = S

local WS = { [" "] = true, ["\t"] = true, ["\n"] = true, ["\r"] = true }
local ESC = { ['"'] = true, ["\\"] = true, ["/"] = true, b = true, f = true,
	n = true, r = true, t = true }
local LIT = { t = "true", f = "false", n = "null" }

--- Makes a validator.
-- @treturn Scanner the validator
function M.new()
	return setmetatable({ st = "value", stack = {} }, S)
end

--- A JSON validator.
-- @type Scanner

function S:value_done()
	if #self.stack == 0 then
		self.st = "end"
		return "done"
	end
	self.st = "after"
	return "ok"
end

function S:open(kind, st)
	self.stack[#self.stack + 1] = kind
	self.st = st
	return "ok"
end

function S:close()
	self.stack[#self.stack] = nil
	return self:value_done()
end

function S:start_value(c)
	if c == "{" then
		return self:open("o", "obj_first")
	elseif c == "[" then
		return self:open("a", "arr_first")
	elseif c == '"' then
		self.st, self.key = "str", false
		return "ok"
	elseif c == "-" then
		self.st = "n_minus"
		return "ok"
	elseif c == "0" then
		self.st = "n_zero"
		return "ok"
	elseif c:find("^%d") then
		self.st = "n_int"
		return "ok"
	elseif LIT[c] then
		self.st, self.lit, self.lp = "lit", LIT[c], 2
		return "ok"
	end
	return "err"
end

S["s_value"] = function(self, c)
	if WS[c] then
		return "ok"
	end
	return self:start_value(c)
end

S["s_arr_first"] = function(self, c)
	if c == "]" then
		return self:close()
	end
	return S.s_value(self, c)
end

S["s_obj_first"] = function(self, c)
	if c == "}" then
		return self:close()
	end
	return S.s_key(self, c)
end

S["s_key"] = function(self, c)
	if WS[c] then
		return "ok"
	elseif c == '"' then
		self.st, self.key = "str", true
		return "ok"
	end
	return "err"
end

S["s_colon"] = function(self, c)
	if WS[c] then
		return "ok"
	elseif c == ":" then
		self.st = "value"
		return "ok"
	end
	return "err"
end

S["s_after"] = function(self, c)
	local top = self.stack[#self.stack]
	if WS[c] then
		return "ok"
	elseif c == "," then
		self.st = top == "o" and "key" or "value"
		return "ok"
	elseif (c == "}" and top == "o") or (c == "]" and top == "a") then
		return self:close()
	end
	return "err"
end

S["s_str"] = function(self, c)
	if c == '"' then
		if self.key then
			self.st = "colon"
			return "ok"
		end
		return self:value_done()
	elseif c == "\\" then
		self.st = "esc"
	elseif c:byte() < 32 then
		return "err"
	end
	return "ok"
end

S["s_esc"] = function(self, c)
	if ESC[c] then
		self.st = "str"
		return "ok"
	elseif c == "u" then
		self.st, self.nhex = "hex", 0
		return "ok"
	end
	return "err"
end

S["s_hex"] = function(self, c)
	if not c:find("^%x") then
		return "err"
	end
	self.nhex = self.nhex + 1
	if self.nhex == 4 then
		self.st = "str"
	end
	return "ok"
end

S["s_lit"] = function(self, c)
	if c ~= self.lit:sub(self.lp, self.lp) then
		return "err"
	end
	self.lp = self.lp + 1
	if self.lp > #self.lit then
		return self:value_done()
	end
	return "ok"
end

S["s_end"] = function()
	return "err"
end

-- Number states: next state per byte class, or nil to end the number.
local NUM = {
	n_minus = { zero = "n_zero", digit = "n_int" },
	n_zero = { dot = "n_dot", e = "n_e" },
	n_int = { zero = "n_int", digit = "n_int", dot = "n_dot", e = "n_e" },
	n_dot = { zero = "n_frac", digit = "n_frac" },
	n_frac = { zero = "n_frac", digit = "n_frac", e = "n_e" },
	n_e = { zero = "n_exp", digit = "n_exp", sign = "n_esign" },
	n_esign = { zero = "n_exp", digit = "n_exp" },
	n_exp = { zero = "n_exp", digit = "n_exp" },
}
local NUM_END = { n_zero = true, n_int = true, n_frac = true, n_exp = true }

local function class(c)
	if c == "0" then
		return "zero"
	elseif c:find("^%d") then
		return "digit"
	elseif c == "." then
		return "dot"
	elseif c == "e" or c == "E" then
		return "e"
	elseif c == "+" or c == "-" then
		return "sign"
	end
end

function S:number(c)
	local nxt = NUM[self.st][class(c)]
	if nxt then
		self.st = nxt
		return "ok"
	elseif not NUM_END[self.st] then
		return "err"
	end
	if self:value_done() == "done" then
		return "done_back"
	end
	return S.s_after(self, c)
end

--- Checks one byte.
-- @tparam string c the byte
-- @treturn string `"ok"`, `"done"`, `"done_back"` or `"err"`
function S:step(c)
	if NUM[self.st] then
		return self:number(c)
	end
	return S["s_" .. self.st](self, c)
end

return M
