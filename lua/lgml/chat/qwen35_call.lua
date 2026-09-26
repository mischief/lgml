--- Byte-at-a-time matcher for one Qwen3.5 XML tool call body.
-- The body follows `<tool_call>\n` and is
-- `<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>`.
-- Arguments come out as compact JSON, as llama.cpp builds them: a
-- string parameter is the raw text, other types are JSON text.
-- `Call:step` returns event, consumed, args_text. Events: `"begin"`,
-- `"fail"` (not a call), `"error"` (bad call after begin), `"end"`.
-- @module lgml.chat.qwen35_call
local tojson = require("lgml.chat.tojson")
local jsonscan = require("lgml.chat.jsonscan")

local M = {}

local C = {}
C.__index = C

-- C isspace.
local WS = { [" "] = true, ["\t"] = true, ["\n"] = true, ["\r"] = true, ["\v"] = true, ["\f"] = true }

local OPEN = "<function="
local PARAM = "<parameter="
local FCLOSE = "</function>\n"
local VCLOSE = "\n</parameter>\n"
local CLOSE = "</tool_call>"
local MAX_NAME = 512

-- True when a schema lets the value be a string, as llama.cpp
-- common_schema_info::resolves_to_string. root resolves "#/..." refs.
local function is_string(s, root, seen)
	if type(s) ~= "table" then
		return false
	end
	local ref = s["$ref"]
	if ref ~= nil then
		if type(ref) ~= "string" or seen[ref] or ref:sub(1, 2) ~= "#/" then
			return false
		end
		seen[ref] = true
		local node = root
		for part in ref:sub(3):gmatch("[^/]+") do
			local key = part:gsub("~1", "/"):gsub("~0", "~")
			if type(node) == "table" and node[key] == nil and tonumber(key) then
				key = tonumber(key) + 1
			end
			node = type(node) == "table" and node[key] or nil
		end
		return is_string(node, root, seen)
	end
	local t = s.type
	if t == "string" then
		return true
	elseif type(t) == "table" then
		for _, v in ipairs(t) do
			if v == "string" then
				return true
			end
		end
	end
	for _, key in ipairs({ "oneOf", "anyOf" }) do
		if type(s[key]) == "table" then
			for _, alt in ipairs(s[key]) do
				if is_string(alt, root, seen) then
					return true
				end
			end
		end
	end
	if type(s.allOf) == "table" and #s.allOf > 0 then
		local all = true
		for _, alt in ipairs(s.allOf) do
			all = all and is_string(alt, root, seen)
		end
		if all then
			return true
		end
	end
	if type(s.const) == "string" then
		return true
	end
	if type(s.enum) == "table" then
		for _, v in ipairs(s.enum) do
			if type(v) == "string" then
				return true
			end
		end
	end
	return false
end

--- Makes the tool table for `new`.
-- A tool given as a name has no schema: any parameter is allowed, and
-- a value that is valid JSON is kept as JSON, else it is a string.
-- @tparam[opt] {string|table,...} tools tool names or OpenAI tools
-- @treturn ?table tools by name, or nil when tools is nil
function M.tools(tools)
	if type(tools) ~= "table" then
		return nil
	end
	local out = {}
	for _, t in ipairs(tools) do
		local f = type(t) == "table" and (t["function"] or t) or t
		if type(f) == "string" then
			out[f] = { any = true }
		elseif type(f) == "table" and type(f.name) == "string" then
			local params = type(f.parameters) == "table" and f.parameters or {}
			local props = type(params.properties) == "table" and params.properties or {}
			local tool = { kinds = {}, required = {} }
			for k, s in pairs(props) do
				if type(k) == "string" then
					tool.kinds[k] = is_string(s, params, {}) and "string" or "json"
				end
			end
			if type(params.required) == "table" then
				for _, k in ipairs(params.required) do
					if tool.kinds[k] then
						tool.required[#tool.required + 1] = k
					end
				end
			end
			out[f.name] = tool
		end
	end
	return out
end

--- Makes a matcher.
-- @tparam[opt] table tools a table from `tools`; nil allows any call
-- @treturn Call the matcher
function M.new(tools)
	return setmetatable({
		phase = "head",
		raw = "",
		name = "",
		tools = tools,
		st = "sep",
		held = "",
		seen = {},
		nargs = 0,
	}, C)
end

local ESC = { ['"'] = '\\"', ["\\"] = "\\\\", ["\n"] = "\\n", ["\r"] = "\\r",
	["\t"] = "\\t", ["\b"] = "\\b", ["\f"] = "\\f" }

-- JSON string body escape.
local function esc(s)
	return (s:gsub('[%c"\\]', function(c)
		if c == "\127" then
			return c
		end
		return ESC[c] or string.format("\\u%04x", c:byte())
	end))
end

-- True when s is one JSON value with optional space around it.
local function is_json(s)
	local v = jsonscan.new()
	s = s:gsub("^[ \t\n\r]+", "") .. " "
	for i = 1, #s do
		local r = v:step(s:sub(i, i))
		if r == "err" then
			return false
		elseif r == "done" or r == "done_back" then
			return s:find("^[ \t\n\r]*$", r == "done" and i + 1 or i) ~= nil
		end
	end
	return false
end

--- A tool call matcher.
-- @type Call

-- Space, then "<function=NAME>\n".
function C:step_head(c)
	if #self.raw >= MAX_NAME then
		return "fail", false
	elseif self.gt then
		if c ~= "\n" then
			return "fail", false
		end
		self.raw = self.raw .. c
		self.phase = "args"
		return "begin", true, "{"
	elseif self.opened then
		if c == ">" then
			local tool = self.tools and self.tools[self.name]
			if self.name == "" or (self.tools and not tool) then
				return "fail", false
			end
			self.tool = tool or { any = true }
			self.gt = true
		elseif c == "\n" or c == "<" then
			return "fail", false
		else
			self.name = self.name .. c
		end
	elseif not (WS[c] and self.held == "") then
		local k = #self.held + 1
		if c ~= OPEN:sub(k, k) then
			return "fail", false
		end
		self.held = self.held .. c
		if #self.held == #OPEN then
			self.held, self.opened = "", true
		end
	end
	self.raw = self.raw .. c
	return nil, true
end

-- A tag between parameters: "<parameter=" or "</function>\n".
function C:step_tag(c)
	local held = self.held .. c
	if held == PARAM then
		self.held, self.st, self.key = "", "key", ""
	elseif held == FCLOSE then
		self.held, self.phase = "", "tail"
		return nil, true, "}"
	elseif PARAM:sub(1, #held) == held or FCLOSE:sub(1, #held) == held then
		self.held = held
	else
		return "error", false
	end
	return nil, true
end

-- The parameter name up to ">", then "\n".
function C:step_key(c)
	if self.st == "key_nl" then
		if c ~= "\n" then
			return "error", false
		end
		local kind = self.tool.any and "any" or self.tool.kinds[self.key]
		self.seen[self.key] = true
		self.st, self.held = kind, ""
		if kind == "json" then
			self.scan = jsonscan.new()
		end
		local head = (self.nargs > 0 and "," or "") .. tojson.string(self.key) .. ":"
		self.nargs = self.nargs + 1
		return nil, true, kind == "string" and head .. '"' or head
	elseif c == ">" then
		if not self.tool.any and not self.tool.kinds[self.key] then
			return "error", false
		end
		self.st = "key_nl"
	elseif c == "\n" or c == "<" or #self.key >= MAX_NAME then
		return "error", false
	else
		self.key = self.key .. c
	end
	return nil, true
end

-- Text up to "\n</parameter>\n". A byte that may start the close
-- waits. Returns the finished flag and the text to add.
function C:value_text(c)
	local held = self.held .. c
	if held == VCLOSE then
		self.held = ""
		return true, nil, true
	elseif held == VCLOSE:sub(1, #held) then
		self.held = held
		return false, nil, true
	end
	self.held = ""
	if #held > 1 then
		-- Not the close: give all but c, then look at c again.
		return false, held:sub(1, -2), false
	end
	return false, c, true
end

function C:step_string(c)
	local done, s, consumed = self:value_text(c)
	if done then
		self.st = "sep"
		return nil, true, '"'
	end
	return nil, consumed, s and esc(s)
end

function C:step_any(c)
	local done, s, consumed = self:value_text(c)
	if done then
		local v = self.value or ""
		self.st, self.value = "sep", nil
		return nil, true, is_json(v) and v or tojson.string(v)
	end
	self.value = (self.value or "") .. (s or "")
	return nil, consumed
end

-- A JSON value, then "\n</parameter>\n" at once.
function C:step_json(c)
	local r = self.scan:step(c)
	if r == "err" then
		return "error", false
	elseif r == "ok" then
		if not self.started and WS[c] then
			return nil, true
		end
		self.started = true
		return nil, true, c
	end
	self.st, self.started, self.held = "json_close", false, ""
	if r == "done" then
		return nil, true, c
	end
	return self:step_json_close(c)
end

function C:step_json_close(c)
	local k = #self.held + 1
	if c ~= VCLOSE:sub(k, k) then
		return "error", false
	end
	self.held = self.held .. c
	if #self.held == #VCLOSE then
		self.held, self.st = "", "sep"
	end
	return nil, true
end

function C:step_args(c)
	local st = self.st
	if st == "sep" then
		if self.held == "" and WS[c] then
			return nil, true
		end
		return self:step_tag(c)
	elseif st == "key" or st == "key_nl" then
		return self:step_key(c)
	elseif st == "string" then
		return self:step_string(c)
	elseif st == "any" then
		return self:step_any(c)
	elseif st == "json" then
		return self:step_json(c)
	end
	return self:step_json_close(c)
end

function C:missing()
	for _, k in ipairs(self.tool.required or {}) do
		if not self.seen[k] then
			return true
		end
	end
	return false
end

-- Space, then "</tool_call>".
function C:step_tail(c)
	local t = self.tail or 0
	if t == 0 and WS[c] then
		return nil, true
	elseif c ~= CLOSE:sub(t + 1, t + 1) then
		return "error", false
	end
	self.tail = t + 1
	if self.tail < #CLOSE then
		return nil, true
	end
	return self:missing() and "error" or "end", true
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

--- Ends the output. A call after `</function>` counts as complete.
-- @treturn string `"end"`, `"fail"` or `"error"`
function C:finish()
	if self.phase == "head" then
		return "fail"
	elseif self.phase == "tail" and not self:missing() then
		return "end"
	end
	return "error"
end

return M
