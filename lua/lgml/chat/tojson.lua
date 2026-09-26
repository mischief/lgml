--- JSON encoder that matches Jinja tojson.
-- The output is Python `json.dumps` with no ASCII escapes. Key order:
-- metatable `__order = { "k1", ... }`; other keys follow, sorted.
-- Metatable `__jsontype = "array"` or `"object"` types an empty table;
-- an unmarked empty table is `{}`. `null` and userdata are null.
-- @module lgml.chat.tojson
local M = {}

-- The same null and array marker as lgml.json, so decoded requests
-- render directly.
local json = require("lgml.json")
--- The null value. This is `lgml.json.null`.
M.null = json.null

local escapes = {
	['"'] = '\\"',
	["\\"] = "\\\\",
	["\n"] = "\\n",
	["\r"] = "\\r",
	["\t"] = "\\t",
	["\b"] = "\\b",
	["\f"] = "\\f",
}

local function esc(c)
	return escapes[c] or string.format("\\u%04x", c:byte())
end

--- Encodes a JSON string.
-- Control bytes, quotes and backslashes are escaped. DEL and the
-- other bytes stay as they are.
-- @tparam string s the string
-- @treturn string the JSON text
function M.string(s)
	return '"' .. s:gsub('[%c"\\]', function(c)
		if c == "\127" then
			return c
		end
		return esc(c)
	end) .. '"'
end

-- Shortest digits that round-trip, as (digits, decimal point position).
local function shortest(x)
	for p = 1, 17 do
		local s = string.format("%." .. (p - 1) .. "e", x)
		if tonumber(s) == x then
			local m, e = s:match("^(%d[%.%d]*)e([-+]%d+)$")
			return m:gsub("%.", ""), tonumber(e) + 1
		end
	end
end

--- Formats a float as Python repr does.
-- It uses fixed notation when -4 < decpt <= 16. NaN and infinities
-- give `NaN`, `Infinity` and `-Infinity`.
-- @tparam number x the number
-- @treturn string the text
function M.float(x)
	if x ~= x then
		return "NaN"
	elseif x == math.huge then
		return "Infinity"
	elseif x == -math.huge then
		return "-Infinity"
	end
	local sign = (x < 0 or 1 / x < 0) and "-" or ""
	if x == 0 then
		return sign .. "0.0"
	end
	local d, pt = shortest(math.abs(x))
	d = d:gsub("0+$", "")
	if d == "" then
		d = "0"
	end
	if pt > -4 and pt <= 16 then
		if pt <= 0 then
			return sign .. "0." .. string.rep("0", -pt) .. d
		elseif pt >= #d then
			return sign .. d .. string.rep("0", pt - #d) .. ".0"
		end
		return sign .. d:sub(1, pt) .. "." .. d:sub(pt + 1)
	end
	local m = d:sub(1, 1)
	if #d > 1 then
		m = m .. "." .. d:sub(2)
	end
	local e = pt - 1
	return string.format("%s%se%s%02d", sign, m, e < 0 and "-" or "+", math.abs(e))
end

local function kind(t)
	if json.isarray(t) then
		return "array"
	end
	local mt = getmetatable(t)
	local jt = mt and mt.__jsontype
	if jt then
		return jt
	end
	local n = #t
	if n == 0 then
		return "object"
	end
	local count = 0
	for _ in pairs(t) do
		count = count + 1
	end
	return count == n and "array" or "object"
end

--- Returns the JSON kind of a table.
-- @function kind
-- @tparam table t the table
-- @treturn string `"array"`, `"object"` or the `__jsontype` of t
M.kind = kind

--- Lists the object keys: `__order` first, then the rest sorted.
-- @tparam table t the table
-- @treturn {any,...} the keys
function M.keys(t)
	local mt = getmetatable(t)
	local out, seen = {}, {}
	for _, k in ipairs(mt and mt.__order or {}) do
		if t[k] ~= nil and not seen[k] then
			seen[k] = true
			out[#out + 1] = k
		end
	end
	local rest = {}
	for k in pairs(t) do
		if not seen[k] then
			rest[#rest + 1] = k
		end
	end
	table.sort(rest, function(a, b)
		return tostring(a) < tostring(b)
	end)
	table.move(rest, 1, #rest, #out + 1, out)
	return out
end

local encode

local function encode_table(t, depth, fl)
	local k = kind(t)
	if k == "null" then
		return "null"
	end
	local parts = {}
	if k == "array" then
		for i = 1, #t do
			parts[i] = encode(t[i], depth, fl)
		end
		return "[" .. table.concat(parts, ", ") .. "]"
	end
	for i, key in ipairs(M.keys(t)) do
		parts[i] = M.string(tostring(key)) .. ": " .. encode(t[key], depth, fl)
	end
	return "{" .. table.concat(parts, ", ") .. "}"
end

function encode(v, depth, fl)
	local tv = type(v)
	if tv == "string" then
		return M.string(v)
	elseif tv == "number" then
		return math.type(v) == "integer" and tostring(v) or fl(v)
	elseif tv == "boolean" then
		return tostring(v)
	elseif tv == "nil" or tv == "userdata" then
		return "null"
	elseif tv == "table" then
		assert(depth < 200, "tojson: nesting too deep")
		return encode_table(v, depth + 1, fl)
	end
	error("tojson: cannot encode " .. tv)
end

--- Encodes a value as JSON.
-- @param v the value
-- @tparam[opt] function float formats a float; default `float`
-- @treturn string the JSON text
-- @raise for a function or a thread, or nesting deeper than 200
function M.encode(v, float)
	return encode(v, 0, float or M.float)
end

return M
