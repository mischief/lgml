--- JSON on lua-cjson, with one private cjson instance.
-- `json.null` is `cjson.null`. Decoded arrays carry the array metatable,
-- so an empty array stays an array. A plain empty table encodes as `{}`.
-- The encoder writes numbers with 16 significant digits and does not
-- escape `/`. Encode and decode stop at a nesting depth of 200.
-- @module lgml.json
local cjson = require("cjson")

local M = {}

local c = cjson.new()
c.decode_array_with_array_mt(true)
c.encode_escape_forward_slash(false)
c.encode_number_precision(16)
c.encode_max_depth(200)
c.decode_max_depth(200)

--- The JSON null sentinel (`cjson.null`).
M.null = c.null
--- Maximum nesting depth for encode and decode (200).
M.max_depth = 200

local array_mt = cjson.array_mt

--- Mark a table as a JSON array.
-- @tparam[opt] table t the table to mark; a new table when nil
-- @treturn table t, with the array metatable
function M.array(t)
	return setmetatable(t or {}, array_mt)
end

--- Mark a table as a JSON object.
-- The function removes any metatable from t.
-- @tparam[opt] table t the table to mark; a new table when nil
-- @treturn table t, with no metatable
function M.object(t)
	return setmetatable(t or {}, nil)
end

--- Test for the JSON array mark.
-- @tparam any t the value to test
-- @treturn boolean true when t carries the array metatable
function M.isarray(t)
	return getmetatable(t) == array_mt
end

-- cjson gives every number as a float. Whole numbers become integers,
-- so 1 and 1.0 both decode as 1, as in llama.cpp.
local function ints(t)
	for k, v in pairs(t) do
		if type(v) == "number" then
			t[k] = math.tointeger(v) or v
		elseif type(v) == "table" then
			ints(v)
		end
	end
	return t
end

--- Decode one JSON document.
-- Whole numbers decode as integers, so 1 and 1.0 both give 1.
-- @tparam string s the JSON text
-- @treturn any the decoded value
-- @return[2] nil on a parse error
-- @treturn[2] string the error message
function M.decode(s)
	local ok, v = pcall(c.decode, s)
	if not ok then
		return nil, tostring(v)
	end
	if type(v) == "number" then
		return math.tointeger(v) or v
	elseif type(v) == "table" then
		return ints(v)
	end
	return v
end

--- Encode a value as JSON text.
-- @tparam any v the value to encode
-- @treturn string the JSON text
-- @return[2] nil when v cannot be encoded
-- @treturn[2] string the error message
function M.encode(v)
	local ok, s = pcall(c.encode, v)
	if not ok then
		return nil, tostring(s)
	end
	return s
end

--- Encode a string as a JSON string literal, with quotes.
-- @tparam string s the string
-- @treturn string the quoted, escaped literal
-- @raise when s is not a string or number
function M.encode_string(s)
	return c.encode(s)
end

return M
