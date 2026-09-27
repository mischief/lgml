--- Checks and replies for OpenAI embedding requests.
-- `FIELDS` lists each field that an embedding request can have, with
-- its check and its compatibility status, as
-- `lgml.server.chat_request.FIELDS` does for chat. A field that is not
-- in the list gives an error, as the OpenAI API gives. `check` makes
-- the request for `Engine:embed`; `reply` makes the response body from
-- its result.
-- @module lgml.server.embeddings
local json = require("lgml.json")
local base64 = require("lgml.base64")

local M = {}

--- The most inputs of one request: 2048, as OpenAI.
M.MAX_INPUTS = 2048

local function is_ids(v)
	if not json.isarray(v) or #v == 0 then
		return false
	end
	for _, x in ipairs(v) do
		local i = math.tointeger(x)
		if not i or i < 0 then
			return false
		end
	end
	return true
end

-- The inputs of an input value as a list, or nil and a message. The
-- value is a string, an array of strings, an array of token ids, or an
-- array of arrays of token ids.
local function inputs(v)
	if type(v) == "string" then
		if v == "" then
			return nil, "input must not be an empty string"
		end
		return { v }
	elseif not json.isarray(v) or #v == 0 then
		return nil, "input must be a string or a non-empty array"
	elseif #v > M.MAX_INPUTS then
		return nil, ("input has more than %d entries"):format(M.MAX_INPUTS)
	elseif type(v[1]) == "number" then
		if not is_ids(v) then
			return nil, "input: token ids must be integers from 0"
		end
		return { v }
	end
	local list = {}
	for i, x in ipairs(v) do
		if type(x) == "string" then
			if x == "" then
				return nil, ("input[%d] must not be an empty string"):format(i - 1)
			end
		elseif not is_ids(x) then
			return nil, ("input[%d] must be a non-empty string or a non-empty array of token ids")
			    :format(i - 1)
		end
		list[i] = x
	end
	return list
end

local function int_from(lo)
	return function(v)
		local i = math.tointeger(v)
		return i ~= nil and i >= lo
	end
end

local function one_of(a, b)
	return function(v)
		return v == a or v == b
	end
end

--- The fields of an embedding request, in the order of the checks.
-- Each entry is `{ name, types, check, text, status = s, label = l }`,
-- as in `lgml.server.chat_request.FIELDS`. Read only.
-- @table FIELDS
M.FIELDS = {
	{ "input", "string|table", function(v) return inputs(v) ~= nil end,
	    "a non-empty string, an array of 1 to 2048 of them, or token ids", status = "supported",
	    label = "OpenAI" },
	{ "model", "string", status = "supported", label = "OpenAI" },
	{ "encoding_format", "string", one_of("float", "base64"), "float or base64", status = "supported",
	    label = "OpenAI" },
	{ "dimensions", "number", int_from(1), "a positive integer", status = "supported", label = "OpenAI" },
	{ "user", "string", status = "no-op", label = "OpenAI" },
}

local by_name = {}
for _, c in ipairs(M.FIELDS) do
	by_name[c[1]] = c
end

--- Checks an embedding request body and makes the engine request.
-- It removes top-level nulls. A field not in `FIELDS` gives code
-- `"unknown_parameter"`, a bad value `"invalid_value"`.
-- @tparam any b the decoded body
-- @treturn[1] table the request: `model`, `input` (a list; each entry
-- is a string or a list of token ids), `dimensions` and
-- `encoding_format` (default `"float"`)
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] ?string the bad parameter name
-- @treturn[2] ?string the OpenAI error code
function M.check(b)
	if type(b) ~= "table" or json.isarray(b) then
		return nil, "request body must be a JSON object"
	end
	for k, v in pairs(b) do
		if v == json.null then
			b[k] = nil
		end
	end
	for k in pairs(b) do
		if type(k) ~= "string" or not by_name[k] then
			return nil, ("unknown field %s: see the embeddings section of docs/api.md"):format(tostring(k)),
			    tostring(k), "unknown_parameter"
		end
	end
	if b.input == nil then
		return nil, "input is required", "input", "invalid_value"
	end
	for _, c in ipairs(M.FIELDS) do
		local v = b[c[1]]
		if v ~= nil then
			local good = false
			for t in c[2]:gmatch("[^|]+") do
				good = good or type(v) == t
			end
			if good and c[3] then
				good = c[3](v)
			end
			if not good then
				local msg = c[1] .. " must be " .. (c[4] or "a " .. c[2])
				if c[1] == "input" then
					msg = select(2, inputs(v)) or msg
				end
				return nil, msg, c[1], "invalid_value"
			end
		end
	end
	return {
		model = b.model, input = inputs(b.input), dimensions = math.tointeger(b.dimensions),
		encoding_format = b.encoding_format or "float",
	}
end

--- The first n values of v, scaled to length 1.
-- @tparam {number,...} v the vector
-- @tparam integer n the count, at most #v
-- @treturn {number,...} a new vector
function M.truncate(v, n)
	local out, s = {}, 0
	for i = 1, n do
		out[i] = v[i]
		s = s + v[i] * v[i]
	end
	s = s > 0 and 1 / math.sqrt(s) or 0
	for i = 1, n do
		out[i] = out[i] * s
	end
	return out
end

--- The base64 text of a vector as little-endian float32 values.
-- @tparam {number,...} v the vector
-- @treturn string the text
function M.encode(v)
	local parts = {}
	for i, x in ipairs(v) do
		parts[i] = string.pack("<f", x)
	end
	return base64.encode(table.concat(parts))
end

--- Makes the OpenAI response body from an engine result.
-- With `dimensions`, each vector keeps its first values and is scaled
-- to length 1 again. With `encoding_format` `"base64"`, each vector is
-- base64 text of little-endian float32 values.
-- @tparam table req the request from `check`
-- @tparam table r the result of `Engine:embed`
-- @tparam string model the model id
-- @treturn[1] table the body
-- @treturn[2] nil
-- @treturn[2] string an error message, when `dimensions` is more than
-- the length of the vectors
function M.reply(req, r, model)
	local data = json.array()
	for i, v in ipairs(r.data) do
		if req.dimensions then
			if req.dimensions > #v then
				return nil, ("dimensions must be at most %d for model %s"):format(#v, model)
			end
			v = M.truncate(v, req.dimensions)
		end
		data[i] = {
			object = "embedding", index = i - 1,
			embedding = req.encoding_format == "base64" and M.encode(v) or json.array(v),
		}
	end
	local n = r.usage and r.usage.prompt_tokens or 0
	return {
		object = "list", data = data, model = model,
		usage = { prompt_tokens = n, total_tokens = n },
	}
end

return M
