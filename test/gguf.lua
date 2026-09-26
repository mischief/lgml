-- Small gguf writer for tests: metadata and optional tensors.
local M = {}

M.UINT32, M.INT32, M.FLOAT32, M.BOOL, M.STRING, M.ARRAY = 4, 5, 6, 7, 8, 9
M.UINT64, M.INT64 = 10, 11

-- ggml tensor type ids.
M.F32, M.F16 = 0, 1

M.ALIGN = 32

local function str(s)
	return string.pack("<I8", #s) .. s
end

local function value(t, v)
	if t == M.UINT32 then
		return string.pack("<I4", v)
	elseif t == M.INT32 then
		return string.pack("<i4", v)
	elseif t == M.FLOAT32 then
		return string.pack("<f", v)
	elseif t == M.UINT64 then
		return string.pack("<I8", v)
	elseif t == M.INT64 then
		return string.pack("<i8", v)
	elseif t == M.BOOL then
		return string.pack("<B", v and 1 or 0)
	elseif t == M.STRING then
		return str(v)
	end
	error("gguf: unsupported type " .. tostring(t))
end

local function pad(s)
	return s .. string.rep("\0", -#s % M.ALIGN)
end

-- Tensor bytes: data is a string of raw bytes, or a list of numbers
-- stored as f32.
local function tensor_bytes(t)
	if type(t.data) == "string" then
		return t.data
	end
	local parts = {}
	for i, x in ipairs(t.data) do
		parts[i] = string.pack("<f", x)
	end
	return table.concat(parts)
end

-- Returns gguf bytes. kv is a list of { key, type, value }; for an
-- array, type is { M.ARRAY, element type } and value is a list.
-- tensors is a list of { name, shape, data [, type] } with shape in
-- torch order; type defaults to M.F32.
function M.encode(kv, tensors)
	tensors = tensors or {}
	local out = { "GGUF", string.pack("<I4I8I8", 3, #tensors, #kv) }
	for _, e in ipairs(kv) do
		local key, t, v = e[1], e[2], e[3]
		out[#out + 1] = str(key)
		if type(t) == "table" then
			out[#out + 1] = string.pack("<I4I4I8", M.ARRAY, t[2], #v)
			for _, x in ipairs(v) do
				out[#out + 1] = value(t[2], x)
			end
		else
			out[#out + 1] = string.pack("<I4", t) .. value(t, v)
		end
	end
	local data, off = {}, 0
	for i, t in ipairs(tensors) do
		local name, shape = t[1], t[2]
		local bytes = tensor_bytes({ data = t[3] })
		out[#out + 1] = str(name) .. string.pack("<I4", #shape)
		for d = #shape, 1, -1 do
			out[#out + 1] = string.pack("<I8", shape[d])
		end
		out[#out + 1] = string.pack("<I4I8", t[4] or M.F32, off)
		data[i] = pad(bytes)
		off = off + #data[i]
	end
	return pad(table.concat(out)) .. table.concat(data)
end

function M.write(path, kv, tensors)
	local f = assert(io.open(path, "wb"))
	f:write(M.encode(kv, tensors))
	f:close()
end

return M
