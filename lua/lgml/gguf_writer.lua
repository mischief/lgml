--- Write gguf files: metadata and f32, f16, bf16 or quantized tensors.
-- The layout is gguf version 3 with 32-byte alignment, as llama.cpp
-- writes it. Use `lgml.gguf` to read a file back.
-- @module lgml.gguf_writer
local M = {}

local ALIGN = 32

-- gguf value types.
local VT = {
	u8 = 0, i8 = 1, u16 = 2, i16 = 3, u32 = 4, i32 = 5, f32 = 6, bool = 7,
	string = 8, array = 9, u64 = 10, i64 = 11, f64 = 12,
}
local PACK = {
	u8 = "<I1", i8 = "<i1", u16 = "<I2", i16 = "<i2", u32 = "<I4", i32 = "<i4", f32 = "<f",
	u64 = "<I8", i64 = "<i8", f64 = "<d",
}
-- ggml tensor type ids. Element sizes of the float types; the others
-- take their row size from `lgml.row_size`.
local TT = {
	f32 = 0, f16 = 1, q4_0 = 2, q4_1 = 3, q5_0 = 6, q5_1 = 7, q8_0 = 8,
	q2_K = 10, q3_K = 11, q4_K = 12, q5_K = 13, q6_K = 14,
	iq4_nl = 20, iq4_xs = 23, bf16 = 30,
}
local ESIZE = { f32 = 4, f16 = 2, bf16 = 2 }

local function str(s)
	return string.pack("<I8", #s) .. s
end

local function scalar(t, v)
	if t == "string" then
		return str(v)
	elseif t == "bool" then
		return string.pack("<I1", v and 1 or 0)
	end
	return string.pack(assert(PACK[t], "gguf_writer: unknown type " .. tostring(t)), v)
end

local function pad(n)
	return string.rep("\0", -n % ALIGN)
end

--- Get the data size of a tensor.
-- @tparam string ty tensor type, such as "bf16" or "q4_K"
-- @tparam table shape torch order
-- @treturn int bytes
-- @raise on an unknown type or a row that is not whole blocks
function M.size(ty, shape)
	assert(TT[ty], "gguf_writer: unknown tensor type " .. tostring(ty))
	local rows = 1
	for d = 1, #shape - 1 do
		rows = rows * shape[d]
	end
	local n = shape[#shape]
	if ESIZE[ty] then
		return rows * n * ESIZE[ty]
	end
	return rows * require("lgml").row_size(ty, n)
end

--- Write a gguf file.
-- kv is a list of { key, type, value }. type is "u8", "i8", "u16",
-- "i16", "u32", "i32", "u64", "i64", "f32", "f64", "bool" or "string",
-- or { "array", element type } with a list value. tensors is a list of
-- { name, type, shape, data }: type is a ggml type name such as "f32",
-- "bf16", "q8_0" or "q4_K", and shape is in torch order. data holds the
-- raw bytes, as from `tensor:bytes`, or is a function. The writer calls
-- it as data(write) when the tensor's turn comes, and it passes each
-- piece of the bytes to write(s) in order. Offsets come from the shapes,
-- so only one tensor at a time needs to be in memory.
-- @tparam string path the file
-- @tparam table kv the metadata
-- @tparam table tensors the tensors
-- @treturn[1] boolean true
-- @return[2] nil
-- @treturn[2] string an error message
-- @raise on a bad type, a data size that does not fit the shape, or an
-- error in a data function
function M.write(path, kv, tensors)
	local head = { "GGUF", string.pack("<I4I8I8", 3, #tensors, #kv) }
	for _, e in ipairs(kv) do
		local key, t, v = e[1], e[2], e[3]
		head[#head + 1] = str(key)
		if type(t) == "table" then
			head[#head + 1] = string.pack("<I4I4I8", VT.array, assert(VT[t[2]]), #v)
			for _, x in ipairs(v) do
				head[#head + 1] = scalar(t[2], x)
			end
		else
			head[#head + 1] = string.pack("<I4", assert(VT[t], "gguf_writer: unknown type " ..
			    tostring(t))) .. scalar(t, v)
		end
	end
	local off, sizes = 0, {}
	for i, t in ipairs(tensors) do
		local name, ty, shape, data = t[1], t[2], t[3], t[4]
		local size = M.size(ty, shape)
		head[#head + 1] = str(name) .. string.pack("<I4", #shape)
		for d = #shape, 1, -1 do
			head[#head + 1] = string.pack("<I8", shape[d])
		end
		assert(type(data) == "function" or #data == size,
		    "gguf_writer: " .. name .. ": data size does not fit the shape")
		head[#head + 1] = string.pack("<I4I8", TT[ty], off)
		sizes[i] = size
		off = off + size + #pad(size)
	end
	local f <close>, err = io.open(path, "wb")
	if not f then
		return nil, err
	end
	local h = table.concat(head)
	local good, werr = f:write(h, pad(#h))
	for i, t in ipairs(tensors) do
		if not good then
			break
		end
		local data = t[4]
		if type(data) == "function" then
			local n = 0
			data(function(s)
				n = n + #s
				if good then
					good, werr = f:write(s)
				end
			end)
			assert(n == sizes[i], "gguf_writer: " .. t[1] .. ": data size does not fit the shape")
		else
			good, werr = f:write(data)
		end
		if good then
			good, werr = f:write(pad(sizes[i]))
		end
	end
	local closed, cerr = f:close()
	if not good or not closed then
		return nil, tostring(werr or cerr)
	end
	return true
end

return M
