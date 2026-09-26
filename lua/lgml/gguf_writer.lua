--- Write gguf files: metadata and f32, f16 or bf16 tensors.
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
-- ggml tensor type ids and element sizes.
local TT = { f32 = { 0, 4 }, f16 = { 1, 2 }, bf16 = { 30, 2 } }

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

--- Write a gguf file.
-- kv is a list of { key, type, value }. type is "u8", "i8", "u16",
-- "i16", "u32", "i32", "u64", "i64", "f32", "f64", "bool" or "string",
-- or { "array", element type } with a list value. tensors is a list of
-- { name, type, shape, data }: type is "f32", "f16" or "bf16", shape is
-- in torch order and data holds the raw bytes, as from `tensor:bytes`.
-- @tparam string path the file
-- @tparam table kv the metadata
-- @tparam table tensors the tensors
-- @treturn[1] boolean true
-- @return[2] nil
-- @treturn[2] string an error message
-- @raise on a bad type or a data size that does not fit the shape
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
	local off = 0
	for _, t in ipairs(tensors) do
		local name, ty, shape, data = t[1], TT[t[2]], t[3], t[4]
		assert(ty, "gguf_writer: tensor type must be f32, f16 or bf16")
		local count = 1
		head[#head + 1] = str(name) .. string.pack("<I4", #shape)
		for d = #shape, 1, -1 do
			head[#head + 1] = string.pack("<I8", shape[d])
			count = count * shape[d]
		end
		assert(#data == count * ty[2], "gguf_writer: " .. name .. ": data size does not fit the shape")
		head[#head + 1] = string.pack("<I4I8", ty[1], off)
		off = off + #data + #pad(#data)
	end
	local f, err = io.open(path, "wb")
	if not f then
		return nil, err
	end
	local h = table.concat(head)
	local good, werr = f:write(h, pad(#h))
	for _, t in ipairs(tensors) do
		if not good then
			break
		end
		good, werr = f:write(t[4], pad(#t[4]))
	end
	local closed, cerr = f:close()
	if not good or not closed then
		return nil, tostring(werr or cerr)
	end
	return true
end

return M
