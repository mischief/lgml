-- Copy a Clef gguf as a qwen35 gguf, so llama.cpp runs its backbone with
-- images: rename the clef.* keys, drop the decision keys and the head
-- tensors. Plain Lua; the header must fit in its first 64 MiB.
-- Usage: lua5.4 clef_qwen35.lua Clef.gguf out.gguf
local src, dst = arg[1], arg[2]
assert(src and dst, "usage: clef_qwen35.lua Clef.gguf out.gguf")
local f = assert(io.open(src, "rb"))
local head = f:read(64 << 20)
local pos = 1
local function u32() local v; v, pos = string.unpack("<I4", head, pos); return v end
local function u64() local v; v, pos = string.unpack("<I8", head, pos); return v end
local function str() local n = u64(); local s = head:sub(pos, pos + n - 1); pos = pos + n; return s end
assert(head:sub(1, 4) == "GGUF")
pos = 5
local ver = u32()
local n_t, n_kv = u64(), u64()
local SZ = { [0] = 1, 1, 2, 2, 4, 4, 4, 1, nil, nil, 8, 8, 8 }
local function skip(t)
	if t == 8 then str()
	elseif t == 9 then
		local et, n = u32(), u64()
		if et == 8 then for _ = 1, n do str() end else pos = pos + SZ[et] * n end
	else pos = pos + SZ[t] end
end
local kvs = {}
for _ = 1, n_kv do
	local key = str()
	local a = pos
	local t = u32()
	skip(t)
	local raw = head:sub(a, pos - 1)
	if key:match("^clef%.decision%.") then
		-- drop
	else
		if key == "general.architecture" then
			raw = string.pack("<I4", 8) .. string.pack("<I8", 6) .. "qwen35"
		end
		key = key:gsub("^clef%.", "qwen35.")
		kvs[#kvs + 1] = string.pack("<I8", #key) .. key .. raw
	end
end
local ts = {}
for _ = 1, n_t do
	local name = str()
	local nd = u32()
	local dims = {}
	for i = 1, nd do dims[i] = u64() end
	local ty = u32()
	local off = u64()
	ts[#ts + 1] = { name = name, dims = dims, ty = ty, off = off }
end
local data0 = (pos - 1 + 31) // 32 * 32
local fsize = f:seek("end")
table.sort(ts, function(a, b) return a.off < b.off end)
for i, t in ipairs(ts) do
	t.size = (ts[i + 1] and ts[i + 1].off or fsize - data0) - t.off
end
local keep = {}
for _, t in ipairs(ts) do
	if not (t.name:match("^dec%.") or t.name:match("^decision%.") or t.name:match("^token_types%.")) then
		keep[#keep + 1] = t
	end
end
local out = { "GGUF", string.pack("<I4I8I8", ver, #keep, #kvs) }
for _, k in ipairs(kvs) do out[#out + 1] = k end
local off = 0
for _, t in ipairs(keep) do
	local s = string.pack("<I8", #t.name) .. t.name .. string.pack("<I4", #t.dims)
	for _, d in ipairs(t.dims) do s = s .. string.pack("<I8", d) end
	out[#out + 1] = s .. string.pack("<I4I8", t.ty, off)
	t.new = off
	off = off + (t.size + 31) // 32 * 32
end
local hdr = table.concat(out)
hdr = hdr .. string.rep("\0", -#hdr % 32)
local o = assert(io.open(dst, "wb"))
o:write(hdr)
for _, t in ipairs(keep) do
	f:seek("set", data0 + t.off)
	local left = t.size
	while left > 0 do
		local c = f:read(math.min(left, 64 << 20))
		o:write(c)
		left = left - #c
	end
	o:write(string.rep("\0", -t.size % 32))
end
o:close()
print(#keep, "tensors", #kvs, "keys")
