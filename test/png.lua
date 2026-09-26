-- A small PNG writer in pure Lua for tests, seeds and the fuzz target.
-- It writes every color type, bit depth and filter, plain or Adam7,
-- with stored (uncompressed) deflate blocks.
local M = {}

local crc_table = {}
for i = 0, 255 do
	local c = i
	for _ = 1, 8 do
		c = c & 1 == 1 and 0xedb88320 ~ (c >> 1) or c >> 1
	end
	crc_table[i] = c
end

--- CRC-32 of a string, as in PNG and zlib.
function M.crc32(s)
	local c = 0xffffffff
	for i = 1, #s do
		c = crc_table[(c ~ s:byte(i)) & 0xff] ~ (c >> 8)
	end
	return c ~ 0xffffffff
end

--- Adler-32 of a string, as in zlib.
function M.adler32(s)
	local a, b = 1, 0
	for i = 1, #s do
		a = (a + s:byte(i)) % 65521
		b = (b + a) % 65521
	end
	return b << 16 | a
end

--- A chunk with its length and CRC.
function M.chunk(kind, data)
	return string.pack(">I4", #data) .. kind .. data .. string.pack(">I4", M.crc32(kind .. data))
end

--- A zlib stream of stored blocks.
function M.zlib(raw)
	local out, pos = { "\x78\x01" }, 1
	repeat
		local n = math.min(65535, #raw - pos + 1)
		out[#out + 1] = string.pack("<BI2I2", pos + n > #raw and 1 or 0, n, ~n & 0xffff)
		out[#out + 1] = raw:sub(pos, pos + n - 1)
		pos = pos + n
	until pos > #raw
	out[#out + 1] = string.pack(">I4", M.adler32(raw))
	return table.concat(out)
end

--- Recompute the CRC of every whole chunk after the signature.
function M.fix_crc(s)
	local out, pos = { s:sub(1, 8) }, 9
	while pos + 11 <= #s do
		local n = string.unpack(">I4", s, pos)
		if n > #s - pos - 11 then
			break
		end
		out[#out + 1] = M.chunk(s:sub(pos + 4, pos + 7), s:sub(pos + 8, pos + 7 + n))
		pos = pos + 12 + n
	end
	out[#out + 1] = s:sub(pos)
	return table.concat(out)
end

local SIG = "\x89PNG\r\n\x1a\n"
local CHANNELS = { [0] = 1, [2] = 3, [3] = 1, [4] = 2, [6] = 4 }
local PASSES = {
	{ 0, 0, 8, 8 }, { 4, 0, 8, 8 }, { 0, 4, 4, 8 }, { 2, 0, 4, 4 },
	{ 0, 2, 2, 4 }, { 1, 0, 2, 2 }, { 0, 1, 1, 2 },
}

local function paeth(a, b, c)
	local p = a + b - c
	local pa, pb, pc = math.abs(p - a), math.abs(p - b), math.abs(p - c)
	if pa <= pb and pa <= pc then
		return a
	end
	return pb <= pc and b or c
end

-- Pack one row of samples at the bit depth.
local function pack(samples, depth)
	local out = {}
	if depth == 16 then
		for i, v in ipairs(samples) do
			out[i] = string.pack(">I2", v)
		end
	elseif depth == 8 then
		for i, v in ipairs(samples) do
			out[i] = string.char(v)
		end
	else
		local acc, bits = 0, 0
		for _, v in ipairs(samples) do
			acc, bits = acc << depth | v, bits + depth
			if bits == 8 then
				out[#out + 1] = string.char(acc)
				acc, bits = 0, 0
			end
		end
		if bits > 0 then
			out[#out + 1] = string.char(acc << (8 - bits))
		end
	end
	return table.concat(out)
end

local function filter(row, prev, bpp, f)
	local out = { string.char(f) }
	for i = 1, #row do
		local a = i > bpp and row:byte(i - bpp) or 0
		local b = prev and prev:byte(i) or 0
		local c = i > bpp and prev and prev:byte(i - bpp) or 0
		local p = ({ 0, a, b, (a + b) // 2, paeth(a, b, c) })[f + 1]
		out[#out + 1] = string.char((row:byte(i) - p) & 0xff)
	end
	return table.concat(out)
end

--- Build a PNG.
-- opts: w, h, depth, ctype, interlace (false), sample(x, y, k) (the
-- sample of channel k, from 0, at the bit depth), palette (a string of
-- RGB triples), trns (tRNS chunk data). Row filters cycle through 0 to 4.
function M.encode(o)
	local nch = CHANNELS[o.ctype]
	local bpp = math.max(1, nch * o.depth // 8)
	local raw, fi = {}, 0
	local passes = o.interlace and PASSES or { { 0, 0, 1, 1 } }
	for _, p in ipairs(passes) do
		local prev
		for y = p[2], o.h - 1, p[4] do
			local samples = {}
			for x = p[1], o.w - 1, p[3] do
				for k = 0, nch - 1 do
					samples[#samples + 1] = o.sample(x, y, k)
				end
			end
			if #samples > 0 then
				local row = pack(samples, o.depth)
				raw[#raw + 1] = filter(row, prev, bpp, fi % 5)
				fi = fi + 1
				prev = row
			end
		end
	end
	local ihdr = string.pack(">I4I4BBBBB", o.w, o.h, o.depth, o.ctype, 0, 0, o.interlace and 1 or 0)
	local parts = { SIG, M.chunk("IHDR", ihdr) }
	if o.palette then
		parts[#parts + 1] = M.chunk("PLTE", o.palette)
	end
	if o.trns then
		parts[#parts + 1] = M.chunk("tRNS", o.trns)
	end
	parts[#parts + 1] = M.chunk("IDAT", M.zlib(table.concat(raw)))
	parts[#parts + 1] = M.chunk("IEND", "")
	return table.concat(parts)
end

--- Wrap IDAT data in a PNG with the given header fields.
function M.wrap(w, h, depth, ctype, interlace, idat, extra)
	local ihdr = string.pack(">I4I4BBBBB", w, h, depth, ctype, 0, 0, interlace)
	return SIG .. M.chunk("IHDR", ihdr) .. (extra or "") .. M.chunk("IDAT", idat) ..
	    M.chunk("IEND", "")
end

M.SIG = SIG

return M
