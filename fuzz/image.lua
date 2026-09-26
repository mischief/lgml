-- PNG and JPEG decode, then encode, resize, crop and tensors. The
-- first byte picks how the rest becomes a file: 0 as is (PNG or JPEG);
-- 1 a PNG with chunk CRCs fixed; 2 and 3 wrap zlib data in a PNG after
-- a 5-byte header (width - 1, height - 1, bit depth, color type,
-- interlace), so inflate sees mutated data. 3 adds PLTE and tRNS.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "../test/?.lua;" .. package.path
local lgml = require("lgml")
local png = require("png")

local be = assert(lgml.backend("CPU"))

local pal = {}
for i = 0, 255 do
	pal[#pal + 1] = string.char(i, 255 - i, i * 7 & 0xff)
end
pal = table.concat(pal)

local function build(s)
	local mode = (s:byte(1) or 0) % 4
	local rest = s:sub(2)
	if mode == 0 then
		return rest
	elseif mode == 1 then
		return png.fix_crc(rest)
	elseif #rest < 5 then
		return rest
	end
	local w, h, depth, ctype, il = rest:byte(1, 5)
	local extra
	if mode == 3 then
		extra = png.chunk("PLTE", pal) .. png.chunk("tRNS", rest:sub(6, 6 + w % 7))
	end
	return png.wrap(w + 1, h + 1, depth, ctype, il, rest:sub(6), extra)
end

return function(s)
	local im = lgml.image(build(s))
	if not im then
		return
	end
	-- A few hundred bytes of JPEG can hold millions of pixels; encode only small images.
	if im.width * im.height > 1 << 20 then
		im:close()
		return
	end
	local enc = im:png()
	local back = assert(lgml.image(enc), "own PNG does not decode")
	assert(back.width == im.width and back.height == im.height and
	    back.channels == im.channels, "PNG round trip changes the shape")
	assert(back:png() == enc, "PNG round trip changes pixels")
	if im.width * im.height <= 4096 then
		local t = im:tensor(be, #s % 3 / 2)
		local again = lgml.image_from(t)
		assert(again.width == im.width and again.channels == 3, "image_from shape")
		local r = im:resize(1 + #s % 37, 1 + #s % 23)
		local c = im:crop(im.width // 3, im.height // 3, (im.width + 1) // 2, (im.height + 1) // 2)
		local j = assert(lgml.image(im:jpeg(1 + #s % 100)), "own JPEG does not decode")
		assert(j.width == im.width and j.height == im.height, "JPEG round trip changes the size")
		j:close()
		r:close()
		c:close()
		again:close()
		t:close()
	end
	back:close()
	im:close()
end
