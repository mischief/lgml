-- PNG and JPEG through lgml.image, image objects, and the FLUX VAE
-- graph with a tiny random VAE, on the CPU. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local png = require("png")
local samples = require("image_samples")
local vae = require("lgml.image.vae")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local be <close> = assert(lgml.backend("CPU"))

-- Largest absolute difference of two lists.
local function maxdiff(a, b)
	if #a ~= #b then
		return math.huge
	end
	local m = 0
	for i = 1, #a do
		m = math.max(m, math.abs(a[i] - b[i]))
	end
	return m
end

-- The (1, 3, H, W) values that image:tensor(be, bg) makes from 8-bit
-- pixels; px(x, y) returns r, g, b, a in 0 to 255.
local function expect(w, h, bg, px)
	local out = {}
	for x = 0, w - 1 do
		for y = 0, h - 1 do
			local p = { px(x, y) }
			local a = p[4] / 255
			for k = 1, 3 do
				out[(k - 1) * w * h + y * w + x + 1] = (p[k] / 255 * a + bg * (1 - a)) * 2 - 1
			end
		end
	end
	return out
end

local function tensor_of(im, bg)
	local t <close> = im:tensor(be, bg)
	return t:get()
end

-- PNG decode: every color type and bit depth, plain and Adam7.

local function to8(v, depth)
	if depth == 16 then
		return (v * 255 + 32767) // 65535
	end
	return v * 255 // ((1 << depth) - 1)
end

local W, H = 13, 7
local function sample(depth)
	local top = (1 << depth) - 1
	return function(x, y, k)
		return (x * 37 + y * 101 + k * 59) * 977 % (top + 1)
	end
end

local cases = {
	{ 0, { 1, 2, 4, 8, 16 } }, { 2, { 8, 16 } }, { 4, { 8, 16 } }, { 6, { 8, 16 } },
}
for _, c in ipairs(cases) do
	local ctype = c[1]
	for _, depth in ipairs(c[2]) do
		for _, interlace in ipairs({ false, true }) do
			local s = sample(depth)
			local name = ("ctype %d depth %d%s"):format(ctype, depth, interlace and " adam7" or "")
			local im = lgml.image(png.encode({ w = W, h = H, depth = depth, ctype = ctype,
			    interlace = interlace, sample = s }))
			local nch = ({ [0] = 1, [2] = 3, [4] = 2, [6] = 4 })[ctype]
			ok(im and im.width == W and im.height == H and im.channels == nch, name .. " shape")
			local want = expect(W, H, 0.25, function(x, y)
				local v = {}
				for k = 0, nch - 1 do
					v[k + 1] = to8(s(x, y, k), depth)
				end
				if nch <= 2 then
					return v[1], v[1], v[1], nch == 2 and v[2] or 255
				end
				return v[1], v[2], v[3], v[4] or 255
			end)
			ok(im and maxdiff(tensor_of(im, 0.25), want) < 1e-5, name .. " pixels")
			if im then
				local back = assert(lgml.image(im:png()))
				ok(back:png() == im:png() and back.channels == nch, name .. " round trip")
			end
		end
	end
end

-- Palette with tRNS, palette at every depth, and gray and RGB color keys.
do
	local pal = "\255\0\0\0\255\0\0\0\255\255\255\255"
	for _, depth in ipairs({ 1, 2, 4, 8 }) do
		local top = math.min(3, (1 << depth) - 1)
		local s = function(x, y) return (x + 2 * y) % (top + 1) end
		local im = lgml.image(png.encode({ w = W, h = H, depth = depth, ctype = 3, palette = pal,
		    trns = "\128\0", sample = s }))
		ok(im and im.channels == 4, "palette depth " .. depth .. " with tRNS is RGBA")
		local want = expect(W, H, 1, function(x, y)
			local i = s(x, y)
			local r, g, b = pal:byte(3 * i + 1, 3 * i + 3)
			return r, g, b, ({ 128, 0, 255, 255 })[i + 1]
		end)
		ok(im and maxdiff(tensor_of(im, 1), want) < 1e-5, "palette depth " .. depth .. " pixels")
	end
	local im = lgml.image(png.encode({ w = 4, h = 1, depth = 8, ctype = 3, palette = pal,
	    sample = function(x) return x end }))
	ok(im and im.channels == 3, "palette without tRNS is RGB")
	im = lgml.image(png.encode({ w = 2, h = 1, depth = 16, ctype = 0, trns = "\1\2",
	    sample = function(x) return x == 0 and 0x102 or 7 end }))
	ok(im and im.channels == 2 and maxdiff(tensor_of(im, 0), { -1, -1, -1, -1, -1, -1 }) < 1e-6,
	    "gray color key makes alpha")
	im = lgml.image(png.encode({ w = 2, h = 1, depth = 8, ctype = 2, trns = "\0\1\0\2\0\3",
	    sample = function(x, _, k) return x == 0 and k + 1 or 255 end }))
	ok(im and im.channels == 4 and maxdiff(tensor_of(im, 0), { -1, 1, -1, 1, -1, 1 }) < 1e-6,
	    "RGB color key makes alpha")
	im = lgml.image(samples.DYNAMIC_PNG)
	ok(im and im.width == 12 and im.height == 6 and im.channels == 3, "dynamic Huffman block")
end

-- Encode: image_from, compression and the fields.
do
	for c = 1, 4 do
		local t <close> = be:tensor("f32", { 1, c, 37, 53 })
		t:fill("uniform", -1.2, 1.2, c)
		local im = lgml.image_from(t)
		local back = assert(lgml.image(im:png()))
		ok(back.width == 53 and back.height == 37 and back.channels == c,
		    "image_from with " .. c .. " channels round trips")
	end
	local t <close> = be:tensor("f32", { 1, 3, 37, 53 })
	t:fill("uniform", -1, 1, 7)
	local back = lgml.image(lgml.image_from(t):png())
	ok(maxdiff(tensor_of(back, 1), t:get()) <= 1 / 255 + 1e-6, "tensor to PNG to tensor within 1/255")
	local b2 <close> = be:tensor("f32", { 2, 3, 2, 2 }, {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		-5, 5, 0 / 0, 0.5, -5, 5, 0 / 0, 0.5, -5, 5, 0 / 0, 0.5,
	})
	local im = lgml.image_from(b2, 2)
	ok(maxdiff(tensor_of(im, 0), { -1, 1, -1, 191 / 127.5 - 1, -1, 1, -1, 191 / 127.5 - 1,
	    -1, 1, -1, 191 / 127.5 - 1 }) < 1e-6, "image_from picks n, clamps and maps NaN to -1")
	ok(not pcall(lgml.image_from, b2, 3), "image_from rejects n out of range")
	-- Patterns large enough for LZ77 matches and several deflate blocks.
	local big <close> = be:tensor("f32", { 1, 3, 200, 300 })
	big:fill("range", -1, 1 / 90000)
	local im2 = lgml.image_from(big)
	local enc = im2:png()
	local dec = assert(lgml.image(enc))
	ok(dec:png() == enc and #enc < 200 * 300 * 3 // 2, "large smooth image compresses and round trips")
	local noise <close> = be:tensor("f32", { 1, 3, 150, 150 })
	noise:fill("uniform", -1, 1, 3)
	local im3 = lgml.image_from(noise)
	ok(lgml.image(im3:png()):png() == im3:png(), "noise round trips through stored blocks")
end

-- Resize and crop.
do
	local t <close> = be:tensor("f32", { 1, 3, 20, 30 })
	t:fill("const", 0.2)
	local im = lgml.image_from(t)
	local r = im:resize(7, 45)
	ok(r.width == 7 and r.height == 45, "resize shape")
	local v, want = tensor_of(r, 0), {}
	for i = 1, #v do
		want[i] = tensor_of(im, 0)[1]
	end
	ok(maxdiff(v, want) < 1e-6, "resize keeps a constant image")
	local grad <close> = be:tensor("f32", { 1, 1, 4, 4 }, {
		-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1 })
	local c = lgml.image_from(grad):crop(1, 1, 2, 2)
	ok(c.width == 2 and c.channels == 1 and maxdiff(tensor_of(c, 0), {
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }) < 1e-6, "crop")
	ok(not pcall(im.crop, im, 25, 0, 6, 1), "crop outside the image is an error")
	ok(not pcall(im.resize, im, 0, 5), "resize to 0 is an error")
end

-- JPEG: decode by magic bytes, CMYK, progressive, restart markers, encode.
do
	for _, name in ipairs({ "CMYK_JPEG", "PROGRESSIVE_JPEG", "RESTART_JPEG" }) do
		local im = lgml.image(samples[name])
		ok(im and im.width == 16 and im.height == 16 and im.channels == 3, name .. " decodes to RGB")
	end
	local t <close> = be:tensor("f32", { 1, 3, 48, 64 })
	t:fill("range", -1, 2 / (3 * 48 * 64))
	local im = lgml.image_from(t)
	local j = assert(lgml.image(im:jpeg(95)))
	ok(j.width == 64 and j.height == 48 and j.channels == 3, "JPEG round trip shape")
	ok(maxdiff(tensor_of(j, 0), t:get()) < 0.05, "JPEG quality 95 stays close")
	ok(#im:jpeg(10) < #im:jpeg(95), "lower quality is smaller")
	local g <close> = be:tensor("f32", { 1, 2, 8, 8 })
	g:fill("uniform", -1, 1)
	local gj = lgml.image(lgml.image_from(g):jpeg())
	ok(gj and gj.channels == 1, "gray and alpha encode as a gray JPEG")
	ok(not pcall(im.jpeg, im, 0), "JPEG quality 0 is an error")
end

-- Bad input gives nil and a message, never a crash.
do
	local good = png.encode({ w = 5, h = 4, depth = 8, ctype = 2, sample = sample(8) })
	local function bad(s, name, pattern)
		local good_call, im, err = pcall(lgml.image, s)
		ok(good_call and im == nil and type(err) == "string" and
		    (not pattern or err:lower():find(pattern, 1, true)), name .. (err and " (" .. err .. ")" or ""))
	end
	-- Decoding can stop after the image data, so only cuts inside it must fail.
	local idat_end = good:find("IEND", 1, true) - 8
	local all = true
	for i = 0, idat_end - 1 do
		local good_call, im = pcall(lgml.image, good:sub(1, i))
		all = all and good_call and im == nil
	end
	ok(all, "every truncation before the end of IDAT is an error")
	bad("", "empty")
	bad("GIF89a", "not a PNG", "not a png or jpeg")
	local crc = idat_end
	bad(good:sub(1, crc - 1) .. string.char(good:byte(crc) ~ 1) .. good:sub(crc + 1), "IDAT CRC mismatch",
	    "crc")
	local function ihdr(w, h, depth, ctype, il, idat_data)
		return png.wrap(w, h, depth, ctype, il or 0, idat_data or png.zlib("\0\0\0\0"))
	end
	bad(ihdr(0, 1, 8, 0), "zero width", "empty")
	bad(ihdr(100000, 100000, 8, 0), "too large", "too large")
	bad(ihdr(10000, 10000, 8, 0), "pixel count too large", "too large")
	bad(ihdr(1, 1, 3, 0), "bad depth")
	bad(ihdr(1, 1, 16, 3), "palette depth 16")
	bad(ihdr(1, 1, 8, 5), "bad color type")
	bad(ihdr(1, 1, 8, 0, 2), "bad interlace")
	bad(ihdr(1, 1, 8, 3, 0, png.zlib("\0\0")), "palette without PLTE")
	bad(ihdr(3, 1, 8, 0, 0, png.zlib("\0\1\2")), "short image data", "not enough")
	bad(ihdr(1, 1, 8, 0, 0, png.zlib("\5\1")), "bad filter type", "filter")
	bad(ihdr(1, 1, 8, 0, 0, "\x78\x01\x01\x02\x00\xfd\xff\0\1\0\0\0\0"), "adler mismatch", "check")
	bad(ihdr(1, 1, 8, 0, 0, "\x79\x01"), "bad zlib header", "header")
	bad(ihdr(1, 1, 8, 0, 0, "\x78\x01\x07"), "bad block type", "block type")
	bad(ihdr(1, 1, 8, 0, 0, "\x78\x01\x01\x02\x00\xfe\xff"), "bad stored length", "stored")
	-- A fixed block with a distance before the start of the output.
	bad(ihdr(4, 1, 8, 0, 0, "\x78\x01\x03\x02\x00"), "distance too far", "distance")
	bad(png.SIG .. png.chunk("IEND", ""), "no IHDR", "ihdr")
	bad(good:sub(1, 33) .. png.chunk("ABCD", "") .. good:sub(34), "unknown critical chunk")
	ok(lgml.image(good:sub(1, 33) .. png.chunk("abCD", "x") .. good:sub(34)) ~= nil,
	    "unknown ancillary chunk is skipped")
	-- An out-of-range palette index and extra image data are not fatal to libpng.
	for _, s in ipairs({
		png.encode({ w = 2, h = 2, depth = 8, ctype = 3, palette = "\1\2\3",
		    sample = function(x) return x end }),
		ihdr(1, 1, 8, 0, 0, png.zlib("\0\1\2")),
	}) do
		ok(pcall(lgml.image, s), "tolerated error does not raise")
	end
	local jpeg = samples.PROGRESSIVE_JPEG
	local cut = true
	for i = 0, #jpeg - 1 do
		cut = cut and pcall(lgml.image, jpeg:sub(1, i))
	end
	ok(cut, "every JPEG truncation decodes or fails cleanly")
	bad(jpeg:sub(1, -40), "JPEG data that ends early", "ends early")
	bad("\xff\xd8\xff", "JPEG with no frame")
	local sof = jpeg:find("\xff\xc2", 1, true)
	bad(jpeg:sub(1, sof + 3) .. "\12" .. jpeg:sub(sof + 5), "12-bit JPEG")
	bad(jpeg:sub(1, sof + 4) .. "\x7f\xff\x7f\xff" .. jpeg:sub(sof + 9), "huge JPEG", "too large")
	-- Random mutations with CRCs fixed: each gives an image or an error.
	local seed, survived = 1, true
	local function rnd(k)
		seed = (seed * 1103515245 + 12345) % 2147483648
		return seed % k
	end
	for i = 1, 3000 do
		local src = i % 2 == 0 and good or samples.RESTART_JPEG
		local m = { src:byte(1, -1) }
		for _ = 1, 1 + rnd(4) do
			local k = 9 + rnd(#m - 8)
			m[k] = rnd(256)
		end
		local b = string.char(table.unpack(m))
		local good_call, im = pcall(lgml.image, i % 2 == 0 and png.fix_crc(b) or b)
		survived = survived and good_call
		if im then
			im:close()
		end
	end
	ok(survived, "3000 mutated PNG and JPEG files decode or fail cleanly")
end

-- graph:scale with a bias and the latent scale helpers.
do
	local z <close> = be:tensor("f32", { 1, 2, 1, 2 }, { -1, 0, 0.5, 2 })
	local g <close> = be:graph()
	local s = vae.scale(g, z)
	local u = vae.unscale(g, s)
	g:compute(s, u)
	local want = {}
	for i, v in ipairs(z:get()) do
		want[i] = (v - vae.SHIFT) * vae.SCALE
	end
	ok(maxdiff(s:get(), want) < 1e-6, "vae.scale")
	ok(maxdiff(u:get(), z:get()) < 1e-6, "vae.unscale inverts vae.scale")
end

-- A tiny VAE with random weights: 2 levels of 32 and 64 channels, 4
-- latent channels, factor 2.
local function tiny_vae(path)
	local entries, seed = {}, 7
	local function rnd()
		seed = (seed * 1103515245 + 12345) % 2147483648
		return seed / 2147483648 - 0.5
	end
	local function add(name, shape, fill)
		local count = 1
		for _, d in ipairs(shape) do
			count = count * d
		end
		local v = {}
		for i = 1, count do
			v[i] = fill == "one" and 1 or fill == "zero" and 0 or rnd() * fill
		end
		entries[#entries + 1] = { name, shape, string.pack("<" .. string.rep("f", count), table.unpack(v)) }
	end
	local function conv(p, cout, cin, k)
		add(p .. ".weight", { cout, cin, k, k }, 2 / math.sqrt(cin * k * k))
		add(p .. ".bias", { cout }, 0.1)
	end
	local function norm(p, c)
		add(p .. ".weight", { c }, "one")
		add(p .. ".bias", { c }, "zero")
	end
	local function res(p, cin, cout)
		norm(p .. ".norm1", cin)
		conv(p .. ".conv1", cout, cin, 3)
		norm(p .. ".norm2", cout)
		conv(p .. ".conv2", cout, cout, 3)
		if cin ~= cout then
			conv(p .. ".nin_shortcut", cout, cin, 1)
		end
	end
	local function mid(p, c)
		res(p .. ".mid.block_1", c, c)
		norm(p .. ".mid.attn_1.norm", c)
		for _, k in ipairs({ "q", "k", "v", "proj_out" }) do
			conv(p .. ".mid.attn_1." .. k, c, c, 1)
		end
		res(p .. ".mid.block_2", c, c)
	end
	conv("encoder.conv_in", 32, 3, 3)
	res("encoder.down.0.block.0", 32, 32)
	conv("encoder.down.0.downsample.conv", 32, 32, 3)
	res("encoder.down.1.block.0", 32, 64)
	mid("encoder", 64)
	norm("encoder.norm_out", 64)
	conv("encoder.conv_out", 8, 64, 3)
	conv("decoder.conv_in", 64, 4, 3)
	mid("decoder", 64)
	res("decoder.up.1.block.0", 64, 64)
	res("decoder.up.1.block.1", 64, 64)
	conv("decoder.up.1.upsample.conv", 64, 64, 3)
	res("decoder.up.0.block.0", 64, 32)
	res("decoder.up.0.block.1", 32, 32)
	norm("decoder.norm_out", 32)
	conv("decoder.conv_out", 3, 32, 3)
	local h, data, off = {}, {}, 0
	for _, e in ipairs(entries) do
		h[#h + 1] = ('"%s":{"dtype":"F32","shape":[%s],"data_offsets":[%d,%d]}'):format(
		    e[1], table.concat(e[2], ","), off, off + #e[3])
		data[#data + 1] = e[3]
		off = off + #e[3]
	end
	local head = "{" .. table.concat(h, ",") .. "}"
	head = head .. string.rep(" ", -#head % 8)
	local f = assert(io.open(path, "wb"))
	f:write(string.pack("<I8", #head), head, table.concat(data))
	f:close()
end

do
	local path = os.tmpname()
	tiny_vae(path)
	local m = vae.load(be, path)
	local m2 = vae.load(be, path, { flash = false, direct = false })
	os.remove(path)
	ok(m.cfg.levels == 2 and m.cfg.blocks == 2 and m.cfg.z == 4, "tiny VAE layout from shapes")
	local x <close> = be:tensor("f32", { 2, 3, 24, 40 })
	x:fill("uniform", -1, 1, 11)
	local g <close> = be:graph()
	local z = m:encode(g, x)
	local z2 = m2:encode(g, x)
	g:compute(z, z2)
	local zs = z:shape()
	ok(zs[1] == 2 and zs[2] == 4 and zs[3] == 12 and zs[4] == 20, "encode shape (N, 4, H/2, W/2)")
	ok(maxdiff(z:get(), z2:get()) < 2e-2, "flash and softmax attention agree in encode")
	local zl <close> = be:tensor("f32", zs, z:bytes())
	local g2 <close> = be:graph()
	local y = m:decode(g2, zl)
	local y2 = m2:decode(g2, zl)
	g2:compute(y, y2)
	local ys = y:shape()
	ok(ys[1] == 2 and ys[2] == 3 and ys[3] == 24 and ys[4] == 40, "decode shape (N, 3, 2h, 2w)")
	local yv = y:get()
	local inside = true
	for i = 1, #yv do
		inside = inside and yv[i] >= -1 and yv[i] <= 1
	end
	ok(inside, "decode clamps to [-1, 1]")
	ok(maxdiff(yv, y2:get()) < 2e-2, "flash and softmax attention agree in decode")
	local one = m:decode_tiled(zl)
	ok(maxdiff(one:get(), yv) == 0, "decode_tiled with one tile is decode")
	local zt = m:encode_tiled(x)
	ok(maxdiff(zt:get(), z:get()) == 0, "encode_tiled with one tile is encode")
	local tiled, peak = m:decode_tiled(zl, { tile = 8, overlap = 3 })
	local ts = tiled:shape()
	local fine = ts[3] == 24 and ts[4] == 40 and peak > 0
	for _, v in ipairs(tiled:get()) do
		fine = fine and v >= -1 and v <= 1
	end
	ok(fine, "decode_tiled with 2 x 3 tiles")
	local zt2 = m:encode_tiled(x, { tile = 6, overlap = 2 })
	ok(maxdiff(zt2:get(), z:get()) < 100 and zt2:shape()[4] == 20, "encode_tiled with 2 x 4 tiles")
	ok(not pcall(m.encode, m, g, be:tensor("f32", { 1, 3, 5, 8 })), "encode needs sides that divide")
	m:close()
	m2:close()
end

-- The engine gives 499 only for a cancel; other pipeline failures give 500.
do
	local pipeline = require("lgml.image.pipeline")
	local load = pipeline.load
	local fail
	pipeline.load = function()
		return { generate = function(_, req)
			if req.step(1, 1) == false then
				return nil, "cancelled"
			end
			return nil, fail
		end }
	end
	local eng = require("lgml.image.engine").new({ dit = "d.gguf", llm = "q.gguf", vae = "ae.safetensors" })
	pipeline.load = load
	fail = "out of memory"
	local r, msg, status = eng:generate({ prompt = "x", width = 64, height = 64 })
	ok(r == nil and msg == "out of memory" and status == 500, "a pipeline failure gives 500")
	r, msg, status = eng:generate({ prompt = "x", width = 64, height = 64 },
	    { cancelled = function() return true end })
	ok(r == nil and msg == "cancelled" and status == 499, "a cancel gives 499")
end

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
