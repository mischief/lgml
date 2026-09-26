-- Vision input on the CPU: image preprocessing as llama.cpp mtmd does
-- it, and the Gemma 4 encoder on a tiny random mmproj made here against
-- a plain Lua forward. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local preprocess = require("lgml.vision.preprocess")
local vision = require("lgml.vision")
local gguf = require("gguf")
local tiny = require("mmproj_tiny")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local be <close> = assert(lgml.backend("CPU"))

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

local function f32(x)
	return (string.unpack("<f", string.pack("<f", x)))
end

-- An RGB image of w x h from px(x, y, k), a byte.
local function make(w, h, px)
	local v = {}
	for k = 0, 2 do
		for y = 0, h - 1 do
			for x = 0, w - 1 do
				v[k * w * h + y * w + x + 1] = px(x, y, k) / 255 * 2 - 1
			end
		end
	end
	local t <close> = be:tensor("f32", { 1, 3, h, w }, v)
	return lgml.image_from(t)
end

-- Bytes of an image in (k, y, x) order.
local function bytes(im)
	local t <close> = im:tensor(be, 0, { 0, 0, 0 }, { 1, 1, 1 })
	local v = t:get()
	for i = 1, #v do
		v[i] = math.floor(v[i] * 255 + 0.5)
	end
	return v
end

-- llama.cpp resize_bilinear on (k, y, x) bytes, in f32 steps.
local function bilinear(src, w, h, tw, th)
	local out = {}
	local xr = tw > 1 and f32((w - 1) / (tw - 1)) or 0
	local yr = th > 1 and f32((h - 1) / (th - 1)) or 0
	for k = 0, 2 do
		for y = 0, th - 1 do
			for x = 0, tw - 1 do
				local px, py = f32(x * xr), f32(y * yr)
				local x0, y0 = math.min(math.floor(px), w - 1), math.min(math.floor(py), h - 1)
				local x1, y1 = math.min(x0 + 1, w - 1), math.min(y0 + 1, h - 1)
				local xf, yf = f32(px - x0), f32(py - y0)
				local function at(xx, yy)
					return src[k * w * h + yy * w + xx + 1]
				end
				local top = f32(at(x0, y0) + f32(f32(at(x1, y0) - at(x0, y0)) * xf))
				local bot = f32(at(x0, y1) + f32(f32(at(x1, y1) - at(x0, y1)) * xf))
				out[k * tw * th + y * tw + x + 1] = math.floor(f32(top + f32(f32(bot - top) * yf)))
			end
		end
	end
	return out
end

-- Image objects: RGB, placement, bilinear resize and normalization.
do
	local function px(x, y, k)
		return (x * 37 + y * 101 + k * 59) * 977 % 256
	end
	local im = make(13, 7, px)
	local want = bilinear(bytes(im), 13, 7, 29, 11)
	local r = im:resize(29, 11, "bilinear")
	ok(r.width == 29 and r.height == 11 and maxdiff(bytes(r), want) == 0,
	    "bilinear resize matches llama.cpp")
	local down = im:resize(5, 3, "bilinear")
	ok(maxdiff(bytes(down), bilinear(bytes(im), 13, 7, 5, 3)) == 0, "bilinear shrink matches llama.cpp")
	ok(not pcall(im.resize, im, 4, 4, "cubic"), "unknown filter is an error")

	local p = im:place(17, 10, 2, -1, 9)
	local got, src = bytes(p), bytes(im)
	local good = p.width == 17 and p.height == 10
	for k = 0, 2 do
		for y = 0, 9 do
			for x = 0, 16 do
				local sx, sy = x - 2, y + 1
				local v = (sx >= 0 and sx < 13 and sy >= 0 and sy < 7) and
				    src[k * 91 + sy * 13 + sx + 1] or 9
				good = good and got[k * 170 + y * 17 + x + 1] == v
			end
		end
	end
	ok(good, "place copies the image onto a filled canvas and cuts it off")
	ok(not pcall(im.place, im, 0, 4, 0, 0), "place on an empty canvas is an error")

	local t <close> = be:tensor("f32", { 1, 2, 2, 3 }, {
		-1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1 })
	local ga = lgml.image_from(t)
	local rgb = ga:rgb()
	ok(rgb.channels == 3 and maxdiff(bytes(rgb), {
		0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255 }) == 0,
	    "rgb copies gray to three channels and drops alpha")
	local nt <close> = im:tensor(be, 1, { 0.5, 0.25, 0 }, { 0.5, 2, 1 })
	local v, b = nt:get(), bytes(im)
	local nw = {}
	for i = 1, #b do
		local k = (i - 1) // 91
		nw[i] = (b[i] / 255 - ({ 0.5, 0.25, 0 })[k + 1]) / ({ 0.5, 2, 1 })[k + 1]
	end
	ok(maxdiff(v, nw) < 1e-6, "tensor with mean and std")
	ok(not pcall(im.tensor, im, be, 1, { 0, 0, 0 }, { 1, 0, 1 }), "std 0 is an error")
end

-- Sizes: smart resize and the padded fit, as llama.cpp computes them.
do
	local A, MIN, MAX = 48, 40 * 48 * 48, 280 * 48 * 48
	local w, h = preprocess.smart_size(640, 488, A, MIN, MAX)
	ok(w == 624 and h == 480, "640x488 keeps its size class: " .. w .. "x" .. h)
	w, h = preprocess.smart_size(4000, 3000, A, MIN, MAX)
	ok(w == 912 and h == 672 and w * h <= MAX, "a large image shrinks: " .. w .. "x" .. h)
	w, h = preprocess.smart_size(20, 10, A, MIN, MAX)
	ok(w == 432 and h == 240 and w * h >= MIN, "a small image grows: " .. w .. "x" .. h)
	w, h = preprocess.smart_size(3000, 20, A, MIN, MAX)
	ok(w % A == 0 and h == 48, "a thin image keeps one row of merged patches: " .. w .. "x" .. h)
	local nw, nh, x, y = preprocess.fit(640, 488, 624, 480)
	ok(nw == 624 and nh == 476 and x == 0 and y == 2, "fit scales and centers")
	nw, nh, x, y = preprocess.fit(10, 30, 48, 48)
	ok(nw == 16 and nh == 48 and x == 16 and y == 0, "fit rounds sides up")
end

-- prepare: decode, resize onto black bars, or keep the size.
do
	local im = make(100, 60, function(x, y, k)
		return (x + y + k * 50) % 256
	end)
	local opts = { align = 48, min_pixels = 40 * 48 * 48, max_pixels = 280 * 48 * 48 }
	local p = assert(preprocess.prepare(im:png(), opts))
	local tw, th = preprocess.smart_size(100, 60, 48, opts.min_pixels, opts.max_pixels)
	ok(p.width == tw and p.height == th and p.channels == 3, "prepare gives the smart size")
	local nw, nh, x, y = preprocess.fit(100, 60, tw, th)
	local placed = im:resize(nw, nh, "bilinear"):place(tw, th, x, y)
	ok(maxdiff(bytes(p), bytes(placed)) == 0, "prepare resizes and places as llama.cpp")
	local same = assert(preprocess.prepare(im, { fixed = { 100, 60 } }))
	ok(maxdiff(bytes(same), bytes(im)) == 0, "a fixed size equal to the image keeps the pixels")
	local bad, err = preprocess.prepare("not an image", opts)
	ok(bad == nil and type(err) == "string", "bad bytes give an error")
end

-- The encoder on a tiny random mmproj against a plain Lua forward.
do
	local NOUT = 6
	local kv, ts, W = tiny.build(NOUT)
	local path = os.tmpname()
	gguf.write(path, kv, ts)
	local enc = vision.load(be, path, { min_tokens = 1, max_tokens = 6 })
	ok(enc.projector == "gemma4v" and enc.hp.n_out == NOUT and enc.hp.act == "gelu_quick",
	    "the mmproj loads as gemma4v")
	local im = make(20, 11, function(x, y, k)
		return (x * 29 + y * 53 + k * 71) % 256
	end)
	local prep = assert(enc:prepare(im))
	ok(prep.width == 18 and prep.height == 12 and enc:tokens(prep) == 6,
	    ("an image prepares to %dx%d, %d tokens"):format(prep.width, prep.height, enc:tokens(prep)))
	local want = tiny.reference(W, bytes(prep), prep.width, prep.height, NOUT)
	local buf, count = assert(enc:encode(im))
	local got = {}
	for i = 1, #buf // 4 do
		got[i] = string.unpack("<f", buf:bytes((i - 1) * 4, 4))
	end
	ok(count == 6 and #got == 6 * NOUT, "encode gives 6 rows of " .. NOUT)
	local d = maxdiff(got, want)
	ok(d < 2e-3, ("the encoder matches a plain forward (max diff %.3g)"):format(d))
	local fa = vision.load(be, path, { min_tokens = 1, max_tokens = 6, flash_attn = true })
	local fbuf = assert(fa:encode(im))
	local fgot = {}
	for i = 1, #fbuf // 4 do
		fgot[i] = string.unpack("<f", fbuf:bytes((i - 1) * 4, 4))
	end
	d = maxdiff(fgot, want)
	ok(d < 3e-2, ("flash attention stays close (max diff %.3g)"):format(d))
	local bad, err = enc:encode("not an image")
	ok(bad == nil and type(err) == "string", "bad bytes give an error")
	ok(not pcall(vision.load, be, here .. "gguf.lua"), "a file that is not gguf is an error")
	os.remove(path)
end

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
