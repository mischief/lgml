-- Vision input on the CPU: image preprocessing as llama.cpp mtmd does
-- it, and the Gemma 4 encoder on a tiny random mmproj made here against
-- a plain Lua forward. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local preprocess = require("lgml.vision.preprocess")
local vision = require("lgml.vision")
local gguf = require("gguf")

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

-- A tiny gemma4v mmproj: E 16, 2 heads of 8, patch 2, merge 3.
local E, NH, S, M, NL, NOUT, NPOS, FF = 16, 2, 2, 3, 2, 6, 12, 24

-- Seeded uniform numbers in [0, 1), the same on every host.
local function rng(seed)
	local x = seed
	return function()
		x = x * 6364136223846793005 + 1442695040888963407
		return (x >> 11) / 2.0 ^ 53
	end
end

local function build_mmproj()
	local r = rng(4242)
	local W = {}
	local ts = {}
	local function t(name, shape, lo, hi)
		local c = 1
		for _, d in ipairs(shape) do
			c = c * d
		end
		local v = {}
		for i = 1, c do
			v[i] = lo + (hi - lo) * r()
		end
		ts[#ts + 1] = { name, shape, v }
		W[name] = v
	end
	local function w(name, rows, cols)
		local sc = math.sqrt(3 / cols)
		t(name, { rows, cols }, -sc, sc)
	end
	t("v.patch_embd.weight", { E, 3, S, S }, -0.5, 0.5)
	t("v.position_embd.weight", { 2, NPOS, E }, -0.3, 0.3)
	t("v.std_bias", { E }, -0.2, 0.2)
	t("v.std_scale", { E }, 0.5, 1.5)
	w("mm.input_projection.weight", NOUT, E)
	for i = 0, NL - 1 do
		local p = "v.blk." .. i .. "."
		t(p .. "ln1.weight", { E }, 0.8, 1.2)
		t(p .. "ln2.weight", { E }, 0.8, 1.2)
		for _, k in ipairs({ "attn_q", "attn_k", "attn_v", "attn_out" }) do
			w(p .. k .. ".weight", E, E)
		end
		t(p .. "attn_q_norm.weight", { E // NH }, 0.8, 1.2)
		t(p .. "attn_k_norm.weight", { E // NH }, 0.8, 1.2)
		t(p .. "attn_post_norm.weight", { E }, 0.8, 1.2)
		t(p .. "ffn_post_norm.weight", { E }, 0.8, 1.2)
		w(p .. "ffn_gate.weight", FF, E)
		w(p .. "ffn_up.weight", FF, E)
		w(p .. "ffn_down.weight", E, FF)
	end
	local C = "clip.vision."
	local kv = {
		{ "general.architecture", gguf.STRING, "clip" },
		{ "clip.has_vision_encoder", gguf.BOOL, true },
		{ C .. "projector_type", gguf.STRING, "gemma4v" },
		{ C .. "embedding_length", gguf.UINT32, E },
		{ C .. "feed_forward_length", gguf.UINT32, FF },
		{ C .. "attention.head_count", gguf.UINT32, NH },
		{ C .. "attention.layer_norm_epsilon", gguf.FLOAT32, 1e-6 },
		{ C .. "block_count", gguf.UINT32, NL },
		{ C .. "patch_size", gguf.UINT32, S },
		{ C .. "projection_dim", gguf.UINT32, NOUT },
		{ C .. "projector.scale_factor", gguf.UINT32, M },
		{ C .. "image_mean", { gguf.ARRAY, gguf.FLOAT32 }, { 0.5, 0.5, 0.5 } },
		{ C .. "image_std", { gguf.ARRAY, gguf.FLOAT32 }, { 0.5, 0.5, 0.5 } },
	}
	return kv, ts, W
end

-- The encoder in plain Lua, on (k, y, x) bytes of a prepared image.
local function reference(W, px, w, h)
	local D, eps = E // NH, 1e-6
	local nx, ny = w // S, h // S
	local P = nx * ny
	local function matvec(name, rows, cols, v)
		local m, out = W[name], {}
		for i = 1, rows do
			local acc = 0
			for j = 1, cols do
				acc = acc + m[(i - 1) * cols + j] * v[j]
			end
			out[i] = acc
		end
		return out
	end
	local function rms(v, wt, from, count)
		from, count = from or 1, count or #v
		local ss = 0
		for i = from, from + count - 1 do
			ss = ss + v[i] * v[i]
		end
		local sc = 1 / math.sqrt(ss / count + eps)
		for i = from, from + count - 1 do
			v[i] = v[i] * sc * (wt and wt[i - from + 1] or 1)
		end
		return v
	end
	local function rope(v, from, pos)
		local half = D // 2
		for i = 0, half // 2 - 1 do
			local th = pos * 100 ^ (-2 * i / half)
			local a, b = v[from + i], v[from + i + half // 2]
			v[from + i] = a * math.cos(th) - b * math.sin(th)
			v[from + i + half // 2] = a * math.sin(th) + b * math.cos(th)
		end
	end
	local x = {}
	local pe = W["v.position_embd.weight"]
	for p = 0, P - 1 do
		local cx, cy = p % nx, p // nx
		local vec = {}
		for c = 0, 2 do
			for ky = 0, S - 1 do
				for kx = 0, S - 1 do
					local b = px[c * w * h + (cy * S + ky) * w + cx * S + kx + 1]
					vec[#vec + 1] = ((b / 255 - 0.5) / 0.5) * 2 - 1
				end
			end
		end
		local e = matvec("v.patch_embd.weight", E, 3 * S * S, vec)
		for i = 1, E do
			e[i] = e[i] + pe[cx * E + i] + pe[(NPOS + cy) * E + i]
		end
		x[p + 1] = e
	end
	for l = 0, NL - 1 do
		local pre = "v.blk." .. l .. "."
		local q, k, v = {}, {}, {}
		for p = 1, P do
			local hn = rms({ table.unpack(x[p]) }, W[pre .. "ln1.weight"])
			q[p] = matvec(pre .. "attn_q.weight", E, E, hn)
			k[p] = matvec(pre .. "attn_k.weight", E, E, hn)
			v[p] = matvec(pre .. "attn_v.weight", E, E, hn)
			local cx, cy = (p - 1) % nx, (p - 1) // nx
			for hd = 0, NH - 1 do
				local from = hd * D + 1
				rms(q[p], W[pre .. "attn_q_norm.weight"], from, D)
				rms(k[p], W[pre .. "attn_k_norm.weight"], from, D)
				rms(v[p], nil, from, D)
				for _, t in ipairs({ q[p], k[p] }) do
					rope(t, from, cx)
					rope(t, from + D // 2, cy)
				end
			end
		end
		for p = 1, P do
			local o = {}
			for hd = 0, NH - 1 do
				local sc, mx = {}, -math.huge
				for j = 1, P do
					local d = 0
					for i = 1, D do
						d = d + q[p][hd * D + i] * k[j][hd * D + i]
					end
					sc[j] = d
					mx = math.max(mx, d)
				end
				local sum = 0
				for j = 1, P do
					sc[j] = math.exp(sc[j] - mx)
					sum = sum + sc[j]
				end
				for i = 1, D do
					local acc = 0
					for j = 1, P do
						acc = acc + sc[j] / sum * v[j][hd * D + i]
					end
					o[hd * D + i] = acc
				end
			end
			local a = rms(matvec(pre .. "attn_out.weight", E, E, o), W[pre .. "attn_post_norm.weight"])
			for i = 1, E do
				a[i] = a[i] + x[p][i]
			end
			local hn = rms({ table.unpack(a) }, W[pre .. "ln2.weight"])
			local gt = matvec(pre .. "ffn_gate.weight", FF, E, hn)
			local up = matvec(pre .. "ffn_up.weight", FF, E, hn)
			for i = 1, FF do
				gt[i] = gt[i] / (1 + math.exp(-1.702 * gt[i])) * up[i]
			end
			local f = rms(matvec(pre .. "ffn_down.weight", E, FF, gt), W[pre .. "ffn_post_norm.weight"])
			for i = 1, E do
				a[i] = a[i] + f[i]
			end
			x[p].next = a
		end
		for p = 1, P do
			x[p] = x[p].next
		end
	end
	local out = {}
	for oy = 0, ny // M - 1 do
		for ox = 0, nx // M - 1 do
			local e = {}
			for i = 1, E do
				local acc = 0
				for dy = 0, M - 1 do
					for dx = 0, M - 1 do
						acc = acc + x[(oy * M + dy) * nx + ox * M + dx + 1][i]
					end
				end
				e[i] = (acc / (M * M) * math.sqrt(E) - W["v.std_bias"][i]) * W["v.std_scale"][i]
			end
			local pr = matvec("mm.input_projection.weight", NOUT, E, rms(e))
			table.move(pr, 1, NOUT, #out + 1, out)
		end
	end
	return out
end

do
	local kv, ts, W = build_mmproj()
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
	local want = reference(W, bytes(prep), prep.width, prep.height)
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
