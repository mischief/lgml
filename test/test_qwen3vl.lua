-- Qwen3-VL image input on the CPU: multi-section rope, the position
-- table resize, the bicubic resize, the qwen3vl encoder on a tiny random
-- mmproj against a plain Lua forward, and M-RoPE passes of a tiny
-- qwen35 model. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local vision = require("lgml.vision")
local transformer = require("lgml.transformer")
local gguf = require("gguf")
local tiny = require("tiny")
local vtiny = require("mmproj_qwen3vl_tiny")

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

local function rand(count, seed)
	local r, v = tiny.rng(seed), {}
	for i = 1, count do
		v[i] = r() * 2 - 1
	end
	return v
end

-- Run one op on a fresh graph and get its values.
local function run(fn)
	local g <close> = be:graph()
	local out = fn(g)
	g:compute(out)
	return out:get()
end

-- Rope: neox pairs (k, k + n_dims / 2) of a head, one angle per pair.
local function rope_ref(v, T, H, D, n_dims, angle)
	local out = table.move(v, 1, #v, 1, {})
	for t = 0, T - 1 do
		for h = 0, H - 1 do
			local o = (t * H + h) * D
			for k = 0, n_dims // 2 - 1 do
				local th = angle(t, k)
				local a, b = v[o + k + 1], v[o + k + n_dims // 2 + 1]
				out[o + k + 1] = a * math.cos(th) - b * math.sin(th)
				out[o + k + n_dims // 2 + 1] = a * math.sin(th) + b * math.cos(th)
			end
		end
	end
	return out
end

-- Multi-section rope against neox and plain Lua.
do
	local T, H, D, R, base = 5, 2, 16, 8, 1e4
	local x = rand(T * H * D, 11)
	local sec = { 2, 1, 1, 0 }
	local p, p4 = {}, {}
	for t = 1, T do
		p[t] = t + 2
	end
	for s = 0, 3 do
		for t = 1, T do
			p4[s * T + t] = p[t]
		end
	end
	local function go(mode, pos)
		return run(function(g)
			local a = be:tensor("f32", { T, H, D }, x)
			return g:rope(a, be:tensor("i32", { #pos }, pos), { mode = mode, n_dims = R, base = base,
				sections = sec })
		end)
	end
	local neox, im = go("neox", p), go("imrope", p4)
	ok(maxdiff(neox, im) < 1e-6, "imrope with equal positions is neox")
	-- Distinct t, h and w: interleaved sections pick by pair index.
	local q = {}
	for t = 0, T - 1 do
		q[t + 1], q[T + t + 1], q[2 * T + t + 1], q[3 * T + t + 1] = t, 10 + 2 * t, 20 - t, 0
	end
	local want = rope_ref(x, T, H, D, R, function(t, k)
		local s = k % 4
		local m = s % 3
		local which = (m == 1 and s < 3) and 1 or (m == 2 and s < 3) and 2 or (m == 0 and s < 6) and 0 or 3
		return q[which * T + t + 1] * base ^ (-2 * k / R)
	end)
	ok(maxdiff(go("imrope", q), want) < 1e-5, "imrope matches a plain rope")
	ok(not pcall(go, "imrope", p), "imrope needs 4 positions per token")
	ok(not pcall(run, function(g)
		return g:rope(be:tensor("f32", { T, H, D }, x), be:tensor("i32", { 4 * T }, p4),
		    { mode = "imrope", n_dims = R })
	end), "a multi-section rope needs sections")
	-- Vision: y on the first quarter of pairs, x on the second.
	local VD = 8
	local vx = rand(T * H * VD, 12)
	local vp = {}
	for t = 0, T - 1 do
		vp[t + 1], vp[T + t + 1], vp[2 * T + t + 1], vp[3 * T + t + 1] = t // 2, t % 3, t // 2, t % 3
	end
	local got = run(function(g)
		return g:rope(be:tensor("f32", { T, H, VD }, vx), be:tensor("i32", { 4 * T }, vp), {
			mode = "vision", n_dims = VD // 2, sections = { 2, 2, 2, 2 }, base = base })
	end)
	want = rope_ref(vx, T, H, VD, VD, function(t, k)
		if k < VD // 4 then
			return vp[t + 1] * base ^ (-2 * k / (VD // 2))
		end
		return vp[T + t + 1] * base ^ (-2 * (k - VD // 4) / (VD // 2))
	end)
	ok(maxdiff(got, want) < 1e-5, "vision rope matches a plain rope")
end

-- Bilinear interpolation with aligned corners.
do
	local C, h, w, oh, ow = 2, 3, 4, 5, 7
	local x = rand(C * h * w, 13)
	local got = run(function(g)
		return g:interpolate(be:tensor("f32", { 1, C, h, w }, x), oh, ow, "bilinear", true)
	end)
	local want = {}
	for c = 0, C - 1 do
		for y = 0, oh - 1 do
			for xx = 0, ow - 1 do
				local sy, sx = y * (h - 1) / (oh - 1), xx * (w - 1) / (ow - 1)
				local y0, x0 = math.floor(sy), math.floor(sx)
				local y1, x1 = math.min(y0 + 1, h - 1), math.min(x0 + 1, w - 1)
				local fy, fx = sy - y0, sx - x0
				local function at(a, b)
					return x[c * h * w + a * w + b + 1]
				end
				want[#want + 1] = (at(y0, x0) * (1 - fx) + at(y0, x1) * fx) * (1 - fy) +
				    (at(y1, x0) * (1 - fx) + at(y1, x1) * fx) * fy
			end
		end
	end
	ok(#got == C * oh * ow and maxdiff(got, want) < 1e-5, "interpolate aligns the corners")
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

-- Pillow's bicubic resample on (k, y, x) bytes: taps per axis, 22-bit
-- fixed-point weights, rows first.
local function bicubic(src, w, h, tw, th)
	local function cubic(x)
		x = math.abs(x)
		if x < 1 then
			return (1.5 * x - 2.5) * x * x + 1
		elseif x < 2 then
			return (((x - 5) * x + 8) * x - 4) * -0.5
		end
		return 0
	end
	local function taps(inn, out)
		local scale = inn / out
		local fs = math.max(scale, 1)
		local support = 2 * fs
		local res = {}
		for xx = 0, out - 1 do
			local center = (xx + 0.5) * scale
			-- C casts toward zero.
			local function trunc(v)
				return math.tointeger(v < 0 and math.ceil(v) or math.floor(v))
			end
			local lo = math.max(0, trunc(center - support + 0.5))
			local hi = math.min(inn, trunc(center + support + 0.5))
			local wt, sum = {}, 0
			for x = lo, hi - 1 do
				wt[#wt + 1] = cubic((x - center + 0.5) / fs)
				sum = sum + wt[#wt]
			end
			for i = 1, #wt do
				-- Round half away from zero, then truncate, as Pillow.
				local v = sum ~= 0 and wt[i] / sum or wt[i]
				local f = v * 2 ^ 22 + (v < 0 and -0.5 or 0.5)
				wt[i] = math.tointeger(f < 0 and math.ceil(f) or math.floor(f))
			end
			res[xx + 1] = { lo = lo, w = wt }
		end
		return res
	end
	local function clip(acc)
		acc = acc // (1 << 22)
		return math.max(0, math.min(255, acc))
	end
	local cur, cw = src, w
	if tw ~= w then
		local tx, out = taps(w, tw), {}
		for k = 0, 2 do
			for y = 0, h - 1 do
				for x = 0, tw - 1 do
					local acc = 1 << 21
					for j, wt in ipairs(tx[x + 1].w) do
						acc = acc + cur[k * w * h + y * w + tx[x + 1].lo + j] * wt
					end
					out[k * tw * h + y * tw + x + 1] = clip(acc)
				end
			end
		end
		cur, cw = out, tw
	end
	if th == h then
		return cur
	end
	local ty, out = taps(h, th), {}
	for k = 0, 2 do
		for y = 0, th - 1 do
			for x = 0, cw - 1 do
				local acc = 1 << 21
				for j, wt in ipairs(ty[y + 1].w) do
					acc = acc + cur[k * cw * h + (ty[y + 1].lo + j - 1) * cw + x + 1] * wt
				end
				out[k * cw * th + y * cw + x + 1] = clip(acc)
			end
		end
	end
	return out
end

do
	local im = make(13, 9, function(x, y, k)
		return (x * 37 + y * 101 + k * 59) * 977 % 256
	end)
	local src = bytes(im)
	local up = im:resize(29, 17, "bicubic")
	ok(maxdiff(bytes(up), bicubic(src, 13, 9, 29, 17)) == 0, "bicubic grow matches Pillow's resample")
	local down = im:resize(6, 4, "bicubic")
	ok(maxdiff(bytes(down), bicubic(src, 13, 9, 6, 4)) == 0, "bicubic shrink matches Pillow's resample")
	local wide = im:resize(20, 9, "bicubic")
	ok(maxdiff(bytes(wide), bicubic(src, 13, 9, 20, 9)) == 0, "bicubic on one axis keeps the rows")
	local flat = make(10, 10, function()
		return 77
	end)
	local f = bytes(flat:resize(7, 13, "bicubic"))
	ok(maxdiff(f, (function()
		local t = {}
		for i = 1, #f do
			t[i] = 77
		end
		return t
	end)()) == 0, "bicubic keeps a flat image flat")
end

-- The encoder on a tiny random mmproj against a plain Lua forward.
do
	local NOUT = 6
	local kv, ts, W = vtiny.build(NOUT)
	local path = os.tmpname()
	gguf.write(path, kv, ts)
	local enc = vision.load(be, path, { min_tokens = 1, max_tokens = 16, flash_attn = false })
	ok(enc.projector == "qwen3vl_merger" and enc.hp.n_out == NOUT and enc.hp.n_deepstack == 1 and
	    enc.causal and enc.mrope, "the mmproj loads as qwen3vl_merger with a deepstack layer")
	local im = make(12, 8, function(x, y, k)
		return (x * 29 + y * 53 + k * 71) % 256
	end)
	local prep = assert(enc:prepare(im))
	local gx, gy = enc:grid(prep)
	ok(prep.width == 12 and prep.height == 8 and gx == 3 and gy == 2 and enc:tokens(prep) == 6,
	    ("an image prepares to %dx%d, a %dx%d grid"):format(prep.width, prep.height, gx, gy))
	local want = vtiny.reference(W, bytes(prep), prep.width, prep.height, NOUT)
	local function values(buf)
		local v = {}
		for i = 1, #buf // 4 do
			v[i] = string.unpack("<f", buf:bytes((i - 1) * 4, 4))
		end
		return v
	end
	local buf, count = assert(enc:encode(im))
	local got = values(buf)
	ok(count == 6 and #got == 6 * 2 * NOUT, "encode gives 6 rows of the embedding and a feature")
	local d = maxdiff(got, want)
	ok(d < 2e-3, ("the encoder matches a plain forward (max diff %.3g)"):format(d))
	local fa = vision.load(be, path, { min_tokens = 1, max_tokens = 16 })
	d = maxdiff(values((assert(fa:encode(im)))), want)
	ok(fa.flash_attn and d < 3e-2, ("flash attention stays close (max diff %.3g)"):format(d))
	-- A larger image: the position table grows to the grid.
	local big = make(20, 12, function(x, y, k)
		return (x * 13 + y * 7 + k * 31) % 256
	end)
	local bp = assert(enc:prepare(big))
	want = vtiny.reference(W, bytes(bp), bp.width, bp.height, NOUT)
	d = maxdiff(values((assert(enc:encode(big)))), want)
	ok(bp.width == 20 and bp.height == 12 and d < 2e-3,
	    ("a 5x3 grid matches a plain forward (max diff %.3g)"):format(d))
	ok(enc:measure() > 0, "measure plans the largest image")
	os.remove(path)
end

-- A tiny qwen35: text passes with and without rope sections, M-RoPE
-- positions and deepstack features.
do
	local secs = { "qwen35.rope.dimension_sections", { gguf.ARRAY, gguf.INT32 }, { 2, 1, 1, 0 } }
	local pa, pb = os.tmpname(), os.tmpname()
	tiny.write(pa, "qwen35", { hybrid = true, kv = { secs } })
	tiny.write(pb, "qwen35", { hybrid = true })
	local ma = transformer.load(be, pa)
	local mb = transformer.load(be, pb)
	ok(ma.hp.sections and ma.hp.rope_multi == "imrope" and not mb.hp.sections, "sections load")
	local ids = { 72, 101, 108, 108, 111, 32, 119, 111 }
	-- The logits of list after the tokens of before; delta and opts
	-- apply to list.
	local function logits(m, list, opts, delta, before)
		local cache = m:cache(64)
		if before then
			local inp = m:inputs(before, cache)
			local g <close> = be:graph()
			g:compute((m:forward(g, inp)))
			cache.n_past = #before
		end
		cache.pos_delta = delta
		local inp = m:inputs(list, cache, opts)
		local g <close> = be:graph()
		local out = m:forward(g, inp)
		g:compute(out)
		return out:get()
	end
	local la, lb = logits(ma, ids), logits(mb, ids)
	ok(maxdiff(la, lb) == 0, "text logits do not change with rope sections")
	local mpos = {}
	for s = 0, 3 do
		for t = 1, #ids do
			mpos[s * #ids + t] = t - 1
		end
	end
	local lm = logits(ma, ids, { mpos = mpos })
	ok(maxdiff(lm, la) < 1e-4, ("equal M-RoPE positions give the text logits (max diff %.2g)")
	    :format(maxdiff(lm, la)))
	-- After 4 tokens, a delta of -2 puts the next ones at 2 and up.
	local head, tail = { 72, 101, 108, 108 }, { 111, 32, 119, 111 }
	local shifted = {}
	for s = 0, 3 do
		for t = 1, #tail do
			shifted[s * #tail + t] = t + 1
		end
	end
	local ld = logits(ma, tail, nil, -2, head)
	local d1 = maxdiff(logits(ma, tail, { mpos = shifted }, nil, head), ld)
	local d2 = maxdiff(logits(ma, tail, nil, nil, head), ld)
	ok(d1 < 1e-4 and d2 > 1e-3, ("a position delta moves text positions (%.2g, %.2g)"):format(d1, d2))
	ok(not pcall(logits, mb, ids, { mpos = mpos }), "M-RoPE positions need rope sections")
	-- Deepstack: layer 1 adds the feature of each row.
	local E = ma.hp.n_embd
	local function hidden(feat)
		local rows = {}
		local x = rand(#ids * E, 21)
		for t = 0, #ids - 1 do
			for i = 1, E do
				rows[#rows + 1] = string.pack("<f", x[t * E + i])
			end
			for i = 1, E do
				rows[#rows + 1] = string.pack("<f", feat(t, i))
			end
		end
		local s = table.concat(rows)
		local buf = lgml.hostbuf(#s)
		buf:set(0, s)
		return logits(ma, ids, { embd = buf, deepstack = 1, hidden = 1, mpos = mpos })
	end
	local h0 = hidden(function()
		return 0
	end)
	local h1 = hidden(function(t, i)
		return (t + 1) * 0.01 * i
	end)
	local worst = 0
	for t = 0, #ids - 1 do
		for i = 1, E do
			worst = math.max(worst, math.abs(h1[t * E + i] - h0[t * E + i] - (t + 1) * 0.01 * i))
		end
	end
	ok(#h1 == #ids * E and worst < 1e-5, "a deepstack feature adds to the output of its layer")
	os.remove(pa)
	os.remove(pb)
end

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
