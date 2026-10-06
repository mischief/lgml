-- A tiny qwen3vl_merger mmproj for tests, and the encoder in plain Lua.
local gguf = require("gguf")

local M = {}
-- E 16, 2 heads of 8, patch 2, merge 2, a 4 x 4 position table, and a
-- deepstack layer 0. nout is the text embedding size.
local E, NH, S, NL, SIDE, FF = 16, 2, 2, 2, 4, 24
M.E, M.PATCH, M.MERGE = E, S, 2

local function rng(seed)
	local x = seed
	return function()
		x = x * 6364136223846793005 + 1442695040888963407
		return (x >> 11) / 2.0 ^ 53
	end
end

--- kv and tensors of the mmproj, and the weights by name.
-- @tparam integer nout the text embedding size
function M.build(nout)
	local r = rng(777)
	local W, ts = {}, {}
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
	local function lin(name, rows, cols)
		local sc = math.sqrt(3 / cols)
		t(name .. ".weight", { rows, cols }, -sc, sc)
		t(name .. ".bias", { rows }, -0.1, 0.1)
	end
	local function norm(name, d)
		t(name .. ".weight", { d }, 0.8, 1.2)
		t(name .. ".bias", { d }, -0.1, 0.1)
	end
	t("v.patch_embd.weight", { E, 3, S, S }, -0.4, 0.4)
	t("v.patch_embd.weight.1", { E, 3, S, S }, -0.4, 0.4)
	t("v.patch_embd.bias", { E }, -0.1, 0.1)
	t("v.position_embd.weight", { SIDE * SIDE, E }, -0.3, 0.3)
	for i = 0, NL - 1 do
		local p = "v.blk." .. i .. "."
		norm(p .. "ln1", E)
		norm(p .. "ln2", E)
		lin(p .. "attn_qkv", 3 * E, E)
		lin(p .. "attn_out", E, E)
		lin(p .. "ffn_up", FF, E)
		lin(p .. "ffn_down", E, FF)
	end
	norm("v.deepstack.0.norm", 4 * E)
	lin("v.deepstack.0.fc1", 4 * E, 4 * E)
	lin("v.deepstack.0.fc2", nout, 4 * E)
	norm("v.post_ln", E)
	lin("mm.0", 4 * E, 4 * E)
	lin("mm.2", nout, 4 * E)
	local C = "clip.vision."
	local kv = {
		{ "general.architecture", gguf.STRING, "clip" },
		{ "clip.has_vision_encoder", gguf.BOOL, true },
		{ "clip.projector_type", gguf.STRING, "qwen3vl_merger" },
		{ "clip.use_gelu", gguf.BOOL, true },
		{ C .. "embedding_length", gguf.UINT32, E },
		{ C .. "feed_forward_length", gguf.UINT32, FF },
		{ C .. "attention.head_count", gguf.UINT32, NH },
		{ C .. "attention.layer_norm_epsilon", gguf.FLOAT32, 1e-6 },
		{ C .. "block_count", gguf.UINT32, NL },
		{ C .. "patch_size", gguf.UINT32, S },
		{ C .. "projection_dim", gguf.UINT32, nout },
		{ C .. "spatial_merge_size", gguf.UINT32, 2 },
		{ C .. "is_deepstack_layers", { gguf.ARRAY, gguf.BOOL }, { true, false } },
		{ C .. "image_mean", { gguf.ARRAY, gguf.FLOAT32 }, { 0.5, 0.5, 0.5 } },
		{ C .. "image_std", { gguf.ARRAY, gguf.FLOAT32 }, { 0.5, 0.5, 0.5 } },
	}
	return kv, ts, W
end

local function gelu(x)
	return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ^ 3)))
end

--- The rows of a prepared image: each the embedding, then the
-- deepstack feature, as a flat list.
-- @tparam table W the weights from `build`
-- @tparam {integer,...} px the image bytes in (k, y, x) order
-- @tparam integer w the width
-- @tparam integer h the height
-- @tparam integer nout the text embedding size
function M.reference(W, px, w, h, nout)
	local D, eps = E // NH, 1e-6
	local nx, ny = w // S, h // S
	local function lin(name, rows, cols, v)
		local m, b, out = W[name .. ".weight"], W[name .. ".bias"], {}
		for i = 1, rows do
			local acc = b and b[i] or 0
			for j = 1, cols do
				acc = acc + m[(i - 1) * cols + j] * v[j]
			end
			out[i] = acc
		end
		return out
	end
	local function ln(v, name)
		local n, mean, var = #v, 0, 0
		for i = 1, n do
			mean = mean + v[i] / n
		end
		for i = 1, n do
			var = var + (v[i] - mean) ^ 2 / n
		end
		local o, wt, b = {}, W[name .. ".weight"], W[name .. ".bias"]
		for i = 1, n do
			o[i] = (v[i] - mean) / math.sqrt(var + eps) * wt[i] + b[i]
		end
		return o
	end
	-- Bilinear position table with the corners aligned.
	local pe = W["v.position_embd.weight"]
	local function pos(y, x, i)
		local sy = ny > 1 and y * (SIDE - 1) / (ny - 1) or 0
		local sx = nx > 1 and x * (SIDE - 1) / (nx - 1) or 0
		local y0, x0 = math.floor(sy), math.floor(sx)
		local y1, x1 = math.min(y0 + 1, SIDE - 1), math.min(x0 + 1, SIDE - 1)
		local fy, fx = sy - y0, sx - x0
		local function at(yy, xx)
			return pe[(yy * SIDE + xx) * E + i]
		end
		return (at(y0, x0) * (1 - fx) + at(y0, x1) * fx) * (1 - fy) +
		    (at(y1, x0) * (1 - fx) + at(y1, x1) * fx) * fy
	end
	-- Patches in merge order: each 2 x 2 block together.
	local x, py, pxs = {}, {}, {}
	for by = 0, ny - 1, 2 do
		for bx = 0, nx - 1, 2 do
			for dy = 0, 1 do
				for dx = 0, 1 do
					local cy, cx = by + dy, bx + dx
					local vec = {}
					for c = 0, 2 do
						for ky = 0, S - 1 do
							for kx = 0, S - 1 do
								local b = px[c * w * h + (cy * S + ky) * w + cx * S + kx + 1]
								vec[#vec + 1] = (b / 255 - 0.5) / 0.5
							end
						end
					end
					local e = {}
					local k0, k1 = W["v.patch_embd.weight"], W["v.patch_embd.weight.1"]
					for i = 1, E do
						local acc = W["v.patch_embd.bias"][i]
						for j = 1, 3 * S * S do
							acc = acc + (k0[(i - 1) * 3 * S * S + j] + k1[(i - 1) * 3 * S * S + j]) * vec[j]
						end
						e[i] = acc + pos(cy, cx, i)
					end
					x[#x + 1], py[#x + 1], pxs[#x + 1] = e, cy, cx
				end
			end
		end
	end
	local P = #x
	-- Rotate halves of a head: y on the first quarter of pairs, x on the
	-- second.
	local function rope(v, from, y, xx)
		local half = D // 2
		for k = 0, half - 1 do
			local th
			if k < D // 4 then
				th = y * 10000 ^ (-2 * k / half)
			else
				th = xx * 10000 ^ (-2 * (k - D // 4) / half)
			end
			local a, b = v[from + k], v[from + k + half]
			v[from + k] = a * math.cos(th) - b * math.sin(th)
			v[from + k + half] = a * math.sin(th) + b * math.cos(th)
		end
	end
	local feats = {}
	for l = 0, NL - 1 do
		local pre = "v.blk." .. l .. "."
		local q, k, v = {}, {}, {}
		for p = 1, P do
			local qkv = lin(pre .. "attn_qkv", 3 * E, E, ln(x[p], pre .. "ln1"))
			q[p] = table.move(qkv, 1, E, 1, {})
			k[p] = table.move(qkv, E + 1, 2 * E, 1, {})
			v[p] = table.move(qkv, 2 * E + 1, 3 * E, 1, {})
			for hd = 0, NH - 1 do
				rope(q[p], hd * D + 1, py[p], pxs[p])
				rope(k[p], hd * D + 1, py[p], pxs[p])
			end
		end
		local nxt = {}
		for p = 1, P do
			local o = {}
			for hd = 0, NH - 1 do
				local sc, mx = {}, -math.huge
				for j = 1, P do
					local d = 0
					for i = 1, D do
						d = d + q[p][hd * D + i] * k[j][hd * D + i]
					end
					sc[j] = d / math.sqrt(D)
					mx = math.max(mx, sc[j])
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
			local a = lin(pre .. "attn_out", E, E, o)
			for i = 1, E do
				a[i] = a[i] + x[p][i]
			end
			local u = lin(pre .. "ffn_up", FF, E, ln(a, pre .. "ln2"))
			for i = 1, FF do
				u[i] = gelu(u[i])
			end
			local f = lin(pre .. "ffn_down", E, FF, u)
			for i = 1, E do
				a[i] = a[i] + f[i]
			end
			nxt[p] = a
		end
		x = nxt
		if W["v.deepstack." .. l .. ".norm.weight"] then
			local d = "v.deepstack." .. l .. "."
			for g = 0, P // 4 - 1 do
				local cat = {}
				for j = 1, 4 do
					table.move(x[g * 4 + j], 1, E, #cat + 1, cat)
				end
				local u = lin(d .. "fc1", 4 * E, 4 * E, ln(cat, d .. "norm"))
				for i = 1, 4 * E do
					u[i] = gelu(u[i])
				end
				feats[g + 1] = lin(d .. "fc2", nout, 4 * E, u)
			end
		end
	end
	local out = {}
	for g = 0, P // 4 - 1 do
		local cat = {}
		for j = 1, 4 do
			table.move(ln(x[g * 4 + j], "v.post_ln"), 1, E, #cat + 1, cat)
		end
		local u = lin("mm.0", 4 * E, 4 * E, cat)
		for i = 1, 4 * E do
			u[i] = gelu(u[i])
		end
		local row = lin("mm.2", nout, 4 * E, u)
		table.move(row, 1, nout, #out + 1, out)
		table.move(feats[g + 1], 1, nout, #out + 1, out)
	end
	return out
end

return M
