-- A tiny gemma4v mmproj for tests, and the encoder in plain Lua.
local gguf = require("gguf")

local M = {}
-- E 16, 2 heads of 8, patch 2, merge 3. nout is the text embedding size.
local E, NH, S, MG, NL, NPOS, FF = 16, 2, 2, 3, 2, 12, 24
M.E, M.PATCH, M.MERGE = E, S, MG

-- Seeded uniform numbers in [0, 1), the same on every host.
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
	w("mm.input_projection.weight", nout, E)
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
		{ C .. "projection_dim", gguf.UINT32, nout },
		{ C .. "projector.scale_factor", gguf.UINT32, MG },
		{ C .. "image_mean", { gguf.ARRAY, gguf.FLOAT32 }, { 0.5, 0.5, 0.5 } },
		{ C .. "image_std", { gguf.ARRAY, gguf.FLOAT32 }, { 0.5, 0.5, 0.5 } },
	}
	return kv, ts, W
end

-- The encoder in plain Lua, on (k, y, x) bytes of a prepared image.
--- The embeddings (a flat list) of a prepared image.
-- @tparam table W the weights from `build`
-- @tparam {integer,...} px the image bytes in (k, y, x) order
-- @tparam integer w the width
-- @tparam integer h the height
-- @tparam integer nout the text embedding size
function M.reference(W, px, w, h, nout)
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
	for oy = 0, ny // MG - 1 do
		for ox = 0, nx // MG - 1 do
			local e = {}
			for i = 1, E do
				local acc = 0
				for dy = 0, MG - 1 do
					for dx = 0, MG - 1 do
						acc = acc + x[(oy * MG + dy) * nx + ox * MG + dx + 1][i]
					end
				end
				e[i] = (acc / (MG * MG) * math.sqrt(E) - W["v.std_bias"][i]) * W["v.std_scale"][i]
			end
			local pr = matvec("mm.input_projection.weight", nout, E, rms(e))
			table.move(pr, 1, nout, #out + 1, out)
		end
	end
	return out
end

return M
