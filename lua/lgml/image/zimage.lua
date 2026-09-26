--- The Z-Image diffusion transformer (DiT), from a gguf file.
-- It follows ZImageModel of stable-diffusion.cpp: 2x2 patches, a text
-- refiner and a noise refiner of 2 blocks each, then 30 single-stream
-- blocks over the text and image tokens with adaLN modulation from the
-- timestep, and a final adaLN layer. RoPE splits each head into 3 axes
-- (text position or image index, row, column) of 32, 48 and 48 dims.
-- Shapes are in torch order; the batch is 1.
-- @module lgml.image.zimage

local M = {}
M.__index = M

--- Patch size in latent pixels.
M.PATCH = 2
--- Text and image token counts are padded to a multiple of this.
M.SEQ_MULTI = 32
--- RoPE base.
M.THETA = 256
-- RoPE dims of each axis, in head order.
M.AXES = { 32, 48, 48 }
--- Size of the sinusoidal timestep features.
M.T_FREQ = 256

-- K and V before their f16 cast, and the FFN down-projection input,
-- scale by 1/128 and back, as the reference does against f16 overflow.
local KV_SCALE = 1 / 128
local FFN_SCALE = 1 / 128

--- Pad count that takes n to a multiple of m.
-- @tparam integer n the count
-- @tparam[opt=32] integer m the multiple
-- @treturn integer the pad count
function M.pad_of(n, m)
	m = m or M.SEQ_MULTI
	return (m - n % m) % m
end

--- RoPE positions of the text and image tokens.
-- Text token i (from 0) is at (i + 1, 0, 0), padding included. Image
-- patch (r, c) is at (T + 1, r, c), with T the padded text count. Image
-- padding is at (0, 0, 0).
-- @tparam integer n_txt the text token count, before padding
-- @tparam integer h patch rows
-- @tparam integer w patch columns
-- @treturn table { txt = { a1, a2, a3 }, img = { a1, a2, a3 } }: one
-- position list per axis, padded counts T and I, and n_img = h * w
function M.positions(n_txt, h, w)
	local T = n_txt + M.pad_of(n_txt)
	local n_img = h * w
	local I = n_img + M.pad_of(n_img)
	local txt, img = { {}, {}, {} }, { {}, {}, {} }
	for i = 1, T do
		txt[1][i], txt[2][i], txt[3][i] = i, 0, 0
	end
	for i = 1, I do
		if i <= n_img then
			img[1][i], img[2][i], img[3][i] = T + 1, (i - 1) // w, (i - 1) % w
		else
			img[1][i], img[2][i], img[3][i] = 0, 0, 0
		end
	end
	return { txt = txt, img = img, T = T, I = I, n_img = n_img }
end

--- Sinusoidal features of a timestep, cos then sin.
-- Frequency j is exp(-ln(10000) * j / 128).
-- @tparam number t the timestep, 0 to 1000
-- @treturn {number,...} `T_FREQ` values
function M.timestep_features(t)
	local half = M.T_FREQ // 2
	local out = {}
	for j = 0, half - 1 do
		local f = math.exp(-math.log(10000) * j / half)
		out[j + 1] = math.cos(t * f)
		out[j + 1 + half] = math.sin(t * f)
	end
	return out
end

local function block_weights(w, p, mod)
	local function get(n)
		return assert(w:get(p .. n), p .. n)
	end
	local B = {
		qkv = get("attention.qkv.weight"), out = get("attention.out.weight"),
		q_norm = get("attention.q_norm.weight"), k_norm = get("attention.k_norm.weight"),
		w1 = get("feed_forward.w1.weight"), w2 = get("feed_forward.w2.weight"),
		w3 = get("feed_forward.w3.weight"),
		attn_norm1 = get("attention_norm1.weight"), attn_norm2 = get("attention_norm2.weight"),
		ffn_norm1 = get("ffn_norm1.weight"), ffn_norm2 = get("ffn_norm2.weight"),
	}
	if mod then
		B.ada_w = get("adaLN_modulation.0.weight")
		B.ada_b = get("adaLN_modulation.0.bias")
	end
	return B
end

local function count(w, prefix)
	local n = 0
	while w:get(prefix .. n .. ".attention.qkv.weight") do
		n = n + 1
	end
	return n
end

--- Load the DiT.
-- The sizes come from the weight shapes.
-- @param be the lgml backend
-- @tparam string path the gguf file
-- @treturn Model the model
-- @raise when the file does not load or a weight is missing
function M.load(be, path)
	local w = assert(be:load(path))
	local function get(n)
		return assert(w:get(n), n)
	end
	local m = setmetatable({ be = be, w = w }, M)
	m.x_emb_w, m.x_emb_b = get("x_embedder.weight"), get("x_embedder.bias")
	m.cap_norm = get("cap_embedder.0.weight")
	m.cap_w, m.cap_b = get("cap_embedder.1.weight"), get("cap_embedder.1.bias")
	m.cap_pad, m.x_pad = get("cap_pad_token"), get("x_pad_token")
	m.t0_w, m.t0_b = get("t_embedder.mlp.0.weight"), get("t_embedder.mlp.0.bias")
	m.t2_w, m.t2_b = get("t_embedder.mlp.2.weight"), get("t_embedder.mlp.2.bias")
	m.final_ada_w = get("final_layer.adaLN_modulation.1.weight")
	m.final_ada_b = get("final_layer.adaLN_modulation.1.bias")
	m.final_w, m.final_b = get("final_layer.linear.weight"), get("final_layer.linear.bias")
	local dim, pp = table.unpack(m.x_emb_w:shape())
	local hd = get("layers.0.attention.q_norm.weight"):shape()[1]
	m.hp = {
		dim = dim, head_dim = hd, n_head = dim // hd,
		in_ch = pp // (M.PATCH * M.PATCH), cap_dim = m.cap_w:shape()[2],
		eps = 1e-5, qk_eps = 1e-6,
	}
	m.ctx_ref, m.noise_ref, m.layers = {}, {}, {}
	for i = 1, count(w, "context_refiner.") do
		m.ctx_ref[i] = block_weights(w, "context_refiner." .. (i - 1) .. ".", false)
	end
	for i = 1, count(w, "noise_refiner.") do
		m.noise_ref[i] = block_weights(w, "noise_refiner." .. (i - 1) .. ".", true)
	end
	for i = 1, count(w, "layers.") do
		m.layers[i] = block_weights(w, "layers." .. (i - 1) .. ".", true)
	end
	assert(#m.layers > 0, path .. ": no layers")
	return m
end

--- A loaded DiT, from `load`.
-- Fields: be, w (the weights), hp (dim, head_dim, n_head, in_ch,
-- cap_dim, eps, qk_eps), layers, ctx_ref and noise_ref.
-- @type Model

--- Close the weights.
function M:close()
	self.w:close()
end

--- Make the graph inputs for one image size and text length.
-- The leaves: x (1, C, H, W) the latent, ctx (n_txt, cap_dim) the text
-- hidden states, tfreq (1, 256) from `timestep_features`, and the
-- position and pad index leaves, which this call fills.
-- @tparam integer n_txt the text token count
-- @tparam integer H latent rows, even
-- @tparam integer W latent columns, even
-- @param[opt] x a latent leaf (1, C, H, W) to read in place of a new one;
-- `close_inputs` leaves it open
-- @treturn table the inputs: the leaves, pos (from `positions`), H, W,
-- h, w and n_txt
function M:inputs(n_txt, H, W, x)
	local be, hp, p = self.be, self.hp, M.PATCH
	assert(H % p == 0 and W % p == 0, "zimage: latent size must be even")
	local h, w = H // p, W // p
	local pos = M.positions(n_txt, h, w)
	local inp = {
		n_txt = n_txt, H = H, W = W, h = h, w = w, pos = pos,
		x = x or be:tensor("f32", { 1, hp.in_ch, H, W }), own_x = x == nil,
		ctx = be:tensor("f32", { n_txt, hp.cap_dim }),
		tfreq = be:tensor("f32", { 1, M.T_FREQ }),
		pos_txt = {}, pos_img = {}, pos_all = {},
	}
	for a = 1, 3 do
		local all = table.move(pos.img[a], 1, pos.I, pos.T + 1, table.move(pos.txt[a], 1, pos.T, 1, {}))
		inp.pos_txt[a] = be:tensor("i32", { pos.T }, pos.txt[a])
		inp.pos_img[a] = be:tensor("i32", { pos.I }, pos.img[a])
		inp.pos_all[a] = be:tensor("i32", { pos.T + pos.I }, all)
	end
	if pos.T > n_txt then
		inp.txt_pad = be:tensor("i32", { pos.T - n_txt })
		inp.txt_pad:zero()
	end
	if pos.I > pos.n_img then
		inp.img_pad = be:tensor("i32", { pos.I - pos.n_img })
		inp.img_pad:zero()
	end
	return inp
end

--- Close the leaves of inputs from `inputs`.
-- @tparam table inp the inputs
function M.close_inputs(inp)
	for _, k in ipairs({ "x", "ctx", "tfreq", "txt_pad", "img_pad" }) do
		if inp[k] and (k ~= "x" or inp.own_x) then
			inp[k]:close()
		end
	end
	for _, k in ipairs({ "pos_txt", "pos_img", "pos_all" }) do
		for _, t in ipairs(inp[k]) do
			t:close()
		end
	end
end

local function linear(g, w, b, x)
	local y = g:mul_mat(w, x)
	if b then
		y = g:add(y, b)
	end
	return y
end

local function rms(g, x, w, eps)
	return g:mul(g:rms_norm(x, eps), w)
end

--- Patchify a latent (1, C, H, W) to (h * w, 4 * C) tokens.
-- Token features are in (row in patch, column in patch, channel) order.
-- This is a plain function.
-- @static
-- @param g the graph
-- @param x the latent
-- @treturn tensor the tokens
function M.patchify(g, x)
	local _, C, H, W = table.unpack(x:shape())
	local p = M.PATCH
	local h, w = H // p, W // p
	x = g:cont(g:permute(g:reshape(x, { C * h, p, w, p }), { 1, 3, 2, 4 }))
	x = g:cont(g:permute(g:reshape(x, { C, h * w, p * p }), { 2, 3, 1 }))
	return g:reshape(x, { h * w, p * p * C })
end

--- Undo `patchify`: (h * w, 4 * C) tokens to a latent (1, C, H, W).
-- This is a plain function.
-- @static
-- @param g the graph
-- @param x the tokens, contiguous
-- @tparam integer h patch rows
-- @tparam integer w patch columns
-- @treturn tensor the latent
function M.unpatchify(g, x, h, w)
	local p = M.PATCH
	local C = x:shape()[2] // (p * p)
	x = g:cont(g:permute(g:reshape(x, { h * w, p * p, C }), { 3, 1, 2 }))
	x = g:cont(g:permute(g:reshape(x, { C * h, w, p, p }), { 1, 3, 2, 4 }))
	return g:reshape(x, { 1, C, h * p, w * p })
end

-- RoPE on each axis part of the heads: interleaved pairs, as ggml
-- "normal" mode. x is (L, H, D); pos holds one i32 leaf per axis.
local function rope(g, x, pos)
	local L, H = x:shape()[1], x:shape()[2]
	local st = x:strides()
	local out, off = nil, 0
	for a, d in ipairs(M.AXES) do
		local part = g:view(x, { L, H, d }, { st[1], st[2] }, off * 4)
		part = g:rope(part, pos[a], { n_dims = d, mode = "normal", base = M.THETA })
		out = out and g:concat(out, part, 3) or part
		off = off + d
	end
	return out
end

local function attention(g, hp, B, x, pos)
	local L, H, D = x:shape()[1], hp.n_head, hp.head_dim
	local qkv = g:reshape(g:mul_mat(B.qkv, x), { L, 3 * H, D })
	local st = qkv:strides()
	local function part(i)
		return g:view(qkv, { L, H, D }, { st[1], st[2] }, i * H * D * 4)
	end
	local q = rope(g, rms(g, part(0), B.q_norm, hp.qk_eps), pos)
	local k = rope(g, rms(g, part(1), B.k_norm, hp.qk_eps), pos)
	local v = part(2)
	k = g:cast(g:permute(g:scale(k, KV_SCALE), { 2, 1, 3 }), "f16")
	v = g:cast(g:permute(g:scale(g:cont(v), KV_SCALE), { 2, 1, 3 }), "f16")
	local o = g:flash_attn(g:permute(q, { 2, 1, 3 }), k, v, nil, 1 / math.sqrt(D) / KV_SCALE)
	o = g:scale(g:reshape(o, { L, H * D }), 1 / KV_SCALE)
	return g:mul_mat(B.out, o)
end

local function ffn(g, B, x)
	local h = g:mul(g:silu(g:mul_mat(B.w1, x)), g:mul_mat(B.w3, x))
	return g:scale(g:mul_mat(B.w2, g:scale(h, FFN_SCALE)), 1 / FFN_SCALE)
end

-- One block. temb is the timestep embedding (1, 256); the text refiner
-- blocks have no modulation and get nil.
local function block(g, hp, B, x, pos, temb)
	local eps = hp.eps
	if not temb then
		x = g:add(rms(g, attention(g, hp, B, rms(g, x, B.attn_norm1, eps), pos),
		    B.attn_norm2, eps), x)
		return g:add(rms(g, ffn(g, B, rms(g, x, B.ffn_norm1, eps)), B.ffn_norm2, eps), x)
	end
	local dim = hp.dim
	local m = linear(g, B.ada_w, B.ada_b, temb)
	local function mod(i)
		return g:view(m, { 1, dim }, nil, i * dim * 4)
	end
	local h = rms(g, x, B.attn_norm1, eps)
	h = g:add(h, g:mul(h, mod(0)))
	h = rms(g, attention(g, hp, B, h, pos), B.attn_norm2, eps)
	x = g:add(g:mul(h, g:tanh(mod(1))), x)
	h = rms(g, x, B.ffn_norm1, eps)
	h = g:add(h, g:mul(h, mod(2)))
	h = rms(g, ffn(g, B, h), B.ffn_norm2, eps)
	return g:add(g:mul(h, g:tanh(mod(3))), x)
end

-- Rows of a (1, dim) pad token for each index in ids.
local function pad_rows(g, tok, ids)
	local dim = tok:shape()[1]
	return g:get_rows(g:reshape(tok, { 1, dim }), ids)
end

--- Build the forward graph: the flow velocity for the latent.
-- The result is the model output of the reference, which negates the
-- final layer: the denoised latent is x - sigma * out.
-- @param g the graph
-- @tparam table inp inputs from `inputs`
-- @return the output (1, C, H, W)
-- @treturn table named nodes: "temb", "cap", "xemb", "txt" and "img"
-- (after the refiners), "layer0" to "layer29", and "final"
function M:forward(g, inp)
	local hp = self.hp
	local taps = {}
	local t = g:silu(linear(g, self.t0_w, self.t0_b, inp.tfreq))
	local temb = linear(g, self.t2_w, self.t2_b, t)
	taps.temb = temb

	local txt = linear(g, self.cap_w, self.cap_b, rms(g, inp.ctx, self.cap_norm, hp.eps))
	if inp.txt_pad then
		txt = g:concat(txt, pad_rows(g, self.cap_pad, inp.txt_pad), 1)
	end
	local img = linear(g, self.x_emb_w, self.x_emb_b, M.patchify(g, inp.x))
	taps.cap, taps.xemb = txt, img
	if inp.img_pad then
		img = g:concat(img, pad_rows(g, self.x_pad, inp.img_pad), 1)
	end
	for _, B in ipairs(self.ctx_ref) do
		txt = block(g, hp, B, txt, inp.pos_txt, nil)
	end
	for _, B in ipairs(self.noise_ref) do
		img = block(g, hp, B, img, inp.pos_img, temb)
	end
	taps.txt, taps.img = txt, img
	local x = g:concat(txt, img, 1)
	for i, B in ipairs(self.layers) do
		x = block(g, hp, B, x, inp.pos_all, temb)
		taps["layer" .. (i - 1)] = x
	end
	local scale = linear(g, self.final_ada_w, self.final_ada_b, g:silu(temb))
	x = g:norm(x, 1e-6)
	x = g:add(x, g:mul(x, scale))
	x = linear(g, self.final_w, self.final_b, x)
	taps.final = x
	local pos = inp.pos
	local cols = x:shape()[2]
	x = g:view(x, { pos.n_img, cols }, nil, pos.T * cols * 4)
	return g:scale(M.unpatchify(g, x, inp.h, inp.w), -1), taps
end

return M
