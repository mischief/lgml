--- Qwen3-TTS-Tokenizer-12Hz decoder.
-- The decoder turns 16 RVQ codes per 12.5 Hz frame into 24 kHz samples.
-- It has an RVQ sum, causal convs, an 8-layer sliding-window
-- transformer, 2 ConvNeXt upsample stages and 4 SnakeBeta decoder blocks.
-- A loaded codec has the hyperparameters in hp; hp.rate is the sample
-- rate and hp.hop is samples per frame.
-- @classmod lgml.qwen3tts.codec
local lgml = require("lgml")

local M = {}
M.__index = M

--- Frames per decode graph.
M.CHUNK = 150
--- Left context frames for a chunk after the first.
M.CTX = 96

local DILATIONS = { 1, 3, 9 }

local function hparams(meta)
	local function k(s)
		return meta["qwen3tts_codec.dec." .. s]
	end
	return {
		n_layer = k("n_layers"), n_embd = k("d_model"), n_head = k("n_heads"),
		head_dim = k("head_dim"), eps = k("rms_norm_eps"), rope_base = k("rope_theta"),
		window = k("sliding_window"), n_q = k("n_quantizers"),
		rates = k("upsample_rates"), up = k("upsampling_ratios"),
		hop = meta["qwen3tts_codec.decode_upsample"],
		rate = meta["qwen3tts_codec.output_sample_rate"],
	}
end

-- Weight preparation graph. Each prep writes into a new leaf.
local Prep = {}
Prep.__index = Prep

function Prep:add(node, type)
	local leaf = self.be:tensor(type or node:type(), node:shape())
	self.outs[#self.outs + 1] = self.g:cpy(node, leaf)
	return leaf
end

-- Conv weight (OC, IC, K) to (K, OC, IC).
function Prep:conv(w)
	local s = w:shape()
	if #s == 2 then
		w = self.g:reshape(w, { 1, s[1], s[2] })
	end
	return self:add(self.g:cont(self.g:permute(w, { 3, 1, 2 })))
end

-- Transposed conv weight (IC, OC, K) to (OC, K, IC).
function Prep:tconv(w)
	return self:add(self.g:cont(self.g:permute(w, { 2, 3, 1 })))
end

-- Depthwise weight (C, 1, K) to f32 (K, C).
function Prep:dw(w)
	local s = w:shape()
	local g = self.g
	return self:add(g:cont(g:transpose(g:reshape(w, { s[1], s[3] }))), "f32")
end

local function load_weights(m, w, prep)
	local function get(name)
		return assert(w:get("codec.dec." .. name), name)
	end
	m.rvq_first = get("rvq_first.codebook")
	m.rvq_first_out = prep:conv(get("rvq_first.out_proj_w"))
	m.rvq_rest, m.rvq_rest_out = {}, prep:conv(get("rvq_rest.out_proj_w"))
	for q = 0, m.hp.n_q - 2 do
		m.rvq_rest[q + 1] = get("rvq_rest." .. q .. ".codebook")
	end
	m.pre_w, m.pre_b = prep:conv(get("pre_conv_w")), get("pre_conv_b")
	m.in_proj_w, m.in_proj_b = get("xfmr.in_proj_w"), get("xfmr.in_proj_b")
	m.out_proj_w, m.out_proj_b = get("xfmr.out_proj_w"), get("xfmr.out_proj_b")
	m.xnorm = get("xfmr.norm_w")
	m.layers = {}
	for i = 0, m.hp.n_layer - 1 do
		local p = "xfmr.blk." .. i .. "."
		local L = {}
		for _, k in ipairs({ "attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v",
		    "attn_o", "attn_ls", "ffn_gate", "ffn_up", "ffn_down", "ffn_ls" }) do
			L[k] = get(p .. k .. "_w")
		end
		m.layers[i + 1] = L
	end
	m.ups = {}
	for s = 0, #m.hp.up - 1 do
		local p = "up." .. s .. "."
		local u = { tconv = prep:tconv(get(p .. "tconv_w")), tconv_b = get(p .. "tconv_b"),
		    dw = prep:dw(get(p .. "cnx.dw_w")), dw_b = get(p .. "cnx.dw_b") }
		for _, k in ipairs({ "norm_w", "norm_b", "pw1_w", "pw1_b", "pw2_w", "pw2_b", "gamma" }) do
			u[k] = get(p .. "cnx." .. k)
		end
		m.ups[s + 1] = u
	end
	m.in_w, m.in_b = prep:conv(get("in_conv_w")), get("in_conv_b")
	m.blocks = {}
	for b = 0, #m.hp.rates - 1 do
		local p = "blk." .. b .. "."
		local blk = { a = get(p .. "snake_a"), b = get(p .. "snake_b"),
		    tconv = prep:tconv(get(p .. "tconv_w")), tconv_b = get(p .. "tconv_b"), res = {} }
		for u = 0, 2 do
			local r = p .. "res." .. u .. "."
			blk.res[u + 1] = {
				a1 = get(r .. "act1_a"), b1 = get(r .. "act1_b"),
				a2 = get(r .. "act2_a"), b2 = get(r .. "act2_b"),
				w1 = prep:conv(get(r .. "conv1_w")), b1c = get(r .. "conv1_b"),
				w2 = prep:conv(get(r .. "conv2_w")), b2c = get(r .. "conv2_b"),
			}
		end
		m.blocks[b + 1] = blk
	end
	m.out_a, m.out_b = get("out_snake_a"), get("out_snake_b")
	m.out_w, m.out_bias = prep:conv(get("out_conv_w")), get("out_conv_b")
end

--- Load a codec gguf and prepare the conv weights.
-- @tparam lgml.backend be backend
-- @string path gguf path
-- @treturn lgml.qwen3tts.codec the codec
-- @raise when the file is not a qwen3tts_tokenizer gguf
-- @static
function M.load(be, path)
	local info = assert(lgml.gguf(path))
	if info.meta["general.architecture"] ~= "qwen3tts_tokenizer" then
		error(path .. ": not a qwen3tts_tokenizer gguf")
	end
	local w = assert(be:load(path))
	local m = setmetatable({ be = be, w = w, hp = hparams(info.meta) }, M)
	local g <close> = be:graph()
	local prep = setmetatable({ be = be, g = g, outs = {} }, Prep)
	load_weights(m, w, prep)
	g:compute(table.unpack(prep.outs))
	return m
end

--- Free the weights.
function M:close()
	self.w:close()
end

-- Sliding-window causal mask (n, n), f16.
local function window_mask(be, n, window)
	local v = {}
	for i = 0, n - 1 do
		for j = 0, n - 1 do
			v[i * n + j + 1] = (j <= i and i - j < window) and 0 or -math.huge
		end
	end
	return be:tensor("f16", { n, n }, v)
end

local function transformer(self, g, x, pos, mask)
	local hp = self.hp
	local n, H, D = x:shape()[1], hp.n_head, hp.head_dim
	local rope = { mode = "neox", base = hp.rope_base }
	x = g:add(g:mul_mat(self.in_proj_w, x), self.in_proj_b)
	for _, L in ipairs(self.layers) do
		local cur = g:mul(g:rms_norm(x, hp.eps), L.attn_norm)
		local q = g:rope(g:reshape(g:mul_mat(L.attn_q, cur), { n, H, D }), pos, rope)
		local k = g:rope(g:reshape(g:mul_mat(L.attn_k, cur), { n, H, D }), pos, rope)
		local v = g:reshape(g:mul_mat(L.attn_v, cur), { n, H, D })
		local o = g:flash_attn(g:permute(q, { 2, 1, 3 }), g:permute(k, { 2, 1, 3 }),
		    g:permute(v, { 2, 1, 3 }), mask, 1 / math.sqrt(D))
		o = g:mul_mat(L.attn_o, g:reshape(o, { n, H * D }))
		x = g:add(g:mul(o, L.attn_ls), x)
		cur = g:mul(g:rms_norm(x, hp.eps), L.ffn_norm)
		local ff = g:mul(g:silu(g:mul_mat(L.ffn_gate, cur)), g:mul_mat(L.ffn_up, cur))
		x = g:add(g:mul(g:mul_mat(L.ffn_down, ff), L.ffn_ls), x)
	end
	x = g:mul(g:rms_norm(x, hp.eps), self.xnorm)
	return g:add(g:mul_mat(self.out_proj_w, x), self.out_proj_b)
end

local function convnext(g, x, u)
	local h = g:qwen3tts_dwconv(x, u.dw, u.dw_b)
	h = g:add(g:mul(g:norm(h, 1e-5), u.norm_w), u.norm_b)
	h = g:gelu_erf(g:add(g:mul_mat(u.pw1_w, h), u.pw1_b))
	h = g:add(g:mul_mat(u.pw2_w, h), u.pw2_b)
	return g:add(g:mul(h, u.gamma), x)
end

local function decoder_block(g, x, blk, rate)
	x = g:qwen3tts_tconv(g:qwen3tts_snake(x, blk.a, blk.b), blk.tconv, blk.tconv_b, rate)
	for u, r in ipairs(blk.res) do
		local h = g:qwen3tts_snake(x, r.a1, r.b1)
		h = g:qwen3tts_conv(h, r.w1, r.b1c, DILATIONS[u])
		h = g:qwen3tts_conv(g:qwen3tts_snake(h, r.a2, r.b2), r.w2, r.b2c)
		x = g:add(h, x)
	end
	return x
end

--- Build the decoder graph.
-- @tparam lgml.graph g graph
-- @tparam lgml.tensor codes i32 leaf (n_q, n), one row per codebook
-- @int n frame count
-- @tparam lgml.tensor pos i32 positions (n)
-- @tparam lgml.tensor mask f16 sliding-window causal mask (n, n)
-- @treturn lgml.tensor samples (n * hop, 1)
function M:graph(g, codes, n, pos, mask)
	local x = g:get_rows(self.rvq_first, g:view(codes, { n }))
	x = g:qwen3tts_conv(x, self.rvq_first_out)
	local r = g:get_rows(self.rvq_rest[1], g:view(codes, { n }, nil, n * 4))
	for q = 2, self.hp.n_q - 1 do
		r = g:add(g:get_rows(self.rvq_rest[q], g:view(codes, { n }, nil, q * n * 4)), r)
	end
	x = g:add(x, g:qwen3tts_conv(r, self.rvq_rest_out))
	x = g:qwen3tts_conv(x, self.pre_w, self.pre_b)
	x = transformer(self, g, x, pos, mask)
	for i, u in ipairs(self.ups) do
		x = convnext(g, g:qwen3tts_tconv(x, u.tconv, u.tconv_b, self.hp.up[i]), u)
	end
	x = g:qwen3tts_conv(x, self.in_w, self.in_b)
	for i, blk in ipairs(self.blocks) do
		x = decoder_block(g, x, blk, self.hp.rates[i])
	end
	x = g:qwen3tts_snake(x, self.out_a, self.out_b)
	return g:qwen3tts_clamp(g:qwen3tts_conv(x, self.out_w, self.out_bias), -1, 1)
end

--- Decode frames [first, last] with ctx frames of left context.
-- @tparam table frames list of frames, each a list of n_q codes
-- @int first first frame (1-based)
-- @int last last frame (inclusive)
-- @int ctx left context frames
-- @treturn lgml.tensor new f32 leaf with the samples of [first, last]
function M:decode_window(frames, first, last, ctx)
	local be, hp = self.be, self.hp
	local s = math.max(1, first - ctx)
	local n = last - s + 1
	local ids = {}
	for q = 1, hp.n_q do
		for t = s, last do
			ids[#ids + 1] = frames[t][q]
		end
	end
	local codes <close> = be:tensor("i32", { hp.n_q, n }, ids)
	local pos <close> = be:tensor("i32", { n })
	local mask <close> = window_mask(be, n, hp.window)
	pos:fill_range(s - 1)
	local g <close> = be:graph()
	local pcm = self:graph(g, codes, n, pos, mask)
	local keep = (last - first + 1) * hp.hop
	local out = be:tensor("f32", { keep })
	g:compute(g:cpy(g:view(pcm, { keep }, nil, (first - s) * hp.hop * 4), out))
	return out
end

--- Decode all frames in chunks of CHUNK.
-- @tparam table frames list of frames, each a list of n_q codes
-- @tparam[opt] function step called after each chunk; return false to stop
-- @treturn {tensor,...} f32 sample leaves in order; the caller closes them
function M:decode(frames, step)
	local out = {}
	for first = 1, #frames, M.CHUNK do
		local last = math.min(#frames, first + M.CHUNK - 1)
		out[#out + 1] = self:decode_window(frames, first, last, M.CTX)
		if step and not step() then
			break
		end
	end
	return out
end

return M
