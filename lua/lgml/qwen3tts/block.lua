--- Qwen3 decoder layers for the Qwen3-TTS talker and code predictor.
-- The layers use q/k norm, neox rope, GQA, SwiGLU and a per-sequence
-- KV cache.
-- @module lgml.qwen3tts.block
local lgml = require("lgml")

local M = {}

--- Layer weights with prefix p.
-- @tparam lgml.weights w loaded gguf weights
-- @string p tensor name prefix, for example "talker.blk.3."
-- @treturn table layer weights: attn_norm, wq, wk, wv, wo, q_norm,
-- k_norm, ffn_norm, gate, up, down
-- @raise when a tensor is missing
function M.layer(w, p)
	local function get(name)
		return assert(w:get(p .. name .. ".weight"), p .. name)
	end
	return {
		attn_norm = get("attn_norm"), wq = get("attn_q"), wk = get("attn_k"),
		wv = get("attn_v"), wo = get("attn_output"), q_norm = get("attn_q_norm"),
		k_norm = get("attn_k_norm"), ffn_norm = get("ffn_norm"),
		gate = get("ffn_gate"), up = get("ffn_up"), down = get("ffn_down"),
	}
end

--- Make a KV cache.
-- The cache holds f16 leaves (n_ctx, kv_heads * head_dim) per layer.
-- @tparam lgml.backend be backend
-- @tparam table hp hyperparameters with n_head_kv and head_dim
-- @int n_layer layer count
-- @int n_ctx cache rows
-- @treturn table cache: n_ctx, n_past (starts at 0), k and v (lists of
-- leaves)
function M.cache(be, hp, n_layer, n_ctx)
	local w = hp.n_head_kv * hp.head_dim
	local c = { n_ctx = n_ctx, n_past = 0, k = {}, v = {} }
	for i = 1, n_layer do
		c.k[i] = be:tensor("f16", { n_ctx, w })
		c.v[i] = be:tensor("f16", { n_ctx, w })
	end
	return c
end

--- Make reusable leaves for one-token steps.
-- @tparam lgml.backend be backend
-- @treturn table pos (i32, 1) and rows (i64, 1)
function M.step_leaves(be)
	return { pos = be:tensor("i32", { 1 }), rows = be:tensor("i64", { 1 }) }
end

--- Make inputs for n tokens at cache.n_past.
-- One token reuses the leaves in one. More tokens get new leaves and a
-- causal mask.
-- @tparam lgml.backend be backend
-- @tparam table cache cache from cache
-- @int n token count
-- @tparam[opt] table one leaves from step_leaves
-- @treturn table inp: n, n_kv, cache, pos, rows and mask (nil for one token)
-- @treturn {tensor,...} leaves for the caller to close
-- @raise when n_past + n is larger than the cache
function M.inputs(be, cache, n, one)
	local n_past = cache.n_past
	local n_kv = n_past + n
	if n_kv > cache.n_ctx then
		error("context full: " .. n_kv .. " > " .. cache.n_ctx)
	end
	local inp = { n = n, n_kv = n_kv, cache = cache }
	if n == 1 and one then
		one.pos:set({ n_past })
		one.rows:set({ n_past })
		inp.pos, inp.rows = one.pos, one.rows
		return inp, {}
	end
	inp.pos = be:tensor("i32", { n })
	inp.rows = be:tensor("i64", { n })
	inp.mask = be:tensor("f16", { n, n_kv })
	inp.pos:fill_range(n_past)
	inp.rows:fill_range(n_past)
	inp.mask:fill_causal(n_past)
	return inp, { inp.pos, inp.rows, inp.mask }
end

-- Write k and v at inp.rows; attend over the first n_kv cache rows.
local function attention(g, inp, hp, il, q, k, v)
	local H, Hkv, D = hp.n_head, hp.n_head_kv, hp.head_dim
	local n, n_kv, c = inp.n, inp.n_kv, inp.cache
	local kw = g:set_rows(c.k[il], g:reshape(k, { n, Hkv * D }), inp.rows)
	local vw = g:set_rows(c.v[il], g:reshape(v, { n, Hkv * D }), inp.rows)
	local row = lgml.row_size("f16", Hkv * D)
	local hd = lgml.row_size("f16", D)
	k = g:permute(g:view(kw, { n_kv, Hkv, D }, { row, hd }), { 2, 1, 3 })
	v = g:permute(g:view(vw, { n_kv, Hkv, D }, { row, hd }), { 2, 1, 3 })
	q = g:permute(q, { 2, 1, 3 })
	local o = g:flash_attn(q, k, v, inp.mask, 1 / math.sqrt(D))
	return g:reshape(o, { n, H * D })
end

--- One decoder layer on x (n, d).
-- The layer writes k and v into the cache rows of inp.
-- @tparam lgml.graph g graph
-- @tparam lgml.tensor x input (n, d)
-- @tparam table L layer weights from layer
-- @tparam table hp hyperparameters
-- @tparam table inp inputs from inputs
-- @int il layer index (1-based)
-- @treturn lgml.tensor output (n, d)
function M.forward_layer(g, x, L, hp, inp, il)
	local n, H, Hkv, D = inp.n, hp.n_head, hp.n_head_kv, hp.head_dim
	local rope = { mode = "neox", base = hp.rope_base, n_ctx_orig = hp.n_ctx_orig }
	local cur = g:mul(g:rms_norm(x, hp.eps), L.attn_norm)
	local q = g:reshape(g:mul_mat(L.wq, cur), { n, H, D })
	local k = g:reshape(g:mul_mat(L.wk, cur), { n, Hkv, D })
	local v = g:reshape(g:mul_mat(L.wv, cur), { n, Hkv, D })
	q = g:rope(g:mul(g:rms_norm(q, hp.eps), L.q_norm), inp.pos, rope)
	k = g:rope(g:mul(g:rms_norm(k, hp.eps), L.k_norm), inp.pos, rope)
	x = g:add(g:mul_mat(L.wo, attention(g, inp, hp, il, q, k, v)), x)
	cur = g:mul(g:rms_norm(x, hp.eps), L.ffn_norm)
	local ff = g:mul(g:silu(g:mul_mat(L.gate, cur)), g:mul_mat(L.up, cur))
	return g:add(g:mul_mat(L.down, ff), x)
end

--- All layers, then the final norm of the last row.
-- @tparam lgml.graph g graph
-- @tparam lgml.tensor x input (n, d)
-- @tparam {table,...} layers layer weights
-- @tparam lgml.tensor out_norm final norm weight
-- @tparam table hp hyperparameters
-- @tparam table inp inputs from inputs
-- @treturn lgml.tensor hidden state (1, d)
function M.forward(g, x, layers, out_norm, hp, inp)
	for il, L in ipairs(layers) do
		x = M.forward_layer(g, x, L, hp, inp, il)
	end
	if inp.n > 1 then
		local row = lgml.row_size("f32", hp.n_embd)
		x = g:view(x, { 1, hp.n_embd }, nil, (inp.n - 1) * row)
	end
	return g:mul(g:rms_norm(x, hp.eps), out_norm)
end

return M
