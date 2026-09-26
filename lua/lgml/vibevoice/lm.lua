--- Qwen2 decoder stack for VibeVoice.
-- q, k and v have biases and there is no q/k norm. KV caches are f32
-- leaves (n_ctx, kv_heads * head_dim) per layer.
-- @module lgml.vibevoice.lm
local lgml = require("lgml")

local M = {}

--- Collect layer weights.
-- @tparam lgml.weights w loaded gguf weights
-- @string prefix name prefix, for example "lm" or "tts_lm"
-- @int n layer count
-- @treturn {table,...} layer weights
-- @raise when a tensor is missing
function M.layers(w, prefix, n)
	local function get(name)
		return assert(w:get(name), name)
	end
	local out = {}
	for i = 0, n - 1 do
		local p = prefix .. ".layers." .. i .. "."
		out[i + 1] = {
			attn_ln = get(p .. "attn_ln.weight"),
			wq = get(p .. "attn.q_proj.weight"), bq = get(p .. "attn.q_proj.bias"),
			wk = get(p .. "attn.k_proj.weight"), bk = get(p .. "attn.k_proj.bias"),
			wv = get(p .. "attn.v_proj.weight"), bv = get(p .. "attn.v_proj.bias"),
			wo = get(p .. "attn.o_proj.weight"),
			ffn_ln = get(p .. "ffn_ln.weight"),
			gate = get(p .. "ffn.gate.weight"),
			up = get(p .. "ffn.up.weight"),
			down = get(p .. "ffn.down.weight"),
		}
	end
	return out
end

--- Make a KV cache.
-- Caches start as zeros. A decode graph may read rows past n_past, and
-- NaN there would reach the output through the masked scores.
-- @tparam lgml.backend be backend
-- @int n_layer layer count
-- @int n_ctx cache rows
-- @int width kv_heads * head_dim
-- @treturn table cache: n_ctx, width, k and v (lists of leaves)
function M.cache(be, n_layer, n_ctx, width)
	local c = { n_ctx = n_ctx, width = width, k = {}, v = {} }
	local z = string.rep("\0", 4 * n_ctx * width)
	for i = 1, n_layer do
		c.k[i] = be:tensor("f32", { n_ctx, width }, z)
		c.v[i] = be:tensor("f32", { n_ctx, width }, z)
	end
	return c
end

--- Make input leaves for n tokens that attend to n_kv cache rows.
-- fill sets them for a position.
-- @tparam lgml.backend be backend
-- @int n token count
-- @int n_kv cache rows to attend to
-- @treturn table inp: n, n_kv, pos, rows and mask
function M.inputs(be, n, n_kv)
	return {
		n = n, n_kv = n_kv,
		pos = be:tensor("i32", { n }),
		rows = be:tensor("i64", { n }),
		mask = be:tensor("f32", { n, n_kv }),
	}
end

--- Set the input leaves for tokens at position n_past.
-- @tparam table inp inputs from inputs
-- @int n_past position of the first token
-- @raise when n_past + n is larger than n_kv
function M.fill(inp, n_past)
	if n_past + inp.n > inp.n_kv then
		error(("cache view too short: %d + %d > %d"):format(n_past, inp.n, inp.n_kv))
	end
	inp.pos:fill_range(n_past)
	inp.rows:fill_range(n_past)
	inp.mask:fill_causal(n_past, inp.n)
end

--- Free the input leaves.
-- @tparam table inp inputs from inputs
function M.close_inputs(inp)
	for _, k in ipairs({ "pos", "rows", "mask" }) do
		inp[k]:close()
	end
end

-- Write k and v at inp.rows, then attend over the first n_kv rows.
local function attention(g, hp, inp, c, il, q, k, v)
	local H, Hkv, D = hp.n_head, hp.n_kv, hp.head_dim
	local n, n_kv, w = inp.n, inp.n_kv, Hkv * D
	local kw = g:set_rows(c.k[il], g:reshape(k, { n, w }), inp.rows)
	local vw = g:set_rows(c.v[il], g:reshape(v, { n, w }), inp.rows)
	local row, hd = lgml.row_size("f32", w), lgml.row_size("f32", D)
	k = g:permute(g:view(kw, { n_kv, Hkv, D }, { row, hd }), { 2, 1, 3 })
	v = g:cont(g:permute(g:view(vw, { n_kv, Hkv, D }, { row, hd }), { 2, 3, 1 }))
	q = g:permute(q, { 2, 1, 3 })
	local kq = g:soft_max(g:mul_mat(k, q), inp.mask, 1 / math.sqrt(D))
	local o = g:mul_mat(v, kq)
	return g:reshape(g:cont(g:permute(o, { 2, 1, 3 })), { n, H * D })
end

local function linear(g, w, b, x)
	return g:add(g:mul_mat(w, x), b)
end

--- Run x (n, d) through the layers.
-- Each layer writes k and v into cache c at inp.rows.
-- @tparam lgml.graph g graph
-- @tparam table hp hyperparameters
-- @tparam {table,...} layers layer weights from layers
-- @tparam lgml.tensor x input (n, d)
-- @tparam table inp inputs from inputs
-- @tparam table c cache from cache
-- @treturn lgml.tensor hidden state (n, d) before any final norm
function M.stack(g, hp, layers, x, inp, c)
	local n = inp.n
	local H, Hkv, D = hp.n_head, hp.n_kv, hp.head_dim
	local rope = { mode = "neox", base = hp.rope_theta }
	for il, L in ipairs(layers) do
		local cur = g:mul(g:rms_norm(x, hp.eps), L.attn_ln)
		local q = g:reshape(linear(g, L.wq, L.bq, cur), { n, H, D })
		local k = g:reshape(linear(g, L.wk, L.bk, cur), { n, Hkv, D })
		local v = g:reshape(linear(g, L.wv, L.bv, cur), { n, Hkv, D })
		q = g:rope(q, inp.pos, rope)
		k = g:rope(k, inp.pos, rope)
		local o = attention(g, hp, inp, c, il, q, k, v)
		x = g:add(g:mul_mat(L.wo, o), x)
		cur = g:mul(g:rms_norm(x, hp.eps), L.ffn_ln)
		local ff = g:mul(g:silu(g:mul_mat(L.gate, cur)), g:mul_mat(L.up, cur))
		x = g:add(g:mul_mat(L.down, ff), x)
	end
	return x
end

--- The last row of x (n, d).
-- @tparam lgml.graph g graph
-- @tparam lgml.tensor x input (n, d)
-- @int n row count
-- @int d row width
-- @treturn lgml.tensor view (1, d)
function M.last(g, x, n, d)
	if n == 1 then
		return x
	end
	local row = lgml.row_size("f32", d)
	return g:view(x, { 1, d }, { row }, (n - 1) * row)
end

return M
