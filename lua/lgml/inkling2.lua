--- The Inkling 2 text model (gguf arch inkling2).
-- Layers [core_start, core_end) run num_loops times with shared weights.
-- The gguf holds 24 physical blocks, and the config lists one layer per
-- visit; each visit has its own KV cache and convolution state. There is
-- no rope: a learned bias by distance carries the positions. Block
-- attention residuals mix the earlier block outputs into each layer input.
-- `lgml.transformer` calls this module for the arch.
-- @module lgml.inkling2
local lgml = require("lgml")

local M = {}

-- Message controllers of a looped layer, in llama.cpp order: q, k, v, ffn.
local CTRL = { "attn", "k", "v", "mlp" }

local function T()
	return package.loaded["lgml.transformer"]
end

--- The physical layer (from 0) of visit v (from 0).
-- @tparam table I hp.ink
-- @tparam integer v the visit
-- @treturn integer the physical layer
function M.phys(I, v)
	if v < I.cs then
		return v
	end
	local nc = I.ce - I.cs
	if v < I.cs + I.passes * nc then
		return I.cs + (v - I.cs) % nc
	end
	return v - (I.passes - 1) * nc
end

--- Read the inkling2 parts of the config.
-- It sets hp.ink and the per-visit layer fields; hp.n_vocab becomes the
-- unpadded vocabulary, as the head drops the padded logits.
-- @tparam table hp the config so far
-- @tparam function k reads a key without the arch prefix, with a default
-- @raise on metadata that the model cannot use
function M.config(hp, k)
	local a = hp.arch
	local function arr(v, i)
		return type(v) == "table" and v[i] or v
	end
	local heads, kvh = k("attention.head_count"), k("attention.head_count_kv")
	hp.n_head = arr(heads, 1)
	hp.n_head_kv = arr(kvh, 1)
	hp.head_dim = k("attention.key_length")
	local I = {
		cs = k("loop.core_start", 0), ce = k("loop.core_end", 0), passes = k("num_loops", 1),
		rank = k("loop.gate_rank", 0),
		rel = k("attention.relative_length"), extent = k("attention.relative_extent"),
		rel_clip = k("attention.relative_clip"), qk_clip = k("attention.qk_clip"),
		log_floor = k("attention.log_scale_floor"), log_alpha = k("attention.log_scale_alpha"),
		block = k("attn_res.block_size"), conv = k("shortconv.l_cache"),
		situ_beta = k("activation.situ_beta"), situ_lb = k("activation.situ_linear_beta"),
		n_vocab_pad = k("vocab_size"),
	}
	if I.passes < 2 then
		I.cs, I.ce, I.passes = 0, 0, 1
	end
	I.phys = hp.n_layer - (I.passes - 1) * (I.ce - I.cs)
	for _, f in ipairs({ "rel", "extent", "rel_clip", "qk_clip", "log_floor", "log_alpha", "block", "conv",
		"situ_beta", "situ_lb" }) do
		if not I[f] then
			error(a .. ": missing metadata for " .. f)
		end
	end
	if I.passes > 1 and (I.rank < 1 or I.cs == 0 or I.cs % I.block ~= 0 or (I.ce - I.cs) % I.block ~= 0) then
		error(a .. ": the looped core must start after the first block and span whole blocks")
	end
	hp.n_vocab = k("vocab_size_unpadded")
	hp.n_swa = k("attention.sliding_window")
	hp.softcap = k("final_logit_softcapping", 0)
	hp.gating = T().gating.SIGMOID
	hp.norm_topk = true
	hp.n_rot = 0
	local swa = k("attention.sliding_window_pattern")
	assert(type(swa) == "table" and #swa == hp.n_layer, a .. ": need a sliding_window_pattern entry per layer")
	for i = 1, hp.n_layer do
		local d = hp.layers[i]
		d.swa = swa[i] == true or swa[i] == 1
		d.D = hp.head_dim
		d.Hkv = arr(kvh, i)
		d.H = arr(heads, i)
		d.n_rot = 0
		d.freqs = false
		d.ip = M.phys(I, i - 1)
	end
	I.channels = 2 * hp.n_head_kv * hp.head_dim + 2 * hp.n_embd
	hp.ink = I
end

-- Rows of a (R, C) f32 leaf with one zero row after them, as a new leaf.
local function pad_row(be, t)
	local sh = t:shape()
	local v = t:get()
	for i = #v + 1, #v + sh[2] do
		v[i] = 0
	end
	return be:tensor("f32", { sh[1] + 1, sh[2] }, v)
end

-- Load physical layer i (from 0).
local function load_phys(be, w, hp, i)
	local I = hp.ink
	local p = "blk." .. i .. "."
	local function get(name)
		return assert(w:get(p .. name), p .. name)
	end
	local function opt(name)
		return w:get(p .. name)
	end
	-- A physical layer has the attention kind of its first visit.
	local v = i < I.ce and i or i + (I.passes - 1) * (I.ce - I.cs)
	local swa = hp.layers[v + 1].swa
	local P = {
		wq = get("attn_q.weight"), wk = get("attn_k.weight"), wv = get("attn_v.weight"),
		wo = get("attn_output.weight"), qkv_gate = get("attn_gate.weight"),
		q_clip = get("attn_q_clip.weight"), sink_proj = get("attn_sink_proj.weight"),
		rel = get("attn_rel.weight"), rel_bias = pad_row(be, get("attn_rel_bias.weight")),
		rel_clip = get("attn_rel_clip.weight"),
		k_conv = get("attn_k_conv.weight"), v_conv = get("attn_v_conv.weight"),
		post_conv = get("attn_post_conv.weight"), ffn_post_conv = get("ffn_post_conv.weight"),
		attn_res_query = opt("attn_res_query.weight"), ffn_res_query = get("ffn_res_query.weight"),
		ffn_norm = get("ffn_norm.weight"), gate_inp = get("ffn_gate_inp.weight"),
		exp_probs_b = opt("exp_probs_b.bias"),
		gate_up_exps = get("ffn_gate_up_exps.weight"), down_exps = get("ffn_down_exps.weight"),
		gate_up_scale = opt("ffn_gate_up_exps.scale"), down_scale = opt("ffn_down_exps.scale"),
	}
	if swa then
		P.attn_norm = get("attn_norm.weight")
	else
		P.q_norm = get("attn_q_inp_norm.weight")
		P.k_norm = get("attn_k_inp_norm.weight")
		P.v_norm = get("attn_v_inp_norm.weight")
		P.k_conv_proj = get("attn_k_conv_proj.weight")
		P.v_conv_proj = get("attn_v_conv_proj.weight")
		P.attn_norm = P.q_norm
	end
	if i > 0 then
		assert(P.attn_res_query, p .. "attn_res_query.weight")
	end
	if I.passes > 1 and i >= I.cs and i < I.ce then
		P.ctrl = {}
		for c, name in ipairs(CTRL) do
			P.ctrl[c] = {
				down = get("loop_" .. name .. "_down.weight"),
				gain = get("loop_" .. name .. "_gain.weight"),
				forget = get("loop_" .. name .. "_forget.weight"),
			}
		end
	end
	return P
end

--- Load a model of this arch.
-- @param be the backend
-- @tparam string path the gguf file
-- @tparam table opts the options of `lgml.transformer.load`
-- @tparam table info the gguf info of path
-- @tparam table hp the config
-- @treturn Model the model, with the metatable of `lgml.transformer`
function M.load(be, path, opts, info, hp)
	if opts.placement then
		error(hp.arch .. ": weight placement is not supported")
	end
	local I = hp.ink
	local w = assert(be:load(path))
	local function get(name)
		return assert(w:get(name), name)
	end
	local m = setmetatable({ be = be, w = w, hp = hp, meta = info.meta }, T())
	m.tok_embd = get("token_embd.weight")
	m.tok_norm = get("token_embd_norm.weight")
	m.output = get("output.weight")
	m.output_norm = get("output_norm.weight")
	m.output_res_query = get("output_res_query.weight")
	if I.passes > 1 then
		m.loop_res_query = get("loop_res_query.weight")
		m.loop_kv_res_query = get("loop_kv_res_query.weight")
	end
	local zeros = {}
	for i = 1, hp.n_head do
		zeros[i] = 0
	end
	m.zero_sinks = be:tensor("f32", { hp.n_head }, zeros)
	local phys = {}
	for i = 0, I.phys - 1 do
		phys[i] = load_phys(be, w, hp, i)
	end
	m.phys = phys
	m.layers = {}
	for i = 1, hp.n_layer do
		local d = hp.layers[i]
		m.layers[i] = setmetatable({ recr = false, swa = d.swa, D = d.D, Hkv = d.Hkv, H = d.H, n_rot = 0,
			base = 0, ip = d.ip }, { __index = phys[d.ip] })
	end
	-- The conv states cannot go back to an earlier position.
	m.recurrent = true
	if opts.replay == false then
		m.replay_key = false
	end
	return m
end

-- Physical layer fields that take an adapter, by gguf name after "blk.N.".
local FIELDS = {
	["attn_q.weight"] = "wq", ["attn_k.weight"] = "wk", ["attn_v.weight"] = "wv", ["attn_output.weight"] = "wo",
	["attn_gate.weight"] = "qkv_gate", ["attn_rel.weight"] = "rel", ["attn_sink_proj.weight"] = "sink_proj",
	["attn_k_conv_proj.weight"] = "k_conv_proj", ["attn_v_conv_proj.weight"] = "v_conv_proj",
	["ffn_gate_inp.weight"] = "gate_inp", ["ffn_gate_up_exps.weight"] = "gate_up_exps",
	["ffn_down_exps.weight"] = "down_exps",
}

--- Apply LoRA adapters, or remove them.
-- Adapters can change the projections of a layer, the routed experts,
-- the token embedding and the output. A looped layer shares its adapter
-- across its passes, as it shares its weights.
-- @tparam Model self the model
-- @tparam[opt] {lgml.lora.Adapter,...} list adapters from `lgml.lora`
-- @raise when an adapter names a weight that takes no adapter, or a shape
-- differs
function M.set_adapters(self, list)
	local maps, embd, out, keys = {}, {}, {}, {}
	for _, ad in ipairs(list or {}) do
		for _, name in ipairs(ad.names) do
			local p = ad.pairs[name]
			local e = { a = p.a, b = p.b, s = ad:scale_of(name) }
			local il, suffix = name:match("^blk%.(%d+)%.(.+)$")
			local P = il and self.phys[tonumber(il)]
			local field = P and FIELDS[suffix]
			if field then
				local ws, bs = P[field]:shape(), p.b:shape()
				if bs[#bs - 1] ~= ws[#ws - 1] then
					error(("lora: %s: B (%s) does not fit the weight (%s)"):format(name, table.concat(bs, ", "),
					    table.concat(ws, ", ")))
				end
				maps[P] = maps[P] or {}
				maps[P][field] = maps[P][field] or {}
				table.insert(maps[P][field], e)
			elseif name == "token_embd.weight" then
				embd[#embd + 1] = e
			elseif name == "output.weight" then
				out[#out + 1] = e
			else
				error("lora: " .. name .. ": the model has no such weight that takes an adapter")
			end
		end
		keys[#keys + 1] = ad.id .. "*" .. ad.scale
	end
	for _, P in pairs(self.phys) do
		P.lora = maps[P]
	end
	self.lora_embd = #embd > 0 and embd or nil
	self.lora_out = #out > 0 and out or nil
	self.adapters = list and #list > 0 and list or nil
	self.lora_key = table.concat(keys, ",")
end

-- The product of weight L[key] and x, plus the adapter terms s B (A x).
local function mm(g, L, key, x, f32)
	local y = g:mul_mat(L[key], x, f32)
	for _, e in ipairs(L.lora and L.lora[key] or {}) do
		y = g:add(y, g:scale(g:mul_mat(e.b, g:mul_mat(e.a, x)), e.s))
	end
	return y
end

-- The routed expert product of L[key] for the picks sel, plus the adapter
-- terms; s scales the expert outputs (per expert, or nil) before them.
local function mm_id(g, L, key, x, sel, scale)
	local y = scale(g:mul_mat_id(L[key], x, sel))
	for _, e in ipairs(L.lora and L.lora[key] or {}) do
		y = g:add(y, g:scale(g:mul_mat_id(e.b, g:mul_mat_id(e.a, x, sel), sel), e.s))
	end
	return y
end

--- Add the conv states of each visit to a per-sequence cache: four f32
-- tensors (1, C, conv - 1) for K, V, the attention output and the FFN
-- output, in c.pconv[4 * (il - 1) + j].
-- @tparam Model self the model
-- @tparam table c the cache
function M.add_cache(self, c)
	if not c.conv then
		return
	end
	local hp = self.hp
	local I = hp.ink
	c.pconv = c.pconv or {}
	local kv = hp.n_head_kv * hp.head_dim
	local widths = { kv, kv, hp.n_embd, hp.n_embd }
	for il = 1, hp.n_layer do
		for j = 1, 4 do
			local t = self.be:tensor("f32", { 1, widths[j], I.conv - 1 })
			t:zero()
			c.pconv[4 * (il - 1) + j] = t
		end
	end
end

--- Bytes of the conv states of one sequence.
-- @tparam table hp the config
-- @treturn integer bytes
function M.state_bytes(hp)
	return 4 * hp.n_layer * hp.ink.channels * (hp.ink.conv - 1)
end

--- Make the inkling2 input leaves of a pass of n tokens: tau (n, 1, 1),
-- the query positions qpos (n, 1) and the cell positions kpos and
-- kpos_swa (1, span), all f32.
-- @tparam Model self the model
-- @tparam table inp the inputs
-- @tparam integer n the tokens
function M.leaves(self, inp, n)
	local be = self.be
	inp.tau = be:tensor("f32", { n, 1, 1 })
	inp.qpos = be:tensor("f32", { n, 1 })
	inp.kpos = be:tensor("f32", { 1, inp.n_attn })
	if inp.mask_swa then
		inp.kpos_swa = be:tensor("f32", { 1, inp.n_swa_attn })
	end
end

--- Fill the inkling2 leaves of a pass at cache.n_past.
-- A cell that holds no position of the sequence gets 0; the mask hides it.
-- @tparam Model self the model
-- @tparam table inp the inputs
-- @tparam[opt] table cache the cache
function M.pass(self, inp, cache)
	local I = self.hp.ink
	local n, n_past = inp.n, inp.n_past
	local tau, qpos = {}, {}
	for i = 1, n do
		local p = n_past + i - 1
		qpos[i] = p
		tau[i] = 1 + I.log_alpha * math.log(math.max(1, (p + 1) / I.log_floor))
	end
	inp.tau:set(tau)
	inp.qpos:set(qpos)
	local na, kpos = inp.n_attn, {}
	for j = 1, na do
		kpos[j] = 0
	end
	if cache and cache.pool then
		local B, lo = cache.pool.block, inp.lo
		for b, blk in ipairs(cache.blocks) do
			for o = 0, B - 1 do
				local col = blk * B + o - lo
				if col >= 0 and col < na then
					kpos[col + 1] = (b - 1) * B + o
				end
			end
		end
	else
		for j = 1, na do
			kpos[j] = j - 1
		end
	end
	inp.kpos:set(kpos)
	if inp.kpos_swa then
		local ns, n_kv, ring = inp.n_swa_attn, inp.n_kv, cache and cache.ring
		local ks = {}
		for r = 0, ns - 1 do
			local p = r
			if ring and r < n_kv then
				p = r + (n_kv - 1 - r) // ring * ring
			end
			ks[r + 1] = p
		end
		inp.kpos_swa:set(ks)
	end
end

-- Divide each row of x (..., D) by max(1, |row| / thr), after the scale s.
local function clip(g, x, s, thr)
	if s then
		x = g:mul(x, s)
	end
	local norm = g:sqrt(g:sum_rows(g:sqr(x)))
	return g:div(x, g:clamp(g:scale(norm, 1 / thr), 1, math.huge))
end

-- Attention over the depth: the rms-normed blocks and cur, mixed by each
-- row of the queries q (Q, E). The result is a list of Q nodes (n, E).
-- cache holds the normed blocks of this graph, by block list.
local function mix(g, hp, cache, blocks, cur, q)
	local n, E = cur:shape()[1], hp.n_embd
	local states = g:rms_norm(g:reshape(cur, { n, 1, E }), 1e-6)
	if #blocks > 0 then
		-- Nodes print their shape, not their identity: number them.
		local ids, key = cache.ids, {}
		for i, b in ipairs(blocks) do
			if not ids[b] then
				cache.n = cache.n + 1
				ids[b] = cache.n
			end
			key[i] = ids[b]
		end
		key = table.concat(key, ",")
		local normed = cache[key]
		if not normed then
			normed = g:reshape(blocks[1], { n, 1, E })
			for i = 2, #blocks do
				normed = g:concat(normed, g:reshape(blocks[i], { n, 1, E }), 2)
			end
			normed = g:rms_norm(normed, 1e-6)
			cache[key] = normed
		end
		states = g:concat(normed, states, 2)
	end
	local J = #blocks + 1
	local Q = #q:shape() == 1 and 1 or q:shape()[1]
	local f4, st = lgml.row_size("f32", E), states:strides()
	local out = {}
	for r = 1, Q do
		local qr = g:view(q, { 1, 1, E }, { f4, f4 }, (r - 1) * f4)
		local w = g:sum_rows(g:mul(states, qr))
		w = g:reshape(g:soft_max(g:reshape(w, { n, J }), nil, 1 / math.sqrt(E)), { n, J, 1 })
		local prod = g:mul(states, w)
		local ps = prod:strides()
		local o = g:view(prod, { n, E }, { ps[1] })
		for j = 1, J - 1 do
			o = g:add(o, g:view(prod, { n, E }, { ps[1] }, j * ps[2]))
		end
		out[r] = J == 1 and g:cont(o) or o
	end
	return out
end

-- Mix a message of the previous pass into a looped layer input.
local function control(g, hp, C, cur, msg)
	local n, E = cur:shape()[1], hp.n_embd
	local h = g:concat(g:reshape(cur, { n, 1, E }), g:reshape(msg, { n, 1, E }), 2)
	h = g:reshape(g:rms_norm(h, 1e-6), { n, 2 * E })
	h = g:mul_mat(C.down, h)
	local gain = g:tanh(g:mul_mat(C.gain, h))
	local forget = g:tanh(g:mul_mat(C.forget, h))
	return g:add(g:sub(cur, g:mul(forget, cur)), g:mul(gain, msg))
end

-- Causal short conv of x (n, C) with weight w (C, K) and a residual. The
-- last K - 1 inputs live in state (1, C, K - 1); the graph writes them
-- back. dyn (n, heads * K) adds input-dependent taps per head.
local function convolve(g, I, x, w, dyn, state)
	local n, C = x:shape()[1], x:shape()[2]
	local K = I.conv
	local f4 = lgml.row_size("f32", 1)
	local xin = g:concat(state, g:transpose(x), 3)
	local s = xin:strides()
	g:expand(g:cpy(g:view(xin, { 1, C, K - 1 }, { s[1], s[2] }, n * f4), state))
	local out = g:reshape(g:ssm_conv(xin, w), { n, C })
	if dyn then
		local nh = dyn:shape()[2] // K
		local d3 = g:reshape(dyn, { n, nh, K })
		local ds = d3:strides()
		for lag = 0, K - 1 do
			local tap = g:view(d3, { n, nh, 1 }, { ds[1], ds[2] }, lag * ds[3])
			local win = g:view(xin, { C, n }, { s[2] }, (K - 1 - lag) * f4)
			win = g:reshape(g:cont(g:transpose(win)), { n, nh, C // nh })
			out = g:add(out, g:reshape(g:mul(win, tap), { n, C }))
		end
	end
	return g:scale(g:add(x, out), 1 / math.sqrt((dyn and 2 or 1) * K + 1))
end

--- The most bytes of the per-head mask of one slice of queries. Global
-- layers build the mask and run attention in slices of at most this size,
-- so the compute buffer does not grow with the pass size.
M.BIAS_BYTES = 256 << 20

-- Rows off+1 .. off+n of an (N, ...) node.
local function rows_of(g, t, off, n)
	local sh, st = t:shape(), t:strides()
	local shape, strides = { n }, {}
	for i = 2, #sh do
		shape[i] = sh[i]
	end
	for i = 1, #sh - 1 do
		strides[i] = st[i]
	end
	return g:view(t, shape, strides, off * st[1])
end

-- Attention of visit il with the cache, with the bias by distance. The
-- bias and the mask go in as one mask per head.
local function attention(self, g, inp, il, L, q, k, v, r, sinks)
	local kv = require("lgml.kv")
	local c, n, D, H = inp.cache, inp.n, L.D, L.H
	local rk, rv = kv.rot_k(c, D), kv.rot_v(c, D)
	q, k, v = kv.rotate(g, q, rk), kv.rotate(g, k, rk), kv.rotate(g, v, rv)
	local w = L.Hkv * D
	local rows = L.swa and inp.rows_swa or inp.rows
	local src = (c.pool and not L.swa) and c.pool or c
	local kw = g:set_rows(src.k[il], g:reshape(k, { n, w }), rows)
	local vw = g:set_rows(src.v[il], g:reshape(v, { n, w }), rows)
	local span = L.swa and inp.n_swa_attn or inp.n_attn
	k, v = kv.views(g, c, kw, vw, span, L.Hkv, D, not L.swa and inp.lo or 0)
	-- Bias table per query and head: r . rel_bias, less the sink logit.
	local tab = g:mul_mat(L.rel_bias, r)
	tab = g:sub(tab, g:reshape(sinks, { n, H, 1 }))
	local mask = L.swa and inp.mask_swa or inp.mask
	local kpos = L.swa and inp.kpos_swa or inp.kpos
	local extent = L.swa and self.hp.n_swa or self.hp.ink.extent
	local scale = 1 / math.sqrt(D)
	local flash = inp.flash_attn
	q = g:permute(q, { 2, 1, 3 })
	k = g:permute(k, { 2, 1, 3 })
	if flash then
		v = g:permute(v, { 2, 1, 3 })
		k, v = kv.for_flash(g, k), kv.for_flash(g, v)
	else
		v = g:cont(g:permute(v, { 2, 3, 1 }))
	end
	-- The mask of queries [off, off + m): the bias gathered by distance,
	-- plus the causal mask.
	local function slice(off, m)
		local mr = rows_of(g, mask, off, m)
		local qp = rows_of(g, inp.qpos, off, m)
		local tr = rows_of(g, tab, off, m)
		if flash then
			return g:rel_mask(tr, qp, kpos, mr)
		end
		local idx = g:cast(g:clamp(g:sub(g:tile(qp, mr), kpos), 0, extent), "i32")
		local bias = g:get_rows(g:cont(g:permute(tr, { 1, 3, 2 })), idx)
		return g:add(g:cont(g:permute(bias, { 3, 1, 2 })), mr)
	end
	local per = flash and math.max(1, M.BIAS_BYTES // (2 * H * span)) or n
	local out
	for off = 0, n - 1, per do
		local m = math.min(per, n - off)
		local qst = q:strides()
		local qs = per < n and g:view(q, { H, m, D }, { qst[1], qst[2] }, off * qst[2]) or q
		local o
		if flash then
			o = g:flash_attn(qs, k, v, slice(off, m), scale, 0, 0, true, self.zero_sinks)
		else
			local kq = g:soft_max(g:mul_mat(k, qs, true), slice(off, m), scale, 0, self.zero_sinks)
			o = g:cont(g:permute(g:mul_mat(v, kq), { 2, 1, 3 }))
		end
		out = out and g:concat(out, o, 1) or o
	end
	return g:reshape(kv.rotate(g, g:cont(out), rv), { n, H * D })
end

-- The routed experts of a layer, with the SITU activation.
local function moe(g, hp, L, h)
	local I = hp.ink
	local n, E, NE, K = h:shape()[1], hp.n_embd, hp.n_expert, hp.n_used
	local probs = g:sigmoid(mm(g, L, "gate_inp", h))
	local sel_in = L.exp_probs_b and g:add(probs, L.exp_probs_b) or probs
	local sel = g:argsort_top_k(sel_in, K)
	local probs3 = g:reshape(probs, { n, NE, 1 })
	local wts = g:reshape(g:get_rows(probs3, sel), { n, K })
	wts = g:div(wts, g:clamp(g:sum_rows(wts), 1e-20, math.huge))
	wts = g:scale(wts, hp.expert_scale or 1)
	wts = g:reshape(wts, { n, K, 1 })
	local function scaler(s)
		return function(y)
			if not s then
				return y
			end
			local st = g:tile(g:reshape(s, { 1, NE, 1 }), probs3)
			return g:mul(y, g:get_rows(st, sel))
		end
	end
	local x3 = g:reshape(h, { n, 1, E })
	local gu = mm_id(g, L, "gate_up_exps", x3, sel, scaler(L.gate_up_scale))
	local F = gu:shape()[3] // 2
	local st = gu:strides()
	local gate = g:cont(g:view(gu, { n, K, F }, { st[1], st[2] }))
	local up = g:cont(g:view(gu, { n, K, F }, { st[1], st[2] }, lgml.row_size("f32", F)))
	local b, lb = I.situ_beta, I.situ_lb
	local act = g:mul(g:scale(g:tanh(g:scale(gate, 1 / b)), b), g:sigmoid(gate))
	act = g:mul(act, g:scale(g:tanh(g:scale(up, 1 / lb)), lb))
	local ex = mm_id(g, L, "down_exps", act, sel, scaler(L.down_scale))
	ex = g:mul(ex, wts)
	local exs = ex:strides()
	local out = g:view(ex, { n, E }, { exs[1] })
	for j = 1, K - 1 do
		out = g:add(out, g:view(ex, { n, E }, { exs[1] }, j * exs[2]))
	end
	return out
end

-- One layer visit. msgs holds the four messages of a looped pass, or nil.
local function layer(self, g, inp, il, L, cur, blocks, msgs)
	local hp, I = self.hp, self.hp.ink
	local n, E, D, H, eps = inp.n, hp.n_embd, L.D, L.H, hp.eps
	local kvw = L.Hkv * D
	local st = inp.cache.pconv
	local s0 = 4 * (il - 1)
	local inpSA = cur
	local ins = { cur, cur, cur }
	if L.attn_res_query then
		local Q = L.swa and 1 or 3
		local mixed = mix(g, hp, inp.mix_cache, blocks, cur, g:reshape(L.attn_res_query, { Q, E }))
		for j = 1, 3 do
			ins[j] = mixed[L.swa and 1 or j]
		end
	end
	if msgs then
		for j = 1, 3 do
			ins[j] = control(g, hp, L.ctrl[j], ins[j], msgs[j])
		end
	end
	local function norm(x, wt)
		return g:mul(g:rms_norm(x, eps), wt)
	end
	if L.swa and msgs then
		for j = 1, 3 do
			ins[j] = norm(ins[j], L.attn_norm)
		end
	elseif L.swa then
		ins[1] = norm(ins[1], L.attn_norm)
		ins[2], ins[3] = ins[1], ins[1]
	else
		ins[1] = norm(ins[1], L.q_norm)
		ins[2] = norm(ins[2], L.k_norm)
		ins[3] = norm(ins[3], L.v_norm)
	end
	local q = mm(g, L, "wq", ins[1])
	local k = convolve(g, I, mm(g, L, "wk", ins[2]), L.k_conv,
	    not L.swa and mm(g, L, "k_conv_proj", ins[2]) or nil, st[s0 + 1])
	local v = convolve(g, I, mm(g, L, "wv", ins[3]), L.v_conv,
	    not L.swa and mm(g, L, "v_conv_proj", ins[3]) or nil, st[s0 + 2])
	local r = mm(g, L, "rel", ins[1])
	q = clip(g, g:reshape(q, { n, H, D }), L.q_clip, I.qk_clip)
	k = clip(g, g:reshape(k, { n, L.Hkv, D }), nil, I.qk_clip)
	v = g:rms_norm(g:reshape(v, { n, L.Hkv, D }), eps)
	r = clip(g, g:reshape(r, { n, H, I.rel }), L.rel_clip, I.rel_clip)
	if not L.swa then
		q = g:mul(q, inp.tau)
		r = g:mul(r, inp.tau)
	end
	local sinks = mm(g, L, "sink_proj", ins[1])
	local o = attention(self, g, inp, il, L, q, k, v, r, sinks)
	o = g:mul(o, g:sigmoid(mm(g, L, "qkv_gate", ins[1])))
	o = convolve(g, I, mm(g, L, "wo", o), L.post_conv, nil, st[s0 + 3])
	-- The first layer of a block starts a new residual stream.
	if L.ip % I.block == 0 then
		blocks[#blocks + 1] = inpSA
		cur = o
	else
		cur = g:add(inpSA, o)
	end
	local ffn_inp = cur
	local h = mix(g, hp, inp.mix_cache, blocks, cur, g:reshape(L.ffn_res_query, { 1, E }))[1]
	if msgs then
		h = control(g, hp, L.ctrl[4], h, msgs[4])
	end
	h = norm(h, L.ffn_norm)
	local ex = convolve(g, I, moe(g, hp, L, h), L.ffn_post_conv, nil, st[s0 + 4])
	return g:add(ffn_inp, ex)
end

local function copy(t)
	return table.move(t, 1, #t, 1, {})
end

--- Build the graph of a pass.
-- @tparam Model self the model
-- @param g the graph
-- @tparam table inp the inputs
-- @return the logits (or the normed hidden state with inp.pool), and the
-- named nodes
function M.forward(self, g, inp)
	local hp, I = self.hp, self.hp.ink
	local n, E = inp.n, hp.n_embd
	assert(inp.cache and not inp.batch, hp.arch .. ": passes need a cache of one sequence")
	local taps = {}
	inp.mix_cache = { ids = {}, n = 0 }
	local x
	if inp.embd then
		x = inp.embd
	else
		x = g:get_rows(self.tok_embd, inp.ids)
		for _, e in ipairs(self.lora_embd or {}) do
			x = g:add(x, g:scale(g:mul_mat(e.b, g:get_rows(e.a, inp.ids)), e.s))
		end
		x = g:mul(g:rms_norm(x, hp.eps), self.tok_norm)
	end
	taps.embd = x
	local nc = I.ce - I.cs
	local blocks, pre_blocks, pre_cur, bank, bank_last = {}, nil, nil, nil, nil
	local cur = x
	for il, L in ipairs(self.layers) do
		local v, ip = il - 1, L.ip
		local msgs
		if I.passes > 1 and ip >= I.cs and ip < I.ce then
			local pass = (v - I.cs) // nc
			if ip == I.cs then
				if pass == 0 then
					pre_blocks, pre_cur = copy(blocks), cur
				else
					bank = table.move(blocks, #pre_blocks + 2, #blocks, 1, {})
					bank_last = cur
					blocks, cur = copy(pre_blocks), pre_cur
				end
			end
			if pass > 0 then
				local rw = 2 * (ip - I.cs)
				local f4 = lgml.row_size("f32", E)
				local qa = g:view(self.loop_res_query, { 2, E }, { f4 }, rw * f4)
				local qk = g:view(self.loop_kv_res_query, { 2, E }, { f4 }, rw * f4)
				local ma = mix(g, hp, inp.mix_cache, bank, bank_last, qa)
				local mk = mix(g, hp, inp.mix_cache, bank, bank_last, qk)
				msgs = { ma[1], mk[1], mk[2], ma[2] }
			end
		end
		cur = layer(self, g, inp, il, L, cur, blocks, msgs)
		taps["l_out-" .. v] = cur
		if il == inp.hidden then
			return cur, taps
		end
	end
	if inp.logits == "rows" then
		cur = g:get_rows(cur, inp.logit_rows)
		for i, b in ipairs(blocks) do
			blocks[i] = g:get_rows(b, inp.logit_rows)
		end
	elseif (inp.logits == "last" or inp.pool == "last") and n > 1 then
		local rs = lgml.row_size("f32", E)
		local function last(t)
			return g:view(t, { 1, E }, nil, (n - 1) * rs)
		end
		cur = last(cur)
		for i, b in ipairs(blocks) do
			blocks[i] = last(b)
		end
	end
	local h = mix(g, hp, inp.mix_cache, blocks, cur, g:reshape(self.output_res_query, { 1, E }))[1]
	h = g:mul(g:rms_norm(h, hp.eps), self.output_norm)
	taps.result_norm = h
	if inp.pool then
		return h, taps
	end
	local logits = mm(g, { output = self.output, lora = self.lora_out and { output = self.lora_out } }, "output", h, true)
	if hp.softcap > 0 then
		logits = g:scale(g:tanh(g:scale(logits, 1 / hp.softcap)), hp.softcap)
	end
	local m, rs = logits:shape()[1], logits:strides()[1]
	logits = g:cont(g:view(logits, { m, hp.n_vocab }, { rs }))
	return logits, taps
end

return M
