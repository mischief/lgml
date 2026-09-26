--- Config-driven decoder for the gguf text models.
-- One module builds llama, qwen3, qwen3moe, qwen3next, qwen35,
-- qwen35moe and gemma4 graphs. `config` reads the gguf metadata into a
-- table; the blocks read that table and the weights that exist.
-- A layer is gated delta net or attention, then a dense FFN, routed
-- experts, or both. Shapes are in torch order.
-- @module lgml.transformer
local lgml = require("lgml")
local kv = require("lgml.kv")
local replay = require("lgml.replay")

local M = {}
M.__index = M

-- Attention reads the cache in blocks of PAD rows, so decode steps keep
-- one graph shape while the span stays inside a block.
local PAD = 256
-- The default count of tokens per pass on a sliding-window cache.
local UBATCH = 512

--- Expert gating functions, as gguf `expert_gating_func` numbers.
-- SOFTMAX: softmax over all router logits. SIGMOID: sigmoid of each
-- logit.
M.gating = { SOFTMAX = 1, SIGMOID = 2 }

-- Layer fields of the weights that adapters can change, by gguf name
-- after "blk.N.". Routed experts and the embeddings take no adapter.
local FIELDS = {
	["attn_q.weight"] = "wq", ["attn_k.weight"] = "wk", ["attn_v.weight"] = "wv",
	["attn_output.weight"] = "wo", ["ffn_gate.weight"] = "gate", ["ffn_up.weight"] = "up",
	["ffn_down.weight"] = "down", ["attn_qkv.weight"] = "qkv", ["attn_gate.weight"] = "qkv_gate",
	["ssm_in.weight"] = "ssm_in", ["ssm_beta.weight"] = "beta", ["ssm_alpha.weight"] = "alpha",
	["ssm_ba.weight"] = "ba", ["ssm_out.weight"] = "ssm_out", ["ffn_gate_inp.weight"] = "gate_inp",
	["ffn_gate_shexp.weight"] = "gate_shexp", ["ffn_up_shexp.weight"] = "up_shexp",
	["ffn_down_shexp.weight"] = "down_shexp", ["inp_gate.weight"] = "ple_gate",
	["proj.weight"] = "ple_proj",
}

-- The product of weight L[key] and x, plus the adapter terms
-- s * B (A x) of L.lora[key].
local function mm(g, L, key, x)
	local y = g:mul_mat(L[key], x)
	for _, e in ipairs(L.lora and L.lora[key] or {}) do
		y = g:add(y, g:scale(g:mul_mat(e.b, g:mul_mat(e.a, x)), e.s))
	end
	return y
end

-- Settings that the metadata does not hold, per architecture.
local family = {
	llama = { rope_mode = "normal", rope_ctx = true },
	qwen3 = { qk_norm = true, rope_ctx = true },
	qwen3moe = { qk_norm = true, rope_ctx = true },
	qwen3next = { qk_norm = true, rope_ctx = true, hybrid = true, ssm_tile = true },
	qwen35 = { qk_norm = true, rope_ctx = true, hybrid = true },
	qwen35moe = { qk_norm = true, rope_ctx = true, hybrid = true },
	gemma4 = { qk_norm = true, gemma = true },
	["diffusion-gemma"] = { qk_norm = true, gemma = true },
}

-- Check that text tokens can use neox rope. Their t, h and w positions
-- are equal, so multi-section rope is neox rope when no rotary pair
-- takes the e section.
local function check_sections(a, sec, n_rot)
	local t, h, w = sec[1] or 0, sec[2] or 0, sec[3] or 0
	local all = t + h + w + (sec[4] or 0)
	for i = 0, n_rot // 2 - 1 do
		local s = i % all
		local m = s % 3
		if not ((m == 0 and s < 3 * t) or (m == 1 and s < 3 * h) or (m == 2 and s < 3 * w)) then
			error(a .. ": rope sections not supported")
		end
	end
end

--- Tell if the transformer handles a gguf architecture.
-- @tparam string arch the `general.architecture` value
-- @treturn boolean true for llama, qwen3, qwen3moe, qwen3next, qwen35,
-- qwen35moe and gemma4
function M.supports(arch)
	return family[arch] ~= nil
end

--- Read the model config from gguf metadata.
-- Keys have the prefix `general.architecture`. The table is the `hp`
-- field of a model. It holds arch, n_layer (without nextn layers),
-- n_embd, n_head, n_head_kv, head_dim, n_rot, eps, rope_base, n_ctx,
-- n_vocab (or nil), and the flags rope_mode, qk_norm, attn_gate, gemma.
-- MoE models add n_expert, n_used, gating and norm_topk. Gemma adds
-- head_dim_swa, n_rot_swa, rope_base_swa, n_swa, swa, softcap,
-- n_kv_shared and n_embd_per_layer. Hybrid models add recr, d_conv,
-- head_k, n_k_heads, n_v_heads and head_v.
-- `layers[i]` describes layer i: recr, swa, D, Hkv, n_rot, base,
-- freqs (true when rope_freqs applies) and kv_src (the layer whose
-- cache it reads, or nil).
-- @tparam table meta the gguf metadata, as from `lgml.gguf(path).meta`
-- @treturn table the config
-- @raise on an unknown architecture or unsupported metadata
function M.config(meta)
	local a = meta["general.architecture"]
	local fam = family[a]
	if not fam then
		error("transformer: unknown architecture " .. tostring(a))
	end
	local function k(s, def)
		local v = meta[a .. "." .. s]
		if v == nil then
			v = def
		end
		return v
	end
	local n_layer = k("block_count") - k("nextn_predict_layers", 0)
	local n_embd, n_head = k("embedding_length"), k("attention.head_count")
	local head_dim = k("attention.key_length") or n_embd // n_head
	local hkv = k("attention.head_count_kv", n_head)
	local hp = {
		arch = a,
		n_layer = n_layer,
		n_embd = n_embd,
		n_head = n_head,
		n_head_kv = type(hkv) == "table" and hkv[1] or hkv,
		head_dim = head_dim,
		n_rot = k("rope.dimension_count", head_dim),
		eps = k("attention.layer_norm_rms_epsilon"),
		rope_base = k("rope.freq_base", 10000),
		n_ctx = k("context_length"),
		n_vocab = k("vocab_size"),
		rope_mode = fam.rope_mode or "neox",
		qk_norm = fam.qk_norm or false,
		attn_gate = fam.hybrid or false,
		gemma = fam.gemma or false,
		n_expert = k("expert_count"),
		layers = {},
	}
	if fam.rope_ctx then
		hp.rope_ctx = hp.n_ctx
	end
	if hp.n_expert then
		hp.n_used = k("expert_used_count")
		hp.gating = k("expert_gating_func", M.gating.SOFTMAX)
		hp.norm_topk = k("expert_weights_norm", true)
		if hp.gating ~= M.gating.SOFTMAX and hp.gating ~= M.gating.SIGMOID then
			error(("%s: expert gating %s not supported"):format(a, tostring(hp.gating)))
		end
	end
	local swa = k("attention.sliding_window_pattern")
	if fam.gemma then
		assert(type(swa) == "table" and #swa == n_layer,
		    a .. ": need a sliding_window_pattern entry per layer")
		hp.head_dim_swa = k("attention.key_length_swa")
		hp.n_rot_swa = k("rope.dimension_count_swa", hp.head_dim_swa)
		hp.rope_base_swa = k("rope.freq_base_swa", 10000)
		hp.n_swa = k("attention.sliding_window")
		hp.swa = swa
		hp.softcap = k("final_logit_softcapping", 0)
		hp.n_kv_shared = k("attention.shared_kv_layers", 0)
		hp.n_embd_per_layer = k("embedding_length_per_layer_input", 0)
		hp.embd_scale = math.sqrt(n_embd)
		hp.attn_scale = 1.0
	end
	if fam.hybrid then
		hp.d_conv = k("ssm.conv_kernel")
		hp.head_k = k("ssm.state_size")
		hp.n_k_heads = k("ssm.group_count")
		hp.n_v_heads = k("ssm.time_step_rank")
		hp.head_v = k("ssm.inner_size") // hp.n_v_heads
		hp.ssm_tile = fam.ssm_tile or false
		hp.recr = {}
		local rl = k("attention.recurrent_layers")
		local every = k("full_attention_interval", 4)
		for i = 1, n_layer do
			if rl then
				hp.recr[i] = rl[i] == true or rl[i] == 1
			else
				hp.recr[i] = i % every ~= 0
			end
		end
		local sec = k("rope.dimension_sections")
		if sec then
			check_sections(a, sec, hp.n_rot)
		end
	end
	local kv_start = n_layer - (hp.n_kv_shared or 0)
	for i = 1, n_layer do
		local s = fam.gemma and swa[i] or false
		local d = {
			recr = hp.recr and hp.recr[i] or false,
			swa = s,
			D = s and hp.head_dim_swa or head_dim,
			Hkv = type(hkv) == "table" and hkv[i] or hkv,
			n_rot = s and hp.n_rot_swa or hp.n_rot,
			base = s and hp.rope_base_swa or hp.rope_base,
			freqs = not s,
		}
		-- Shared layers read the last layer of the same kind before
		-- kv_start.
		if i > kv_start then
			for j = kv_start, 1, -1 do
				if hp.layers[j].swa == s then
					d.kv_src = j
					break
				end
			end
			assert(d.kv_src, a .. ": no layer to share the cache of layer " .. i)
		end
		hp.layers[i] = d
	end
	return hp
end

--- Load the weights of one layer.
-- The config says which weights must exist. Other weights are
-- optional, and the blocks use them when they exist.
-- @param w the weights, from `backend:load`
-- @tparam table hp the config, from `config`
-- @tparam integer i the layer index, from 1
-- @param[opt] rope_freqs the rope frequency factors
-- @treturn table the layer: weights and the fields of `hp.layers[i]`
-- @raise when a required weight is missing
function M.load_layer(w, hp, i, rope_freqs)
	local p = "blk." .. (i - 1) .. "."
	local function get(name)
		return assert(w:get(p .. name), p .. name)
	end
	local function opt(name)
		return w:get(p .. name)
	end
	local d = hp.layers[i]
	local L = {
		recr = d.recr, swa = d.swa, D = d.D, Hkv = d.Hkv, n_rot = d.n_rot, base = d.base,
		kv_src = d.kv_src, freqs = d.freqs and rope_freqs or nil,
		attn_norm = get("attn_norm.weight"),
	}
	if d.recr then
		-- Older Qwen3-Next files have one ssm_in in place of attn_qkv
		-- and attn_gate, and one ssm_ba in place of beta and alpha.
		L.qkv = opt("attn_qkv.weight")
		if L.qkv then
			L.qkv_gate = get("attn_gate.weight")
		else
			L.ssm_in = get("ssm_in.weight")
		end
		L.ba = opt("ssm_ba.weight")
		if not L.ba then
			L.beta = get("ssm_beta.weight")
			L.alpha = get("ssm_alpha.weight")
		end
		L.conv = get("ssm_conv1d.weight")
		L.dt = get("ssm_dt.bias")
		L.a = get("ssm_a")
		L.ssm_norm = get("ssm_norm.weight")
		L.ssm_out = get("ssm_out.weight")
	else
		L.wq = get("attn_q.weight")
		L.wo = get("attn_output.weight")
		if hp.qk_norm then
			L.q_norm = get("attn_q_norm.weight")
		end
		if not d.kv_src then
			L.wk = get("attn_k.weight")
			-- Gemma global layers have no attn_v: V is K.
			L.wv = opt("attn_v.weight")
			if not hp.gemma then
				L.wv = get("attn_v.weight")
			end
			if hp.qk_norm then
				L.k_norm = get("attn_k_norm.weight")
			end
		end
	end
	if hp.gemma then
		L.post_attn_norm = get("post_attention_norm.weight")
		L.ffn_norm = get("ffn_norm.weight")
		L.post_ffn_norm = get("post_ffw_norm.weight")
		L.gate_inp = opt("ffn_gate_inp.weight")
		L.router_scale = opt("ffn_gate_inp.scale")
		L.post_ffn_norm_1 = opt("post_ffw_norm_1.weight")
		L.pre_ffn_norm_2 = opt("pre_ffw_norm_2.weight")
		L.post_ffn_norm_2 = opt("post_ffw_norm_2.weight")
		L.down_exps_scale = opt("ffn_down_exps.scale")
		L.out_scale = opt("layer_output_scale.weight")
		L.ple_gate = opt("inp_gate.weight")
		L.ple_proj = opt("proj.weight")
		L.ple_post_norm = opt("post_norm.weight")
	else
		L.ffn_norm = get(hp.attn_gate and "post_attention_norm.weight" or "ffn_norm.weight")
		if hp.n_expert then
			L.gate_inp = get("ffn_gate_inp.weight")
		end
	end
	if hp.gemma or not hp.n_expert then
		L.gate = get("ffn_gate.weight")
		L.up = get("ffn_up.weight")
		L.down = get("ffn_down.weight")
	end
	if L.gate_inp then
		-- gate_up is one (E, 2F, n_embd) tensor in some files.
		L.gate_up_exps = opt("ffn_gate_up_exps.weight")
		if not L.gate_up_exps then
			L.gate_exps = get("ffn_gate_exps.weight")
			L.up_exps = get("ffn_up_exps.weight")
		end
		L.down_exps = get("ffn_down_exps.weight")
		L.shexp_gate = opt("ffn_gate_inp_shexp.weight")
		if L.shexp_gate then
			L.gate_shexp = get("ffn_gate_shexp.weight")
			L.up_shexp = get("ffn_up_shexp.weight")
			L.down_shexp = get("ffn_down_shexp.weight")
		end
	end
	return L
end

--- Load a model.
-- The output weight falls back to the token embedding.
-- @param be the lgml backend
-- @tparam string path path to the gguf file
-- @tparam[opt] table opts options:
-- opts.replay: false builds a new graph for every pass (default true).
-- @treturn Model the model
-- @raise when the file does not open, a weight is missing or the
-- architecture is not supported
function M.load(be, path, opts)
	opts = opts or {}
	local info = assert(lgml.gguf(path))
	local w = assert(be:load(path))
	local hp = M.config(info.meta)
	local function get(name)
		return assert(w:get(name), name)
	end
	local m = setmetatable({ be = be, w = w, hp = hp, meta = info.meta }, M)
	m.tok_embd = get("token_embd.weight")
	m.output = w:get("output.weight") or m.tok_embd
	m.output_norm = get("output_norm.weight")
	m.rope_freqs = w:get("rope_freqs.weight")
	hp.n_vocab = hp.n_vocab or m.output:shape()[1]
	if (hp.n_embd_per_layer or 0) > 0 then
		m.ple_embd = get("per_layer_token_embd.weight")
		m.ple_proj = get("per_layer_model_proj.weight")
		m.ple_norm = get("per_layer_proj_norm.weight")
	end
	m.layers = {}
	for i = 1, hp.n_layer do
		local L = M.load_layer(w, hp, i, m.rope_freqs)
		if m.ple_embd then
			assert(L.ple_gate and L.ple_proj and L.ple_post_norm,
			    "blk." .. (i - 1) .. ": no per-layer input weights")
		end
		m.recurrent = m.recurrent or L.recr or nil
		m.layers[i] = L
	end
	if opts.replay == false then
		m.replay_key = false
	end
	return m
end

--- A loaded model, from `load`.
-- Fields: be (backend), w (weights), hp (from `config`), meta (gguf
-- metadata), tok_embd, output, output_norm, rope_freqs (or nil),
-- layers, recurrent (true when a layer keeps a recurrent state,
-- which cannot go back to an earlier position), and adapters (from
-- `set_adapters`, or nil). A layer with adapters has `lora`: adapter
-- terms { a, b, s } by weight field.
-- @type Model

--- Close the weights.
function M:close()
	self.w:close()
end

--- Apply LoRA adapters to the model, or remove them.
-- Each adapter adds s * B (A x) to the product of each weight it names,
-- as llama.cpp does, with s from `adapter:scale_of`. The list replaces
-- the adapters of an earlier call; an empty list removes them. Passes
-- with other adapters or scales build their own replay graph.
-- @tparam[opt] {lgml.lora.Adapter,...} list adapters from `lgml.lora`
-- @raise when an adapter names a weight that the model lacks or that
-- takes no adapter (routed experts, embeddings), or a shape differs
function M:set_adapters(list)
	local maps, out, keys = {}, {}, {}
	for _, ad in ipairs(list or {}) do
		for _, name in ipairs(ad.names) do
			local p = ad.pairs[name]
			local il, suffix = name:match("^blk%.(%d+)%.(.+)$")
			local L = il and self.layers[tonumber(il) + 1]
			local field = L and FIELDS[suffix]
			local w = field and L[field]
			if name == "output.weight" and self.output ~= self.tok_embd then
				w = self.output
			end
			if not w then
				error("lora: " .. name .. ": the model has no such weight that takes an adapter")
			end
			local ws, as, bs = w:shape(), p.a:shape(), p.b:shape()
			if #ws ~= 2 or as[2] ~= ws[2] or bs[1] ~= ws[1] or as[1] ~= bs[2] then
				error(("lora: %s: A (%s) and B (%s) do not fit the weight (%s)"):format(name,
				    table.concat(as, ", "), table.concat(bs, ", "), table.concat(ws, ", ")))
			end
			local e = { a = p.a, b = p.b, s = ad:scale_of(name) }
			if L then
				maps[L] = maps[L] or {}
				maps[L][field] = maps[L][field] or {}
				table.insert(maps[L][field], e)
			else
				out[#out + 1] = e
			end
		end
		keys[#keys + 1] = ad.id .. "*" .. ad.scale
	end
	for _, L in ipairs(self.layers) do
		L.lora = maps[L]
	end
	self.lora_out = #out > 0 and out or nil
	self.adapters = list and #list > 0 and list or nil
	self.lora_key = table.concat(keys, ",")
end

local function conv_dim(hp)
	return 2 * hp.n_k_heads * hp.head_k + hp.n_v_heads * hp.head_v
end

--- Make a cache for one sequence.
-- Attention layers get zeroed K and V leaves (rows, Hkv * D). Global
-- layers hold n_ctx rows. Sliding layers hold a ring of
-- min(n_ctx, n_swa - 1 + ubatch) rows, so one pass takes at most
-- ubatch tokens once the ring wraps. Layers that share a cache get
-- none. Recurrent layers get an f32 conv state (1, conv_dim,
-- d_conv - 1) and a delta net state (1, Hv, S, S), both zero.
-- @tparam integer n_ctx the maximum count of positions
-- @tparam[opt] table opts options:
-- opts.type, opts.type_k, opts.type_v: cache types from `lgml.kv`
-- (default f16).
-- opts.ubatch: the most tokens per pass on a ring (default 512).
-- @treturn table cache with n_ctx, n_past (0), type_k, type_v, k, v;
-- ring, ubatch and hi with sliding layers; rs_pos (the position the
-- states are at), conv and ssm with recurrent layers
function M:cache(n_ctx, opts)
	opts = opts or {}
	local hp, be = self.hp, self.be
	local dims = {}
	for _, L in ipairs(self.layers) do
		if not L.recr then
			dims[#dims + 1] = L.D
		end
	end
	local tk, tv = kv.types_of(opts, dims)
	local c = { n_ctx = n_ctx, n_past = 0, be = be, type_k = tk, type_v = tv, k = {}, v = {} }
	if hp.n_swa then
		c.ubatch = opts.ubatch or UBATCH
		c.ring = math.min(n_ctx, hp.n_swa - 1 + c.ubatch)
		c.hi = 0
	end
	if self.recurrent then
		c.rs_pos, c.conv, c.ssm = 0, {}, {}
	end
	for il, L in ipairs(self.layers) do
		if L.recr then
			c.conv[il] = be:tensor("f32", { 1, conv_dim(hp), hp.d_conv - 1 })
			c.ssm[il] = be:tensor("f32", { 1, hp.n_v_heads, hp.head_v, hp.head_v })
			c.conv[il]:zero()
			c.ssm[il]:zero()
		elseif not L.kv_src then
			local rows = L.swa and c.ring or n_ctx
			c.k[il] = be:tensor(tk, { rows, L.Hkv * L.D })
			c.v[il] = be:tensor(tv, { rows, L.Hkv * L.D })
			-- Rows past n_past are read under a -inf mask; zero keeps
			-- them finite.
			c.k[il]:zero()
			c.v[il]:zero()
		end
	end
	return c
end

--- Bytes a cache from `cache` takes on the device.
-- @tparam[opt] table opts the cache options, as for `cache`
-- @treturn integer bytes per position (full-attention layers)
-- @treturn integer bytes that do not grow with n_ctx (rings, recurrent
-- states)
function M:cache_size(opts)
	opts = opts or {}
	local hp = self.hp
	local dims = {}
	for _, L in ipairs(self.layers) do
		if not L.recr then
			dims[#dims + 1] = L.D
		end
	end
	local tk, tv = kv.types_of(opts, dims)
	local ring = hp.n_swa and hp.n_swa - 1 + (opts.ubatch or UBATCH)
	local per, fixed = 0, 0
	for _, L in ipairs(self.layers) do
		if L.recr then
			fixed = fixed + 4 * (conv_dim(hp) * (hp.d_conv - 1) + hp.n_v_heads * hp.head_v * hp.head_v)
		elseif not L.kv_src then
			local row = lgml.row_size(tk, L.Hkv * L.D) + lgml.row_size(tv, L.Hkv * L.D)
			if L.swa then
				fixed = fixed + ring * row
			else
				per = per + row
			end
		end
	end
	return per, fixed
end

--- Find how many cached positions stay valid after a rewind.
-- The engine calls this before it sets cache.n_past to a smaller value.
-- A ring must still hold the n_swa - 1 positions before n. Other
-- caches keep every position.
-- @tparam table cache a cache from `cache`
-- @tparam integer n the wanted count of kept positions
-- @tparam[opt] boolean peek true only checks: cache.hi stays
-- @treturn integer n, or 0 when the ring lost those positions
function M:rewind(cache, n, peek)
	if not cache.ring then
		return n
	end
	if n > 0 and n < cache.hi and n + cache.ring - self.hp.n_swa + 1 < cache.hi then
		n = 0
	end
	if n == 0 and not peek then
		cache.hi = 0
	end
	return n
end

-- The parts of a snapshot: { tensor, offset, bytes } in layer order.
-- Full-attention layers give rows [0, rows) when rows is set. Rings give
-- their rows up to hi; recurrent layers give their whole states.
local function snap_parts(self, cache, rows, state)
	local parts = {}
	local function add(t, n)
		if n > 0 then
			parts[#parts + 1] = { t, 0, n }
		end
	end
	for il, L in ipairs(self.layers) do
		if L.recr and state then
			add(cache.conv[il], cache.conv[il]:nbytes())
			add(cache.ssm[il], cache.ssm[il]:nbytes())
		elseif cache.k[il] and (L.swa and state or not L.swa and rows) then
			local n = L.swa and math.min(cache.hi, cache.ring) or rows
			for _, t in ipairs({ cache.k[il], cache.v[il] }) do
				add(t, n * (t:nbytes() // t:shape()[1]))
			end
		end
	end
	return parts
end

--- Bytes that `snapshot` copies now.
-- @tparam table cache a cache from `cache`
-- @tparam[opt] boolean full true for a full snapshot
-- @treturn integer bytes
function M:snapshot_bytes(cache, full)
	local bytes = 0
	for _, p in ipairs(snap_parts(self, cache, full and cache.n_past, true)) do
		bytes = bytes + p[3]
	end
	return bytes
end

--- Copy the cache state at cache.n_past to host memory.
-- The state is the recurrent conv and delta net states and the sliding
-- rings with hi: the parts that cannot go back to an earlier position.
-- A full snapshot also holds rows [0, n_past) of the full-attention
-- layers.
-- @tparam table cache a cache from `cache`
-- @tparam[opt] boolean full true for a full snapshot
-- @treturn table the snapshot: n (the position), hi, full, bytes and
-- buf (an `lgml.hostbuf`; close it to free the memory at once)
function M:snapshot(cache, full)
	local n = cache.n_past
	local parts = snap_parts(self, cache, full and n, true)
	local bytes = 0
	for _, p in ipairs(parts) do
		bytes = bytes + p[3]
	end
	local buf = lgml.hostbuf(bytes)
	local off = 0
	for _, p in ipairs(parts) do
		p[1]:to_host(buf, off, p[2], p[3])
		off = off + p[3]
	end
	return { n = n, hi = cache.hi, full = full or false, bytes = bytes, buf = buf }
end

--- Write a snapshot back into the cache it came from.
-- Without opts it sets n_past, rs_pos and hi to the snapshot's. The
-- full-attention rows before n of a snapshot without them must still
-- hold the same tokens. The cached replay graph stays valid.
-- @tparam table cache the cache
-- @tparam table snap a snapshot from `snapshot`
-- @tparam[opt] table opts options:
-- opts.rows: restore only the first rows full-attention rows of a full
-- snapshot and set n_past to rows (default snap.n).
-- opts.state: false leaves the state, rs_pos and hi as they are (default
-- true); the caller then restores a snapshot at opts.rows.
function M:restore(cache, snap, opts)
	opts = opts or {}
	local rows = opts.rows or snap.n
	local state = opts.state ~= false
	assert(rows <= snap.n and (snap.full or rows == snap.n), "restore: rows not in the snapshot")
	-- Walk the parts in snapshot order; skip the ones not restored.
	local hi = cache.hi
	cache.hi = snap.hi
	local all = snap_parts(self, cache, snap.full and snap.n, true)
	local want = snap_parts(self, cache, snap.full and rows, state)
	cache.hi = hi
	local off, j = 0, 1
	for _, p in ipairs(all) do
		local w = want[j]
		if w and w[1] == p[1] then
			p[1]:from_host(snap.buf, off, 0, w[3])
			j = j + 1
		end
		off = off + p[3]
	end
	cache.n_past = rows
	if state then
		cache.hi = snap.hi
		if cache.rs_pos then
			cache.rs_pos = snap.n
		end
	end
end

-- The attention spans of a pass that ends at n_kv: global layers and
-- sliding layers. With a cache both are padded to PAD rows.
local function spans(cache, n_kv)
	if not cache then
		return n_kv, n_kv
	end
	local n_attn = math.min(cache.n_ctx, (n_kv + PAD - 1) // PAD * PAD)
	local ring = cache.ring or cache.n_ctx
	return n_attn, n_kv > ring and ring or math.min(ring, n_attn)
end

-- Raise when a pass of n tokens does not fit the cache.
local function check_pass(hp, cache, n)
	local n_kv = cache.n_past + n
	if n_kv > cache.n_ctx then
		error("context full: " .. n_kv .. " > " .. cache.n_ctx)
	end
	if cache.ring and cache.ring < cache.n_ctx and n > cache.ring - hp.n_swa + 1 then
		error(("%s: %d tokens in one pass; the ring takes %d"):format(hp.arch, n,
		    cache.ring - hp.n_swa + 1))
	end
	if cache.rs_pos then
		if cache.n_past == 0 then
			for i in pairs(cache.conv) do
				cache.conv[i]:zero()
				cache.ssm[i]:zero()
			end
		elseif cache.n_past ~= cache.rs_pos then
			error(("recurrent state is at %d, not %d"):format(cache.rs_pos, cache.n_past))
		end
	end
end

-- Fill the input leaves for the pass at cache.n_past, or at 0 with
-- no cache. ids is nil when the leaf already holds them.
local function fill(hp, inp, cache, ids, opts)
	local n = inp.n
	local n_past = cache and cache.n_past or 0
	inp.n_past, inp.n_kv = n_past, n_past + n
	if ids then
		inp.ids:set(ids)
	end
	if inp.embd then
		inp.embd:from_host(opts.embd, opts.embd_off or 0, 0, inp.embd:nbytes())
	end
	inp.pos:fill_range(n_past)
	if inp.causal then
		inp.mask:fill_causal(n_past)
	else
		inp.mask:fill_mask({ n_past = n_past, causal = false })
	end
	if inp.mask_swa then
		inp.mask_swa:fill_mask({ n_past = n_past, window = hp.n_swa, ring = cache ~= nil,
			causal = inp.causal })
	end
	if cache then
		inp.rows:fill_range(n_past)
		if inp.rows_swa then
			local r = {}
			for i = 1, n do
				r[i] = (n_past + i - 1) % cache.ring
			end
			inp.rows_swa:set(r)
			cache.hi = math.max(cache.hi, inp.n_kv)
		end
		if cache.rs_pos then
			cache.rs_pos = inp.n_kv
		end
	end
end

--- Make the graph inputs for n tokens at positions n_past and up.
-- C fills pos, rows and the masks. A recurrent model without a cache
-- gets a new one for this pass. n_past 0 clears the recurrent state.
-- Any other n_past must be where the state is.
-- @tparam {integer,...} ids the token ids
-- @tparam[opt] table cache a cache from `cache`
-- @tparam[opt] table opts options:
-- opts.flash_attn: true for flash attention (f16 masks).
-- opts.logits: "all" or "last" (default "all").
-- opts.hidden: a layer count; `forward` stops after that many layers
-- and gives the hidden state in place of the logits (default nil).
-- opts.embd: an `lgml.hostbuf` with n rows of n_embd f32 values, such
-- as image embeddings; they take the place of the token embeddings,
-- with no scale. ids still give the per-layer inputs of gemma models;
-- use 0 for them there, as llama.cpp does.
-- opts.embd_off: the byte offset of the first row in opts.embd
-- (default 0).
-- opts.causal: false lets the tokens of this pass see each other both
-- ways, as llama.cpp does for gemma image tokens (default true).
-- @treturn table fields n, n_past, n_kv, n_attn, n_swa_attn, cache,
-- flash_attn, logits, hidden, causal, and the tensors ids, pos, mask,
-- mask_swa (with sliding layers), rows and rows_swa (with a cache), and
-- embd (with opts.embd)
-- @raise when the pass does not fit the cache, or n_past is not where
-- the recurrent state is
function M:inputs(ids, cache, opts)
	opts = opts or {}
	local be, hp, n = self.be, self.hp, #ids
	if self.recurrent and not cache then
		cache = self:cache(n)
	end
	if cache then
		check_pass(hp, cache, n)
	end
	kv.check(cache, opts.flash_attn)
	local n_kv = (cache and cache.n_past or 0) + n
	local n_attn, n_swa_attn = spans(cache, n_kv)
	local mt = opts.flash_attn and "f16" or "f32"
	local inp = {
		n = n, n_attn = n_attn, n_swa_attn = n_swa_attn, cache = cache,
		flash_attn = opts.flash_attn, logits = opts.logits or "all",
		hidden = opts.hidden, causal = opts.causal ~= false,
		ids = be:tensor("i32", { n }, ids),
		pos = be:tensor("i32", { n }),
		mask = be:tensor(mt, { n, n_attn }),
	}
	if hp.n_swa then
		inp.mask_swa = be:tensor(mt, { n, n_swa_attn })
	end
	if opts.embd then
		inp.embd = be:tensor("f32", { n, hp.n_embd })
	end
	if cache then
		inp.rows = be:tensor("i64", { n })
		if cache.ring then
			inp.rows_swa = be:tensor("i64", { n })
		end
	end
	fill(hp, inp, cache, nil, opts)
	return inp
end

--- The replay key of a pass: the pass shape, options and adapters.
-- Passes with the same key share one graph; see `lgml.replay`.
-- @tparam table cache the cache
-- @tparam integer n the token count
-- @tparam table opts the input options
-- @treturn string the key
function M:replay_key(cache, n, opts)
	local n_attn, n_swa_attn = spans(cache, cache.n_past + n)
	return ("%d:%d:%d:%s:%s:%s:%s:%s:%s"):format(n, n_attn, n_swa_attn, tostring(opts.flash_attn),
	    opts.logits or "all", tostring(opts.hidden), tostring(opts.embd ~= nil),
	    tostring(opts.causal ~= false), self.lora_key or "")
end

--- Load new ids into the inputs of an earlier pass with the same key.
-- @tparam table inp inputs from `inputs`
-- @tparam {integer,...} ids the token ids
-- @tparam table cache the cache
-- @tparam[opt] table opts the input options; opts.embd and
-- opts.embd_off give the new embeddings
-- @raise as `inputs`
function M:refill(inp, ids, cache, opts)
	check_pass(self.hp, cache, #ids)
	fill(self.hp, inp, cache, ids, opts)
end

--- Build the normed, projected and roped queries, keys and values.
-- A layer without attn_v uses the keys as values. A layer that shares
-- a cache returns no keys and values. This is a plain function.
-- @static
-- @param g the graph
-- @tparam table hp the config
-- @tparam table L the layer
-- @param x the layer input (n, n_embd)
-- @param pos the i32 positions (n)
-- @return queries (n, n_head, D)
-- @return keys (n, Hkv, D), or nil
-- @return values (n, Hkv, D), or nil
-- @return the output gate (n, n_head, D), or nil
function M.qkv(g, hp, L, x, pos)
	local eps, n, H, D = hp.eps, x:shape()[1], hp.n_head, L.D
	local rope = { mode = hp.rope_mode, base = L.base, n_dims = L.n_rot, n_ctx_orig = hp.rope_ctx }
	local cur = g:mul(g:rms_norm(x, eps), L.attn_norm)
	local q, gate, k, v
	if hp.attn_gate then
		-- q_proj gives [q | gate] per head.
		local f4 = lgml.row_size("f32", 1)
		local qg = mm(g, L, "wq", cur)
		local st = { H * 2 * D * f4, 2 * D * f4 }
		q = g:view(qg, { n, H, D }, st)
		gate = g:view(qg, { n, H, D }, st, D * f4)
	else
		q = g:reshape(mm(g, L, "wq", cur), { n, H, D })
	end
	if L.q_norm then
		q = g:mul(g:rms_norm(q, eps), L.q_norm)
	end
	q = g:rope(q, pos, rope, L.freqs)
	if not L.wk then
		return q, nil, nil, gate
	end
	local kraw = mm(g, L, "wk", cur)
	k = g:reshape(kraw, { n, L.Hkv, D })
	v = g:reshape(L.wv and mm(g, L, "wv", cur) or kraw, { n, L.Hkv, D })
	if hp.gemma then
		v = g:rms_norm(v, eps)
	end
	if L.k_norm then
		k = g:mul(g:rms_norm(k, eps), L.k_norm)
	end
	return q, g:rope(k, pos, rope, L.freqs), v, gate
end

--- Build attention on queries, keys and values.
-- This is a plain function.
-- @static
-- @param g the graph
-- @param q queries (n, H, D)
-- @param k keys (n_kv, Hkv, D)
-- @param v values (n_kv, Hkv, D)
-- @param mask the mask (n, n_kv), f16 for flash attention
-- @tparam number scale the scale on q . k
-- @tparam[opt] boolean flash true for flash attention
-- @param[opt] rv the V rotation from `lgml.kv.rot_v`, to undo
-- @return the output (n, H * D)
function M.attend(g, q, k, v, mask, scale, flash, rv)
	local n, H, D = q:shape()[1], q:shape()[2], q:shape()[3]
	q = g:permute(q, { 2, 1, 3 })
	k = g:permute(k, { 2, 1, 3 })
	if flash then
		v = g:permute(v, { 2, 1, 3 })
		local o = g:flash_attn(q, kv.for_flash(g, k), kv.for_flash(g, v), mask, scale)
		return g:reshape(kv.rotate(g, o, rv), { n, H * D })
	end
	-- v as (kv_heads, dim, n_kv)
	v = g:cont(g:permute(v, { 2, 3, 1 }))
	local kq = g:soft_max(g:mul_mat(k, q), mask, scale)
	local o = g:mul_mat(v, kq)
	return g:reshape(kv.rotate(g, g:cont(g:permute(o, { 2, 1, 3 })), rv), { n, H * D })
end

-- Attention of layer il with the cache. shared holds the keys and
-- values of earlier layers in this graph, for layers that share them.
local function attention(hp, g, inp, il, L, q, k, v, shared)
	local c, n, D = inp.cache, inp.n, L.D
	local rk, rv = kv.rot_k(c, D), kv.rot_v(c, D)
	if c then
		q = kv.rotate(g, q, rk)
	end
	if L.kv_src then
		k, v = shared[L.kv_src].k, shared[L.kv_src].v
	elseif c then
		k, v = kv.rotate(g, k, rk), kv.rotate(g, v, rv)
		local w = L.Hkv * D
		local rows = L.swa and inp.rows_swa or inp.rows
		local kw = g:set_rows(c.k[il], g:reshape(k, { n, w }), rows)
		local vw = g:set_rows(c.v[il], g:reshape(v, { n, w }), rows)
		k, v = kv.views(g, c, kw, vw, L.swa and inp.n_swa_attn or inp.n_attn, L.Hkv, D)
	end
	shared[il] = { k = k, v = v }
	local mask = L.swa and inp.mask_swa or inp.mask
	return M.attend(g, q, k, v, mask, hp.attn_scale or 1 / math.sqrt(D), inp.flash_attn, rv)
end

--- Build the output projection, post norm and residual of attention.
-- This is a plain function.
-- @static
-- @param g the graph
-- @tparam table hp the config
-- @tparam table L the layer
-- @param o the attention output (n, n_head * D)
-- @param gate the output gate from `qkv`, or nil
-- @param x the layer input
-- @return the attention block output (n, n_embd)
function M.attn_out(g, hp, L, o, gate, x)
	if gate then
		local n = o:shape()[1]
		o = g:mul(o, g:sigmoid(g:reshape(g:cont(gate), { n, hp.n_head * L.D })))
	end
	o = mm(g, L, "wo", o)
	if L.post_attn_norm then
		o = g:mul(g:rms_norm(o, hp.eps), L.post_attn_norm)
	end
	return g:add(o, x)
end

--- Build the routed experts of one layer.
-- The router picks the top n_used experts; their weights sum to 1
-- when hp.norm_topk is set. Gemma routes on the rms-normed residual.
-- This is a plain function.
-- @static
-- @param g the graph
-- @tparam table hp the config
-- @tparam table L the layer
-- @param x the residual (n, n_embd)
-- @param h the normed expert input (n, n_embd)
-- @tparam[opt] table taps gets "ffn_moe_topk" and "ffn_moe_weights"
-- @tparam[opt] string sfx the tap name suffix
-- @return the weighted sum of the chosen expert outputs (n, n_embd)
function M.moe(g, hp, L, x, h, taps, sfx)
	local n, E, NE, K = x:shape()[1], hp.n_embd, hp.n_expert, hp.n_used
	local r_in = h
	if hp.gemma then
		r_in = g:scale(g:rms_norm(x, hp.eps), 1 / math.sqrt(E))
		if L.router_scale then
			r_in = g:mul(r_in, L.router_scale)
		end
	end
	local logits = mm(g, L, "gate_inp", r_in)
	local probs = hp.gating == M.gating.SIGMOID and g:sigmoid(logits) or g:soft_max(logits)
	local sel = g:argsort_top_k(probs, K)
	local probs3 = g:reshape(probs, { n, NE, 1 })
	local w = g:reshape(g:get_rows(probs3, sel), { n, K })
	if hp.norm_topk then
		w = g:div(w, g:clamp(g:sum_rows(w), 6.103515625e-5, math.huge))
	end
	w = g:reshape(w, { n, K, 1 })
	if L.down_exps_scale then
		local s = g:tile(g:reshape(L.down_exps_scale, { 1, NE, 1 }), probs3)
		w = g:mul(w, g:get_rows(s, sel))
	end
	local x3 = g:reshape(h, { n, 1, E })
	local gate, up
	if L.gate_up_exps then
		local gu = g:mul_mat_id(L.gate_up_exps, x3, sel)
		local F = gu:shape()[3] // 2
		local st = gu:strides()
		gate = g:view(gu, { n, K, F }, { st[1], st[2] })
		up = g:view(gu, { n, K, F }, { st[1], st[2] }, lgml.row_size("f32", F))
	else
		gate = g:mul_mat_id(L.gate_exps, x3, sel)
		up = g:mul_mat_id(L.up_exps, x3, sel)
	end
	local act = hp.gemma and g:geglu_split(gate, up) or g:swiglu_split(gate, up)
	local ex = g:mul(g:mul_mat_id(L.down_exps, act, sel), w)
	if taps then
		taps["ffn_moe_topk" .. sfx] = sel
		taps["ffn_moe_weights" .. sfx] = w
	end
	local st = ex:strides()
	local out = g:view(ex, { n, E }, { st[1] })
	for j = 1, K - 1 do
		out = g:add(out, g:view(ex, { n, E }, { st[1] }, j * st[2]))
	end
	if K == 1 then
		out = g:cont(out)
	end
	return out
end

--- Build the FFN block of one layer, with its residual.
-- A layer has a dense FFN (SwiGLU, or GeGLU on gemma), routed experts,
-- or both. A shared expert adds its output times sigmoid(gate . h).
-- This is a plain function.
-- @static
-- @param g the graph
-- @tparam table hp the config
-- @tparam table L the layer
-- @param x the attention block output (n, n_embd)
-- @tparam[opt] table taps named nodes to fill
-- @tparam[opt] string sfx the tap name suffix
-- @return the block output (n, n_embd)
function M.ffn(g, hp, L, x, taps, sfx)
	local eps = hp.eps
	local h = g:mul(g:rms_norm(x, eps), L.ffn_norm)
	local out
	if L.gate then
		local gate, up = mm(g, L, "gate", h), mm(g, L, "up", h)
		local act = hp.gemma and g:geglu_split(gate, up) or g:mul(g:silu(gate), up)
		out = mm(g, L, "down", act)
	end
	if L.gate_inp then
		if out and L.post_ffn_norm_1 then
			out = g:mul(g:rms_norm(out, eps), L.post_ffn_norm_1)
		end
		local e_in = L.pre_ffn_norm_2 and g:mul(g:rms_norm(x, eps), L.pre_ffn_norm_2) or h
		local ex = M.moe(g, hp, L, x, e_in, taps, sfx)
		if L.post_ffn_norm_2 then
			ex = g:mul(g:rms_norm(ex, eps), L.post_ffn_norm_2)
		end
		out = out and g:add(out, ex) or ex
		if L.shexp_gate then
			local sh = g:swiglu_split(mm(g, L, "gate_shexp", h), mm(g, L, "up_shexp", h))
			sh = mm(g, L, "down_shexp", sh)
			out = g:add(out, g:mul(sh, g:sigmoid(g:mul_mat(L.shexp_gate, h))))
		end
	end
	if L.post_ffn_norm then
		out = g:mul(g:rms_norm(out, eps), L.post_ffn_norm)
	end
	if taps then
		taps["ffn_out" .. sfx] = out
	end
	return g:add(out, x)
end

--- Build the output head: final norm, projection and soft cap.
-- This is a plain function.
-- @static
-- @param g the graph
-- @tparam table hp the config
-- @param output_norm the final norm weight
-- @param output the output projection
-- @param x the hidden state (n, n_embd)
-- @tparam[opt] table lora adapter terms on the output projection, as
-- `layer.lora` entries hold
-- @return the logits (n, n_vocab)
-- @return the normed hidden state
function M.head(g, hp, output_norm, output, x, lora)
	x = g:mul(g:rms_norm(x, hp.eps), output_norm)
	local logits = mm(g, { output = output, lora = lora and { output = lora } }, "output", x)
	if (hp.softcap or 0) > 0 then
		logits = g:scale(g:tanh(g:scale(logits, 1 / hp.softcap)), hp.softcap)
	end
	return logits, x
end

-- q, k, v and z from the old fused ssm_in. Per key head it holds
-- [q | k | v | z], with v and z for that head's r value heads.
local function split_qkvz(g, hp, L, x, n)
	local nk, hk, hv = hp.n_k_heads, hp.head_k, hp.head_v
	local r = hp.n_v_heads // nk
	local f4 = lgml.row_size("f32", 1)
	local m = g:reshape(mm(g, L, "ssm_in", x), { n, nk, 2 * hk + 2 * hv * r })
	local s = m:strides()
	local function part(off, w)
		local t = g:cont(g:view(m, { n, nk, w }, { s[1], s[2] }, off * f4))
		return g:reshape(t, { n, nk * w })
	end
	local q, k = part(0, hk), part(hk, hk)
	local v, z = part(2 * hk, hv * r), part(2 * hk + hv * r, hv * r)
	return g:concat(g:concat(q, k, 2), v, 2), z
end

-- beta and alpha (n, Hv). The fused ssm_ba holds [beta | alpha] per key
-- head, for that head's value heads.
local function beta_alpha(g, hp, L, x, n)
	if not L.ba then
		return mm(g, L, "beta", x), mm(g, L, "alpha", x)
	end
	local nk, nv = hp.n_k_heads, hp.n_v_heads
	local r = nv // nk
	local ba = g:reshape(mm(g, L, "ba", x), { n, nk, 2 * r })
	local s = ba:strides()
	local b = g:cont(g:view(ba, { n, nk, r }, { s[1], s[2] }))
	local a = g:cont(g:view(ba, { n, nk, r }, { s[1], s[2] }, lgml.row_size("f32", r)))
	return g:reshape(b, { n, nv }), g:reshape(a, { n, nv })
end

-- Qwen3-Next repeats each key head for its r value heads in turn.
-- Qwen3.5 cycles through the key heads, which the delta net op does.
local function repeat_heads(g, hp, t, n)
	local nk, hk = hp.n_k_heads, hp.head_k
	local r = hp.n_v_heads // nk
	if not hp.ssm_tile or r == 1 then
		return t
	end
	local one = g:reshape(t, { n, nk, 1, hk })
	local rep = one
	for _ = 2, r do
		rep = g:concat(rep, one, 3)
	end
	return g:reshape(rep, { 1, n, nk * r, hk })
end

-- A gated delta net layer on the normed input x. The conv and delta
-- net states come from the cache; the graph writes the new states back.
local function linear_attention(hp, g, inp, L, il, x, taps, sfx)
	local n, c = inp.n, inp.cache
	local nk, nv, hk, hv = hp.n_k_heads, hp.n_v_heads, hp.head_k, hp.head_v
	local kd, vd, cd = nk * hk, nv * hv, conv_dim(hp)
	local f4 = lgml.row_size("f32", 1)
	local qkv, z
	if L.qkv then
		qkv, z = mm(g, L, "qkv", x), mm(g, L, "qkv_gate", x)
	else
		qkv, z = split_qkvz(g, hp, L, x, n)
	end
	local beta, alpha = beta_alpha(g, hp, L, x, n)
	beta = g:reshape(g:sigmoid(beta), { 1, n, nv, 1 })
	local gate = g:mul(g:softplus(g:add(alpha, L.dt)), L.a)
	gate = g:reshape(gate, { 1, n, nv, 1 })

	-- Conv over the saved d_conv - 1 columns and the new ones.
	local xin = g:concat(c.conv[il], g:transpose(qkv), 3)
	local s = xin:strides()
	local last = g:view(xin, { 1, cd, hp.d_conv - 1 }, { s[1], s[2] }, n * f4)
	g:expand(g:cpy(last, c.conv[il]))
	local conv = g:silu(g:ssm_conv(xin, L.conv))
	taps["conv_output_silu" .. sfx] = conv

	local st = { n * cd * f4, cd * f4 }
	local q = g:view(conv, { 1, n, nk, hk }, { st[1], st[2], hk * f4 })
	local k = g:view(conv, { 1, n, nk, hk }, { st[1], st[2], hk * f4 }, kd * f4)
	local v = g:view(conv, { 1, n, nv, hv }, { st[1], st[2], hv * f4 }, 2 * kd * f4)
	q = repeat_heads(g, hp, g:l2_norm(q, hp.eps), n)
	k = repeat_heads(g, hp, g:l2_norm(k, hp.eps), n)

	local out = g:gated_delta_net(q, k, v, gate, beta, c.ssm[il])
	local o = g:view(out, { 1, n, nv, hv }, { n * vd * f4, vd * f4, hv * f4 })
	local ns = g:view(out, { 1, nv, hv, hv }, { nv * hv * hv * f4, hv * hv * f4, hv * f4 },
	    n * vd * f4)
	g:expand(g:cpy(ns, c.ssm[il]))
	taps["attn_output" .. sfx] = o

	z = g:reshape(z, { 1, n, nv, hv })
	o = g:mul(g:mul(g:rms_norm(o, hp.eps), L.ssm_norm), g:silu(z))
	return mm(g, L, "ssm_out", g:reshape(o, { n, vd }))
end

-- Per-layer inputs (n_layer, n, n_embd_per_layer) from the token ids
-- and the scaled embedding x.
local function per_layer_inputs(self, g, inp, x)
	local hp, n = self.hp, inp.n
	local P, NL = hp.n_embd_per_layer, hp.n_layer
	local tok = g:reshape(g:get_rows(self.ple_embd, inp.ids), { n, NL, P })
	tok = g:scale(tok, math.sqrt(P))
	local proj = g:scale(g:mul_mat(self.ple_proj, x), 1 / math.sqrt(hp.n_embd))
	proj = g:mul(g:rms_norm(g:reshape(proj, { n, NL, P }), hp.eps), self.ple_norm)
	local all = g:scale(g:add(proj, tok), 1 / math.sqrt(2))
	return g:cont(g:permute(all, { 2, 1, 3 }))
end

--- Build the forward graph.
-- Recurrent state writes are in the graph; computing the logits runs
-- them.
-- @param g the graph
-- @tparam table inp inputs from `inputs`
-- @return logits (n, n_vocab), or (1, n_vocab) when inp.logits is "last";
-- with inp.hidden, the hidden state (n, n_embd) after that many layers
-- @treturn table named nodes for comparison, such as "embd",
-- "ffn_out-0", "l_out-0" and "result_norm"
function M:forward(g, inp)
	local hp, n = self.hp, inp.n
	local taps, shared = {}, {}
	local x
	if inp.embd then
		x = inp.embd
	else
		x = g:get_rows(self.tok_embd, inp.ids)
		if hp.embd_scale then
			x = g:scale(x, hp.embd_scale)
		end
	end
	taps.embd = x
	local ple = self.ple_embd and per_layer_inputs(self, g, inp, x)
	for il, L in ipairs(self.layers) do
		local sfx = "-" .. (il - 1)
		if L.recr then
			local cur = g:mul(g:rms_norm(x, hp.eps), L.attn_norm)
			x = g:add(linear_attention(hp, g, inp, L, il, cur, taps, sfx), x)
		else
			local q, k, v, gate = M.qkv(g, hp, L, x, inp.pos)
			local o = attention(hp, g, inp, il, L, q, k, v, shared)
			x = M.attn_out(g, hp, L, o, gate, x)
		end
		x = M.ffn(g, hp, L, x, taps, sfx)
		if ple then
			local P = hp.n_embd_per_layer
			local e = g:view(ple, { n, P }, { P * 4 }, (il - 1) * n * P * 4)
			local t = g:mul(g:gelu(mm(g, L, "ple_gate", x)), e)
			t = g:mul(g:rms_norm(mm(g, L, "ple_proj", t), hp.eps), L.ple_post_norm)
			x = g:add(x, t)
		end
		if L.out_scale then
			x = g:mul(x, L.out_scale)
		end
		taps["l_out" .. sfx] = x
		if il == inp.hidden then
			return x, taps
		end
	end
	if inp.logits == "last" and n > 1 then
		local row = lgml.row_size("f32", hp.n_embd)
		x = g:view(x, { 1, hp.n_embd }, nil, (n - 1) * row)
	end
	local logits
	logits, taps.result_norm = M.head(g, hp, self.output_norm, self.output, x, self.lora_out)
	return logits, taps
end

--- Run ids through the model at cache.n_past and advance the cache.
-- A cache with a ring takes passes of at most cache.ubatch tokens.
-- Passes go through `lgml.replay`. The call computes only the last row
-- of logits.
-- @tparam table cache a cache from `cache`
-- @tparam {integer,...} ids the token ids
-- @tparam[opt] table opts options for `inputs`
-- @treturn integer the greedy next token id
-- @raise when the context is full
function M:step(cache, ids, opts)
	local o = { logits = "last" }
	for k, v in pairs(opts or {}) do
		o[k] = v
	end
	local ub = cache.ubatch or #ids
	local id
	for s = 1, #ids, ub do
		local part = table.move(ids, s, math.min(#ids, s + ub - 1), 1, {})
		local best = replay.run(self, cache, part, o, function(g, logits)
			return g:argmax(logits)
		end)
		id = best:get()[1]
	end
	return id
end

return M
