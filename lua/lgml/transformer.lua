--- Config-driven decoder for the gguf text models.
-- One module builds llama, qwen3, qwen3moe, qwen3next, qwen35,
-- qwen35moe, qwen4exp (see `lgml.qwen4exp`) and gemma4 graphs.
-- `config` reads the gguf metadata into a table; the blocks read that
-- table and the weights that exist.
-- A layer is gated delta net or attention, then a dense FFN, routed
-- experts, or both. Shapes are in torch order.
-- @module lgml.transformer
local lgml = require("lgml")
local kv = require("lgml.kv")
local kvpool = require("lgml.kvpool")
local placement = require("lgml.placement")
local replay = require("lgml.replay")
local budget = require("lgml.budget")
local log = require("lgml.log")
local q4 = require("lgml.qwen4exp")

local LOG = log.new("kv")

local M = {}
M.__index = M

-- Attention reads the cache in blocks of PAD rows, so decode steps keep
-- one graph shape while the span stays inside a block.
local PAD = 256
-- The default count of tokens per pass on a sliding-window cache.
local UBATCH = 512
-- Cells per block of a KV pool. It divides PAD, so a pool window in
-- PAD steps holds whole blocks.
local BLOCK = 64

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
	qwen4exp = { qk_norm = true, rope_ctx = true, hybrid = true, hc = true },
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
-- qwen35moe, qwen4exp and gemma4
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
	if fam.hc then
		q4.config(hp, k)
	end
	return hp
end

--- The tensors of a gguf that `load` puts on a backend, for weight
-- placement plans.
-- @tparam table info the gguf info, as from `lgml.gguf`
-- @tparam table hp the config, from `config`
-- @treturn table the tensor list
function M.weight_tensors(info, hp)
	return hp.hc and q4.weight_tensors(info.tensors, hp) or info.tensors
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
-- opts.placement: rules from `lgml.placement.rules`; the weights they
-- place on the CPU load to `be:cpu()`, and so do the caches of layers
-- whose attn_norm is there (default: all on be).
-- opts.pinned: false to put those CPU weights in plain memory, not in
-- pinned host memory of the device (default true).
-- @treturn Model the model
-- @raise when the file does not open, a weight is missing, the
-- architecture is not supported, or the placement moves weights and be
-- has no CPU backend
function M.load(be, path, opts)
	opts = opts or {}
	local info = assert(lgml.gguf(path))
	local hp = M.config(info.meta)
	if hp.hc then
		return q4.load(be, path, opts, info, hp)
	end
	local cpu = opts.placement and placement.plan(opts.placement, info.tensors, hp.n_layer)
	if cpu and not be:cpu() then
		error(path .. ": the placement moves weights to the CPU; open the backend with cpu = true")
	end
	local w = assert(be:load(path, cpu and { cpu = cpu, pinned = opts.pinned }))
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
		L.cpu = L.attn_norm:backend() ~= be or nil
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
	if self.ngram then
		self.ngram:close()
	end
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

-- Caches get an id, so a batch graph knows which caches it writes.
local next_cache_id = 0

-- Head sizes of the attention layers, for the cache type check.
local function attn_dims(self)
	local dims = {}
	for _, L in ipairs(self.layers) do
		if not L.recr then
			dims[#dims + 1] = L.D
		end
	end
	return dims
end

-- Rows for cap cells at first: at least PAD, in PAD steps, at most n.
local function first_cap(n, initial)
	if initial and initial < n then
		return math.min(n, math.max(PAD, (initial + PAD - 1) // PAD * PAD))
	end
	return n
end

-- Add the per-sequence parts to cache c: rings, recurrent states, and
-- with global true the full-attention rows (cap of them).
local function add_layers(self, c, opts, global)
	local hp, be = self.hp, self.be
	if hp.n_swa then
		c.ubatch = opts.ubatch or UBATCH
		c.ring = math.min(c.n_ctx, hp.n_swa - 1 + c.ubatch)
		c.hi = 0
	end
	if self.recurrent then
		c.rs_pos, c.conv, c.ssm = 0, {}, {}
	end
	for il, L in ipairs(self.layers) do
		local lbe = L.cpu and be:cpu() or be
		if L.recr then
			c.conv[il] = lbe:tensor("f32", { 1, conv_dim(hp), hp.d_conv - 1 })
			c.ssm[il] = lbe:tensor("f32", { 1, hp.n_v_heads, hp.head_v, hp.head_v })
			c.conv[il]:zero()
			c.ssm[il]:zero()
		elseif not L.kv_src and (L.swa or global) then
			local rows = L.swa and c.ring or c.cap
			c.k[il] = lbe:tensor(c.type_k, { rows, L.Hkv * L.D })
			c.v[il] = lbe:tensor(c.type_v, { rows, L.Hkv * L.D })
			-- Rows past n_past are read under a -inf mask; zero keeps
			-- them finite.
			c.k[il]:zero()
			c.v[il]:zero()
		end
	end
	if hp.hc then
		q4.add_cache(self, c, c.cap, global)
	end
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
-- opts.initial: rows the full-attention layers get at first (default
-- n_ctx). A pass past them grows the cache to twice the rows, or to the
-- rows the pass needs, at most n_ctx; see `grow`.
-- @treturn table cache with n_ctx, cap (the rows the full-attention
-- layers hold now), n_past (0), type_k, type_v, k, v;
-- ring, ubatch and hi with sliding layers; rs_pos (the position the
-- states are at), conv and ssm with recurrent layers
function M:cache(n_ctx, opts)
	opts = opts or {}
	local tk, tv = kv.types_of(opts, attn_dims(self))
	next_cache_id = next_cache_id + 1
	local c = { n_ctx = n_ctx, cap = first_cap(n_ctx, opts.initial), n_past = 0, be = self.be,
		type_k = tk, type_v = tv, k = {}, v = {}, id = next_cache_id, gen = 0 }
	add_layers(self, c, opts, true)
	return c
end

--- Make a KV pool: one buffer of full-attention rows that several
-- sequences share.
-- The pool holds the K and V rows of the global attention layers in
-- blocks of `block` cells. A sequence from `seq` takes blocks as it
-- grows; attention of a sequence reads a window of the pool under a
-- mask that shows only the cells of that sequence. Rings and recurrent
-- states stay per sequence.
-- @tparam integer n_cells the pool size in cells (rounded up to 256)
-- @tparam[opt] table opts options:
-- opts.type, opts.type_k, opts.type_v: cache types from `lgml.kv`
-- (default f16).
-- opts.initial: cells the tensors hold at first (default all). The
-- pool doubles them when no free block is left, up to n_cells.
-- opts.block: cells per block (default 64); it must divide 256.
-- @treturn table the pool: n_ctx (cells), cap (cells the tensors hold
-- now), block, type_k, type_v, k, v (by layer) and alloc (the
-- `lgml.kvpool` bookkeeping)
function M:pool(n_cells, opts)
	opts = opts or {}
	local B = opts.block or BLOCK
	assert(B >= 1 and PAD % B == 0, "pool: the block size must divide " .. PAD)
	local tk, tv = kv.types_of(opts, attn_dims(self))
	local size = (n_cells + PAD - 1) // PAD * PAD
	local cap = first_cap(size, opts.initial)
	next_cache_id = next_cache_id + 1
	local p = { n_ctx = size, cap = cap, block = B, be = self.be, type_k = tk, type_v = tv,
		k = {}, v = {}, id = next_cache_id, gen = 0,
		alloc = kvpool.new({ blocks = size // B, size = B, cap = cap // B }),
		seqs = setmetatable({}, { __mode = "k" }) }
	for il, L in ipairs(self.layers) do
		if not L.recr and not L.kv_src and not L.swa then
			local lbe = L.cpu and self.be:cpu() or self.be
			p.k[il] = lbe:tensor(tk, { cap, L.Hkv * L.D })
			p.v[il] = lbe:tensor(tv, { cap, L.Hkv * L.D })
			p.k[il]:zero()
			p.v[il]:zero()
		end
	end
	if self.hp.hc then
		q4.add_cache(self, p, cap, true)
	end
	return p
end

--- Make a cache for one sequence on a pool.
-- It works as a cache from `cache` does, but the global attention rows
-- live in pool blocks: `blocks` lists the block ids in position order,
-- and `keys` the keys of its full blocks that have one (see
-- `lgml.kvpool`). Rings and recurrent states belong to the sequence.
-- @tparam table pool a pool from `pool`
-- @tparam[opt] table opts options: n_ctx (the most positions, default
-- the pool size) and ubatch (as for `cache`)
-- @treturn table the cache, with pool and blocks
function M:seq(pool, opts)
	opts = opts or {}
	next_cache_id = next_cache_id + 1
	local c = { pool = pool, blocks = {}, keys = {}, n_ctx = math.min(opts.n_ctx or pool.n_ctx, pool.n_ctx),
		n_past = 0, be = pool.be, type_k = pool.type_k, type_v = pool.type_v, k = {}, v = {},
		id = next_cache_id, gen = 0 }
	add_layers(self, c, opts, false)
	pool.seqs[c] = true
	return c
end

-- Drop the last batch graph. With done true, its output rows close too.
-- With done "detach", the rows get their own storage (see
-- `tensor:detach`) and the compute buffers go now.
local function drop_batch(self, done)
	local r = self.batch_replay
	self.batch_replay = nil
	if done == "detach" then
		for _, t in ipairs(r.rows) do
			t:detach()
		end
		done = false
	end
	r.g:close()
	for _, part in ipairs({ r.inp, table.unpack(r.inp.batch) }) do
		for _, t in pairs(part) do
			if type(t) == "userdata" then
				t:close()
			end
		end
	end
	if done then
		for _, t in ipairs(r.rows) do
			t:close()
		end
	end
end

--- Drop the graph of the last batch pass.
-- The next batch pass builds a graph again. Its compute buffers go when
-- its logits rows close. With done false, a request can still read a
-- row; a garbage collection then frees the buffers when none does.
-- @tparam[opt] boolean done true to close the rows now (default true)
function M:drop_batch(done)
	if self.batch_replay then
		drop_batch(self, done ~= false)
		if done == false then
			collectgarbage()
		end
	end
end

-- Drop every graph that reads the tensors of pool p.
-- The outputs stay readable in their own storage, and the compute
-- buffers go now: a pass after a pool change plans larger graphs.
local function drop_pool_graphs(self, p)
	for c in pairs(p.seqs) do
		replay.drop(c, "detach")
	end
	local r = self.batch_replay
	if r and r.pools and r.pools[p] then
		drop_batch(self, "detach")
	end
end

--- Free a pool sequence now: its blocks, graphs and state tensors.
-- The sequence is not valid after this.
-- @tparam table c a cache from `seq`
function M:free_seq(c)
	self:truncate(c, 0)
	replay.drop(c)
	local r = self.batch_replay
	if r then
		for _, b in ipairs(r.inp.batch) do
			if b.cache == c then
				drop_batch(self)
				break
			end
		end
	end
	for _, part in ipairs({ c.k, c.v, c.conv or {}, c.ssm or {}, c.pconv or {}, { c.mtp_h } }) do
		for il, t in pairs(part) do
			t:close()
			part[il] = nil
		end
	end
	c.pool.seqs[c] = nil
end

-- Raise a memory budget error when the tensors cannot grow to cap
-- rows. A grow holds each old tensor until its new one has the rows,
-- so the largest old tensor counts too.
local function check_grow(list, cap, what)
	local by = {}
	for _, t in ipairs(list) do
		local rows = t:shape()[1]
		local be = t:backend()
		local e = by[be] or { add = 0, top = 0 }
		by[be] = e
		e.add = e.add + (cap - rows) * (t:nbytes() // rows)
		e.top = math.max(e.top, t:nbytes())
	end
	for be, e in pairs(by) do
		local ok, err = lgml.memory_check(be, e.add + e.top, what)
		if not ok then
			error(err, 0)
		end
	end
end

-- Give pool p cap cells, or twice its cells up to p.n_ctx. The new
-- tensors get the old cells that fit, when a block holds cells.
local function pool_grow(self, p, cap)
	cap = cap or math.min(p.n_ctx, p.cap * 2)
	local list = {}
	for _, part in ipairs({ p.k, p.v, p.x or {} }) do
		for _, t in pairs(part) do
			list[#list + 1] = t
		end
	end
	if cap > p.cap then
		check_grow(list, cap, ("the KV pool from %d to %d cells"):format(p.cap, cap))
	end
	drop_pool_graphs(self, p)
	local empty = p.alloc.used == 0 and p.alloc.cached == 0
	local buf
	for _, part in ipairs({ p.k, p.v, p.x or {} }) do
		for il, old in pairs(part) do
			local bytes = math.min(old:nbytes(), cap * (old:nbytes() // old:shape()[1]))
			local new = old:backend():tensor(old:type(), { cap, old:shape()[2] })
			new:zero()
			if not empty then
				if not buf or #buf < bytes then
					if buf then
						buf:close()
					end
					buf = lgml.hostbuf(bytes)
				end
				old:to_host(buf, 0, 0, bytes)
				new:from_host(buf, 0, 0, bytes)
			end
			old:close()
			part[il] = new
		end
	end
	if buf then
		buf:close()
	end
	LOG:info(cap > p.cap and "pool grew" or "pool shrank", "cells", cap, "from", p.cap, "size", p.n_ctx)
	p.cap = cap
	p.gen = p.gen + 1
	p.alloc:set_cap(cap // p.block)
end

--- Set the cells that the tensors of a pool hold.
-- The new tensors get the old cells that fit. Blocks at or past the new
-- size must be free.
-- @tparam table pool the pool, from `pool`
-- @tparam integer cells the cells, a multiple of the block size, at most
-- pool.n_ctx
function M:pool_resize(pool, cells)
	assert(cells % pool.block == 0 and cells <= pool.n_ctx, "pool_resize: bad cell count")
	for b in pairs(pool.alloc.refs) do
		assert(b < cells // pool.block, "pool_resize: a block past the new size is in use")
	end
	for b in pairs(pool.alloc.keys) do
		assert(b < cells // pool.block, "pool_resize: a block past the new size is cached")
	end
	if cells ~= pool.cap then
		pool_grow(self, pool, cells)
	end
end

-- A block for pool p after block min - 1 of sequence c: a free one,
-- else a new one when the pool can grow, else a cached one, else one
-- that pool.pressure frees for c. The second result is true when the
-- pool grew. A grow past the memory budget raises its error when no
-- other block is there.
local function new_block(self, p, c, min)
	local b = p.alloc:take(min)
	if b then
		return b, false
	end
	local grow_err
	if p.cap < p.n_ctx then
		local ok, err = pcall(pool_grow, self, p)
		if ok then
			return p.alloc:take(min), true
		elseif not budget.is_budget(err) then
			error(err, 0)
		end
		grow_err = err
	end
	b = p.alloc:evict()
	if not b and p.pressure then
		p.pressure(p, c)
		b = p.alloc:take(min) or p.alloc:evict()
	end
	if not b and grow_err then
		error(grow_err, 0)
	end
	return b, false
end

-- Give sequence c the blocks for positions [0, n). The new blocks go
-- in cell order, so attention reads the cells of a pass in position
-- order. Returns true when the pool grew.
local function reserve(self, c, n)
	local p = c.pool
	local B, grew, first = p.block, false, #c.blocks + 1
	while #c.blocks * B < n do
		local last = c.blocks[#c.blocks]
		local b, g = new_block(self, p, c, last and last + 1)
		grew = grew or g
		if not b then
			error(("kv pool full: %d of %d cells in use"):format(p.alloc.used * B, p.n_ctx))
		end
		c.blocks[#c.blocks + 1] = b
	end
	if #c.blocks > first then
		local new = table.move(c.blocks, first, #c.blocks, 1, {})
		table.sort(new)
		table.move(new, 1, #new, first, c.blocks)
	end
	return grew
end

--- Copy the first n cells of block src to block dst of a pool, in
-- every layer.
-- @tparam table pool the pool
-- @tparam integer src the block to read
-- @tparam integer dst the block to write
-- @tparam integer n the cell count, at most the block size
function M.copy_block(pool, src, dst, n)
	local buf
	for _, part in ipairs({ pool.k, pool.v, pool.x or {} }) do
		for _, t in pairs(part) do
			local row = t:nbytes() // t:shape()[1]
			if not buf or #buf < n * row then
				if buf then
					buf:close()
				end
				buf = lgml.hostbuf(n * row)
			end
			t:to_host(buf, 0, src * pool.block * row, n * row)
			t:from_host(buf, 0, dst * pool.block * row, n * row)
		end
	end
	if buf then
		buf:close()
	end
end

-- The K and V tensors of a pool, in a fixed order.
local function pool_tensors(pool)
	local list = {}
	for il in pairs(pool.k) do
		list[#list + 1] = il
	end
	table.sort(list)
	local out = {}
	for _, il in ipairs(list) do
		out[#out + 1] = pool.k[il]
		out[#out + 1] = pool.v[il]
	end
	local xs = {}
	for il in pairs(pool.x or {}) do
		xs[#xs + 1] = il
	end
	table.sort(xs)
	for _, il in ipairs(xs) do
		out[#out + 1] = pool.x[il]
	end
	return out
end

--- Bytes of the cells of one pool block, in all layers.
-- @tparam table pool the pool
-- @treturn integer bytes
function M.block_bytes(pool)
	local n = 0
	for _, t in ipairs(pool_tensors(pool)) do
		n = n + pool.block * (t:nbytes() // t:shape()[1])
	end
	return n
end

--- Copy the cells of a pool block to host memory, or back.
-- @tparam table pool the pool
-- @tparam integer b the block id
-- @param buf an `lgml.hostbuf` of `block_bytes` bytes
-- @tparam boolean out true to copy to buf, false to copy from it
function M.copy_block_host(pool, b, buf, out)
	local off = 0
	for _, t in ipairs(pool_tensors(pool)) do
		local n = pool.block * (t:nbytes() // t:shape()[1])
		if out then
			t:to_host(buf, off, b * n, n)
		else
			t:from_host(buf, off, b * n, n)
		end
		off = off + n
	end
end

--- Take a block of a pool for sequence c, as its passes do: a free
-- block past min when there is one, else a new one when the pool can
-- grow, else a cached one, else one that pool.pressure frees.
-- @tparam table pool the pool
-- @tparam[opt] table c the sequence that asks
-- @tparam[opt] integer min the lowest block wanted
-- @treturn integer the block id, with one reference
-- @raise when the pool is full
function M:take_block(pool, c, min)
	local b = new_block(self, pool, c, min)
	if not b then
		error(("kv pool full: %d of %d cells in use"):format(pool.alloc.used * pool.block, pool.n_ctx))
	end
	return b
end

--- Give back the blocks of a pool sequence past its first n positions.
-- It also sets n_past to n when n_past is larger, and drops the block
-- keys past n. When n ends inside a block that other sequences share
-- or that has a key, the sequence gets a copy of that block: later
-- writes must not change the shared cells.
-- @tparam table cache a cache from `seq`
-- @tparam integer n the positions to keep
function M:truncate(cache, n)
	local p = cache.pool
	local B = p.block
	local keep = (n + B - 1) // B
	for i = #cache.blocks, keep + 1, -1 do
		p.alloc:unref(cache.blocks[i])
		cache.blocks[i] = nil
	end
	local keys = cache.keys
	for i = #keys, n // B + 1, -1 do
		keys[i] = nil
	end
	local last = cache.blocks[keep]
	if n % B ~= 0 and last and (p.alloc:refs_of(last) > 1 or p.alloc:key_of(last)) then
		local prev = cache.blocks[keep - 1]
		local b = new_block(self, p, cache, prev and prev + 1)
		if not b then
			error(("kv pool full: %d of %d cells in use"):format(p.alloc.used * B, p.n_ctx))
		end
		M.copy_block(p, last, b, n % B)
		cache.blocks[keep] = b
		p.alloc:unref(last)
	end
	cache.n_past = math.min(cache.n_past, n)
end

--- Start an empty pool sequence on shared blocks.
-- The sequence takes a reference to each block and holds their
-- positions. n_past becomes the positions of the blocks.
-- @tparam table cache a cache from `seq`, with no blocks
-- @tparam {integer,...} blocks the block ids in position order
-- @tparam {string,...} keys the key of each block
function M:share(cache, blocks, keys)
	assert(#cache.blocks == 0, "share: the sequence has blocks")
	for i, b in ipairs(blocks) do
		cache.pool.alloc:ref(b, cache.prio)
		cache.blocks[i], cache.keys[i] = b, keys[i]
	end
	cache.n_past = #blocks * cache.pool.block
end

-- The pool cells of positions [from, from + n) of a pool sequence.
local function seq_cells(c, from, n)
	local B, bl, r = c.pool.block, c.blocks, {}
	for i = 1, n do
		local p = from + i - 1
		r[i] = bl[p // B + 1] * B + p % B
	end
	return r
end

--- The cells of positions [from, from + n) of a cache: pool cells
-- for a pool sequence, else the positions.
-- @tparam table c the cache
-- @tparam integer from the first position
-- @tparam integer n the count
-- @treturn {integer,...} the cells
function M.cells(c, from, n)
	if c.pool then
		return seq_cells(c, from, n)
	end
	local r = {}
	for i = 1, n do
		r[i] = from + i - 1
	end
	return r
end

-- The pool window that holds positions [0, n_kv) of a pool sequence:
-- the first cell and the width, both in PAD steps.
local function seq_window(c, n_kv)
	local p = c.pool
	local B = p.block
	local nb = (n_kv + B - 1) // B
	local lo, hi = math.huge, 0
	for i = 1, nb do
		local s = c.blocks[i] * B
		if s < lo then
			lo = s
		end
		local e = i < nb and s + B or s + (n_kv - 1) % B + 1
		if e > hi then
			hi = e
		end
	end
	lo = lo // PAD * PAD
	hi = math.min(p.cap, (hi + PAD - 1) // PAD * PAD)
	return lo, hi - lo
end

-- Bytes per position of a QSA layer's indexer keys and block keys.
local function idx_row(hp, L)
	return lgml.row_size("f16", L.idx) + lgml.row_size("f16", L.idx // hp.idx.ratio)
end

--- Bytes a cache from `cache` takes on the device.
-- A layer with `cpu` set keeps its cache on the CPU; its bytes are the
-- third and fourth results.
-- @tparam[opt] table opts the cache options, as for `cache`
-- @treturn integer bytes per position (full-attention layers)
-- @treturn integer bytes that do not grow with n_ctx (rings, recurrent
-- states)
-- @treturn integer bytes per position on the CPU
-- @treturn integer fixed bytes on the CPU
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
	local s = { dev = { 0, 0 }, cpu = { 0, 0 } }
	for _, L in ipairs(self.layers) do
		local p = L.cpu and s.cpu or s.dev
		if L.recr then
			p[2] = p[2] + 4 * (conv_dim(hp) * (hp.d_conv - 1) + hp.n_v_heads * hp.head_v * hp.head_v)
			if L.ple then
				p[2] = p[2] + 4 * hp.hc * hp.n_embd * hp.ple.hist
			end
		elseif not L.kv_src then
			local row = lgml.row_size(tk, L.Hkv * L.D) + lgml.row_size(tv, L.Hkv * L.D)
			if L.idx then
				row = row + idx_row(hp, L)
			end
			if L.swa then
				p[2] = p[2] + ring * row
			else
				p[1] = p[1] + row
			end
		end
	end
	-- Token ids of the cells, for the n-gram input.
	if hp.ple then
		s.cpu[1] = s.cpu[1] + 4
	end
	local mt = self.mtp
	if mt then
		local L = mt.L
		local p = L.cpu and s.cpu or s.dev
		p[1] = p[1] + lgml.row_size(tk, L.Hkv * L.D) + lgml.row_size(tv, L.Hkv * L.D) +
		    (L.idx and idx_row(hp, L) or 0)
		p[2] = p[2] + 4 * hp.hc * hp.n_embd
	end
	return s.dev[1], s.dev[2], s.cpu[1], s.cpu[2]
end

--- Give the full-attention layers room for rows positions.
-- The new tensors get the old rows [0, n_past); the old tensors are
-- freed. The cached replay graph refers to the old tensors, so it is
-- dropped. Rings and recurrent states do not change.
-- A pool sequence from `seq` takes pool blocks for rows positions.
-- @tparam table cache a cache from `cache` or `seq`
-- @tparam integer rows the positions needed, at most cache.n_ctx
-- @treturn boolean true when the cache or its pool grew
function M:grow(cache, rows)
	if cache.pool then
		assert(rows <= cache.n_ctx, "grow: past n_ctx")
		return reserve(self, cache, rows)
	end
	-- A cache made by hand without cap holds all n_ctx rows.
	if rows <= (cache.cap or cache.n_ctx) then
		return false
	end
	assert(rows <= cache.n_ctx, "grow: past n_ctx")
	local cap = math.min(cache.n_ctx, math.max(cache.cap * 2, (rows + PAD - 1) // PAD * PAD))
	-- The tensors that grow, as { table, key } pairs.
	local slots, list = {}, {}
	for il, L in ipairs(self.layers) do
		if cache.k[il] and not L.swa then
			slots[#slots + 1] = { cache.k, il }
			slots[#slots + 1] = { cache.v, il }
		end
	end
	for il in pairs(cache.x or {}) do
		slots[#slots + 1] = { cache.x, il }
	end
	-- Rows of layers past the trunk, such as an MTP layer.
	for il in pairs(cache.k) do
		if not self.layers[il] then
			slots[#slots + 1] = { cache.k, il }
			slots[#slots + 1] = { cache.v, il }
		end
	end
	for i, sl in ipairs(slots) do
		list[i] = sl[1][sl[2]]
	end
	check_grow(list, cap, ("the KV cache from %d to %d rows"):format(cache.cap, cap))
	replay.drop(cache, "detach")
	local keep = math.min(cache.n_past, cache.cap)
	local buf
	for _, sl in ipairs(slots) do
		local old = sl[1][sl[2]]
		local bytes = keep * (old:nbytes() // old:shape()[1])
		local new = old:backend():tensor(old:type(), { cap, old:shape()[2] })
		new:zero()
		if bytes > 0 then
			if not buf or #buf < bytes then
				buf = lgml.hostbuf(bytes)
			end
			old:to_host(buf, 0, 0, bytes)
			new:from_host(buf, 0, 0, bytes)
		end
		old:close()
		sl[1][sl[2]] = new
	end
	if buf then
		buf:close()
	end
	LOG:info("cache grew", "rows", cap, "from", cache.cap, "n_ctx", cache.n_ctx, "n_past", cache.n_past)
	cache.cap = cap
	cache.gen = (cache.gen or 0) + 1
	return true
end

--- Device bytes the cache holds now.
-- @tparam table cache a cache from `cache`
-- @treturn integer bytes of the K, V, ring and recurrent tensors
function M.cache_bytes(cache)
	local n = 0
	for _, part in ipairs({ cache.k, cache.v, cache.conv or {}, cache.ssm or {}, cache.x or {}, cache.pconv or {}, { cache.mtp_h } }) do
		for _, t in pairs(part) do
			n = n + t:nbytes()
		end
	end
	return n
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

-- Positions [a, b) of a cache as runs of rows: { pos, row, count }. A
-- pool sequence maps positions to the cells of its blocks.
local function runs(cache, a, b)
	if not cache.pool then
		return a < b and { { a, a, b - a } } or {}
	end
	local B, out, p = cache.pool.block, {}, a
	while p < b do
		local row = cache.blocks[p // B + 1] * B + p % B
		local n = math.min(b - p, B - p % B)
		local last = out[#out]
		if last and last[2] + last[3] == row then
			last[3] = last[3] + n
		else
			out[#out + 1] = { p, row, n }
		end
		p = p + n
	end
	return out
end

-- The parts of a snapshot in buffer order: { t, bytes, row }. A part
-- with row (bytes per row) holds positions [0, rows) of a
-- full-attention layer, in position order. The others copy the first
-- bytes of t: rings up to hi, and whole recurrent states.
local function snap_parts(self, cache, rows, state, hi)
	local parts = {}
	local function add(t, n, row)
		if n > 0 then
			parts[#parts + 1] = { t, n, row }
		end
	end
	for il, L in ipairs(self.layers) do
		if L.recr then
			if state then
				add(cache.conv[il], cache.conv[il]:nbytes())
				add(cache.ssm[il], cache.ssm[il]:nbytes())
				if L.ple then
					add(cache.pconv[il], cache.pconv[il]:nbytes())
				end
			end
		elseif L.swa then
			if state and cache.k[il] then
				for _, t in ipairs({ cache.k[il], cache.v[il] }) do
					add(t, math.min(hi, cache.ring) * (t:nbytes() // t:shape()[1]))
				end
			end
		elseif rows and not L.kv_src then
			local src = cache.pool or cache
			for _, t in ipairs({ src.k[il], src.v[il], src.x and src.x[il], src.x and src.x[-il] }) do
				local row = t:nbytes() // t:shape()[1]
				add(t, rows * row, row)
			end
		end
	end
	local mt = self.mtp
	if mt then
		local src = cache.pool or cache
		if rows then
			for _, t in ipairs({ src.k[mt.il], src.v[mt.il], src.x and src.x[mt.il], src.x and src.x[-mt.il] }) do
				local row = t:nbytes() // t:shape()[1]
				add(t, rows * row, row)
			end
		end
		if state and cache.mtp_h then
			add(cache.mtp_h, cache.mtp_h:nbytes())
		end
	end
	local tok = rows and (cache.pool or cache).x
	tok = tok and tok[0]
	if tok then
		add(tok, rows * (tok:nbytes() // tok:shape()[1]), tok:nbytes() // tok:shape()[1])
	end
	return parts
end

-- Copy part p between the device and buf at off: to the host with out
-- true, else back. A row part copies its first n positions.
local function copy_part(cache, p, buf, off, n, out)
	local t, row = p[1], p[3]
	if not row then
		if out then
			t:to_host(buf, off, 0, p[2])
		else
			t:from_host(buf, off, 0, p[2])
		end
		return
	end
	for _, r in ipairs(runs(cache, 0, n)) do
		if out then
			t:to_host(buf, off + r[1] * row, r[2] * row, r[3] * row)
		else
			t:from_host(buf, off + r[1] * row, r[2] * row, r[3] * row)
		end
	end
end

--- Bytes that `snapshot` copies now.
-- @tparam table cache a cache from `cache` or `seq`
-- @tparam[opt] boolean full true for a full snapshot
-- @treturn integer bytes
function M:snapshot_bytes(cache, full)
	local bytes = 0
	for _, p in ipairs(snap_parts(self, cache, full and cache.n_past, true, cache.hi)) do
		bytes = bytes + p[2]
	end
	return bytes
end

--- Copy the cache state at cache.n_past to host memory.
-- The state is the recurrent conv and delta net states and the sliding
-- rings with hi: the parts that cannot go back to an earlier position.
-- A full snapshot also holds rows [0, n_past) of the full-attention
-- layers, in position order, so it restores into another cache or pool
-- sequence of the same model.
-- @tparam table cache a cache from `cache` or `seq`
-- @tparam[opt] boolean full true for a full snapshot
-- @treturn table the snapshot: n (the position), hi, full, bytes and
-- buf (an `lgml.hostbuf`; close it to free the memory at once)
function M:snapshot(cache, full)
	local n = cache.n_past
	local parts = snap_parts(self, cache, full and n, true, cache.hi)
	local bytes = 0
	for _, p in ipairs(parts) do
		bytes = bytes + p[2]
	end
	local buf = lgml.hostbuf(bytes)
	local off = 0
	for _, p in ipairs(parts) do
		copy_part(cache, p, buf, off, n, true)
		off = off + p[2]
	end
	return { n = n, hi = cache.hi, full = full or false, bytes = bytes, buf = buf }
end

--- Write a snapshot back into a cache.
-- Without opts it sets n_past, rs_pos and hi to the snapshot's. The
-- full-attention rows before n of a snapshot without them must still
-- hold the same tokens. The cached replay graph stays valid. A pool
-- sequence takes the blocks that the rows need.
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
	if snap.full then
		-- Shared blocks must not change: a pool sequence starts over on
		-- blocks of its own.
		if cache.pool then
			self:truncate(cache, 0)
		end
		self:grow(cache, rows)
	end
	-- Walk the parts in snapshot order; skip the ones not restored.
	local off = 0
	for _, p in ipairs(snap_parts(self, cache, snap.full and snap.n, true, snap.hi)) do
		if p[3] or state then
			copy_part(cache, p, snap.buf, off, rows, false)
		end
		off = off + p[2]
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
-- sliding layers, and the first row of the global span. With a cache
-- the spans are padded to PAD rows. A pool sequence reads the pool
-- window that holds its cells.
local function spans(cache, n_kv)
	if not cache then
		return n_kv, n_kv, 0
	end
	if cache.pool then
		local lo, w = seq_window(cache, n_kv)
		local ring = cache.ring
		if not ring then
			return w, w, lo
		end
		return w, n_kv > ring and ring or math.min(ring, (n_kv + PAD - 1) // PAD * PAD), lo
	end
	local cap = cache.cap or cache.n_ctx
	local n_attn = math.min(cap, (n_kv + PAD - 1) // PAD * PAD)
	local ring = cache.ring or cap
	return n_attn, n_kv > ring and ring or math.min(ring, n_attn), 0
end

-- Raise when a pass of n tokens does not fit the cache. A pass past
-- the rows the cache holds now grows it.
local function check_pass(self, cache, n)
	local hp = self.hp
	local n_kv = cache.n_past + n
	if n_kv > cache.n_ctx then
		error("context full: " .. n_kv .. " > " .. cache.n_ctx)
	end
	self:grow(cache, n_kv)
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
			for _, t in pairs(cache.pconv or {}) do
				t:zero()
			end
			if cache.mtp_h then
				cache.mtp_h:zero()
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
	if cache and cache.pool then
		inp.mask:fill_blocks({ n_past = n_past, block = cache.pool.block, blocks = cache.blocks,
			lo = inp.lo, causal = inp.causal })
	elseif inp.causal then
		inp.mask:fill_causal(n_past)
	else
		inp.mask:fill_mask({ n_past = n_past, causal = false })
	end
	if inp.mask_swa then
		inp.mask_swa:fill_mask({ n_past = n_past, window = hp.n_swa, ring = cache ~= nil,
			causal = inp.causal })
	end
	if cache then
		if cache.pool then
			inp.rows:set(seq_cells(cache, n_past, n))
		else
			inp.rows:fill_range(n_past)
		end
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
-- @tparam[opt] table cache a cache from `cache` or `seq`
-- @tparam[opt] table opts options:
-- opts.flash_attn: true for flash attention (f16 masks).
-- opts.logits: "all" or "last" (default "all").
-- opts.hidden: a layer count; `forward` stops after that many layers
-- and gives the hidden state in place of the logits (default nil).
-- opts.pool: "last" makes `forward` give the normed hidden state of the
-- last token in place of the logits, as embedding models use (default
-- nil).
-- opts.embd: an `lgml.hostbuf` with n rows of n_embd f32 values, such
-- as image embeddings; they take the place of the token embeddings,
-- with no scale. ids still give the per-layer inputs of gemma models;
-- use 0 for them there, as llama.cpp does.
-- opts.embd_off: the byte offset of the first row in opts.embd
-- (default 0).
-- opts.causal: false lets the tokens of this pass see each other both
-- ways, as llama.cpp does for gemma image tokens (default true).
-- @treturn table fields n, n_past, n_kv, n_attn, n_swa_attn, lo, cache,
-- flash_attn, logits, hidden, pool, causal, and the tensors ids, pos, mask,
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
		check_pass(self, cache, n)
	end
	kv.check(cache, opts.flash_attn)
	local n_kv = (cache and cache.n_past or 0) + n
	local n_attn, n_swa_attn, lo = spans(cache, n_kv)
	local mt = opts.flash_attn and "f16" or "f32"
	local inp = {
		n = n, n_attn = n_attn, n_swa_attn = n_swa_attn, lo = lo, cache = cache,
		flash_attn = opts.flash_attn, logits = opts.logits or "all",
		hidden = opts.hidden, pool = opts.pool, causal = opts.causal ~= false,
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
	if hp.hc then
		q4.leaves(self, inp, n)
		q4.seq_leaves(self, inp, n_attn)
	end
	fill(hp, inp, cache, nil, opts)
	if hp.hc then
		q4.pass(self, inp, cache, ids)
	end
	return inp
end

--- The replay key of a pass: the pass shape, options and adapters.
-- Passes with the same key share one graph; see `lgml.replay`.
-- @tparam table cache the cache
-- @tparam integer n the token count
-- @tparam table opts the input options
-- @treturn string the key
function M:replay_key(cache, n, opts)
	-- Grow first: a grow drops the cached graph, and the key holds the
	-- span of the grown cache.
	if cache.n_past + n <= cache.n_ctx then
		self:grow(cache, cache.n_past + n)
	end
	local n_attn, n_swa_attn, lo = spans(cache, cache.n_past + n)
	local key = ("%d:%d:%d:%s:%s:%s:%s:%s:%s:%s"):format(n, n_attn, n_swa_attn, tostring(opts.flash_attn),
	    opts.logits or "all", tostring(opts.hidden), tostring(opts.pool), tostring(opts.embd ~= nil),
	    tostring(opts.causal ~= false), self.lora_key or "")
	if cache.pool then
		key = key .. (":%d@%d.%d"):format(lo, cache.pool.id, cache.pool.gen)
	end
	return key
end

--- Load new ids into the inputs of an earlier pass with the same key.
-- @tparam table inp inputs from `inputs`
-- @tparam {integer,...} ids the token ids
-- @tparam table cache the cache
-- @tparam[opt] table opts the input options; opts.embd and
-- opts.embd_off give the new embeddings
-- @raise as `inputs`
function M:refill(inp, ids, cache, opts)
	check_pass(self, cache, #ids)
	fill(self.hp, inp, cache, ids, opts)
	if self.hp.hc then
		q4.pass(self, inp, cache, ids)
	end
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
	local cur = L.attn_norm and g:mul(g:rms_norm(x, eps), L.attn_norm) or x
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
	if inp.batch then
		return M.attention_batch(hp, g, inp, il, L, q, k, v, shared)
	end
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
		-- A pool sequence writes its global rows into the pool.
		local src = (c.pool and not L.swa) and c.pool or c
		local kw = g:set_rows(src.k[il], g:reshape(k, { n, w }), rows)
		local vw = g:set_rows(src.v[il], g:reshape(v, { n, w }), rows)
		k, v = kv.views(g, c, kw, vw, L.swa and inp.n_swa_attn or inp.n_attn, L.Hkv, D,
		    not L.swa and inp.lo or 0)
	end
	shared[il] = { k = k, v = v }
	local mask = L.swa and inp.mask_swa or inp.mask
	return M.attend(g, q, k, v, mask, hp.attn_scale or 1 / math.sqrt(D), inp.flash_attn, rv)
end
M.attention = attention

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

-- The weighted outputs (n, K, n_embd) of the experts of W that sel picks.
local function experts(g, hp, W, x3, sel, w)
	local gate, up
	if W.gate_up_exps then
		local gu = g:mul_mat_id(W.gate_up_exps, x3, sel)
		local n, K, F = gu:shape()[1], gu:shape()[2], gu:shape()[3] // 2
		local st = gu:strides()
		gate = g:view(gu, { n, K, F }, { st[1], st[2] })
		up = g:view(gu, { n, K, F }, { st[1], st[2] }, lgml.row_size("f32", F))
	else
		gate = g:mul_mat_id(W.gate_exps, x3, sel)
		up = g:mul_mat_id(W.up_exps, x3, sel)
	end
	local act = hp.gemma and g:geglu_split(gate, up) or g:swiglu_split(gate, up)
	return g:mul(g:mul_mat_id(W.down_exps, act, sel), w)
end

-- The sum over the K picks of ex (n, K, E). The views go into the graph
-- first, so the adds are adjacent nodes: ggml-vulkan fuses them into
-- one multi-add dispatch.
local function sum_picks(g, ex)
	local n, K, E = table.unpack(ex:shape())
	local st = ex:strides()
	local views = {}
	for j = 0, K - 1 do
		views[j] = g:view(ex, { n, E }, { st[1] }, j * st[2])
		g:expand(views[j])
	end
	local out = views[0]
	for j = 1, K - 1 do
		out = g:add(out, views[j])
	end
	if K == 1 then
		out = g:cont(out)
	end
	return out
end

--- Passes of fewer tokens take the expert cache path. It is the op
-- offload batch of ggml-vulkan (GGML_OP_OFFLOAD_MIN_BATCH, default 32).
M.XCACHE_MAX_BATCH = tonumber(os.getenv("GGML_OP_OFFLOAD_MIN_BATCH") or 32)

--- True to count the picks and hits of a zero-copy expert cache, for
-- `expert_cache:stats` and profiles. It slows decode: set it with
-- LGML_XCACHE_COUNT=1 for profiling runs only.
M.XCACHE_COUNT = os.getenv("LGML_XCACHE_COUNT") == "1"

-- The expert outputs of a layer with an expert cache (see
-- `lgml.xcache`). The device computes the picks in slots, the CPU the
-- rest; a mask of the router weights keeps each pick on one side.
local function cached_experts(g, hp, L, x3, sel, w)
	local xc = L.xc
	local n, K = sel:shape()[1], sel:shape()[2]
	local flat = g:reshape(g:cont(sel), { n * K })
	local slot = g:reshape(g:get_rows(xc.map, flat), { n, K })
	if xc.full then
		return sum_picks(g, experts(g, hp, xc, x3, g:cast(slot, "i32"), w))
	end
	-- A pick that no slot holds goes to the zero slot of its position.
	local hit = g:reshape(g:get_rows(xc.hit, flat), { n, K })
	local dsel = g:cast(g:add(g:mul(g:sub(slot, xc.zero), hit), xc.zero), "i32")
	local wd = g:mul(w, g:reshape(hit, { n, K, 1 }))
	if xc.host then
		-- Zero-copy: the device reads the misses from pinned host memory.
		-- Held picks read a dummy row at weight 0: one shared row on the
		-- matvec path (8 tokens or fewer), else one row per position, as
		-- the batched path computes too few rows for a repeated id.
		if M.XCACHE_COUNT then
			-- The remap op counts the picks and hits on the CPU. A CPU
			-- leaf sends the graph through the scheduler, which runs
			-- the op there; the sync per layer slows decode.
			xc.count_leaf = xc.count_leaf or x3:backend():cpu():tensor("f32", { 1 }, { 0 })
			g:expand(g:scale(xc.count_leaf, 1))
			g:expand(xc.state:remap(g, xc.il, g:reshape(flat, { n, K })))
		end
		local hrow = g:reshape(g:get_rows(xc.hmap, flat), { n, K })
		local hsel = n <= 8 and g:cast(hrow, "i32") or
		    g:cast(g:sub(hrow, g:mul(g:sub(hrow, xc.hzero), hit)), "i32")
		local dev = sum_picks(g, experts(g, hp, xc, x3, dsel, wd))
		return g:add(dev, sum_picks(g, experts(g, hp, xc.host, x3, hsel, g:sub(w, wd))))
	end
	if not hp.gemma and not xc.router_cpu then
		-- The CPU computes only the picks that no slot holds.
		local up = not L.gate_up_exps and L.up_exps or nil
		local dev = sum_picks(g, experts(g, hp, xc, x3, dsel, wd))
		local cpu = xc.state:misses(g, xc.il, g:cont(x3), g:reshape(flat, { n, K }), w,
		    L.gate_up_exps or L.gate_exps, up, L.down_exps)
		return g:add(dev, cpu)
	end
	local wc = g:sub(w, wd)
	local csel = xc.state:remap(g, xc.il, g:reshape(flat, { n, K }))
	-- With the router on the CPU, the routing nodes go first, so that the
	-- scheduler keeps them there: only x, the device ids and weights go
	-- to the device, and one sum comes back.
	if xc.router_cpu then
		g:expand(w, wd, wc, csel, dsel)
	end
	local dev = sum_picks(g, experts(g, hp, xc, x3, dsel, wd))
	return g:add(dev, sum_picks(g, experts(g, hp, L, x3, csel, wc)))
end

--- Build the routed experts of one layer.
-- The router picks the top n_used experts; their weights sum to 1
-- when hp.norm_topk is set. Gemma routes on the rms-normed residual.
-- This is a plain function.
-- @static
-- @param g the graph
-- @tparam table hp the config
-- @tparam table L the layer; L.n_expert overrides hp.n_expert
-- @param x the residual (n, n_embd)
-- @param h the normed expert input (n, n_embd)
-- @tparam[opt] table taps gets "ffn_moe_topk" and "ffn_moe_weights"
-- @tparam[opt] string sfx the tap name suffix
-- @return the weighted sum of the chosen expert outputs (n, n_embd)
function M.moe(g, hp, L, x, h, taps, sfx)
	local n, E, NE, K = x:shape()[1], hp.n_embd, L.n_expert or hp.n_expert, hp.n_used
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
	if taps then
		taps["ffn_moe_topk" .. sfx] = sel
		taps["ffn_moe_weights" .. sfx] = w
	end
	-- From XCACHE_MAX_BATCH tokens up, the scheduler moves the CPU experts
	-- to the device with the used experts copied: the cache adds nothing,
	-- and its CPU ids there gave NaN on Vulkan. A zero-copy cache has no
	-- CPU part, so every pass takes it.
	if L.xc and L.xc.n > 0 and (n < M.XCACHE_MAX_BATCH or L.xc.host) then
		return cached_experts(g, hp, L, x3, sel, w)
	end
	return sum_picks(g, experts(g, hp, L, x3, sel, w))
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
	z = hp.gdn_gate == "sigmoid" and g:sigmoid(z) or g:silu(z)
	o = g:mul(g:mul(g:rms_norm(o, hp.eps), L.ssm_norm), z)
	return mm(g, L, "ssm_out", g:reshape(o, { n, vd }))
end
M.linear_attention = linear_attention

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
-- @return logits (n, n_vocab), or (1, n_vocab) when inp.logits is "last",
-- or one row per entry of the i32 leaf inp.logit_rows when it is "rows";
-- with inp.hidden, the hidden state (n, n_embd) after that many layers;
-- with inp.pool "last", the normed hidden state (1, n_embd) of the last
-- token; with inp.pool "rows", the normed hidden states (m, n_embd) of
-- the rows that inp.pool_rows lists
-- @treturn table named nodes for comparison, such as "embd",
-- "ffn_out-0", "l_out-0" and "result_norm"
function M:forward(g, inp)
	if self.hp.hc then
		return q4.forward(self, g, inp)
	end
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
			local la = inp.batch and M.linear_attention_batch or linear_attention
			x = g:add(la(hp, g, inp, L, il, cur, taps, sfx), x)
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
	if inp.logits == "rows" then
		x = g:get_rows(x, inp.logit_rows)
	elseif (inp.logits == "last" or inp.pool == "last") and n > 1 then
		local row = lgml.row_size("f32", hp.n_embd)
		x = g:view(x, { 1, hp.n_embd }, nil, (n - 1) * row)
	end
	if inp.pool == "rows" then
		x = g:get_rows(x, inp.pool_rows)
	end
	if inp.pool then
		taps.result_norm = g:mul(g:rms_norm(x, hp.eps), self.output_norm)
		return taps.result_norm, taps
	end
	local logits
	logits, taps.result_norm = M.head(g, hp, self.output_norm, self.output, x, self.lora_out)
	return logits, taps
end

-- Rows off+1 .. off+n of an (N, ...) node, as (n, ...).
local function rows_of(g, t, off, n)
	local sh = t:shape()
	local st = t:strides()
	local shape = { n }
	for i = 2, #sh do
		shape[i] = sh[i]
	end
	local strides = {}
	for i = 1, #sh - 1 do
		strides[i] = st[i]
	end
	return g:view(t, shape, strides, off * st[1])
end

--- Attention of layer il for a batch of pool sequences.
-- Rows off+1 .. off+n of q, k and v belong to one sequence. Their keys
-- and values go into the cells of that sequence: global layers in the
-- pool, sliding layers in the sequence's ring. The queries of a
-- sequence read the pool window of its cells, under a mask that shows
-- only its own cells.
-- @return the output (N, H * D)
function M.attention_batch(hp, g, inp, il, L, q, k, v, shared)
	local D, out = L.D, nil
	shared[il] = shared[il] or {}
	for i, b in ipairs(inp.batch) do
		local c, n = b.cache, b.n
		local rk, rv = kv.rot_k(c, D), kv.rot_v(c, D)
		local qi = kv.rotate(g, g:cont(rows_of(g, q, b.off, n)), rk)
		local ki, vi
		if L.kv_src then
			ki, vi = shared[L.kv_src][i].k, shared[L.kv_src][i].v
		else
			ki = kv.rotate(g, g:cont(rows_of(g, k, b.off, n)), rk)
			vi = kv.rotate(g, g:cont(rows_of(g, v, b.off, n)), rv)
			local w = L.Hkv * D
			local rows = L.swa and b.rows_swa or b.rows
			local src = L.swa and c or c.pool
			local kw = g:set_rows(src.k[il], g:reshape(ki, { n, w }), rows)
			local vw = g:set_rows(src.v[il], g:reshape(vi, { n, w }), rows)
			ki, vi = kv.views(g, c, kw, vw, L.swa and b.n_swa_attn or b.n_attn, L.Hkv, D,
			    not L.swa and b.lo or 0)
		end
		shared[il][i] = { k = ki, v = vi }
		local mask = L.swa and b.mask_swa or b.mask
		local o = M.attend(g, qi, ki, vi, mask, hp.attn_scale or 1 / math.sqrt(D), inp.flash_attn, rv)
		out = out and g:concat_qwen3tts(out, o, 1) or o
	end
	return out
end

--- Recurrent layer il for a batch of sequences: each sequence runs its
-- rows through its own conv and delta net states.
-- @return the output (N, n_embd)
function M.linear_attention_batch(hp, g, inp, L, il, x, taps, sfx)
	local out
	for _, b in ipairs(inp.batch) do
		local xi = g:cont(rows_of(g, x, b.off, b.n))
		local o = linear_attention(hp, g, { n = b.n, cache = b.cache }, L, il, xi, taps, sfx)
		out = out and g:concat_qwen3tts(out, o, 1) or o
	end
	return out
end

-- Fill the input leaves of a batch pass: sequence i gets ids[i] (a
-- list) at its cache.n_past.
local function fill_batch(hp, inp, ids)
	if ids then
		local all = {}
		for _, list in ipairs(ids) do
			table.move(list, 1, #list, #all + 1, all)
		end
		inp.ids:set(all)
	end
	local pos = {}
	for _, b in ipairs(inp.batch) do
		local c, n = b.cache, b.n
		local n_past = c.n_past
		for j = 1, n do
			pos[#pos + 1] = n_past + j - 1
		end
		if c.rs_pos and n_past == 0 then
			for l in pairs(c.conv) do
				c.conv[l]:zero()
				c.ssm[l]:zero()
			end
			for _, t in pairs(c.pconv or {}) do
				t:zero()
			end
			if c.mtp_h then
				c.mtp_h:zero()
			end
		end
		if b.mask then
			b.mask:fill_blocks({ n_past = n_past, block = c.pool.block, blocks = c.blocks, lo = b.lo })
		end
		if b.mask_swa then
			b.mask_swa:fill_mask({ n_past = n_past, window = hp.n_swa, ring = true, causal = true })
		end
		if b.rows then
			b.rows:set(seq_cells(c, n_past, n))
		end
		if b.rows_swa then
			local r = {}
			for j = 1, n do
				r[j] = (n_past + j - 1) % c.ring
			end
			b.rows_swa:set(r)
			c.hi = math.max(c.hi, n_past + n)
		end
		if c.rs_pos then
			c.rs_pos = n_past + n
		end
	end
	inp.pos:set(pos)
end

--- Tell if the model can run several pool sequences in one pass. Every
-- chat model can: attention reads the pool under a mask per sequence,
-- and recurrent layers run per sequence on their states.
-- @treturn boolean
function M:batchable()
	return true
end

--- Make the inputs of a pass over several pool sequences.
-- Sequence i runs ids[i], a list of token ids, at its cache.n_past: one
-- id for a decode step, or a chunk of a prompt.
-- @tparam {table,...} caches the sequences, from `seq`
-- @tparam {table,...} ids a list of token ids per cache
-- @tparam[opt] table opts options: flash_attn
-- @treturn table inputs for `forward`, with `batch` (one entry per
-- sequence: cache, off, n, rows, masks, spans and lo) and pools (the
-- set of pools it reads)
-- @raise when a cache is full or is not a pool sequence
function M:batch_inputs(caches, ids, opts)
	opts = opts or {}
	local be, hp = self.be, self.hp
	local mt = opts.flash_attn and "f16" or "f32"
	local N, attn = 0, false
	for _, L in ipairs(self.layers) do
		attn = attn or not L.recr
	end
	local inp = { flash_attn = opts.flash_attn, logits = "rows", causal = true, batch = {}, pools = {} }
	for i, c in ipairs(caches) do
		assert(c.pool, "batch_inputs: a batch runs pool sequences")
		local n = #ids[i]
		check_pass(self, c, n)
		kv.check(c, opts.flash_attn)
		local n_attn, n_swa_attn, lo = spans(c, c.n_past + n)
		local b = { cache = c, off = N, n = n, n_attn = n_attn, n_swa_attn = n_swa_attn, lo = lo }
		inp.pools[c.pool] = true
		if attn then
			b.rows, b.mask = be:tensor("i64", { n }), be:tensor(mt, { n, n_attn })
			if hp.n_swa then
				b.mask_swa = be:tensor(mt, { n, n_swa_attn })
			end
			if c.ring then
				b.rows_swa = be:tensor("i64", { n })
			end
			if hp.hc then
				q4.seq_leaves(self, b, n_attn)
			end
		end
		inp.batch[i] = b
		N = N + n
	end
	inp.n = N
	inp.ids = be:tensor("i32", { N })
	inp.pos = be:tensor("i32", { N })
	if hp.hc then
		q4.leaves(self, inp, N)
	end
	-- Only the last token of each sequence needs logits.
	local last = {}
	for i, b in ipairs(inp.batch) do
		last[i] = b.off + b.n - 1
	end
	inp.logit_rows = be:tensor("i32", { #last }, last)
	fill_batch(hp, inp, ids)
	if hp.hc then
		q4.batch(self, inp, ids)
	end
	return inp
end

--- The replay key of a batch pass: the token counts and spans of the
-- sequences, and the options. Sequences with rings or recurrent states
-- add their ids, since the graph reads their own tensors.
-- @tparam {table,...} caches the sequences
-- @tparam {table,...} ids a list of token ids per cache
-- @tparam table opts the input options
-- @treturn string the key
function M:batch_key(caches, ids, opts)
	local parts = { tostring(opts.flash_attn), self.lora_key or "" }
	for i, c in ipairs(caches) do
		local n = #ids[i]
		if c.n_past + n <= c.n_ctx then
			self:grow(c, c.n_past + n)
		end
		local a, w, lo = spans(c, c.n_past + n)
		local own = (c.ring or c.rs_pos) and c.id or 0
		parts[#parts + 1] = ("%d.%d.%d/%d/%d/%d/%d"):format(own, c.pool.id, c.pool.gen, n, a, w, lo)
	end
	return table.concat(parts, ":")
end

--- Run several pool sequences in one pass.
-- Sequence i runs ids[i] (a list of token ids) at its cache.n_past and
-- advances by #ids[i]. The pass reuses the last batch graph when the key
-- matches. Recurrent states and rings follow each sequence, as a pass
-- of one sequence does; a ring takes at most cache.ubatch tokens per
-- pass.
-- @tparam {table,...} caches the sequences, from `seq`
-- @tparam {table,...} ids a list of token ids per cache
-- @tparam[opt] table opts options: flash_attn
-- @treturn {userdata,...} one (1, n_vocab) logits node per sequence,
-- for its last token; valid until the next batch pass
function M:run_batch(caches, ids, opts)
	opts = opts or {}
	local key = self:batch_key(caches, ids, opts)
	local r = self.batch_replay
	if r and r.key ~= key then
		-- Requests can still read rows of the last pass; the collector
		-- frees its buffers when none does.
		drop_batch(self)
		collectgarbage()
		r = nil
	end
	if r then
		for _, b in ipairs(r.inp.batch) do
			check_pass(self, b.cache, b.n)
		end
		fill_batch(self.hp, r.inp, ids)
		if self.hp.hc then
			q4.batch(self, r.inp, ids)
		end
	else
		local inp = self:batch_inputs(caches, ids, opts)
		local g = self.be:graph()
		local out = self:forward(g, inp)
		r = { key = key, g = g, inp = inp, rows = {}, pools = inp.pools }
		local V = out:shape()[2]
		local rb = lgml.row_size("f32", V)
		for i in ipairs(inp.batch) do
			r.rows[i] = g:cont(g:view(out, { 1, V }, nil, (i - 1) * rb))
		end
		self.batch_replay = r
	end
	if self.before_pass then
		self:before_pass()
	end
	r.g:compute(table.unpack(r.rows))
	for i, c in ipairs(caches) do
		c.n_past = c.n_past + #ids[i]
	end
	return r.rows
end

--- Decode one token for each of several pool sequences in one pass.
-- @tparam {table,...} caches the sequences, from `seq`
-- @tparam {integer,...} ids one token id per cache
-- @tparam[opt] table opts options: flash_attn
-- @treturn {userdata,...} one (1, n_vocab) logits node per sequence
function M:decode_batch(caches, ids, opts)
	local lists = {}
	for i, id in ipairs(ids) do
		lists[i] = { id }
	end
	return self:run_batch(caches, lists, opts)
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

--- Run ids from position 0 and give the pooled hidden state.
-- The cache starts empty. Passes take at most ubatch tokens and go
-- through `lgml.replay`; each pass computes one pooled row, so full
-- chunks share one graph. The last pass gives the result.
-- @tparam table cache a cache from `cache`
-- @tparam {integer,...} ids the token ids, at least one
-- @tparam[opt] table opts options for `inputs`, and opts.ubatch: the
-- most tokens per pass (default 512; a ring takes at most cache.ubatch)
-- @treturn {number,...} the normed hidden state of the last token,
-- n_embd values
-- @raise when the ids do not fit the cache
function M:embed(cache, ids, opts)
	local o = { pool = "last" }
	for k, v in pairs(opts or {}) do
		if k ~= "ubatch" then
			o[k] = v
		end
	end
	local ub = math.min(opts and opts.ubatch or UBATCH, cache.ubatch or math.huge)
	self:rewind(cache, 0)
	cache.n_past = 0
	local out
	for s = 1, #ids, ub do
		local part = table.move(ids, s, math.min(#ids, s + ub - 1), 1, {})
		out = replay.run(self, cache, part, o)
	end
	return out:get()
end

-- A causal mask (n, n) of type mt in which each token list of lists
-- sees only its own tokens.
local function packed_mask(lists, n, mt)
	local neg = mt == "f16" and "\0\252" or string.pack("<f", -math.huge)
	local zero = mt == "f16" and "\0\0" or string.pack("<f", 0)
	local rows, a = {}, 0
	for _, l in ipairs(lists) do
		for k = 1, #l do
			rows[#rows + 1] = neg:rep(a) .. zero:rep(k) .. neg:rep(n - a - k)
		end
		a = a + #l
	end
	return table.concat(rows)
end

--- Tell if `embed_packed` can run the model: it has no recurrent and
-- no sliding-window layers.
-- @treturn boolean
function M:packable()
	return not self.recurrent and not self.hp.n_swa
end

--- Run several token lists in one pass and give the pooled hidden
-- state of each.
-- The lists share the pass but not their attention: each token sees
-- only the earlier tokens of its own list, at positions from 0. The
-- pass uses no cache and no replay, so the lists together must fit
-- one pass.
-- @tparam {{integer,...},...} lists the token lists, each non-empty
-- @tparam[opt] table opts options: flash_attn
-- @treturn {{number,...},...} the normed hidden state of the last
-- token of each list, n_embd values each
-- @raise when the model is not `packable`
function M:embed_packed(lists, opts)
	assert(self:packable(), "embed_packed: the model has recurrent or sliding-window layers")
	opts = opts or {}
	local be, E = self.be, self.hp.n_embd
	local ids, pos, last = {}, {}, {}
	for i, l in ipairs(lists) do
		for j, id in ipairs(l) do
			ids[#ids + 1], pos[#pos + 1] = id, j - 1
		end
		last[i] = #ids - 1
	end
	local n, mt = #ids, opts.flash_attn and "f16" or "f32"
	local inp = {
		n = n, n_past = 0, n_kv = n, n_attn = n, n_swa_attn = n, flash_attn = opts.flash_attn,
		logits = "all", pool = "rows", causal = true,
		ids = be:tensor("i32", { n }, ids), pos = be:tensor("i32", { n }, pos),
		mask = be:tensor(mt, { n, n }), pool_rows = be:tensor("i32", { #lists }, last),
	}
	inp.mask:set(packed_mask(lists, n, mt))
	local g = be:graph()
	local out = self:forward(g, inp)
	g:compute(out)
	local flat = out:get()
	g:close()
	for _, t in pairs(inp) do
		if type(t) == "userdata" then
			t:close()
		end
	end
	local vs = {}
	for i = 1, #lists do
		vs[i] = table.move(flat, (i - 1) * E + 1, i * E, 1, {})
	end
	return vs
end

return M
