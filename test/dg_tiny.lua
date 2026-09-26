-- A tiny random-weight diffusion-gemma gguf for tests: two layers (one
-- sliding, one global), n_embd 64, 4 experts, a 288-token vocab, all
-- f32. The weights come from seeded C fills, so each run writes the
-- same bytes.
local lgml = require("lgml")
local gguf = require("gguf")

local M = {}

M.hp = {
	n_embd = 64, n_head = 4, n_vocab = 288, n_ff = 48, n_expert = 4,
	n_expert_used = 2, n_ff_exp = 16, canvas = 8, n_swa = 4,
	head_dim = 32, head_dim_swa = 16,
}

-- Vocab: 5 control tokens, 256 byte tokens, then "▁" and "a" to "z".
local function vocab()
	local tokens = { "<pad>", "<eos>", "<bos>", "<unk>", "<mask>" }
	local types = { 3, 3, 3, 3, 3 }
	for b = 0, 255 do
		tokens[#tokens + 1] = string.format("<0x%02X>", b)
		types[#types + 1] = 6
	end
	tokens[#tokens + 1] = "\u{2581}"
	types[#types + 1] = 1
	for c = string.byte("a"), string.byte("z") do
		tokens[#tokens + 1] = string.char(c)
		types[#types + 1] = 1
	end
	local scores = {}
	for i = 1, #tokens do
		scores[i] = i <= 5 and -1000 or -i
	end
	return tokens, types, scores
end

-- Write the model to path. n_layer defaults to 2; with more layers,
-- every sixth layer is global.
function M.write(path, n_layer)
	local hp = M.hp
	n_layer = n_layer or 2
	local E, H, V = hp.n_embd, hp.n_head, hp.n_vocab
	local be <close> = assert(lgml.backend("CPU"))
	local seed = 0
	local tensors = {}
	-- A tensor of normal values with mean m and std s.
	local function add(name, shape, m, s)
		local t <close> = be:tensor("f32", shape)
		seed = seed + 1
		t:fill("normal", m, s, seed)
		tensors[#tensors + 1] = { name, shape, t:bytes() }
	end
	local function const(name, shape, v)
		local t <close> = be:tensor("f32", shape)
		t:fill("const", v)
		tensors[#tensors + 1] = { name, shape, t:bytes() }
	end

	add("token_embd.weight", { V, E }, 0, 0.125)
	add("output_norm.weight", { E }, 5, 0.5)
	add("self_cond_pre_norm.weight", { E }, 1, 0.1)
	add("self_cond_gate.weight", { hp.n_ff, E }, 0, 0.125)
	add("self_cond_up.weight", { hp.n_ff, E }, 0, 0.125)
	add("self_cond_down.weight", { E, hp.n_ff }, 0, 0.02)
	-- Proportional rope: the second half of the pairs does not rotate.
	local freqs = {}
	for i = 1, hp.head_dim // 2 do
		freqs[i] = i <= hp.head_dim // 4 and 1 or 1e30
	end
	tensors[#tensors + 1] = { "rope_freqs.weight", { hp.head_dim // 2 }, freqs }

	local swa, n_head_kv = { true, false }, { 2, 1 }
	if n_layer > 2 then
		for il = 1, n_layer do
			swa[il] = il % 6 ~= 0
			n_head_kv[il] = swa[il] and 2 or 1
		end
	end
	for il = 0, n_layer - 1 do
		local p = "blk." .. il .. "."
		local D = swa[il + 1] and hp.head_dim_swa or hp.head_dim
		local Hkv = n_head_kv[il + 1]
		add(p .. "attn_norm.weight", { E }, 1, 0.1)
		add(p .. "attn_q.weight", { H * D, E }, 0, 0.125)
		add(p .. "attn_k.weight", { Hkv * D, E }, 0, 0.125)
		if swa[il + 1] then
			add(p .. "attn_v.weight", { Hkv * D, E }, 0, 0.125)
		end
		add(p .. "attn_output.weight", { E, H * D }, 0, 0.1)
		add(p .. "attn_q_norm.weight", { D }, 1, 0.1)
		add(p .. "attn_k_norm.weight", { D }, 1, 0.1)
		add(p .. "post_attention_norm.weight", { E }, 1, 0.1)
		add(p .. "ffn_norm.weight", { E }, 1, 0.1)
		add(p .. "ffn_gate.weight", { hp.n_ff, E }, 0, 0.125)
		add(p .. "ffn_up.weight", { hp.n_ff, E }, 0, 0.125)
		add(p .. "ffn_down.weight", { E, hp.n_ff }, 0, 0.15)
		add(p .. "post_ffw_norm_1.weight", { E }, 1, 0.1)
		add(p .. "pre_ffw_norm_2.weight", { E }, 1, 0.1)
		add(p .. "post_ffw_norm_2.weight", { E }, 1, 0.1)
		add(p .. "post_ffw_norm.weight", { E }, 1, 0.1)
		add(p .. "ffn_gate_inp.weight", { hp.n_expert, E }, 0, 1)
		add(p .. "ffn_gate_inp.scale", { E }, 1, 0.2)
		-- Layer 0 has merged gate and up experts, layer 1 separate ones.
		if il == 0 then
			add(p .. "ffn_gate_up_exps.weight", { hp.n_expert, 2 * hp.n_ff_exp, E }, 0, 0.125)
		else
			add(p .. "ffn_gate_exps.weight", { hp.n_expert, hp.n_ff_exp, E }, 0, 0.125)
			add(p .. "ffn_up_exps.weight", { hp.n_expert, hp.n_ff_exp, E }, 0, 0.125)
		end
		add(p .. "ffn_down_exps.weight", { hp.n_expert, E, hp.n_ff_exp }, 0, 0.25)
		add(p .. "ffn_down_exps.scale", { hp.n_expert }, 1, 0.3)
		const(p .. "layer_output_scale.weight", { 1 }, il == 0 and 1.3 or 1.1)
		const(p .. "enc_layer_output_scale.weight", { 1 }, il == 0 and 0.7 or 0.9)
	end

	local A = "diffusion-gemma."
	local tokens, types, scores = vocab()
	local S, U, F, B, I = gguf.STRING, gguf.UINT32, gguf.FLOAT32, gguf.BOOL, gguf.INT32
	local AR = gguf.ARRAY
	gguf.write(path, {
		{ "general.architecture", S, "diffusion-gemma" },
		{ "general.name", S, "dg-tiny" },
		{ A .. "block_count", U, n_layer },
		{ A .. "context_length", U, 512 },
		{ A .. "embedding_length", U, E },
		{ A .. "feed_forward_length", U, hp.n_ff },
		{ A .. "attention.head_count", U, H },
		{ A .. "attention.head_count_kv", { AR, I }, n_head_kv },
		{ A .. "attention.key_length", U, hp.head_dim },
		{ A .. "attention.value_length", U, hp.head_dim },
		{ A .. "attention.key_length_swa", U, hp.head_dim_swa },
		{ A .. "attention.value_length_swa", U, hp.head_dim_swa },
		{ A .. "attention.layer_norm_rms_epsilon", F, 1e-6 },
		{ A .. "attention.sliding_window", U, hp.n_swa },
		{ A .. "attention.sliding_window_pattern", { AR, B }, swa },
		{ A .. "attention.causal", B, false },
		{ A .. "attention.shared_kv_layers", U, 0 },
		{ A .. "rope.dimension_count", U, hp.head_dim },
		{ A .. "rope.dimension_count_swa", U, hp.head_dim_swa },
		{ A .. "rope.freq_base", F, 1e6 },
		{ A .. "rope.freq_base_swa", F, 1e4 },
		{ A .. "expert_count", U, hp.n_expert },
		{ A .. "expert_used_count", U, hp.n_expert_used },
		{ A .. "expert_feed_forward_length", U, hp.n_ff_exp },
		{ A .. "final_logit_softcapping", F, 30 },
		{ A .. "embedding_length_per_layer_input", U, 0 },
		{ "diffusion.canvas_length", U, hp.canvas },
		{ "tokenizer.ggml.model", S, "gemma4" },
		{ "tokenizer.ggml.tokens", { AR, S }, tokens },
		{ "tokenizer.ggml.token_type", { AR, I }, types },
		{ "tokenizer.ggml.scores", { AR, F }, scores },
		{ "tokenizer.ggml.merges", { AR, S }, {} },
		{ "tokenizer.ggml.bos_token_id", U, 2 },
		{ "tokenizer.ggml.eos_token_id", U, 1 },
		{ "tokenizer.ggml.padding_token_id", U, 0 },
		{ "tokenizer.ggml.unknown_token_id", U, 3 },
		{ "tokenizer.ggml.mask_token_id", U, 4 },
		{ "tokenizer.ggml.add_bos_token", B, true },
		{ "tokenizer.ggml.add_space_prefix", B, false },
	}, tensors)
end

return M
