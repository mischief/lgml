-- A tiny random-weight llada-moe-v2 gguf for tests: two layers,
-- n_embd 32, 8 experts (3 used), an ungated shared expert, a byte vocab
-- with the LLaDA special tokens, all f32. `write` also returns the
-- weights as Lua lists (torch order, row-major) for reference math.
local gguf = require("gguf")

local M = {}

M.hp = {
	n_layer = 2, n_embd = 32, n_head = 4, n_head_kv = 2, head_dim = 16, n_expert = 8,
	n_used = 3, n_ff_exp = 8, n_ff_shexp = 16, scale = 2.5, eps = 1e-5, base = 500000,
}

M.SPECIALS = { "<|endoftext|>", "<|mask|>", "<|role_end|>", "<role>", "</role>" }

-- Seeded uniform numbers in [0, 1), the same on every host.
local function rng(seed)
	local x = seed
	return function()
		x = x * 6364136223846793005 + 1442695040888963407
		return (x >> 11) / 2.0 ^ 53
	end
end

-- The 256 GPT-2 byte pieces, one merge, then the specials.
local function vocab()
	local tokens, types, k = {}, {}, 0
	for b = 0, 255 do
		if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or b >= 174 then
			tokens[#tokens + 1] = utf8.char(b)
		else
			tokens[#tokens + 1] = utf8.char(256 + k)
			k = k + 1
		end
		types[#types + 1] = 1
	end
	tokens[#tokens + 1], types[#types + 1] = "he", 1
	for _, s in ipairs(M.SPECIALS) do
		tokens[#tokens + 1], types[#types + 1] = s, 3
	end
	return tokens, types
end

-- The id of special token s, from 0.
function M.id(s)
	for i, t in ipairs(M.SPECIALS) do
		if t == s then
			return 256 + i
		end
	end
end

M.n_vocab = 257 + #M.SPECIALS

--- Write the model to path.
-- opts.out_gain scales the output head (default 1). opts.copy is
-- { dst, src, f }: the output row of dst becomes f times that of src.
-- @treturn table weights by tensor name
function M.write(path, opts)
	opts = opts or {}
	local hp = M.hp
	local E, H, HKV, D, X = hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_expert
	local V = M.n_vocab
	local r = rng(20260)
	local ts, W = {}, {}
	local function add(name, shape, lo, hi)
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
		return v
	end
	local function w(name, shape, gain)
		local s = (gain or 1) * math.sqrt(3 / shape[#shape])
		return add(name, shape, -s, s)
	end
	w("token_embd.weight", { V, E }, 3)
	add("output_norm.weight", { E }, 0.9, 1.1)
	local out = w("output.weight", { V, E }, opts.out_gain or 1)
	if opts.copy then
		local dst, src, f = table.unpack(opts.copy)
		for i = 1, E do
			out[dst * E + i] = f * out[src * E + i]
		end
	end
	for il = 0, hp.n_layer - 1 do
		local p = "blk." .. il .. "."
		add(p .. "attn_norm.weight", { E }, 0.9, 1.1)
		w(p .. "attn_q.weight", { H * D, E })
		w(p .. "attn_k.weight", { HKV * D, E })
		w(p .. "attn_v.weight", { HKV * D, E })
		w(p .. "attn_output.weight", { E, H * D })
		add(p .. "attn_q_norm.weight", { D }, 0.9, 1.1)
		add(p .. "attn_k_norm.weight", { D }, 0.9, 1.1)
		add(p .. "ffn_norm.weight", { E }, 0.9, 1.1)
		w(p .. "ffn_gate_inp.weight", { X, E }, 2)
		w(p .. "ffn_gate_exps.weight", { X, hp.n_ff_exp, E })
		w(p .. "ffn_up_exps.weight", { X, hp.n_ff_exp, E })
		w(p .. "ffn_down_exps.weight", { X, E, hp.n_ff_exp })
		w(p .. "ffn_gate_shexp.weight", { hp.n_ff_shexp, E })
		w(p .. "ffn_up_shexp.weight", { hp.n_ff_shexp, E })
		w(p .. "ffn_down_shexp.weight", { E, hp.n_ff_shexp })
	end
	local tokens, types = vocab()
	local A = "llada-moe-v2."
	local U, F, S, B = gguf.UINT32, gguf.FLOAT32, gguf.STRING, gguf.BOOL
	gguf.write(path, {
		{ "general.architecture", S, "llada-moe-v2" },
		{ "general.name", S, "llada-tiny" },
		{ A .. "block_count", U, hp.n_layer },
		{ A .. "context_length", U, 512 },
		{ A .. "embedding_length", U, E },
		{ A .. "feed_forward_length", U, hp.n_ff_shexp },
		{ A .. "attention.head_count", U, H },
		{ A .. "attention.head_count_kv", U, HKV },
		{ A .. "attention.key_length", U, D },
		{ A .. "attention.value_length", U, D },
		{ A .. "attention.layer_norm_rms_epsilon", F, hp.eps },
		{ A .. "rope.freq_base", F, hp.base },
		{ A .. "expert_count", U, X },
		{ A .. "expert_used_count", U, hp.n_used },
		{ A .. "expert_feed_forward_length", U, hp.n_ff_exp },
		{ A .. "expert_shared_feed_forward_length", U, hp.n_ff_shexp },
		{ A .. "expert_weights_scale", F, hp.scale },
		{ A .. "expert_weights_norm", B, true },
		{ "tokenizer.ggml.model", S, "gpt2" },
		{ "tokenizer.ggml.pre", S, "qwen2" },
		{ "tokenizer.ggml.tokens", { gguf.ARRAY, S }, tokens },
		{ "tokenizer.ggml.token_type", { gguf.ARRAY, gguf.INT32 }, types },
		{ "tokenizer.ggml.merges", { gguf.ARRAY, S }, { "h e" } },
		{ "tokenizer.ggml.eos_token_id", U, M.id("<|endoftext|>") },
		{ "tokenizer.ggml.padding_token_id", U, M.id("<|endoftext|>") },
		{ "tokenizer.ggml.mask_token_id", U, M.id("<|mask|>") },
		{ "tokenizer.chat_template", S, "{{- '<role>SYSTEM</role>' }}{{- '<role>HUMAN</role>' }}" },
	}, ts)
	return W
end

return M
