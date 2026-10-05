-- A tiny qwen4exp gguf with random weights, for tests: 4 layers
-- (delta net, QSA, delta net with the n-gram input, QSA), 4 streams,
-- routed and shared experts. Shapes are in torch order.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local gguf = require("gguf")
local tiny = require("tiny")

local M = {}

M.E, M.HC, M.R = 64, 4, 16
M.H, M.HKV, M.D, M.ROT = 4, 2, 32, 8
M.NK, M.NV, M.S, M.K = 2, 4, 16, 4
M.X, M.USED, M.F, M.FS = 8, 2, 32, 32
M.IH, M.ID, M.TOP_K, M.RATIO = 2, 16, 8, 4
M.NL, M.PLE_LAYER = 4, 2
M.EOS = 258

-- The real model's hash multipliers: their products wrap 64 bits.
local MULT = { 23703573157769, 20109073645365, 8052911324071 }
local VOCAB = { 53, 59, 61, 67, 71, 73, 79, 83 }

-- Byte-level vocab: the 256 GPT-2 byte pieces, one merge, the chat
-- specials.
local function vocab_kv()
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
	for _, t in ipairs({ { "he", 1 }, { "<|im_start|>", 3 }, { "<|im_end|>", 3 },
	    { "<think>", 1 }, { "</think>", 1 } }) do
		tokens[#tokens + 1] = t[1]
		types[#types + 1] = t[2]
	end
	return {
		{ "tokenizer.ggml.model", gguf.STRING, "gpt2" },
		{ "tokenizer.ggml.pre", gguf.STRING, "qwen2" },
		{ "tokenizer.ggml.tokens", { gguf.ARRAY, gguf.STRING }, tokens },
		{ "tokenizer.ggml.token_type", { gguf.ARRAY, gguf.INT32 }, types },
		{ "tokenizer.ggml.merges", { gguf.ARRAY, gguf.STRING }, { "h e" } },
		{ "tokenizer.ggml.eos_token_id", gguf.UINT32, M.EOS },
	}, #tokens
end

-- The kv and tensors of the model. opts.top_k sets the indexer budget
-- (default M.TOP_K); opts.n_ctx the context length (default 4096).
function M.build(opts)
	opts = opts or {}
	local r = tiny.rng(4242)
	local kv, nv = vocab_kv()
	local ts = {}
	local E, HC, R = M.E, M.HC, M.R
	local CE = HC * E
	local function key(k, t, v)
		kv[#kv + 1] = { "qwen4exp." .. k, t, v }
	end
	local function rand(n, lo, hi)
		local v = {}
		for i = 1, n do
			v[i] = lo + (hi - lo) * r()
		end
		return v
	end
	local function w(name, shape, gain)
		local s = (gain or 1) * math.sqrt(3 / shape[#shape])
		local n = 1
		for _, d in ipairs(shape) do
			n = n * d
		end
		ts[#ts + 1] = { name, shape, rand(n, -s, s) }
	end
	local function norm(name, d)
		ts[#ts + 1] = { name, { d }, rand(d, 0.9, 1.1) }
	end
	local function vec(name, d, lo, hi)
		ts[#ts + 1] = { name, { d }, rand(d, lo, hi) }
	end
	-- Store tensor t as f16, as the real model stores some.
	local function f16(t)
		local be <close> = assert(lgml.backend("CPU"))
		local h <close> = be:tensor("f16", t[2], t[3])
		t[3], t[4] = h:bytes(), gguf.F16
	end
	local function hc(prefix, inject)
		norm(prefix .. "_norm.weight", CE)
		w(prefix .. "_down.weight", { R, CE })
		w(prefix .. "_up.weight", { CE, R })
		if inject then
			w(prefix .. "_inject.weight", { HC, CE })
		end
	end
	kv[#kv + 1] = { "general.architecture", gguf.STRING, "qwen4exp" }
	kv[#kv + 1] = { "general.name", gguf.STRING, "tiny qwen4exp" }
	key("block_count", gguf.UINT32, M.NL)
	key("context_length", gguf.UINT32, opts.n_ctx or 4096)
	key("embedding_length", gguf.UINT32, E)
	key("feed_forward_length", gguf.UINT32, 2 * E)
	key("attention.head_count", gguf.UINT32, M.H)
	key("attention.head_count_kv", gguf.UINT32, M.HKV)
	key("attention.key_length", gguf.UINT32, M.D)
	key("attention.value_length", gguf.UINT32, M.D)
	key("attention.layer_norm_rms_epsilon", gguf.FLOAT32, 1e-6)
	key("rope.freq_base", gguf.FLOAT32, 1e6)
	key("rope.dimension_count", gguf.UINT32, M.ROT)
	key("rope.dimension_sections", { gguf.ARRAY, gguf.INT32 }, { 2, 1, 1, 0 })
	key("expert_count", gguf.UINT32, M.X)
	key("expert_used_count", gguf.UINT32, M.USED)
	key("expert_feed_forward_length", gguf.UINT32, M.F)
	key("expert_shared_feed_forward_length", gguf.UINT32, M.FS)
	key("ssm.conv_kernel", gguf.UINT32, M.K)
	key("ssm.state_size", gguf.UINT32, M.S)
	key("ssm.group_count", gguf.UINT32, M.NK)
	key("ssm.time_step_rank", gguf.UINT32, M.NV)
	key("ssm.inner_size", gguf.UINT32, M.NV * M.S)
	key("full_attention_interval", gguf.UINT32, 2)
	key("hyper_connection.count", gguf.UINT32, HC)
	key("hyper_connection.low_rank", gguf.UINT32, R)
	key("attention.indexer.head_count", gguf.UINT32, M.IH)
	key("attention.indexer.key_length", gguf.UINT32, M.ID)
	key("attention.indexer.top_k", gguf.UINT32, opts.top_k or M.TOP_K)
	local ratios = {}
	for i = 1, M.NL do
		ratios[i] = i % 2 == 0 and M.RATIO or 0
	end
	key("attention.compress_ratios", { gguf.ARRAY, gguf.INT32 }, ratios)
	-- The n-gram embedding: bigram and trigram heads, two of each.
	local per, n = 4, 3
	local heads = (n - 1) * per
	local PD = E // heads
	local off, rows = {}, 0
	for h = 1, heads do
		off[h] = rows
		rows = rows + VOCAB[h]
	end
	key("ple.layers", { gguf.ARRAY, gguf.INT32 }, { M.PLE_LAYER })
	key("ple.ngram_size", gguf.UINT32, n)
	key("ple.heads_per_ngram", gguf.UINT32, per)
	key("ple.conv_kernel", gguf.UINT32, 4)
	key("ple.layer_multipliers", { gguf.ARRAY, gguf.UINT64 }, MULT)
	key("ple.head_offsets", { gguf.ARRAY, gguf.UINT64 }, off)
	key("ple.head_vocab_sizes", { gguf.ARRAY, gguf.UINT64 }, table.move(VOCAB, 1, heads, 1, {}))
	key("ple.eos_token_id", gguf.UINT32, M.EOS)
	key("embedding_length_per_layer_input", gguf.UINT32, PD)

	w("token_embd.weight", { nv, E }, 3)
	w("output.weight", { nv, E })
	hc("output_hc", false)
	-- The n-gram table is f16, as the reader takes no f32 table.
	w("per_layer_token_embd.weight", { rows + 8, PD }, 3)
	f16(ts[#ts])
	local kd, vd = M.NK * M.S, M.NV * M.S
	for i = 0, M.NL - 1 do
		local p = "blk." .. i .. "."
		hc(p .. "hc_attn", true)
		hc(p .. "hc_ffn", true)
		if i % 2 == 0 then
			w(p .. "attn_qkv.weight", { 2 * kd + vd, E })
			w(p .. "attn_gate.weight", { vd, E })
			w(p .. "ssm_beta.weight", { M.NV, E })
			w(p .. "ssm_alpha.weight", { M.NV, E })
			w(p .. "ssm_conv1d.weight", { 2 * kd + vd, M.K })
			vec(p .. "ssm_dt.bias", M.NV, -1, 1)
			vec(p .. "ssm_a", M.NV, -1.5, -0.2)
			norm(p .. "ssm_norm.weight", M.S)
			w(p .. "ssm_out.weight", { E, vd })
		else
			w(p .. "attn_q.weight", { 2 * M.H * M.D, E })
			w(p .. "attn_k.weight", { M.HKV * M.D, E })
			w(p .. "attn_v.weight", { M.HKV * M.D, E })
			w(p .. "attn_output.weight", { E, M.H * M.D })
			norm(p .. "attn_q_norm.weight", M.D)
			norm(p .. "attn_k_norm.weight", M.D)
			w(p .. "indexer.q_proj.weight", { M.IH * M.ID, E }, 2)
			w(p .. "indexer.k_proj.weight", { M.ID, E }, 2)
			norm(p .. "indexer.q_norm.weight", M.ID)
			norm(p .. "indexer.k_norm.weight", M.ID)
		end
		if i == M.PLE_LAYER then
			w(p .. "ple_key.weight", { CE, E })
			w(p .. "ple_value.weight", { E, E })
			norm(p .. "ple_norm_key.weight", CE)
			norm(p .. "ple_norm_query.weight", CE)
			norm(p .. "ple_norm_conv.weight", CE)
			w(p .. "ple_conv1d.weight", { CE, 4 })
			f16(ts[#ts])
		end
		w(p .. "ffn_gate_inp.weight", { M.X, E }, 3)
		w(p .. "ffn_gate_exps.weight", { M.X, M.F, E })
		w(p .. "ffn_up_exps.weight", { M.X, M.F, E })
		w(p .. "ffn_down_exps.weight", { M.X, E, M.F })
		w(p .. "ffn_gate_inp_shexp.weight", { E })
		w(p .. "ffn_gate_shexp.weight", { M.FS, E })
		w(p .. "ffn_up_shexp.weight", { M.FS, E })
		w(p .. "ffn_down_shexp.weight", { E, M.FS })
	end
	return kv, ts
end

-- Write the model to path.
function M.write(path, opts)
	gguf.write(path, M.build(opts))
	return path
end

return M
