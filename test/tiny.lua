-- Tiny dense llama and qwen3 ggufs for tests, with a byte-level vocab
-- and random weights. Shapes are in torch order.
local lgml = require("lgml")
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local gguf = require("gguf")

local M = {}

M.E, M.H, M.HKV, M.D, M.F, M.NL = 64, 4, 2, 16, 128, 2

-- ggml type ids of quantized weights.
local QTYPE = { q8_0 = 8, q4_0 = 2 }

-- Seeded uniform numbers in [0, 1), the same on every host.
function M.rng(seed)
	local x = seed
	return function()
		x = x * 6364136223846793005 + 1442695040888963407
		return (x >> 11) / 2.0 ^ 53
	end
end

-- Byte-level vocab: the 256 GPT-2 byte pieces, one merge, and the
-- chat specials.
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
		{ "tokenizer.ggml.eos_token_id", gguf.UINT32, 258 },
	}, #tokens
end

-- Write a tiny model of arch to path: "llama", "qwen3", or "qwen3next"
-- (a delta net layer, then an attention layer). opts.quant ("q8_0" or
-- "q4_0") quantizes the 2-D layer weights. opts.tied leaves out
-- output.weight. opts.n_ctx: context length (default 256). opts.override
-- gives f32 values by tensor name. opts.kv adds { name, type, value }
-- entries. Returns the tensors, { name = { shape, values } }.
function M.write(path, arch, opts)
	opts = opts or {}
	local r = M.rng(#arch * 7919 + (opts.seed or 0))
	local kv, nv = vocab_kv()
	local E, H, HKV, D, F = M.E, M.H, M.HKV, M.D, M.F
	local ts, byname = {}, {}
	local function key(k, t, v)
		kv[#kv + 1] = { arch .. "." .. k, t, v }
	end
	local function add(name, shape, v, quant)
		if opts.override and opts.override[name] then
			v, quant = opts.override[name], false
		end
		byname[name] = { shape, v }
		if quant and opts.quant then
			local raw = {}
			for i, x in ipairs(v) do
				raw[i] = string.pack("<f", x)
			end
			local q = lgml.quantize(opts.quant, table.concat(raw), shape[#shape])
			ts[#ts + 1] = { name, shape, q, QTYPE[opts.quant] }
			-- Keep the values the model sees.
			local be = lgml.backend("CPU")
			local qt = be:tensor(opts.quant, shape, q)
			local ft = be:tensor("f32", shape)
			local g = be:graph()
			g:compute(g:cpy(qt, ft))
			byname[name] = { shape, ft:get() }
			g:close()
			be:close()
		else
			ts[#ts + 1] = { name, shape, v }
		end
	end
	local function w(name, shape, gain)
		local s = (gain or 1) * math.sqrt(3 / shape[#shape])
		local v = {}
		for i = 1, shape[1] * shape[2] do
			v[i] = -s + 2 * s * r()
		end
		add(name, shape, v, name:match("^blk"))
	end
	-- Values in [lo, lo + width); the default keeps the values of
	-- older files exact.
	local function norm(name, d, lo, width)
		lo, width = lo or 0.9, width or 0.2
		local v = {}
		for i = 1, d do
			v[i] = lo + width * r()
		end
		add(name, { d }, v)
	end
	local hybrid = arch == "qwen3next"
	-- Delta net sizes: 2 key heads, 4 value heads, head size 16.
	local NK, NV, S, K = 2, 4, 16, 4
	if hybrid then
		key("rope.dimension_count", gguf.UINT32, 8)
		key("ssm.conv_kernel", gguf.UINT32, K)
		key("ssm.state_size", gguf.UINT32, S)
		key("ssm.group_count", gguf.UINT32, NK)
		key("ssm.time_step_rank", gguf.UINT32, NV)
		key("ssm.inner_size", gguf.UINT32, NV * S)
		key("full_attention_interval", gguf.UINT32, 2)
	end
	kv[#kv + 1] = { "general.architecture", gguf.STRING, arch }
	kv[#kv + 1] = { "general.name", gguf.STRING, "tiny " .. arch }
	key("block_count", gguf.UINT32, M.NL)
	key("context_length", gguf.UINT32, opts.n_ctx or 256)
	key("embedding_length", gguf.UINT32, E)
	key("feed_forward_length", gguf.UINT32, F)
	key("attention.head_count", gguf.UINT32, H)
	key("attention.head_count_kv", gguf.UINT32, HKV)
	key("attention.key_length", gguf.UINT32, D)
	key("attention.value_length", gguf.UINT32, D)
	key("attention.layer_norm_rms_epsilon", gguf.FLOAT32, 1e-6)
	key("rope.freq_base", gguf.FLOAT32, 1e4)
	w("token_embd.weight", { nv, E }, 3)
	norm("output_norm.weight", E)
	if not opts.tied then
		w("output.weight", { nv, E })
	end
	for i = 0, M.NL - 1 do
		local p = "blk." .. i .. "."
		norm(p .. "attn_norm.weight", E)
		if hybrid and i % 2 == 0 then
			local kd, vd = NK * S, NV * S
			w(p .. "ssm_in.weight", { 2 * kd + 2 * vd, E })
			w(p .. "ssm_ba.weight", { 2 * NV, E })
			w(p .. "ssm_conv1d.weight", { 2 * kd + vd, K })
			norm(p .. "ssm_dt.bias", NV, -1, 2)
			norm(p .. "ssm_a", NV, -1.5, 1.3)
			norm(p .. "ssm_norm.weight", S)
			w(p .. "ssm_out.weight", { E, vd })
		else
			w(p .. "attn_q.weight", { H * D * (hybrid and 2 or 1), E })
			w(p .. "attn_k.weight", { HKV * D, E })
			w(p .. "attn_v.weight", { HKV * D, E })
			w(p .. "attn_output.weight", { E, H * D })
			if arch ~= "llama" then
				norm(p .. "attn_q_norm.weight", D)
				norm(p .. "attn_k_norm.weight", D)
			end
		end
		norm(p .. (hybrid and "post_attention_norm.weight" or "ffn_norm.weight"), E)
		w(p .. "ffn_gate.weight", { F, E })
		w(p .. "ffn_up.weight", { F, E })
		w(p .. "ffn_down.weight", { E, F })
	end
	for _, e in ipairs(opts.kv or {}) do
		kv[#kv + 1] = e
	end
	gguf.write(path, kv, ts)
	return byname, nv
end

-- Write a tiny gemma4 model to path: a sliding layer (window 6) and a
-- global layer, with the gemma4 test vocab. opts.n_ctx sets the
-- context length (default 256).
function M.write_gemma4(path, opts)
	opts = opts or {}
	local bpe_vocab = require("bpe_vocab")
	local r = M.rng(7 * 7919 + (opts.seed or 0))
	local kv = bpe_vocab.gemma4_kv()
	local nv
	for _, e in ipairs(kv) do
		if e[1] == "tokenizer.ggml.tokens" then
			nv = #e[3]
		end
	end
	local E, H, D, DS, FF = 64, 4, 32, 16, 96
	local ts = {}
	local function key(k, t, v)
		kv[#kv + 1] = { "gemma4." .. k, t, v }
	end
	local function vals(c, lo, hi)
		local v = {}
		for i = 1, c do
			v[i] = lo + (hi - lo) * r()
		end
		return v
	end
	local function w(name, shape, gain)
		local s = (gain or 1) * math.sqrt(3 / shape[#shape])
		ts[#ts + 1] = { name, shape, vals(shape[1] * shape[2], -s, s) }
	end
	local function norm(name, d)
		ts[#ts + 1] = { name, { d }, vals(d, 0.8, 1.2) }
	end
	local swa, hkv = { true, false }, { 2, 1 }
	kv[#kv + 1] = { "general.architecture", gguf.STRING, "gemma4" }
	kv[#kv + 1] = { "general.name", gguf.STRING, "tiny gemma4" }
	key("block_count", gguf.UINT32, 2)
	key("context_length", gguf.UINT32, opts.n_ctx or 256)
	key("embedding_length", gguf.UINT32, E)
	key("feed_forward_length", gguf.UINT32, FF)
	key("attention.head_count", gguf.UINT32, H)
	key("attention.head_count_kv", { gguf.ARRAY, gguf.INT32 }, hkv)
	key("attention.key_length", gguf.UINT32, D)
	key("attention.value_length", gguf.UINT32, D)
	key("attention.key_length_swa", gguf.UINT32, DS)
	key("attention.value_length_swa", gguf.UINT32, DS)
	key("attention.layer_norm_rms_epsilon", gguf.FLOAT32, 1e-6)
	key("attention.sliding_window", gguf.UINT32, 6)
	key("attention.sliding_window_pattern", { gguf.ARRAY, gguf.BOOL }, swa)
	key("rope.dimension_count", gguf.UINT32, D)
	key("rope.dimension_count_swa", gguf.UINT32, DS)
	key("rope.freq_base", gguf.FLOAT32, 1e6)
	key("rope.freq_base_swa", gguf.FLOAT32, 1e4)
	key("final_logit_softcapping", gguf.FLOAT32, 30)
	w("token_embd.weight", { nv, E }, 0.3)
	norm("output_norm.weight", E)
	local freqs = {}
	for i = 1, D // 2 do
		freqs[i] = i <= D // 4 and 1 or 1e30
	end
	ts[#ts + 1] = { "rope_freqs.weight", { D // 2 }, freqs }
	for i = 0, 1 do
		local p = "blk." .. i .. "."
		local d = swa[i + 1] and DS or D
		norm(p .. "attn_norm.weight", E)
		w(p .. "attn_q.weight", { H * d, E })
		w(p .. "attn_k.weight", { hkv[i + 1] * d, E })
		norm(p .. "attn_k_norm.weight", d)
		if swa[i + 1] then
			w(p .. "attn_v.weight", { hkv[i + 1] * d, E })
		end
		w(p .. "attn_output.weight", { E, H * d })
		norm(p .. "attn_q_norm.weight", d)
		norm(p .. "post_attention_norm.weight", E)
		norm(p .. "ffn_norm.weight", E)
		w(p .. "ffn_gate.weight", { FF, E })
		w(p .. "ffn_up.weight", { FF, E })
		w(p .. "ffn_down.weight", { E, FF })
		norm(p .. "post_ffw_norm.weight", E)
		ts[#ts + 1] = { p .. "layer_output_scale.weight", { 1 }, { 0.8 + 0.3 * i } }
	end
	gguf.write(path, kv, ts)
end

-- A temporary file name that os.remove cleans up.
function M.tmp(suffix)
	local p = os.tmpname()
	os.remove(p)
	return p .. (suffix or "")
end

return M
