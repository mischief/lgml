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

-- Write a tiny model of arch ("llama" or "qwen3") to path. opts.quant
-- ("q8_0" or "q4_0") stores the 2-D weights of the layers quantized.
-- opts.tied leaves out output.weight. opts.n_ctx sets the context length
-- (default 256). opts.override gives f32 values
-- by tensor name in place of the random ones. opts.kv adds metadata
-- entries { name, type, value }. Returns the tensors, { name =
-- { shape, values } }, for tests that compute by hand.
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
	local function norm(name, d)
		local v = {}
		for i = 1, d do
			v[i] = 0.9 + 0.2 * r()
		end
		add(name, { d }, v)
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
		w(p .. "attn_q.weight", { H * D, E })
		w(p .. "attn_k.weight", { HKV * D, E })
		w(p .. "attn_v.weight", { HKV * D, E })
		w(p .. "attn_output.weight", { E, H * D })
		if arch ~= "llama" then
			norm(p .. "attn_q_norm.weight", D)
			norm(p .. "attn_k_norm.weight", D)
		end
		norm(p .. "ffn_norm.weight", E)
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

-- A temporary file name that os.remove cleans up.
function M.tmp(suffix)
	local p = os.tmpname()
	os.remove(p)
	return p .. (suffix or "")
end

return M
