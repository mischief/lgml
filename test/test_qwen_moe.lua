-- llama, qwen3, qwen3moe, qwen3next, qwen35moe and qwen35 on tiny ggufs
-- made here, against llama.cpp logits, and replay against a new graph
-- per pass. Output is TAP. LGML_LLAMA_LOGITS names a llama_logits.c
-- binary to compare every logit live; LGML_REF_DUMP=1 prints new rows.
-- LGML_QWEN3MOE, LGML_QWEN3NEXT, LGML_QWEN35MOE, LGML_QWEN35 name real
-- ggufs to run.
local lgml = require("lgml")
-- GPU backends accumulate in f16, so whole-pass comparisons get more room.
local LOOSE = (os.getenv("LGML_DEVICE") or "CPU") == "CPU" and 1e-2 or 3e-2

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local gguf = require("gguf")
local rc = require("replay_check")

local n = 0
local failed = false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local function skip(name)
	n = n + 1
	print("ok " .. n .. " # SKIP " .. name)
end

-- Seeded uniform numbers in [0, 1), the same on every host.
local function rng(seed)
	local x = seed
	return function()
		x = x * 6364136223846793005 + 1442695040888963407
		return (x >> 11) / 2.0 ^ 53
	end
end

-- Byte-level vocab: the 256 GPT-2 byte pieces, one merge, two specials.
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
	for _, t in ipairs({ { "he", 1 }, { "<|im_start|>", 3 }, { "<|im_end|>", 3 } }) do
		tokens[#tokens + 1] = t[1]
		types[#types + 1] = t[2]
	end
	return {
		{ "tokenizer.ggml.model", gguf.STRING, "gpt2" },
		{ "tokenizer.ggml.pre", gguf.STRING, "qwen2" },
		{ "tokenizer.ggml.tokens", { gguf.ARRAY, gguf.STRING }, tokens },
		{ "tokenizer.ggml.token_type", { gguf.ARRAY, gguf.INT32 }, types },
		{ "tokenizer.ggml.merges", { gguf.ARRAY, gguf.STRING }, { "h e" } },
		{ "tokenizer.ggml.eos_token_id", gguf.UINT32, #tokens - 1 },
	}, #tokens
end

local E, H, HKV, D, X, USED, F = 64, 4, 2, 16, 8, 2, 32

-- kv and tensors for a tiny model of arch. Shapes are in torch order.
local function build(arch)
	local r = rng(#arch * 7919)
	local kv, nv = vocab_kv()
	local ts = {}
	local function key(k, t, v)
		kv[#kv + 1] = { k:gsub("^%.", arch .. "."), t, v }
	end
	local function rand(shape, lo, hi)
		local c = 1
		for _, d in ipairs(shape) do
			c = c * d
		end
		local v = {}
		for i = 1, c do
			v[i] = lo + (hi - lo) * r()
		end
		return v
	end
	-- Weight scaled for the fan-in (last dimension).
	local function w(name, shape, gain)
		local s = (gain or 1) * math.sqrt(3 / shape[#shape])
		ts[#ts + 1] = { name, shape, rand(shape, -s, s) }
	end
	local function norm(name, d)
		ts[#ts + 1] = { name, { d }, rand({ d }, 0.9, 1.1) }
	end
	local function vec(name, d, lo, hi)
		ts[#ts + 1] = { name, { d }, rand({ d }, lo, hi) }
	end

	local hybrid = arch == "qwen3next" or arch == "qwen35moe" or arch == "qwen35"
	local dense = arch == "qwen3" or arch == "qwen35" or arch == "llama"
	local q35 = arch == "qwen35moe" or arch == "qwen35"
	local nl = hybrid and 4 or 2
	kv[#kv + 1] = { "general.architecture", gguf.STRING, arch }
	kv[#kv + 1] = { "general.name", gguf.STRING, "tiny " .. arch }
	key(".block_count", gguf.UINT32, nl)
	key(".context_length", gguf.UINT32, 256)
	key(".embedding_length", gguf.UINT32, E)
	key(".feed_forward_length", gguf.UINT32, 2 * E)
	key(".attention.head_count", gguf.UINT32, H)
	key(".attention.head_count_kv", gguf.UINT32, HKV)
	key(".attention.key_length", gguf.UINT32, D)
	key(".attention.value_length", gguf.UINT32, D)
	key(".attention.layer_norm_rms_epsilon", gguf.FLOAT32, 1e-6)
	key(".rope.freq_base", gguf.FLOAT32, 1e6)
	if not dense then
		key(".expert_count", gguf.UINT32, X)
		key(".expert_used_count", gguf.UINT32, USED)
		key(".expert_feed_forward_length", gguf.UINT32, F)
	end

	-- Delta net sizes: 2 key heads, 4 value heads, head size 16.
	local NK, NV, S, K = 2, 4, 16, 4
	local FS = 48
	if hybrid then
		key(".rope.dimension_count", gguf.UINT32, 8)
		if not dense then
			key(".expert_shared_feed_forward_length", gguf.UINT32, FS)
		end
		key(".ssm.conv_kernel", gguf.UINT32, K)
		key(".ssm.state_size", gguf.UINT32, S)
		key(".ssm.group_count", gguf.UINT32, NK)
		key(".ssm.time_step_rank", gguf.UINT32, NV)
		key(".ssm.inner_size", gguf.UINT32, NV * S)
		key(".full_attention_interval", gguf.UINT32, 2)
		if q35 then
			key(".rope.dimension_sections", { gguf.ARRAY, gguf.INT32 }, { 2, 1, 1, 0 })
		end
	end

	w("token_embd.weight", { nv, E }, 3)
	norm("output_norm.weight", E)
	w("output.weight", { nv, E })
	if arch == "llama" then
		-- llama3 frequency factors.
		key(".rope.dimension_count", gguf.UINT32, D)
		vec("rope_freqs.weight", D // 2, 1, 8)
	end
	for i = 0, nl - 1 do
		local p = "blk." .. i .. "."
		norm(p .. "attn_norm.weight", E)
		if hybrid and i % 2 == 0 then
			local kd, vd = NK * S, NV * S
			if arch == "qwen3next" then
				w(p .. "ssm_in.weight", { 2 * kd + 2 * vd, E })
				w(p .. "ssm_ba.weight", { 2 * NV, E })
			else
				w(p .. "attn_qkv.weight", { 2 * kd + vd, E })
				w(p .. "attn_gate.weight", { vd, E })
				w(p .. "ssm_beta.weight", { NV, E })
				w(p .. "ssm_alpha.weight", { NV, E })
			end
			w(p .. "ssm_conv1d.weight", { 2 * kd + vd, K })
			vec(p .. "ssm_dt.bias", NV, -1, 1)
			vec(p .. "ssm_a", NV, -1.5, -0.2)
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
		if dense then
			w(p .. "ffn_gate.weight", { 2 * E, E })
			w(p .. "ffn_up.weight", { 2 * E, E })
			w(p .. "ffn_down.weight", { E, 2 * E })
		else
			w(p .. "ffn_gate_inp.weight", { X, E }, 3)
			if arch == "qwen35moe" then
				w(p .. "ffn_gate_up_exps.weight", { X, 2 * F, E })
			else
				w(p .. "ffn_gate_exps.weight", { X, F, E })
				w(p .. "ffn_up_exps.weight", { X, F, E })
			end
			w(p .. "ffn_down_exps.weight", { X, E, F })
		end
		if hybrid and not dense then
			w(p .. "ffn_gate_inp_shexp.weight", { E })
			w(p .. "ffn_gate_shexp.weight", { FS, E })
			w(p .. "ffn_up_shexp.weight", { FS, E })
			w(p .. "ffn_down_shexp.weight", { E, FS })
		end
	end
	return kv, ts
end

-- Two prompt batches, then greedy steps on llama.cpp's choices.
local batches = { { 104, 101, 108, 108, 111 }, { 256, 32, 119 } }
local N_DECODE = 6

-- Reference rows: input id, argmax, logits at IDX. From llama.cpp on
-- the CPU with f16 KV cache.
local IDX = { 1, 33, 65, 101, 151, 201, 256, 259 }
local REF = {}
REF.llama = {
	{ id = 104, best = 27, v = { -1.05115, -0.0685001, 0.381098, 1.38403, -0.139574, 0.617577, 1.08773, 0.283959 } },
	{ id = 101, best = 139, v = { -0.746684, -1.88775, -0.430884, -0.934096, 0.853949, -0.803914, 2.07841, 0.628068 } },
	{ id = 108, best = 110, v = { -0.48671, -1.42679, 2.06304, 1.42211, 0.0042516, 0.0617156, -1.11522, 0.219221 } },
	{ id = 108, best = 64, v = { -0.329015, -0.895649, 2.46447, 1.52424, 0.543325, 0.432585, -1.44716, 0.0523283 } },
	{ id = 111, best = 31, v = { -0.139485, -0.687294, 1.84334, 0.948356, 0.0718723, 0.537472, -0.343468, -0.00484884 } },
	{ id = 256, best = 179, v = { -0.0920151, 0.0817786, 1.30437, -0.0913485, -0.141847, -0.0627955, -0.668146, 0.674637 } },
	{ id = 32, best = 143, v = { -0.883974, 0.0939144, 2.12966, 0.0245771, -0.140124, 1.99303, -1.06771, -0.438986 } },
	{ id = 119, best = 94, v = { -2.20769, 0.79694, 1.69301, -1.03928, 1.14025, 0.0840331, 0.953195, 0.895152 } },
	{ id = 94, best = 201, v = { -0.553102, 0.339777, -0.367432, -0.517332, 0.464785, -0.0201371, 0.260383, 1.45374 } },
	{ id = 201, best = 64, v = { 0.222636, 1.9625, 2.55442, -0.207547, 1.23181, 0.45706, 1.03088, -1.018 } },
	{ id = 64, best = 29, v = { 0.232239, 0.549152, 1.50869, -1.01467, 1.01234, -0.394239, -0.0856171, 0.0592265 } },
	{ id = 29, best = 247, v = { -0.329641, 0.901714, 1.38295, 0.212162, 1.48048, 1.97504, -0.168128, -0.676739 } },
	{ id = 247, best = 64, v = { 0.346251, -0.0819248, 3.30467, -1.16438, 0.657267, 0.509867, 0.186653, -0.581156 } },
	{ id = 64, best = 96, v = { -0.171444, 0.580061, 1.42343, -0.927338, 0.730668, -0.3473, 0.281601, 0.84549 } },
}
REF.qwen3 = {
	{ id = 104, best = 201, v = { -1.28208, -1.07746, 0.296883, 0.409393, 1.51223, 2.25051, -0.436495, 0.212937 } },
	{ id = 101, best = 17, v = { -0.751454, -1.40469, 0.610081, 0.331726, -0.207181, 2.47268, -0.566524, 0.801569 } },
	{ id = 108, best = 200, v = { -0.894077, -0.439106, 1.17438, 0.0512241, 0.794092, 3.56739, -0.66045, 1.09793 } },
	{ id = 108, best = 200, v = { -1.054, -0.270842, 1.22301, 0.198575, 0.701514, 3.31456, -0.653716, 1.1818 } },
	{ id = 111, best = 21, v = { -1.11188, -0.0926493, -0.379794, 0.460695, 0.516054, 2.04142, -0.720588, 1.6887 } },
	{ id = 256, best = 224, v = { -2.17088, -0.659625, 1.11436, -0.967621, 1.19859, 1.58668, -0.745333, 0.317392 } },
	{ id = 32, best = 72, v = { -1.1445, -0.713143, -0.354835, -0.840118, -1.61469, 1.78081, -0.98997, 0.776097 } },
	{ id = 119, best = 246, v = { -2.38766, -0.606108, 1.97219, -0.0744514, 0.627293, 1.83918, -1.01291, 0.168866 } },
	{ id = 246, best = 226, v = { -1.31587, -1.89684, 1.1825, 2.22483, 0.630171, 2.1202, -0.444982, 0.920047 } },
	{ id = 226, best = 75, v = { -0.405636, -0.0379642, 0.746998, 1.58409, 1.57735, 1.53925, 0.0322377, 0.0594707 } },
	{ id = 75, best = 129, v = { -1.12805, -0.373538, 1.21592, 0.74982, 1.52807, 1.15422, 0.60599, 0.235278 } },
	{ id = 129, best = 140, v = { -1.70418, -1.74701, 1.53398, 0.7721, -0.461653, 1.59674, -0.986596, 1.19246 } },
	{ id = 140, best = 129, v = { -0.938116, -1.7604, 1.11478, 0.550438, 2.27, 2.60255, 0.0443451, -1.48176 } },
	{ id = 129, best = 224, v = { -1.65095, -2.01039, 0.732394, 0.53116, -0.533183, 2.06186, -1.63194, 1.0771 } },
}
REF.qwen3moe = {
	{ id = 104, best = 162, v = { 0.422612, 0.675456, -1.43958, -0.876634, -0.199726, 0.230231, -1.22371, 0.902244 } },
	{ id = 101, best = 162, v = { 0.749849, -0.795744, -0.849307, -0.417871, 0.131926, 0.208432, 1.01161, 0.0317917 } },
	{ id = 108, best = 208, v = { 1.38582, 0.72651, 0.113369, -0.454437, -1.55815, 0.302924, 0.678891, -1.48816 } },
	{ id = 108, best = 208, v = { 1.39788, 0.648028, 0.252159, -0.58532, -1.65346, -0.208891, 0.899008, -1.80362 } },
	{ id = 111, best = 86, v = { 0.643173, 0.839277, -0.712702, 1.66899, -0.504484, -0.123913, 2.05132, -0.950302 } },
	{ id = 256, best = 216, v = { -0.144441, -0.710417, 0.172642, 0.200605, -1.05638, -0.220543, 1.14916, -1.06902 } },
	{ id = 32, best = 31, v = { -1.46681, 0.263598, -0.305398, -0.759618, 0.527873, 0.664876, 0.742315, 0.0713511 } },
	{ id = 119, best = 171, v = { 0.790346, 1.02256, -0.0112842, 1.89189, -0.208932, -0.18186, 1.2016, 0.0037207 } },
	{ id = 171, best = 31, v = { -0.678178, -0.0935246, -0.464812, 0.25034, -0.540423, 1.31923, 0.200541, -0.535848 } },
	{ id = 31, best = 92, v = { -0.328159, -0.636344, -1.3434, 1.10646, -0.734217, -0.390656, 0.855375, -1.10627 } },
	{ id = 92, best = 255, v = { 0.955459, 0.177183, 0.0216756, 0.91036, -0.866549, -0.215155, 2.64175, -1.5438 } },
	{ id = 255, best = 213, v = { 2.10047, 0.567903, -0.144471, 0.474486, -0.178422, 1.32404, 0.560431, -0.176087 } },
	{ id = 213, best = 83, v = { 0.139672, -0.0980108, -0.40917, -0.538707, -0.210372, 0.920649, 1.16178, 0.131741 } },
	{ id = 83, best = 162, v = { -1.13694, -0.929626, 0.340415, 1.77103, 0.1129, 0.546885, 1.61102, 0.992177 } },
}
REF.qwen3next = {
	{ id = 104, best = 74, v = { 0.900627, 0.155941, 0.600175, 0.0636306, 0.123724, 0.237035, 1.07896, -0.649485 } },
	{ id = 101, best = 195, v = { 0.251232, 1.25618, -0.682534, -1.02653, -0.297548, -0.67786, 0.585092, -0.275161 } },
	{ id = 108, best = 238, v = { -0.691614, 0.820521, 0.363753, -0.383941, 0.625701, 0.141862, 0.0144778, -0.632857 } },
	{ id = 108, best = 206, v = { -0.727992, 0.302546, 0.632632, -0.539311, 1.53437, -0.115795, 0.0934349, 0.302281 } },
	{ id = 111, best = 14, v = { -1.24587, 0.882382, 0.168385, -0.375093, -0.132613, -0.94747, -0.475912, 1.04245 } },
	{ id = 256, best = 235, v = { -0.608139, -0.524057, -0.0844897, -0.000210896, 0.514741, 0.00207654, 0.320347, 0.712436 } },
	{ id = 32, best = 219, v = { 0.208593, -0.468982, -0.57658, -0.685077, -1.04127, 0.834757, -0.174975, 0.771034 } },
	{ id = 119, best = 239, v = { -0.340904, 1.36759, -0.517791, 2.1593, -0.539037, -0.567017, -0.975742, -0.844581 } },
	{ id = 239, best = 170, v = { -0.888299, -0.73465, 0.177091, 0.382456, -0.598027, 0.298383, 1.22475, -0.775758 } },
	{ id = 170, best = 223, v = { -0.792613, -0.488571, 0.199659, -0.256183, -0.395046, 0.699435, 0.679537, 0.412106 } },
	{ id = 223, best = 236, v = { -0.818922, 0.206162, -0.216754, -0.405276, -0.67275, 0.509297, -2.32779, -0.403158 } },
	{ id = 236, best = 172, v = { 1.04212, -2.10643, 0.737711, 0.882096, 0.89825, 0.748135, 0.054401, 1.66404 } },
	{ id = 172, best = 101, v = { 0.267992, 0.162176, -0.0055826, -2.04546, -0.7993, -1.20063, 0.285796, -0.85923 } },
	{ id = 101, best = 74, v = { 1.1139, 1.43934, -1.28673, -1.29438, -0.8429, -1.20468, 0.240952, -0.155441 } },
}
REF.qwen35moe = {
	{ id = 104, best = 132, v = { -0.0594387, 0.540082, -0.268126, -0.550753, 0.713221, -0.132219, 0.247489, -0.42194 } },
	{ id = 101, best = 6, v = { 0.908493, -0.0355893, 0.975039, -0.698991, -2.48283, -0.554227, 0.273623, -1.48216 } },
	{ id = 108, best = 109, v = { 1.37471, -0.474631, 0.703978, -0.557054, -0.962872, 0.0952681, -0.443808, -0.534326 } },
	{ id = 108, best = 59, v = { 0.826599, 0.247777, 0.496898, -1.92239, -1.03182, -0.718233, 0.0855267, -0.853547 } },
	{ id = 111, best = 78, v = { 0.387135, 0.161247, 0.808935, -0.719035, 1.01377, 0.064768, 0.716713, 0.579997 } },
	{ id = 256, best = 148, v = { 0.179954, -0.136735, 1.76514, 0.330454, -0.305743, -0.6184, -0.385547, -1.35429 } },
	{ id = 32, best = 93, v = { 1.88197, 0.366352, 0.107659, 0.285531, 0.703169, 1.05096, 0.740242, -0.0638491 } },
	{ id = 119, best = 39, v = { -2.58262, 0.390587, -0.152811, -2.21282, 0.917855, 0.381178, 0.987708, 0.614725 } },
	{ id = 39, best = 99, v = { 0.914595, -1.33728, 0.779541, -0.650884, -1.40355, 0.779296, -0.816349, 0.685588 } },
	{ id = 99, best = 45, v = { 1.29669, 1.99703, -0.883421, -1.05293, 0.58618, 0.359916, 1.09266, -1.7076 } },
	{ id = 45, best = 30, v = { 0.914281, 0.668082, -0.377891, -1.61714, 1.96703, -0.0971248, 1.0856, 0.663464 } },
	{ id = 30, best = 40, v = { -0.723525, 1.21584, -1.2772, -0.621114, -0.682331, 0.0987821, -1.51515, -1.86596 } },
	{ id = 40, best = 35, v = { 0.772328, 0.449265, 0.675351, -1.43461, 0.0159395, -0.933473, 0.339232, -0.0713653 } },
	{ id = 35, best = 182, v = { -2.06998, 0.100364, -1.12857, -2.42489, -0.228877, -1.12356, -0.267117, 0.316472 } },
}
REF.qwen35 = {
	{ id = 104, best = 91, v = { 0.166369, -0.921372, -0.294668, -1.30905, 1.90535, 0.204631, 1.85323, 1.19078 } },
	{ id = 101, best = 46, v = { -1.73047, -0.894844, 1.05722, 1.20938, -0.736321, -0.148385, -0.794118, -1.32213 } },
	{ id = 108, best = 97, v = { -1.12896, 0.897962, -0.576969, 1.1955, 2.17654, -1.76422, -1.05239, 0.548551 } },
	{ id = 108, best = 70, v = { -1.71752, 1.26575, 0.548496, 0.485094, 1.64008, 0.163604, 1.0453, 0.862631 } },
	{ id = 111, best = 95, v = { -0.0986442, -0.546889, -0.376962, -0.242349, -0.0884756, -0.471766, -0.0783929, 0.0190737 } },
	{ id = 256, best = 175, v = { 0.05864, 0.136585, -0.535786, -0.283116, -1.0227, 0.139287, -0.520613, 0.361859 } },
	{ id = 32, best = 169, v = { -1.31325, -0.422233, -0.341607, 0.251139, -2.58415, 0.117314, 1.3213, -0.00168404 } },
	{ id = 119, best = 106, v = { -0.288017, -0.433156, 1.32729, -0.440948, -0.0353969, -0.0746102, -3.2486, -0.359862 } },
	{ id = 106, best = 104, v = { -0.720023, -0.658187, -1.90512, -1.15654, 0.435009, -0.812422, -0.904968, 0.663198 } },
	{ id = 104, best = 142, v = { -1.85213, 1.35935, 0.90301, 2.11566, -1.51501, -0.135683, 1.65785, -1.45613 } },
	{ id = 142, best = 75, v = { 0.295593, -0.961254, 0.120598, -0.17996, -0.163078, -1.13308, 0.610732, 0.570884 } },
	{ id = 75, best = 84, v = { -0.781914, 0.701409, -0.221661, -0.142553, -1.00675, -1.26597, -0.295839, -0.672643 } },
	{ id = 84, best = 157, v = { -1.47991, -0.322878, -0.822329, 0.498202, -0.260715, -0.395344, 1.11967, 0.329531 } },
	{ id = 157, best = 73, v = { 0.215747, 1.56929, -1.19288, -1.1045, 1.31623, -0.355461, 2.341, 0.557646 } },
}

-- Rows from the llama_logits binary: { id, logits }.
local function llama_rows(bin, path)
	local args = { bin, path, N_DECODE }
	for i, b in ipairs(batches) do
		if i > 1 then
			args[#args + 1] = "/"
		end
		for _, id in ipairs(b) do
			args[#args + 1] = id
		end
	end
	local p = assert(io.popen("nice -n 19 " .. table.concat(args, " ")))
	local rows = {}
	for line in p:lines() do
		local v = {}
		for x in line:gmatch("%S+") do
			v[#v + 1] = tonumber(x)
		end
		rows[#rows + 1] = { id = math.tointeger(table.remove(v, 1)), logits = v }
	end
	p:close()
	return rows
end

local function argmax(v)
	local b = 1
	for i = 2, #v do
		if v[i] > v[b] then
			b = i
		end
	end
	return b - 1
end

-- Run ids at cache.n_past. Returns one logits list per token.
local function eval(m, cache, ids, opts)
	local inp = m:inputs(ids, cache, opts)
	local g <close> = m.be:graph()
	local logits = m:forward(g, inp)
	g:compute(logits)
	if cache then
		cache.n_past = inp.n_kv
	end
	local flat, nv = logits:get(), logits:shape()[2]
	local rows = {}
	for i = 1, #ids do
		rows[i] = table.move(flat, (i - 1) * nv + 1, i * nv, 1, {})
	end
	return rows
end

-- Our rows for the batches and the decode ids of ref.
local function our_rows(m, ref, opts)
	local c = m:cache(64)
	local rows = {}
	local function add(ids)
		for _, r in ipairs(eval(m, c, ids, opts)) do
			rows[#rows + 1] = r
		end
	end
	local nb = 0
	for _, b in ipairs(batches) do
		add(b)
		nb = nb + #b
	end
	for i = nb + 1, #ref do
		add({ ref[i].id })
	end
	return rows, c
end

local TOL = 2e-3

-- Largest difference to the reference and whether every argmax agrees.
local function compare(rows, ref)
	local err, same = 0, #rows == #ref
	for i, r in ipairs(ref) do
		local got = rows[i] or {}
		if r.logits then
			for j = 1, #r.logits do
				err = math.max(err, math.abs((got[j] or math.huge) - r.logits[j]))
			end
			same = same and argmax(got) == argmax(r.logits)
		else
			for j, ix in ipairs(IDX) do
				err = math.max(err, math.abs((got[ix] or math.huge) - r.v[j]))
			end
			same = same and argmax(got) == r.best
		end
	end
	return err, same
end

local function best(r)
	return r.logits and argmax(r.logits) or r.best
end

-- Checks beyond the logits: greedy steps, flash attention, a pass
-- without a cache, and the recurrent state position rules.
local function more(arch, m, ref)
	local gen = require("lgml.generate")
	local prompt, nb = {}, 0
	for _, b in ipairs(batches) do
		table.move(b, 1, #b, #prompt + 1, prompt)
	end
	nb = #prompt
	local want = {}
	for i = nb + 1, #ref do
		want[#want + 1] = ref[i].id
	end
	want[#want + 1] = best(ref[#ref])
	local c = m:cache(64)
	local ids = gen.greedy(m, c, prompt, #want)
	ok(table.concat(ids, ",") == table.concat(want, ","),
	    arch .. ": greedy steps match llama.cpp")
	ok(c.n_past == nb + #want - 1, arch .. ": cache advanced")

	local err, same = compare(our_rows(m, ref, { flash_attn = true }), ref)
	ok(err < LOOSE and same, ("%s: flash_attn rows agree (max error %.3g)"):format(arch, err))

	local seq = table.move(prompt, 1, nb, 1, {})
	for i = nb + 1, #ref do
		seq[#seq + 1] = ref[i].id
	end
	err, same = compare(eval(m, nil, seq), ref)
	ok(err < LOOSE and same, ("%s: pass without a cache agrees (max error %.3g)"):format(arch, err))

	if m.recurrent then
		c = m:cache(64)
		eval(m, c, { 1, 2, 3 })
		c.n_past = 2
		ok(not pcall(eval, m, c, { 4 }), arch .. ": recurrent state cannot rewind")
		c.n_past = 0
		local rows = {}
		for _, b in ipairs(batches) do
			for _, r in ipairs(eval(m, c, b)) do
				rows[#rows + 1] = r
			end
		end
		local e2 = compare(rows, table.move(ref, 1, nb, 1, {}))
		ok(e2 < TOL, arch .. ": n_past 0 clears the recurrent state")
	end
end

local be <close> = assert(lgml.backend(os.getenv("LGML_DEVICE") or "CPU"))
local bin = os.getenv("LGML_LLAMA_LOGITS")

-- A prompt that ends near the first padding block of 256 rows.
local long = {}
for i = 1, 250 do
	long[i] = (i * 37) % 250 + 1
end

for _, arch in ipairs({ "llama", "qwen3", "qwen3moe", "qwen3next", "qwen35moe", "qwen35" }) do
	local path = os.tmpname()
	gguf.write(path, build(arch))
	local model = require("lgml." .. arch)
	local m = model.load(be, path)
	local ref
	if bin and bin ~= "" then
		ref = llama_rows(bin, path)
	else
		ref = REF[arch]
	end
	if not ref then
		skip(arch .. ": no reference")
	else
		local rows = our_rows(m, ref)
		ok(#rows == #ref, arch .. ": row count " .. #rows)
		local err, same = compare(rows, ref)
		io.stderr:write(("%s: %d rows, max logit error %.3g\n"):format(arch, #ref, err))
		ok(err < TOL, ("%s: logits match llama.cpp (max error %.3g)"):format(arch, err))
		ok(same, arch .. ": argmax matches llama.cpp on every row")
		if os.getenv("LGML_REF_DUMP") and ref[1].logits then
			print("REF." .. arch .. " = {")
			for _, r in ipairs(ref) do
				local v = {}
				for j, ix in ipairs(IDX) do
					v[j] = ("%.6g"):format(r.logits[ix])
				end
				print(("\t{ id = %d, best = %d, v = { %s } },"):format(r.id,
				    argmax(r.logits), table.concat(v, ", ")))
			end
			print("}")
		end
		more(arch, m, ref)
	end
	-- Decode steps cross n_past 256: three graphs for prefill and the
	-- two attention spans.
	for _, fa in ipairs({ false, true }) do
		rc.decode(ok, arch .. (fa and " flash_attn: " or ": "), m,
		    { n_ctx = 320, prompt = long, n = 12, flash_attn = fa, builds = 3 })
	end
	if m.recurrent then
		rc.restart(ok, arch .. ": ", m)
	end
	m:close()
	os.remove(path)
end

-- Real models on LGML_DEVICE (default CPU). Run this file with lua5.4
-- directly for Vulkan: the meson test hides the Vulkan driver.
local real = {
	{ "qwen3moe", "LGML_QWEN3MOE" },
	{ "qwen3next", "LGML_QWEN3NEXT" },
	{ "qwen35moe", "LGML_QWEN35MOE" },
	{ "qwen35", "LGML_QWEN35" },
}
for _, r in ipairs(real) do
	local arch, env = r[1], r[2]
	local path = os.getenv(env)
	if path == nil or path == "" then
		skip(env .. " not set")
	else
		local gen = require("lgml.generate")
		local dev = os.getenv("LGML_DEVICE") or "CPU"
		local rb <close> = assert(lgml.backend(dev))
		local tag = arch .. " " .. dev .. ": "
		local info = assert(lgml.gguf(path))
		ok(info.meta["general.architecture"] == arch, tag .. "architecture")
		local m = require("lgml." .. arch).load(rb, path)
		local tk <close> = assert(lgml.tokenizer(path))
		local prompt = tk:encode("The capital of France is", { add_bos = false })
		local t0 = lgml.clock()
		local ids, t = gen.greedy(m, m:cache(256), prompt, 32)
		local text = tk:decode(ids)
		io.stderr:write(("%s%q, %.1f tok/s decode\n"):format(tag, text,
		    t.n_decode / math.max(t.decode, 1e-9)))
		ok(text:find("Paris") ~= nil, tag .. "greedy text names Paris")
		local fa = gen.greedy(m, m:cache(256), prompt, 8, { flash_attn = true })
		ok(table.concat(fa, ",", 1, 4) == table.concat(ids, ",", 1, 4),
		    tag .. "flash_attn agrees on the first tokens")
		-- Prompt and output in two parts give the same next token.
		local c = m:cache(256)
		local all = table.move(ids, 1, 8, #prompt + 1, table.move(prompt, 1, #prompt, 1, {}))
		m:step(c, table.move(all, 1, #prompt + 4, 1, {}))
		local id = m:step(c, table.move(all, #prompt + 5, #all, 1, {}))
		ok(id == ids[9], tag .. "chunked prefill agrees")
		io.stderr:write(("%stotal %.1f s\n"):format(tag, lgml.clock() - t0))
		m:close()
	end
end

print("1.." .. n)
os.exit(failed and 1 or 0)
