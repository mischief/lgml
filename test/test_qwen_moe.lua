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

-- Largest difference of two logit lists.
local function maxdiff(a, b)
	local d = #a == #b and 0 or math.huge
	for i = 1, #a do
		d = math.max(d, math.abs(a[i] - (b[i] or math.huge)))
	end
	return d
end

-- Snapshots on the model: a full snapshot and a state checkpoint go
-- back to their position, and the next pass gives the same logits.
-- Batched passes: three sequences of one KV pool prefill their prompts
-- in one pass, then decode in one pass per step; each sequence's logits
-- match the sequence run alone on its own cache, and so do its greedy
-- tokens.
local function batch_check(arch, m)
	local prompts = { { 11, 12, 13, 14, 15 }, { 40, 41, 42, 43, 44, 45, 46, 47, 48 }, { 7 } }
	local steps = 10
	local want, first = {}, {}
	for i, p in ipairs(prompts) do
		local c = m:cache(128)
		local rows = eval(m, c, p)
		first[i] = rows[#rows]
		local id = argmax(rows[#rows])
		want[i] = { id }
		for _ = 2, steps do
			id = argmax(eval(m, c, { id })[1])
			want[i][#want[i] + 1] = id
		end
	end
	local caches, got = {}, {}
	local pool = m:pool(3 * 128)
	for i = 1, #prompts do
		caches[i] = m:seq(pool, { n_ctx = 128 })
	end
	local rows = m:run_batch(caches, prompts)
	local err, ids = 0, {}
	for i = 1, #prompts do
		local v = rows[i]:get()
		err = math.max(err, maxdiff(v, first[i]))
		ids[i] = argmax(v)
		got[i] = { ids[i] }
	end
	for _ = 2, steps do
		local r = m:decode_batch(caches, ids)
		for i = 1, #prompts do
			ids[i] = argmax(r[i]:get())
			got[i][#got[i] + 1] = ids[i]
		end
	end
	local same = true
	for i = 1, #prompts do
		for j = 1, steps do
			same = same and got[i][j] == want[i][j]
		end
	end
	ok(err == 0, ("%s: batched prefill logits match each sequence alone (max error %.3g)"):format(arch, err))
	ok(same, arch .. ": batched decode gives the same tokens as each sequence alone")
end

local function snapshot_check(arch, m)
	local replay = require("lgml.replay")
	local c = m:cache(64)
	local o = { logits = "last" }
	local function run(ids)
		return replay.run(m, c, ids, o):get()
	end
	run({ 5, 6, 7, 8, 9, 10 })
	local full = m:snapshot(c, true)
	local st = m:snapshot(c)
	local want = run({ 11, 12 })
	run({ 13, 14, 15 })
	m:restore(c, full)
	ok(c.n_past == 6 and (not c.rs_pos or c.rs_pos == 6), arch .. ": restore sets n_past")
	ok(maxdiff(run({ 11, 12 }), want) == 0, arch .. ": full snapshot restores the logits")
	run({ 20, 21, 22, 23 })
	m:restore(c, st)
	ok(maxdiff(run({ 11, 12 }), want) == 0, arch .. ": state checkpoint restores the logits")
	ok(st.bytes < full.bytes and (m.recurrent and st.bytes > 0 or st.bytes == 0),
	    ("%s: checkpoint %d bytes, full snapshot %d"):format(arch, st.bytes, full.bytes))
	full.buf:close()
	st.buf:close()
	replay.drop(c)
end

-- A request through the engine outside the server. Returns the reply.
local function engine_chat(e, messages)
	local uv = require("luv")
	local r = { parts = {} }
	local sink = {}
	function sink.content(_, t)
		r.parts[#r.parts + 1] = t
		return true
	end
	sink.reasoning, sink.tool_arguments = sink.content, sink.content
	function sink.tool_call()
		return 0
	end
	function sink.finish(_, reason, usage, timings)
		r.reason, r.usage, r.timings = reason, usage, timings
	end
	function sink.cancelled()
		return false
	end
	local done
	coroutine.wrap(function()
		r.res, r.err = e:chat({ messages = messages, max_tokens = 3, top_k = 1,
		    chat_template_kwargs = { enable_thinking = false } }, sink)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	assert(r.res, r.err)
	r.text = table.concat(r.parts)
	return r
end

-- The logits the engine gives for messages after its cache logic, and
-- those of a new engine without a cache. Returns both and the cached count.
local function prompt_logits(e, fresh, messages)
	local sink = { cancelled = function() return false end }
	local text = e.family.render(messages, { add_generation_prompt = true, enable_thinking = false })
	local p = e.tk:encode(text)
	local got, want, cached, done
	coroutine.wrap(function()
		cached = e:reuse(p)
		got = e:prefill(p, cached + 1, sink):get()
		-- The reference computes the whole prompt: no slot tokens and
		-- no cached blocks.
		fresh.tokens = {}
		fresh:clear_cache(fresh.cache)
		local a = fresh.pool.alloc
		while true do
			local b = a:evict()
			if not b then
				break
			end
			a:unref(b)
		end
		want = fresh:prefill(p, fresh:reuse(p) + 1, sink):get()
		done = true
	end)()
	while not done do
		require("luv").run("once")
	end
	return got, want, cached, #p
end

local function long_text(seed, n)
	local r = rng(seed)
	local w = {}
	for i = 1, n do
		w[i] = string.char(97 + math.floor(r() * 26), 97 + math.floor(r() * 26))
	end
	return table.concat(w, " ")
end

-- The position before the last turn start of the prompt for messages.
local function last_start(e, messages)
	local text = e.family.render(messages, { add_generation_prompt = true, enable_thinking = false })
	local p = e.tk:encode(text)
	local id = e.tk:special().tokens["<|im_start|>"]
	for i = #p, 1, -1 do
		if p[i] == id then
			return i - 1
		end
	end
end

-- Checkpoints and the RAM store through the engine. Without flash
-- attention, passes split at other positions give the same logits, so
-- a restore must give them exactly.
-- Concurrent chats on a two-slot engine: the prompts prefill and decode
-- in batched passes and give the same replies as one engine per chat.
-- The two chats fit the pool of 256 cells together.
local function concurrent_check(arch, path)
	local engine = require("lgml.engine")
	local uv = require("luv")
	local convs = {
		{ { role = "system", content = "You are a test. " .. long_text(4, 12) },
		  { role = "user", content = long_text(5, 8) } },
		{ { role = "user", content = long_text(6, 20) } },
	}
	local want = {}
	for i, m in ipairs(convs) do
		local one = engine.new({ model = path, backend = "CPU", n_ctx = 512, flash_attn = false })
		want[i] = engine_chat(one, m).text
	end
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 512, flash_attn = false, slots = 2 })
	local got, done = {}, 0
	for i, m in ipairs(convs) do
		local text = {}
		local sink = {
			content = function(_, t) text[#text + 1] = t return true end,
			reasoning = function() return true end, tool_arguments = function() return true end,
			tool_call = function() return 0 end, finish = function() end,
			cancelled = function() return false end,
		}
		coroutine.wrap(function()
			e:chat({ messages = m, max_tokens = 3, top_k = 1,
			    chat_template_kwargs = { enable_thinking = false } }, sink)
			got[i] = table.concat(text)
			done = done + 1
		end)()
	end
	while done < #convs do
		uv.run("once")
	end
	ok(got[1] == want[1] and got[2] == want[2], arch .. ": concurrent chats match one engine each")
	ok((e.batch_rows or 0) > (e.batch_passes or 0), ("%s: prompts and steps shared passes (%d rows in %d passes, %d tokens)")
	    :format(arch, e.batch_rows or 0, e.batch_passes or 0, e.batch_tokens or 0))
end

-- Evict the cached blocks of the engine's pool, so that only the RAM
-- store holds earlier prompts.
local function drop_cached(e)
	local a = e.pool.alloc
	while true do
		local b = a:evict()
		if not b then
			break
		end
		a:unref(b)
	end
end

local function prompt_cache_check(arch, path)
	local engine = require("lgml.engine")
	local tag = arch .. " prompt cache: "
	local fresh = engine.new({ model = path, backend = "CPU", n_ctx = 256, cache_ram = 0,
	    ctx_checkpoints = 0, flash_attn = false, slots = 1 })
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 256, flash_attn = false, slots = 1 })
	local sys = { role = "system", content = "You are a test. " .. long_text(1, 10) }
	local u1 = { role = "user", content = long_text(2, 12) }
	-- Turn 2 extends turn 1 up to its last turn start, then differs.
	local a = engine_chat(e, { sys, u1 })
	local turn2 = { sys, u1, { role = "assistant", content = "not what it said" },
	    { role = "user", content = long_text(3, 8) } }
	local got, want, cached, np = prompt_logits(e, fresh, turn2)
	local d = maxdiff(got, want)
	io.stderr:write(("%sturn 2: %d of %d tokens cached, %d restored, max error %.3g\n"):format(
	    tag, cached, np, e.pc.restored, d))
	if e.model.recurrent then
		ok(cached == last_start(e, { sys, u1 }) and e.pc.restored == cached,
		    tag .. "turn 2 restores the checkpoint at the last turn start of turn 1")
	else
		ok(cached > a.usage.prompt_tokens / 2 and e.pc.restored == 0,
		    tag .. "turn 2 keeps the prefix without a restore")
	end
	ok(d == 0, ("%sturn 2 logits equal a fresh prefill (max error %.3g)"):format(tag, d))

	-- Two conversations in turns: once the pool has lost their blocks,
	-- the second visit restores from the store. Cells taken back from the
	-- cache can be out of position order, which changes the rounding of
	-- attention, so the first visits start with no cached blocks.
	local ua = { role = "user", content = long_text(4, 20) }
	local ub = { role = "user", content = long_text(5, 20) }
	drop_cached(e)
	local ra1 = engine_chat(e, { sys, ua })
	drop_cached(e)
	engine_chat(e, { sys, ub })
	drop_cached(e)
	local ra2 = engine_chat(e, { sys, ua })
	drop_cached(e)
	local rb2 = engine_chat(e, { sys, ub })
	io.stderr:write(("%sA again: %d of %d cached, %d restored; B again: %d cached, %d restored\n")
	    :format(tag, ra2.timings.cache_n, ra2.usage.prompt_tokens, ra2.timings.restored_n,
	    rb2.timings.cache_n, rb2.timings.restored_n))
	-- A hybrid goes back to the checkpoint at the last turn start.
	local wa = e.model.recurrent and last_start(e, { sys, ua }) or ra1.usage.prompt_tokens - 1
	local wb = e.model.recurrent and last_start(e, { sys, ub }) or rb2.usage.prompt_tokens - 1
	ok(ra2.timings.restored_n == wa and ra2.timings.cache_n == wa,
	    tag .. "the second visit of A restores it from the store")
	ok(rb2.timings.restored_n == wb and rb2.timings.cache_n == wb,
	    tag .. "the second visit of B restores it from the store")
	ok(e.pc.stats.loaded >= 2 and e.pc.stats.evicted == 0, tag .. "store loads, no evictions")
	drop_cached(e)
	got, want = prompt_logits(e, fresh, { sys, ua })
	d = maxdiff(got, want)
	-- The pool of 4 blocks is full here: the restore takes blocks back
	-- from the cache, out of position order, so a hybrid rounds
	-- differently.
	ok(d == 0 or (e.model.recurrent and d < 1e-5),
	    ("%slogits after a store restore equal a fresh prefill (max error %.3g)"):format(tag, d))

	-- A budget for about one entry: older entries go.
	e.pc:clear()
	local one = e.pc.bytes
	engine_chat(e, { sys, ua })
	engine_chat(e, { sys, ub })
	one = e.pc.store[1] and e.pc.store[1].bytes or 0
	e.pc.ram = one * 3 // 2
	local uc = { role = "user", content = long_text(6, 20) }
	engine_chat(e, { sys, uc })
	ok(#e.pc.store == 1 and e.pc.bytes <= e.pc.ram and e.pc.stats.evicted >= 1,
	    ("%sthe budget evicts the oldest entry (%d bytes each)"):format(tag, one))
	drop_cached(e)
	local r = engine_chat(e, { sys, ua })
	ok(r.timings.restored_n < r.usage.prompt_tokens // 2, tag .. "an evicted prompt prefills again")
	e.pc:clear()
	fresh.pc = nil
end

-- One request through the engine. Returns the result, the error, the
-- status, the content text and the finish reason.
local function engine_req(e, req)
	local parts, reason = {}, nil
	local sink = {}
	function sink.content(_, t)
		parts[#parts + 1] = t
		return true
	end
	sink.reasoning, sink.tool_arguments = sink.content, sink.content
	function sink.tool_call()
		return 0
	end
	function sink.finish(_, r)
		reason = r
	end
	function sink.cancelled()
		return false
	end
	local res, err, status, done
	coroutine.wrap(function()
		res, err, status = e:chat(req, sink)
		done = true
	end)()
	while not done do
		require("luv").run("once")
	end
	return res, err, status, table.concat(parts), reason
end

-- Random weights give adversarial output: the grammar must hold anyway.
local function grammar_engine_check(arch, path)
	local json = require("lgml.json")
	local e = require("lgml.engine").new({ model = path, backend = "CPU", n_ctx = 512 })
	local tag = arch .. " grammar: "
	local msgs = { { role = "user", content = "Give JSON." } }
	local schema = {
		type = "object",
		properties = {
			a = { type = "integer", minimum = 0, maximum = 99 },
			b = { enum = { "x", "y" } },
			c = { type = "array", items = { type = "boolean" }, minItems = 1, maxItems = 3 },
		},
		required = { "a", "b", "c" },
	}
	local good = 0
	for seed = 1, 8 do
		local res, err, _, text, reason = engine_req(e, {
			messages = msgs, max_tokens = 300, seed = seed, temperature = 1.5,
			chat_template_kwargs = { enable_thinking = false },
			response_format = { type = "json_schema", json_schema = { schema = schema } },
		})
		local v = res and json.decode(text)
		if v and reason == "stop" and math.type(v.a) == "integer" and v.a >= 0 and v.a <= 99 and
		    (v.b == "x" or v.b == "y") and #v.c >= 1 and #v.c <= 3 and type(v.c[1]) == "boolean" then
			good = good + 1
		else
			io.stderr:write(("seed %d: %s %s %q\n"):format(seed, tostring(err), tostring(reason), text))
		end
	end
	ok(good == 8, tag .. good .. " of 8 random-weight answers match the schema")
	local res, _, _, text, reason = engine_req(e, {
		messages = msgs, max_tokens = 50, seed = 3, temperature = 1.5,
		chat_template_kwargs = { enable_thinking = false },
		grammar = 'root ::= "ab" [0-9]{2} "!"',
	})
	ok(res and text:match("^ab%d%d!$") and reason == "stop", tag .. "raw GBNF: " .. text)
	res, _, _, text = engine_req(e, {
		messages = msgs, max_tokens = 40, seed = 3, temperature = 1.5,
		response_format = { type = "json_object" },
	})
	ok(res and (#text == 0 or text:match("^%s*{")), tag .. "json_object with thinking starts an object")
	local _, err, status = engine_req(e, { messages = msgs, grammar = "root ::= nope" })
	ok(status == 400 and err:find("undefined rule"), tag .. "bad grammar is 400: " .. tostring(err))
	_, err, status = engine_req(e, { messages = msgs,
	    response_format = { type = "json_schema", json_schema = { schema = { type = "nope" } } } })
	ok(status == 400, tag .. "bad schema is 400: " .. tostring(err):gsub("%c", " "))
	local r2 = engine_req(e, { messages = msgs, max_tokens = 3, top_k = 1,
	    chat_template_kwargs = { enable_thinking = false } })
	ok(r2, tag .. "the engine serves after errors")
end

-- Weight placement: parts of the model on a second CPU backend give
-- the same logits as one backend, through the scheduler, and replay
-- keeps one graph per shape.
local function offload_check(arch, path, m, ref, long)
	local placement = require("lgml.placement")
	local transformer = require("lgml.transformer")
	local sb <close> = assert(lgml.backend(os.getenv("LGML_DEVICE") or "CPU", { cpu = true }))
	local cb = sb:cpu()
	local want = our_rows(m, ref)
	local cases = {
		{ "cpu_moe", { cpu_moe = true } },
		{ "n_cpu_moe 1", { n_cpu_moe = 1 } },
		{ "n_gpu_layers 1", { n_gpu_layers = 1 } },
		{ "override", { override = { "attn_output=GPU", "attn=CPU" } } },
	}
	for _, c in ipairs(cases) do
		local tag = ("%s %s: "):format(arch, c[1])
		local s = transformer.load(sb, path, { placement = placement.rules(c[2]) })
		local L1, L2 = s.layers[1], s.layers[#s.layers]
		if c[1] == "cpu_moe" then
			ok(L1.down_exps:backend() == cb and L2.down_exps:backend() == cb and
			    L1.gate_inp:backend() == sb and s.output:backend() == sb,
			    tag .. "experts on the CPU, router and output on the backend")
		elseif c[1] == "n_cpu_moe 1" then
			ok(L1.down_exps:backend() == cb and L2.down_exps:backend() == sb,
			    tag .. "experts of the first layer on the CPU")
		elseif c[1] == "n_gpu_layers 1" then
			ok(L1.attn_norm:backend() == cb and L2.attn_norm:backend() == sb and
			    s.tok_embd:backend() == cb and s.output:backend() == cb and L1.cpu and not L2.cpu,
			    tag .. "the first layers, embeddings and output on the CPU")
		else
			local wq = L2.wq or L2.qkv
			ok((not wq or wq:backend() == cb) and L2.attn_norm:backend() == cb and
			    L2.wo:backend() == sb, tag .. "patterns in order")
		end
		local dev, host = s.w:size()
		ok(host > 0 and dev > 0, ("%sweights split: %d bytes on the device, %d on the CPU")
		    :format(tag, dev, host))
		local got = our_rows(s, ref)
		local d = 0
		for i = 1, #want do
			d = math.max(d, maxdiff(got[i] or {}, want[i]))
		end
		ok(#got == #want and d < ((os.getenv("LGML_DEVICE") or "CPU") == "CPU" and 1e-4 or LOOSE), ("%slogits equal one backend (max error %.3g)"):format(tag, d))
		rc.decode(ok, tag, s, { n_ctx = 320, prompt = long, n = 12, flash_attn = true, builds = 3 })
		s:close()
	end
	-- The engine places weights only on a GPU backend.
	local dev = os.getenv("LGML_DEVICE") or "CPU"
	local e = require("lgml.engine").new({ model = path, backend = dev, n_ctx = 64, n_cpu_moe = 1,
	    override = { "output=CPU" } })
	local L1 = e.model.layers[1]
	if dev == "CPU" then
		ok(L1.down_exps:backend() == e.be and not e.be:cpu(), arch .. " engine: no placement on the CPU")
	else
		ok(L1.down_exps:backend() == e.be:cpu() and e.model.output:backend() == e.be:cpu() and
		    e.model.layers[2].down_exps:backend() == e.be, arch .. " engine: n_cpu_moe and override")
	end
	local r = engine_chat(e, { { role = "user", content = "hi" } })
	ok(r.usage.completion_tokens >= 1, arch .. " engine: chat with placement")
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
	local model = require("lgml.transformer")
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
	snapshot_check(arch, m)
	batch_check(arch, m)
	if ref and (arch == "qwen3moe" or arch == "qwen3next" or arch == "qwen35moe") then
		offload_check(arch, path, m, ref, long)
	end
	if arch == "qwen35moe" or arch == "qwen3" then
		prompt_cache_check(arch, path)
		concurrent_check(arch, path)
	end
	if arch == "qwen3" then
		grammar_engine_check(arch, path)
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
		local m = require("lgml.transformer").load(rb, path)
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
