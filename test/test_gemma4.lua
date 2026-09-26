-- Gemma 4 (lgml.transformer) on tiny random-weight ggufs made here, against
-- llama.cpp logits, and replay against a new graph per pass. Output is TAP. Set LGML_LLAMA_LOGITS to a
-- scripts/llama_logits.c binary to compare every logit live;
-- LGML_REF_DUMP=1 prints new reference rows. LGML_GEMMA4 names a real
-- gguf to run on LGML_DEVICE (default CPU).
local lgml = require("lgml")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local gguf = require("gguf")
local bpe_vocab = require("bpe_vocab")
local rc = require("replay_check")
local transformer = require("lgml.transformer")

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

local E, H, D, DS, FF, NSWA = 64, 4, 32, 16, 96, 6
local X, USED, FX, PLE = 4, 2, 16, 8

-- kv and tensors for a tiny model. "dense" has a sliding and a global
-- layer. "moe" has four layers with routed experts, per-layer inputs
-- and two layers that share the cache of earlier layers. extra lists
-- control tokens to add at the end of the vocab.
local function build(kind, extra)
	local r = rng(#kind * 7919)
	local kv = bpe_vocab.gemma4_kv()
	local nv
	for _, e in ipairs(kv) do
		if e[1] == "tokenizer.ggml.tokens" or e[1] == "tokenizer.ggml.token_type" then
			for _, piece in ipairs(extra or {}) do
				table.insert(e[3], e[1]:find("type") and 3 or piece)
			end
		end
		if e[1] == "tokenizer.ggml.tokens" then
			nv = #e[3]
		end
	end
	local ts = {}
	local A = "gemma4."
	local function key(k, t, v)
		kv[#kv + 1] = { A .. k, t, v }
	end
	local function rand(count, lo, hi)
		local v = {}
		for i = 1, count do
			v[i] = lo + (hi - lo) * r()
		end
		return v
	end
	-- Weight scaled for the fan-in (last dimension).
	local function w(name, shape, gain)
		local c = 1
		for _, d in ipairs(shape) do
			c = c * d
		end
		local s = (gain or 1) * math.sqrt(3 / shape[#shape])
		ts[#ts + 1] = { name, shape, rand(c, -s, s) }
	end
	local function norm(name, d)
		ts[#ts + 1] = { name, { d }, rand(d, 0.8, 1.2) }
	end

	local moe = kind == "moe"
	local nl = moe and 4 or 2
	local swa, hkv = {}, {}
	for i = 1, nl do
		swa[i] = i % 2 == 1
		hkv[i] = swa[i] and 2 or 1
	end
	kv[#kv + 1] = { "general.architecture", gguf.STRING, "gemma4" }
	kv[#kv + 1] = { "general.name", gguf.STRING, "tiny gemma4 " .. kind }
	key("block_count", gguf.UINT32, nl)
	key("context_length", gguf.UINT32, 256)
	key("embedding_length", gguf.UINT32, E)
	key("feed_forward_length", gguf.UINT32, FF)
	key("attention.head_count", gguf.UINT32, H)
	key("attention.head_count_kv", { gguf.ARRAY, gguf.INT32 }, hkv)
	key("attention.key_length", gguf.UINT32, D)
	key("attention.value_length", gguf.UINT32, D)
	key("attention.key_length_swa", gguf.UINT32, DS)
	key("attention.value_length_swa", gguf.UINT32, DS)
	key("attention.layer_norm_rms_epsilon", gguf.FLOAT32, 1e-6)
	key("attention.sliding_window", gguf.UINT32, NSWA)
	key("attention.sliding_window_pattern", { gguf.ARRAY, gguf.BOOL }, swa)
	key("attention.shared_kv_layers", gguf.UINT32, moe and 2 or 0)
	key("rope.dimension_count", gguf.UINT32, D)
	key("rope.dimension_count_swa", gguf.UINT32, DS)
	key("rope.freq_base", gguf.FLOAT32, 1e6)
	key("rope.freq_base_swa", gguf.FLOAT32, 1e4)
	key("final_logit_softcapping", gguf.FLOAT32, 30)
	key("embedding_length_per_layer_input", gguf.UINT32, moe and PLE or 0)
	if moe then
		key("expert_count", gguf.UINT32, X)
		key("expert_used_count", gguf.UINT32, USED)
		key("expert_feed_forward_length", gguf.UINT32, FX)
	end

	w("token_embd.weight", { nv, E }, 0.3)
	norm("output_norm.weight", E)
	-- Proportional rope: the second half of the pairs does not rotate.
	local freqs = {}
	for i = 1, D // 2 do
		freqs[i] = i <= D // 4 and 1 or 1e30
	end
	ts[#ts + 1] = { "rope_freqs.weight", { D // 2 }, freqs }
	if moe then
		w("per_layer_token_embd.weight", { nv, nl * PLE }, 0.5)
		w("per_layer_model_proj.weight", { nl * PLE, E })
		norm("per_layer_proj_norm.weight", PLE)
	end
	for i = 0, nl - 1 do
		local p = "blk." .. i .. "."
		local d = swa[i + 1] and DS or D
		local shared = moe and i >= 2
		norm(p .. "attn_norm.weight", E)
		w(p .. "attn_q.weight", { H * d, E })
		if not shared then
			w(p .. "attn_k.weight", { hkv[i + 1] * d, E })
			norm(p .. "attn_k_norm.weight", d)
			-- Global layers of the real models have no attn_v.
			if swa[i + 1] then
				w(p .. "attn_v.weight", { hkv[i + 1] * d, E })
			end
		end
		w(p .. "attn_output.weight", { E, H * d })
		norm(p .. "attn_q_norm.weight", d)
		norm(p .. "post_attention_norm.weight", E)
		norm(p .. "ffn_norm.weight", E)
		w(p .. "ffn_gate.weight", { FF, E })
		w(p .. "ffn_up.weight", { FF, E })
		w(p .. "ffn_down.weight", { E, FF })
		norm(p .. "post_ffw_norm.weight", E)
		if moe then
			norm(p .. "post_ffw_norm_1.weight", E)
			norm(p .. "pre_ffw_norm_2.weight", E)
			norm(p .. "post_ffw_norm_2.weight", E)
			-- This gain keeps the second and third router probabilities
			-- at least 3e-3 apart, so f16 rounding picks the same experts.
			w(p .. "ffn_gate_inp.weight", { X, E }, 8)
			norm(p .. "ffn_gate_inp.scale", E)
			if i == 0 then
				w(p .. "ffn_gate_up_exps.weight", { X, 2 * FX, E })
			else
				w(p .. "ffn_gate_exps.weight", { X, FX, E })
				w(p .. "ffn_up_exps.weight", { X, FX, E })
			end
			w(p .. "ffn_down_exps.weight", { X, E, FX })
			ts[#ts + 1] = { p .. "ffn_down_exps.scale", { X }, rand(X, 0.7, 1.3) }
			w(p .. "inp_gate.weight", { PLE, E })
			w(p .. "proj.weight", { E, PLE })
			norm(p .. "post_norm.weight", E)
		else
			ts[#ts + 1] = { p .. "layer_output_scale.weight", { 1 }, { 0.8 + 0.3 * i } }
		end
	end
	return kv, ts, nv
end

-- A prompt longer than the sliding window, a batch at n_past, then
-- greedy steps on llama.cpp's choices.
local batches = {
	{ 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 21, 41, 61, 81, 101, 121 },
	{ 270, 30, 50, 70, 90 },
}
local N_DECODE = 6
local IDX = { 1, 37, 73, 109, 145, 181, 217, 253, 289 }
-- Reference rows: input id, argmax, logits at IDX. From llama.cpp on
-- the CPU with f16 KV cache.
local REF = {}
REF.dense = {
	{ id = 20, best = 66, v = { -0.259937, -0.121262, -0.106518, 0.0242489, -0.10353, 0.074525, -0.716905, -0.262208, 0.295995 } },
	{ id = 40, best = 275, v = { -0.523163, -0.271153, -0.451639, 0.285485, -0.0463441, 0.0541316, -1.07747, -0.289506, 0.193628 } },
	{ id = 60, best = 96, v = { 0.00189176, -0.385258, 0.187099, -0.0705348, 0.205595, -0.409999, -0.00600769, -0.308243, -0.189253 } },
	{ id = 80, best = 69, v = { 0.106275, -0.0139313, -0.251412, -0.024601, 0.0216769, -0.472591, -0.268039, 0.312823, 0.540558 } },
	{ id = 100, best = 125, v = { 0.0757409, -0.403967, -0.0894041, 0.0133389, -0.321089, -0.134211, -0.289164, -0.44633, 0.437401 } },
	{ id = 120, best = 217, v = { -0.284697, -0.0295383, 0.526762, -0.0905151, -0.430256, -0.0134743, 0.022642, -0.104178, 0.159136 } },
	{ id = 140, best = 232, v = { 0.187754, -0.497159, 0.357348, 0.107882, -0.0872565, 0.446308, 0.346884, 0.0637101, -0.294103 } },
	{ id = 160, best = 158, v = { -0.291647, -0.309106, -0.290024, -0.14284, 0.176059, 0.200506, -0.485508, 0.0628057, 0.410098 } },
	{ id = 180, best = 231, v = { -0.364495, -0.00359919, -0.309481, 0.349206, 0.439976, 0.644278, 0.101631, -0.505123, -0.344131 } },
	{ id = 200, best = 118, v = { -0.141502, -0.0191725, -0.285144, 0.0728029, 0.278722, 0.45029, -0.344081, -0.243616, -0.10797 } },
	{ id = 220, best = 70, v = { 0.453045, 0.0354898, -0.115118, -0.00904548, -0.385089, 0.124667, 0.338921, -0.421713, -0.0387589 } },
	{ id = 240, best = 222, v = { -0.327783, 0.0270996, -0.240241, -0.168434, 0.345149, 0.0304449, -0.269096, -0.510553, 0.0875987 } },
	{ id = 260, best = 99, v = { -0.0889372, -0.252446, 0.256237, 0.0332501, -0.0148597, 0.25422, -0.00946314, 0.408821, 0.274484 } },
	{ id = 280, best = 280, v = { 0.164544, -0.225023, 0.0160821, -0.0854894, 0.281747, 0.295275, 0.278378, 0.0985577, 0.355725 } },
	{ id = 21, best = 17, v = { -0.0201341, -0.109054, -0.0709063, -0.0454963, -0.351438, 0.379682, -0.206519, -0.39333, 0.221194 } },
	{ id = 41, best = 111, v = { 0.164185, -0.146206, -0.00202846, -0.46102, 0.105892, -0.0773358, 0.0691046, -0.0626011, 0.119397 } },
	{ id = 61, best = 201, v = { 0.529817, -0.139087, -0.103554, 0.0980644, 0.517398, -0.197555, -0.03556, 0.0688648, 0.0158529 } },
	{ id = 81, best = 181, v = { -0.0899597, 0.185036, -0.249783, 0.287919, -0.301647, 0.23173, -0.083815, 0.298695, 0.292458 } },
	{ id = 101, best = 77, v = { 0.355559, 0.212655, -0.339118, 0.00100878, -0.430998, -0.257582, -0.127174, 0.402835, 0.0446248 } },
	{ id = 121, best = 272, v = { 0.0480036, 0.00767002, 0.226041, -0.0761361, 0.215652, -0.290832, 0.109503, 0.199781, -0.255872 } },
	{ id = 270, best = 142, v = { 0.143818, -0.484245, 0.362997, 0.138738, 0.823721, -0.0062772, 0.43333, 0.0144995, -0.174645 } },
	{ id = 30, best = 273, v = { 0.0431483, -0.021273, -0.0980124, -0.195805, -0.376527, -0.101572, 0.148678, -0.364582, -0.142932 } },
	{ id = 50, best = 137, v = { -0.0723759, -0.241656, 0.365101, -0.550366, 0.155309, -0.72687, 0.500639, -0.250338, -0.278344 } },
	{ id = 70, best = 70, v = { 0.0243836, -0.318614, 0.0537053, -0.183117, 0.199991, -0.0408372, 0.0581787, 0.1824, -0.0213682 } },
	{ id = 90, best = 95, v = { -0.154618, -1.02663, 0.491205, -0.302612, 0.118511, -0.269613, 0.646288, 0.278693, -0.371697 } },
	{ id = 95, best = 147, v = { -0.04521, -0.538659, 0.267049, -0.360457, 0.29126, -0.648116, 0.192302, -0.144606, -0.165687 } },
	{ id = 147, best = 233, v = { 0.220219, 0.0599778, -0.013781, 0.224286, -0.15807, -0.0653272, 0.386717, 0.31238, -0.0562207 } },
	{ id = 233, best = 2, v = { -0.104499, -0.0362872, -0.498213, 0.497651, -0.37031, -0.375431, -0.383823, -0.0156035, -0.140097 } },
	{ id = 2, best = 95, v = { -0.0833117, -0.614188, -0.20717, 0.527614, 0.218383, 0.121834, 0.326023, -0.0825663, -0.00222066 } },
	{ id = 95, best = 272, v = { -0.0919804, -0.231077, 0.0665824, -0.168583, 0.629983, -0.136297, 0.572278, 0.326275, -0.745022 } },
	{ id = 272, best = 272, v = { 0.239055, -0.636477, -0.0428899, -0.0526921, 0.0789291, -0.193016, 0.602036, -0.105941, -0.250266 } },
}
REF.moe = {
	{ id = 20, best = 28, v = { 0.404688, -0.0326983, -0.0713767, -0.164673, -0.307094, -0.246597, -0.114804, 0.0739947, -0.16143 } },
	{ id = 40, best = 66, v = { -0.399575, -0.11791, 0.322234, 0.377007, -0.304361, -0.133334, -0.437123, 0.288525, 0.173697 } },
	{ id = 60, best = 66, v = { -0.83646, -0.188536, -0.268833, 0.236242, -0.0592452, 0.0439757, -0.475485, 0.184527, 0.467745 } },
	{ id = 80, best = 104, v = { -0.513515, 0.1196, -0.42269, -0.322465, 0.0647145, 0.139256, -0.421758, 0.460163, -0.0368253 } },
	{ id = 100, best = 147, v = { -0.307781, 0.0504023, -0.239773, 0.298596, -0.139786, -0.242996, 0.0747281, 0.341034, -0.030034 } },
	{ id = 120, best = 179, v = { -0.839642, -0.617117, 0.149244, 0.289111, -0.433029, -0.0492982, -0.200309, 0.253696, 0.176058 } },
	{ id = 140, best = 260, v = { -0.576745, 0.198312, 0.0485453, -0.184308, 0.119481, 0.511551, -0.66776, 0.011624, 0.50136 } },
	{ id = 160, best = 157, v = { -0.147485, 0.244362, -0.272007, 0.141016, -0.316179, 0.153397, -0.257119, 0.157559, 0.298822 } },
	{ id = 180, best = 82, v = { 0.186052, 0.423193, 0.00016515, 0.220155, -0.176311, 0.229034, 0.178865, -0.260129, 0.0903595 } },
	{ id = 200, best = 226, v = { -0.174574, 0.537203, -0.0872922, 0.179064, -0.11763, -0.0998023, -0.145161, 0.00175647, 0.306542 } },
	{ id = 220, best = 95, v = { 0.230223, 0.553318, 0.195501, 0.202191, -0.821957, -0.1841, 0.163259, -0.143292, -0.171113 } },
	{ id = 240, best = 212, v = { 0.11838, 0.561515, -0.0490796, -0.23395, -0.15112, -0.115036, 0.0162774, -0.493553, -0.179127 } },
	{ id = 260, best = 188, v = { -0.719117, 0.143073, -0.105524, 0.0713514, -0.020555, -0.421315, -0.138605, 0.0461324, 0.130471 } },
	{ id = 280, best = 107, v = { 0.492527, 0.309811, -0.284195, -0.0691065, 0.347339, 0.179619, -0.314014, -0.108103, 0.212207 } },
	{ id = 21, best = 273, v = { -0.0925451, 0.270599, -0.118333, -0.071209, 0.173951, -0.231889, 0.35202, 0.142707, -0.285726 } },
	{ id = 41, best = 102, v = { -0.462814, -0.422943, -0.567944, 0.026478, -0.160937, -0.162814, -0.220224, 0.57257, 0.0973087 } },
	{ id = 61, best = 169, v = { -0.340555, -0.0353158, -0.148359, 0.0534515, 0.542802, -0.184971, -0.559223, 0.00565551, -0.0633387 } },
	{ id = 81, best = 255, v = { 0.0936001, 0.183806, -0.11592, -0.0822957, 0.117601, 0.220076, -0.111306, -0.0708486, 0.0466021 } },
	{ id = 101, best = 187, v = { -0.466744, 0.150582, 0.033692, 0.102709, 0.173968, -0.0454132, 0.0036763, 0.364903, 0.353084 } },
	{ id = 121, best = 81, v = { 0.0742622, -0.373434, -0.153735, -0.143757, 0.345292, -0.256792, -0.275796, 0.0873866, -0.0910779 } },
	{ id = 270, best = 265, v = { 0.159511, 0.552726, -0.143074, -0.306849, 0.385714, 0.000634879, -0.522535, 0.481159, 0.456383 } },
	{ id = 30, best = 182, v = { -0.122178, 0.619996, 0.0897393, -0.201036, 0.507612, -0.208626, -0.342169, 0.401211, -0.349615 } },
	{ id = 50, best = 232, v = { -0.276159, -0.354788, -0.108741, 0.0145938, 0.0364097, 0.337783, 0.110124, 0.607354, 0.336784 } },
	{ id = 70, best = 31, v = { -0.277844, -0.594481, 0.101215, 0.187086, -0.135361, 0.116047, 0.0850603, -0.134103, -0.0846746 } },
	{ id = 90, best = 176, v = { 0.159892, -0.0489648, -0.42791, -0.333479, 0.050872, -0.0684839, 0.191049, -0.0376361, -0.201671 } },
	{ id = 176, best = 201, v = { 0.114168, 0.425604, -0.262515, 0.0110816, -0.38146, 0.176623, 0.400688, -0.118454, -0.260461 } },
	{ id = 201, best = 92, v = { 0.151921, 0.496807, -0.00256911, 0.194917, 0.351149, 0.338794, 0.421482, -0.156349, -0.614727 } },
	{ id = 92, best = 43, v = { 0.0844731, -0.703216, -0.613065, -0.30878, -0.0567145, -0.164013, -0.0321885, 0.13685, -0.344658 } },
	{ id = 43, best = 157, v = { -0.265981, 0.0169102, -0.0972319, -0.0176686, -0.181558, 0.193656, 0.414893, 0.0182965, -0.314794 } },
	{ id = 157, best = 40, v = { -0.549941, -0.563464, -0.185138, -0.205164, -0.223048, 0.0397418, -0.437084, 0.591623, 0.237077 } },
	{ id = 40, best = 190, v = { -0.429717, -0.201886, 0.659937, 0.57934, -0.331019, -0.271844, -0.406541, 0.303611, -0.610613 } },
}

-- Rows from the llama_logits binary: { id, logits }.
local function llama_rows(bin, path)
	local args = { "nice -n 19", bin, path, N_DECODE }
	for i, b in ipairs(batches) do
		if i > 1 then
			args[#args + 1] = "/"
		end
		for _, id in ipairs(b) do
			args[#args + 1] = id
		end
	end
	local p = assert(io.popen("VK_DRIVER_FILES=/nonexistent VK_ICD_FILENAMES=/nonexistent " ..
	    table.concat(args, " ")))
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

-- Run ids at cache.n_past in passes of at most ub tokens. Returns one
-- logits list per token.
local function eval(m, cache, ids, opts, ub)
	local rows = {}
	ub = ub or #ids
	for s = 1, #ids, ub do
		local part = table.move(ids, s, math.min(#ids, s + ub - 1), 1, {})
		local inp = m:inputs(part, cache, opts)
		local g <close> = m.be:graph()
		local logits = m:forward(g, inp)
		g:compute(logits)
		if cache then
			cache.n_past = inp.n_kv
		end
		local flat, nv = logits:get(), logits:shape()[2]
		for i = 1, #part do
			rows[#rows + 1] = table.move(flat, (i - 1) * nv + 1, i * nv, 1, {})
		end
		for _, t in pairs(inp) do
			if type(t) == "userdata" then
				t:close()
			end
		end
	end
	return rows
end

-- Our rows for the batches and the decode ids of ref. copts go to
-- the cache; opts go to inputs.
local function our_rows(m, ref, copts, opts)
	local c = m:cache(64, copts)
	local rows = {}
	local function add(ids)
		for _, r in ipairs(eval(m, c, ids, opts, c.ubatch)) do
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

-- GPU backends round to f16, as llama.cpp's own GPU run does.
local GPU = (os.getenv("LGML_DEVICE") or "CPU") ~= "CPU"
local TOL, LOOSE = GPU and 2e-2 or 2e-3, GPU and 5e-2 or 3e-2

-- Checks beyond the logits: cache layouts, flash attention, a pass
-- without a cache, greedy steps and rewinds of the ring.
local function more(kind, m, ref)
	local tag = kind .. ": "
	local err, same = compare(our_rows(m, ref, { ubatch = 64 }), ref)
	ok(err < TOL and same, ("%sfull sliding cache agrees (max error %.3g)"):format(tag, err))
	err, same = compare(our_rows(m, ref, { ubatch = 8 }, { flash_attn = true }), ref)
	ok(err < LOOSE and same, ("%sflash_attn on the ring agrees (max error %.3g)"):format(tag, err))
	-- An f32 cache does not round K and V as the reference does.
	err, same = compare(our_rows(m, ref, { ubatch = 8, type = "f32" }), ref)
	ok(err < LOOSE and same, ("%sf32 ring agrees (max error %.3g)"):format(tag, err))

	local prompt = {}
	for _, b in ipairs(batches) do
		table.move(b, 1, #b, #prompt + 1, prompt)
	end
	local nb = #prompt
	local seq = table.move(prompt, 1, nb, 1, {})
	for i = nb + 1, #ref do
		seq[#seq + 1] = ref[i].id
	end
	err, same = compare(eval(m, nil, seq), ref)
	ok(err < LOOSE and same, ("%spass without a cache agrees (max error %.3g)"):format(tag, err))

	local gen = require("lgml.generate")
	local want = {}
	for i = nb + 1, #ref do
		want[#want + 1] = ref[i].id
	end
	want[#want + 1] = best(ref[#ref])
	local c = m:cache(64, { ubatch = 8 })
	local ids = gen.greedy(m, c, prompt, #want)
	ok(table.concat(ids, ",") == table.concat(want, ","), tag .. "greedy steps match llama.cpp")
	ok(c.n_past == nb + #want - 1, tag .. "cache advanced")

	-- The ring holds 13 rows; 31 positions went in. A rewind to 25 keeps
	-- positions 20 to 24, which the ring still holds; 10 does not.
	ok(m:rewind(c, 25) == 25, tag .. "rewind inside the ring keeps the prefix")
	c.n_past = 25
	err = compare(eval(m, c, table.move(seq, 26, #seq, 1, {}), nil, 8),
	    table.move(ref, 26, #ref, 1, {}))
	ok(err < TOL, ("%srows after a rewind agree (max error %.3g)"):format(tag, err))
	ok(m:rewind(c, 10) == 0 and c.hi == 0, tag .. "rewind past the ring starts over")
	c.n_past = 0
	ok(not pcall(m.inputs, m, seq, c), tag .. "a pass over ubatch fails once the ring wraps")
end

-- Ring checkpoints: go back past what the ring holds by a state
-- snapshot, and past it by a full snapshot. Passes of 8 tokens repeat,
-- so the logits must match exactly.
local function ring_check(kind, m)
	local replay = require("lgml.replay")
	local tag = kind .. " ring snapshot: "
	local c = m:cache(64, { ubatch = 8 })
	local function run(from, to)
		local out
		for s = from, to, 8 do
			local ids = {}
			for i = s, math.min(to, s + 7) do
				ids[#ids + 1] = (i * 37) % 200 + 1
			end
			out = replay.run(m, c, ids, { logits = "last" }):get()
		end
		return out
	end
	local function same(a, b)
		for i = 1, #a do
			if a[i] ~= b[i] then
				return false
			end
		end
		return #a == #b
	end
	run(1, 24)
	local st, full = m:snapshot(c), m:snapshot(c, true)
	local want = run(25, 32)
	run(33, 48)
	ok(m:rewind(c, 24, true) == 0, tag .. "the ring lost position 24")
	m:restore(c, st)
	ok(c.n_past == 24 and c.hi == 24, tag .. "a checkpoint sets n_past and hi")
	ok(same(run(25, 32), want), tag .. "logits after a checkpoint restore match")
	run(33, 48)
	c.n_past = 0
	m:rewind(c, 0)
	m:restore(c, full)
	ok(same(run(25, 32), want), tag .. "logits after a full restore match")
	ok(st.bytes > 0 and st.bytes < full.bytes, ("%scheckpoint %d bytes, full %d"):format(tag,
	    st.bytes, full.bytes))
	st.buf:close()
	full.buf:close()
	replay.drop(c)
end

-- Embedding passes: token embeddings given as opts.embd give the same
-- logits as the ids, and a non-causal span sees its later rows.
local function embd_check(kind, m)
	local replay = require("lgml.replay")
	local tag = kind .. " embd: "
	local be, E = m.be, m.hp.n_embd
	local ids = batches[1]
	local n = #ids
	-- Scaled token embeddings of ids, in a host buffer.
	local buf = lgml.hostbuf(n * E * 4)
	do
		local leaf <close> = be:tensor("i32", { n }, ids)
		local g <close> = be:graph()
		local x = g:scale(g:get_rows(m.tok_embd, leaf), m.hp.embd_scale)
		g:compute(x)
		x:to_host(buf)
	end
	local function logits(c, part, opts)
		local inp = m:inputs(part, c, opts)
		local g <close> = be:graph()
		local out = m:forward(g, inp)
		g:compute(out)
		if c then
			c.n_past = inp.n_kv
		end
		local v = out:get()
		for _, t in pairs(inp) do
			if type(t) == "userdata" then
				t:close()
			end
		end
		return v
	end
	local function maxdiff(a, b)
		local d = #a == #b and 0 or math.huge
		for i = 1, math.min(#a, #b) do
			d = math.max(d, math.abs(a[i] - b[i]))
		end
		return d
	end
	local a = logits(m:cache(64, { ubatch = 32 }), ids)
	local b = logits(m:cache(64, { ubatch = 32 }), ids, { embd = buf })
	ok(maxdiff(a, b) < 1e-4, ("%sembeddings match ids (max diff %.3g)"):format(tag, maxdiff(a, b)))

	-- Text, a span of 6 embedding rows, then text. The span rows are
	-- token embeddings 7 to 12 of ids, at byte offset 6 rows.
	local zeros = { 0, 0, 0, 0, 0, 0 }
	local head, tail = { 5, 6, 7, 8 }, { 9, 10, 11 }
	local off = 6 * E * 4
	local function mixed(copts, fa)
		local c = m:cache(64, copts)
		local o = { flash_attn = fa }
		logits(c, head, o)
		local img = logits(c, zeros, { embd = buf, embd_off = off, causal = false,
			flash_attn = fa })
		return img, logits(c, tail, o)
	end
	local img, after = mixed({ ubatch = 64 })
	local img2, after2 = mixed({ ubatch = 8 })
	local d = math.max(maxdiff(img, img2), maxdiff(after, after2))
	ok(d < TOL, ("%sa non-causal span on the ring agrees with a full cache (max diff %.3g)")
	    :format(tag, d))
	local img3, after3 = mixed({ ubatch = 8 }, true)
	ok(maxdiff(img, img3) < 3e-2 and maxdiff(after, after3) < 3e-2,
	    ("%sflash_attn agrees (max diff %.3g)"):format(tag, math.max(maxdiff(img, img3),
	    maxdiff(after, after3))))
	-- The first row of a non-causal span sees the later rows.
	local c = m:cache(64, { ubatch = 8 })
	logits(c, head)
	local causal = logits(c, zeros, { embd = buf, embd_off = off })
	local nv = m.hp.n_vocab
	local first = function(v) return table.move(v, 1, nv, 1, {}) end
	ok(maxdiff(first(causal), first(img2)) > 1e-2, tag .. "a non-causal span sees later rows")
	-- Replay: one graph for two spans with different embeddings.
	local function replayed(off)
		local rc2 = m:cache(64, { ubatch = 8 })
		replay.run(m, rc2, head, {})
		local out = replay.run(m, rc2, zeros, { embd = buf, embd_off = off, causal = false })
		return out:get(), rc2
	end
	local r1, rc1 = replayed(off)
	ok(maxdiff(r1, img2) < 1e-4, tag .. "replay gives the same span logits")
	rc1.n_past = 4
	local r2 = replay.run(m, rc1, zeros, { embd = buf, embd_off = 0, causal = false }):get()
	local fresh = replayed(0)
	ok(maxdiff(r2, fresh) < 1e-4, tag .. "a replayed span loads new embeddings")
	replay.drop(rc1)
	buf:close()
end

-- Run one chat request on engine e; returns { res, err, status,
-- reason, usage, text }.
local function chat_on(e, messages, extra)
	local uv = require("luv")
	local r = { text = {} }
	local sink = {}
	function sink.content(_, t)
		r.text[#r.text + 1] = t
		return true
	end
	sink.reasoning, sink.tool_arguments = sink.content, sink.content
	function sink.tool_call()
		return 0
	end
	function sink.finish(_, reason, usage)
		r.reason, r.usage = reason, usage
	end
	function sink.cancelled()
		return false
	end
	local req = { messages = messages, max_tokens = 4 }
	for k, v in pairs(extra or {}) do
		req[k] = v
	end
	local done
	coroutine.wrap(function()
		r.res, r.err, r.status = e:chat(req, sink)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	r.text = table.concat(r.text)
	return r
end

-- Image input on a tiny model and a tiny mmproj: prompt positions, the
-- cached prefix, and request errors.
local function vision_check()
	local tiny = require("mmproj_tiny")
	local base64 = require("lgml.base64")
	local path, mpath = os.tmpname(), os.tmpname()
	local kv, ts = build("dense", { "<|image>", "<|image|>", "<image|>" })
	gguf.write(path, kv, ts)
	local mkv, mts = tiny.build(E)
	gguf.write(mpath, mkv, mts)
	local e = require("lgml.engine").new({ model = path, backend = "CPU", n_ctx = 96,
		mmproj = mpath, image_min_tokens = 1, image_max_tokens = 6 })
	local function url(seed)
		local t <close> = e.be:tensor("f32", { 1, 3, 12, 18 })
		t:fill("uniform", -1, 1, seed)
		return "data:image/png;base64," .. base64.encode(lgml.image_from(t):png())
	end
	local a, b = url(1), url(2)
	local greedy = { temperature = 0 }
	local function msg(u, text)
		return { { role = "user", content = {
			{ type = "image_url", image_url = { url = u } }, { type = "text", text = text } } } }
	end
	local plain = chat_on(e, { { role = "user", content = "hello world" } }, greedy)
	local r1 = chat_on(e, msg(a, "hello world"), greedy)
	ok(r1.res == true and r1.usage.prompt_tokens == plain.usage.prompt_tokens + 8,
	    ("vision: an image adds 6 positions and 2 tokens (%s, %s)"):format(
	    r1.usage and r1.usage.prompt_tokens, tostring(r1.err)))
	local keys = 0
	for _, v in ipairs(e.tokens) do
		if type(v) == "string" and v:match("^img:%x+:%d+$") then
			keys = keys + 1
		end
	end
	ok(keys == 6, "vision: the token list holds 6 image keys")
	local r2 = chat_on(e, msg(a, "hello world"), greedy)
	ok(r2.usage.cached_tokens == r2.usage.prompt_tokens - 1 and r2.text == r1.text,
	    "vision: the same image reuses the cache and gives the same answer")
	local r3 = chat_on(e, msg(b, "hello world"), greedy)
	local start
	for i, v in ipairs(e.tokens) do
		if type(v) == "string" then
			start = start or i
		end
	end
	ok(r3.res == true and r3.usage.cached_tokens == start - 1,
	    ("vision: another image reuses only the text before it (%d)"):format(r3.usage.cached_tokens))
	local r4 = chat_on(e, msg(a, "hello world"), greedy)
	ok(r4.text == r1.text, "vision: the first image again gives the first answer")
	local two = chat_on(e, { { role = "user", content = {
		{ type = "image_url", image_url = { url = a } }, { type = "text", text = "x" },
		{ type = "image_url", image_url = { url = b } } } } })
	ok(two.res == true and two.usage.prompt_tokens == plain.usage.prompt_tokens - 2 + 17,
	    "vision: two images in one message " .. tostring(two.err))
	local bad = chat_on(e, msg("https://example.com/a.png", "hi"))
	ok(bad.res == nil and bad.status == 400 and bad.err:find("data URL", 1, true),
	    "vision: an http URL gives 400: " .. tostring(bad.err))
	bad = chat_on(e, msg("data:image/png;base64,aGVsbG8=", "hi"))
	ok(bad.res == nil and bad.status == 400, "vision: bytes that are not an image give 400")
	bad = chat_on(e, { { role = "user", content = "<|image|>" } })
	ok(bad.res == nil and bad.status == 400, "vision: an image token without an image gives 400")
	ok(e:models()[1].vision == true and e:props().modalities.vision == true,
	    "vision: models and props report vision")
	local text_only = require("lgml.engine").new({ model = path, backend = "CPU", n_ctx = 96 })
	bad = chat_on(text_only, msg(a, "hi"))
	ok(bad.res == nil and bad.status == 400 and bad.err:find("--mmproj", 1, true),
	    "vision: no mmproj gives 400")
	os.remove(path)
	os.remove(mpath)
end


-- Runs a decision on e in a coroutine. Returns the result, the error
-- message and the status.
local function decide_on(e, req)
	local uv = require("luv")
	local r, done = {}, false
	coroutine.wrap(function()
		r.res, r.err, r.status = e:decide(req, {})
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	return r.res, r.err, r.status
end

-- Decisions with images on the tiny model and mmproj: the images
-- change the readout, text-only requests do not change, and questions
-- on one image share its cache rows.
local function decide_vision_check()
	local tiny = require("mmproj_tiny")
	local base64 = require("lgml.base64")
	local so = require("lgml.systemone")
	local decide = require("lgml.decide")
	local path, mpath = os.tmpname(), os.tmpname()
	local kv, ts = build("dense", { "<|image>", "<|image|>", "<image|>" })
	for _, x in ipairs(kv) do
		if x[1] == "gemma4.context_length" then
			x[3] = 2048
		end
	end
	gguf.write(path, kv, ts)
	local mkv, mts = tiny.build(E)
	gguf.write(mpath, mkv, mts)
	local engine = require("lgml.engine")
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 2048,
		mmproj = mpath, image_min_tokens = 1, image_max_tokens = 6 })
	local plain = engine.new({ model = path, backend = "CPU", n_ctx = 2048 })
	local encodes = 0
	local encode = e.vision.encode
	e.vision.encode = function(...)
		encodes = encodes + 1
		return encode(...)
	end
	local function url(seed)
		local t <close> = e.be:tensor("f32", { 1, 3, 12, 18 })
		t:fill("uniform", -1, 1, seed)
		return "data:image/png;base64," .. base64.encode(lgml.image_from(t):png())
	end
	local a, b = url(1), url(2)
	local Q = {
		c = '{"type": "choice", "instructions": "Which color?", ' ..
		    '"criteria": {"red": null, "green": null, "blue": null}}',
		n = '{"type": "noul", "instructions": "Is it a cat?"}',
	}
	local function req(qs, images, extra)
		local parts = {}
		for i, id in ipairs(qs) do
			parts[i] = ('"%s": %s'):format(id, Q[id])
		end
		local text = '{"state": "a picture", "questions": {' .. table.concat(parts, ", ") ..
		    '}, "lgml": {"raw": true' .. (extra or "") .. "}"
		if images then
			text = text .. ', "attachments": ["' .. table.concat(images, '", "') .. '"]'
		end
		local r, body = assert(so.check(text .. "}"))
		return r, body
	end
	local function ask(eng, qs, images, extra)
		local r, body = req(qs, images, extra)
		local res, err, status = decide_on(eng, r)
		return res and so.reply(r, body, res), res, err, status
	end
	-- The largest difference of two score lists from scores.
	local function diff(x, y)
		local xs, ys, m = {}, {}, 0
		for v in x:gmatch("[^,]+") do
			xs[#xs + 1] = tonumber(v)
		end
		for v in y:gmatch("[^,]+") do
			ys[#ys + 1] = tonumber(v)
		end
		if #xs ~= #ys then
			return math.huge
		end
		for i = 1, #xs do
			m = math.max(m, math.abs(xs[i] - ys[i]))
		end
		return m
	end
	local function scores(res, id)
		local out = {}
		for _, run in ipairs(res.answers[id].lgml.runs) do
			for _, v in ipairs(run.logscores) do
				out[#out + 1] = ("%.17g"):format(v)
			end
		end
		return table.concat(out, ",")
	end

	-- Text only: the same prompt, the same bytes, with or without mmproj.
	local t1, r1 = ask(plain, { "c", "n" })
	local t2, r2 = ask(e, { "c", "n" })
	ok(t1 and t1 == t2 and scores(r1, "c") == scores(r2, "c") and scores(r1, "n") == scores(r2, "n"),
	    "decide vision: text only gives the same bytes with an mmproj")
	ok(r2.timings.images == 0 and encodes == 0, "decide vision: text only encodes no image")

	-- One image: the prompt holds it before the state text.
	e.tokens, e.cache.n_past = {}, 0
	local ta, ra, err, status = ask(e, { "c" }, { a })
	ok(ta and scores(ra, "c") ~= scores(r2, "c"),
	    "decide vision: an image changes the readout" .. (err and ": " .. err .. " " .. status or ""))
	local keys = 0
	for _, v in ipairs(e.tokens) do
		if type(v) == "string" and v:match("^img:%x+:%d+$") then
			keys = keys + 1
		end
	end
	ok(keys == 6 and encodes == 1 and ra.timings.images == 1 and ra.timings.image_encode_ms > 0,
	    ("decide vision: 6 image positions, one encode for two option orders (%d, %d)"):format(
	    keys, encodes))
	local text = decide.chat_prompt(e.family, decide.render("a picture", req({ "c" }).questions[1],
	    { identifiers = e.decide_ids, images = e.image_marker }).prefix)
	ok(text:find("State:\n" .. e.image_marker .. "a picture\n\nQuestion: Which color?", 1, true),
	    "decide vision: the image goes before the state text")
	local tb, rb = ask(e, { "c" }, { b })
	ok(tb and scores(rb, "c") ~= scores(ra, "c") and encodes == 2, "decide vision: another image, another readout")

	-- Another question on the first image reuses the rows up to the state.
	local before = encodes
	local tn, rn = ask(e, { "n" }, { a })
	ok(tn and encodes == before, "decide vision: the image embeddings come from the store")
	local _, rn2 = ask(e, { "c" }, { a })
	local d = diff(scores(rn2, "c"), scores(ra, "c"))
	ok(rn2.timings.cache_n > 6 and d < 1e-3,
	    ("decide vision: a second question reuses the image rows (%d cached), same answer (%.3g)")
	    :format(rn2.timings.cache_n, d))
	local _, both = ask(e, { "c", "n" }, { a })
	d = math.max(diff(scores(both, "c"), scores(ra, "c")), diff(scores(both, "n"), scores(rn, "n")))
	ok(d < 1e-3, ("decide vision: two questions in one request give the same scores (%.3g)"):format(d))

	-- Prior correction runs on a text-only content-free prompt.
	local _, rp = ask(e, { "n" }, { a }, ', "recipe": {"prior_weight": 1}')
	local marker = false
	for prefix in pairs(e.decide_priors.runs) do
		marker = marker or prefix:find(e.image_marker, 1, true) ~= nil
	end
	ok(rp and rp.answers.n.lgml.prior and not marker, "decide vision: the prior prompt has no image")

	-- Two images go in order.
	local _, r12 = ask(e, { "c" }, { a, b })
	local _, r21 = ask(e, { "c" }, { b, a })
	ok(r12 and r21 and scores(r12, "c") ~= scores(r21, "c") and r12.timings.images == 2,
	    "decide vision: two images, in order")

	local _, _, perr, pstatus = ask(plain, { "c" }, { a })
	ok(pstatus == 400 and perr:find("image input is not supported", 1, true),
	    "decide vision: no mmproj gives 400")
	local bad = assert(so.check('{"state": "x", "questions": {"n": ' .. Q.n .. '}}'))
	bad.images = { { bytes = "\137PNG\r\n\26\nnot really", mime = "image/png" } }
	local _, berr, bstatus = decide_on(e, bad)
	ok(bstatus == 400 and berr:find("image 1", 1, true), "decide vision: a bad image gives 400")
	os.remove(path)
	os.remove(mpath)
end

-- The chat engine on the tiny model: a second request with the same
-- first message reuses the cached prefix.
local function engine_check(path)
	local e = require("lgml.engine").new({ model = path, backend = "CPU", n_ctx = 96 })
	require("lgml.server.engine").check(e)
	local function chat(messages)
		return chat_on(e, messages)
	end
	local first = { role = "user", content = "hello world" }
	local a = chat({ first })
	local b = chat({ first, { role = "assistant", content = "hello" },
		{ role = "user", content = "xyz" } })
	ok(a.res == true and b.res == true and a.usage.completion_tokens >= 1,
	    "engine: chat runs " .. tostring(a.err or b.err))
	ok(b.usage.cached_tokens > 0, "engine: the second request reuses the prefix")
end

local be <close> = assert(lgml.backend(os.getenv("LGML_DEVICE") or "CPU"))
local bin = os.getenv("LGML_LLAMA_LOGITS")

for _, kind in ipairs({ "dense", "moe" }) do
	local path = os.tmpname()
	local kv, ts = build(kind)
	gguf.write(path, kv, ts)
	local m = transformer.load(be, path)
	local ref
	if bin and bin ~= "" then
		ref = llama_rows(bin, path)
	else
		ref = REF[kind]
	end
	if not ref or #ref == 0 then
		skip(kind .. ": no reference")
	else
		local rows = our_rows(m, ref, { ubatch = 8 })
		ok(#rows == #ref, kind .. ": row count " .. #rows)
		local err, same = compare(rows, ref)
		io.stderr:write(("%s: %d rows, max logit error %.3g\n"):format(kind, #ref, err))
		ok(err < TOL, ("%s: logits match llama.cpp (max error %.3g)"):format(kind, err))
		ok(same, kind .. ": argmax matches llama.cpp on every row")
		if os.getenv("LGML_REF_DUMP") and ref[1].logits then
			print("REF." .. kind .. " = {")
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
		more(kind, m, ref)
	end
	ring_check(kind, m)
	embd_check(kind, m)
	-- The ring holds 13 rows. A long prompt in passes of 8 and decode
	-- steps across n_past 256 build four graphs.
	local long = {}
	for i = 1, 250 do
		long[i] = (i * 37) % 250 + 1
	end
	for _, fa in ipairs({ false, true }) do
		local tag = kind .. (fa and " flash_attn: " or ": ")
		rc.decode(ok, tag, m, { n_ctx = 320, cache = { ubatch = 8 }, ubatch = 8,
		    prompt = long, n = 12, flash_attn = fa, builds = 4 })
		-- Decode steps from n_past 5 to 45 wrap the ring on one graph.
		rc.decode(ok, tag .. "ring wrap: ", m, { n_ctx = 64, cache = { ubatch = 8 },
		    ubatch = 8, prompt = { 20, 40, 60, 80, 100 }, n = 40, flash_attn = fa, builds = 2 })
	end
	if kind == "dense" then
		engine_check(path)
		vision_check()
		decide_vision_check()
	end
	m:close()
	os.remove(path)
end

-- A real model on LGML_DEVICE (default CPU). Run this file with lua5.4
-- directly for Vulkan: the meson test hides the Vulkan driver.
local path = os.getenv("LGML_GEMMA4")
if path == nil or path == "" then
	skip("LGML_GEMMA4 not set")
else
	local chat = require("lgml.chat")
	local tokenizer = require("lgml.tokenizer")
	local gen = require("lgml.generate")
	local dev = os.getenv("LGML_DEVICE") or "CPU"
	local rb <close> = assert(lgml.backend(dev))
	local tag = "gemma4 " .. dev .. ": "
	local info = assert(lgml.gguf(path))
	ok(info.meta["general.architecture"] == "gemma4", tag .. "architecture")
	local t0 = lgml.clock()
	local m = transformer.load(rb, path)
	local tk = assert(tokenizer.open(path))
	local text = chat.load(info.meta).render({ { role = "user",
		content = "What is the capital of France? Answer in one sentence." } },
		{ add_generation_prompt = true, enable_thinking = false })
	local prompt = tk:encode(text)
	local stop = {}
	for _, id in ipairs(tk:special().eog or {}) do
		stop[id] = true
	end
	local ids, t = gen.greedy(m, m:cache(512), prompt, 32, { stop = stop })
	local out = tk:decode(ids)
	io.stderr:write(("%s%q, %.1f tok/s decode\n"):format(tag, out,
	    t.n_decode / math.max(t.decode, 1e-9)))
	ok(out:find("Paris") ~= nil, tag .. "greedy answer names Paris")
	local fa = gen.greedy(m, m:cache(512), prompt, 4, { flash_attn = true, stop = stop })
	ok(table.concat(fa, ",") == table.concat(ids, ",", 1, #fa),
	    tag .. "flash_attn agrees on the first tokens")
	io.stderr:write(("%stotal %.1f s\n"):format(tag, lgml.clock() - t0))
	m:close()
end

print("1.." .. n)
os.exit(failed and 1 or 0)
