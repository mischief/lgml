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
-- Reference rows: input id, argmax, logits at IDX. From llama.cpp (ggml 1.0) on
-- the CPU with f16 KV cache.
local REF = {}
REF.dense = {
	{ id = 20, best = 66, v = { -0.259937, -0.121262, -0.106518, 0.0242489, -0.10353, 0.074525, -0.716905, -0.262208, 0.295995 } },
	{ id = 40, best = 275, v = { -0.52323, -0.271088, -0.451832, 0.285434, -0.0466356, 0.0540984, -1.07772, -0.289144, 0.193511 } },
	{ id = 60, best = 96, v = { 0.00196849, -0.385248, 0.187068, -0.0705279, 0.205605, -0.410038, -0.00597528, -0.308256, -0.189243 } },
	{ id = 80, best = 69, v = { 0.106189, -0.0139341, -0.251335, -0.024651, 0.021689, -0.472358, -0.26812, 0.313025, 0.540881 } },
	{ id = 100, best = 125, v = { 0.0758534, -0.403845, -0.0895399, 0.0133566, -0.321362, -0.134079, -0.288962, -0.446257, 0.437008 } },
	{ id = 120, best = 217, v = { -0.284925, -0.0295329, 0.526588, -0.0906476, -0.430494, -0.0138343, 0.022541, -0.104311, 0.159165 } },
	{ id = 140, best = 232, v = { 0.187596, -0.497156, 0.357433, 0.108167, -0.0873198, 0.446413, 0.346618, 0.0636307, -0.29385 } },
	{ id = 160, best = 158, v = { -0.291459, -0.30915, -0.289731, -0.142971, 0.176118, 0.200387, -0.485413, 0.063019, 0.410211 } },
	{ id = 180, best = 231, v = { -0.364391, -0.00340294, -0.309568, 0.349665, 0.440251, 0.643956, 0.101593, -0.504932, -0.344439 } },
	{ id = 200, best = 118, v = { -0.141733, -0.0190182, -0.28488, 0.0728754, 0.278602, 0.450003, -0.344327, -0.243673, -0.108062 } },
	{ id = 220, best = 70, v = { 0.453278, 0.0356921, -0.115578, -0.00909758, -0.384781, 0.12431, 0.339043, -0.421181, -0.0383241 } },
	{ id = 240, best = 222, v = { -0.327875, 0.0270534, -0.240033, -0.168587, 0.345, 0.0304061, -0.26897, -0.510504, 0.0877995 } },
	{ id = 260, best = 99, v = { -0.0889895, -0.252931, 0.256389, 0.0331263, -0.0147032, 0.254446, -0.00936673, 0.409043, 0.274557 } },
	{ id = 280, best = 280, v = { 0.16444, -0.224947, 0.0161463, -0.0852799, 0.281723, 0.295387, 0.278239, 0.0984466, 0.355781 } },
	{ id = 21, best = 17, v = { -0.0197735, -0.108482, -0.0707749, -0.0450774, -0.350878, 0.379917, -0.206794, -0.393346, 0.221526 } },
	{ id = 41, best = 111, v = { 0.163975, -0.146536, -0.00187892, -0.461291, 0.106836, -0.0772738, 0.0688432, -0.0629921, 0.119256 } },
	{ id = 61, best = 201, v = { 0.529629, -0.139622, -0.10322, 0.0978006, 0.517611, -0.197709, -0.035363, 0.0691804, 0.0155704 } },
	{ id = 81, best = 181, v = { -0.0901326, 0.184663, -0.249307, 0.287809, -0.301566, 0.231783, -0.0840825, 0.298581, 0.292409 } },
	{ id = 101, best = 77, v = { 0.355531, 0.212691, -0.339058, 0.000833511, -0.431251, -0.257179, -0.127281, 0.402784, 0.0447788 } },
	{ id = 121, best = 272, v = { 0.0477792, 0.00755435, 0.226307, -0.0764042, 0.215752, -0.29038, 0.109516, 0.199804, -0.255823 } },
	{ id = 270, best = 142, v = { 0.144092, -0.484496, 0.363654, 0.138508, 0.82354, -0.0062539, 0.433921, 0.0142421, -0.17468 } },
	{ id = 30, best = 273, v = { 0.0434089, -0.0211502, -0.0981256, -0.19577, -0.376875, -0.101481, 0.148847, -0.364497, -0.143138 } },
	{ id = 50, best = 137, v = { -0.0723666, -0.241965, 0.365688, -0.550397, 0.155328, -0.727179, 0.500442, -0.250253, -0.278107 } },
	{ id = 70, best = 70, v = { 0.0249175, -0.31813, 0.0539749, -0.183273, 0.199752, -0.040774, 0.0585718, 0.182646, -0.0208357 } },
	{ id = 90, best = 95, v = { -0.154699, -1.02662, 0.491132, -0.302565, 0.118543, -0.269762, 0.646277, 0.27881, -0.371724 } },
	{ id = 95, best = 147, v = { -0.0452597, -0.538507, 0.267223, -0.360023, 0.291279, -0.647422, 0.19229, -0.144636, -0.166163 } },
	{ id = 147, best = 233, v = { 0.219804, 0.0598699, -0.0140303, 0.223932, -0.158292, -0.0653279, 0.386244, 0.313278, -0.0557155 } },
	{ id = 233, best = 2, v = { -0.104425, -0.037801, -0.497392, 0.497574, -0.36944, -0.377, -0.384177, -0.0166035, -0.140949 } },
	{ id = 2, best = 95, v = { -0.083207, -0.614453, -0.207117, 0.527907, 0.218659, 0.121815, 0.325777, -0.0828093, -0.00204062 } },
	{ id = 95, best = 272, v = { -0.0917644, -0.230898, 0.0663843, -0.168948, 0.629768, -0.135868, 0.572344, 0.326714, -0.744808 } },
	{ id = 272, best = 272, v = { 0.239104, -0.636676, -0.0432361, -0.0526013, 0.0789084, -0.192653, 0.60187, -0.106161, -0.250626 } },
}
REF.moe = {
	{ id = 20, best = 28, v = { 0.404688, -0.0326983, -0.0713768, -0.164673, -0.307094, -0.246597, -0.114804, 0.0739948, -0.16143 } },
	{ id = 40, best = 66, v = { -0.399641, -0.118008, 0.322459, 0.377069, -0.304364, -0.133199, -0.437394, 0.288638, 0.173684 } },
	{ id = 60, best = 66, v = { -0.836578, -0.188749, -0.268896, 0.236231, -0.0594705, 0.0438759, -0.47537, 0.184842, 0.467912 } },
	{ id = 80, best = 104, v = { -0.513566, 0.119487, -0.422482, -0.322171, 0.0644554, 0.139267, -0.421915, 0.460529, -0.0370848 } },
	{ id = 100, best = 147, v = { -0.30784, 0.0504773, -0.23977, 0.298661, -0.1399, -0.242881, 0.0745847, 0.341099, -0.0300271 } },
	{ id = 120, best = 179, v = { -0.839739, -0.61659, 0.149581, 0.288312, -0.432851, -0.0490543, -0.200265, 0.25337, 0.176016 } },
	{ id = 140, best = 260, v = { -0.576623, 0.198379, 0.0487028, -0.184579, 0.119656, 0.511412, -0.667689, 0.0115685, 0.50108 } },
	{ id = 160, best = 157, v = { -0.147214, 0.243505, -0.272286, 0.141049, -0.316163, 0.153481, -0.256435, 0.156972, 0.298389 } },
	{ id = 180, best = 82, v = { 0.18584, 0.42282, 0.000544742, 0.22023, -0.176953, 0.229602, 0.179269, -0.25981, 0.090247 } },
	{ id = 200, best = 226, v = { -0.1751, 0.537353, -0.0874965, 0.178727, -0.117743, -0.100242, -0.144607, 0.00165899, 0.306517 } },
	{ id = 220, best = 95, v = { 0.229513, 0.552937, 0.195459, 0.202453, -0.822063, -0.183922, 0.162887, -0.14304, -0.171068 } },
	{ id = 240, best = 212, v = { 0.118428, 0.561983, -0.048675, -0.233672, -0.151137, -0.114242, 0.015955, -0.492688, -0.179491 } },
	{ id = 260, best = 188, v = { -0.719592, 0.143297, -0.105382, 0.0719296, -0.0208779, -0.421252, -0.138749, 0.0459391, 0.130317 } },
	{ id = 280, best = 107, v = { 0.49151, 0.309136, -0.283768, -0.0693015, 0.347378, 0.178894, -0.314411, -0.106707, 0.212636 } },
	{ id = 21, best = 273, v = { -0.0929038, 0.270949, -0.11772, -0.0708248, 0.17366, -0.232057, 0.351895, 0.14273, -0.286154 } },
	{ id = 41, best = 102, v = { -0.464339, -0.421882, -0.568913, 0.0256581, -0.161913, -0.161956, -0.220295, 0.57298, 0.0978171 } },
	{ id = 61, best = 169, v = { -0.340635, -0.0353052, -0.148723, 0.0532044, 0.543018, -0.185101, -0.55905, 0.00571889, -0.0633894 } },
	{ id = 81, best = 255, v = { 0.093657, 0.183738, -0.116089, -0.0823212, 0.117636, 0.219704, -0.111236, -0.0712172, 0.046382 } },
	{ id = 101, best = 187, v = { -0.466983, 0.150157, 0.0332511, 0.102473, 0.174114, -0.0452999, 0.00314784, 0.364728, 0.353097 } },
	{ id = 121, best = 81, v = { 0.0738571, -0.373933, -0.153695, -0.143687, 0.345029, -0.257156, -0.275789, 0.0873169, -0.091082 } },
	{ id = 270, best = 265, v = { 0.159549, 0.552809, -0.142688, -0.306477, 0.38574, 0.000528529, -0.522581, 0.481257, 0.456469 } },
	{ id = 30, best = 182, v = { -0.122506, 0.619247, 0.0894106, -0.200807, 0.507675, -0.208685, -0.34242, 0.401875, -0.348933 } },
	{ id = 50, best = 232, v = { -0.276306, -0.354788, -0.108277, 0.0142966, 0.036445, 0.337805, 0.110269, 0.607237, 0.336653 } },
	{ id = 70, best = 31, v = { -0.278965, -0.594278, 0.101005, 0.187311, -0.135727, 0.115226, 0.085424, -0.134305, -0.0846261 } },
	{ id = 90, best = 176, v = { 0.160063, -0.0489704, -0.428085, -0.333486, 0.0510247, -0.0684967, 0.191186, -0.0377871, -0.20204 } },
	{ id = 176, best = 201, v = { 0.11366, 0.425635, -0.262663, 0.0111647, -0.381179, 0.176349, 0.400527, -0.118491, -0.260922 } },
	{ id = 201, best = 92, v = { 0.151615, 0.496894, -0.00256544, 0.195143, 0.350764, 0.33865, 0.42179, -0.156255, -0.615094 } },
	{ id = 92, best = 43, v = { 0.0848556, -0.703265, -0.612792, -0.308612, -0.0561337, -0.163879, -0.0328775, 0.137749, -0.345455 } },
	{ id = 43, best = 157, v = { -0.265696, 0.0166092, -0.0970049, -0.0177445, -0.181502, 0.193746, 0.415005, 0.0183861, -0.314486 } },
	{ id = 157, best = 40, v = { -0.547575, -0.562009, -0.182614, -0.207427, -0.22104, 0.0391633, -0.435741, 0.591657, 0.233985 } },
	{ id = 40, best = 190, v = { -0.429761, -0.202015, 0.660124, 0.579317, -0.33085, -0.271394, -0.406582, 0.303862, -0.611058 } },
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

-- Two sequences of one KV pool in batched passes of at most 8 tokens:
-- the global layers share the pool, and each sequence keeps its own
-- ring. The logits of each sequence equal the sequence alone on its
-- own cache, prompt and decode.
local function pool_check(kind, m)
	local prompts = { {}, {} }
	for i = 1, 40 do
		prompts[1][i] = (i * 37) % 250 + 1
	end
	for i = 1, 27 do
		prompts[2][i] = (i * 53) % 250 + 1
	end
	local steps = 8
	local want = {}
	for s, p in ipairs(prompts) do
		local c = m:cache(128, { ubatch = 8 })
		local rows = eval(m, c, p, nil, 8)
		want[s] = { rows[#rows] }
		for _ = 2, steps do
			rows = eval(m, c, { argmax(want[s][#want[s]]) })
			want[s][#want[s] + 1] = rows[1]
		end
	end
	local pool = m:pool(256, { initial = 256 })
	local seqs = { m:seq(pool, { n_ctx = 128, ubatch = 8 }), m:seq(pool, { n_ctx = 128, ubatch = 8 }) }
	local got, at = { {}, {} }, { 1, 1 }
	while at[1] <= #prompts[1] or at[2] <= #prompts[2] do
		local cs, ids, who = {}, {}, {}
		for s, p in ipairs(prompts) do
			if at[s] <= #p then
				local j = math.min(#p, at[s] + 7)
				cs[#cs + 1], ids[#ids + 1], who[#who + 1] = seqs[s], table.move(p, at[s], j, 1, {}), s
				at[s] = j + 1
			end
		end
		local rows = m:run_batch(cs, ids)
		for i, s in ipairs(who) do
			if at[s] > #prompts[s] then
				got[s][1] = rows[i]:get()
			end
		end
	end
	for _ = 2, steps do
		local rows = m:decode_batch(seqs, { argmax(got[1][#got[1]]), argmax(got[2][#got[2]]) })
		got[1][#got[1] + 1], got[2][#got[2] + 1] = rows[1]:get(), rows[2]:get()
	end
	local err = 0
	for s = 1, 2 do
		for k = 1, steps do
			for j = 1, #want[s][k] do
				err = math.max(err, math.abs(want[s][k][j] - got[s][k][j]))
			end
		end
	end
	ok(err == 0, ("%s pool: batched sequences with rings match each sequence alone (max error %.3g)")
	    :format(kind, err))
	ok(seqs[1].k[1] ~= nil and pool.k[1] == nil and seqs[1].ring == 13,
	    kind .. " pool: sliding layers keep a ring per sequence")
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
-- Runs fn with log lines at level into a list. Returns the lines as
-- one text.
local function logged(level, fn)
	local log = require("lgml.log")
	local lines = {}
	log.set_output(function(l) lines[#lines + 1] = l end)
	log.set_level(level)
	local ok_, err = pcall(fn)
	log.set_level("info")
	log.set_output(nil)
	assert(ok_, err)
	return table.concat(lines, "\n")
end

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
	local e = require("lgml.engine").new({ model = path, backend = "CPU", n_ctx = 96, slots = 1,
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
	local all = logged("trace", function()
		chat_on(e, msg(a, "hello world"), greedy)
	end)
	ok(all:find('msg="chat prompt" model=', 1, true) and all:find("<image 1 img:%x+ %d+B 6 tokens>") and
	    not all:find("iVBOR", 1, true), "log: the chat prompt shows the image as a marker")
	ok(all:find('msg=sampling preset=qwen3', 1, true) and all:find(' settings="{', 1, true) and all:find("msg=prefill tokens=%d+ cached=%d+"),
	    "log: debug gives the sampling settings and the prefill")
	ok(all:find("msg=token n=1 id=%d+") and all:find("msg=\"graph %a+\" n=%d+"),
	    "log: trace gives each token and graph pass")
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
	local all
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

	-- Renderings after the first run together on copies of the state;
	-- they give the same scores as one at a time.
	local forks = 0
	local fork_rows = plain.fork_rows
	plain.fork_rows = function(...)
		forks = forks + 1
		return fork_rows(...)
	end
	local alone = engine.new({ model = path, backend = "CPU", n_ctx = 2048 })
	alone.fork_rows = false
	local long = ('{"state": "%s", "questions": {"c": %s, "n": %s}, "lgml": {"raw": true}}'):format(
	    ("a long picture of many things "):rep(20), Q.c, Q.n)
	local f1 = decide_on(plain, assert(so.check(long)))
	local s1 = decide_on(alone, assert(so.check(long)))
	local df = math.max(diff(scores(f1, "c"), scores(s1, "c")), diff(scores(f1, "n"), scores(s1, "n")))
	ok(forks == 1 and df < 1e-3, ("decide: forked renderings give the same scores (%d forks, %.3g)"):format(forks, df))

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

	all = logged("debug", function()
		local t, _, e2, s2 = ask(e, { "c" }, { a })
		assert(t, tostring(e2) .. " " .. tostring(s2))
	end)
	ok(all:find('msg="decide prompt" model=', 1, true) and all:find("question=c type=choice order=2 seed=1001", 1, true)
	    and all:find("<image 1 img:%x+ %d+B 6 tokens><image|>a picture"), "log: each option order logs its prompt")
	ok(all:find("msg=candidates question=c", 1, true) and all:find("msg=logscores question=c", 1, true) and
	    all:find("msg=recipe recipe=", 1, true) and all:find("msg=answer question=c", 1, true),
	    "log: candidates, log-scores, recipe and answer")
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
-- A chat on a model with rings whose slot went to other requests
-- starts again from cached pool blocks and a checkpoint of the state
-- index, which holds the rings, and answers as a fresh engine does.
local function ring_index_check()
	local path = os.tmpname()
	local kv, ts = build("dense")
	for _, x in ipairs(kv) do
		if x[1] == "gemma4.context_length" then
			x[3] = 2048
		end
	end
	gguf.write(path, kv, ts)
	local engine = require("lgml.engine")
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 2, flash_attn = false })
	local fresh = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 1, cache_ram = 0,
		ctx_checkpoints = 0, flash_attn = false })
	local g = { temperature = 0 }
	local words = {}
	for i = 1, 120 do
		words[i] = ("w%d"):format(i * 7 % 97)
	end
	local first = { role = "user", content = table.concat(words, " ") }
	chat_on(e, { first }, g)
	chat_on(e, { { role = "user", content = "one other chat" } }, g)
	chat_on(e, { { role = "user", content = "and a third one" } }, g)
	local turn2 = { first, { role = "assistant", content = "not what it said" },
		{ role = "user", content = "more words here" } }
	local hits = e.pool.alloc.stats.hits
	local r = chat_on(e, turn2, g)
	local want = chat_on(fresh, turn2, g)
	ok(e.fixed and e.slots[1].cache.ring < 2048, "ring index: the model keeps rings per slot")
	ok(r.usage.cached_tokens > 100 and e.pool.alloc.stats.hits > hits and #e.pc.states > 0,
	    ("ring index: turn 2 starts from cached blocks and a checkpoint (cached %d of %d)")
	    :format(r.usage.cached_tokens, r.usage.prompt_tokens))
	ok(r.text == want.text, "ring index: turn 2 answers as a fresh engine")
	os.remove(path)
end

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

-- Expert cache: with every routed expert on the CPU and some of them in
-- device slots that change between passes, decode steps give the
-- logits of one backend. The KV cache is f32, so that f16 rounding
-- does not grow the small differences of the sum order.
local function xcache_check(tag, path, ids, tol)
	local placement = require("lgml.placement")
	local replay = require("lgml.replay")
	local xcache = require("lgml.xcache")
	local transformer = require("lgml.transformer")
	local dev = os.getenv("LGML_DEVICE") or "CPU"
	local function rows(m)
		local c = m:cache(#ids + 8, { type = "f32" })
		local out = {}
		for i, id in ipairs(ids) do
			out[i] = replay.run(m, c, { id }, { logits = "last" }):get()
		end
		return out
	end
	local one <close> = assert(lgml.backend(dev))
	local want = rows(transformer.load(one, path))
	local sb <close> = assert(lgml.backend(dev, { cpu = true }))
	local rules = placement.rules({ cpu_moe = true })
	local ne
	for _, frac in ipairs({ 0.1, 0.5, 0.9 }) do
		local m = transformer.load(sb, path, { placement = rules })
		ne = m.hp.n_expert
		local n = math.max(1, math.min(ne - 1, math.floor(ne * frac)))
		local xc = xcache.new(m, { slots = n, swap = 1 << 30, halflife = 3, margin = 0 })
		local got = rows(m)
		local d = #got == #want and 0 or math.huge
		for i = 1, #want do
			for j = 1, #want[i] do
				d = math.max(d, math.abs(want[i][j] - (got[i][j] or math.huge)))
			end
		end
		local h, mi = xc:stats()
		ok(d < tol and h > 0 and mi > 0 and xc.moved > xc.bytes // 2,
		    ("%s: expert cache of %d slots: logits equal one backend (max error %.3g, %d of %d picks held)")
		    :format(tag, n, d, h, h + mi))
		m:close()
	end
end
-- Weight placement: experts or the first layers on a second CPU backend
-- give the same logits as one backend, and replay keeps its graphs.
local function offload_check(kind, path, m, ref, long)
	local placement = require("lgml.placement")
	local sb <close> = assert(lgml.backend(os.getenv("LGML_DEVICE") or "CPU", { cpu = true }))
	local cb = sb:cpu()
	local want = our_rows(m, ref, { ubatch = 8 })
	for _, c in ipairs({ { "cpu_moe", { cpu_moe = true } }, { "n_gpu_layers 2", { n_gpu_layers = 2 } } }) do
		local tag = ("%s %s: "):format(kind, c[1])
		local s = transformer.load(sb, path, { placement = placement.rules(c[2]) })
		local L1, L4 = s.layers[1], s.layers[4]
		if c[1] == "cpu_moe" then
			ok(L1.down_exps:backend() == cb and L1.gate:backend() == sb and
			    L1.down_exps_scale:backend() == sb, tag .. "routed experts on the CPU")
		else
			ok(L1.cpu and not L4.cpu and s.ple_embd:backend() == cb and s.output:backend() == cb,
			    tag .. "the first layers and the embeddings on the CPU")
		end
		local got = our_rows(s, ref, { ubatch = 8 })
		local d = 0
		for i = 1, #want do
			for j = 1, #want[i] do
				d = math.max(d, math.abs(want[i][j] - ((got[i] or {})[j] or math.huge)))
			end
		end
		ok(#got == #want and d < (GPU and TOL or 1e-4),
		    ("%slogits equal one backend (max error %.3g)"):format(tag, d))
		rc.decode(ok, tag, s, { n_ctx = 320, cache = { ubatch = 8 }, ubatch = 8,
		    prompt = long, n = 12, flash_attn = true, builds = 4 })
		s:close()
	end
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
	pool_check(kind, m)
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
	if kind == "moe" and ref and #ref > 0 then
		offload_check(kind, path, m, ref, long)
	end
	if kind == "moe" then
		xcache_check(kind, path, table.move(long, 1, 24, 1, {}), GPU and 2e-3 or 1e-4)
	end
	if kind == "dense" then
		engine_check(path)
		ring_index_check()
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
