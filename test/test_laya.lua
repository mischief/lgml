-- Tests for lgml.modernbert, lgml.laya and lgml.laya_engine. TAP output.
-- A tiny random model and tokenizer built here run on the CPU and
-- match a plain Lua reference of the same math. With LGML_LAYA set to
-- a Laya model directory, it also checks the real model on clear cases
-- (backend LGML_LAYA_BACKEND, default CPU).
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local json = require("lgml.json")
local bpe_vocab = require("bpe_vocab")
local so = require("lgml.systemone")
local modernbert = require("lgml.modernbert")
local laya = require("lgml.laya")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function skip(name, why)
	n = n + 1
	print("ok " .. n .. " - " .. name .. " # SKIP " .. why)
end

local function write(path, data)
	local f = assert(io.open(path, "wb"))
	f:write(data)
	f:close()
end

-- tiny model

local D, H, NL, FF, WINDOW, ACT = 32, 2, 4, 24, 4, 8
local TYPES = { "full_attention", "sliding_attention", "sliding_attention", "full_attention" }

math.randomseed(20260926)

local function randn(k, scale, base)
	local t = {}
	for i = 1, k do
		t[i] = (base or 0) + (math.random() * 2 - 1) * scale
	end
	return t
end

-- name -> { shape, values }, in row-major order.
local W = {}
local function param(name, shape, scale, base)
	local k = 1
	for _, s in ipairs(shape) do
		k = k * s
	end
	W[name] = { shape = shape, v = randn(k, scale, base) }
end

local tj, ids = bpe_vocab.hf_json()
local NV = ids["[MASK]"] + 1

param("encoder.embeddings.tok_embeddings.weight", { NV, D }, 1.0)
param("encoder.embeddings.norm.weight", { D }, 0.1, 1)
for i = 0, NL - 1 do
	local p = "encoder.layers." .. i .. "."
	if i > 0 then
		param(p .. "attn_norm.weight", { D }, 0.1, 1)
	end
	param(p .. "attn.Wqkv.weight", { 3 * D, D }, 0.3)
	param(p .. "attn.Wo.weight", { D, D }, 0.2)
	param(p .. "mlp_norm.weight", { D }, 0.1, 1)
	param(p .. "mlp.Wi.weight", { 2 * FF, D }, 0.3)
	param(p .. "mlp.Wo.weight", { D, FF }, 0.2)
end
param("encoder.final_norm.weight", { D }, 0.1, 1)
param("type_emb.weight", { 3, D }, 0.5)
for i = 0, 1 do
	local p = "head.layers." .. i .. "."
	param(p .. "norm1.weight", { D }, 0.1, 1)
	param(p .. "norm1.bias", { D }, 0.1)
	param(p .. "norm2.weight", { D }, 0.1, 1)
	param(p .. "norm2.bias", { D }, 0.1)
	param(p .. "self_attn.in_proj_weight", { 3 * D, D }, 0.3)
	param(p .. "self_attn.in_proj_bias", { 3 * D }, 0.1)
	param(p .. "self_attn.out_proj.weight", { D, D }, 0.2)
	param(p .. "self_attn.out_proj.bias", { D }, 0.1)
	param(p .. "linear1.weight", { 4 * D, D }, 0.2)
	param(p .. "linear1.bias", { 4 * D }, 0.1)
	param(p .. "linear2.weight", { D, 4 * D }, 0.1)
	param(p .. "linear2.bias", { D }, 0.1)
end
param("scorer.0.weight", { D }, 0.1, 1)
param("scorer.0.bias", { D }, 0.1)
param("scorer.1.weight", { D, D }, 0.3)
param("scorer.1.bias", { D }, 0.1)
param("scorer.3.weight", { 1, D }, 0.5)
param("scorer.3.bias", { 1 }, 0.1)
param("act_head.0.weight", { ACT, D + 4 }, 0.3)
param("act_head.0.bias", { ACT }, 0.1)
param("act_head.2.weight", { 2, ACT }, 0.3)
param("act_head.2.bias", { 2 }, 0.1)

local function safetensors()
	local names = {}
	for k in pairs(W) do
		names[#names + 1] = k
	end
	table.sort(names)
	local h, data, off = {}, {}, 0
	for _, k in ipairs(names) do
		local b = string.pack("<" .. string.rep("f", #W[k].v), table.unpack(W[k].v))
		h[k] = { dtype = "F32", shape = json.array(W[k].shape), data_offsets = { off, off + #b } }
		data[#data + 1] = b
		off = off + #b
	end
	local s = json.encode(h)
	return string.pack("<I8", #s) .. s .. table.concat(data)
end

local CFG = {
	head_layers = 2, max_len = 64, head_max_len = 24,
	temperature = { 1.5, 1.25, 2.0 },
	temperature_by_options = { ["choice:2"] = 1.9, ["choice:3-5"] = 1.7, ["noul:2"] = 2.0 },
}
local ECFG = {
	model_type = "modernbert", hidden_size = D, num_hidden_layers = NL, num_attention_heads = H,
	intermediate_size = FF, norm_eps = 1e-5, local_attention = WINDOW, max_position_embeddings = 512,
	vocab_size = NV, global_attn_every_n_layers = 3, layer_types = TYPES,
	rope_parameters = { full_attention = { rope_theta = 160000 }, sliding_attention = { rope_theta = 10000 } },
}

local dir = os.tmpname()
os.remove(dir)
assert(os.execute("mkdir -p " .. dir .. "/encoder " .. dir .. "/tokenizer"))
write(dir .. "/model.safetensors", safetensors())
write(dir .. "/rl_agent_config.json", assert(json.encode(CFG)))
write(dir .. "/encoder/config.json", assert(json.encode(ECFG)))
write(dir .. "/tokenizer/tokenizer.json", assert(json.encode(tj)))

-- plain Lua reference

local function w(name)
	return W[name].v
end

-- y = W x (+ b); W is (out, in).
local function linear(x, wv, b, nout, nin)
	local y = {}
	for o = 1, nout do
		local s, base = b and b[o] or 0, (o - 1) * nin
		for i = 1, nin do
			s = s + wv[base + i] * x[i]
		end
		y[o] = s
	end
	return y
end

local function layer_norm(x, g, b)
	local m, v = 0, 0
	for _, a in ipairs(x) do
		m = m + a
	end
	m = m / #x
	for _, a in ipairs(x) do
		v = v + (a - m) ^ 2
	end
	v = v / #x
	local y = {}
	for i, a in ipairs(x) do
		y[i] = (a - m) / math.sqrt(v + 1e-5) * g[i] + (b and b[i] or 0)
	end
	return y
end

local function erf(x)
	-- Series for |x| < 3, else the sign.
	if math.abs(x) >= 3 then
		return x > 0 and 1 - 2.2e-5 or -1 + 2.2e-5
	end
	local s, t, k = x, x, 0
	repeat
		k = k + 1
		t = -t * x * x / k
		local d = t / (2 * k + 1)
		s = s + d
	until math.abs(d) < 1e-15
	return 2 / math.sqrt(math.pi) * s
end

local function gelu(x)
	return 0.5 * x * (1 + erf(x / math.sqrt(2)))
end

local function rope(v, pos, base, hd)
	local half = hd // 2
	local out = table.move(v, 1, #v, 1, {})
	for i = 0, half - 1 do
		local th = pos * base ^ (-2 * i / hd)
		local c, s = math.cos(th), math.sin(th)
		local a, b = v[i + 1], v[i + 1 + half]
		out[i + 1], out[i + 1 + half] = a * c - b * s, b * c + a * s
	end
	return out
end

-- Attention of rows xs with heads of hd; qkv packs [q | k | v].
local function attention(xs, qkv, nh, hd, allowed, rot)
	local n, d = #xs, nh * hd
	local out = {}
	for t = 1, n do
		out[t] = {}
	end
	for h = 0, nh - 1 do
		local q, k, v = {}, {}, {}
		for t = 1, n do
			local function slice(off)
				return table.move(qkv[t], off + h * hd + 1, off + (h + 1) * hd, 1, {})
			end
			q[t], k[t], v[t] = slice(0), slice(d), slice(2 * d)
			if rot then
				q[t], k[t] = rot(q[t], t - 1), rot(k[t], t - 1)
			end
		end
		for t = 1, n do
			local s, max = {}, -math.huge
			for u = 1, n do
				if allowed(t, u) then
					local dot = 0
					for i = 1, hd do
						dot = dot + q[t][i] * k[u][i]
					end
					s[u] = dot / math.sqrt(hd)
					max = math.max(max, s[u])
				end
			end
			local sum = 0
			for u in pairs(s) do
				s[u] = math.exp(s[u] - max)
				sum = sum + s[u]
			end
			for i = 1, hd do
				local a = 0
				for u, p in pairs(s) do
					a = a + p / sum * v[u][i]
				end
				out[t][h * hd + i] = a
			end
		end
	end
	return out
end

local function add(a, b)
	local y = {}
	for i = 1, #a do
		y[i] = a[i] + b[i]
	end
	return y
end

-- The marker logits and the pooled state, as rl_common.DecisionModel.
local function reference(tokens, markers, qtype)
	local x = {}
	for t, id in ipairs(tokens) do
		local e = table.move(w("encoder.embeddings.tok_embeddings.weight"), id * D + 1, (id + 1) * D, 1, {})
		x[t] = layer_norm(e, w("encoder.embeddings.norm.weight"))
	end
	for i = 0, NL - 1 do
		local p = "encoder.layers." .. i .. "."
		local global = TYPES[i + 1] == "full_attention"
		local base = global and 160000 or 10000
		local qkv = {}
		for t = 1, #x do
			local h = i > 0 and layer_norm(x[t], w(p .. "attn_norm.weight")) or x[t]
			qkv[t] = linear(h, w(p .. "attn.Wqkv.weight"), nil, 3 * D, D)
		end
		local o = attention(x, qkv, H, D // H, function(a, b)
			return global or math.abs(a - b) <= WINDOW // 2
		end, function(v, pos)
			return rope(v, pos, base, D // H)
		end)
		for t = 1, #x do
			x[t] = add(x[t], linear(o[t], w(p .. "attn.Wo.weight"), nil, D, D))
			local h = linear(layer_norm(x[t], w(p .. "mlp_norm.weight")), w(p .. "mlp.Wi.weight"), nil, 2 * FF, D)
			local gg = {}
			for j = 1, FF do
				gg[j] = gelu(h[j]) * h[FF + j]
			end
			x[t] = add(x[t], linear(gg, w(p .. "mlp.Wo.weight"), nil, D, FF))
		end
	end
	local te = table.move(w("type_emb.weight"), qtype * D + 1, (qtype + 1) * D, 1, {})
	for t = 1, #x do
		x[t] = add(layer_norm(x[t], w("encoder.final_norm.weight")), te)
	end
	for i = 0, 1 do
		local p = "head.layers." .. i .. "."
		local qkv = {}
		for t = 1, #x do
			qkv[t] = linear(layer_norm(x[t], w(p .. "norm1.weight"), w(p .. "norm1.bias")),
			    w(p .. "self_attn.in_proj_weight"), w(p .. "self_attn.in_proj_bias"), 3 * D, D)
		end
		local o = attention(x, qkv, 1, D, function()
			return true
		end)
		for t = 1, #x do
			x[t] = add(x[t], linear(o[t], w(p .. "self_attn.out_proj.weight"), w(p .. "self_attn.out_proj.bias"), D, D))
			local f = linear(layer_norm(x[t], w(p .. "norm2.weight"), w(p .. "norm2.bias")),
			    w(p .. "linear1.weight"), w(p .. "linear1.bias"), 4 * D, D)
			for j = 1, #f do
				f[j] = math.max(f[j], 0)
			end
			x[t] = add(x[t], linear(f, w(p .. "linear2.weight"), w(p .. "linear2.bias"), D, 4 * D))
		end
	end
	local z = {}
	for j, m in ipairs(markers) do
		local h = layer_norm(x[m + 1], w("scorer.0.weight"), w("scorer.0.bias"))
		h = linear(h, w("scorer.1.weight"), w("scorer.1.bias"), D, D)
		for i = 1, D do
			h[i] = gelu(h[i])
		end
		z[j] = linear(h, w("scorer.3.weight"), w("scorer.3.bias"), 1, D)[1]
	end
	return z, x[1]
end

local function close(a, b, tol)
	if #a ~= #b then
		return false
	end
	for i = 1, #a do
		if math.abs(a[i] - b[i]) > tol then
			io.stderr:write(("index %d: got %.6f, want %.6f\n"):format(i, a[i], b[i]))
			return false
		end
	end
	return true
end

-- tests

local be = assert(lgml.backend("CPU", { threads = 2 }))

local function question(body)
	local req = assert(so.check(body))
	return req, req.questions
end

local function test_helpers()
	ok(laya.bucket("choice", 2) == "choice:2" and laya.bucket("choice", 5) == "choice:3-5" and
	    laya.bucket("choice", 6) == "choice:6-10" and laya.bucket("choice", 11) == "choice:11+",
	    "temperature buckets")
	ok(laya.temperature(CFG, "choice", 4) == 1.7 and laya.temperature(CFG, "score", 3) == 1.25 and
	    laya.temperature(CFG, "noul", 2) == 2.0, "temperature by bucket, else by type")
	local p = laya.softmax({ 1, 2, 3 }, 2)
	ok(math.abs(p[1] + p[2] + p[3] - 1) < 1e-12 and p[3] > p[2], "softmax with temperature")
	ok(math.abs(laya.confidence({ 0.5, 0.5 })) < 1e-12 and laya.confidence({ 1, 0 }) > 0.999,
	    "confidence is 1 minus normalized entropy")
	local _, qs = question([[{"state": "s", "questions": {
		"a": {"type": "choice", "criteria": {"x": "the x", "y": null}},
		"b": {"type": "score", "criteria": ["low", "high"]},
		"c": {"type": "noul", "criteria": {"true": "it holds"}}}}]])
	local o = { laya.options(qs[1]), laya.options(qs[2]), laya.options(qs[3]) }
	ok(o[1][1] == "x: the x" and o[1][2] == "y", "choice options")
	ok(o[2][1] == "level 0: low" and o[2][2] == "level 1: high", "score levels")
	ok(o[3][1] == "false: no, the statement does not hold" and o[3][2] == "true: it holds",
	    "noul options: false first")
end

local function test_modernbert()
	local hp = modernbert.config(ECFG)
	ok(hp.head_dim == D // H and hp.global[1] and not hp.global[2] and hp.global[4] and
	    hp.theta_global == 160000 and hp.theta_local == 10000, "config")
	local old = { model_type = "modernbert", hidden_size = 8, num_hidden_layers = 4, num_attention_heads = 2,
		intermediate_size = 8, global_attn_every_n_layers = 3, global_rope_theta = 160000,
		local_rope_theta = 10000, vocab_size = 10 }
	local hp2 = modernbert.config(old)
	ok(hp2.global[1] and not hp2.global[3] and hp2.global[4], "config without layer_types")
end

local function test_forward(m)
	local req, qs = question([[{"state": "the troll blocks the door and the player has 3 hp",
		"questions": {"act": {"type": "choice", "instructions": "Pick an action.",
		"criteria": {"attack": "hit it", "run": "leave", "talk": null}}}}]])
	local tokens, markers, opts = m:sequence(req.state, qs[1])
	ok(#markers == 3 and #opts == 3, "one marker per option")
	local good = tokens[1] == ids["[CLS]"] and tokens[#tokens] == ids["[SEP]"]
	for _, p in ipairs(markers) do
		good = good and tokens[p + 1] == ids["[MASK]"]
	end
	ok(good, "markers are [MASK] tokens; [CLS] first, [SEP] last")
	ok(#tokens <= CFG.max_len, "the state is cut to max_len")
	local res = assert(m:decide(req.state, qs))
	local zr, pooled = reference(tokens, markers, laya.QTYPES.choice)
	ok(close(res[1].logits, zr, 2e-3), "logits match the Lua reference")
	local p = laya.softmax(zr, 1.7)
	ok(close(res[1].probabilities, p, 1e-3) and res[1].temperature == 1.7, "calibrated probabilities")
	local act = m:act_probability(pooled, zr)
	ok(math.abs(res[1].act_probability - act) < 1e-3, "act head")
	local again = assert(m:decide(req.state, qs))
	ok(close(again[1].logits, res[1].logits, 0), "stable outputs")
	-- Short sequences skip the local mask; long ones use it.
	local short = { ids["[CLS]"], ids.x, ids["[MASK]"], ids["[SEP]"] }
	local zs = m.enc:encode(short)
	ok(#zs == #short * D, "encoder output shape")
end

local function test_budget(m)
	local many = {}
	for i = 1, 9 do
		many[#many + 1] = ('"o%d": "a long description of option number %d"'):format(i, i)
	end
	local req, qs = question('{"state": "s", "questions": {"q": {"type": "choice", "criteria": {' ..
	    table.concat(many, ", ") .. "}}}}")
	local tokens, markers = m:sequence(req.state, qs[1])
	ok(#markers == 9, "options shrink to share head_max_len")
	local steps = {}
	for i = 2, #markers do
		steps[#steps + 1] = markers[i] - markers[i - 1]
	end
	ok(steps[1] == math.max(4, (CFG.head_max_len - 16) // 9), "each option keeps the same share")
	ok(tokens[markers[1]] == ids["[SEP]"], "options follow the instructions and [SEP]")
	local big = {}
	for i = 1, 40 do
		big[#big + 1] = ('"option%d": null'):format(i)
	end
	req, qs = question('{"state": "s", "questions": {"q": {"type": "choice", "criteria": {' ..
	    table.concat(big, ", ") .. "}}}}")
	local r, err = m:decide(req.state, qs)
	ok(r == nil and err:find("do not fit"), "too many options give an error")
end

local function test_engine()
	local eng = require("lgml.laya_engine").new({ model = dir, backend = "CPU", threads = 2 })
	local mm = eng:models()
	ok(#mm == 1 and mm[1].capabilities[1] == "decide" and #mm[1].capabilities == 1 and
	    mm[1].context_length == CFG.max_len, "models: decide only")
	local text = [[{"state": {"hp": 3, "enemy": "troll"}, "questions": {
		"flee": {"type": "noul", "instructions": "Should the player flee?"},
		"act": {"type": "choice", "instructions": "Pick an action.",
			"criteria": {"attack": "Hit the enemy.", "run": "Leave the room."}},
		"mood": {"type": "score", "criteria": ["calm", "tense", "panic"]}},
		"lgml": {"raw": true}}]]
	local req, body = assert(so.check(text))
	ok(req.state_json == '{"hp": 3, "enemy": "troll"}', "one-line JSON state")
	local r = assert(eng:decide(req))
	ok(r.answers.flee.yes and #r.answers.act.probabilities == 2 and #r.answers.mood.probabilities == 3,
	    "answers by type")
	ok(r.usage.input_tokens > 0 and r.usage.output_tokens == 0, "usage")
	local out = assert(json.decode(so.reply(req, body, r)))
	local a = out.answers
	ok(a.flee.type == "noul" and type(a.flee.noul) == "number" and a.act.choice and
	    a.mood.legend["2"] == "panic" and a.flee.lgml.temperature == 2.0, "wire reply")
	local _, e1, s1 = eng:decide(assert(so.check('{"state": "s", "questions": {"q": {"type": "noul"}}, ' ..
	    '"lgml": {"recipe": {"temperature": 2}}}')))
	ok(s1 == 400 and e1:find("recipe"), "recipe gives 400")
	local ir = assert(so.check('{"state": "s", "questions": {"q": {"type": "noul"}}}'))
	ir.images = { { bytes = "x", mime = "image/png" } }
	local _, _, s2 = eng:decide(ir)
	ok(s2 == 400, "images give 400")
	eng:close()
end

-- real model

local function test_model(path)
	local backend = os.getenv("LGML_LAYA_BACKEND") or "CPU"
	local mbe = assert(lgml.backend(backend))
	local m = laya.load(mbe, path)
	local req, qs = question([[{"state": {"from": "user@acme.com",
		"subject": "Duplicate charge on invoice #4411",
		"body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
		"questions": {
		"department": {"type": "choice", "instructions": "Which department should handle this request?",
			"criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
			"sales": "pricing, new contracts", "other": "everything else"}},
		"refund": {"type": "noul", "instructions": "Does the user explicitly request a refund?"},
		"green": {"type": "noul", "instructions": "Is the email about gardening?"}}}]])
	local tk = m.tk
	ok(table.concat(tk:encode("Hello world"), ",") == "12092,1533", "model tokenizer: Hello world")
	local res = assert(m:decide(req.state_json, qs))
	local p = res[1].probabilities
	ok(p[1] > 0.8, ("billing email picks billing (p %.3f)"):format(p[1]))
	ok(res[2].probabilities[2] > 0.7, ("refund request: noul high (p %.3f)"):format(res[2].probabilities[2]))
	ok(res[3].probabilities[2] < 0.2, ("gardening: noul low (p %.3f)"):format(res[3].probabilities[2]))
	req, qs = question([[{"state": "The capital of France is Paris.", "questions": {
		"cap": {"type": "choice", "instructions": "Which city is the capital of France?",
			"criteria": {"Berlin": null, "Paris": null, "Madrid": null, "Rome": null}}}}]])
	res = assert(m:decide(req.state, qs))
	local best = 1
	for i, v in ipairs(res[1].probabilities) do
		if v > res[1].probabilities[best] then
			best = i
		end
	end
	ok(best == 2, "capital of France: Paris")
	m:close()
	mbe:close()
end

test_helpers()
test_modernbert()
local m = laya.load(be, dir)
test_forward(m)
test_budget(m)
m:close()
test_engine()
local model = os.getenv("LGML_LAYA") or ""
if model == "" then
	skip("Laya model", "LGML_LAYA not set")
else
	test_model(model)
end
be:close()
os.execute("rm -rf " .. dir)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
