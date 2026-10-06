-- Tests for lgml.decide_head: the Kev and Clef prompts, and both heads
-- on tiny models with random head weights against a plain Lua
-- reference of the model code. TAP output.

local uv = require("luv")
local json = require("lgml.json")
local so = require("lgml.systemone")
local decide = require("lgml.decide")
local head = require("lgml.decide_head")
local engine = require("lgml.engine")
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local tiny = require("tiny")
local gguf = require("gguf")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function check(text)
	return assert(so.check(text))
end

-- plain Lua math

local function lin(w, b, x, nout)
	local nin = #x
	local y = {}
	for o = 1, nout do
		local s = b and b[o] or 0.0
		local base = (o - 1) * nin
		for i = 1, nin do
			s = s + w[base + i] * x[i]
		end
		y[o] = s
	end
	return y
end

local function add(a, b)
	local y = {}
	for i = 1, #a do
		y[i] = a[i] + b[i]
	end
	return y
end

local function dot(a, b)
	local s = 0.0
	for i = 1, #a do
		s = s + a[i] * b[i]
	end
	return s
end

local function softmax(s, t)
	local m, z, p = -math.huge, 0.0, {}
	for _, v in ipairs(s) do
		m = math.max(m, v)
	end
	for i, v in ipairs(s) do
		p[i] = math.exp((v - m) / (t or 1))
		z = z + p[i]
	end
	for i = 1, #p do
		p[i] = p[i] / z
	end
	return p
end

local function layer_norm(x, w, b, eps)
	local m, v = 0.0, 0.0
	for _, a in ipairs(x) do
		m = m + a
	end
	m = m / #x
	for _, a in ipairs(x) do
		v = v + (a - m) ^ 2
	end
	local r = 1 / math.sqrt(v / #x + eps)
	local y = {}
	for i, a in ipairs(x) do
		y[i] = (a - m) * r * w[i] + b[i]
	end
	return y
end

local function l2(x)
	local s = math.max(math.sqrt(dot(x, x)), 1e-12)
	local y = {}
	for i, a in ipairs(x) do
		y[i] = a / s
	end
	return y
end

-- erf by Abramowitz and Stegun 7.1.26 (error below 1.5e-7).
local function erf(x)
	local s = x < 0 and -1 or 1
	x = math.abs(x)
	local t = 1 / (1 + 0.3275911 * x)
	local y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
	    0.254829592) * t * math.exp(-x * x)
	return s * y
end

local function gelu(x)
	local y = {}
	for i, a in ipairs(x) do
		y[i] = 0.5 * a * (1 + erf(a / math.sqrt(2)))
	end
	return y
end

local function mean(rows, a, b)
	local y = {}
	for j = 1, #rows[a] do
		local s = 0.0
		for i = a, b do
			s = s + rows[i][j]
		end
		y[j] = s / (b - a + 1)
	end
	return y
end

-- The normed hidden states of all tokens of ids, from one plain pass.
local function hidden_rows(model, ids, out_norm)
	local hp = model.hp
	local inp = model:inputs(ids, model:cache(#ids + 8), { hidden = hp.n_layer })
	local g = model.be:graph()
	local x = model:forward(g, inp)
	g:compute(x)
	local flat = x:get()
	g:close()
	local E, rows = hp.n_embd, {}
	for i = 1, #ids do
		local r = table.move(flat, (i - 1) * E + 1, i * E, 1, {})
		local ms = dot(r, r) / E
		local s = 1 / math.sqrt(ms + hp.eps)
		for j = 1, E do
			r[j] = r[j] * s * out_norm[j]
		end
		rows[i] = r
	end
	return rows
end

local function rand(r, count, lo, width)
	local v = {}
	for i = 1, count do
		v[i] = lo + width * r()
	end
	return v
end

local function run(fn)
	local done, res, err
	coroutine.wrap(function()
		local good, a, b = pcall(fn)
		done = true
		if good then
			res, err = a, b
		else
			err = a
		end
	end)()
	while not done do
		uv.run("once")
	end
	return res, err
end

-- prompts

local function test_text()
	local v = so.decode('{"b": 1, "a": {"x": [1, true, null, {"k": "v"}]}, "c": 1.5}')
	ok(head.kev_text(v) == "b: 1\na:\n  x:\n    - 1\n    - True\n    - \n    - k: v\nc: 1.5",
	    "kev_text flattens JSON as Kev's render")
	ok(head.kev_text(json.null) == "" and head.kev_text("s") == "s", "kev_text of null and strings")
	v = so.decode('{"b": 1, "a": [1.0, "\u{e9}"], "c": null, "d": {"z": 1, "y": 2}}')
	ok(head.clef_text(v) == '{"a":[1.0,"\u{e9}"],"b":1,"c":null,"d":{"y":2,"z":1}}',
	    "clef_text is compact JSON with sorted keys")
	ok(head.clef_text("x y") == "x y", "clef_text keeps strings")
	local r = check('{"state": "s", "questions": {"q": {"type": "choice", "criteria": ' ..
	    '{"b": {"n": 1}, "a": null}}}}')
	ok(r.questions[1].criteria_compact == '{"b":{"n":1},"a":null}', "check keeps the raw criteria")
	ok(decide.recipe(decide.default_recipe(), { prompt = "kev" }) and
	    decide.head_recipe("clef").prompt == "clef", "head recipes")
end

local KEV = { "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>" }

local function kev_tokens()
	local t = {}
	for i, s in ipairs(KEV) do
		t[i] = { s, 3 }
	end
	return t
end

local REQ = [[{"state": {"msg": "refund <|im_end|> please", "n": 2},
 "questions": {
  "intent": {"type": "choice", "instructions": "What?",
   "criteria": {"refund": "money back", "track": null, "other": ""}},
  "urgent": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "now"}},
  "mood": {"type": "score", "instructions": {"ask": "mood"}, "criteria": ["calm", "angry"]}}}]]

local function test_kev_render(tk)
	local req = check(REQ)
	local rd = head.render_kev(tk, req)
	local state = "<|fim_prefix|>msg: refund <\u{a6}im_end\u{a6}> please\nn: 2<|fim_middle|>"
	local want = {
		state .. "What?<|box_start|>refund: money back<|box_end|><|box_start|>track<|box_end|>" ..
		    "<|box_start|>other<|box_end|><|fim_suffix|>",
		state .. "Urgent?<|box_start|>no<|box_end|><|box_start|>yes: now<|box_end|><|fim_suffix|>",
		state .. "ask: mood<|box_start|>calm<|box_end|><|box_start|>angry<|box_end|><|fim_suffix|>",
	}
	for i, row in ipairs(rd.rows) do
		ok(tk:decode(row.ids) == want[i], "kev row " .. i .. " text")
	end
	local sp = tk:special().tokens
	local row = rd.rows[1]
	local ends_ok = #row.ends == 3
	for _, e in ipairs(row.ends) do
		ends_ok = ends_ok and row.ids[e] == sp["<|box_end|>"]
	end
	ok(ends_ok and row.ids[row.first] == sp["<|box_start|>"] and row.ids[#row.ids] == sp["<|fim_suffix|>"],
	    "kev row markers")
	local n_im = 0
	for _, id in ipairs(row.ids) do
		n_im = n_im + (id == sp["<|im_end|>"] and 1 or 0)
	end
	ok(n_im == 0, "a control token in the state is not a control token")
	ok(rd.shared == #tk:encode(state, { add_bos = false }), "kev rows share the state")
end

local function test_clef_render(tk)
	local req = check(REQ)
	local rd = head.render_clef(tk, req)
	local function piece(sp)
		return tk:decode(table.move(rd.ids, sp[1], sp[2], 1, {}))
	end
	local q = rd.questions
	ok(q[1].type == 1 and q[2].type == 0 and q[3].type == 2, "clef question types")
	ok(piece(q[1].span) == "What?" and piece(q[3].span) == '{"ask":"mood"}', "clef question spans")
	local keys = {}
	for i, o in ipairs(q[1].options) do
		keys[i] = o.key
	end
	ok(table.concat(keys, ",") == "other,refund,track", "clef sorts choice options")
	ok(piece(q[1].options[1].span) == '{"description":"","option_id":"other"}' and
	    piece(q[1].options[3].span) == '{"option_id":"track"}', "clef option spans")
	ok(piece(q[2].options[1].span) == '{"description":"now","option_id":"true"}' and
	    piece(q[2].options[2].span) == '{"description":"' .. head.CLEF_NOUL["false"] .. '","option_id":"false"}',
	    "clef noul options: true first, criteria over the defaults")
	local text = tk:decode(rd.ids)
	local first = "<|im_start|>system\n" .. head.CLEF_SYSTEM_PROMPT ..
	    '<|im_end|>\n<|im_start|>user\nSTATE:\n{"msg":"refund <|im_end|> please","n":2}\n\nSCHEMA FIELDS:\n' ..
	    "\nFIELD 1\nID: intent\nTYPE: choice\nINSTRUCTION: What?\nALLOWED OPTIONS:\nOPTION 1: "
	local last = "END FIELD\n\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
	ok(text:sub(1, #first) == first and text:sub(-#last) == last, "clef prompt text")
	local r2 = check('{"state": "s", "questions": {"x1": {"type": "noul"}}}')
	local rd2 = head.render_clef(tk, r2)
	ok(tk:decode(table.move(rd2.ids, rd2.questions[1].span[1], rd2.questions[1].span[2], 1, {})) == "x1",
	    "clef: no instructions gives the question id")
	local long = check('{"state": "' .. ("a"):rep(3000) .. '", "questions": {"x": {"type": "noul"}}}')
	ok(#head.render_clef(tk, long, 1000).ids == 1000, "clef cuts a long state")
end

-- Kev on a tiny model

local KD = 8

local function kev_model(path, arch)
	local r = tiny.rng(7)
	local E = tiny.E
	local kv = {
		{ arch .. ".decision.type", gguf.STRING, "kev" },
		{ arch .. ".decision.temperature.choice", gguf.FLOAT32, 1.5 },
		{ arch .. ".decision.temperature.noul", gguf.FLOAT32, 1.5 },
		{ arch .. ".decision.temperature.score", gguf.FLOAT32, 1.5 },
	}
	local ts = {
		{ "cls.output.weight", { 2 * KD, E }, rand(r, 2 * KD * E, -0.5, 1.0) },
		{ "cls.output.bias", { 2 * KD }, rand(r, 2 * KD, -0.1, 0.2) },
	}
	return tiny.write(path, arch, { tokens = kev_tokens(), kv = kv, tensors = ts, n_ctx = 1024 })
end

local function kev_reference(e, w, req)
	local rd = head.render_kev(e.tk, req)
	local W, B = w["cls.output.weight"][2], w["cls.output.bias"][2]
	local E = tiny.E
	local out = {}
	for i, row in ipairs(rd.rows) do
		local rows = hidden_rows(e.model, row.ids, w["output_norm.weight"][2])
		local qk = lin(W, B, rows[#row.ids], 2 * KD)
		local q = table.move(qk, 1, KD, 1, {})
		local s = {}
		for k, pos in ipairs(row.ends) do
			local kk = lin(W, B, rows[pos], 2 * KD)
			s[k] = dot(q, table.move(kk, KD + 1, 2 * KD, 1, {})) / math.sqrt(KD)
		end
		out[req.questions[i].id] = softmax(s, 1.5)
	end
	assert(E)
	return out
end

local function close_to(a, b, eps)
	if #a ~= #b then
		return false
	end
	for i = 1, #a do
		if math.abs(a[i] - b[i]) > eps then
			return false
		end
	end
	return true
end

local function test_kev(arch)
	local path = tiny.tmp(".gguf")
	local w = kev_model(path, arch)
	-- Soft_max attention with an f16 cache matches the plain pass.
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 1024, ubatch = 16, flash_attn = false })
	ok(e.recipe.prompt == "kev" and e.decide_head.dim == KD, arch .. ": kev head picked from the metadata")
	local req, body = check(REQ)
	local ref = kev_reference(e, w, req)
	for round = 1, 2 do
		local r, err = run(function()
			return e:decide(req, {})
		end)
		ok(r ~= nil, arch .. ": kev decide runs" .. (r and "" or ": " .. tostring(err)))
		if not r then
			break
		end
		local good = true
		for _, q in ipairs(req.questions) do
			local a = r.answers[q.id]
			local want = ref[q.id]
			if q.type == "noul" then
				good = good and math.abs(a.yes - want[2]) < 1e-3
			else
				good = good and close_to(a.probabilities, want, 1e-3)
			end
		end
		ok(good, ("%s: kev probabilities match the reference (round %d, cached %d)"):format(arch, round,
		    r.timings.cache_n))
		if round == 2 then
			ok(r.timings.cache_n > 0, arch .. ": the second request reuses the state")
		end
		local reply = json.decode(so.reply(req, body, r))
		local s = 0
		for _, p in pairs(reply.answers.intent.probabilities) do
			s = s + p
		end
		ok(math.abs(s - 1) < 1e-9 and reply.answers.urgent.noul and reply.answers.mood.score,
		    arch .. ": kev reply")
	end
	local r, msg = run(function()
		local x = check('{"state": "s", "questions": {"q": {"type": "noul"}}, ' ..
		    '"attachments": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk' ..
		    '+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="]}')
		return e:decide(x, {})
	end)
	ok(r == nil and msg, arch .. ": kev refuses images")
	os.remove(path)
end

-- Clef on a tiny model

local CW, CH, CF = 16, 2, 32

local function clef_model(path)
	local r = tiny.rng(11)
	local E, arch = tiny.E, "qwen3next"
	local ts = {}
	local function t(name, shape, lo, width)
		local count = 1
		for _, s in ipairs(shape) do
			count = count * s
		end
		ts[#ts + 1] = { name, shape, rand(r, count, lo, width) }
	end
	local function nrm(p, d)
		t(p .. ".weight", { d }, 0.8, 0.4)
		t(p .. ".bias", { d }, -0.1, 0.2)
	end
	local function attn(p)
		for _, x in ipairs({ "q", "k", "v", "o" }) do
			t(p .. "_" .. x .. ".weight", { CW, CW }, -0.5, 1.0)
			t(p .. "_" .. x .. ".bias", { CW }, -0.1, 0.2)
		end
	end
	for i = 0, 1 do
		local p = "dec.blk." .. i .. "."
		nrm(p .. "cross_attn_norm", CW)
		attn(p .. "cross_attn")
		nrm(p .. "ffn_norm", CW)
		t(p .. "ffn_up.weight", { CF, CW }, -0.5, 1.0)
		t(p .. "ffn_up.bias", { CF }, -0.1, 0.2)
		t(p .. "ffn_down.weight", { CW, CF }, -0.3, 0.6)
		t(p .. "ffn_down.bias", { CW }, -0.1, 0.2)
		if i == 0 then
			nrm(p .. "cross_attn_norm_kv", CW)
		else
			nrm(p .. "attn_norm", CW)
			attn(p .. "attn")
		end
	end
	nrm("decision.hidden_norm", E)
	for _, x in ipairs({ "option_summary_norm", "field_norm", "option_norm" }) do
		nrm("decision." .. x, CW)
	end
	for _, x in ipairs({ "memory", "question", "option_question", "global", "option_context", "option_lexical" }) do
		t("decision.proj_" .. x .. ".weight", { CW, E }, -0.3, 0.6)
	end
	t("token_types.weight", { 3, CW }, -0.5, 1.0)
	t("decision.scorer.weight", { CW, 4 * CW }, -0.3, 0.6)
	t("decision.scorer.bias", { CW }, -0.1, 0.2)
	t("decision.scorer_out.weight", { 1, CW }, -0.5, 1.0)
	t("decision.scorer_out.bias", { 1 }, -0.1, 0.2)
	ts[#ts + 1] = { "decision.scales", { 3 }, { 2.5, 3.0, 0.6 } }
	local kv = {
		{ arch .. ".decision.type", gguf.STRING, "clef" },
		{ arch .. ".decision.routing_block_count", gguf.UINT32, 1 },
		{ arch .. ".decision.block_count", gguf.UINT32, 1 },
		{ arch .. ".decision.head_count", gguf.UINT32, CH },
		{ arch .. ".attention.layer_norm_epsilon", gguf.FLOAT32, 1e-5 },
	}
	return tiny.write(path, arch, { kv = kv, tensors = ts, n_ctx = 2048 })
end

-- JointSchemaHead.forward of the model code, in plain Lua.
local function clef_reference(e, w, req)
	local rd = head.render_clef(e.tk, req)
	local function v(name)
		return w[name][2]
	end
	local eps = 1e-5
	local hidden = hidden_rows(e.model, rd.ids, v("output_norm.weight"))
	local x = {}
	for i, h in ipairs(hidden) do
		x[i] = layer_norm(h, v("decision.hidden_norm.weight"), v("decision.hidden_norm.bias"), eps)
	end
	local function ln(a, p)
		return layer_norm(a, v(p .. ".weight"), v(p .. ".bias"), eps)
	end
	local function proj(name, a)
		return lin(v("decision.proj_" .. name .. ".weight"), nil, a, CW)
	end
	local function mha(p, qs, kvs)
		local function l(xx, a)
			return lin(v(p .. "_" .. xx .. ".weight"), v(p .. "_" .. xx .. ".bias"), a, CW)
		end
		local Q, K, V = {}, {}, {}
		for i, a in ipairs(qs) do
			Q[i] = l("q", a)
		end
		for i, a in ipairs(kvs) do
			K[i], V[i] = l("k", a), l("v", a)
		end
		local d, out = CW // CH, {}
		for i = 1, #qs do
			local o = {}
			for hh = 0, CH - 1 do
				local s = {}
				for j = 1, #kvs do
					local acc = 0.0
					for c = 1, d do
						acc = acc + Q[i][hh * d + c] * K[j][hh * d + c]
					end
					s[j] = acc / math.sqrt(d)
				end
				local p = softmax(s)
				for c = 1, d do
					local acc = 0.0
					for j = 1, #kvs do
						acc = acc + p[j] * V[j][hh * d + c]
					end
					o[hh * d + c] = acc
				end
			end
			out[i] = l("o", o)
		end
		return out
	end
	local function ffn(p, a)
		local y = gelu(lin(v(p .. "ffn_up.weight"), v(p .. "ffn_up.bias"), ln(a, p .. "ffn_norm"), CF))
		return lin(v(p .. "ffn_down.weight"), v(p .. "ffn_down.bias"), y, CW)
	end
	local mem = {}
	for i, a in ipairs(x) do
		mem[i] = proj("memory", a)
	end
	local glob = x[#x]
	local out = v("output.weight")
	local E = tiny.E
	local qv, opts, lexs, owner = {}, {}, {}, {}
	for qi, q in ipairs(rd.questions) do
		qv[qi] = mean(x, q.span[1], q.span[2])
		for _, o in ipairs(q.options) do
			local rows = {}
			for k = o.span[1], o.span[2] do
				local id = rd.ids[k]
				rows[#rows + 1] = table.move(out, id * E + 1, (id + 1) * E, 1, {})
			end
			local lex = mean(rows, 1, #rows)
			lexs[#lexs + 1] = lex
			owner[#owner + 1] = qi
			local a = add(proj("option_context", mean(x, o.span[1], o.span[2])), proj("option_lexical", lex))
			opts[#opts + 1] = add(a, proj("option_question", qv[qi]))
		end
	end
	local p0 = "dec.blk.0."
	local kvn = {}
	for i, m in ipairs(mem) do
		kvn[i] = ln(m, p0 .. "cross_attn_norm_kv")
	end
	local qn = {}
	for i, o in ipairs(opts) do
		qn[i] = ln(o, p0 .. "cross_attn_norm")
	end
	local routed = mha(p0 .. "cross_attn", qn, kvn)
	for i = 1, #opts do
		opts[i] = add(opts[i], routed[i])
		opts[i] = add(opts[i], ffn(p0, opts[i]))
	end
	local fields = {}
	for qi in ipairs(rd.questions) do
		local f = proj("question", qv[qi])
		local mine, s = {}, {}
		for oi, o in ipairs(opts) do
			if owner[oi] == qi then
				mine[#mine + 1] = o
				s[#s + 1] = dot(o, f) / math.sqrt(CW)
			end
		end
		local p = softmax(s)
		local sum = {}
		for c = 1, CW do
			local acc = 0.0
			for j, o in ipairs(mine) do
				acc = acc + p[j] * o[c]
			end
			sum[c] = acc
		end
		f = add(f, ln(sum, "decision.option_summary_norm"))
		f = add(f, proj("global", glob))
		local te = v("token_types.weight")
		f = add(f, table.move(te, rd.questions[qi].type * CW + 1, (rd.questions[qi].type + 1) * CW, 1, {}))
		fields[qi] = f
	end
	local p1 = "dec.blk.1."
	local c = {}
	for i, f in ipairs(fields) do
		c[i] = ln(f, p1 .. "attn_norm")
	end
	local sa = mha(p1 .. "attn", c, c)
	for i = 1, #fields do
		fields[i] = add(fields[i], sa[i])
		c[i] = ln(fields[i], p1 .. "cross_attn_norm")
	end
	local ca = mha(p1 .. "cross_attn", c, mem)
	for i = 1, #fields do
		fields[i] = add(fields[i], ca[i])
		fields[i] = add(fields[i], ffn(p1, fields[i]))
		fields[i] = ln(fields[i], "decision.field_norm")
	end
	local scales = v("decision.scales")
	local logits = {}
	for oi, o in ipairs(opts) do
		local qi = owner[oi]
		local prior = scales[1] * dot(l2(lexs[oi]), l2(add(qv[qi], glob)))
		local on = ln(o, "decision.option_norm")
		local f = fields[qi]
		local cos = dot(l2(f), l2(on))
		local feat = {}
		for k = 1, CW do
			feat[k], feat[CW + k], feat[2 * CW + k], feat[3 * CW + k] = f[k], on[k], f[k] * on[k],
			    math.abs(f[k] - on[k])
		end
		local hdn = gelu(lin(v("decision.scorer.weight"), v("decision.scorer.bias"), feat, CW))
		local res = lin(v("decision.scorer_out.weight"), v("decision.scorer_out.bias"), hdn, 1)[1]
		logits[#logits + 1] = prior + scales[3] * (scales[2] * cos + res)
	end
	local out_p, k = {}, 0
	for qi, q in ipairs(rd.questions) do
		local ls = table.move(logits, k + 1, k + #q.options, 1, {})
		k = k + #q.options
		local p = softmax(ls)
		local by = {}
		for j, o in ipairs(q.options) do
			by[o.key] = p[j]
		end
		out_p[req.questions[qi].id] = by
	end
	return out_p
end

local function test_clef()
	local path = tiny.tmp(".gguf")
	local w = clef_model(path)
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 2048, ubatch = 64, flash_attn = false })
	ok(e.recipe.prompt == "clef" and e.decide_head.n_routing == 1, "clef head picked from the metadata")
	local req, body = check(REQ)
	local ref = clef_reference(e, w, req)
	local r, err = run(function()
		return e:decide(req, {})
	end)
	ok(r ~= nil, "clef decide runs" .. (r and "" or ": " .. tostring(err)))
	if not r then
		return
	end
	local good = true
	for _, q in ipairs(req.questions) do
		local a, want = r.answers[q.id], ref[q.id]
		if q.type == "noul" then
			good = good and math.abs(a.yes - want["true"]) < 1e-3
		elseif q.type == "choice" then
			for j, o in ipairs(q.options) do
				good = good and math.abs(a.probabilities[j] - want[o.key]) < 1e-3
			end
		else
			for j = 1, #q.levels do
				good = good and math.abs(a.probabilities[j] - want[tostring(j - 1)]) < 1e-3
			end
		end
	end
	ok(good, "clef probabilities match the reference")
	ok(r.usage.input_tokens == #head.render_clef(e.tk, req).ids, "clef usage")
	local reply = json.decode(so.reply(req, body, r))
	local s = 0
	for _, p in pairs(reply.answers.mood.probabilities) do
		s = s + p
	end
	ok(math.abs(s - 1) < 1e-9, "clef reply")
	os.remove(path)
end

local function main()
	test_text()
	local path = tiny.tmp(".gguf")
	tiny.write(path, "qwen3", { tokens = kev_tokens() })
	local tk = assert(require("lgml.tokenizer").open(path))
	test_kev_render(tk)
	test_clef_render(tk)
	os.remove(path)
	test_kev("qwen3")
	test_kev("qwen3next")
	test_clef()
	print("1.." .. n)
end

local good, err = xpcall(main, debug.traceback)
if not good then
	print("not ok - " .. tostring(err))
	failed = failed + 1
end
os.exit(failed == 0 and 0 or 1)
