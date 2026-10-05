-- qwen4exp on a tiny gguf made here: logits against llama.cpp, QSA
-- against dense attention under the budget, replay, snapshots, batched
-- pool sequences, prefix reuse across turns and concurrent slots.
-- Output is TAP. LGML_LLAMA_LOGITS names a llama_logits.c binary built
-- against a llama.cpp with qwen4exp, to compare every logit live;
-- without it the rows in REF stand in. LGML_REF_DUMP=1 prints them.
local lgml = require("lgml")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local q4t = require("q4_tiny")
local tiny = require("tiny")
local rc = require("replay_check")
local transformer = require("lgml.transformer")
local replay = require("lgml.replay")

local LOOSE = 1e-2
local TOL = 2e-3

local n = 0
local failed = false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
	return cond
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

local function maxdiff(a, b)
	local d = #a == #b and 0 or math.huge
	for i = 1, #a do
		d = math.max(d, math.abs(a[i] - (b[i] or math.huge)))
	end
	return d
end

-- Two prompt batches, then greedy steps on llama.cpp's choices. The
-- first batch holds more blocks than QSA keeps, so tokens past it pick
-- blocks.
local batches = { {}, { 256, 32, 119, 258, 40, 41, 42, 43, 44, 45, 46, 47 } }
for i = 1, 24 do
	batches[1][i] = (i * 37) % 250 + 1
end
local N_DECODE = 6

-- Reference rows: input id, argmax, logits at IDX, from llama.cpp on
-- the CPU with an f16 cache.
local IDX = { 1, 33, 65, 101, 151, 201, 256, 259 }
local REF = {}
REF.qwen4exp = {
	{ id = 38, best = 191, v = { -0.0660873, 0.0404728, 0.0719145, -0.75568, 0.483401, -0.23356, -0.444648, -0.66464 } },
	{ id = 75, best = 252, v = { -0.0421746, 0.233806, -0.111072, 0.618781, 0.413401, 0.660207, -0.239843, -0.576422 } },
	{ id = 112, best = 149, v = { 0.375785, -0.191538, -0.277471, 0.493285, -0.20825, 0.237303, 0.940415, -0.42307 } },
	{ id = 149, best = 97, v = { -0.155305, 0.00418949, 0.403033, -0.625235, 0.0495216, 0.25079, -0.212071, -0.357097 } },
	{ id = 186, best = 153, v = { 0.50302, -0.920059, -0.191753, -0.178352, -0.29892, -0.745764, -0.0774762, -0.26861 } },
	{ id = 223, best = 196, v = { -0.0137927, -0.777606, -0.470972, 0.61059, 0.514302, -0.146706, 0.305868, 0.24146 } },
	{ id = 10, best = 89, v = { -0.277042, -0.0353189, 0.035085, -0.0830097, 0.132275, 0.335096, 0.476035, -0.17986 } },
	{ id = 47, best = 23, v = { -0.53041, 0.105469, 0.114064, -0.247794, 0.269412, -0.395875, 0.217729, -0.35417 } },
	{ id = 84, best = 23, v = { -0.496504, -0.495978, -0.0337761, -1.05027, -0.230554, 0.331403, 0.0879018, -0.319152 } },
	{ id = 121, best = 149, v = { 0.81409, 0.110682, 0.502775, 0.603746, -0.54944, 0.63838, 0.474622, -0.356521 } },
	{ id = 158, best = 109, v = { -0.252376, 0.20413, -0.253596, -0.309834, -0.342395, -0.235653, 0.268521, -0.0558692 } },
	{ id = 195, best = 195, v = { 0.120479, 0.673776, -0.218028, -0.474283, 0.0318199, 0.0163394, -0.306694, 0.170782 } },
	{ id = 232, best = 106, v = { 0.468591, 0.59983, 0.386816, -0.353348, 0.128076, -0.117075, -0.263758, 0.0171622 } },
	{ id = 19, best = 97, v = { -0.061353, 0.147166, -0.363506, -0.227226, -0.442939, 0.325848, -0.281524, -0.0307085 } },
	{ id = 56, best = 40, v = { 0.0104282, 0.282552, 0.507229, -0.738687, 0.133515, -0.104311, -0.833311, 0.147277 } },
	{ id = 93, best = 249, v = { -0.124683, 0.134104, 0.728731, 0.00795617, -0.591609, 0.325244, 0.761039, -0.215142 } },
	{ id = 130, best = 159, v = { -0.380862, 0.326523, -0.972076, -0.538043, -0.479756, 0.212217, 1.09551, -0.0986051 } },
	{ id = 167, best = 129, v = { -0.727028, 0.338781, -0.0394232, -0.00799428, -0.0816023, -0.00818276, 0.270011, 0.699435 } },
	{ id = 204, best = 100, v = { -0.145922, 0.225417, 0.107109, 1.30033, 0.305939, 0.785619, 0.33807, -0.373437 } },
	{ id = 241, best = 31, v = { -0.320449, -0.42698, -0.974221, -1.09707, -0.063148, -0.556845, 0.321464, 0.0770013 } },
	{ id = 28, best = 26, v = { -0.456056, -0.367253, 0.44877, -0.646503, -0.218391, 0.226044, -0.326123, -0.165062 } },
	{ id = 65, best = 26, v = { -0.621879, -0.102513, 0.26501, -1.13398, -0.804364, -0.051437, 0.265238, -0.170582 } },
	{ id = 102, best = 43, v = { 0.0704501, 0.485335, -0.0273116, -0.780686, -0.194305, 0.049992, -0.253824, -0.55947 } },
	{ id = 139, best = 228, v = { 0.00755371, -0.0901106, 0.0202598, -0.340314, 0.121968, 0.122673, 0.252887, 0.262954 } },
	{ id = 256, best = 92, v = { 0.396818, -0.0122362, -1.08356, 0.547901, -0.0147379, -0.0491633, 0.48457, -0.366672 } },
	{ id = 32, best = 97, v = { 0.0177897, -0.27755, 0.229497, -0.457128, -0.191673, -0.297201, -0.464088, 0.479746 } },
	{ id = 119, best = 109, v = { 0.22999, 0.604355, 0.0286117, 0.324815, -0.356258, 0.408587, 0.182256, 0.299847 } },
	{ id = 258, best = 189, v = { 0.0886941, -0.00752734, 0.362283, 0.0534076, -0.20452, 0.22141, 0.298622, 0.525962 } },
	{ id = 40, best = 105, v = { -0.909354, -0.179814, 0.517808, 0.0565536, -0.851511, 0.0939736, -0.807586, -0.0715542 } },
	{ id = 41, best = 124, v = { 0.490538, -0.498989, 0.162972, 0.563501, -0.330064, 0.213211, 0.279644, -0.068989 } },
	{ id = 42, best = 173, v = { -0.835311, 0.185269, 0.163407, 0.279273, -0.283068, -0.247224, 0.313528, 0.25852 } },
	{ id = 43, best = 120, v = { -0.0289495, -0.257171, 0.497051, 0.552515, 0.49723, -0.116804, -0.174819, -0.477835 } },
	{ id = 44, best = 142, v = { 0.0187282, 0.0549271, -0.122973, -0.654071, -0.279796, -0.402235, 0.685679, -0.197668 } },
	{ id = 45, best = 106, v = { -0.265359, 0.102379, 0.093068, -1.02683, -0.557311, -0.546857, -0.395345, 0.0611512 } },
	{ id = 46, best = 232, v = { -0.158926, 0.965786, 0.103339, -0.246313, -0.198204, 0.976837, 0.258517, -0.233758 } },
	{ id = 47, best = 235, v = { -0.327881, 0.324062, -0.0474933, -0.490477, 0.47182, 0.55878, 0.749607, 0.069862 } },
	{ id = 235, best = 169, v = { -0.725652, 0.0937785, 0.4091, -0.801309, 0.565224, 0.40447, 0.349411, 0.34244 } },
	{ id = 169, best = 232, v = { -0.517975, 0.537022, -0.164796, -0.551465, -0.642332, 0.412673, -0.319392, -0.580014 } },
	{ id = 232, best = 233, v = { 0.151051, 0.674685, -0.257766, -0.865591, -0.308843, -0.11748, -0.696961, -0.26308 } },
	{ id = 233, best = 43, v = { -0.0608586, 0.383817, -0.170114, -0.200255, 0.411268, -0.0686515, 0.0197626, 0.303671 } },
	{ id = 43, best = 154, v = { -0.275251, -0.154392, -0.319901, -0.360061, -0.0323996, 0.436716, 0.784058, -0.23705 } },
	{ id = 154, best = 25, v = { 0.707843, -0.5953, -0.643481, -0.0504971, -0.105551, -0.365725, 1.014, 0.256191 } },
}

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

-- Run ids at cache.n_past. Returns one logits list per token.
local function eval(m, cache, ids, opts)
	local inp = m:inputs(ids, cache, opts)
	local g <close> = m.be:graph()
	local logits = m:forward(g, inp)
	g:compute(logits)
	cache.n_past = inp.n_kv
	local flat, nv = logits:get(), logits:shape()[2]
	local rows = {}
	for i = 1, #ids do
		rows[i] = table.move(flat, (i - 1) * nv + 1, i * nv, 1, {})
	end
	for _, t in pairs(inp) do
		if type(t) == "userdata" then
			t:close()
		end
	end
	return rows
end

local function our_rows(m, ref, opts)
	local c = m:cache(256)
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
	return rows
end

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

-- QSA under the budget is dense attention: a short prompt gives the
-- logits of the same model without the indexer. A long one differs.
local function dense_check(m, path)
	local be = m.be
	local d = transformer.load(be, path)
	d.hp.idx = nil
	for _, L in ipairs(d.layers) do
		L.idx = nil
	end
	local short = table.move(batches[1], 1, 11, 1, {})
	local a = eval(m, m:cache(256), short)
	local b = eval(d, d:cache(256), short)
	local e = 0
	for i = 1, #a do
		e = math.max(e, maxdiff(a[i], b[i]))
	end
	ok(e < 1e-5, ("qsa: within the budget the logits equal dense attention (max error %.3g)"):format(e))
	a = eval(m, m:cache(256), batches[1])
	b = eval(d, d:cache(256), batches[1])
	e = 0
	for i = 1, #a do
		e = math.max(e, maxdiff(a[i], b[i]))
	end
	ok(e > 1e-4, ("qsa: past the budget the logits differ from dense attention (max difference %.3g)"):format(e))
	d:close()
end

-- The n-gram rows of the Lua hash for a context with an EOS and a
-- sequence start, against the formula by hand.
local function hash_check(m)
	local q4 = require("lgml.qwen4exp")
	local p = m.hp.ple
	local rows = q4.ngram_rows(p, { 5 }, { 7, q4t.EOS, 9 }, {})
	local function row(h, ctx)
		local x = ctx[1] * p.mult[1]
		for j = 2, #ctx do
			x = x ~ (ctx[j] * p.mult[j])
		end
		return x % p.vocab[h] + p.off[h]
	end
	local want, H, per = {}, p.heads, p.per
	local ctxs = { { 7, 5, q4t.EOS }, { q4t.EOS, 7, 5 }, { 9, q4t.EOS, q4t.EOS } }
	for i, ctx in ipairs(ctxs) do
		for h = 1, H do
			local m_ = h <= per and 2 or 3
			want[(i - 1) * H + h] = row(h, table.move(ctx, 1, m_, 1, {}))
		end
	end
	local same = #rows == #want
	for i = 1, #want do
		same = same and rows[i] == want[i]
	end
	ok(same, "ngram: rows of the Lua hash match the formula")
end

-- Snapshots: a full snapshot and a state checkpoint go back to their
-- position, and the next pass gives the same logits.
local function snapshot_check(m)
	local c = m:cache(256)
	local o = { logits = "last" }
	local function run(ids)
		return replay.run(m, c, ids, o):get()
	end
	run(batches[1])
	local full = m:snapshot(c, true)
	local st = m:snapshot(c)
	local want = run({ 11, 12 })
	run({ 13, 14, 15 })
	m:restore(c, full)
	ok(c.n_past == #batches[1] and c.rs_pos == #batches[1], "snapshot: restore sets n_past")
	ok(maxdiff(run({ 11, 12 }), want) == 0, "snapshot: a full snapshot restores the logits")
	run({ 20, 21, 22, 23 })
	m:restore(c, st)
	ok(maxdiff(run({ 11, 12 }), want) == 0, "snapshot: a state checkpoint restores the logits")
	full.buf:close()
	st.buf:close()
	replay.drop(c)
end

-- Three sequences of one pool prefill in one pass, then decode in one
-- pass per step: logits and tokens match each sequence alone.
local function batch_check(m)
	local prompts = { batches[1], { 40, 41, 42, 43, 44, 45, 46, 47, 48 }, { 7 } }
	local steps = 10
	local want, first = {}, {}
	for i, p in ipairs(prompts) do
		local c = m:cache(256)
		local rows = eval(m, c, p)
		first[i] = rows[#rows]
		local id = argmax(rows[#rows])
		want[i] = { id }
		for _ = 2, steps do
			id = argmax(eval(m, c, { id })[1])
			want[i][#want[i] + 1] = id
		end
	end
	local pool = m:pool(3 * 256)
	local caches = {}
	for i = 1, #prompts do
		caches[i] = m:seq(pool, { n_ctx = 256 })
	end
	local rows = m:run_batch(caches, prompts)
	local err, ids, got = 0, {}, {}
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
		same = same and table.concat(got[i], ",") == table.concat(want[i], ",")
	end
	ok(err < 1e-4, ("batch: prefill logits match each sequence alone (max error %.3g)"):format(err))
	ok(same, "batch: decode gives the same tokens as each sequence alone")
	m:drop_batch()
	for _, c in ipairs(caches) do
		m:free_seq(c)
	end
end

-- A request through the engine outside the server.
local function engine_chat(e, messages)
	local uv = require("luv")
	local r = { parts = {} }
	local sink = {
		content = function(_, t)
			r.parts[#r.parts + 1] = t
			return true
		end,
		tool_call = function() return 0 end,
		finish = function(_, reason, usage, timings)
			r.reason, r.usage, r.timings = reason, usage, timings
		end,
		cancelled = function() return false end,
	}
	sink.reasoning, sink.tool_arguments = sink.content, sink.content
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
-- those of a fresh engine.
local function prompt_logits(e, fresh, messages)
	local sink = { cancelled = function() return false end }
	local text = e.family.render(messages, { add_generation_prompt = true, enable_thinking = false })
	local p = e.tk:encode(text)
	local got, want, cached, done
	coroutine.wrap(function()
		cached = e:reuse(p)
		got = e:prefill(p, cached + 1, sink):get()
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

local function long_text(seed, k)
	local r = tiny.rng(seed)
	local w = {}
	for i = 1, k do
		w[i] = string.char(97 + math.floor(r() * 26), 97 + math.floor(r() * 26))
	end
	return table.concat(w, " ")
end

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

-- Turn 2 of a chat starts from the cached prefix of turn 1 and gives
-- the logits of a fresh prefill.
local function prefix_check(path)
	local engine = require("lgml.engine")
	local fresh = engine.new({ model = path, backend = "CPU", n_ctx = 1024, cache_ram = 0,
	    ctx_checkpoints = 0, flash_attn = false, slots = 1 })
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 1024, flash_attn = false, slots = 1 })
	local sys = { role = "system", content = "You are a test. " .. long_text(1, 10) }
	local u1 = { role = "user", content = long_text(2, 30) }
	engine_chat(e, { sys, u1 })
	local turn2 = { sys, u1, { role = "assistant", content = "not what it said" },
	    { role = "user", content = long_text(3, 8) } }
	local got, want, cached, np = prompt_logits(e, fresh, turn2)
	local k = last_start(e, { sys, u1 })
	ok(cached == k and e.pc.restored == k and k > 40,
	    ("prefix: turn 2 reuses %d of %d tokens from the checkpoint at the last turn start (%d)"):format(
	    cached, np, k))
	local d = maxdiff(got, want)
	ok(d == 0, ("prefix: turn 2 logits equal a fresh prefill (max error %.3g)"):format(d))
end

-- Two chats on a two-slot engine run in batched passes and give the
-- replies of one engine each.
local function concurrent_check(path)
	local engine = require("lgml.engine")
	local uv = require("luv")
	local convs = {
		{ { role = "system", content = "You are a test. " .. long_text(4, 12) },
		  { role = "user", content = long_text(5, 8) } },
		{ { role = "user", content = long_text(6, 20) } },
	}
	local want = {}
	for i, msgs in ipairs(convs) do
		local one = engine.new({ model = path, backend = "CPU", n_ctx = 512, flash_attn = false })
		want[i] = engine_chat(one, msgs).text
	end
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 512, flash_attn = false, slots = 2 })
	local got, done = {}, 0
	for i, msgs in ipairs(convs) do
		local text = {}
		local sink = {
			content = function(_, t) text[#text + 1] = t return true end,
			reasoning = function() return true end, tool_arguments = function() return true end,
			tool_call = function() return 0 end, finish = function() end,
			cancelled = function() return false end,
		}
		coroutine.wrap(function()
			e:chat({ messages = msgs, max_tokens = 3, top_k = 1,
			    chat_template_kwargs = { enable_thinking = false } }, sink)
			got[i] = table.concat(text)
			done = done + 1
		end)()
	end
	while done < #convs do
		uv.run("once")
	end
	ok(got[1] == want[1] and got[2] == want[2], "concurrent: two slots give the replies of one engine each")
	ok((e.batch_rows or 0) > (e.batch_passes or 0),
	    ("concurrent: prompts and steps shared passes (%d rows in %d passes)"):format(e.batch_rows or 0,
	    e.batch_passes or 0))
end

local be <close> = assert(lgml.backend("CPU"))
local path = tiny.tmp(".gguf")
local kv, ts = q4t.build()
-- A template that the qwen35 family takes, for the engine checks.
kv[#kv + 1] = { "tokenizer.chat_template", 8,
	"<|im_start|> <tool_call> <function= <think> tool | tojson sysns reasoning_effort" }
require("gguf").write(path, kv, ts)

local m = transformer.load(be, path)
ok(m.hp.hc == 4 and m.hp.idx and m.hp.idx.ratio == 4 and m.hp.ple and m.hp.ple.layer == 3,
    "config: streams, indexer and n-gram layer")

local bin = os.getenv("LGML_LLAMA_LOGITS")
local fh = bin and bin ~= "" and io.open(bin)
local ref
if fh then
	fh:close()
	ref = llama_rows(bin, path)
else
	ref = REF.qwen4exp
end
if not ref then
	n = n + 1
	print("ok " .. n .. " # SKIP no llama.cpp reference")
else
	local rows = our_rows(m, ref)
	local err, same = compare(rows, ref)
	io.stderr:write(("qwen4exp: %d rows, max logit error %.3g\n"):format(#ref, err))
	ok(#rows == #ref, "rows: " .. #rows)
	ok(err < TOL, ("logits match llama.cpp (max error %.3g)"):format(err))
	ok(same, "argmax matches llama.cpp on every row")
	if os.getenv("LGML_REF_DUMP") and ref[1].logits then
		print("REF.qwen4exp = {")
		for _, r in ipairs(ref) do
			local v = {}
			for j, ix in ipairs(IDX) do
				v[j] = ("%.6g"):format(r.logits[ix])
			end
			print(("\t{ id = %d, best = %d, v = { %s } },"):format(r.id, argmax(r.logits), table.concat(v, ", ")))
		end
		print("}")
	end
	err, same = compare(our_rows(m, ref, { flash_attn = true }), ref)
	ok(err < LOOSE and same, ("flash_attn rows agree (max error %.3g)"):format(err))
end

hash_check(m)
dense_check(m, path)

local long = {}
for i = 1, 250 do
	long[i] = (i * 37) % 250 + 1
end
for _, fa in ipairs({ false, true }) do
	rc.decode(ok, fa and "flash_attn: " or "", m, { n_ctx = 320, prompt = long, n = 12, flash_attn = fa,
	    builds = 3 })
end
rc.restart(ok, "", m)
snapshot_check(m)
batch_check(m)
prefix_check(path)
concurrent_check(path)

m:close()
os.remove(path)
print("1.." .. n)
os.exit(failed and 1 or 0)
