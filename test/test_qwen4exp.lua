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
	{ id = 38, best = 191, v = { -0.0660587, 0.0404418, 0.0718169, -0.755638, 0.483457, -0.233498, -0.444489, -0.664739 } },
	{ id = 75, best = 252, v = { -0.0422149, 0.233776, -0.111074, 0.618694, 0.413375, 0.66025, -0.239646, -0.576357 } },
	{ id = 112, best = 149, v = { 0.375697, -0.191593, -0.277472, 0.493606, -0.208377, 0.237213, 0.940673, -0.423187 } },
	{ id = 149, best = 97, v = { -0.155264, 0.00404158, 0.402989, -0.625315, 0.049518, 0.250864, -0.211963, -0.357149 } },
	{ id = 186, best = 153, v = { 0.503104, -0.920133, -0.191858, -0.17845, -0.298921, -0.745662, -0.0774352, -0.26849 } },
	{ id = 223, best = 196, v = { -0.0135683, -0.777641, -0.470764, 0.610659, 0.514271, -0.146682, 0.30589, 0.241604 } },
	{ id = 10, best = 89, v = { -0.277182, -0.0355069, 0.0351672, -0.083088, 0.132181, 0.335135, 0.47597, -0.179775 } },
	{ id = 47, best = 23, v = { -0.530474, 0.105607, 0.113991, -0.247647, 0.26946, -0.396045, 0.217759, -0.354113 } },
	{ id = 84, best = 23, v = { -0.496553, -0.495999, -0.033868, -1.05028, -0.230436, 0.331401, 0.0879302, -0.319225 } },
	{ id = 121, best = 149, v = { 0.814041, 0.110828, 0.502672, 0.603742, -0.549465, 0.638335, 0.474592, -0.356547 } },
	{ id = 158, best = 109, v = { -0.252539, 0.204062, -0.253475, -0.309772, -0.342315, -0.23571, 0.26838, -0.0558064 } },
	{ id = 195, best = 195, v = { 0.120782, 0.673655, -0.218215, -0.474372, 0.0319101, 0.0161403, -0.306877, 0.170554 } },
	{ id = 232, best = 106, v = { 0.468504, 0.599817, 0.386642, -0.353212, 0.128255, -0.11723, -0.263731, 0.0172794 } },
	{ id = 19, best = 97, v = { -0.0614455, 0.147296, -0.363529, -0.227468, -0.442945, 0.325847, -0.281648, -0.0308049 } },
	{ id = 56, best = 40, v = { 0.0103762, 0.282515, 0.507202, -0.738882, 0.133572, -0.104272, -0.833376, 0.147356 } },
	{ id = 93, best = 249, v = { -0.124688, 0.134135, 0.728807, 0.00795843, -0.591641, 0.325184, 0.760995, -0.21505 } },
	{ id = 130, best = 159, v = { -0.380837, 0.326602, -0.972209, -0.538072, -0.479633, 0.212183, 1.09555, -0.0985637 } },
	{ id = 167, best = 129, v = { -0.727049, 0.338812, -0.0393991, -0.00800879, -0.0815908, -0.0081825, 0.270088, 0.699561 } },
	{ id = 204, best = 100, v = { -0.145912, 0.22533, 0.107119, 1.30026, 0.305924, 0.785701, 0.33807, -0.373556 } },
	{ id = 241, best = 31, v = { -0.320324, -0.426898, -0.974378, -1.09721, -0.0630042, -0.556737, 0.321284, 0.076995 } },
	{ id = 28, best = 26, v = { -0.456199, -0.367216, 0.448745, -0.646676, -0.218301, 0.225872, -0.326334, -0.165174 } },
	{ id = 65, best = 26, v = { -0.621807, -0.102598, 0.264979, -1.13397, -0.804338, -0.0513456, 0.265222, -0.170684 } },
	{ id = 102, best = 43, v = { 0.070708, 0.48534, -0.0267242, -0.780083, -0.194205, 0.0499793, -0.254433, -0.559499 } },
	{ id = 139, best = 228, v = { 0.00756524, -0.0899954, 0.0202861, -0.340216, 0.121971, 0.122718, 0.252797, 0.262952 } },
	{ id = 256, best = 92, v = { 0.396908, -0.0123956, -1.08373, 0.547669, -0.0147665, -0.049217, 0.484632, -0.366676 } },
	{ id = 32, best = 97, v = { 0.017775, -0.277574, 0.229508, -0.457195, -0.19166, -0.297194, -0.46421, 0.479781 } },
	{ id = 119, best = 109, v = { 0.229914, 0.604422, 0.0285699, 0.32477, -0.35625, 0.40861, 0.18223, 0.299824 } },
	{ id = 258, best = 189, v = { 0.0886779, -0.00743698, 0.362226, 0.0532729, -0.204493, 0.22138, 0.298657, 0.526067 } },
	{ id = 40, best = 105, v = { -0.909448, -0.179964, 0.517799, 0.0565553, -0.851522, 0.0939917, -0.807517, -0.0716389 } },
	{ id = 41, best = 124, v = { 0.490492, -0.499012, 0.162948, 0.563488, -0.330078, 0.213214, 0.279646, -0.0689131 } },
	{ id = 42, best = 173, v = { -0.835284, 0.185219, 0.163302, 0.279389, -0.283306, -0.247255, 0.313439, 0.258422 } },
	{ id = 43, best = 120, v = { -0.0291383, -0.257238, 0.497036, 0.552503, 0.497153, -0.116744, -0.174667, -0.47782 } },
	{ id = 44, best = 142, v = { 0.0187444, 0.0551117, -0.122961, -0.654045, -0.279992, -0.402286, 0.685597, -0.197699 } },
	{ id = 45, best = 106, v = { -0.265418, 0.102395, 0.0930151, -1.02676, -0.557283, -0.546856, -0.395312, 0.061173 } },
	{ id = 46, best = 232, v = { -0.159012, 0.965861, 0.10342, -0.246369, -0.198317, 0.976915, 0.258436, -0.233656 } },
	{ id = 47, best = 235, v = { -0.327873, 0.323973, -0.0474791, -0.490509, 0.471787, 0.558839, 0.74963, 0.0698865 } },
	{ id = 235, best = 169, v = { -0.725662, 0.0936843, 0.408967, -0.801452, 0.565214, 0.404488, 0.349522, 0.34243 } },
	{ id = 169, best = 232, v = { -0.517932, 0.53709, -0.164905, -0.551621, -0.642296, 0.412701, -0.319476, -0.580005 } },
	{ id = 232, best = 233, v = { 0.151054, 0.67466, -0.257809, -0.865783, -0.308771, -0.117544, -0.696996, -0.263176 } },
	{ id = 233, best = 43, v = { -0.0608972, 0.383909, -0.17018, -0.200227, 0.411382, -0.0687622, 0.0197626, 0.303635 } },
	{ id = 43, best = 154, v = { -0.275167, -0.154404, -0.320035, -0.360086, -0.0324624, 0.4368, 0.784134, -0.237067 } },
	{ id = 154, best = 25, v = { 0.707868, -0.595287, -0.643505, -0.0504183, -0.105563, -0.365765, 1.01401, 0.256202 } },
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

-- A pass within the budget stores its indexer keys: a prompt in two
-- passes, the second past the budget, gives the keys and logits of one
-- pass. A large budget puts the first pass within it.
local function split_check(m)
	local one = m:cache(256)
	local all = eval(m, one, batches[1])
	local c = m:cache(256)
	local top_k = m.hp.idx.top_k
	m.hp.idx.top_k = 1 << 20
	eval(m, c, table.move(batches[1], 1, 11, 1, {}))
	m.hp.idx.top_k = top_k
	local rest = eval(m, c, table.move(batches[1], 12, #batches[1], 1, {}))
	local e = 0
	for i = 1, #rest do
		e = math.max(e, maxdiff(rest[i], all[11 + i]))
	end
	local ek = 0
	for il, x in pairs(one.x) do
		if il > 0 then
			ek = math.max(ek, maxdiff(c.x[il]:get(), x:get()))
		end
	end
	ok(e < TOL and ek < TOL, ("qsa: a pass within the budget keeps its keys (max error %.3g, keys %.3g)"):format(e, ek))
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

-- The n-gram reader of lgml.ngram and get_rows on the table give the
-- same rows and the same logits. m uses the reader.
local function reader_check(m, path)
	local q4 = require("lgml.qwen4exp")
	local p = m.hp.ple
	local ids, prev = { 7, q4t.EOS, 9, 30, 31 }, { 5, 6 }
	local got = m.ngram:rows(ids, prev)
	local want = q4.ngram_rows(p, prev, ids, {})
	local same = #got == #want
	for i = 1, #want do
		same = same and got[i] == want[i]
	end
	ok(same, "ngram: the reader's rows match the Lua hash")
	local o = transformer.load(m.be, path, { ngram = false })
	ok(not o.ngram and o.ple_table and o.ple_mapped, "ngram: without the reader the table is mapped")
	local a = eval(m, m:cache(256), batches[1])
	local b = eval(o, o:cache(256), batches[1])
	local e = 0
	for i = 1, #a do
		e = math.max(e, maxdiff(a[i], b[i]))
	end
	ok(e == 0, ("ngram: the reader gives the logits of get_rows on the table (max error %.3g)"):format(e))
	o:close()
end

-- A placement split loads the weights to a backend and its CPU, as on
-- a GPU, but not the n-gram table: the reader serves it. The model
-- files are split as the real ones are, with the table alone in the
-- second. The memory budget holds every other weight and no more.
local function placement_check(path)
	local gguf = require("gguf")
	local placement = require("lgml.placement")
	local kv, ts = q4t.build()
	local one, two = {}, {}
	for _, t in ipairs(ts) do
		table.insert(t[1] == "per_layer_token_embd.weight" and two or one, t)
	end
	local base = tiny.tmp("")
	local p1, p2 = base .. "-00001-of-00002.gguf", base .. "-00002-of-00002.gguf"
	kv[#kv + 1] = { "split.count", gguf.UINT32, 2 }
	gguf.write(p1, kv, one)
	gguf.write(p2, { { "general.architecture", gguf.STRING, "qwen4exp" }, { "split.count", gguf.UINT32, 2 } }, two)
	local want = 0
	for _, t in ipairs(transformer.weight_tensors(assert(lgml.gguf(p1)), transformer.config(lgml.gguf(p1).meta))) do
		want = want + t.size
		-- The hc norms on the CPU get a scaled copy (see fold_hc).
		if t.name:match("hc_%a*_*norm%.weight$") then
			want = want + t.size
		end
	end
	local sb <close> = assert(lgml.backend("CPU", { cpu = true }))
	collectgarbage()
	collectgarbage()
	local function weights()
		local n = 0
		for _, v in pairs(lgml.memory()) do
			n = n + v.weights
		end
		return n
	end
	local w0 = weights()
	local s = transformer.load(sb, p1, { placement = placement.rules({ n_cpu_moe = 2 }) })
	local dev, host = s.w:size()
	ok(s.ngram and s.w:get("per_layer_token_embd.weight") == nil and host > 0 and dev > 0,
	    "placement: the weights split and the table stays in its file")
	-- Buffers pad their size a little; the table is far larger.
	local got, tbl = weights() - w0, assert(lgml.gguf(p2)).tensors[1].size
	ok(got >= want and got - want < 1024 and tbl > 4096,
	    ("placement: the budget holds the weights (%d bytes for %d) and not the table (%d)"):format(got, want, tbl))
	local a = eval(s, s:cache(256), batches[1])
	local m1 = transformer.load(sb, p1, { ngram = false, mmap = false })
	local b = eval(m1, m1:cache(256), batches[1])
	local e = 0
	for i = 1, #a do
		e = math.max(e, maxdiff(a[i], b[i]))
	end
	ok(e < 1e-4, ("placement: the split model gives the logits of one backend (max error %.3g)"):format(e))
	s:close()
	m1:close()
	os.remove(p1)
	os.remove(p2)
end

-- Expert cache: every routed expert on the CPU and some in device slots
-- that change between passes give the logits of one backend. The KV
-- cache is f32, so that f16 rounding does not grow sum-order changes.
local function xcache_check(m, path)
	local placement = require("lgml.placement")
	local replay = require("lgml.replay")
	local xcache = require("lgml.xcache")
	local ids = {}
	for i = 1, 20 do
		ids[i] = (i * 37) % 250 + 1
	end
	local function rows(model)
		local c = model:cache(64, { type = "f32" })
		local out = {}
		for i, id in ipairs(ids) do
			out[i] = replay.run(model, c, { id }, { logits = "last" }):get()
		end
		return out
	end
	local want = rows(m)
	local sb <close> = assert(lgml.backend("CPU", { cpu = true }))
	local s = transformer.load(sb, path, { placement = placement.rules({ cpu_moe = true }) })
	local n = math.max(1, s.hp.n_expert // 2)
	local xc = xcache.new(s, { slots = n, swap = 1 << 30, halflife = 3, margin = 0 })
	local got = rows(s)
	local e = 0
	for i = 1, #want do
		e = math.max(e, maxdiff(want[i], got[i]))
	end
	local h, mi = xc:stats()
	ok(e < 1e-4 and h > 0 and mi > 0 and xc.moved > xc.bytes // 2,
	    ("expert cache of %d slots: logits equal one backend (max error %.3g, %d of %d picks held)")
	    :format(n, e, h, h + mi))
	s:close()
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
ok(m.hp.hc == 4 and m.hp.idx and m.hp.idx.ratio == 4 and m.hp.ple and m.hp.ple.layer == 3 and m.ngram,
    "config: streams, indexer, n-gram layer and reader")

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
reader_check(m, path)
placement_check(path)
xcache_check(m, path)
dense_check(m, path)
split_check(m)

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
