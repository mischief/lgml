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
	{ id = 38, best = 191, v = { -0.066053, 0.0404824, 0.0718477, -0.755605, 0.483435, -0.233501, -0.444522, -0.664737 } },
	{ id = 75, best = 252, v = { -0.0422199, 0.233823, -0.111078, 0.618722, 0.413342, 0.660222, -0.239713, -0.576367 } },
	{ id = 112, best = 149, v = { 0.375609, -0.191562, -0.277492, 0.493639, -0.208398, 0.237206, 0.940699, -0.423209 } },
	{ id = 149, best = 97, v = { -0.155266, 0.00407636, 0.402979, -0.625292, 0.0494761, 0.250874, -0.211971, -0.357204 } },
	{ id = 186, best = 153, v = { 0.503083, -0.920129, -0.191841, -0.178401, -0.29891, -0.745663, -0.0774316, -0.26852 } },
	{ id = 223, best = 196, v = { -0.0135536, -0.777598, -0.470715, 0.61066, 0.514264, -0.146679, 0.305918, 0.241602 } },
	{ id = 10, best = 89, v = { -0.277122, -0.0354819, 0.0351709, -0.0830715, 0.132121, 0.335121, 0.476011, -0.17978 } },
	{ id = 47, best = 23, v = { -0.530516, 0.105573, 0.113999, -0.247637, 0.269452, -0.396033, 0.217785, -0.354123 } },
	{ id = 84, best = 23, v = { -0.49654, -0.496002, -0.0338996, -1.0503, -0.230445, 0.331402, 0.0879837, -0.319202 } },
	{ id = 121, best = 149, v = { 0.814051, 0.110796, 0.502655, 0.603779, -0.549475, 0.638299, 0.474585, -0.356568 } },
	{ id = 158, best = 109, v = { -0.252446, 0.204021, -0.253492, -0.30979, -0.342275, -0.235702, 0.268423, -0.0558141 } },
	{ id = 195, best = 195, v = { 0.12066, 0.673647, -0.218135, -0.474301, 0.0319072, 0.0161757, -0.306813, 0.170643 } },
	{ id = 232, best = 106, v = { 0.468507, 0.599799, 0.386652, -0.353149, 0.128299, -0.117331, -0.263763, 0.0172342 } },
	{ id = 19, best = 97, v = { -0.0613828, 0.147208, -0.363486, -0.227451, -0.442929, 0.325824, -0.281608, -0.0308029 } },
	{ id = 56, best = 40, v = { 0.0103642, 0.28251, 0.507246, -0.738818, 0.133547, -0.104321, -0.833345, 0.147315 } },
	{ id = 93, best = 249, v = { -0.124721, 0.134113, 0.728842, 0.00799328, -0.591663, 0.325181, 0.761024, -0.215061 } },
	{ id = 130, best = 159, v = { -0.380769, 0.326567, -0.972106, -0.538019, -0.479717, 0.212226, 1.09559, -0.0986236 } },
	{ id = 167, best = 129, v = { -0.727051, 0.338787, -0.0394685, -0.00809036, -0.0815548, -0.00819725, 0.270041, 0.699457 } },
	{ id = 204, best = 100, v = { -0.145925, 0.225294, 0.107152, 1.30027, 0.30593, 0.785685, 0.338082, -0.373516 } },
	{ id = 241, best = 31, v = { -0.320345, -0.427001, -0.974319, -1.09719, -0.0630217, -0.556771, 0.321392, 0.0769184 } },
	{ id = 28, best = 26, v = { -0.456175, -0.367333, 0.448745, -0.646667, -0.218312, 0.225924, -0.326238, -0.165188 } },
	{ id = 65, best = 26, v = { -0.621827, -0.102625, 0.265011, -1.13397, -0.804422, -0.0513938, 0.265232, -0.170688 } },
	{ id = 102, best = 43, v = { 0.0706005, 0.485359, -0.0268618, -0.780227, -0.194255, 0.049995, -0.254304, -0.559566 } },
	{ id = 139, best = 228, v = { 0.00752535, -0.0900558, 0.0202943, -0.340202, 0.121993, 0.122645, 0.252793, 0.262955 } },
	{ id = 256, best = 92, v = { 0.39692, -0.0124767, -1.08372, 0.54766, -0.0147957, -0.0492137, 0.484607, -0.366773 } },
	{ id = 32, best = 97, v = { 0.0178158, -0.277654, 0.229508, -0.457195, -0.191695, -0.297265, -0.464147, 0.479788 } },
	{ id = 119, best = 109, v = { 0.229959, 0.604388, 0.0286461, 0.324793, -0.356257, 0.408643, 0.182236, 0.299813 } },
	{ id = 258, best = 189, v = { 0.0887312, -0.00746047, 0.362266, 0.0533178, -0.204473, 0.221362, 0.298663, 0.52609 } },
	{ id = 40, best = 105, v = { -0.909396, -0.179947, 0.517819, 0.0565982, -0.851553, 0.0939627, -0.807512, -0.0716568 } },
	{ id = 41, best = 124, v = { 0.490478, -0.498995, 0.162932, 0.563477, -0.330052, 0.21321, 0.279635, -0.0689117 } },
	{ id = 42, best = 173, v = { -0.835305, 0.185178, 0.163385, 0.279373, -0.283209, -0.247289, 0.313504, 0.258455 } },
	{ id = 43, best = 120, v = { -0.0290957, -0.257228, 0.497023, 0.552503, 0.497157, -0.116789, -0.174685, -0.477844 } },
	{ id = 44, best = 142, v = { 0.0187313, 0.0550146, -0.122975, -0.654086, -0.279922, -0.402282, 0.685639, -0.197636 } },
	{ id = 45, best = 106, v = { -0.265432, 0.102407, 0.093007, -1.02676, -0.557284, -0.546867, -0.395238, 0.0612167 } },
	{ id = 46, best = 232, v = { -0.158987, 0.965805, 0.103424, -0.246389, -0.198344, 0.97687, 0.258513, -0.233639 } },
	{ id = 47, best = 235, v = { -0.327859, 0.323982, -0.0475424, -0.490603, 0.471747, 0.558791, 0.749569, 0.0698425 } },
	{ id = 235, best = 169, v = { -0.725651, 0.0936199, 0.408984, -0.801401, 0.565206, 0.40449, 0.349559, 0.342389 } },
	{ id = 169, best = 232, v = { -0.517938, 0.537096, -0.164898, -0.551653, -0.642301, 0.412706, -0.319424, -0.580011 } },
	{ id = 232, best = 233, v = { 0.151068, 0.674724, -0.257769, -0.86575, -0.308786, -0.117531, -0.696984, -0.263156 } },
	{ id = 233, best = 43, v = { -0.0608659, 0.383877, -0.170123, -0.200154, 0.411339, -0.0687611, 0.019801, 0.303679 } },
	{ id = 43, best = 154, v = { -0.275166, -0.154373, -0.320011, -0.360061, -0.0324438, 0.436773, 0.784104, -0.237072 } },
	{ id = 154, best = 25, v = { 0.707852, -0.595283, -0.64348, -0.0504611, -0.105531, -0.365837, 1.01399, 0.256184 } },
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
