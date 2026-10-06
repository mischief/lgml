-- Speculative decoding on tiny models: greedy replies match plain
-- decoding, rejected guesses leave the cache as plain decoding does
-- (attention, delta net and ring models), and sampled replies keep
-- their distribution. Output is TAP. The engines run without flash
-- attention: its CPU kernel gives slightly other logits for a pass of
-- several rows than for one row, and these comparisons are exact.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local uv = require("luv")
local engine = require("lgml.engine")
local spec = require("lgml.spec")
local tiny = require("tiny")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

-- One request through the engine; the reply text, usage and timings.
local function chat(e, req)
	local r = { text = {}, lp = {} }
	r.sink = {
		content = function(_, t, lp)
			r.text[#r.text + 1] = t
			for _, x in ipairs(lp or {}) do
				r.lp[#r.lp + 1] = x
			end
			return true
		end,
		reasoning = function(_, t)
			r.text[#r.text + 1] = t
			return true
		end,
		tool_call = function() return 0 end,
		tool_arguments = function() return true end,
		finish = function(_, reason, usage, timings)
			r.reason, r.usage, r.timings = reason, usage, timings
		end,
		cancelled = function() return false end,
	}
	local done
	coroutine.wrap(function()
		r.res, r.err = e:chat(req, r.sink)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	r.out = table.concat(r.text)
	return r
end

-- A greedy request; extra fields override. End tokens get a bias of
-- -100, so replies run to max_tokens.
local function req_of(e, content, extra)
	local lb = {}
	for _, id in ipairs(e.tk:special().eog or {}) do
		lb[tostring(id)] = -100
	end
	local r = { messages = { { role = "user", content = content } }, max_tokens = 120, temperature = 0,
		chat_template_kwargs = { enable_thinking = false }, logit_bias = lb }
	for k, v in pairs(extra or {}) do
		r[k] = v
	end
	return r
end

local REPEAT = ("the cat sat on the mat and the dog sat on the log "):rep(6)

-- The drafter of the next request, for engines made with `test_drafter`.
local next_drafter
local function test_drafter()
	return next_drafter
end

-- A drafter that keeps the sequence it sees and guesses nothing.
local function recorder()
	return { draft = function(self, ctx)
		self.seq = ctx.seq
		return {}
	end }
end

-- A drafter that guesses the reference reply ref (prompt and reply ids)
-- but spoils every third guess, so passes keep some guesses and reject
-- the rest at changing places.
local function oracle(ref)
	local d = { calls = 0 }
	function d:draft(ctx, k)
		local L = #ctx.seq
		local out = {}
		for j = 1, k do
			local id = ref[L + j]
			if not id then
				break
			end
			self.calls = self.calls + 1
			out[j] = self.calls % 3 == 0 and (id + 1) % 250 or id
		end
		return out
	end
	return d
end

-- 1. Prompt lookup on a dense model: greedy replies match.
local qpath = tiny.tmp(".gguf")
tiny.write(qpath, "qwen3", { n_ctx = 2048 })
for _, slots in ipairs({ 1, 2 }) do
	local e = engine.new({ model = qpath, backend = "CPU", n_ctx = 2048, slots = slots, kv_initial = 256,
		flash_attn = false, speculative = true })
	local tag = ("qwen3 slots %d: "):format(slots)
	local plain = chat(e, req_of(e, REPEAT, { speculative = false }))
	local sp = chat(e, req_of(e, REPEAT))
	ok(plain.res and sp.res and sp.out == plain.out, tag .. "greedy reply matches plain decoding")
	ok(sp.usage.completion_tokens == plain.usage.completion_tokens and
	    sp.timings.predicted_n == sp.usage.completion_tokens, tag .. "token counts match")
	ok(plain.timings.draft_n == nil and sp.timings.draft_accepted > 0 and
	    sp.timings.draft_accepted <= sp.timings.draft_n,
	    ("%sguesses kept (%s of %s)"):format(tag, sp.timings.draft_accepted, sp.timings.draft_n))
	-- The cache after a speculative reply serves the next turn as a
	-- fresh engine computes it.
	local turn2 = { { role = "user", content = REPEAT }, { role = "assistant", content = sp.out },
		{ role = "user", content = "again" } }
	local fresh = engine.new({ model = qpath, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256,
		flash_attn = false })
	local a = chat(e, req_of(e, nil, { messages = turn2 }))
	local b = chat(fresh, req_of(fresh, nil, { messages = turn2, speculative = false }))
	ok(a.out == b.out and a.usage.cached_tokens > 0,
	    ("%snext turn matches a fresh engine (cached %d)"):format(tag, a.usage.cached_tokens))
end

-- 2. Penalties, a grammar, logprobs and stop strings act on each row
-- as on a decode step.
do
	local e = engine.new({ model = qpath, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256,
		flash_attn = false, speculative = { max = 4, adaptive = false } })
	local function same(name, extra)
		local x = {}
		for k, v in pairs(extra) do
			x[k] = v
		end
		x.speculative = false
		local plain = chat(e, req_of(e, REPEAT, x))
		x.speculative = nil
		local sp = chat(e, req_of(e, REPEAT, x))
		ok(plain.res and sp.res and sp.out == plain.out and sp.reason == plain.reason and
		    sp.timings.draft_n > 0, ("%s: reply matches (%d of %d guesses kept)"):format(name,
		    sp.timings.draft_accepted, sp.timings.draft_n))
		return plain, sp
	end
	same("penalties", { repeat_penalty = 1.3, presence_penalty = 0.4, frequency_penalty = 0.2 })
	same("grammar", { grammar = 'root ::= ("the " | "cat " | "sat " | "on " | "mat ")+' })
	local plain, sp = same("logprobs", { logprobs = true, top_logprobs = 2 })
	local d = 0
	for i, l in ipairs(plain.lp) do
		local m = sp.lp[i]
		d = math.max(d, m and math.abs(m.logprob - l.logprob) or math.huge)
	end
	ok(#plain.lp == #sp.lp and #plain.lp > 0 and d < 1e-3,
	    ("logprobs match (max difference %.3g)"):format(d))
	local stop = plain.out:sub(31, 34)
	same("stop string", { stop = { stop } })
	same("max_tokens", { max_tokens = 7 })
end

-- The repeat guard counts kept guesses too: a reply that repeats one
-- token ends at the same token as plain decoding. Guesses stop before
-- the reasoning budget, and the wrap-up text runs plainly.
do
	local e = engine.new({ model = qpath, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256,
		flash_attn = false, repeat_stop_tokens = 20, speculative = { max = 8, adaptive = false } })
	local lb = { ["120"] = 100 }
	local plain = chat(e, req_of(e, REPEAT, { logit_bias = lb, speculative = false }))
	local sp = chat(e, req_of(e, REPEAT, { logit_bias = lb }))
	ok(plain.timings.repeat_stop == 1 and sp.timings.repeat_stop == 1 and sp.out == plain.out and
	    sp.usage.completion_tokens == 20 and sp.timings.draft_accepted > 0,
	    ("repeat guard: the reply ends after 20 tokens (%d of %d guesses kept)"):format(
	    sp.timings.draft_accepted or 0, sp.timings.draft_n or 0))
	local rec = recorder()
	local d = engine.new({ model = qpath, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256,
		flash_attn = false, speculative = { max = 8, adaptive = false, drafter = test_drafter } })
	next_drafter = rec
	-- A bias on <think> keeps the model in its reasoning block.
	local tid
	for id = 0, d.tk:n_vocab() - 1 do
		tid = tid or (d.tk:token(id) == "<think>" and id or nil)
	end
	local think = { chat_template_kwargs = { enable_thinking = true }, reasoning_budget = 12, max_tokens = 80,
		logit_bias = { [tostring(tid)] = 100 } }
	local want = chat(d, req_of(d, REPEAT, think))
	local od = oracle(rec.seq)
	next_drafter = od
	local got = chat(d, req_of(d, REPEAT, think))
	ok(got.out == want.out and got.usage.reasoning_tokens == want.usage.reasoning_tokens and
	    got.usage.reasoning_tokens >= 12 and got.timings.draft_accepted > 0, ("reasoning budget: the reply matches (%d reasoning tokens, %d of %d kept)"):format(
	    got.usage.reasoning_tokens, got.timings.draft_accepted, got.timings.draft_n))
end

-- 3. Rejected guesses roll back: an oracle drafter with spoiled guesses
-- on a dense model, a delta net hybrid and a sliding ring model whose
-- reply wraps the ring.
local gpath, hpath = tiny.tmp(".gguf"), tiny.tmp(".gguf")
tiny.write_gemma4(gpath, { n_ctx = 2048 })
tiny.write(hpath, "qwen3next", { n_ctx = 2048 })
for _, m in ipairs({ { "qwen3", qpath, 200 }, { "qwen3next", hpath, 200 }, { "gemma4", gpath, 700 } }) do
	local tag = m[1] .. " rollback: "
	local function mk(slots)
		return engine.new({ model = m[2], backend = "CPU", n_ctx = 2048, slots = slots, kv_initial = 256,
			flash_attn = false, speculative = { max = 4, adaptive = false, drafter = test_drafter } })
	end
	local e = mk(1)
	local rec = recorder()
	next_drafter = rec
	local plain = chat(e, req_of(e, REPEAT, { max_tokens = m[3] }))
	local ref = rec.seq
	for _, slots in ipairs({ 1, 2 }) do
		local es = mk(slots)
		local od = oracle(ref)
		next_drafter = od
		local sp = chat(es, req_of(es, REPEAT, { max_tokens = m[3] }))
		local t = sp.timings
		ok(sp.res and sp.out == plain.out and t.draft_accepted > 0 and t.draft_accepted < t.draft_n,
		    ("%sslots %d: reply matches (%d of %d guesses kept)"):format(tag, slots, t.draft_accepted,
		    t.draft_n))
		local turn2 = { { role = "user", content = REPEAT }, { role = "assistant", content = sp.out },
			{ role = "user", content = "again" } }
		next_drafter = recorder()
		local a = chat(es, req_of(es, nil, { messages = turn2, max_tokens = 40 }))
		local fresh = mk(1)
		local b = chat(fresh, req_of(fresh, nil, { messages = turn2, max_tokens = 40, speculative = false }))
		ok(a.out == b.out, ("%sslots %d: next turn matches a fresh engine (cached %d)"):format(tag, slots,
		    a.usage.cached_tokens))
	end
end

-- A drafter with hidden = true gets the state after the last layer at
-- the row of the last token, as a model-native draft head needs.
do
	local e = engine.new({ model = qpath, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256,
		flash_attn = false, speculative = { max = 4, adaptive = false, drafter = test_drafter } })
	local rec = recorder()
	next_drafter = rec
	local want = chat(e, req_of(e, REPEAT, { max_tokens = 40 })).out
	local ref = rec.seq
	local seen, good = 0, true
	local d = oracle(ref)
	d.hidden = true
	local draft = d.draft
	function d:draft(ctx, k)
		if ctx.hidden then
			local sh = ctx.hidden:shape()
			seen = seen + 1
			good = good and sh[2] == tiny.E and ctx.hidden_row >= 1 and ctx.hidden_row <= sh[1]
		end
		return draft(self, ctx, k)
	end
	next_drafter = d
	local r = chat(e, req_of(e, REPEAT, { max_tokens = 40 }))
	ok(r.out == want and seen >= 10 and good, ("hidden states reach the drafter (%d steps)"):format(seen))
end

-- A request that speculates alone goes on in batched passes when a
-- second request comes; kept tokens that wait for a pass go with it.
do
	local function mk(slots)
		return engine.new({ model = hpath, backend = "CPU", n_ctx = 2048, slots = slots, kv_initial = 256,
			flash_attn = false, speculative = { max = 4, adaptive = false, drafter = test_drafter } })
	end
	local one = mk(1)
	local rec = recorder()
	next_drafter = rec
	local want_a = chat(one, req_of(one, REPEAT, { max_tokens = 150 })).out
	local ref = rec.seq
	next_drafter = recorder()
	local want_b = chat(one, req_of(one, "a different prompt here", { max_tokens = 60 })).out
	local e = mk(2)
	local steps, done = 0, 0
	next_drafter = oracle(ref)
	local function start(req)
		local r = { text = {} }
		local sink = {
			content = function(_, t)
				r.text[#r.text + 1] = t
				steps = steps + 1
				return true
			end,
			reasoning = function() return true end,
			tool_call = function() return 0 end,
			tool_arguments = function() return true end,
			finish = function(_, _, _, timings) r.timings = timings end,
			cancelled = function() return false end,
		}
		coroutine.wrap(function()
			r.res = e:chat(req, sink)
			done = done + 1
		end)()
		return r
	end
	local ra = start(req_of(e, REPEAT, { max_tokens = 150 }))
	while steps < 30 do
		uv.run("once")
	end
	next_drafter = recorder()
	local rb = start(req_of(e, "a different prompt here", { max_tokens = 60 }))
	while done < 2 do
		uv.run("once")
	end
	ok(table.concat(ra.text) == want_a and table.concat(rb.text) == want_b and ra.timings.draft_n > 0,
	    ("qwen3next: a second request joins a speculating one; both replies match (%d of %d kept)"):format(
	    ra.timings.draft_accepted, ra.timings.draft_n))
end

-- 4. Sampled replies keep their distribution. The guesses follow the
-- greedy reply. The token counts at each position, of plain replies and
-- of speculative replies with other seeds, must pass a chi-square test
-- of homogeneity at p 0.001. Tokens with fewer than 10 counts in all
-- share one class.
do
	local e = engine.new({ model = qpath, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256,
		flash_attn = false, speculative = { max = 4, adaptive = false, drafter = test_drafter } })
	local POS, N = 6, 300
	local rec = recorder()
	next_drafter = rec
	chat(e, req_of(e, REPEAT, { max_tokens = POS }))
	local ref = rec.seq
	local P = #ref - POS
	-- The reply ids of seeds first .. first + N - 1, and their counts at
	-- each position. With guess true, the drafter guesses ref.
	local function run(first, guess)
		local c, outs = {}, {}
		for i = 1, POS do
			c[i] = {}
		end
		for seed = first, first + N - 1 do
			local d = { draft = function(self, ctx, k)
				self.seq = ctx.seq
				return guess and table.move(ref, #ctx.seq + 1, math.min(#ref, #ctx.seq + k), 1, {}) or {}
			end }
			next_drafter = d
			local r = chat(e, req_of(e, REPEAT, { max_tokens = POS, temperature = 1, top_k = 3, top_p = 1,
				min_p = 0, seed = seed }))
			outs[seed] = r.out
			for i = 1, POS do
				local id = d.seq and d.seq[P + i]
				if id then
					c[i][id] = (c[i][id] or 0) + 1
				end
			end
		end
		return c, outs
	end
	local cp, plain = run(1, false)
	local _, guessed = run(1, true)
	local same = 0
	for seed = 1, N do
		same = same + (plain[seed] == guessed[seed] and 1 or 0)
	end
	ok(same == N, ("sampling: a seed gives the same reply with guesses (%d of %d)"):format(same, N))
	local cs = run(N + 1, true)
	local worst, at = 0, ""
	for i = 1, POS do
		local classes = { { 0, 0 } }
		local keys = {}
		for id in pairs(cp[i]) do
			keys[id] = true
		end
		for id in pairs(cs[i]) do
			keys[id] = true
		end
		for id in pairs(keys) do
			local a, b = cp[i][id] or 0, cs[i][id] or 0
			if a + b < 10 then
				classes[1][1], classes[1][2] = classes[1][1] + a, classes[1][2] + b
			else
				classes[#classes + 1] = { a, b }
			end
		end
		local chi, df = 0, -1
		for _, c in ipairs(classes) do
			local m = (c[1] + c[2]) / 2
			if m > 0 then
				chi, df = chi + (c[1] - m) ^ 2 / m + (c[2] - m) ^ 2 / m, df + 1
			end
		end
		-- The p 0.001 point of chi-square with df degrees (Wilson-Hilferty).
		local crit = df * (1 - 2 / (9 * df) + 3.09 * math.sqrt(2 / (9 * df))) ^ 3
		if df > 0 and chi / crit > worst then
			worst, at = chi / crit, ("%.1f of %.1f, %d degrees"):format(chi, crit, df)
		end
	end
	ok(worst < 1, "sampling: token counts match plain sampling (worst chi-square " .. at .. ")")
end

-- 5. The draft length choice and the lookup drafter.
do
	local d = spec.lookup()
	local seq = { 1, 2, 3, 4, 5, 1, 2, 3 }
	local g = d:draft({ seq = seq }, 4)
	ok(table.concat(g, ",") == "4,5,1,2", "lookup: the guesses follow the earlier match")
	seq[#seq + 1] = 7
	ok(#d:draft({ seq = seq }, 4) == 0, "lookup: no match, no guesses")
	local loop = { 9, 8, 9, 8, 9 }
	ok(table.concat(spec.lookup():draft({ seq = loop }, 5), ",") == "8,9,8,9,8",
	    "lookup: a match at the end repeats with its period")
	local c = spec.costs()
	c:add(1, 0.010)
	c:add(5, 0.012)
	local s = spec.session(d, c, { max = 16 })
	s.hit, s.miss = 9, 1
	local k = s:choose(16)
	ok(k >= 4, "choose: a high acceptance rate drafts long (" .. k .. ")")
	s.hit, s.miss = 0.05, 5
	local zeros = 0
	for _ = 1, 8 do
		zeros = zeros + (s:choose(16) == 0 and 1 or 0)
	end
	ok(zeros >= 5, "choose: a low acceptance rate mostly decodes plainly, with probes (" .. zeros .. " of 8)")
	ok(s:choose(0) == 0, "choose: no room, no draft")
end

-- 6. The request field and the metrics.
do
	local cr = require("lgml.server.chat_request")
	local msgs = require("lgml.json").array({ { role = "user", content = "hi" } })
	ok(cr.check({ messages = msgs, speculative = true }) == true, "request: speculative true passes")
	ok(not cr.check({ messages = msgs, speculative = 3 }), "request: speculative must be a boolean or an object")
	ok(cr.check({ messages = msgs, speculative = { max = 2, min_p = 0.5 } }) == true,
	    "request: speculative takes max and min_p")
	ok(not cr.check({ messages = msgs, speculative = { max = 1.5 } }) and
	    not cr.check({ messages = msgs, speculative = { min_p = 2 } }) and
	    not cr.check({ messages = msgs, speculative = { other = 1 } }), "request: bad speculative objects fail")
	local metrics = require("lgml.server.metrics")
	metrics.reset()
	metrics.chat("m", { prompt_n = 1, predicted_n = 9, draft_n = 8, draft_accepted = 6 })
	local text = metrics.render()
	ok(text:find('lgml_draft_tokens_total{model="m"} 8', 1, true) and
	    text:find('lgml_draft_accepted_tokens_total{model="m"} 6', 1, true), "metrics count guesses")
end

os.remove(qpath)
os.remove(gpath)
os.remove(hpath)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
