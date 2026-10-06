-- LLaDA-MoE-v2 on a tiny random model. Output is TAP. The forward is
-- checked against plain Lua math of the reference model; the sampler
-- against a plain Lua copy of the reference README sampler that reads
-- the same logits. LGML_DEVICE picks the backend (default CPU).
local lgml = require("lgml")
local uv = require("luv")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local tiny = require("llada_tiny")
local llada = require("lgml.llada-moe-v2")

local DEVICE = os.getenv("LGML_DEVICE") or "CPU"
-- Flash attention takes K and V in f16.
local TOL = DEVICE == "CPU" and 2e-3 or 3e-2
local TOL_FA = DEVICE == "CPU" and 1e-2 or 3e-2

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
	return cond
end

local function same(a, b)
	if #a ~= #b then
		return false
	end
	for i = 1, #a do
		if a[i] ~= b[i] then
			return false
		end
	end
	return true
end

-- Plain Lua forward of the reference model. Returns logits rows.
local function ref_forward(W, ids)
	local hp = tiny.hp
	local E, H, HKV, D, X, K = hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_expert, hp.n_used
	local V, N = tiny.n_vocab, #ids
	local function mv(w, rows, cols, x, r0)
		local y = {}
		r0 = r0 or 0
		for o = 1, rows do
			local s, base = 0, (r0 + o - 1) * cols
			for i = 1, cols do
				s = s + w[base + i] * x[i]
			end
			y[o] = s
		end
		return y
	end
	local function rms(x, w, off, len)
		off, len = off or 0, len or #x
		local s = 0
		for i = 1, len do
			s = s + x[off + i] ^ 2
		end
		local r = 1 / math.sqrt(s / len + hp.eps)
		for i = 1, len do
			x[off + i] = x[off + i] * r * w[i]
		end
	end
	local function norm(x, w)
		local y = table.move(x, 1, #x, 1, {})
		rms(y, w)
		return y
	end
	local function rope(v, heads, pos)
		for h = 0, heads - 1 do
			for i = 0, D // 2 - 1 do
				local th = pos * hp.base ^ (-2 * i / D)
				local a, b = v[h * D + i + 1], v[h * D + i + D // 2 + 1]
				v[h * D + i + 1] = a * math.cos(th) - b * math.sin(th)
				v[h * D + i + D // 2 + 1] = a * math.sin(th) + b * math.cos(th)
			end
		end
	end
	local function silu(x)
		return x / (1 + math.exp(-x))
	end
	local function swiglu(gw, uw, dw, F, x, r0)
		local g = mv(gw, F, E, x, r0 and r0 * F)
		local u = mv(uw, F, E, x, r0 and r0 * F)
		for i = 1, F do
			g[i] = silu(g[i]) * u[i]
		end
		return mv(dw, E, F, g, r0 and r0 * E)
	end
	local x = {}
	for t = 1, N do
		x[t] = table.move(W["token_embd.weight"], ids[t] * E + 1, ids[t] * E + E, 1, {})
	end
	for il = 0, hp.n_layer - 1 do
		local p = "blk." .. il .. "."
		local q, k, v = {}, {}, {}
		for t = 1, N do
			local h = norm(x[t], W[p .. "attn_norm.weight"])
			q[t] = mv(W[p .. "attn_q.weight"], H * D, E, h)
			k[t] = mv(W[p .. "attn_k.weight"], HKV * D, E, h)
			v[t] = mv(W[p .. "attn_v.weight"], HKV * D, E, h)
			for hh = 0, H - 1 do
				rms(q[t], W[p .. "attn_q_norm.weight"], hh * D, D)
			end
			for hh = 0, HKV - 1 do
				rms(k[t], W[p .. "attn_k_norm.weight"], hh * D, D)
			end
			rope(q[t], H, t - 1)
			rope(k[t], HKV, t - 1)
		end
		for t = 1, N do
			local o = {}
			for hh = 0, H - 1 do
				local kh = hh // (H // HKV)
				local sc, mx = {}, -math.huge
				for s = 1, N do
					local d = 0
					for i = 1, D do
						d = d + q[t][hh * D + i] * k[s][kh * D + i]
					end
					sc[s] = d / math.sqrt(D)
					mx = math.max(mx, sc[s])
				end
				local z = 0
				for s = 1, N do
					sc[s] = math.exp(sc[s] - mx)
					z = z + sc[s]
				end
				for i = 1, D do
					local a = 0
					for s = 1, N do
						a = a + sc[s] / z * v[s][kh * D + i]
					end
					o[hh * D + i] = a
				end
			end
			local y = mv(W[p .. "attn_output.weight"], E, H * D, o)
			for i = 1, E do
				x[t][i] = x[t][i] + y[i]
			end
		end
		for t = 1, N do
			local h = norm(x[t], W[p .. "ffn_norm.weight"])
			local lg = mv(W[p .. "ffn_gate_inp.weight"], X, E, h)
			local order = {}
			for e = 1, X do
				order[e] = e
			end
			table.sort(order, function(a, b) return lg[a] > lg[b] end)
			local mx, z, wts = lg[order[1]], 0, {}
			for j = 1, K do
				wts[j] = math.exp(lg[order[j]] - mx)
				z = z + wts[j]
			end
			local out = swiglu(W[p .. "ffn_gate_shexp.weight"], W[p .. "ffn_up_shexp.weight"],
			    W[p .. "ffn_down_shexp.weight"], hp.n_ff_shexp, h)
			for j = 1, K do
				local e = order[j] - 1
				local y = swiglu(W[p .. "ffn_gate_exps.weight"], W[p .. "ffn_up_exps.weight"],
				    W[p .. "ffn_down_exps.weight"], hp.n_ff_exp, h, e)
				for i = 1, E do
					out[i] = out[i] + wts[j] / z * hp.scale * y[i]
				end
			end
			for i = 1, E do
				x[t][i] = x[t][i] + out[i]
			end
		end
	end
	local logits = {}
	for t = 1, N do
		logits[t] = mv(W["output.weight"], V, E, norm(x[t], W["output_norm.weight"]))
	end
	return logits
end

-- The largest |a - b| over rows r0+1 .. r0+C of ref against flat got.
local function maxdiff(ref, got, r0, C)
	local V, m = tiny.n_vocab, 0
	for r = 1, C do
		for j = 1, V do
			m = math.max(m, math.abs(ref[r0 + r][j] - got[(r - 1) * V + j]))
		end
	end
	return m
end

-- detect reads only files named .gguf.
local path = os.tmpname()
os.remove(path)
path = path .. ".gguf"
local W = tiny.write(path)
local be = assert(lgml.backend(DEVICE))
local MASK = tiny.id("<|mask|>")

-- Forward against the plain Lua model, with and without flash attention.
local ids = { 72, 101, 108, MASK, MASK, 111, MASK }
local ref = ref_forward(W, ids)
for _, fa in ipairs({ false, true }) do
	local m = llada.load(be, path, { flash_attn = fa })
	ok(m.hp.mask_id == MASK and m.hp.canvas == 64 and m.hp.expert_scale == 2.5, "hparams")
	local d = maxdiff(ref, m:logits(ids), 0, #ids)
	ok(d < (fa and TOL_FA or TOL), ("logits match plain Lua, flash %s (max diff %.2g)"):format(fa, d))
	d = maxdiff(ref, m:logits(ids, 3, 3), 3, 3)
	ok(d < (fa and TOL_FA or TOL), ("logits rows 4 to 6 match, flash %s (max diff %.2g)"):format(fa, d))
	m:close()
end

ok(require("lgml.detect").engine_of({ model = path }) == "lgml.diffusion_engine", "detect picks the diffusion engine")

-- bf16 rounding and the transfer schedule.
ok(llada.bf16(1.0) == 1.0 and llada.bf16(0.998046875) == 1.0 and llada.bf16(0.998) < 1.0 and
    llada.bf16(1 + 2 ^ -8) == 1.0 and llada.bf16(1 + 3 * 2 ^ -8) == 1 + 2 ^ -6, "bf16 rounds to nearest even")
ok(same(llada.schedule(10, 4), { 3, 3, 2, 2 }) and same(llada.schedule(64, 64), (function()
	local t = {}
	for i = 1, 64 do
		t[i] = 1
	end
	return t
end)()) and same(llada.schedule(3, 5), { 1, 1, 1, 0, 0 }), "schedule as get_num_transfer_tokens")
ok(llada.options({ threshold = 1.0, steps = 8, block_length = 64 }) == nil, "steps * minimal_topk must cover a block")
ok(llada.options({ threshold = false, steps = 8, block_length = 64 }) ~= nil, "a schedule takes any steps")
ok(llada.options({ top_k = 3 }) == nil and llada.options({ temperature = -1 }) == nil, "bad options")

-- The README selection, written over the whole block as the reference
-- does: confidence -inf off the mask, top k, keep >= threshold.
local function ref_select(block, conf, threshold, minimal_topk, scheduled)
	local C, full, k = #block, {}, 0
	for i = 1, C do
		full[i] = block[i] == MASK and conf[i] or -math.huge
		k = k + (block[i] == MASK and 1 or 0)
	end
	if threshold then
		k = k
	else
		k = scheduled
	end
	if k == 0 then
		return {}
	end
	local idx = {}
	for i = 1, C do
		idx[i] = i
	end
	table.sort(idx, function(a, b)
		return full[a] > full[b] or (full[a] == full[b] and a < b)
	end)
	local sel = {}
	for j = 1, k do
		if not threshold or j <= math.min(minimal_topk, k) or full[idx[j]] >= threshold then
			sel[#sel + 1] = idx[j]
		end
	end
	return sel
end

do
	local seed, good = 99, true
	local function rnd()
		seed = (seed * 1103515245 + 12345) % 2147483648
		return seed / 2147483648
	end
	for trial = 1, 300 do
		local C = 1 + trial % 16
		local block, conf, masked = {}, {}, {}
		for i = 1, C do
			block[i] = rnd() < 0.6 and MASK or 5
			-- Coarse values give ties.
			conf[i] = math.floor(rnd() * 8) / 8
			if block[i] == MASK then
				masked[#masked + 1] = i
			end
		end
		local th = ({ 1.0, 0.5, 0.25, false })[trial % 4 + 1]
		-- A schedule never asks for more than the masks left.
		local mt, sch = 1 + trial % 3, math.min(trial % 5, #masked)
		local a = llada.select(masked, conf, th, mt, sch)
		local b = ref_select(block, conf, th, mt, sch)
		if not same(a, b) then
			good = false
			io.stderr:write(("trial %d: %s vs %s\n"):format(trial, table.concat(a, ","), table.concat(b, ",")))
		end
	end
	ok(good, "select matches the README selection on 300 random cases")
end

-- The README sampler over one block, on logits from the model. With
-- temperature 0 it is deterministic.
local function ref_block(m, prefix, o)
	local P, C, V = #prefix, o.block_length, tiny.n_vocab
	local x = table.move(prefix, 1, P, 1, {})
	for i = 1, C do
		x[P + i] = MASK
	end
	local sched = llada.schedule(C, o.steps)
	local steps = 0
	for step = 1, o.steps do
		local any = false
		for i = 1, C do
			any = any or x[P + i] == MASK
		end
		if not any then
			break
		end
		local lg = m:logits(x, P, C)
		steps = steps + 1
		local block, conf = {}, {}
		local x0 = {}
		for i = 1, C do
			block[i] = x[P + i]
			local best, bv = 0, -math.huge
			for j = 1, V do
				if lg[(i - 1) * V + j] > bv then
					best, bv = j - 1, lg[(i - 1) * V + j]
				end
			end
			local z = 0
			for j = 1, V do
				z = z + math.exp(lg[(i - 1) * V + j] - bv)
			end
			x0[i], conf[i] = best, llada.bf16(1 / z)
		end
		for _, i in ipairs(ref_select(block, conf, o.threshold, o.minimal_topk, sched[step])) do
			x[P + i] = x0[i]
		end
	end
	return table.move(x, P + 1, P + C, 1, {}), steps
end

do
	local p2 = os.tmpname()
	local W2 = tiny.write(p2, { out_gain = 30 })
	local m = llada.load(be, p2, { flash_attn = DEVICE ~= "CPU" })
	local prefix = { 72, 101, 108, 108, 111 }
	local cases = {
		{ threshold = 1.0, steps = 8, block_length = 8, minimal_topk = 1 },
		{ threshold = 0.9, steps = 8, block_length = 8, minimal_topk = 1 },
		{ threshold = 0.5, steps = 4, block_length = 8, minimal_topk = 2 },
		{ threshold = false, steps = 3, block_length = 8, minimal_topk = 1 },
		{ threshold = false, steps = 12, block_length = 8, minimal_topk = 1 },
	}
	local first
	for _, o in ipairs(cases) do
		local want, ws = ref_block(m, prefix, o)
		local got, gs = m:denoise(prefix, o)
		first = first or gs
		ok(same(got, want) and gs == ws, ("denoise matches the README sampler, threshold %s, steps %d (%d steps)")
		    :format(tostring(o.threshold), o.steps, gs))
	end
	-- Fewer steps than tokens: some steps unmask more than the minimum.
	ok(first > 1 and first < 8, "threshold 1.0 unmasks several tokens in a step (" .. first .. " steps)")
	-- Temperature: seeded noise repeats, and the block has no mask left.
	local a = m:denoise(prefix, { temperature = 1, threshold = 0.5, steps = 8, block_length = 8, seed = 3 })
	local b = m:denoise(prefix, { temperature = 1, threshold = 0.5, steps = 8, block_length = 8, seed = 3 })
	local masks = 0
	for _, t in ipairs(a) do
		masks = masks + (t == MASK and 1 or 0)
	end
	ok(same(a, b) and masks == 0, "temperature runs repeat with a seed")
	m:close()
	os.remove(p2)
	local _ = W2
end

-- The engine against the README generate loop, with eos early stop on
-- <|role_end|>. Its output row is a scaled copy of the row of a token
-- that the model makes in the third block, so it wins there.
do
	local p3 = os.tmpname()
	local END = tiny.id("<|role_end|>")
	tiny.write(p3, { out_gain = 4, copy = { END, 153, 1.1 } })
	local engine = require("lgml.diffusion_engine")
	local e = engine.new({ model = p3, backend = DEVICE, n_ctx = 256, seed = 1,
		flash_attn = DEVICE ~= "CPU", sampler = { block_length = 8, steps = 8 } })
	require("lgml.server.engine").check(e)
	local r = { text = {} }
	function r.content(_, t)
		r.text[#r.text + 1] = t
		return true
	end
	r.reasoning = r.content
	function r.tool_call()
		return 0
	end
	function r.tool_arguments()
		return true
	end
	function r.finish(_, reason, usage, timings)
		r.reason, r.usage, r.timings = reason, usage, timings
	end
	function r.cancelled()
		return false
	end
	local function chat(extra)
		r.text = {}
		local req = { messages = { { role = "user", content = "hi" } }, max_tokens = 40 }
		for k, v in pairs(extra or {}) do
			req[k] = v
		end
		local done, res, err, status
		coroutine.wrap(function()
			res, err, status = e:chat(req, r)
			done = true
		end)()
		while not done do
			uv.run("once")
		end
		return res, err, status
	end

	-- The README generate: whole blocks, stop at the first end token.
	local tk = e.tk
	local prompt = tk:encode(e.family.render({ { role = "user", content = "hi" } }, { add_generation_prompt = true }))
	local o = { threshold = 1.0, steps = 8, block_length = 8, minimal_topk = 1 }
	local x, gen, stop = table.move(prompt, 1, #prompt, 1, {}), {}, false
	while #gen < 40 and not stop do
		local blk = ref_block(e.model, x, o)
		for _, t in ipairs(blk) do
			x[#x + 1] = t
			if t == END or e.eog[t] then
				stop = true
				break
			end
			gen[#gen + 1] = t
		end
	end
	local res, err = chat()
	ok(res == true, "engine chat returns true " .. tostring(err))
	-- The decoder cleans bad UTF-8; the hermes parser drops trailing space.
	local want = tk:decode(table.move(gen, 1, math.min(#gen, 40), 1, {}), { special = false })
	want = require("lgml.tokenizer").clean(want):gsub("%s+$", "")
	ok(table.concat(r.text) == want, ("engine text matches the README generate (%d tokens, stop %s)")
	    :format(#gen, tostring(stop)))
	ok(stop and r.reason == "stop" and r.usage.completion_tokens == #gen + 1,
	    "the end token stops the reply: " .. tostring(r.reason) .. " " .. tostring(r.usage.completion_tokens))
	ok(r.timings.blocks_n >= 1 and r.timings.steps_n >= r.timings.blocks_n, "timings count blocks and steps")

	local _, msg, status = chat({ diffusion = { steps = 1 } })
	ok(status == 400 and msg:find("cover one block", 1, true), "bad diffusion options give 400")
	_, msg, status = chat({ top_k = 3 })
	ok(status == 400 and msg:find("top_k", 1, true), "top_k gives 400")
	res = chat({ temperature = 0.7, seed = 5, diffusion = { threshold = false, steps = 4 } })
	ok(res == true and r.usage.completion_tokens >= 1, "temperature and a schedule run")
	e.model:close()
	os.remove(p3)
end

os.remove(path)
print("1.." .. n)
os.exit(failed and 1 or 0)
