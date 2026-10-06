-- The MTP draft head of qwen4exp on the tiny model: head logits
-- against llama.cpp, and speculation with it through the engine, where
-- greedy replies must equal plain decoding. Output is TAP.
-- LGML_LLAMA_MTP names the mtp_logits binary (scripts/mtp_logits.cpp,
-- built against llama.cpp) to compare every logit live; without it the
-- rows in REF stand in. LGML_REF_DUMP=1 prints them.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local uv = require("luv")
local lgml = require("lgml")
local engine = require("lgml.engine")
local transformer = require("lgml.transformer")
local q4 = require("lgml.qwen4exp")
local q4t = require("q4_tiny")
local tiny = require("tiny")
local gguf = require("gguf")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
	return cond
end

-- A prompt of 30 tokens: past the QSA budget of the tiny model.
local PROMPT = {}
for i = 1, 30 do
	PROMPT[i] = (i * 37) % 250 + 1
end
local STEPS = 2
local IDX = { 1, 33, 65, 101, 151, 201, 256, 259 }
-- The head logits at IDX after the prompt, then after each step on its
-- own guess, from llama.cpp on the CPU.
local REF = {}
REF.mtp = {
	{ id = 111, best = 199, v = { 0.242977, 1.00164, -0.021498, -0.0662982, 0.361919, -0.208028, 0.462854, 0.855497 } },
	{ id = 199, best = 33, v = { 0.975312, -0.257349, 0.457183, 0.0792754, -0.297274, 0.24536, 0.388522, -0.453244 } },
	{ id = 33, best = 164, v = { 0.239158, 0.00144151, 0.412074, 0.41706, 0.199649, 0.651167, 0.28299, 0.285452 } },
}

local function argmax(v)
	local b = 1
	for i = 2, #v do
		if v[i] > v[b] then
			b = i
		end
	end
	return b - 1
end

-- Rows of the mtp_logits binary: { id, logits }.
local function llama_rows(bin, path)
	local args = { bin, "-m", path, "--spec-type", "draft-mtp", "-c", "256", "-fa", "off", "-t", "2", "--",
		STEPS, table.concat(PROMPT, " ") }
	local p = assert(io.popen("nice -n 19 " .. table.concat(args, " ") .. " 2>/dev/null"))
	local rows = {}
	for line in p:lines() do
		local v = {}
		for x in line:gmatch("%S+") do
			v[#v + 1] = tonumber(x)
		end
		-- A row is a tag ("p" or "s", no number), the input id and the logits.
		if #v > 2 then
			rows[#rows + 1] = { id = math.tointeger(table.remove(v, 1)), logits = v }
		end
	end
	p:close()
	return rows
end

-- One request through the engine.
local function chat(e, req)
	local r = { text = {} }
	r.sink = {
		content = function(_, t)
			r.text[#r.text + 1] = t
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

-- A greedy request that runs to max_tokens.
local function req_of(e, content, extra)
	local lb = {}
	for _, id in ipairs(e.tk:special().eog or {}) do
		lb[tostring(id)] = -100
	end
	local r = { messages = { { role = "user", content = content } }, max_tokens = 60, temperature = 0,
		chat_template_kwargs = { enable_thinking = false }, logit_bias = lb }
	for k, v in pairs(extra or {}) do
		r[k] = v
	end
	return r
end

local path = tiny.tmp(".gguf")
local kv, ts = q4t.build()
kv[#kv + 1] = { "tokenizer.chat_template", gguf.STRING,
	"<|im_start|> <tool_call> <function= <think> tool | tojson sysns reasoning_effort" }
gguf.write(path, kv, ts)

-- 1. The head: loaded only when asked, and its logits match llama.cpp.
do
	local be <close> = assert(lgml.backend("CPU"))
	local plain = transformer.load(be, path)
	ok(plain.mtp == nil and plain.w:get("blk.4.nextn.eh_proj.weight") == nil,
	    "head: a plain load leaves the MTP layer in the file")
	plain:close()
	local m = transformer.load(be, path, { mtp = true })
	ok(m.mtp and m.mtp.il == 5, "head: mtp = true loads the MTP layer")
	local c = m:cache(256)
	require("lgml.replay").run(m, c, table.move(PROMPT, 1, #PROMPT - 1, 1, {}), { logits = "last" })
	local rows = {}
	local d = q4.mtp_draft(m, c, { PROMPT[#PROMPT] }, nil, STEPS + 1, false, rows)
	ok(#d == STEPS + 1 and c.n_past == #PROMPT - 1, "head: drafts leave the trunk's position")
	local again = q4.mtp_draft(m, c, { PROMPT[#PROMPT] }, nil, STEPS + 1)
	ok(table.concat(again, ",") == table.concat(d, ","), "head: a second draft gives the same guesses")
	local bin = os.getenv("LGML_LLAMA_MTP")
	local fh = bin and bin ~= "" and io.open(bin)
	local ref
	if fh then
		fh:close()
		ref = llama_rows(bin, path)
	else
		ref = REF.mtp
	end
	if not ref or #ref == 0 then
		n = n + 1
		print("ok " .. n .. " # SKIP no llama.cpp reference")
	else
		local err, same = 0, #ref == #rows
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
		io.stderr:write(("mtp head: %d rows, max logit error %.3g\n"):format(#ref, err))
		ok(err < 2e-3 and same, ("head: logits match llama.cpp (max error %.3g)"):format(err))
		if os.getenv("LGML_REF_DUMP") and ref[1].logits then
			print("REF.mtp = {")
			for _, r in ipairs(ref) do
				local v = {}
				for j, ix in ipairs(IDX) do
					v[j] = ("%.6g"):format(r.logits[ix])
				end
				print(("\t{ id = %d, best = %d, v = { %s } },"):format(r.id, argmax(r.logits), table.concat(v, ", ")))
			end
			print("}")
		end
	end
	m:close()
end

-- 2. Speculation through the engine: the MTP head drafts, and greedy
-- replies equal plain decoding, with one slot and with two.
local REPEAT = ("the cat sat on the mat and the dog sat on the log "):rep(4)
for _, slots in ipairs({ 1, 2 }) do
	local tag = ("engine, %d slot%s: "):format(slots, slots > 1 and "s" or "")
	-- The random head is unsure of its guesses: min_p 0 keeps them.
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = slots, kv_initial = 256,
		flash_attn = false, speculative = { max = 3, adaptive = false, min_p = 0 } })
	ok(e.model.mtp ~= nil, tag .. "speculation loads the MTP layer")
	local plain = chat(e, req_of(e, REPEAT, { speculative = false }))
	local d0, a0 = e.spec_drafted or 0, e.spec_accepted or 0
	local sp = chat(e, req_of(e, REPEAT, { speculative = true }))
	ok(plain.res and sp.res and #sp.out > 0 and sp.out == plain.out, tag .. "greedy reply equals plain decoding")
	local drafted, accepted = (e.spec_drafted or 0) - d0, (e.spec_accepted or 0) - a0
	ok(drafted > 0 and accepted >= 0 and accepted <= drafted and sp.timings.draft_n == drafted and
	    sp.timings.draft_accepted == accepted,
	    ("%sguesses counted (%d drafted, %d kept)"):format(tag, drafted, accepted))
	-- A second turn on the same slot, and a cold engine for it.
	local turn2 = { { role = "user", content = REPEAT }, { role = "assistant", content = sp.out },
		{ role = "user", content = "again: " .. REPEAT } }
	local a = chat(e, req_of(e, nil, { messages = turn2, speculative = true }))
	local fresh = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256,
		flash_attn = false, cache_ram = 0 })
	local b = chat(fresh, req_of(fresh, nil, { messages = turn2 }))
	ok(a.res and a.out == b.out, tag .. "the next turn equals a cold engine")
	-- The default gate drops the unsure guesses of the random head; the
	-- reply stays that of plain decoding.
	if slots == 1 then
		local g = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256,
			flash_attn = false, speculative = { max = 3, adaptive = false } })
		local gp = chat(g, req_of(g, REPEAT, { speculative = true }))
		ok(gp.res and gp.out == plain.out and (g.spec_drafted or 0) < drafted,
		    ("%sthe default gate drops unsure guesses (%d drafted)"):format(tag, g.spec_drafted or 0))
	end
end

-- 3. The head drafts after a rejection too: the redo tokens run with
-- the trunk rows of the pass that the state lost.
do
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256,
		flash_attn = false, speculative = { max = 3, adaptive = false, min_p = 0 } })
	local used, made = 0, nil
	local mk = e.spec_opts.drafter
	e.spec_opts.drafter = function(req)
		made = mk(req)
		local draft = made.draft
		function made.draft(self, ctx, k)
			if #ctx.seq - ctx.cache.n_past > 1 then
				used = used + 1
			end
			return draft(self, ctx, k)
		end
		return made
	end
	local plain = chat(e, req_of(e, "count: 1 2 3 4 5 6 7 8 9", { speculative = false }))
	local sp = chat(e, req_of(e, "count: 1 2 3 4 5 6 7 8 9"))
	ok(sp.out == plain.out and used > 0 and made.used > 0,
	    ("redo: greedy reply equals plain decoding; %d drafts after a rejection"):format(used))
end

os.remove(path)
print("1.." .. n)
os.exit(failed and 1 or 0)
