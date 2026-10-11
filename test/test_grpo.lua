-- lgml.grpo on a tiny qwen3 model: sampling, advantages, and short runs
-- with a toy reward. CPU only. Output is TAP.
local lgml = require("lgml")
local chat = require("lgml.chat")
local tokenizer = require("lgml.tokenizer")
local transformer = require("lgml.transformer")
local lora = require("lgml.lora")
local grpo = require("lgml.grpo")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local tiny = require("tiny")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local path = tiny.tmp(".gguf")
tiny.write(path, "qwen3")
local info = assert(lgml.gguf(path))
local family = chat.load(info.meta)
local tk = assert(tokenizer.open(path))
local be <close> = assert(lgml.backend("CPU"))
local model = transformer.load(be, path)
local msgs = { { role = "user", content = "Say something." } }

-- Sampling: the same seed gives the same answer; max_new holds.
do
	local ad = lora.new(model, { rank = 2, seed = 1 })
	model:set_adapters({ ad })
	local a1, ids1 = grpo.generate(model, tk, family, msgs, { seed = 5, max_new = 6 })
	local a2 = grpo.generate(model, tk, family, msgs, { seed = 5, max_new = 6 })
	local a3 = grpo.generate(model, tk, family, msgs, { seed = 6, max_new = 6 })
	ok(a1 == a2 and #ids1 <= 6, ("generate: the same seed gives the same answer (%q)"):format(a1))
	ok(a3 ~= a1, "generate: another seed gives another answer")
end

-- Advantages: mean 0, spread 1; equal rewards give zeros.
do
	local a, sd = grpo.advantages({ 1, 2, 3, 4 })
	local mean = (a[1] + a[2] + a[3] + a[4]) / 4
	ok(math.abs(mean) < 1e-9 and a[4] > 0 and a[1] < 0 and sd > 1, "advantages center on the group mean")
	local z = grpo.advantages({ 0.5, 0.5, 0.5 })
	ok(z[1] == 0 and z[2] == 0 and z[3] == 0, "equal rewards give no advantage")
end

-- A toy reward: 1 when the first answer token has an even id.
local function even(_, _, ids)
	return (ids[1] or 1) % 2 == 0 and 1 or 0, { parse = true }
end
local function run(kl, lr, steps, dump)
	local ad = lora.new(model, { rank = 4, seed = 2 })
	local ref = lora.new(model, { rank = 4, seed = 2 })
	local rewards, kls = {}, {}
	grpo.train({
		model = model, tk = tk, family = family, adapter = ad, ref = ref,
		prompts = { { messages = msgs } }, samples = 8, per_step = 2, steps = steps, kl = kl,
		lr = lr, max_new = 3, temperature = 1.0, clip = 1, reward = even, dump = dump,
		log = function(e)
			rewards[#rewards + 1] = e.reward
			kls[#kls + 1] = e.kl
		end,
	})
	return rewards, kls, ad
end
-- The share of 40 sampled answers that the toy reward pays.
local function share(ad)
	model:set_adapters({ ad })
	local hits = 0
	for s = 1, 40 do
		local _, ids = grpo.generate(model, tk, family, msgs, { seed = 900 + s, max_new = 1, temperature = 1.0 })
		hits = hits + even(nil, nil, ids)
	end
	return hits / 40
end
do
	-- One step raises the log-prob of answers with a positive advantage
	-- and lowers the others.
	local items = {}
	local _, _, ad = run(0, 1e-2, 1, function(r)
		items[#items + 1] = r
	end)
	model:set_adapters({ ad })
	local moved = 0
	for _, it in ipairs(items) do
		local now = -require("lgml.finetune").loss(model, it.mb) * it.mb.m
		moved = moved + it.advantage * (now - it.logp)
	end
	ok(moved > 0, ("one step moves log-probs with the advantages: %.4f"):format(moved))

	local before = share(lora.new(model, { rank = 4, seed = 2 }))
	local _, _, trained = run(0, 2e-2, 20)
	local after = share(trained)
	ok(after > before, ("a toy reward rises: %.2f -> %.2f"):format(before, after))
	local _, k0 = run(0, 5e-2, 6)
	local _, k1 = run(50, 5e-2, 6)
	ok(math.abs(k1[#k1]) <= math.abs(k0[#k0]) + 1e-6,
	    ("a large kl keeps the policy nearer the reference: |kl| %.3f against %.3f"):format(k1[#k1], k0[#k0]))
end

-- The tool: two steps from an adapter file, with a dump and a checkpoint.
do
	local init, out, rw, data, dump = tiny.tmp(".gguf"), tiny.tmp(".gguf"), tiny.tmp(".lua"),
	    tiny.tmp(".jsonl"), tiny.tmp(".jsonl")
	assert(lora.new(model, { rank = 2, seed = 3 }):save(init))
	local f = assert(io.open(rw, "w"))
	f:write("return function(answer, rec, ids) return #ids % 2, { parse = true } end\n")
	f:close()
	f = assert(io.open(data, "w"))
	f:write('{"messages":[{"role":"user","content":"Say something."}]}\n')
	f:close()
	local cmd = ("%s %s/../bin/lgml-grpo -m %s -d %s --init %s --reward %s -o %s -b CPU --samples 4 " ..
	    "--per-step 1 --steps 2 --max-new 3 --lr 1e-2 --dump %s 2>&1"):format(arg[-1] or "lua5.4", here,
	    path, data, init, rw, out, dump)
	local p = assert(io.popen(cmd))
	local log = p:read("a")
	local closed = p:close()
	local steps = select(2, log:gsub("\nstep %d+ reward", ""))
	local lines = 0
	for _ in io.lines(dump) do
		lines = lines + 1
	end
	ok(closed and steps + (log:match("^step %d+ reward") and 1 or 0) == 2 and lines == 8 and
	    lgml.gguf(out) ~= nil, ("lgml-grpo runs 2 steps and dumps 8 answers (%d lines)"):format(lines))
	if not closed then
		io.stderr:write(log)
	end
	for _, x in ipairs({ init, out, rw, data, dump }) do
		os.remove(x)
	end
end

model:close()
os.remove(path)
print(("1..%d"):format(n))
os.exit(failed and 1 or 0)
