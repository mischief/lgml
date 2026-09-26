-- DiffusionGemma 26B-A4B on the default backend (the GPU). Output is
-- TAP. LGML_DIFFUSION_GEMMA is the gguf path; the test skips when it is
-- empty. With LGML_DG_REF set to a dg_ref binary (scripts/dg_ref.cpp),
-- the llama.cpp fork runs the same seeded block for comparison.
local lgml = require("lgml")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local path = os.getenv("LGML_DIFFUSION_GEMMA")
if path == nil or path == "" then
	print("ok 1 # SKIP LGML_DIFFUSION_GEMMA not set")
	print("1..1")
	os.exit(0)
end

local dg = require("lgml.diffusion-gemma")
local chat = require("lgml.chat")
local tokenizer = require("lgml.tokenizer")

local be <close> = assert(lgml.backend(os.getenv("LGML_BACKEND")))
print("# backend " .. be:name())
local t0 = lgml.clock()
local m = dg.load(be, path)
print(("# load %.1f s"):format(lgml.clock() - t0))
local hp = m.hp
ok(hp.n_layer == 30 and hp.canvas == 256 and hp.n_vocab == 262144 and hp.n_expert == 128 and
    hp.n_used == 8, "hparams")

local info = assert(lgml.gguf(path))
local fam = chat.load(info.meta)
local tk = assert(tokenizer.open(path))
local text = fam.render({ { role = "user",
	content = "What is the capital of France? Answer with one word." } },
	{ add_generation_prompt = true, enable_thinking = false })
local prompt = tk:encode(text)
print("# prompt " .. table.concat(prompt, ","))

-- The first step without self-conditioning: unified and cached passes
-- agree on the argmax and the entropy of every canvas row.
do
	local C, P = hp.canvas, #prompt
	local canvas = {}
	for i = 1, C do
		canvas[i] = (i * 7919) % hp.n_vocab
	end
	local u <close> = be:tensor("f32", { C, 1 })
	u:fill("uniform", 0, 1, 7)
	local function step(inp)
		local g, best, ent = m:eval(inp, function(g, logits)
			return dg.sample_nodes(g, logits, 1.25, u)
		end)
		local a, e = best:get(), ent:get()
		g:close()
		dg.close_inputs(inp)
		return a, e
	end
	local ids = table.move(canvas, 1, C, P + 1, table.move(prompt, 1, P, 1, {}))
	local t1 = lgml.clock()
	local a1, e1 = step(m:inputs(ids, { P = P }))
	print(("# unified step %.2f s"):format(lgml.clock() - t1))
	local c = m:kv_cache(P, "f32")
	m:prefill(c, prompt)
	local a2, e2 = step(m:inputs(canvas, { mode = "decode", P = P, cache = c }))
	dg.close_cache(c)
	local same, de = 0, 0
	for i = 1, C do
		same = same + (a1[i] == a2[i] and 1 or 0)
		de = math.max(de, math.abs(e1[i] - e2[i]))
	end
	print(("# cached vs unified: %d/%d argmax equal, max entropy diff %.2e"):format(same, C, de))
	ok(same >= C - 2 and de < 1e-2, "cached prompt agrees with the unified pass")
end

-- One seeded block with the prompt KV cache.
local steps_seen = {}
local t2 = lgml.clock()
local function on_step(s, a)
	steps_seen[s] = table.concat(a, " ")
end
local canvas, steps = m:denoise(prompt, { seed = 42, kv_cache = true, cache_type = "f32",
	on_step = on_step })
local dt = lgml.clock() - t2
local eog = {}
for _, id in ipairs(tk:special().eog or {}) do
	eog[id] = true
end
local cut = dg.trim(canvas, eog)
local dec = tokenizer.decoder(tk, { special = false })
local out = {}
for i = 1, cut do
	out[#out + 1] = dec:push(canvas[i])
end
out[#out + 1] = dec:flush()
local reply = table.concat(out)
print(("# %d steps in %.2f s (%.1f ms/step): %q"):format(steps, dt, dt * 1000 / steps, reply))
ok(reply:lower():find("paris", 1, true) ~= nil, "the reply names Paris")

-- Free the device before the reference loads its own copy.
m:close()

-- The llama.cpp fork on the same prompt and seed: count the leading
-- steps whose argmax canvas matches.
local ref = os.getenv("LGML_DG_REF")
if ref and ref ~= "" then
	local cmd = ("%s denoise %s %s 42 1 2>/dev/null"):format(ref, path, table.concat(prompt, ","))
	local p = assert(io.popen(cmd))
	local want = {}
	for line in p:lines() do
		local s, toks = line:match("^step (%d+): (.*)$")
		if s then
			want[tonumber(s)] = toks
		end
	end
	p:close()
	local match = 0
	while want[match] and want[match] == steps_seen[match] do
		match = match + 1
	end
	print(("# llama.cpp: first %d steps match; it ran %d, we ran %d"):format(match, #want + 1, steps))
	ok(want[0] ~= nil and want[0] == steps_seen[0], "step 0 matches llama.cpp")
end

print("1.." .. n)
os.exit(failed and 1 or 0)
