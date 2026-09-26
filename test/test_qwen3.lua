-- Qwen3 KV cache and greedy decode. Output is TAP. Set LGML_MODEL to a
-- Qwen3-0.6B gguf (BF16 or Q8_0) to run the model tests.
local lgml = require("lgml")

local n = 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		os.exit(1)
	end
end

local function skip(name)
	n = n + 1
	print("ok " .. n .. " # SKIP " .. name)
end

local function same(got, want)
	if #got ~= #want then
		return false
	end
	for i = 1, #want do
		if got[i] ~= want[i] then
			return false
		end
	end
	return true
end

local function check(got, want, name)
	local good = same(got, want)
	if not good then
		io.stderr:write(name, ": got {", table.concat(got, ", "), "}\n")
	end
	ok(good, name)
end

-- Masks and ranges filled in C.
do
	local be <close> = assert(lgml.backend("CPU"))
	local ninf = -math.huge
	local m <close> = be:tensor("f32", { 3, 5 })
	m:fill_causal(2, 2)
	check(m:get(), {
		0, 0, 0, ninf, ninf,
		0, 0, 0, 0, ninf,
		ninf, ninf, ninf, ninf, ninf,
	}, "fill_causal f32")
	local h <close> = be:tensor("f16", { 2, 2 })
	h:fill_causal(0)
	check(h:get(), { 0, ninf, 0, 0 }, "fill_causal f16")
	ok(not pcall(m.fill_causal, m, 4, 2), "fill_causal past width is an error")
	local r <close> = be:tensor("i64", { 4 })
	r:fill_range(7)
	check(r:get(), { 7, 8, 9, 10 }, "fill_range i64")
	local p <close> = be:tensor("i32", { 3 })
	p:fill_range(5, 2)
	check(p:get(), { 5, 7, 9 }, "fill_range i32 step")
	ok(not pcall(m.fill_range, m, 0), "fill_range on f32 is an error")
end

-- "The capital of France is", tokenized by llama-tokenize.
local prompt = { 785, 6722, 315, 9625, 374 }
-- llama.cpp greedy output, 32 tokens, BF16 and Q8_0 without flash attn.
local want = {
	12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315,
	17689, 374, 24081, 13, 576, 6722, 315, 5616, 374, 26549, 13, 576,
	6722, 315, 6323, 374, 26194, 13, 576, 6722,
}
-- llama.cpp Q8_0 CPU with flash attn. Token 6 is a near tie (0.01 to 0.2
-- logits) between " Italy" and " France".
local want_fa = {
	12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279,
	5429, 315, 9625, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722,
	315, 279, 7513, 9145, 13, 576, 6722, 315,
}

local path = os.getenv("LGML_MODEL")
local devs = {}
for _, d in ipairs(lgml.devices()) do
	if d.name == "CPU" or d.name == "Vulkan0" then
		devs[#devs + 1] = d.name
	end
end

if path == nil or path == "" then
	skip("LGML_MODEL not set")
else
	local qwen3 = require("lgml.qwen3")
	local gen = require("lgml.generate")
	for _, dev in ipairs(devs) do
		local be <close> = assert(lgml.backend(dev))
		local tag = dev .. ": "
		local m = qwen3.load(be, path)

		local c = m:cache(64)
		local ids = gen.greedy(m, c, prompt, 32)
		check(ids, want, tag .. "greedy matches llama.cpp")
		ok(c.n_past == #prompt + 31, tag .. "cache advanced")

		c = m:cache(64)
		ids = gen.greedy(m, c, prompt, 32, { flash_attn = true })
		local good = same(ids, want) or same(ids, want_fa)
		if not good then
			io.stderr:write("flash_attn: got {", table.concat(ids, ", "), "}\n")
		end
		ok(good, tag .. "flash_attn greedy matches llama.cpp")

		-- A full pass without a cache (f32 k and v) predicts the same
		-- tokens up to the near tie at token 6.
		local seq = table.move(prompt, 1, #prompt, 1, {})
		table.move(want, 1, 4, #seq + 1, seq)
		local inp = m:inputs(seq)
		local g <close> = be:graph()
		local best = g:argmax((m:forward(g, inp)))
		g:compute(best)
		local got = table.move(best:get(), #prompt, #seq, 1, {})
		check(got, table.move(want, 1, 5, 1, {}), tag .. "no-cache pass agrees")

		-- Quantized caches round K and V after the Hadamard rotation. q8_0
		-- keeps the f16 tokens up to the near tie at token 6; q4_0 gives
		-- llama.cpp's q4_0 tokens (", and the capital").
		for ty, w in pairs({ q8_0 = table.move(want, 1, 5, 1, {}),
		    q4_0 = { 12095, 11, 323, 279, 6722 } }) do
			c = m:cache(64, { type = ty })
			ids = gen.greedy(m, c, prompt, 5, { flash_attn = true })
			check(ids, w, tag .. ty .. " cache greedy matches llama.cpp")
		end
		ok(not pcall(gen.greedy, m, m:cache(64, { type_k = "q8_0", type_v = "f16" }), prompt, 1),
		    tag .. "q8_0 cache without flash_attn is an error")

		c = m:cache(8)
		ok(not pcall(m.step, m, c, { 1, 2, 3, 4, 5, 6, 7, 8, 9 }),
		    tag .. "context overflow is an error")
		m:close()
	end
end

print("1.." .. n)
