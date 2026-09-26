-- Qwen3 KV cache and greedy decode. Output is TAP. Set LGML_MODEL to a
-- Qwen3-14B Q4_K_M gguf to run the model tests.
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
-- llama.cpp CPU greedy output, with and without flash attn and with
-- q8_0 and q4_0 caches. Token 26 is a near tie (" located" or " a"), so
-- the tests stop before it.
local want = {
	12095, 13, 3555, 374, 279, 6722, 315, 279, 3639, 4180, 30, 576,
	6722, 315, 279, 3639, 4180, 374, 6515, 11, 422, 727, 13, 1084,
	374,
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
	local transformer = require("lgml.transformer")
	local gen = require("lgml.generate")
	for _, dev in ipairs(devs) do
		local be <close> = assert(lgml.backend(dev))
		local tag = dev .. ": "
		local m = transformer.load(be, path)

		local c = m:cache(64)
		local ids = gen.greedy(m, c, prompt, #want)
		check(ids, want, tag .. "greedy matches llama.cpp")
		ok(c.n_past == #prompt + #want - 1, tag .. "cache advanced")

		c = m:cache(64)
		ids = gen.greedy(m, c, prompt, #want, { flash_attn = true })
		check(ids, want, tag .. "flash_attn greedy matches llama.cpp")

		-- A full pass without a cache (f32 k and v) predicts the same
		-- tokens.
		local seq = table.move(prompt, 1, #prompt, 1, {})
		table.move(want, 1, 4, #seq + 1, seq)
		local inp = m:inputs(seq)
		local g <close> = be:graph()
		local best = g:argmax((m:forward(g, inp)))
		g:compute(best)
		local got = table.move(best:get(), #prompt, #seq, 1, {})
		check(got, table.move(want, 1, 5, 1, {}), tag .. "no-cache pass agrees")

		-- Quantized caches round K and V after the Hadamard rotation.
		-- llama.cpp's q8_0 and q4_0 caches keep the first tokens.
		for ty, w in pairs({ q8_0 = table.move(want, 1, 5, 1, {}),
		    q4_0 = table.move(want, 1, 5, 1, {}) }) do
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
