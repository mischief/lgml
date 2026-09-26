-- Sampler tests on CPU and Vulkan0. Output is TAP.
local lgml = require("lgml")
local presets = require("lgml.sampler")

local n = 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		os.exit(1)
	end
end

-- Neutral chain: only the options given in o are active.
local function neutral(o)
	local t = { temperature = 1, top_k = 0, top_p = 1, min_p = 0, seed = 1 }
	for k, v in pairs(o or {}) do
		t[k] = v
	end
	return lgml.sampler(t)
end

-- Draw count times; return a map id -> hits.
local function hist(s, x, count)
	local h = {}
	for _ = 1, count do
		local id = s:sample(x)
		h[id] = (h[id] or 0) + 1
	end
	return h
end

local function keys(h)
	local r = {}
	for k in pairs(h) do
		r[#r + 1] = k
	end
	table.sort(r)
	return table.concat(r, ",")
end

-- Check hit rates against probabilities within 4 standard deviations.
local function fits(h, p, count)
	for i, pi in ipairs(p) do
		local got = (h[i - 1] or 0) / count
		local sd = math.sqrt(pi * (1 - pi) / count)
		if math.abs(got - pi) > 4 * sd + 1e-9 then
			io.stderr:write(("id %d: got %.4f want %.4f\n"):format(i - 1, got, pi))
			return false
		end
	end
	return true
end

local function softmax(l, temp)
	local m, sum, p = -math.huge, 0, {}
	for _, v in ipairs(l) do
		m = math.max(m, v)
	end
	for i, v in ipairs(l) do
		p[i] = math.exp((v - m) / temp)
		sum = sum + p[i]
	end
	for i = 1, #p do
		p[i] = p[i] / sum
	end
	return p
end

local function logs(p)
	local l = {}
	for i, v in ipairs(p) do
		l[i] = math.log(v)
	end
	return l
end

local names = {}
for _, d in ipairs(lgml.devices()) do
	if d.name == "CPU" or d.name == "Vulkan0" then
		names[#names + 1] = d.name
	end
end
ok(#names >= 1, "devices found")

for _, name in ipairs(names) do
	local be <close> = assert(lgml.backend(name))
	local tag = name .. ": "
	local function tensor(l, rows)
		return be:tensor("f32", rows and { rows, #l // rows } or { 1, #l }, l)
	end

	-- greedy and rows
	local x <close> = tensor({ 1, 5, 3, 2 })
	local g <close> = lgml.sampler({ temperature = 0 })
	ok(g:sample(x) == 1, tag .. "greedy")
	local x2 <close> = tensor({ 9, 1, 2, 3, 1, 2, 7, 3 }, 2)
	ok(g:sample(x2, 1) == 0, tag .. "row 1")
	ok(g:sample(x2) == 2, tag .. "default row is last")
	ok(not pcall(g.sample, g, x2, 3), tag .. "bad row is an error")
	local v1 <close> = be:tensor("f32", { 4 }, { 0, 0, 8, 0 })
	ok(g:sample(v1) == 2, tag .. "rank 1 logits")
	local h16 <close> = be:tensor("f16", { 1, 4 }, { 0, 3, 1, 0 })
	ok(g:sample(h16) == 1, tag .. "f16 logits")

	-- top_k = 1 at high temperature is still the argmax
	local k1 = neutral({ top_k = 1, temperature = 5 })
	ok(keys(hist(k1, x, 200)) == "1", tag .. "top_k = 1")

	-- determinism
	local p4 = { 0.1, 0.2, 0.3, 0.4 }
	local xp <close> = tensor(logs(p4))
	local a, b, c = neutral({ seed = 42 }), neutral({ seed = 42 }), neutral({ seed = 43 })
	local same, diff = true, false
	for _ = 1, 100 do
		local ia, ib, ic = a:sample(xp), b:sample(xp), c:sample(xp)
		same = same and ia == ib
		diff = diff or ia ~= ic
	end
	ok(same, tag .. "same seed, same draws")
	ok(diff, tag .. "other seed, other draws")

	-- penalties
	local xr <close> = tensor({ 1, 2.0, 1.9, 0 })
	local r = lgml.sampler({ temperature = 0, repeat_penalty = 1.1 })
	ok(r:sample(xr) == 1, tag .. "no history, no penalty")
	r:accept(1)
	ok(r:sample(xr) == 2, tag .. "repeat penalty")
	r:reset()
	ok(r:sample(xr) == 1, tag .. "reset clears history")
	local xn <close> = tensor({ -1, -1.05 })
	local rn = lgml.sampler({ temperature = 0, repeat_penalty = 1.1 })
	rn:accept(0)
	ok(rn:sample(xn) == 1, tag .. "repeat penalty on negative logit")
	local pr = lgml.sampler({ temperature = 0, presence_penalty = 0.2 })
	pr:accept(1)
	ok(pr:sample(xr) == 2, tag .. "presence penalty")
	local fq = lgml.sampler({ temperature = 0, frequency_penalty = 0.06 })
	fq:accept(1)
	ok(fq:sample(xr) == 1, tag .. "frequency penalty, one hit")
	fq:accept(1)
	ok(fq:sample(xr) == 2, tag .. "frequency penalty, two hits")
	local ln = lgml.sampler({ temperature = 0, presence_penalty = 1, repeat_last_n = 1 })
	ln:accept(1)
	ln:accept(3)
	ok(ln:sample(xr) == 1, tag .. "repeat_last_n drops old ids")

	-- top_p and min_p
	local q = { 0.5, 0.3, 0.15, 0.05 }
	local xq <close> = tensor(logs(q))
	ok(keys(hist(neutral({ top_p = 0.7 }), xq, 2000)) == "0,1", tag .. "top_p 0.7")
	ok(keys(hist(neutral({ top_p = 0.85 }), xq, 2000)) == "0,1,2", tag .. "top_p 0.85")
	ok(keys(hist(neutral({ min_p = 0.25 }), xq, 2000)) == "0,1,2", tag .. "min_p 0.25")
	ok(keys(hist(neutral({ min_p = 0.5 }), xq, 2000)) == "0,1", tag .. "min_p 0.5")
	ok(keys(hist(neutral({ top_k = 3, min_p = 0.5 }), xq, 2000)) == "0,1",
	    tag .. "min_p after top_k")

	-- temperature
	ok(keys(hist(neutral({ temperature = 0.01 }), xq, 500)) == "0", tag .. "low temperature")
	local count = 20000
	ok(fits(hist(neutral({ seed = 7 }), xq, count), q, count), tag .. "softmax draws")
	ok(fits(hist(neutral({ seed = 8, temperature = 2 }), xq, count),
	    softmax(logs(q), 2), count), tag .. "temperature 2 draws")

	-- large vocab: partial selection paths
	local V, big = 5000, {}
	local rnd = 12345
	for i = 1, V do
		rnd = (rnd * 1103515245 + 12345) % 2147483648
		big[i] = rnd / 2147483648 * 20
	end
	local xb <close> = tensor(big)
	local order = {}
	for i = 1, V do
		order[i] = i
	end
	table.sort(order, function(i, j) return big[i] > big[j] end)
	local top50 = {}
	for i = 1, 50 do
		top50[order[i] - 1] = true
	end
	local inside = true
	for id in pairs(hist(neutral({ top_k = 50 }), xb, 2000)) do
		inside = inside and top50[id] == true
	end
	ok(inside, tag .. "top_k 50 of 5000")

	-- top_p without top_k: expected set from a full sort in Lua
	local pb = softmax(big, 1)
	local cum, allowed = 0, {}
	for i = 1, V do
		allowed[order[i] - 1] = true
		cum = cum + pb[order[i]]
		if cum >= 0.9 then
			break
		end
	end
	local hb = hist(neutral({ top_p = 0.9 }), xb, 2000)
	inside = true
	for id in pairs(hb) do
		inside = inside and allowed[id] == true
	end
	ok(inside, tag .. "top_p 0.9 of 5000")

	-- lifetime
	local cl = lgml.sampler()
	cl:close()
	cl:close()
	ok(not pcall(cl.sample, cl, x), tag .. "closed sampler is an error")
	local gr <close> = be:graph()
	local node = gr:add(x, x)
	ok(not pcall(g.sample, g, node), tag .. "uncomputed node is an error")
end

-- presets
do
	local s = presets.new("greedy")
	local be <close> = assert(lgml.backend("CPU"))
	local x <close> = be:tensor("f32", { 1, 3 }, { 0, 1, 0 })
	ok(s:sample(x) == 1, "greedy preset")
	ok(pcall(presets.new, "qwen3", { seed = 3 }), "preset with overrides")
	ok(not pcall(presets.new, "nope"), "unknown preset is an error")
end

-- timing: full Qwen3 vocab on Vulkan0, including the device read
local ok_vk, vk = pcall(lgml.backend, "Vulkan0")
if ok_vk and vk then
	local V, l = 151936, {}
	for i = 1, V do
		l[i] = math.random() * 30 - 15
	end
	local x <close> = vk:tensor("f32", { 1, V }, l)
	local has_uv, uv = pcall(require, "luv")
	local clock = has_uv and function() return uv.hrtime() / 1e9 end or os.clock
	local function bench(label, opts)
		local s = lgml.sampler(opts)
		s:sample(x)
		local iters = 200
		local t0 = clock()
		for _ = 1, iters do
			s:accept(s:sample(x))
		end
		local us = (clock() - t0) / iters * 1e6
		io.stderr:write(("# %s: %.0f us/sample\n"):format(label, us))
	end
	bench("greedy", { temperature = 0 })
	bench("default chain", { seed = 1, repeat_penalty = 1.1 })
	bench("top_p only", { seed = 1, top_k = 0, min_p = 0 })
	bench("no truncation", { seed = 1, top_k = 0, top_p = 1, min_p = 0 })
	vk:close()
end

print("1.." .. n)
