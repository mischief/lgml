-- DiffusionGemma on a tiny random model, CPU only. Output is TAP. The
-- expected values come from the llama.cpp DiffusionGemma fork on the
-- same gguf (scripts/dg_ref.cpp): logits of one [prompt | canvas]
-- forward, and argmax canvases of seeded entropy-bound runs.
local lgml = require("lgml")
local uv = require("luv")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local tiny = require("dg_tiny")
local dg = require("lgml.diffusion-gemma")
local mt19937 = require("lgml.mt19937")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local function maxdiff(a, b)
	local m = #a == #b and 0 or math.huge
	for i = 1, math.min(#a, #b) do
		m = math.max(m, math.abs(a[i] - b[i]))
	end
	return m
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

-- libstdc++ std::mt19937(42) through uniform_real_distribution<float>
-- and uniform_int_distribution<int32_t>(0, 287), interleaved.
do
	local r = mt19937.new(42)
	local u, v = {}, {}
	for i = 1, 5 do
		u[i], v[i] = r:uniform(), r:below(288)
	end
	ok(maxdiff(u, { 0.37454012, 0.95071429, 0.731993914, 0.598658502, 0.156018645 }) < 1e-8 and
	    same(v, { 229, 52, 224, 171, 128 }), "mt19937 matches libstdc++")
end

-- diffusion_trim_canvas rules.
do
	local eog = { [1] = true }
	local c, s = dg.trim({ 5, 6, 1, 7 }, eog)
	ok(c == 2 and s == 1, "trim stops at an end token")
	c, s = dg.trim({ 5, 9, 9, 9, 9, 9, 9, 9, 8 }, eog)
	ok(c == 1 and s == nil, "trim cuts a stride 1 loop")
	c = dg.trim({ 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7 }, eog)
	ok(c == 1, "trim cuts a stride 2 loop")
	c = dg.trim({ 5, 6, 7, 8, 5, 6, 7, 8 }, eog)
	ok(c == 8, "trim keeps text without loops")
end

local device = os.getenv("LGML_DEVICE") or "CPU"
local be <close> = assert(lgml.backend(device))
-- GPU backends compute in f16 and differ from the CPU reference by
-- about 0.05, as llama.cpp's own GPU run does; seeded runs then diverge.
local gpu = device ~= "CPU"
local TOL = gpu and 0.1 or 1e-3

-- Device sampling against the host formulas.
do
	local logits = { 2, 0, -1, 1, 0.1, 0.7, 0.3, 0.5 }
	local l <close> = be:tensor("f32", { 2, 4 }, logits)
	local u <close> = be:tensor("f32", { 2, 1 }, { 0.9, 0.3 })
	local g <close> = be:graph()
	local best, ent, drawn = dg.sample_nodes(g, l, 2, u)
	g:compute(best, ent, drawn)
	local want_e, want_d = {}, {}
	for r = 0, 1 do
		local z, p, h, cum, pick = 0, {}, 0, 0, nil
		for v = 1, 4 do
			z = z + math.exp(2 * logits[r * 4 + v])
		end
		for v = 1, 4 do
			p[v] = math.exp(2 * logits[r * 4 + v]) / z
			h = h - p[v] * math.log(p[v])
			cum = cum + p[v]
			if not pick and cum >= ({ 0.9, 0.3 })[r + 1] then
				pick = v - 1
			end
		end
		want_e[r + 1], want_d[r + 1] = h, pick
	end
	ok(same(best:get(), { 0, 1 }), "sample argmax")
	ok(maxdiff(ent:get(), want_e) < 1e-5, "sample entropy")
	ok(same(drawn:get(), want_d), "sample multinomial draw")
end

local path = os.tmpname()
tiny.write(path)
local m = dg.load(be, path)

local prompt = { 2, 10, 20, 30, 40, 50, 60, 70 }
local canvas = { 100, 110, 120, 130, 140, 150, 160, 170 }
local ids = table.move(canvas, 1, 8, 9, table.move(prompt, 1, 8, 1, {}))

-- Canvas logits (8, 288) of one forward. opts go to inputs.
local function logits(model, input, opts)
	local inp = model:inputs(input, opts)
	local g, l = model:eval(inp, function(g, x)
		return g:cont(x)
	end)
	local v = l:get()
	g:close()
	dg.close_inputs(inp)
	return v
end

-- Every 23rd logit and the argmax of each row.
local function sample(v)
	local s, am = {}, {}
	for i = 1, #v, 23 do
		s[#s + 1] = v[i]
	end
	for r = 0, 7 do
		local b, bi = -math.huge, nil
		for i = 1, 288 do
			if v[r * 288 + i] > b then
				b, bi = v[r * 288 + i], i - 1
			end
		end
		am[#am + 1] = bi
	end
	return s, am
end

local ref0 = {
	-2.559906, 7.754752, -10.18867, 7.160268, -0.3839312, -1.281566, -2.618919, -2.086194,
	-1.313983, 5.205554, 10.70543, 0.6754102, -4.189247, -5.13571, -1.628817, 3.385177,
	-2.736738, 2.55688, -0.3907267, -5.326577, -0.3116405, -4.632416, 0.6445489, 0.01528251,
	-10.43732, -8.158317, 0.3444152, -7.026465, -2.244967, 5.58122, 3.316753, -0.5774987,
	-0.358446, -0.04973121, 1.538762, -2.674959, -3.468137, 4.919565, 3.231714, 2.869359,
	8.294851, -3.58377, -6.508507, 1.053414, 10.18659, -2.816695, 4.286823, 2.398256,
	6.132071, -1.314181, -0.9891088, 0.2346655, 5.731667, 2.660228, 5.756602, 3.574636,
	-1.595233, -8.020778, -2.836612, 3.770837, -1.36778, -7.036488, -4.225495, 11.89243,
	3.791964, -3.41001, -1.224132, -0.7687162, -2.575243, 2.661458, -1.181455, 0.6157203,
	8.43564, 6.361197, -3.442442, -2.957584, -5.397395, 0.7557924, 2.904768, -3.901214,
	5.324946, 3.351054, -0.8029888, -2.062149, 4.368034, 1.151984, 0.2461532, 4.856414,
	-1.38156, -11.89141, -2.750436, -6.722673, -5.611532, -4.583353, 9.704824, -1.586111,
	1.069702, 0.921051, 2.366065, 3.176302, -0.837498,
}
-- With self-conditioning from N(0, 3) logits (seed 99) at temp_inv 1.5.
local ref1 = {
	-2.514644, 7.621825, -10.14537, 7.191843, 0.4886977, -2.580781, -2.659754, -1.698684,
	-1.31197, 4.509191, 10.67095, -0.313315, -3.812036, -0.6830838, -1.64876, 0.1978359,
	0.1632043, 1.088752, -1.242711, -2.935488, -2.202882, -3.23187, 0.9589882, -1.860437,
	-7.351949, -8.653616, 2.494411, -8.748192, -2.276316, 6.367048, 3.715916, -0.2067333,
	-2.461888, -0.9831008, -2.787295, -4.216475, -5.045133, 4.115727, 1.415113, 5.556456,
	6.176993, -3.01505, -1.572772, -2.601893, 9.948477, -5.413919, 3.407092, -2.671467,
	5.599403, 0.778015, 0.4759327, 1.706439, 2.344632, 1.491554, 2.504498, -1.353875,
	-5.813008, -5.885127, -2.29402, 4.07015, -2.615189, -7.258493, -3.259214, 10.41716,
	2.498053, -2.102698, -1.806326, -1.6686, -1.085545, 1.97954, -1.306765, 2.215377,
	9.254097, 5.937617, -3.179885, -1.969414, -5.917386, 0.8292639, 3.612474, -4.490321,
	5.218266, 4.043591, -3.383225, -3.369751, 4.752471, 0.3127042, 3.202116, 6.224388,
	-2.481954, -3.623305, -3.325648, -4.084021, -3.103203, -3.127543, 8.601881, -1.381964,
	2.660383, 5.699792, -0.3125467, -1.025896, 3.307562,
}

local base = logits(m, ids, { P = 8 })
local s, am = sample(base)
local err0 = maxdiff(s, ref0)
print("# max logit error vs llama.cpp: " .. err0)
ok(err0 < TOL, "logits match llama.cpp")
ok(same(am, canvas), "canvas argmax matches llama.cpp")

do
	local prev <close> = be:tensor("f32", { 8, 288 })
	prev:fill("normal", 0, 3, 99)
	local v = logits(m, ids, { P = 8, sc = { prev = prev, temp_inv = 1.5 } })
	s, am = sample(v)
	local err1 = maxdiff(s, ref1)
	print("# max logit error with self-conditioning: " .. err1)
	ok(err1 < TOL, "self-conditioned logits match llama.cpp")
	ok(same(am, { 100, 110, 120, 66, 140, 150, 160, 170 }), "self-conditioned argmax")
end

do
	local m1 = dg.load(be, path, { layers_per_graph = 1 })
	ok(maxdiff(logits(m1, ids, { P = 8 }), base) < 1e-6, "one graph per layer gives the same logits")
	m1:close()
end

-- Flash attention rounds K and V to f16.
do
	local mf = dg.load(be, path, { flash_attn = true })
	local d = maxdiff(logits(mf, ids, { P = 8 }), base)
	print("# flash attention vs soft_max: " .. d)
	ok(d < 0.1, "flash attention stays close")
	mf:close()
end

-- The prompt KV cache: prefill (in chunks of 3), then a canvas-only pass.
do
	local c = m:kv_cache(16, "f32")
	m:prefill(c, prompt, 3)
	ok(c.n == 8, "prefill fills the cache")
	local v = logits(m, canvas, { mode = "decode", P = 8, cache = c })
	print("# kv cache vs unified: " .. maxdiff(v, base))
	ok(maxdiff(v, base) < (gpu and 0.1 or 1e-4), "cached prompt gives the unified logits")
	ok(m:prefill(c, { 2, 10, 20, 30, 99 }, 3) == 4 and c.n == 5, "prefill keeps the shared prefix")
	dg.close_cache(c)
	c = m:kv_cache(16, "f16")
	m:prefill(c, prompt)
	local d = maxdiff(logits(m, canvas, { mode = "decode", P = 8, cache = c }), base)
	print("# f16 kv cache vs unified: " .. d)
	ok(d < 0.1, "f16 cache stays close")
	dg.close_cache(c)
end

-- Seeded entropy-bound runs: the argmax canvas of the last step, and
-- the step count, from llama.cpp.
local runs = {
	{ seed = 42, steps = 25, want = { 107, 28, 122, 94, 41, 271, 162, 191 } },
	{ seed = 3, steps = 26, want = { 188, 6, 41, 233, 107, 41, 24, 271 } },
}
for _, r in ipairs(runs) do
	for _, kv in ipairs({ false, true }) do
		local got, steps = m:denoise(prompt, { seed = r.seed, kv_cache = kv, cache_type = "f32" })
		local tag = ("seed %d%s"):format(r.seed, kv and ", kv cache" or "")
		if gpu then
			ok(#got == #r.want and steps >= 1 and steps <= 48, "denoise runs, " .. tag)
		elseif not same(got, r.want) or steps ~= r.steps then
			io.stderr:write(tag, ": got {", table.concat(got, ", "), "} in ", steps, " steps\n")
		end
		if not gpu then
			ok(same(got, r.want) and steps == r.steps, "denoise matches llama.cpp, " .. tag)
		end
	end
end

-- Thirty layers need more than one graph.
do
	local p30 = os.tmpname()
	tiny.write(p30, 30)
	local m30 = dg.load(be, p30)
	local got = m30:denoise(prompt, { seed = 1, steps = 2, kv_cache = true })
	ok(#got == 8, "thirty layers run over several graphs")
	m30:close()
	os.remove(p30)
end

m:close()

-- The chat engine, with the Gemma 4 template and the tiny vocab.
do
	local engine = require("lgml.diffusion_engine")
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 96, seed = 42 })
	require("lgml.server.engine").check(e)
	ok(e:models()[1].context_length == 96 and e:props().total_slots == 1, "engine models and props")
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
	function r.finish(_, reason, usage)
		r.reason, r.usage = reason, usage
	end
	function r.cancelled()
		return false
	end
	local done, res, err
	coroutine.wrap(function()
		res, err = e:chat({ messages = { { role = "user", content = "hi" } }, max_tokens = 20 }, r)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	ok(res == true, "engine chat returns true " .. tostring(err))
	ok((r.reason == "stop" or r.reason == "length") and r.usage.completion_tokens <= 20 and
	    r.usage.completion_tokens >= 1, "engine finishes within max_tokens")
end

os.remove(path)
print("1.." .. n)
os.exit(failed and 1 or 0)
