-- The MoE, state-space, conv and mask ops on the CPU backend. Output is TAP.
local lgml = require("lgml")

local n = 0
local failed = false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local function near(got, want, name, tol)
	if got == nil then
		n = n + 1
		print("ok " .. n .. " - " .. name .. " # SKIP op not supported on this backend")
		return
	end
	local good = #got == #want
	for i = 1, #want do
		if not good or math.abs(got[i] - want[i]) > (tol or 1e-4) then
			good = false
			break
		end
	end
	if not good then
		io.stderr:write(name, ": got {", table.concat(got, ", "), "}\n")
	end
	ok(good, name)
end

local be <close> = assert(lgml.backend(os.getenv("LGML_DEVICE") or "CPU"))
local function f32(shape, v)
	return be:tensor("f32", shape, v)
end

-- Compute a one-output graph and return the values and shape, or nil
-- when the backend lacks the op.
local function run(fn)
	local g <close> = be:graph()
	local r = fn(g)
	local good, err = pcall(g.compute, g, r)
	if not good then
		if tostring(err):find("not supported", 1, true) then
			io.stderr:write(tostring(err), "\n")
			return nil
		end
		error(err, 0)
	end
	return r:get(), r:shape()
end

local function silu(x)
	return x / (1 + math.exp(-x))
end

do
	local a <close> = f32({ 4 }, { 0.5, 1, 2, 4 })
	near(run(function(g) return g:log(a) end),
	    { math.log(0.5), 0, math.log(2), math.log(4) }, "log")
end

do
	local a <close> = f32({ 4 }, { -1, 0, 0.5, 3 })
	near(run(function(g) return g:step(a) end), { 0, 0, 1, 1 }, "step")
	near(run(function(g) return g:softplus(a) end),
	    { math.log(1 + math.exp(-1)), math.log(2), math.log(1 + math.exp(0.5)),
	    math.log(1 + math.exp(3)) }, "softplus")
	near(run(function(g) return g:expm1(a) end),
	    { math.exp(-1) - 1, 0, math.exp(0.5) - 1, math.exp(3) - 1 }, "expm1")
end

do
	local a <close> = f32({ 2, 3 }, { 1, 2, 3, 4, 5, 6 })
	near(run(function(g) return g:cumsum(a) end), { 1, 3, 6, 4, 9, 15 }, "cumsum by row")
end

do
	-- Each row is [x | gate].
	local a <close> = f32({ 2, 4 }, { 1, -2, 3, 4, 0.5, 2, -1, 1 })
	local want = { silu(1) * 3, silu(-2) * 4, silu(0.5) * -1, silu(2) * 1 }
	local got, shape = run(function(g) return g:swiglu(a) end)
	near(got, want, "swiglu")
	near(shape, { 2, 2 }, "swiglu shape")
	local x <close> = f32({ 2 }, { 1, -2 })
	local gate <close> = f32({ 2 }, { 3, 4 })
	near(run(function(g) return g:swiglu_split(x, gate) end), { silu(1) * 3, silu(-2) * 4 },
	    "swiglu_split")
	local ge = run(function(g) return g:geglu_split(x, gate) end)
	local gl = run(function(g) return g:mul(g:gelu(x), gate) end)
	near(ge, gl, "geglu_split matches gelu * gate", 1e-3)
	ok(not pcall(function() return run(function(g) return g:swiglu(f32({ 3 })) end) end),
	    "swiglu odd width is an error")
end

do
	local a <close> = f32({ 2, 2 }, { 3, 4, 0, 2 })
	near(run(function(g) return g:l2_norm(a, 0) end), { 0.6, 0.8, 0, 1 }, "l2_norm")
end

do
	-- (C=2, H=1, W=2), one group: normalize all four values together.
	local a <close> = f32({ 2, 1, 2 }, { 1, 2, 3, 4 })
	local m, var = 2.5, 1.25
	local want = {}
	for i, v in ipairs({ 1, 2, 3, 4 }) do
		want[i] = (v - m) / math.sqrt(var + 1e-5)
	end
	near(run(function(g) return g:group_norm(a, 1, 1e-5) end), want, "group_norm one group")
	-- Two groups: each channel alone.
	local s = 1 / math.sqrt(0.25 + 1e-5) * 0.5
	near(run(function(g) return g:group_norm(a, 2, 1e-5) end), { -s, s, -s, s },
	    "group_norm per channel")
end

do
	local a <close> = f32({ 2, 4 }, { 0.3, 0.1, 0.4, 0.2, 5, 7, 6, 8 })
	near(run(function(g) return g:argsort(a) end), { 1, 3, 0, 2, 0, 2, 1, 3 }, "argsort asc")
	near(run(function(g) return g:argsort(a, "desc") end), { 2, 0, 3, 1, 3, 1, 2, 0 },
	    "argsort desc")
	local got, shape = run(function(g) return g:cont(g:argsort_top_k(a, 2)) end)
	near(got, { 2, 0, 3, 1 }, "argsort_top_k values")
	near(shape, { 2, 2 }, "argsort_top_k shape")
end

do
	-- 3 experts, each (M=2, K=2): expert e is (e+1) * identity.
	local w <close> = f32({ 3, 2, 2 }, { 1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3 })
	-- T=2 tokens, U=1: one input row each.
	local x <close> = f32({ 2, 1, 2 }, { 1, 2, 10, 20 })
	local ids <close> = be:tensor("i32", { 2, 2 }, { 0, 2, 1, 1 })
	local got, shape = run(function(g) return g:mul_mat_id(w, x, ids) end)
	near(shape, { 2, 2, 2 }, "mul_mat_id shape")
	near(got, { 1, 2, 3, 6, 20, 40, 20, 40 }, "mul_mat_id picks experts")
	local b <close> = f32({ 3, 2 }, { 0, 0, 100, 100, 1000, 1000 })
	near(run(function(g) return g:add_id(g:mul_mat_id(w, x, ids), b, ids) end),
	    { 1, 2, 1003, 1006, 120, 140, 120, 140 }, "add_id adds per-expert bias")
	local bad <close> = be:tensor("i32", { 3, 2 })
	ok(not pcall(function() return run(function(g) return g:mul_mat_id(w, x, bad) end) end),
	    "mul_mat_id ids count mismatch is an error")
end

do
	-- S=1, D=2, d_conv=2, T=3: out[t, d] = x[d, t] * w[d, 1] + x[d, t + 1] * w[d, 2].
	local x <close> = f32({ 1, 2, 4 }, { 1, 2, 3, 4, 10, 20, 30, 40 })
	local w <close> = f32({ 2, 2 }, { 1, 1, 2, -1 })
	local got, shape = run(function(g) return g:ssm_conv(x, w) end)
	near(shape, { 1, 3, 2 }, "ssm_conv shape")
	near(got, { 3, 0, 5, 10, 7, 20 }, "ssm_conv values")
end

do
	-- One head, S=2, one token. Zero state, gate 0 (decay 1), beta 1:
	-- the new state is k v^T, so the output is v * (k . q) / sqrt(S).
	local q <close> = f32({ 1, 1, 1, 2 }, { 1, 0 })
	local k <close> = f32({ 1, 1, 1, 2 }, { 1, 0 })
	local v <close> = f32({ 1, 1, 1, 2 }, { 3, 5 })
	local gate <close> = f32({ 1, 1, 1, 1 }, { 0 })
	local beta <close> = f32({ 1, 1, 1, 1 }, { 1 })
	local st <close> = f32({ 1, 1, 2, 2 }, { 0, 0, 0, 0 })
	local got, shape = run(function(g) return g:gated_delta_net(q, k, v, gate, beta, st) end)
	near(shape, { 3, 2 }, "gated_delta_net shape")
	near(got and { got[1], got[2] }, { 3 / math.sqrt(2), 5 / math.sqrt(2) }, "gated_delta_net output")
end

do
	local a <close> = f32({ 2, 2 }, { 1, 2, 3, 4 })
	local got, shape = run(function(g) return g:pad(a, { { 1, 0 }, { 0, 2 } }) end)
	near(shape, { 3, 4 }, "pad shape")
	near(got, { 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0 }, "pad values")
end

do
	local a <close> = f32({ 1, 1, 2, 2 }, { 1, 2, 3, 4 })
	local got, shape = run(function(g) return g:upscale(a, 2) end)
	near(shape, { 1, 1, 4, 4 }, "upscale shape")
	near(got, { 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4 }, "upscale nearest")
end

do
	-- 2x2 kernel of ones over a 3x3 input: sums of each 2x2 window.
	local w <close> = be:tensor("f16", { 1, 1, 2, 2 }, { 1, 1, 1, 1 })
	local x <close> = f32({ 1, 1, 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 })
	local got, shape = run(function(g) return g:conv2d(w, x) end)
	near(shape, { 1, 1, 2, 2 }, "conv2d shape")
	near(got, { 12, 16, 24, 28 }, "conv2d values")
	local p = run(function(g) return g:conv2d(w, x, 2, 1) end)
	near(p, { 1, 5, 11, 28 }, "conv2d stride 2 pad 1")
end

do
	-- The cpy is not on the path to the output; expand makes compute run it.
	local a <close> = f32({ 2 }, { 3, 4 })
	local c <close> = f32({ 2 }, { 0, 0 })
	local g <close> = be:graph()
	g:expand(g:cpy(g:scale(a, 2), c))
	local r = g:add(a, a)
	g:compute(r)
	near(c:get(), { 6, 8 }, "expand runs a cpy into a leaf")
	near(r:get(), { 6, 8 }, "expand output")
end

do
	-- H * H = I, and g:hadamard turns each row the same as mul_mat.
	local h <close> = f32({ 4, 4 })
	h:fill("hadamard")
	near(run(function(g) return g:mul_mat(h, h) end),
	    { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }, "hadamard fill is its own inverse")
	local x <close> = f32({ 2, 4 }, { 1, 2, 3, 4, -1, 0, 2, 5 })
	near(run(function(g) return g:hadamard(h, x) end), run(function(g) return g:mul_mat(h, x) end),
	    "hadamard op matches mul_mat")
	ok(not pcall(h.fill, f32({ 3, 3 }), "hadamard"), "hadamard fill needs a power of 2")
end

-- Masks: 1 is visible, 0 is -inf.
local function mask(rows, nkv, opts)
	local m <close> = be:tensor("f32", { rows, nkv })
	m:fill_mask(opts)
	local out = {}
	for i, v in ipairs(m:get()) do
		out[i] = v == 0 and 1 or 0
	end
	return out
end

near(mask(3, 3, {}), { 1, 0, 0, 1, 1, 0, 1, 1, 1 }, "fill_mask causal")
near(mask(2, 4, { n_past = 2 }), { 1, 1, 1, 0, 1, 1, 1, 1 }, "fill_mask n_past")
near(mask(3, 3, { window = 2 }), { 1, 0, 0, 1, 1, 0, 0, 1, 1 }, "fill_mask window")
near(mask(2, 2, { causal = false }), { 1, 1, 1, 1 }, "fill_mask full")
-- Prompt of 3, canvas of 2: prompt rows causal, canvas rows see all.
near(mask(5, 5, { canvas = 3 }), {
	1, 0, 0, 0, 0,
	1, 1, 0, 0, 0,
	1, 1, 1, 0, 0,
	1, 1, 1, 1, 1,
	1, 1, 1, 1, 1,
}, "fill_mask canvas")
-- With window 2, canvas rows see only the last prompt key.
near(mask(5, 5, { canvas = 3, window = 2 }), {
	1, 0, 0, 0, 0,
	1, 1, 0, 0, 0,
	0, 1, 1, 0, 0,
	0, 0, 1, 1, 1,
	0, 0, 1, 1, 1,
}, "fill_mask canvas window")
-- A ring of 4 slots after positions 0 to 6: slots hold 4, 5, 6 and 3.
near(mask(2, 4, { n_past = 5, window = 3, ring = true }), { 1, 1, 0, 1, 1, 1, 1, 0 },
    "fill_mask ring")
near(mask(2, 4, { window = 3, ring = true }), { 1, 0, 0, 0, 1, 1, 0, 0 },
    "fill_mask ring before it fills")

-- Host quantization with the ggml reference quantizer.
do
	local v, parts = {}, {}
	for i = 1, 64 do
		v[i] = (i - 32) / 8
		parts[i] = string.pack("<f", v[i])
	end
	local q = lgml.quantize("q8_0", table.concat(parts), 32)
	ok(#q == 2 * lgml.row_size("q8_0", 32), "quantize q8_0 size")
	local t <close> = be:tensor("q8_0", { 2, 32 }, q)
	near(run(function(g) return g:cast(t, "f32") end), v, "quantize q8_0 round trip", 0.02)
	ok(not pcall(lgml.quantize, "q8_0", table.concat(parts), 30),
	    "quantize row length not a block multiple is an error")
end

-- The test gguf writer: tensors load back with their shapes and values.
do
	local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
	package.path = here .. "?.lua;" .. package.path
	local gguf = require("gguf")
	local path = os.tmpname()
	gguf.write(path, { { "general.architecture", gguf.STRING, "test" } }, {
		{ "a", { 2, 3 }, { 1, 2, 3, 4, 5, 6 } },
		{ "b", { 3 }, { 7, 8, 9 } },
	})
	local w <close> = be:load(path)
	os.remove(path)
	near(w:get("a"):shape(), { 2, 3 }, "gguf writer shape")
	near(w:get("a"):get(), { 1, 2, 3, 4, 5, 6 }, "gguf writer values")
	near(w:get("b"):get(), { 7, 8, 9 }, "gguf writer second tensor")
end

print("1.." .. n)
os.exit(failed and 1 or 0)
