-- Autograd against finite differences, and the optimizer steps, on the
-- CPU backend. Output is TAP.
local lgml = require("lgml")
local optim = require("lgml.optim")

local n = 0
local failed = false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local be <close> = assert(lgml.backend(os.getenv("LGML_DEVICE") or "CPU"))

local function maxdiff(a, b)
	local d = #a == #b and 0 or math.huge
	for i = 1, #a do
		d = math.max(d, math.abs(a[i] - b[i]))
	end
	return d
end

local function numel(shape)
	local c = 1
	for _, d in ipairs(shape) do
		c = c * d
	end
	return c
end

local seed = 1
local function randn(shape, mean, sd)
	local t = be:tensor("f32", shape)
	seed = seed + 1
	t:fill("normal", mean or 0, sd or 1, seed)
	return t
end

-- Loss of the output y of build: sum(y * r), with a fixed random r. The
-- random weights give each element of y its own gradient.
local function weighted(g, y, r)
	return g:sum(g:mul(g:cont(y), r))
end

-- Compare the gradient of each param with central differences of the
-- loss. build(g, params, extra) returns the output node. Returns the
-- largest error, scaled by max(1, |numeric|).
local EPS, TOL = 1e-2, 1e-2
local function fdcheck(name, params, build, opts)
	opts = opts or {}
	for _, p in ipairs(params) do
		p:requires_grad()
		p:grad():zero()
	end
	local function loss_of(g, r)
		local y = build(g, params)
		if opts.scalar then
			return y
		end
		return weighted(g, y, r)
	end
	-- r has the shape of the output; a probe graph finds it.
	local r
	if not opts.scalar then
		local pg <close> = be:graph()
		r = randn(build(pg, params):shape())
	end
	local g <close> = be:graph(65536)
	local loss = loss_of(g, r)
	loss:backward({ lower = opts.lower })
	g:compute(loss)
	local fg <close> = be:graph(65536)
	local floss = loss_of(fg, r)
	local worst = 0
	for pi, p in ipairs(params) do
		local grad = p:grad():get()
		local v = p:get()
		-- At most 24 elements per param, spread over the tensor.
		local step = math.max(1, #v // 24)
		for i = 1, #v, step do
			local x0 = v[i]
			v[i] = x0 + EPS
			p:set(v)
			fg:compute(floss)
			local lp = floss:get()[1]
			v[i] = x0 - EPS
			p:set(v)
			fg:compute(floss)
			local lm = floss:get()[1]
			v[i] = x0
			p:set(v)
			local num = (lp - lm) / (2 * EPS)
			local err = math.abs(num - grad[i]) / math.max(1, math.abs(num))
			if err > worst then
				worst = err
			end
			if err > TOL then
				io.stderr:write(("%s: param %d elem %d: autograd %g, numeric %g\n"):format(name,
				    pi, i, grad[i], num))
			end
		end
	end
	for _, p in ipairs(params) do
		p:requires_grad(false)
	end
	ok(worst <= TOL, ("%s: max error %.2e"):format(name, worst))
	return worst
end

local report = {}
local function check(name, params, build, opts)
	report[#report + 1] = { name, fdcheck(name, params, build, opts) }
end

-- Elementwise ops.
do
	local a, b = randn({ 3, 5 }), randn({ 3, 5 })
	check("add", { a, b }, function(g, p) return g:add(p[1], p[2]) end)
	check("sub", { a, b }, function(g, p) return g:sub(p[1], p[2]) end)
	check("mul", { a, b }, function(g, p) return g:mul(p[1], p[2]) end)
	local pos = randn({ 3, 5 }, 3, 0.5)
	check("div", { a, pos }, function(g, p) return g:div(p[1], p[2]) end)
	check("scale", { a }, function(g, p) return g:scale(p[1], 0.7) end)
	check("sqr", { a }, function(g, p) return g:sqr(p[1]) end)
	check("sqrt", { pos }, function(g, p) return g:sqrt(p[1]) end)
	check("log", { pos }, function(g, p) return g:log(p[1]) end)
	check("exp", { a }, function(g, p) return g:exp(p[1]) end)
	check("silu", { a }, function(g, p) return g:silu(p[1]) end)
	check("softplus", { a }, function(g, p) return g:softplus(p[1]) end)
	check("neg", { a }, function(g, p) return g:neg(p[1]) end)
end

-- Broadcast operands: a norm weight and a bias row.
do
	local x, w = randn({ 4, 6 }), randn({ 6 }, 1, 0.1)
	check("mul broadcast", { x, w }, function(g, p) return g:mul(p[1], p[2]) end)
	check("add broadcast", { x, w }, function(g, p) return g:add(p[1], p[2]) end)
end

-- Norms and reductions.
do
	local x, w = randn({ 4, 16 }), randn({ 16 }, 1, 0.1)
	check("rms_norm", { x }, function(g, p) return g:rms_norm(p[1], 1e-6) end)
	check("rms_norm * weight", { x, w }, function(g, p)
		return g:mul(g:rms_norm(p[1], 1e-6), p[2])
	end)
	check("sum_rows", { x }, function(g, p) return g:sum_rows(p[1]) end)
	check("sum", { x }, function(g, p) return g:sum(p[1]) end, { scalar = true })
end

-- Matrix products.
do
	local w, x = randn({ 5, 8 }), randn({ 3, 8 })
	check("mul_mat", { w, x }, function(g, p) return g:mul_mat(p[1], p[2]) end)
	check("mul_mat lowered", { w, x }, function(g, p) return g:mul_mat(p[1], p[2]) end,
	    { lower = true })
	local y = randn({ 8, 4 })
	check("matmul", { x, y }, function(g, p) return g:matmul(p[1], p[2]) end)
	-- Grouped heads: k has 2 heads, q has 4.
	local k, q = randn({ 2, 6, 8 }), randn({ 4, 3, 8 })
	check("mul_mat broadcast", { k, q }, function(g, p) return g:mul_mat(p[1], p[2]) end)
	check("mul_mat broadcast lowered", { k, q }, function(g, p) return g:mul_mat(p[1], p[2]) end,
	    { lower = true })
end

-- out_prod forward: (K, M) and (K, N) give b^T a.
do
	local a = be:tensor("f32", { 2, 3 }, { 1, 2, 3, 4, 5, 6 })
	local b = be:tensor("f32", { 2, 2 }, { 1, 0, 2, 1 })
	local g <close> = be:graph()
	local r = g:out_prod(a, b)
	g:compute(r)
	local got, want = r:get(), { 9, 12, 15, 4, 5, 6 }
	local good = #got == 6
	for i = 1, 6 do
		good = good and got[i] == want[i]
	end
	ok(good and r:shape()[1] == 2 and r:shape()[2] == 3, "out_prod")
end

-- Views.
do
	local x = randn({ 2, 3, 4 })
	check("reshape", { x }, function(g, p) return g:reshape(p[1], { 6, 4 }) end)
	check("permute", { x }, function(g, p) return g:permute(p[1], { 2, 1, 3 }) end)
	check("transpose", { x }, function(g, p) return g:transpose(p[1]) end)
	check("cont", { x }, function(g, p) return g:cont(g:permute(p[1], { 3, 1, 2 })) end)
	check("view", { x }, function(g, p)
		return g:view(p[1], { 2, 3, 2 }, { 48, 16 }, 8)
	end)
	local e = randn({ 10, 4 })
	local ids = be:tensor("i32", { 5 }, { 3, 0, 9, 3, 7 })
	check("get_rows", { e }, function(g, p) return g:get_rows(p[1], ids) end)
end

-- Softmax with a causal mask and a scale, and swiglu.
do
	local s = randn({ 2, 4, 4 })
	local mask = be:tensor("f32", { 4, 4 })
	mask:fill_causal(0)
	check("soft_max", { s }, function(g, p) return g:soft_max(p[1]) end)
	check("soft_max mask scale", { s }, function(g, p) return g:soft_max(p[1], mask, 0.5) end)
	local a, b = randn({ 3, 6 }), randn({ 3, 6 })
	check("swiglu_split", { a, b }, function(g, p) return g:swiglu_split(p[1], p[2]) end)
end

-- Rotary embedding: neox and normal, with frequency factors.
do
	local x = randn({ 5, 2, 8 })
	local pos = be:tensor("i32", { 5 }, { 0, 1, 2, 7, 30 })
	local freq = be:tensor("f32", { 4 }, { 1, 2, 4, 8 })
	check("rope neox", { x }, function(g, p)
		return g:rope(p[1], pos, { mode = "neox", base = 10000 })
	end)
	check("rope normal freq", { x }, function(g, p)
		return g:rope(p[1], pos, { mode = "normal", base = 500 }, freq)
	end)
	check("rope partial", { x }, function(g, p)
		return g:rope(p[1], pos, { mode = "neox", n_dims = 4, base = 10000 })
	end)
end

-- Cross-entropy, with rows picked by get_rows as the trainer does.
do
	local logits = randn({ 4, 7 })
	local labels = be:tensor("f32", { 3, 7 })
	labels:fill_onehot({ 2, 6, 0 })
	local rows = be:tensor("i32", { 3 }, { 0, 2, 3 })
	check("cross_entropy_loss", { logits }, function(g, p)
		return g:cross_entropy_loss(g:get_rows(p[1], rows), labels)
	end, { scalar = true })
end

-- Attention as the transformer builds it, with grouped heads.
do
	local transformer = require("lgml.transformer")
	local q, k, v = randn({ 3, 4, 8 }), randn({ 3, 2, 8 }), randn({ 3, 2, 8 })
	local mask = be:tensor("f32", { 3, 3 })
	mask:fill_causal(0)
	check("attention", { q, k, v }, function(g, p)
		return transformer.attend(g, p[1], p[2], p[3], mask, 1 / math.sqrt(8))
	end)
	check("attention lowered", { q, k, v }, function(g, p)
		return transformer.attend(g, p[1], p[2], p[3], mask, 1 / math.sqrt(8))
	end, { lower = true })
end

-- A frozen quantized weight between the loss and a trainable leaf. The
-- CPU quantizes x in the forward pass, so the check compares with the
-- gradient through an f32 copy of the weight.
do
	local wq, wf = be:tensor("q8_0", { 6, 32 }), be:tensor("f32", { 6, 32 })
	local g <close> = be:graph()
	g:compute(g:cpy(randn({ 6, 32 }), wq))
	local g1 <close> = be:graph()
	g1:compute(g1:cpy(wq, wf))
	local x, r = randn({ 3, 32 }), randn({ 3, 6 })
	x:requires_grad()
	local function grad_through(w, lower)
		x:grad():zero()
		local g2 <close> = be:graph()
		local loss = weighted(g2, g2:mul_mat(w, x), r)
		loss:backward({ lower = lower })
		g2:compute(loss)
		return x:grad():get()
	end
	local want = grad_through(wf)
	for _, lower in ipairs({ false, true }) do
		local got, worst = grad_through(wq, lower), 0
		for i = 1, #want do
			worst = math.max(worst, math.abs(got[i] - want[i]))
		end
		ok(worst < 1e-4, ("mul_mat q8_0 weight%s: max error %.2e"):format(lower and " lowered" or "",
		    worst))
	end
	x:requires_grad(false)
end

-- Ops without a gradient raise an error, not an abort.
do
	local x = randn({ 2, 4 })
	x:requires_grad()
	for _, op in ipairs({ "sigmoid", "tanh", "gelu", "mean" }) do
		local g <close> = be:graph()
		local loss = g:sum(g[op](g, x))
		local good, err = pcall(loss.backward, loss)
		ok(not good and tostring(err):find(op, 1, true) ~= nil, op .. " has no gradient: " ..
		    tostring(err))
	end
	local g <close> = be:graph()
	local good, err = pcall(function()
		local q = randn({ 2, 4, 4 })
		local kv = be:tensor("f16", { 2, 4, 4 })
		q:requires_grad()
		local loss = g:sum(g:flash_attn(q, kv, kv))
		loss:backward()
	end)
	ok(not good and tostring(err):find("flash", 1, true) ~= nil, "flash_attn has no gradient")
	local g2 <close> = be:graph()
	local y = randn({ 3 })
	local good2, err2 = pcall(function() g2:sum(g2:sqr(y)):backward() end)
	ok(not good2 and tostring(err2):find("trainable", 1, true) ~= nil, "no trainable leaf")
	x:requires_grad(false)
	ok(not x:is_param(), "requires_grad(false) freezes")
end

-- Gradients add up over computes; zero clears them.
do
	local w = randn({ 4 })
	w:requires_grad()
	w:grad():zero()
	local g <close> = be:graph()
	local loss = g:sum(g:scale(w, 3))
	loss:backward()
	g:compute(loss)
	g:compute(loss)
	local gv = w:grad():get()
	ok(math.abs(gv[1] - 6) < 1e-6 and math.abs(gv[4] - 6) < 1e-6, "gradients accumulate")
	w:grad():zero()
	ok(w:grad():get()[1] == 0, "zero clears the gradient")
	local good = pcall(loss.backward, loss)
	ok(not good, "a graph takes one backward pass")
	w:requires_grad(false)
end

-- Optimizer steps against the formulas.
do
	local p = be:tensor("f32", { 3 }, { 1, -2, 3 })
	p:requires_grad()
	local g <close> = be:graph()
	local loss = g:sum(g:sqr(p))
	loss:backward()
	local sgd = optim.sgd({ p }, { lr = 0.1, weight_decay = 0.5 })
	sgd:zero_grad()
	g:compute(loss)
	sgd:step()
	-- p - lr * (grad + wd * p): grad = 2p
	local want = { 1 - 0.1 * (2 + 0.5), -2 - 0.1 * (-4 - 1), 3 - 0.1 * (6 + 1.5) }
	local got = p:get()
	local good = true
	for i = 1, 3 do
		good = good and math.abs(got[i] - want[i]) < 1e-5
	end
	ok(good, "sgd step")

	p:set({ 1, -2, 3 })
	local adam = optim.adamw({ p }, { lr = 0.01, betas = { 0.9, 0.99 }, eps = 1e-8,
	    weight_decay = 0.1 })
	local m, v, x = { 0, 0, 0 }, { 0, 0, 0 }, { 1, -2, 3 }
	good = true
	for t = 1, 3 do
		adam:zero_grad()
		g:compute(loss)
		adam:step()
		for i = 1, 3 do
			local gr = 2 * x[i]
			m[i] = 0.9 * m[i] + 0.1 * gr
			v[i] = 0.99 * v[i] + 0.01 * gr * gr
			local mh, vh = m[i] / (1 - 0.9 ^ t), v[i] / (1 - 0.99 ^ t)
			x[i] = x[i] * (1 - 0.01 * 0.1) - 0.01 * mh / (math.sqrt(vh) + 1e-8)
		end
	end
	got = p:get()
	for i = 1, 3 do
		good = good and math.abs(got[i] - x[i]) < 1e-5
	end
	ok(good, "adamw steps")
	ok(adam.t == 3, "adamw counts steps")
	p:requires_grad(false)
end

-- A graph grows past its size hint: forward, backward and optimizer
-- steps in graphs made with room for 8 tensors match a large graph.
do
	local L, W = 40, 16
	local ws, x0, r = {}, randn({ 3, W }), randn({ 3, W })
	for i = 1, L do
		ws[i] = randn({ W, W }, 0, 0.3)
	end
	local function run(size)
		local ps = {}
		for i = 1, L do
			ps[i] = be:tensor("f32", { W, W }, ws[i]:get())
			ps[i]:requires_grad()
		end
		local g = be:graph(size)
		local x = x0
		for i = 1, L do
			x = g:add(g:silu(g:mul_mat(ps[i], x)), x)
		end
		local loss = g:sum(g:mul(g:rms_norm(x, 1e-6), r))
		loss:backward()
		g:compute(loss)
		g:compute(loss)
		local step = be:graph(size)
		local hp = be:tensor("f32", { 2 }, { 1, 0 })
		local nodes = {}
		for i = 1, L do
			nodes[i] = step:opt_step_sgd(ps[i], ps[i]:grad(), hp)
		end
		step:compute(table.unpack(nodes))
		local out = { loss = loss:get()[1], p = ps[1]:get(), q = ps[L]:get(), g = g }
		step:close()
		for i = 1, L do
			ps[i]:requires_grad(false)
		end
		return out
	end
	local small, big = run(8), run(1 << 16)
	ok(small.loss == big.loss, ("grown graph loss %g"):format(small.loss))
	ok(maxdiff(small.p, big.p) == 0 and maxdiff(small.q, big.q) == 0 and
	    maxdiff(small.p, ws[1]:get()) > 1e-4, "grown graph gradients and steps match")
	small.g:close()
	big.g:close()
	ok(not pcall(be.graph, be, (1 << 20) + 1), "size hint above the limit raises")
end

io.stderr:write("finite differences, max scaled error:\n")
for _, r in ipairs(report) do
	io.stderr:write(("  %-30s %.2e\n"):format(r[1], r[2]))
end
print("1.." .. n)
os.exit(failed and 1 or 0)
