--- Diffusion head and DPM-Solver++ (2M, v-prediction, cosine betas).
-- One graph runs all steps for one frame. The noise, the two
-- conditions and the latent stay on the device.
-- @module lgml.vibevoice.diffusion
local lgml = require("lgml")

local M = {}

-- Cosine schedule with clipped betas. Returns alphas_cumprod (index t + 1).
local function alphas_cumprod(n)
	local function bar(t)
		return math.cos((t + 0.008) / 1.008 * math.pi / 2) ^ 2
	end
	local ac, prod = {}, 1
	for i = 0, n - 1 do
		local beta = math.min(1 - bar((i + 1) / n) / bar(i / n), 0.999)
		prod = prod * (1 - beta)
		ac[i + 1] = prod
	end
	return ac
end

--- Timesteps and update coefficients.
-- Step i sets x0 = sa * z - s1 * v and z = A * z + B * x0 + C * x0_prev.
-- @int steps inference steps
-- @int train_steps training steps of the cosine schedule
-- @treturn {table,...} one entry per step with fields t, sa, s1, A, B, C
function M.schedule(steps, train_steps)
	local ac = alphas_cumprod(train_steps)
	local ts = {}
	for i = 0, steps - 1 do
		ts[i + 1] = math.floor((train_steps - 1) * (steps - i) / steps + 0.5)
	end
	local function ab(t)
		if t == nil then
			return 1, 0, 20
		end
		local a, s = math.sqrt(ac[t + 1]), math.sqrt(1 - ac[t + 1])
		return a, s, math.log(a / s)
	end
	local out = {}
	for i = 1, steps do
		local at, st, lt = ab(ts[i])
		local ap, sp, lp = ab(ts[i + 1])
		local h, eh = lp - lt, math.exp(-(lp - lt)) - 1
		local c = { t = ts[i], sa = ac[ts[i] + 1] ^ 0.5, s1 = (1 - ac[ts[i] + 1]) ^ 0.5 }
		c.A, c.B, c.C = sp / st, -ap * eh, 0
		if i > 1 and i < steps then
			local _, _, ls = ab(ts[i - 1])
			local r = (lt - ls) / h
			c.B = -ap * eh - 0.5 * ap * eh / r
			c.C = 0.5 * ap * eh / r
		end
		out[i] = c
	end
	return out
end

--- Sinusoidal embedding of each timestep.
-- @tparam {table,...} sched schedule from schedule
-- @int dim embedding size
-- @treturn {number,...} flat list: per step, dim / 2 cos values, then
-- dim / 2 sin values
function M.timestep_table(sched, dim)
	local t, half = {}, dim // 2
	for _, c in ipairs(sched) do
		for i = 0, half - 1 do
			local a = c.t * math.exp(-math.log(10000) * i / half)
			t[#t + 1] = math.cos(a)
		end
		for i = 0, half - 1 do
			local a = c.t * math.exp(-math.log(10000) * i / half)
			t[#t + 1] = math.sin(a)
		end
	end
	return t
end

--- Collect the diffusion head weights.
-- @tparam lgml.weights w loaded gguf weights
-- @int n_layer head layer count
-- @treturn table head weights
-- @raise when a tensor is missing
function M.load(w, n_layer)
	local function get(name)
		return assert(w:get(name), name)
	end
	local p = { layers = {} }
	p.noisy_proj = get("pred.noisy_proj.weight")
	p.cond = get("pred.cond.weight")
	p.t0 = get("pred.t_emb.0.weight")
	p.t2 = get("pred.t_emb.2.weight")
	p.final_adaln = get("pred.final.adaln.weight")
	p.final_linear = get("pred.final.linear.weight")
	for i = 0, n_layer - 1 do
		local b = "pred.layers." .. i .. "."
		p.layers[i + 1] = {
			adaln = get(b .. "adaln.weight"),
			norm = get(b .. "norm.weight"),
			gate = get(b .. "ffn.gate_proj.weight"),
			up = get(b .. "ffn.up_proj.weight"),
			down = get(b .. "ffn.down_proj.weight"),
		}
	end
	return p
end

-- Rows j * d .. (j + 1) * d - 1 of an adaLN weight (k * d, d).
local function chunk(g, wt, j, d)
	local row = lgml.row_size(wt:type(), d)
	return g:view(wt, { d, d }, { row }, j * d * row)
end

-- Modulation for all steps: silu(c) through each chunk of the adaLN
-- weight. c is (steps, 2, d). Returns shift, weight and gate nodes.
local function modulation(g, wt, sc, norm, ones, d, n)
	local out = {}
	for j = 0, n - 1 do
		out[j + 1] = g:mul_mat(chunk(g, wt, j, d), sc)
	end
	local ws
	if norm then
		ws = g:add(g:mul(out[2], norm), norm)
	else
		ws = g:add(out[2], ones)
	end
	return out[1], ws, out[3]
end

local function step_view(g, t, s, d)
	local row = lgml.row_size("f32", d)
	return g:view(t, { 2, d }, { row }, (s - 1) * 2 * row)
end

-- One head evaluation for step s. z is (1, vae). Returns v (2, vae).
local function head(g, p, mods, z, s, d, eps)
	local x = g:mul_mat(p.noisy_proj, z)
	for i, L in ipairs(p.layers) do
		local m = mods[i]
		local h = g:mul(step_view(g, m.ws, s, d), g:rms_norm(x, eps))
		h = g:add(h, step_view(g, m.shift, s, d))
		local ff = g:mul(g:silu(g:mul_mat(L.gate, h)), g:mul_mat(L.up, h))
		ff = g:mul(g:mul_mat(L.down, ff), step_view(g, m.gate, s, d))
		x = g:add(ff, x)
	end
	local m = mods.final
	local h = g:mul(step_view(g, m.ws, s, d), g:rms_norm(x, eps))
	h = g:add(h, step_view(g, m.shift, s, d))
	return g:mul_mat(p.final_linear, h)
end

--- Build the graph for one frame.
-- @tparam lgml.graph g graph
-- @tparam table p head weights from load
-- @tparam table io input leaves
-- io.noise: (1, vae).
-- io.cond: (2, d), rows positive and negative.
-- io.temb: (steps, d). io.zeros: (steps, 2, d). io.ones: (d).
-- io.latents: (frames, vae). io.fid: i32 (1).
-- @tparam {table,...} sched schedule from schedule
-- @number cfg classifier-free guidance scale
-- @tparam table hp hyperparameters: d, vae_dim, head_eps
-- @treturn lgml.tensor node that writes the latent into row fid of latents
-- @treturn {tensor,...} the guided v of each step
function M.graph(g, p, io, sched, cfg, hp)
	local d, vae, n = hp.d, hp.vae_dim, #sched
	local c = g:add(g:add(io.zeros, g:mul_mat(p.cond, io.cond)),
	    g:reshape(io.temb, { n, 1, d }))
	local sc = g:silu(c)
	local mods = {}
	for i, L in ipairs(p.layers) do
		local shift, ws, gate = modulation(g, L.adaln, sc, L.norm, nil, d, 3)
		mods[i] = { shift = shift, ws = ws, gate = gate }
	end
	local shift, ws = modulation(g, p.final_adaln, sc, nil, io.ones, d, 2)
	mods.final = { shift = shift, ws = ws }

	local row = lgml.row_size("f32", vae)
	local z, x0prev, vs = io.noise, nil, {}
	for s, k in ipairs(sched) do
		local v = head(g, p, mods, z, s, d, hp.head_eps)
		local vp = g:view(v, { 1, vae }, { row }, 0)
		local vn = g:view(v, { 1, vae }, { row }, row)
		v = g:add(g:scale(vn, 1 - cfg), g:scale(vp, cfg))
		vs[s] = v
		local x0 = g:add(g:scale(z, k.sa), g:scale(v, -k.s1))
		local zn = g:add(g:scale(z, k.A), g:scale(x0, k.B))
		if k.C ~= 0 then
			zn = g:add(zn, g:scale(x0prev, k.C))
		end
		z, x0prev = zn, x0
	end
	return g:set_rows(io.latents, z, io.fid), vs
end

return M
