--- Z-Image text-to-image and image-to-image.
-- The text encoder (`lgml.image.text_encoder`), the DiT
-- (`lgml.image.zimage`) and the FLUX autoencoder (`lgml.image.vae`)
-- share one backend. The noise is the Philox "cuda" RNG of
-- stable-diffusion.cpp, so a seed gives the same start latent. The
-- latent stays on the device: each step runs the DiT graph once for
-- each condition, then a small graph adds (sigma_next - sigma) * v to
-- the latent. The DiT graphs are built once per request and run again
-- for every step with new timestep features.
-- @module lgml.image.pipeline
local lgml = require("lgml")
local zimage = require("lgml.image.zimage")
local text_encoder = require("lgml.image.text_encoder")
local sampler = require("lgml.image.sampler")

local M = {}
M.__index = M

--- Default step count (Z-Image-Turbo).
M.STEPS = 8
--- Pixels per latent position.
M.FACTOR = 8
--- Image sides must be multiples of this: the VAE factor times the patch.
M.MULTIPLE = 16
--- The largest image side.
M.MAX_SIDE = 2048

--- Load the three models.
-- @tparam table opts the options:
--
-- - `dit`: the Z-Image gguf
-- - `llm`: the Qwen3-4B gguf
-- - `vae`: the FLUX ae.safetensors
-- - `backend`: backend name (default: the best one)
-- @treturn Pipeline the pipeline
-- @raise when a model does not load
function M.load(opts)
	local be = assert(lgml.backend(opts.backend))
	local vae = require("lgml.image.vae")
	local p = setmetatable({ be = be, vae_mod = vae }, M)
	p.te = text_encoder.load(be, opts.llm)
	p.dit = zimage.load(be, opts.dit)
	p.vae = vae.load(be, opts.vae)
	return p
end

--- A loaded pipeline, from `load`.
-- Fields: be, te, dit and vae.
-- @type Pipeline

--- Close the models.
function M:close()
	self.te:close()
	self.dit:close()
	self.vae:close()
end

--- Check the size of an image.
-- @tparam integer width pixels
-- @tparam integer height pixels
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string the error
function M.check_size(width, height)
	for _, v in ipairs({ width, height }) do
		if math.type(v) ~= "integer" or v < M.MULTIPLE or v > M.MAX_SIDE or v % M.MULTIPLE ~= 0 then
			return nil, ("image sides must be multiples of %d from %d to %d"):format(M.MULTIPLE,
			    M.MULTIPLE, M.MAX_SIDE)
		end
	end
	return true
end

-- Set x = noise * s0 + init * (1 - s0). Without init, the reference
-- starts from the latent shift.
local function start_latent(self, x, seed, s0, init)
	local be = self.be
	local shape = x:shape()
	local n = shape[1] * shape[2] * shape[3] * shape[4]
	local noise <close> = be:tensor("f32", shape, lgml.randn_philox(seed, n))
	local g <close> = be:graph()
	local v = g:scale(noise, s0)
	if init then
		v = g:add(v, g:scale(init, 1 - s0))
	elseif s0 ~= 1 then
		local z <close> = be:tensor("f32", shape)
		z:fill("const", self.vae_mod.SHIFT)
		v = g:add(v, g:scale(z, 1 - s0))
	end
	g:compute(g:cpy(v, x))
end

-- The model latent of an image: VAE mean, then the latent scale.
local function encode_image(self, image, width, height)
	if image.width ~= width or image.height ~= height then
		image = image:resize(width, height)
	end
	local t <close> = image:tensor(self.be)
	local g <close> = self.be:graph()
	local z = self.vae_mod.scale(g, self.vae:encode(g, t))
	g:compute(z)
	return self.be:tensor("f32", z:shape(), z:bytes())
end

-- The DiT graph of one condition: its inputs and the node that writes
-- the velocity into v.
local function dit_graph(self, ctx, n, x, v)
	local H, W = x:shape()[3], x:shape()[4]
	local inp = self.dit:inputs(n, H, W, x)
	inp.ctx:set(ctx)
	local g = self.be:graph()
	local out = self.dit:forward(g, inp)
	return { g = g, inp = inp, w = g:cpy(out, v) }
end

--- Generate one image.
-- req fields: prompt; `negative_prompt` (""), for the unconditional
-- pass; width and height, multiples of 16 (1024); steps (8); seed (42);
-- cfg, the guidance scale, where 1 runs one pass per step (1); shift,
-- the flow time shift (3); image, an `lgml.image` to start from, resized
-- to width x height; strength, 0 to 1, the part of the schedule that
-- image-to-image runs (0.75); step, a function(i, n) called after each
-- step, which returns false to stop; latent, true to also return the
-- final model latent as f32 bytes.
-- @tparam table req the request
-- @treturn[1] image the `lgml.image`
-- @treturn[1] table encode_ms, sample_ms, decode_ms, total_ms, steps
-- (the steps run) and latent
-- @treturn[2] nil
-- @treturn[2] string "cancelled" or an error
-- @raise on a bad request
function M:generate(req)
	local be = self.be
	local width, height = req.width or 1024, req.height or 1024
	assert(M.check_size(width, height))
	local steps = req.steps or M.STEPS
	local cfg = req.cfg or 1
	local t0 = lgml.clock()
	local ctx_c, n_c = self.te:encode(req.prompt)
	local ctx_u, n_u
	if cfg ~= 1 then
		ctx_u, n_u = self.te:encode(req.negative_prompt or "")
	end
	local sig = sampler.sigmas(steps, req.shift)
	local init
	if req.image then
		sig = sampler.img2img(sig, req.strength or 0.75)
		init = encode_image(self, req.image, width, height)
	end
	local t1 = lgml.clock()

	local C, H, W = self.dit.hp.in_ch, height // M.FACTOR, width // M.FACTOR
	local x <close> = be:tensor("f32", { 1, C, H, W })
	local vc <close> = be:tensor("f32", { 1, C, H, W })
	local vu <close> = be:tensor("f32", { 1, C, H, W })
	local dt <close> = be:tensor("f32", { 1 })
	start_latent(self, x, req.seed or 42, sig[1], init)
	if init then
		init:close()
	end
	local runs = { dit_graph(self, ctx_c, n_c, x, vc) }
	if ctx_u then
		runs[2] = dit_graph(self, ctx_u, n_u, x, vu)
	end
	local ge <close> = be:graph()
	local v = vc
	if ctx_u then
		v = ge:add(vu, ge:scale(ge:sub(vc, vu), cfg))
	end
	local upd = ge:cpy(ge:add(x, ge:mul(v, dt)), x)
	local n, done = #sig - 1, 0
	local ok, err = pcall(function()
		for i = 1, n do
			local tf = zimage.timestep_features(sampler.timestep(sig[i]))
			for _, r in ipairs(runs) do
				r.inp.tfreq:set(tf)
				r.g:compute(r.w)
			end
			dt:set({ sig[i + 1] - sig[i] })
			ge:compute(upd)
			done = i
			if req.step and req.step(i, n) == false then
				return
			end
		end
	end)
	for _, r in ipairs(runs) do
		r.g:close()
		zimage.close_inputs(r.inp)
	end
	-- Graph memory stays until the Lua node tensors are collected.
	runs = nil
	collectgarbage()
	if not ok then
		error(err, 0)
	end
	if done < n then
		return nil, "cancelled"
	end
	local t2 = lgml.clock()
	local latent = req.latent and x:bytes() or nil
	local gd = be:graph()
	local px = self.vae:decode(gd, self.vae_mod.unscale(gd, x))
	gd:compute(px)
	local image = lgml.image_from(px)
	gd:close()
	gd, px = nil, nil
	collectgarbage()
	local t3 = lgml.clock()
	return image, {
		encode_ms = (t1 - t0) * 1000, sample_ms = (t2 - t1) * 1000,
		decode_ms = (t3 - t2) * 1000, total_ms = (t3 - t0) * 1000, steps = n,
		latent = latent,
	}
end

return M
