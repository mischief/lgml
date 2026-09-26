--- Flow-matching schedule and Euler steps for Z-Image.
-- The math follows the DiscreteFlowDenoiser, the discrete scheduler
-- and the Euler sampler of stable-diffusion.cpp. The model predicts a
-- velocity v; one step is x = x + (sigma_next - sigma) * v. Values
-- round to f32 as in the reference.
-- @module lgml.image.sampler
local M = {}

--- Timestep count of the training schedule.
M.TIMESTEPS = 1000
--- Default time shift of Z-Image.
M.SHIFT = 3.0

local function f32(x)
	return (string.unpack("<f", string.pack("<f", x)))
end

--- Sigma of a discrete timestep: the shifted time (t + 1) / 1000.
-- @tparam number t the timestep, 0 to 999
-- @tparam[opt] number shift the time shift (default `SHIFT`)
-- @treturn number sigma, 0 to 1
function M.t_to_sigma(t, shift)
	shift = shift or M.SHIFT
	local s = f32(f32(t + 1) / M.TIMESTEPS)
	if shift == 1 then
		return s
	end
	return f32(f32(shift * s) / f32(1 + f32((shift - 1) * s)))
end

--- Sigmas of a run: steps values from 1 down, then 0.
-- The timesteps are evenly spaced from 999 to 0.
-- @tparam integer steps the step count, at least 1
-- @tparam[opt] number shift the time shift (default `SHIFT`)
-- @treturn {number,...} steps + 1 sigmas
function M.sigmas(steps, shift)
	assert(steps >= 1, "need at least one step")
	local t_max = M.TIMESTEPS - 1
	local out = {}
	if steps == 1 then
		out[1] = M.t_to_sigma(t_max, shift)
	else
		local step = f32(t_max / (steps - 1))
		for i = 0, steps - 1 do
			out[i + 1] = M.t_to_sigma(f32(t_max - f32(step * i)), shift)
		end
	end
	out[#out + 1] = 0
	return out
end

--- The sigmas that image-to-image runs.
-- The run skips the first steps and keeps floor(steps * strength) + 1
-- of them, at most steps.
-- @tparam {number,...} sigmas from `sigmas`
-- @tparam number strength 0 to 1; 1 starts from pure noise
-- @treturn {number,...} the tail of sigmas, from the start sigma
function M.img2img(sigmas, strength)
	local steps = #sigmas - 1
	local t_enc = math.floor(f32(steps * f32(strength)))
	if t_enc >= steps then
		t_enc = steps - 1
	end
	return table.move(sigmas, steps - t_enc, #sigmas, 1, {})
end

--- The model timestep of a sigma: 1000 * (1 - sigma).
-- @tparam number sigma the sigma
-- @treturn number the timestep
function M.timestep(sigma)
	return f32(M.TIMESTEPS - f32(sigma * M.TIMESTEPS))
end

--- One Euler step on Lua numbers, for tests and small latents.
-- With a guidance scale, v is vu + cfg * (vc - vu).
-- @tparam {number,...} x the latent
-- @tparam {number,...} vc the conditional velocity
-- @tparam number sigma the current sigma
-- @tparam number next the next sigma
-- @tparam[opt] {number,...} vu the unconditional velocity
-- @tparam[opt=1] number cfg the guidance scale
-- @treturn {number,...} the new latent
function M.euler(x, vc, sigma, next, vu, cfg)
	local dt, out = next - sigma, {}
	for i = 1, #x do
		local v = vc[i]
		if vu then
			v = vu[i] + (cfg or 1) * (vc[i] - vu[i])
		end
		out[i] = x[i] + dt * v
	end
	return out
end

return M
