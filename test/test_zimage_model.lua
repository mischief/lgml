-- Z-Image with real weights. Output is TAP. LGML_ZIMAGE names a
-- directory with z_image_turbo-Q8_0.gguf, Qwen3-4B-Instruct-2507-Q8_0.gguf
-- and ae.safetensors; the test skips without it. LGML_BACKEND and
-- LGML_TEXT_BACKEND pick the backends of the DiT and the text encoder.
-- The expected values come from stable-diffusion.cpp on Vulkan with the
-- same files.
local lgml = require("lgml")

local dir = os.getenv("LGML_ZIMAGE")
if not dir or dir == "" then
	print("1..0 # SKIP LGML_ZIMAGE not set")
	os.exit(0)
end

local pipeline = require("lgml.image.pipeline")
local zimage = require("lgml.image.zimage")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local tb = os.getenv("LGML_TEXT_BACKEND")
local p = pipeline.load({
	dit = dir .. "/z_image_turbo-Q8_0.gguf", llm = dir .. "/Qwen3-4B-Instruct-2507-Q8_0.gguf",
	vae = dir .. "/ae.safetensors", backend = os.getenv("LGML_BACKEND"),
	text_backend = tb ~= "" and tb or nil,
})
local prompt = "a red fox sitting in a snowy forest at sunrise, detailed fur, soft golden light"

-- Text encoder: 25 tokens; hidden states at (token, dim) points.
local ctx, nt, ne = p.te:encode(prompt)
ok(nt == 25 and ne == 2560, "template and prompt give 25 tokens")
local want = {
	{ 10, 0, 323.75916 }, { 10, 1, -11.12397 }, { 10, 2, -13.77580 }, { 10, 3, 13.53832 },
	{ 10, 4, 362.05194 }, { 24, 100, 0.86552 }, { 24, 101, 5.39040 }, { 24, 102, -14.04603 },
}
local good = true
for _, w in ipairs(want) do
	local v = string.unpack("<f", ctx, (w[1] * ne + w[2]) * 4 + 1)
	good = good and math.abs(v - w[3]) <= 0.01 * math.abs(w[3]) + 0.2
end
ok(good, "hidden states match the reference")

-- One DiT pass at 256x256 on seed 42 noise, sigma 1: channel means and
-- mean magnitudes.
local ref_mean = { 0.0234, -0.3114, -0.1336, -0.0806, 0.0393, -0.3550, 0.1073, -0.0929,
	-0.0108, -0.2457, -0.0938, -0.4050, 0.0106, 1.0852, 0.4836, 0.2842 }
local ref_abs = { 1.1074, 1.3835, 1.0092, 0.9326, 0.8805, 1.0193, 1.2777, 1.1639,
	1.0696, 1.0473, 1.0085, 1.0869, 1.2082, 1.3739, 1.1662, 1.1514 }
local inp = p.dit:inputs(nt, 32, 32)
inp.x:set(lgml.randn_philox(42, 16 * 32 * 32))
inp.ctx:set(ctx)
inp.tfreq:set(zimage.timestep_features(0))
local g = p.be:graph()
local out = p.dit:forward(g, inp)
g:compute(out)
local v = out:get()
local worst = 0
for c = 1, 16 do
	local s, a = 0, 0
	for i = (c - 1) * 1024 + 1, c * 1024 do
		s, a = s + v[i], a + math.abs(v[i])
	end
	worst = math.max(worst, math.abs(s / 1024 - ref_mean[c]), math.abs(a / 1024 - ref_abs[c]))
end
ok(worst < 0.03, ("DiT output matches the reference (worst %.4f)"):format(worst))
g:close()
zimage.close_inputs(inp)

-- Whole runs: same seed, same PNG; a different seed, a different image.
local req = { prompt = prompt, width = 256, height = 256, steps = 4, seed = 1 }
local a = p:generate(req)
local b, tm = p:generate(req)
ok(a and b and a:png() == b:png() and tm.steps == 4, "a seed gives the same image")
req.seed = 2
local c = p:generate(req)
ok(c:png() ~= a:png() and c.width == 256 and c.height == 256, "another seed gives another image")
local t = a:tensor(lgml.backend("CPU")):get()
local mean, var = 0, 0
for _, x in ipairs(t) do
	mean = mean + x
end
mean = mean / #t
for _, x in ipairs(t) do
	var = var + (x - mean) ^ 2
end
ok(var / #t > 0.01, "the image is not flat")

-- Image-to-image from that image, and a cancel.
local e, tm2 = p:generate({ prompt = "a gray wolf", width = 256, height = 256, steps = 4,
    seed = 3, image = a, strength = 0.5 })
ok(e and tm2.steps == 3, "image-to-image runs floor(4 * 0.5) + 1 steps")
local r, err = p:generate({ prompt = prompt, width = 256, height = 256, steps = 4, seed = 1,
    step = function(i) return i < 2 end })
ok(r == nil and err == "cancelled", "a step callback can stop the run")

p:close()
print("1.." .. n)
os.exit(failed and 1 or 0)
