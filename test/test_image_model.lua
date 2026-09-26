-- The FLUX VAE with real weights. Output is TAP. Set LGML_ZIMAGE_VAE to
-- ae.safetensors; the test skips without it. LGML_DEVICE picks the
-- backend (default: the best one), LGML_VAE_SIZE the image side (512).
-- LGML_VAE_REF names a vae_ref binary (scripts/vae_ref.cpp) for a
-- comparison with stable-diffusion.cpp.
local lgml = require("lgml")
local vae = require("lgml.image.vae")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local path = os.getenv("LGML_ZIMAGE_VAE")
if not path or path == "" then
	print("1..0 # SKIP LGML_ZIMAGE_VAE not set")
	os.exit(0)
end

local dev = os.getenv("LGML_DEVICE")
local be <close> = assert(lgml.backend(dev ~= "" and dev or nil))
local S = tonumber(os.getenv("LGML_VAE_SIZE") or "") or 512

-- A test card: a sky gradient, a disc, a square, stripes and a checker.
local function card(s)
	local v, plane = {}, s * s
	for y = 0, s - 1 do
		for x = 0, s - 1 do
			local u, w = x / s, y / s
			local r, g, b = 0.3 + 0.4 * w, 0.5 + 0.3 * w, 0.9 - 0.3 * w
			if (u - 0.35) ^ 2 + (w - 0.4) ^ 2 < 0.04 then
				r, g, b = 0.9, 0.2 + 0.3 * u, 0.1
			elseif u > 0.6 and u < 0.85 and w > 0.55 and w < 0.8 then
				r, g, b = 0.1, 0.2, 0.7 + 0.2 * math.sin(u * 40)
			elseif w > 0.85 then
				local k = (x // 8 + y // 8) % 2
				r, g, b = k, k, k
			elseif (x + y) // 12 % 2 == 0 and u > 0.7 and w < 0.4 then
				r, g, b = 0.2, 0.8, 0.3
			end
			local i = y * s + x + 1
			v[i], v[plane + i], v[2 * plane + i] = r * 2 - 1, g * 2 - 1, b * 2 - 1
		end
	end
	return v
end

local function psnr(a, b)
	local se = 0
	for i = 1, #a do
		local d = (math.max(-1, math.min(1, a[i])) - b[i]) / 2
		se = se + d * d
	end
	return 10 * math.log(#a / se, 10)
end

local m = vae.load(be, path)
ok(m.cfg.levels == 4 and m.cfg.blocks == 3 and m.cfg.z == 16, "FLUX layout: 4 levels, 16 latent channels")

local xv = card(S)
local x <close> = be:tensor("f32", { 1, 3, S, S }, xv)
local t0 = lgml.clock()
local g <close> = be:graph()
local z = m:encode(g, x)
local zs = vae.scale(g, z)
g:compute(z, zs)
local te = lgml.clock() - t0
local zv, sum, sq = zs:get(), 0, 0
for _, v in ipairs(zv) do
	sum, sq = sum + v, sq + v * v
end
local sd = math.sqrt(sq / #zv - (sum / #zv) ^ 2)
ok(sd > 0.3 and sd < 3, ("scaled latent std %.3f"):format(sd))

local zl <close> = be:tensor("f32", z:shape(), z:bytes())
t0 = lgml.clock()
local g2 <close> = be:graph()
local y = m:decode(g2, zl)
g2:compute(y)
local td = lgml.clock() - t0
local yv = y:get()
local p = psnr(yv, xv)
print(("# %s %dx%d: encode %.2fs (%d MB), decode %.2fs (%d MB)"):format(be:name(), S, S, te,
    g:buffer_size() >> 20, td, g2:buffer_size() >> 20))
ok(p > 27, ("round trip PSNR %.2f dB"):format(p))

local tile = S // 16
local yt, peak = m:decode_tiled(zl, { tile = tile, overlap = tile // 4 })
local pt = psnr(yt:get(), yv)
ok(pt > 30, ("decode_tiled (tile %d) vs decode PSNR %.2f dB, peak %d MB"):format(tile, pt, peak >> 20))
ok(peak < g2:buffer_size(), "tiles use less compute memory")
yt:close()

local ref = os.getenv("LGML_VAE_REF")
if ref and ref ~= "" then
	local zf, yf = os.tmpname(), os.tmpname()
	local f = assert(io.open(zf, "wb"))
	f:write(z:bytes())
	f:close()
	local refdev = be:name() == "CPU" and "cpu" or "vk"
	local cmd = ("VAE_DEV=%s %s decode %s %s %s 1 %d %d >/dev/null 2>&1"):format(refdev, ref, path,
	    zf, yf, S // 8, S // 8)
	ok(os.execute(cmd), "vae_ref runs")
	f = assert(io.open(yf, "rb"))
	local s = f:read("a")
	f:close()
	os.remove(zf)
	os.remove(yf)
	local rv = {}
	for i = 1, #s // 4 do
		rv[i] = string.unpack("<f", s, 4 * i - 3)
	end
	local pr = psnr(rv, yv)
	-- The reference keeps kernels in f16; the CPU backend gets about 80 dB.
	ok(#rv == #yv and pr > 50, ("decode vs stable-diffusion.cpp PSNR %.1f dB"):format(pr))
end

m:close()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
