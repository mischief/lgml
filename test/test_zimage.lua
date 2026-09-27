-- Z-Image parts on the CPU with a tiny random DiT. Output is TAP. The
-- noise and schedule values come from stable-diffusion.cpp.
local lgml = require("lgml")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local gguf = require("gguf")
local zimage = require("lgml.image.zimage")
local sampler = require("lgml.image.sampler")
local base64 = require("lgml.base64")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local function near(a, b, tol)
	if #a ~= #b then
		return false
	end
	for i = 1, #a do
		if math.abs(a[i] - b[i]) > tol then
			return false
		end
	end
	return true
end

local function floats(s)
	local t = {}
	for i = 1, #s // 4 do
		t[i] = string.unpack("<f", s, (i - 1) * 4 + 1)
	end
	return t
end

-- Philox noise: the "cuda" RNG of stable-diffusion.cpp.
local r = floats(lgml.randn_philox(42, 4))
ok(near(r, { 0.19401887, 2.1613739, -0.1720506, 0.8490601 }, 1e-7), "philox seed 42")
ok(floats(lgml.randn_philox(42, 1000))[4] == r[4], "philox draw i does not depend on n")
ok(floats(lgml.randn_philox(42, 4, 1))[1] ~= r[1], "philox offset changes the stream")
ok(not pcall(lgml.randn_philox, 1, 0), "philox rejects n = 0")

-- Schedule: discrete flow sigmas with shift 3.
local sig = sampler.sigmas(8)
ok(near(sig, { 1.0, 0.9474265, 0.8824981, 0.8002798, 0.6928045, 0.5463216, 0.3348869,
    0.0029942, 0 }, 1e-6), "8-step sigmas")
ok(near(sampler.sigmas(1), { 1.0, 0 }, 0), "1-step sigmas")
ok(near(sampler.sigmas(4, 1), { 1.0, 0.667, 0.334, 0.001, 0 }, 1e-6), "shift 1 is linear")
local i2i = sampler.img2img(sig, 0.75)
ok(#i2i == 8 and i2i[1] == sig[2], "strength 0.75 runs 7 of 8 steps")
ok(#sampler.img2img(sig, 1) == 9, "strength 1 runs every step")
ok(#sampler.img2img(sig, 0.1) == 2 and sampler.img2img(sig, 0.1)[1] == sig[8],
    "small strength runs one step")
ok(sampler.timestep(1) == 0 and sampler.timestep(0) == 1000 and sampler.timestep(0.25) == 750,
    "timestep is 1000 * (1 - sigma)")
ok(near(sampler.euler({ 1, 2 }, { 0.5, -1 }, 1, 0.5), { 0.75, 2.5 }, 1e-12), "euler step")
ok(near(sampler.euler({ 1 }, { 2 }, 1, 0.5, { 1 }, 3), { -1 }, 1e-12), "euler step with cfg")

-- Positions and timestep features.
local pos = zimage.positions(25, 2, 3)
ok(pos.T == 32 and pos.I == 32 and pos.n_img == 6, "positions pad to 32")
ok(pos.txt[1][1] == 1 and pos.txt[1][32] == 32 and pos.txt[2][32] == 0, "text positions")
ok(pos.img[1][1] == 33 and pos.img[2][6] == 1 and pos.img[3][6] == 2 and pos.img[3][5] == 1,
    "image positions")
ok(pos.img[1][7] == 0 and pos.img[2][32] == 0, "image pad positions are 0")
local tf = zimage.timestep_features(0)
ok(#tf == 256 and tf[1] == 1 and tf[128] == 1 and tf[129] == 0, "timestep features cos then sin")

local be = assert(lgml.backend("CPU"))

-- Patchify: tokens hold (row in patch, column in patch, channel).
do
	local C, H, W = 3, 6, 8
	local v = {}
	for c = 0, C - 1 do
		for y = 0, H - 1 do
			for x = 0, W - 1 do
				v[#v + 1] = c * 10000 + y * 100 + x
			end
		end
	end
	local t = be:tensor("f32", { 1, C, H, W }, v)
	local g = be:graph()
	local p = zimage.patchify(g, t)
	local u = zimage.unpatchify(g, p, H // 2, W // 2)
	g:compute(p, u)
	local P, bad = p:get(), 0
	for row = 0, 2 do
		for col = 0, 3 do
			for c = 0, C - 1 do
				for p1 = 0, 1 do
					for p2 = 0, 1 do
						local got = P[(row * 4 + col) * C * 4 + (p1 * 2 + p2) * C + c + 1]
						if got ~= c * 10000 + (2 * row + p1) * 100 + 2 * col + p2 then
							bad = bad + 1
						end
					end
				end
			end
		end
	end
	ok(bad == 0 and table.concat(p:shape(), "x") == "12x12", "patchify feature order")
	ok(near(u:get(), v, 0), "unpatchify undoes patchify")
end

-- A tiny random DiT: 2 heads of 128, dim 256, one refiner block each.
local function tiny_gguf(path)
	local seed = 1
	local function rnd(k, scale)
		local t = {}
		for i = 1, k do
			seed = (seed * 1103515245 + 12345) % 2147483648
			t[i] = (seed / 2147483648 - 0.5) * scale
		end
		return t
	end
	local function ones(k)
		local t = {}
		for i = 1, k do
			t[i] = 1
		end
		return t
	end
	local D, F, CAP, TM = 256, 512, 32, 64
	local ts = {
		{ "x_embedder.weight", { D, 64 }, rnd(D * 64, 0.2) }, { "x_embedder.bias", { D }, rnd(D, 0.1) },
		{ "cap_embedder.0.weight", { CAP }, ones(CAP) },
		{ "cap_embedder.1.weight", { D, CAP }, rnd(D * CAP, 0.2) },
		{ "cap_embedder.1.bias", { D }, rnd(D, 0.1) },
		{ "cap_pad_token", { D }, rnd(D, 1) }, { "x_pad_token", { D }, rnd(D, 1) },
		{ "t_embedder.mlp.0.weight", { TM, 256 }, rnd(TM * 256, 0.2) },
		{ "t_embedder.mlp.0.bias", { TM }, rnd(TM, 0.1) },
		{ "t_embedder.mlp.2.weight", { 256, TM }, rnd(256 * TM, 0.2) },
		{ "t_embedder.mlp.2.bias", { 256 }, rnd(256, 0.1) },
		{ "final_layer.adaLN_modulation.1.weight", { D, 256 }, rnd(D * 256, 0.1) },
		{ "final_layer.adaLN_modulation.1.bias", { D }, rnd(D, 0.1) },
		{ "final_layer.linear.weight", { 64, D }, rnd(64 * D, 0.2) },
		{ "final_layer.linear.bias", { 64 }, rnd(64, 0.1) },
	}
	local function block(p, mod)
		local list = {
			{ "attention.qkv.weight", { 3 * D, D }, 0.1 }, { "attention.out.weight", { D, D }, 0.1 },
			{ "feed_forward.w1.weight", { F, D }, 0.1 }, { "feed_forward.w2.weight", { D, F }, 0.1 },
			{ "feed_forward.w3.weight", { F, D }, 0.1 },
		}
		for _, e in ipairs(list) do
			ts[#ts + 1] = { p .. e[1], e[2], rnd(e[2][1] * e[2][2], e[3]) }
		end
		for _, name in ipairs({ "attention.q_norm.weight", "attention.k_norm.weight" }) do
			ts[#ts + 1] = { p .. name, { 128 }, ones(128) }
		end
		for _, name in ipairs({ "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2" }) do
			ts[#ts + 1] = { p .. name .. ".weight", { D }, ones(D) }
		end
		if mod then
			ts[#ts + 1] = { p .. "adaLN_modulation.0.weight", { 4 * D, 256 }, rnd(4 * D * 256, 0.1) }
			ts[#ts + 1] = { p .. "adaLN_modulation.0.bias", { 4 * D }, rnd(4 * D, 0.1) }
		end
	end
	block("context_refiner.0.", false)
	block("noise_refiner.0.", true)
	block("layers.0.", true)
	block("layers.1.", true)
	gguf.write(path, { { "general.architecture", gguf.STRING, "z_image" } }, ts)
end

local path = os.tmpname()
tiny_gguf(path)
local m = zimage.load(be, path)
os.remove(path)
ok(m.hp.dim == 256 and m.hp.n_head == 2 and m.hp.in_ch == 16 and m.hp.cap_dim == 32 and
    #m.layers == 2 and #m.ctx_ref == 1 and #m.noise_ref == 1, "tiny DiT sizes from shapes")

-- 5 text tokens and 3 x 3 patches: both pad paths run.
local inp = m:inputs(5, 6, 6)
inp.x:set(lgml.randn_philox(1, 16 * 36))
inp.ctx:set(lgml.randn_philox(2, 5 * 32))
inp.tfreq:set(zimage.timestep_features(250))
local g = be:graph()
local out, taps = m:forward(g, inp)
g:compute(out, taps.txt, taps.img)
local a = out:get()
local finite = true
for _, x in ipairs(a) do
	finite = finite and x == x and math.abs(x) < 1e6
end
ok(table.concat(out:shape(), "x") == "1x16x6x6" and finite, "tiny DiT output shape")
ok(table.concat(taps.txt:shape(), "x") == "32x256" and table.concat(taps.img:shape(), "x") == "32x256",
    "refiners run on padded tokens")
g:compute(out)
ok(near(out:get(), a, 0), "replay gives the same output")
inp.tfreq:set(zimage.timestep_features(750))
g:compute(out)
ok(not near(out:get(), a, 1e-6), "a new timestep changes the output")
inp.tfreq:set(zimage.timestep_features(250))
inp.ctx:set(lgml.randn_philox(3, 5 * 32))
g:compute(out)
ok(not near(out:get(), a, 1e-6), "new text changes the output")
g:close()
zimage.close_inputs(inp)

-- The pipeline plan: two DiT passes (one per condition), the VAE graphs
-- and three latents, with the text encoder on another backend.
do
	local pipeline = require("lgml.image.pipeline")
	local function scale(g2, z)
		return g2:scale(z, 2)
	end
	local p = setmetatable({ be = be, dit = m, te = { be = false },
		vae = { decode = function(_, g2, z) return scale(g2, z) end,
			encode = function(_, g2, px) return scale(g2, px) end, has_encoder = true },
		vae_mod = { scale = scale, unscale = scale } }, pipeline)
	local before = lgml.memory().CPU
	local bytes = p:measure(64, 16)
	local x <close> = be:tensor("f32", { 1, 16, 8, 8 })
	local v <close> = be:tensor("f32", { 1, 16, 8, 8 })
	local di = m:inputs(16, 8, 8, x)
	local g2 <close> = be:graph()
	local w = g2:cpy((m:forward(g2, di)), v)
	local dit = g2:measure(w)
	zimage.close_inputs(di)
	ok(bytes > 2 * dit + 3 * x:nbytes() and lgml.memory().CPU.compute == before.compute,
	    ("the pipeline plan holds two DiT passes (%d > 2 x %d) and allocates nothing"):format(bytes, dit))
end
m:close()

-- Base64 for the image API.
ok(base64.encode("foobar") == "Zm9vYmFy" and base64.encode("fo") == "Zm8=" and
    base64.encode("f") == "Zg==", "base64 encode")
local bytes = {}
for i = 0, 255 do
	bytes[#bytes + 1] = string.char(i)
end
bytes = table.concat(bytes):rep(3)
ok(base64.decode(base64.encode(bytes)) == bytes and base64.decode("Zm9v\nYg==") == "foob",
    "base64 round trip")
ok(base64.decode("Zm9") == nil and base64.decode("Zm9*") == nil, "base64 rejects bad text")

print("1.." .. n)
os.exit(failed and 1 or 0)
