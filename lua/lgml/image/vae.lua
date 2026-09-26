--- FLUX autoencoder on lgml, as used by Z-Image and FLUX.1.
-- Images are (N, 3, H, W) f32 in [-1, 1]. Latents are (N, 16, H / 8,
-- W / 8) f32. `VAE:decode` and `VAE:encode` work in VAE latent space.
-- The diffusion model works in scaled space: `M.scale` maps VAE
-- latents to model latents, and `M.unscale` maps them back. So a
-- sampler calls `M.unscale` before `VAE:decode`, and `M.scale` after
-- `VAE:encode`.
--
-- Compute memory grows with H * W: about 3 KB per pixel to decode and
-- 1.5 KB per pixel to encode, so 3 GB and 1.5 GB at 1024 x 1024. The
-- tiled calls bound it by the tile size.
-- @module lgml.image.vae

local M = {}
local VAE = {}
VAE.__index = VAE

--- Latent scale factor of the FLUX autoencoder (0.3611).
M.SCALE = 0.3611
--- Latent shift factor of the FLUX autoencoder (0.1159).
M.SHIFT = 0.1159
--- Pixels per latent position on each axis (8).
M.FACTOR = 8
--- Default tile size of the tiled calls, in latent positions (128, or 1024 pixels).
M.TILE = 128
--- Default tile overlap, in latent positions (32, or 256 pixels).
M.OVERLAP = 32

local GROUPS = 32
local EPS = 1e-6

--- Map VAE latents to model latents: (z - SHIFT) * SCALE.
-- @tparam lgml.graph g the graph
-- @tparam lgml.tensor z contiguous f32 latents
-- @treturn lgml.tensor node with the shape of z
function M.scale(g, z)
	return g:scale(z, M.SCALE, -M.SHIFT * M.SCALE)
end

--- Map model latents to VAE latents: z / SCALE + SHIFT.
-- @tparam lgml.graph g the graph
-- @tparam lgml.tensor z contiguous f32 latents
-- @treturn lgml.tensor node with the shape of z
function M.unscale(g, z)
	return g:scale(z, 1 / M.SCALE, M.SHIFT)
end

-- Levels, blocks and latent channels from the tensor shapes.
local function config(w)
	local function shape(name)
		local t = w:get(name)
		return t and t:shape()
	end
	local cfg = { levels = 0 }
	while shape("decoder.up." .. cfg.levels .. ".block.0.conv1.weight") do
		cfg.levels = cfg.levels + 1
	end
	assert(cfg.levels > 0, "not a FLUX autoencoder: no decoder.up blocks")
	cfg.blocks = 0
	while shape("decoder.up.0.block." .. cfg.blocks .. ".conv1.weight") do
		cfg.blocks = cfg.blocks + 1
	end
	cfg.z = assert(shape("decoder.conv_in.weight"), "no decoder.conv_in")[2]
	return cfg
end

--- Load the autoencoder from a safetensors file.
-- The file holds decoder.* and, for encoding, encoder.* tensors, as in
-- the ae.safetensors of FLUX.1 and Z-Image. The layout (levels, blocks
-- and channels) comes from the tensor shapes.
-- @tparam lgml.backend be the backend that holds the weights
-- @tparam string path the safetensors file
-- @tparam[opt] table opts `direct` (true): run convolutions as the direct
-- op; false uses im2col, which needs much more memory. `dtype` ("f32"):
-- "f16" stores the convolution kernels as f16. `flash` (true): run the
-- attention as flash attention with f16 keys and values; false uses a
-- softmax whose memory grows with the square of H * W
-- @treturn VAE the autoencoder
-- @raise when the file does not load or a tensor is missing
function M.load(be, path, opts)
	opts = opts or {}
	local w = be:load_safetensors(path, { f32_1d = true })
	local m = setmetatable({ be = be, w = w, c = {}, k = {}, direct = opts.direct ~= false,
	    flash = opts.flash ~= false, dtype = opts.dtype or "f32" }, VAE)
	assert(m.dtype == "f32" or m.dtype == "f16", "dtype must be f32 or f16")
	m.has_encoder = w:get("encoder.conv_in.weight") ~= nil
	m.cfg = config(w)
	return m
end

--- Close the weights.
function VAE:close()
	for _, set in ipairs({ self.c, self.k }) do
		for _, t in pairs(set) do
			t:close()
		end
	end
	self.c, self.k = {}, {}
	self.w:close()
end

local function get(self, name)
	return assert(self.w:get(name), "missing tensor " .. name)
end

-- A 1-D parameter as a (C, 1, 1) leaf, which broadcasts over (N, C, H, W).
local function chan(self, name)
	local t = self.c[name]
	if not t then
		local v = get(self, name)
		t = self.be:tensor("f32", { v:shape()[1], 1, 1 }, v:bytes())
		self.c[name] = t
	end
	return t
end

-- A convolution kernel in the kernel type.
local function kernel(self, name)
	local w = get(self, name)
	if w:type() == self.dtype then
		return w
	end
	local t = self.k[name]
	if not t then
		local g <close> = self.be:graph()
		local c = g:cast(w, self.dtype)
		g:compute(c)
		t = self.be:tensor(self.dtype, w:shape(), c:bytes())
		self.k[name] = t
	end
	return t
end

local function conv(self, g, p, x, stride, pad)
	local w = kernel(self, p .. ".weight")
	local k = w:shape()[3]
	local y = g:conv2d(w, x, stride or 1, pad or (k - 1) // 2, 1, self.direct)
	return g:add(y, chan(self, p .. ".bias"))
end

local function norm(self, g, p, x)
	local y = g:group_norm(x, GROUPS, EPS)
	return g:add(g:mul(y, chan(self, p .. ".weight")), chan(self, p .. ".bias"))
end

local function resblock(self, g, p, x)
	local h = g:silu(norm(self, g, p .. ".norm1", x))
	h = conv(self, g, p .. ".conv1", h)
	h = g:silu(norm(self, g, p .. ".norm2", h))
	h = conv(self, g, p .. ".conv2", h)
	if self.w:get(p .. ".nin_shortcut.weight") then
		x = conv(self, g, p .. ".nin_shortcut", x)
	end
	return g:add(x, h)
end

-- Single-head self-attention over the H * W positions.
local function attn(self, g, p, x)
	local n, c, hh, ww = table.unpack(x:shape())
	local hw = hh * ww
	local h = norm(self, g, p .. ".norm", x)
	-- ht holds a row of C channels for each position.
	local ht = g:cont(g:permute(g:reshape(h, { n, c, hw }), { 1, 3, 2 }))
	local function lin(name)
		local wt = g:reshape(get(self, p .. "." .. name .. ".weight"), { c, c })
		return g:add(g:mul_mat(wt, ht), get(self, p .. "." .. name .. ".bias"))
	end
	local q, k = lin("q"), lin("k")
	local o
	if self.flash then
		-- The batch takes the place of the heads. The result is (HW, N, C).
		-- Decoder scores overflow f16, so the op accumulates in f32.
		local v = g:cast(lin("v"), "f16")
		o = g:flash_attn(q, g:cast(k, "f16"), v, nil, 1 / math.sqrt(c), 0, 0, true)
		o = g:reshape(g:cont(g:permute(o, { 2, 3, 1 })), { n, c, hh, ww })
	else
		local v = g:reshape(conv(self, g, p .. ".v", h), { n, c, hw })
		local s = g:soft_max(g:mul_mat(k, q), nil, 1 / math.sqrt(c))
		-- (N, C, HW): each output channel row weights v by the scores.
		o = g:reshape(g:mul_mat(s, v), { n, c, hh, ww })
	end
	return g:add(x, conv(self, g, p .. ".proj_out", o))
end

local function mid(self, g, p, h)
	h = resblock(self, g, p .. ".mid.block_1", h)
	h = attn(self, g, p .. ".mid.attn_1", h)
	return resblock(self, g, p .. ".mid.block_2", h)
end

local function check_shape(t, c, what)
	local s = t:shape()
	assert(#s == 4 and s[2] == c, ("%s must be (N, %d, H, W)"):format(what, c))
	return s
end

--- Add the decoder to a graph.
-- The whole image is one graph. Use `VAE:decode_tiled` to bound memory
-- for large images.
-- @tparam lgml.graph g the graph
-- @tparam lgml.tensor z f32 VAE latents (N, 16, h, w), not scaled; see `M.unscale`
-- @treturn lgml.tensor node (N, 3, 8 h, 8 w), clamped to [-1, 1]
function VAE:decode(g, z)
	check_shape(z, self.cfg.z, "latents")
	local cfg = self.cfg
	local h = conv(self, g, "decoder.conv_in", z)
	h = mid(self, g, "decoder", h)
	for i = cfg.levels - 1, 0, -1 do
		for j = 0, cfg.blocks - 1 do
			h = resblock(self, g, ("decoder.up.%d.block.%d"):format(i, j), h)
		end
		if i > 0 then
			h = g:upscale(h, 2)
			h = conv(self, g, ("decoder.up.%d.upsample.conv"):format(i), h)
		end
	end
	h = g:silu(norm(self, g, "decoder.norm_out", h))
	h = conv(self, g, "decoder.conv_out", h)
	return g:clamp(h, -1, 1)
end

--- Add the encoder to a graph.
-- The whole image is one graph. Use `VAE:encode_tiled` to bound memory
-- for large images.
-- @tparam lgml.graph g the graph
-- @tparam lgml.tensor x f32 image (N, 3, H, W) in [-1, 1]; H and W are
-- multiples of 8
-- @treturn lgml.tensor node (N, 16, H / 8, W / 8): the mean of the latent
-- distribution, in VAE space; see `M.scale`
-- @raise when the file has no encoder, or on a bad shape
function VAE:encode(g, x)
	assert(self.has_encoder, "this autoencoder has no encoder")
	local s = check_shape(x, 3, "image")
	local f = 1 << (self.cfg.levels - 1)
	assert(s[3] % f == 0 and s[4] % f == 0, ("image sides must be multiples of %d"):format(f))
	local cfg = self.cfg
	local h = conv(self, g, "encoder.conv_in", x)
	for i = 0, cfg.levels - 1 do
		for j = 0, cfg.blocks - 2 do
			h = resblock(self, g, ("encoder.down.%d.block.%d"):format(i, j), h)
		end
		if i < cfg.levels - 1 then
			h = g:pad(h, { { 0, 0 }, { 0, 0 }, { 0, 1 }, { 0, 1 } })
			h = conv(self, g, ("encoder.down.%d.downsample.conv"):format(i), h, 2, 0)
		end
	end
	h = mid(self, g, "encoder", h)
	h = g:silu(norm(self, g, "encoder.norm_out", h))
	h = conv(self, g, "encoder.conv_out", h)
	-- The first half of the channels is the mean, the second the log variance.
	local n, c, hh, ww = table.unpack(h:shape())
	local mean = g:view(h, { n, cfg.z, hh, ww }, { c * hh * ww * 4, hh * ww * 4, ww * 4 })
	return g:cont(mean)
end

-- Tile starts on one axis of size n: one tile, or tiles of size t that
-- overlap by at least o, spread evenly.
local function starts(n, t, o)
	if n <= t then
		return { 0 }, n
	end
	local k = math.ceil((n - o) / (t - o))
	local s = {}
	for i = 0, k - 1 do
		s[i + 1] = (n - t) * i // (k - 1)
	end
	return s, t
end

-- Blend weights of each tile on one axis, in output units: a ramp over
-- each overlap, divided by the sum over tiles so the weights sum to 1.
local function weights(s, t, f)
	local n = (s[#s] + t) * f
	local w, sum = {}, {}
	for x = 1, n do
		sum[x] = 0
	end
	for i, a in ipairs(s) do
		local lo = i > 1 and (s[i - 1] + t - a) * f or 0
		local hi = i < #s and (a + t - s[i + 1]) * f or 0
		local v = {}
		for x = 0, t * f - 1 do
			local r = 1
			if lo > 0 then
				r = math.min(r, (x + 0.5) / lo)
			end
			if hi > 0 then
				r = math.min(r, (t * f - x - 0.5) / hi)
			end
			v[x + 1] = r
			sum[a * f + x + 1] = sum[a * f + x + 1] + r
		end
		w[i] = v
	end
	for i, a in ipairs(s) do
		for x = 1, t * f do
			w[i][x] = w[i][x] / sum[a * f + x]
		end
	end
	return w
end

-- Run fn on tiles of x and blend the results into one leaf. Tile sizes
-- are in latent units; fin and fout are input and output units per
-- latent position.
local function tiled(self, x, fin, fout, cout, opts, fn)
	opts = opts or {}
	local t, o = opts.tile or M.TILE, opts.overlap or M.OVERLAP
	assert(t >= 1 and o >= 0 and o < t, "need tile >= 1 and 0 <= overlap < tile")
	local be = self.be
	local shape = x:shape()
	local n, cin, hh, ww = shape[1], shape[2], shape[3] // fin, shape[4] // fin
	local src = be:tensor("f32", shape, x:bytes())
	local out = be:tensor("f32", { n, cout, hh * fout, ww * fout })
	out:zero()
	local ys, th = starts(hh, t, o)
	local xs, tw = starts(ww, t, o)
	local wy, wx = weights(ys, th, fout), weights(xs, tw, fout)
	local peak = 0
	for i, y0 in ipairs(ys) do
		local cy <close> = be:tensor("f32", { th * fout, 1 }, wy[i])
		for j, x0 in ipairs(xs) do
			local cx <close> = be:tensor("f32", { tw * fout }, wx[j])
			local g <close> = be:graph()
			local nb = { cin * shape[3] * shape[4] * 4, shape[3] * shape[4] * 4, shape[4] * 4 }
			local part = g:cont(g:view(src, { n, cin, th * fin, tw * fin }, nb,
			    (y0 * fin * shape[4] + x0 * fin) * 4))
			local r = g:mul(g:mul(fn(g, part), cx), cy)
			local Y, X = y0 * fout, x0 * fout
			r = g:pad(r, { { 0, 0 }, { 0, 0 }, { Y, (hh - y0 - th) * fout },
			    { X, (ww - x0 - tw) * fout } })
			g:compute(g:cpy(g:add(out, r), out))
			peak = math.max(peak, g:buffer_size())
		end
	end
	src:close()
	return out, peak
end

--- Decode latents in tiles, one graph per tile.
-- Tiles overlap and blend with linear ramps. Each tile runs group norm
-- on its own statistics, so tiled output differs a little from
-- `VAE:decode`. When the latents fit in one tile, the result is the
-- same as `VAE:decode`.
-- @tparam lgml.tensor z computed f32 VAE latents (N, 16, h, w), not scaled
-- @tparam[opt] table opts `tile` (`M.TILE`) and `overlap` (`M.OVERLAP`),
-- in latent positions
-- @treturn lgml.tensor a new leaf (N, 3, 8 h, 8 w) in [-1, 1]
-- @treturn int largest compute buffer of a tile, in bytes
function VAE:decode_tiled(z, opts)
	check_shape(z, self.cfg.z, "latents")
	local f = 1 << (self.cfg.levels - 1)
	return tiled(self, z, 1, f, 3, opts, function(g, part)
		return self:decode(g, part)
	end)
end

--- Encode an image in tiles, one graph per tile.
-- Tiles overlap and blend as in `VAE:decode_tiled`.
-- @tparam lgml.tensor x computed f32 image (N, 3, H, W) in [-1, 1]; H
-- and W are multiples of 8
-- @tparam[opt] table opts `tile` (`M.TILE`) and `overlap` (`M.OVERLAP`),
-- in latent positions
-- @treturn lgml.tensor a new leaf (N, 16, H / 8, W / 8), the latent mean
-- in VAE space
-- @treturn int largest compute buffer of a tile, in bytes
function VAE:encode_tiled(x, opts)
	local s = check_shape(x, 3, "image")
	local f = 1 << (self.cfg.levels - 1)
	assert(s[3] % f == 0 and s[4] % f == 0, ("image sides must be multiples of %d"):format(f))
	return tiled(self, x, f, 1, self.cfg.z, opts, function(g, part)
		return self:encode(g, part)
	end)
end

return M
