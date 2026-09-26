--- SNAC 24 kHz decoder (hubertsiuzdak/snac_24khz) on lgml.
-- It reads the GGUF from CrispASR (cstr/snac-24khz-GGUF). Signals are
-- (C, T) with time innermost.
-- @module lgml.snac
local lgml = require("lgml")

local M = {}
M.__index = M

local CTX = 16		-- context frames on each side of a chunk
local CHUNK = 192	-- output frames per chunk

local function hparams(meta)
	local function k(s, def)
		local v = meta["snac." .. s]
		return v == nil and def or v
	end
	return {
		rate = k("sample_rate", 24000),
		hop = k("hop_length", 512),
		n_codebooks = k("n_codebooks", 3),
		codebook_size = k("codebook_size", 4096),
		vq_strides = k("vq_strides", { 4, 2, 1 }),
		strides = k("decoder_strides", { 8, 8, 4, 2 }),
		dilations = k("residual_dilations", { 1, 3, 9 }),
	}
end

--- Load the SNAC decoder from a gguf.
-- @tparam lgml.backend be the backend that holds the weights
-- @tparam string path the gguf file
-- @treturn Snac the decoder
-- @raise when the file does not load or a tensor is missing
function M.load(be, path)
	local info = assert(lgml.gguf(path))
	local w = assert(be:load(path))
	local function get(name)
		return assert(w:get(name), name)
	end
	local m = setmetatable({ be = be, w = w, hp = hparams(info.meta) }, M)
	m.q = {}
	for k = 0, m.hp.n_codebooks - 1 do
		local p = "snac.q." .. k .. "."
		m.q[k + 1] = {
			codebook = get(p .. "codebook"),
			w = get(p .. "out_proj.weight"),
			b = get(p .. "out_proj.bias"),
		}
	end
	m.in0_w, m.in0_b = get("snac.dec.in0.weight"), get("snac.dec.in0.bias")
	m.in1_w, m.in1_b = get("snac.dec.in1.weight"), get("snac.dec.in1.bias")
	m.blocks = {}
	for b = 0, #m.hp.strides - 1 do
		local p = "snac.dec." .. b .. "."
		local blk = { alpha = get(p .. "alpha"), up_w = get(p .. "up.weight"),
		    up_b = get(p .. "up.bias"), res = {} }
		for r = 0, #m.hp.dilations - 1 do
			local q = p .. "res." .. r .. "."
			blk.res[r + 1] = {
				alpha0 = get(q .. "alpha0"), alpha1 = get(q .. "alpha1"),
				w0 = get(q .. "conv0.weight"), b0 = get(q .. "conv0.bias"),
				w1 = get(q .. "conv1.weight"), b1 = get(q .. "conv1.bias"),
			}
		end
		m.blocks[b + 1] = blk
	end
	m.out_alpha = get("snac.dec.out.alpha")
	m.out_w, m.out_b = get("snac.dec.out.weight"), get("snac.dec.out.bias")
	return m
end

--- A loaded SNAC decoder.
-- Fields: `be`, `w` (weights) and `hp` (hyperparameters: `rate`,
-- `hop`, `n_codebooks`, `codebook_size`, `vq_strides`, `strides`,
-- `dilations`).
-- @type Snac

--- Close the weights.
function M:close()
	self.w:close()
end

-- y + bias, with one bias value per channel of a (C, T) signal.
local function bias(g, y, b)
	local c = y:shape()[1]
	return g:add(y, g:reshape(b, { c, 1 }))
end

local function residual(g, x, u, dil)
	local y = g:orpheus_snake(x, u.alpha0)
	y = bias(g, g:conv1d_dw_orpheus(u.w0, y, 3 * dil, dil), u.b0)
	y = g:orpheus_snake(y, u.alpha1)
	y = bias(g, g:conv1d(u.w1, y), u.b1)
	return g:add(x, y)
end

local function block(g, x, blk, stride, dilations)
	x = g:orpheus_snake(x, blk.alpha)
	x = g:conv_transpose1d_orpheus(blk.up_w, x, stride, (stride + 1) // 2)
	x = bias(g, x, blk.up_b)
	for i, d in ipairs(dilations) do
		x = residual(g, x, blk.res[i], d)
	end
	return x
end

--- Build the latent from codes.
-- @tparam lgml.graph g the graph
-- @tparam {lgml.tensor,...} ids ids[k] is an i32 leaf (T): the codes of
-- codebook k, each repeated vq_strides[k] times
-- @treturn lgml.tensor the latent (768, T)
function M:latent(g, ids)
	local z
	for k, q in ipairs(self.q) do
		local e = g:get_rows(q.codebook, ids[k])
		local s = q.w:shape()
		local y = g:mul_mat(g:reshape(q.w, { s[1], s[2] }), e)
		y = g:add(y, q.b)
		z = z and g:add(z, y) or y
	end
	return g:cont(g:transpose(z))
end

--- Build the decoder graph.
-- @tparam lgml.graph g the graph
-- @tparam {lgml.tensor,...} ids code leaves as for `latent`
-- @treturn lgml.tensor samples (T * hop)
-- @treturn table named stages: `quant_out`, `dec_pre`, `dec_blk0` and on
function M:forward(g, ids)
	local taps = {}
	local x = self:latent(g, ids)
	taps.quant_out = x
	x = bias(g, g:conv1d_dw_orpheus(self.in0_w, x, 3, 1), self.in0_b)
	x = bias(g, g:conv1d(self.in1_w, x), self.in1_b)
	taps.dec_pre = x
	for i, blk in ipairs(self.blocks) do
		x = block(g, x, blk, self.hp.strides[i], self.hp.dilations)
		taps["dec_blk" .. (i - 1)] = x
	end
	x = g:orpheus_snake(x, self.out_alpha)
	local s = self.out_w:shape()
	local w = g:reshape(self.out_w, { 1, s[#s - 1], s[#s] })
	x = g:tanh(bias(g, g:conv1d(w, x, 1, 3), self.out_b))
	local n = x:shape()[2]
	return g:reshape(x, { n }), taps
end

--- Expand codes to one code per latent frame.
-- Codebook k repeats each code vq_strides[k] times.
-- @tparam {{int,...},...} codes codes[k] are 0-based codebook entries
-- @treturn ?table the expanded codes per codebook, or nil
-- @return the frame count, or an error message
function M:frames(codes)
	local st = self.hp.vq_strides
	local n = #codes[1] * st[1]
	local out = {}
	for k = 1, self.hp.n_codebooks do
		if #codes[k] * st[k] ~= n then
			return nil, "codebook " .. k .. " has the wrong length"
		end
		local f = {}
		for i, c in ipairs(codes[k]) do
			if math.type(c) ~= "integer" or c < 0 or c >= self.hp.codebook_size then
				return nil, "code out of range"
			end
			for j = 1, st[k] do
				f[(i - 1) * st[k] + j] = c
			end
		end
		out[k] = f
	end
	return out, n
end

--- Decode frames [a, b) with context into pcm.
-- @tparam table frames expanded codes from `frames`
-- @tparam int n the frame count
-- @tparam int a first frame, 0-based
-- @tparam int b end frame, exclusive
-- @tparam lgml.tensor pcm f32 leaf of samples (n * hop)
function M:chunk(frames, n, a, b, pcm)
	local lo, hi = math.max(0, a - CTX), math.min(n, b + CTX)
	local ids = {}
	for k, f in ipairs(frames) do
		ids[k] = self.be:tensor("i32", { hi - lo }, table.move(f, lo + 1, hi, 1, {}))
	end
	local g <close> = self.be:graph()
	local x = self:forward(g, ids)
	local hop, f32 = self.hp.hop, lgml.row_size("f32", 1)
	local src = g:view(x, { (b - a) * hop }, nil, (a - lo) * hop * f32)
	local dst = g:view(pcm, { (b - a) * hop }, nil, a * hop * f32)
	g:compute(g:cpy(src, dst))
	for _, t in ipairs(ids) do
		t:close()
	end
end

--- Decode codes to samples.
-- @tparam {{int,...},...} codes { c0, c1, c2 }, 0-based codebook entries
-- @tparam[opt] function between runs after each chunk; false stops
-- @treturn ?lgml.tensor an f32 leaf of samples, or nil
-- @treturn string error: bad codes, "no codes" or "cancelled"
function M:decode(codes, between)
	local frames, n = self:frames(codes)
	if not frames then
		return nil, n
	end
	if n == 0 then
		return nil, "no codes"
	end
	local pcm = self.be:tensor("f32", { n * self.hp.hop })
	for a = 0, n - 1, CHUNK do
		self:chunk(frames, n, a, math.min(n, a + CHUNK), pcm)
		if between and between() == false then
			pcm:close()
			return nil, "cancelled"
		end
	end
	return pcm
end

return M
