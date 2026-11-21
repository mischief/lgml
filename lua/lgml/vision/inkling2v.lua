--- Inkling 2 vision encoder (mmproj projector type "inkling2v").
-- A ViT with a CLS token, learned position embeddings resized to the
-- patch grid, 2-D rope and layer scales. The CLS output is dropped, each
-- 2 x 2 block of patches merges into one vector, and a GELU MLP projects
-- it into the text model. The graph follows llama.cpp mtmd_graph_inkling2v.
-- @module lgml.vision.inkling2v
local lgml = require("lgml")

local M = {}
M.__index = M

--- The most tokens of one image by default, as llama.cpp.
M.MAX_TOKENS = 4096
--- The image tokens: the template's image block, one placeholder per embedding.
M.image = { begin = "<|content_image|>", token = "<|unused_200054|>", finish = "" }

--- Read the encoder config from mmproj metadata.
-- @tparam table meta the gguf metadata
-- @tparam[opt] table opts options: `max_tokens`, the most tokens of one
-- image (default `MAX_TOKENS`)
-- @treturn table the config
function M.config(meta, opts)
	opts = opts or {}
	local function k(s, def)
		local v = meta["clip.vision." .. s]
		if v == nil then
			v = def
		end
		return v
	end
	local n_embd, n_head = k("embedding_length"), k("attention.head_count")
	local patch, merge = k("patch_size"), k("spatial_merge_size", 2)
	return {
		n_embd = n_embd, n_head = n_head, head_dim = n_embd // n_head, n_layer = k("block_count"),
		eps = k("attention.layer_norm_epsilon", 1e-5), patch = patch, merge = merge,
		n_out = k("projection_dim"), mean = k("image_mean", { 0.5, 0.5, 0.5 }),
		std = k("image_std", { 0.5, 0.5, 0.5 }),
		max_pixels = (opts.max_tokens or M.MAX_TOKENS) * (patch * merge) * (patch * merge),
	}
end

--- Load the encoder weights.
-- @param be the lgml backend
-- @param w the weights, from `backend:load` on the mmproj file
-- @tparam table meta the gguf metadata
-- @tparam[opt] table opts options: `max_tokens`, as for `config`, and
-- `flash_attn` (false for f32 soft_max attention; default true). Large
-- images need flash attention: soft_max holds a square of the patch count.
-- @treturn Encoder the encoder
-- @raise when a weight is missing
function M.load(be, w, meta, opts)
	local hp = M.config(meta, opts)
	assert(hp.merge == 2, "inkling2v: only a spatial merge of 2 is supported")
	local function get(name)
		return assert(w:get(name), name)
	end
	local function norm(p)
		return { w = get(p .. ".weight"), b = get(p .. ".bias") }
	end
	local function lin(p)
		return { w = get(p .. ".weight"), b = w:get(p .. ".bias") }
	end
	local e = setmetatable({ be = be, w = w, hp = hp, layers = {}, flash_attn = (opts or {}).flash_attn ~= false }, M)
	e.patch_embd = get("v.patch_embd.weight")
	e.patch_bias = w:get("v.patch_embd.bias")
	e.pos_embd = get("v.position_embd.weight")
	e.cls = get("v.class_embd")
	e.cls_pos = get("v.class_position_embd.weight")
	e.pre_ln = norm("v.pre_ln")
	e.mm1, e.mm2 = lin("mm.1"), lin("mm.2")
	for i = 1, hp.n_layer do
		local p = "v.blk." .. (i - 1) .. "."
		e.layers[i] = {
			ln1 = norm(p .. "ln1"), ln2 = norm(p .. "ln2"),
			q = lin(p .. "attn_q"), k = lin(p .. "attn_k"), v = lin(p .. "attn_v"), o = lin(p .. "attn_out"),
			up = lin(p .. "ffn_up"), down = lin(p .. "ffn_down"),
			ls1 = w:get(p .. "ls1.weight"), ls2 = w:get(p .. "ls2.weight"),
		}
	end
	return e
end

--- A loaded encoder, from `load`. Its embeddings are causal and take
-- text positions.
-- @type Encoder
M.causal, M.mrope = true, false

--- Decode an image and pad it to the patch grid.
-- The image goes in the top left corner of a black canvas whose sides
-- are multiples of patch * merge. An image above max_pixels shrinks
-- first, with its aspect ratio kept.
-- @tparam string|userdata img file bytes or an `lgml.image`
-- @treturn[1] userdata the RGB `lgml.image`
-- @treturn[2] nil
-- @treturn[2] string an error message
function M:prepare(img)
	if type(img) == "string" then
		local err
		img, err = lgml.image(img)
		if not img then
			return nil, err
		end
	end
	local hp = self.hp
	local a = hp.patch * hp.merge
	local function align(x)
		return (x + a - 1) // a * a
	end
	local rgb <close> = img:rgb()
	local w, h = rgb.width, rgb.height
	if align(w) * align(h) > hp.max_pixels then
		local s = math.sqrt(hp.max_pixels / (w * h))
		local nw = math.max(a, math.floor(w * s / a) * a)
		local nh = math.max(a, math.floor(h * s / a) * a)
		local small <close> = rgb:resize(nw, nh, "bilinear")
		return small:place(nw, nh, 0, 0)
	end
	return rgb:place(align(w), align(h), 0, 0)
end

--- The merged patch grid of a prepared image.
-- @tparam userdata img an image from `prepare`
-- @treturn integer columns
-- @treturn integer rows
function M:grid(img)
	local s = self.hp.patch * self.hp.merge
	return img.width // s, img.height // s
end

--- Count the text tokens of a prepared image.
-- @tparam userdata img an image from `prepare`
-- @treturn integer the token count
function M:tokens(img)
	local nx, ny = self:grid(img)
	return nx * ny
end

local function lin(g, p, x)
	local y = g:mul_mat(p.w, x, true)
	return p.b and g:add(y, p.b) or y
end

local function layer_norm(g, x, p, eps)
	return g:add(g:mul(g:norm(x, eps), p.w), p.b)
end

-- 2-D rope: the first half of each head turns by the column, the second
-- half by the row.
local function rope2d(g, x, pw, ph)
	local n, H, D = x:shape()[1], x:shape()[2], x:shape()[3]
	local st, f4 = x:strides(), lgml.row_size("f32", 1)
	local opt = { mode = "normal", n_dims = D // 2, base = 10000 }
	local a = g:rope(g:cont(g:view(x, { n, H, D // 2 }, { st[1], st[2] })), pw, opt)
	local b = g:rope(g:cont(g:view(x, { n, H, D // 2 }, { st[1], st[2] }, D // 2 * f4)), ph, opt)
	return g:concat(a, b, 3)
end

--- Build the encoder graph for one prepared image.
-- @param g the graph
-- @tparam table inp inputs from `inputs`
-- @return the embeddings (n_tokens, n_out)
-- @treturn table named nodes: "patch_embd", "embeddings", "layer_out-0" and up
function M:forward(g, inp)
	local hp, taps = self.hp, {}
	local E, H, D, eps = hp.n_embd, hp.n_head, hp.head_dim, hp.eps
	local nx, ny = inp.nx, inp.ny
	local P = nx * ny
	local S = hp.patch
	-- The patch conv as one product over rows of (c, ky, kx).
	local rows, rb = nil, nx * S * 4
	for c = 0, 2 do
		local plane = g:view(inp.pixels, { ny, S, nx, S }, { S * rb, rb, S * 4 }, c * P * S * S * 4)
		local t = g:reshape(g:cont(g:permute(plane, { 1, 3, 2, 4 })), { P, S * S })
		rows = rows and g:concat(rows, t, 2) or t
	end
	local x = g:mul_mat(g:reshape(self.patch_embd, { E, 3 * S * S }), rows, true)
	if self.patch_bias then
		x = g:add(x, self.patch_bias)
	end
	taps.patch_embd = x
	local pe = self.pos_embd
	local side = math.tointeger(math.sqrt(pe:shape()[1]))
	if side ~= nx or side ~= ny then
		pe = g:cont(g:permute(g:reshape(pe, { 1, side, side, E }), { 1, 4, 2, 3 }))
		pe = g:interpolate(pe, ny, nx, "bilinear", false)
		pe = g:reshape(g:cont(g:permute(pe, { 1, 3, 4, 2 })), { P, E })
	end
	x = g:add(x, pe)
	local cls = g:reshape(g:add(self.cls, self.cls_pos), { 1, E })
	x = g:concat(cls, x, 1)
	taps.embeddings = x
	local n = P + 1
	x = layer_norm(g, x, self.pre_ln, eps)
	local scale = 1 / math.sqrt(D)
	for il, L in ipairs(self.layers) do
		local h = layer_norm(g, x, L.ln1, eps)
		local q = rope2d(g, g:reshape(lin(g, L.q, h), { n, H, D }), inp.pos_w, inp.pos_h)
		local k = rope2d(g, g:reshape(lin(g, L.k, h), { n, H, D }), inp.pos_w, inp.pos_h)
		local v = g:reshape(lin(g, L.v, h), { n, H, D })
		q = g:permute(q, { 2, 1, 3 })
		k = g:permute(k, { 2, 1, 3 })
		local o
		if self.flash_attn then
			v = g:permute(v, { 2, 1, 3 })
			o = g:flash_attn(q, g:cast(k, "f16"), g:cast(v, "f16"), nil, scale, 0, 0, true)
		else
			v = g:cont(g:permute(v, { 2, 3, 1 }))
			local kq = g:soft_max(g:mul_mat(k, q, true), nil, scale)
			o = g:cont(g:permute(g:mul_mat(v, kq, true), { 2, 1, 3 }))
		end
		local a = lin(g, L.o, g:reshape(o, { n, E }))
		if L.ls1 then
			a = g:mul(a, L.ls1)
		end
		x = g:add(x, a)
		local f = lin(g, L.down, g:gelu_erf(lin(g, L.up, layer_norm(g, x, L.ln2, eps))))
		if L.ls2 then
			f = g:mul(f, L.ls2)
		end
		x = g:add(x, f)
		taps["layer_out-" .. (il - 1)] = x
	end
	-- Drop the CLS row; each 2 x 2 block of patches becomes one row of
	-- (dy, dx, E).
	local st = x:strides()
	x = g:view(x, { P, E }, { st[1] }, st[1])
	x = g:reshape(g:cont(x), { ny // 2, 2, nx // 2, 2 * E })
	x = g:reshape(g:cont(g:permute(x, { 1, 3, 2, 4 })), { P // 4, 4 * E })
	x = lin(g, self.mm2, g:gelu_erf(lin(g, self.mm1, x)))
	return x, taps
end

-- The rope positions of the CLS token and an nx x ny patch grid: CLS
-- at 0, patches from 1.
local function pos_leaves(be, nx, ny)
	local pw, ph = { 0 }, { 0 }
	for y = 0, ny - 1 do
		for x = 0, nx - 1 do
			pw[#pw + 1] = x + 1
			ph[#ph + 1] = y + 1
		end
	end
	return be:tensor("i32", { #pw }, pw), be:tensor("i32", { #ph }, ph)
end

--- Make the graph inputs for a prepared image.
-- @tparam userdata img an image from `prepare`
-- @treturn table fields nx and ny (patches per row and column) and the
-- leaves pixels (1, 3, H, W), pos_w and pos_h
function M:inputs(img)
	local hp, be = self.hp, self.be
	local nx, ny = img.width // hp.patch, img.height // hp.patch
	local pw, ph = pos_leaves(be, nx, ny)
	return { nx = nx, ny = ny, pos_w = pw, pos_h = ph, pixels = img:tensor(be, 1, hp.mean, hp.std) }
end

local function close_leaves(inp)
	for _, t in pairs(inp) do
		if type(t) == "userdata" then
			t:close()
		end
	end
end

--- Encode an image to text embeddings.
-- @tparam string|userdata img file bytes or an `lgml.image`
-- @treturn[1] userdata an `lgml.hostbuf` of n_tokens rows of n_out f32
-- @treturn[1] integer n_tokens
-- @treturn[2] nil
-- @treturn[2] string an error message when the image does not decode
function M:encode(img)
	local prep, err = self:prepare(img)
	if not prep then
		return nil, err
	end
	local _ <close> = prep
	local inp = self:inputs(prep)
	local g <close> = self.be:graph()
	local out = self:forward(g, inp)
	g:compute(out)
	local buf = lgml.hostbuf(out:nbytes())
	out:to_host(buf)
	local n = out:shape()[1]
	out:close()
	close_leaves(inp)
	return buf, n
end

--- The compute buffer bytes of an image of the most tokens. It plans
-- the graph and runs nothing.
-- @treturn integer the bytes on the backend
function M:measure()
	local hp, be = self.hp, self.be
	local most = hp.max_pixels // ((hp.patch * hp.merge) * (hp.patch * hp.merge))
	local rows = math.max(1, math.floor(math.sqrt(most)))
	local cols = math.max(1, most // rows)
	local nx, ny = cols * hp.merge, rows * hp.merge
	local pw, ph = pos_leaves(be, nx, ny)
	local inp = { nx = nx, ny = ny, pos_w = pw, pos_h = ph,
		pixels = be:tensor("f32", { 1, 3, ny * hp.patch, nx * hp.patch }) }
	local g <close> = be:graph()
	local bytes = g:measure((self:forward(g, inp)))
	close_leaves(inp)
	return bytes
end

return M
