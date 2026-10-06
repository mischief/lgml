--- Qwen3-VL and Qwen3.5 vision encoder (mmproj projector type
-- "qwen3vl_merger").
-- A ViT with layer norms, 2-D rope and a GELU FFN. Its learned position
-- table is resized to the patch grid. Each 2 x 2 block of patches merges
-- into one text embedding. Deepstack layers add features for the first
-- text layers. The graph follows llama.cpp clip_graph_qwen3vl.
-- @module lgml.vision.qwen3vl
local lgml = require("lgml")
local preprocess = require("lgml.vision.preprocess")

local M = {}
M.__index = M

--- The fewest tokens of one image by default, as llama.cpp for Qwen-VL.
M.MIN_TOKENS = 8
--- The most tokens of one image by default.
M.MAX_TOKENS = 4096
--- The image tokens of the Qwen chat templates.
M.image = { begin = "<|vision_start|>", token = "<|image_pad|>", finish = "<|vision_end|>" }

--- Read the encoder config from mmproj metadata.
-- @tparam table meta the gguf metadata
-- @tparam[opt] table opts options: `min_tokens` and `max_tokens`, the
-- image token range (default `MIN_TOKENS` and `MAX_TOKENS`)
-- @treturn table fields n_embd, n_head, head_dim, n_layer, eps, patch,
-- merge, n_out (the text embedding size), act ("gelu", "silu" or
-- "gelu_quick"), deepstack (a set of 0-based layers), mean, std,
-- min_pixels and max_pixels
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
	local area = patch * patch * merge * merge
	local ds = {}
	for i, on in ipairs(k("is_deepstack_layers", {})) do
		if on then
			ds[i - 1] = true
		end
	end
	return {
		n_embd = n_embd, n_head = n_head, head_dim = n_embd // n_head,
		n_layer = k("block_count"), eps = k("attention.layer_norm_epsilon", 1e-6),
		patch = patch, merge = merge, n_out = k("projection_dim"), deepstack = ds,
		act = meta["clip.use_gelu"] and "gelu" or meta["clip.use_silu"] and "silu" or "gelu_quick",
		mean = k("image_mean", { 0.5, 0.5, 0.5 }), std = k("image_std", { 0.5, 0.5, 0.5 }),
		min_pixels = (opts.min_tokens or M.MIN_TOKENS) * area,
		max_pixels = (opts.max_tokens or M.MAX_TOKENS) * area,
	}
end

--- Load the encoder weights.
-- @param be the lgml backend
-- @param w the weights, from `backend:load` on the mmproj file
-- @tparam table meta the gguf metadata
-- @tparam[opt] table opts options:
-- opts.flash_attn: false for f32 soft_max attention (default true:
-- flash attention with f16 keys and values, as llama.cpp by default).
-- Large images need it: soft_max holds a square of the patch count.
-- opts.min_tokens, opts.max_tokens: as for `config`.
-- @treturn Encoder the encoder
-- @raise when a weight is missing
function M.load(be, w, meta, opts)
	local hp = M.config(meta, opts)
	assert(hp.merge == 2, "qwen3vl: only a spatial merge of 2 is supported")
	local function get(name)
		return assert(w:get(name), name)
	end
	local function norm(p)
		return { w = get(p .. ".weight"), b = get(p .. ".bias") }
	end
	local function lin(p)
		return { w = get(p .. ".weight"), b = get(p .. ".bias") }
	end
	local e = setmetatable({ be = be, w = w, hp = hp, layers = {},
		flash_attn = (opts or {}).flash_attn ~= false }, M)
	e.patch_embd = { get("v.patch_embd.weight"), w:get("v.patch_embd.weight.1") }
	e.patch_bias = get("v.patch_embd.bias")
	e.pos_embd = get("v.position_embd.weight")
	e.post_ln = norm("v.post_ln")
	e.mm0, e.mm1 = lin("mm.0"), lin("mm.2")
	hp.n_deepstack = 0
	for i = 1, hp.n_layer do
		local p = "v.blk." .. (i - 1) .. "."
		local L = {
			ln1 = norm(p .. "ln1"), ln2 = norm(p .. "ln2"), qkv = lin(p .. "attn_qkv"),
			o = lin(p .. "attn_out"), up = lin(p .. "ffn_up"), down = lin(p .. "ffn_down"),
		}
		if hp.deepstack[i - 1] then
			local d = "v.deepstack." .. (i - 1) .. "."
			L.ds = { norm = norm(d .. "norm"), fc1 = lin(d .. "fc1"), fc2 = lin(d .. "fc2") }
			hp.n_deepstack = hp.n_deepstack + 1
		end
		e.layers[i] = L
	end
	return e
end

--- A loaded encoder, from `load`.
-- Fields: be, w, hp (from `config`, with n_deepstack), layers and
-- flash_attn. Its embeddings are causal and take M-RoPE positions.
-- @type Encoder
M.causal, M.mrope = true, true

--- Decode and resize an image for the encoder.
-- @tparam string|userdata img file bytes or an `lgml.image`
-- @treturn[1] userdata the RGB `lgml.image`, its sides multiples of
-- patch * merge
-- @treturn[2] nil
-- @treturn[2] string an error message
function M:prepare(img)
	local hp = self.hp
	return preprocess.prepare(img, {
		align = hp.patch * hp.merge, min_pixels = hp.min_pixels, max_pixels = hp.max_pixels,
		filter = "bicubic",
	})
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
	return g:add(g:mul_mat(p.w, x, true), p.b)
end

local function layer_norm(g, x, p, eps)
	return g:add(g:mul(g:norm(x, eps), p.w), p.b)
end

local ACT = {
	gelu = function(g, x) return g:gelu(x) end,
	silu = function(g, x) return g:silu(x) end,
	-- x * sigmoid(1.702 x)
	gelu_quick = function(g, x) return g:mul(x, g:sigmoid(g:scale(x, 1.702))) end,
}

local function ffn(g, up, down, x, act)
	return lin(g, down, ACT[act](g, lin(g, up, x)))
end

--- Build the encoder graph for one prepared image.
-- @param g the graph
-- @tparam table inp inputs from `inputs`
-- @return the embeddings (n_tokens, n_out * (1 + n_deepstack)): each row
-- is the embedding, then the deepstack features
-- @treturn table named nodes for comparison with llama.cpp: "inp",
-- "inp_pos_emb", "layer_out-0" and up
function M:forward(g, inp)
	local hp, taps = self.hp, {}
	local E, H, D, eps = hp.n_embd, hp.n_head, hp.head_dim, hp.eps
	local nx, ny = inp.nx, inp.ny
	local P = nx * ny
	-- The patch conv as one product: rows of (c, ky, kx). A still image is
	-- two equal frames, so the two temporal kernels add up.
	local S = hp.patch
	local rows, rb = nil, nx * S * 4
	for c = 0, 2 do
		local plane = g:view(inp.pixels, { ny, S, nx, S }, { S * rb, rb, S * 4 }, c * P * S * S * 4)
		local t = g:reshape(g:cont(g:permute(plane, { 1, 3, 2, 4 })), { P, S * S })
		rows = rows and g:concat(rows, t, 2) or t
	end
	local kern = self.patch_embd[1]
	if self.patch_embd[2] then
		kern = g:add(kern, self.patch_embd[2])
	end
	local x = g:mul_mat(g:reshape(kern, { E, 3 * S * S }), rows, true)
	-- Rows go from raster order to merge order: each 2 x 2 block together.
	x = g:get_rows(g:add(x, self.patch_bias), inp.order)
	taps.inp = x
	local tbl = self.pos_embd
	local side = math.tointeger(math.sqrt(tbl:shape()[1]))
	local pe = tbl
	if side ~= nx or side ~= ny then
		pe = g:cont(g:permute(g:reshape(tbl, { 1, side, side, E }), { 1, 4, 2, 3 }))
		pe = g:interpolate(pe, ny, nx, "bilinear", true)
		pe = g:reshape(g:cont(g:permute(pe, { 1, 3, 4, 2 })), { P, E })
	end
	x = g:add(x, g:get_rows(pe, inp.order))
	taps.inp_pos_emb = x
	local f4 = lgml.row_size("f32", 1)
	local rope = { mode = "vision", n_dims = D // 2, sections = { D // 4, D // 4, D // 4, D // 4 },
		base = 10000, n_ctx_orig = 32768 }
	local scale = 1 / math.sqrt(D)
	local feats
	for il, L in ipairs(self.layers) do
		local h = layer_norm(g, x, L.ln1, eps)
		local qkv = lin(g, L.qkv, h)
		local st = { 3 * E * f4, D * f4 }
		local q = g:rope(g:view(qkv, { P, H, D }, st), inp.pos, rope)
		local k = g:rope(g:view(qkv, { P, H, D }, st, E * f4), inp.pos, rope)
		local v = g:view(qkv, { P, H, D }, st, 2 * E * f4)
		taps["ln1-" .. (il - 1)], taps["Qcur_rope-" .. (il - 1)] = h, q
		q = g:permute(q, { 2, 1, 3 })
		k = g:permute(k, { 2, 1, 3 })
		local o
		if inp.flash_attn then
			v = g:permute(v, { 2, 1, 3 })
			o = g:flash_attn(q, g:cast(k, "f16"), g:cast(v, "f16"), nil, scale, 0, 0, true)
		else
			v = g:cont(g:permute(v, { 2, 3, 1 }))
			local kq = g:soft_max(g:mul_mat(k, q, true), nil, scale)
			o = g:cont(g:permute(g:mul_mat(v, kq, true), { 2, 1, 3 }))
		end
		local a = lin(g, L.o, g:reshape(o, { P, E }))
		x = g:add(x, a)
		local fo = ffn(g, L.up, L.down, layer_norm(g, x, L.ln2, eps), hp.act)
		taps["attn_out-" .. (il - 1)], taps["ffn_out-" .. (il - 1)] = a, fo
		x = g:add(x, fo)
		taps["layer_out-" .. (il - 1)] = x
		if L.ds then
			local f = layer_norm(g, g:reshape(x, { P // 4, 4 * E }), L.ds.norm, eps)
			f = ffn(g, L.ds.fc1, L.ds.fc2, f, "gelu")
			feats = feats and g:concat(feats, f, 2) or f
		end
	end
	x = layer_norm(g, x, self.post_ln, eps)
	x = ffn(g, self.mm0, self.mm1, g:reshape(x, { P // 4, 4 * E }), "gelu")
	if feats then
		x = g:concat(x, feats, 2)
	end
	return x, taps
end

-- The leaves of an nx x ny patch grid: the merge order and the rope
-- positions (row, column, row, column).
local function grid_leaves(be, nx, ny)
	local order, pos = {}, {}
	local P = nx * ny
	for y = 0, ny - 1, 2 do
		for x = 0, nx - 1, 2 do
			for dy = 0, 1 do
				for dx = 0, 1 do
					local i = #order
					order[i + 1] = (y + dy) * nx + x + dx
					pos[i + 1], pos[P + i + 1] = y + dy, x + dx
					pos[2 * P + i + 1], pos[3 * P + i + 1] = y + dy, x + dx
				end
			end
		end
	end
	return be:tensor("i32", { P }, order), be:tensor("i32", { 4 * P }, pos)
end

--- Make the graph inputs for a prepared image.
-- @tparam userdata img an image from `prepare`
-- @treturn table fields nx and ny (patches per row and column),
-- flash_attn, and the leaves pixels (1, 3, H, W), order (the merge order
-- of the patches) and pos (their rope positions)
function M:inputs(img)
	local hp, be = self.hp, self.be
	local nx, ny = img.width // hp.patch, img.height // hp.patch
	local order, pos = grid_leaves(be, nx, ny)
	return {
		nx = nx, ny = ny, flash_attn = self.flash_attn, order = order, pos = pos,
		pixels = img:tensor(be, 1, hp.mean, hp.std),
	}
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
-- @treturn[1] userdata an `lgml.hostbuf` of n_tokens rows of
-- n_out * (1 + n_deepstack) f32
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

--- The compute buffer bytes of an image of the most tokens.
-- The image is a square grid of tokens, or close to one. It plans the
-- graph and runs nothing.
-- @treturn integer the bytes on the backend
function M:measure()
	local hp, be = self.hp, self.be
	local most = hp.max_pixels // (hp.patch * hp.patch * hp.merge * hp.merge)
	local rows = math.max(1, math.floor(math.sqrt(most)))
	local cols = math.max(1, most // rows)
	local nx, ny = cols * hp.merge, rows * hp.merge
	local order, pos = grid_leaves(be, nx, ny)
	local inp = {
		nx = nx, ny = ny, flash_attn = self.flash_attn, order = order, pos = pos,
		pixels = be:tensor("f32", { 1, 3, ny * hp.patch, nx * hp.patch }),
	}
	local g <close> = be:graph()
	local bytes = g:measure((self:forward(g, inp)))
	close_leaves(inp)
	return bytes
end

return M
