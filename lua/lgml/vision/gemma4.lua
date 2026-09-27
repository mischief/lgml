--- Gemma 4 vision encoder (mmproj projector type "gemma4v").
-- A ViT with rms norms, per-head q and k norms, v norms, 2-D neox rope
-- and a gated FFN. A 3 x 3 average pool merges the patches; a projection
-- makes one text embedding per merged patch. The graph follows
-- llama.cpp clip_graph_gemma4v.
-- @module lgml.vision.gemma4
local lgml = require("lgml")
local preprocess = require("lgml.vision.preprocess")

local M = {}
M.__index = M

--- The fewest tokens of one image by default, as llama.cpp for gemma4v.
M.MIN_TOKENS = 40
--- The most tokens of one image by default.
M.MAX_TOKENS = 280

--- Read the encoder config from mmproj metadata.
-- As in llama.cpp, the FFN gate is gelu with `clip.use_gelu`, silu with
-- `clip.use_silu`, and else the quick gelu.
-- @tparam table meta the gguf metadata
-- @tparam[opt] table opts options: `min_tokens` and `max_tokens`, the
-- image token range (default `MIN_TOKENS` and `MAX_TOKENS`)
-- @treturn table fields n_embd, n_head, head_dim, n_layer, eps, patch,
-- merge, rope_base, n_out (the text embedding size), act ("gelu",
-- "silu" or "gelu_quick"), mean, std, min_pixels and max_pixels
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
	local patch, merge = k("patch_size"), k("projector.scale_factor", 3)
	local area = patch * patch * merge * merge
	return {
		n_embd = n_embd, n_head = n_head, head_dim = n_embd // n_head,
		n_layer = k("block_count"), eps = k("attention.layer_norm_epsilon", 1e-6),
		patch = patch, merge = merge, rope_base = 100,
		n_out = k("projection_dim"),
		act = meta["clip.use_gelu"] and "gelu" or meta["clip.use_silu"] and "silu" or "gelu_quick",
		mean = k("image_mean", { 0, 0, 0 }), std = k("image_std", { 1, 1, 1 }),
		min_pixels = (opts.min_tokens or M.MIN_TOKENS) * area,
		max_pixels = (opts.max_tokens or M.MAX_TOKENS) * area,
	}
end

--- Load the encoder weights.
-- @param be the lgml backend
-- @param w the weights, from `backend:load` on the mmproj file
-- @tparam table meta the gguf metadata
-- @tparam[opt] table opts options:
-- opts.flash_attn: true for flash attention with f16 keys and values,
-- as llama.cpp does by default (default false: f32 soft_max attention).
-- opts.min_tokens, opts.max_tokens: as for `config`.
-- @treturn Encoder the encoder
-- @raise when a weight is missing
function M.load(be, w, meta, opts)
	local hp = M.config(meta, opts)
	local function get(name)
		return assert(w:get(name), name)
	end
	local e = setmetatable({ be = be, w = w, hp = hp, layers = {},
		flash_attn = (opts or {}).flash_attn or false }, M)
	e.patch_embd = get("v.patch_embd.weight")
	e.pos_embd = get("v.position_embd.weight")
	e.std_bias = w:get("v.std_bias")
	e.std_scale = w:get("v.std_scale")
	e.proj = get("mm.input_projection.weight")
	for i = 1, hp.n_layer do
		local p = "v.blk." .. (i - 1) .. "."
		e.layers[i] = {
			ln1 = get(p .. "ln1.weight"), ln2 = get(p .. "ln2.weight"),
			q = get(p .. "attn_q.weight"), k = get(p .. "attn_k.weight"),
			v = get(p .. "attn_v.weight"), o = get(p .. "attn_out.weight"),
			q_norm = get(p .. "attn_q_norm.weight"), k_norm = get(p .. "attn_k_norm.weight"),
			attn_post = get(p .. "attn_post_norm.weight"),
			gate = get(p .. "ffn_gate.weight"), up = get(p .. "ffn_up.weight"),
			down = get(p .. "ffn_down.weight"), ffn_post = get(p .. "ffn_post_norm.weight"),
			prefix = p,
		}
	end
	-- Clamp ranges of Gemma4ClippableLinear, where the file has them.
	e.clamp = {}
	for _, name in ipairs(w:names()) do
		local base = name:match("^(.*)%.input_max$")
		if base then
			local function scalar(s, def)
				local t = w:get(base .. s)
				return t and t:get()[1] or def
			end
			e.clamp[base .. ".weight"] = {
				scalar(".input_min", -math.huge), scalar(".input_max", math.huge),
				scalar(".output_min", -math.huge), scalar(".output_max", math.huge),
			}
		end
	end
	return e
end

--- A loaded encoder, from `load`.
-- Fields: be, w, hp (from `config`), layers and flash_attn.
-- @type Encoder

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
	})
end

--- Count the text tokens of a prepared image.
-- @tparam userdata img an image from `prepare`
-- @treturn integer the token count
function M:tokens(img)
	local s = self.hp.patch * self.hp.merge
	return (img.width // s) * (img.height // s)
end

-- A matrix product, with the clamps of that weight when it has them.
local function mm(self, g, wt, name, x)
	local c = self.clamp[name]
	if not c then
		return g:mul_mat(wt, x, true)
	end
	return g:clamp(g:mul_mat(wt, g:clamp(x, c[1], c[2]), true), c[3], c[4])
end

-- Gated FFN activations: act(gate) * up.
local GLU = {
	gelu = function(g, a, b) return g:geglu_split(a, b) end,
	silu = function(g, a, b) return g:swiglu_split(a, b) end,
	gelu_quick = function(g, a, b) return g:geglu_quick_split(a, b) end,
}

-- Neox rope on each half of the head: x positions on the first half,
-- y positions on the second.
local function rope2d(g, t, px, py, base)
	local n, H, D = t:shape()[1], t:shape()[2], t:shape()[3]
	local st = t:strides()
	local half = D // 2
	local opts = { mode = "neox", base = base, n_dims = half }
	local a = g:rope(g:view(t, { n, H, half }, { st[1], st[2] }), px, opts)
	local b = g:rope(g:view(t, { n, H, half }, { st[1], st[2] }, lgml.row_size("f32", half)), py,
	    opts)
	return g:concat(a, b, 3)
end

--- Build the encoder graph for one prepared image.
-- @param g the graph
-- @tparam table inp inputs from `inputs`
-- @return the embeddings (n_tokens, n_out)
-- @treturn table named nodes for comparison with llama.cpp: "inp",
-- "pos_embd", "layer_out-0" and up, and "pooled"
function M:forward(g, inp)
	local hp, taps = self.hp, {}
	local E, H, D, eps = hp.n_embd, hp.n_head, hp.head_dim, hp.eps
	local nx, ny = inp.nx, inp.ny
	local P = nx * ny
	-- The patch conv as one product with f32 sums: rows of (c, ky, kx).
	local S = hp.patch
	local px = g:scale(inp.pixels, 2, -1)
	local rows, rb = nil, nx * S * 4
	for c = 0, 2 do
		local plane = g:view(px, { ny, S, nx, S }, { S * rb, rb, S * 4 }, c * P * S * S * 4)
		local t = g:reshape(g:cont(g:permute(plane, { 1, 3, 2, 4 })), { P, S * S })
		rows = rows and g:concat(rows, t, 2) or t
	end
	local x = g:mul_mat(g:reshape(self.patch_embd, { E, 3 * S * S }), rows, true)
	taps.inp = x
	local tbl = self.pos_embd
	local n_pos = tbl:shape()[2]
	local row = lgml.row_size(tbl:type(), E)
	local tx = g:view(tbl, { n_pos, E }, { row })
	local ty = g:view(tbl, { n_pos, E }, { row }, n_pos * row)
	x = g:add(g:add(x, g:get_rows(tx, inp.px)), g:get_rows(ty, inp.py))
	taps.pos_embd = x
	for il, L in ipairs(self.layers) do
		local p = L.prefix
		local h = g:mul(g:rms_norm(x, eps), L.ln1)
		local q = g:reshape(mm(self, g, L.q, p .. "attn_q.weight", h), { P, H, D })
		local k = g:reshape(mm(self, g, L.k, p .. "attn_k.weight", h), { P, H, D })
		local v = g:reshape(mm(self, g, L.v, p .. "attn_v.weight", h), { P, H, D })
		q = rope2d(g, g:mul(g:rms_norm(q, eps), L.q_norm), inp.px, inp.py, hp.rope_base)
		k = rope2d(g, g:mul(g:rms_norm(k, eps), L.k_norm), inp.px, inp.py, hp.rope_base)
		v = g:rms_norm(v, eps)
		q = g:permute(q, { 2, 1, 3 })
		k = g:permute(k, { 2, 1, 3 })
		local o
		if inp.flash_attn then
			-- f16 keys and values, as llama.cpp clip.
			v = g:permute(v, { 2, 1, 3 })
			o = g:flash_attn(q, g:cast(k, "f16"), g:cast(v, "f16"), nil, 1.0, 0, 0, true)
		else
			v = g:cont(g:permute(v, { 2, 3, 1 }))
			local kq = g:soft_max(g:mul_mat(k, q, true), nil, 1.0)
			o = g:cont(g:permute(g:mul_mat(v, kq, true), { 2, 1, 3 }))
		end
		o = mm(self, g, L.o, p .. "attn_out.weight", g:reshape(o, { P, E }))
		x = g:add(x, g:mul(g:rms_norm(o, eps), L.attn_post))
		h = g:mul(g:rms_norm(x, eps), L.ln2)
		local f = GLU[hp.act](g, mm(self, g, L.gate, p .. "ffn_gate.weight", h),
		    mm(self, g, L.up, p .. "ffn_up.weight", h))
		f = mm(self, g, L.down, p .. "ffn_down.weight", f)
		x = g:add(x, g:mul(g:rms_norm(f, eps), L.ffn_post))
		taps["layer_out-" .. (il - 1)] = x
	end
	local m = hp.merge
	x = g:reshape(g:cont(g:transpose(x)), { 1, E, ny, nx })
	x = g:pool2d(x, m)
	local n = (nx // m) * (ny // m)
	x = g:cont(g:transpose(g:reshape(x, { E, n })))
	x = g:scale(x, math.sqrt(E))
	taps.pooled = x
	if self.std_bias and self.std_scale then
		x = g:mul(g:sub(x, self.std_bias), self.std_scale)
	end
	return mm(self, g, self.proj, "mm.input_projection.weight", g:rms_norm(x, eps)), taps
end

--- Make the graph inputs for a prepared image.
-- @tparam userdata img an image from `prepare`
-- @treturn table fields nx and ny (patches per row and column),
-- flash_attn, and the leaves pixels (1, 3, H, W), px and py (patch
-- columns and rows)
function M:inputs(img)
	local hp, be = self.hp, self.be
	local nx, ny = img.width // hp.patch, img.height // hp.patch
	local px, py = {}, {}
	for i = 0, nx * ny - 1 do
		px[i + 1] = i % nx
		py[i + 1] = i // nx
	end
	return {
		nx = nx, ny = ny, flash_attn = self.flash_attn,
		pixels = img:tensor(be, 1, hp.mean, hp.std),
		px = be:tensor("i32", { nx * ny }, px),
		py = be:tensor("i32", { nx * ny }, py),
	}
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
	-- With its output closed, the graph frees its compute buffer at close.
	out:close()
	for _, t in pairs(inp) do
		if type(t) == "userdata" then
			t:close()
		end
	end
	return buf, n
end

--- The compute buffer bytes of an image of the most tokens.
-- The image is a rows x cols grid of tokens, with rows * cols at most
-- the token limit and rows the square root. It plans the graph and runs
-- nothing.
-- @treturn integer the bytes on the backend
function M:measure()
	local hp, be = self.hp, self.be
	local most = hp.max_pixels // (hp.patch * hp.patch * hp.merge * hp.merge)
	local rows = math.max(1, math.floor(math.sqrt(most)))
	local cols = math.max(1, most // rows)
	local nx, ny = cols * hp.merge, rows * hp.merge
	local px, py = {}, {}
	for i = 0, nx * ny - 1 do
		px[i + 1] = i % nx
		py[i + 1] = i // nx
	end
	local inp = {
		nx = nx, ny = ny, flash_attn = self.flash_attn,
		pixels = be:tensor("f32", { 1, 3, ny * hp.patch, nx * hp.patch }),
		px = be:tensor("i32", { nx * ny }, px),
		py = be:tensor("i32", { nx * ny }, py),
	}
	local g <close> = be:graph()
	local bytes = g:measure((self:forward(g, inp)))
	for _, t in pairs(inp) do
		if type(t) == "userdata" then
			t:close()
		end
	end
	return bytes
end

return M
