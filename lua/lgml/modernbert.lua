--- ModernBERT encoder on lgml, from Hugging Face safetensors.
-- Attention is bidirectional. Layers alternate between global
-- attention and local sliding-window attention, and each kind has its
-- own RoPE base. The MLP is GeGLU with the exact (erf) GELU. The norms
-- are LayerNorm. Linear layers and norms take a bias only when the
-- weights have one. The forward pass gives the last hidden state.
-- @module lgml.modernbert
local lgml = require("lgml")
local json = require("lgml.json")

local M = {}
M.__index = M

local function read_json(path)
	local f = assert(io.open(path, "rb"))
	local s = f:read("a")
	f:close()
	return assert(json.decode(s))
end

--- Reads the hyperparameters from a config.json table.
-- The result has `d` (hidden size), `n_layer`, `n_head`, `head_dim`,
-- `n_ff` (the GeGLU width), `eps`, `window` (the local attention span,
-- `local_attention`), `max_pos`, `n_vocab`, `theta_global`,
-- `theta_local`, and `global`, a list of booleans by layer from 1.
-- A layer is global when `layer_types` says `full_attention`, or else
-- when its index from 0 is a multiple of `global_attn_every_n_layers`.
-- @tparam table cfg the decoded config.json
-- @treturn table the hyperparameters
-- @raise when cfg is not a ModernBERT config
function M.config(cfg)
	assert(cfg.model_type == "modernbert", "not a ModernBERT config")
	local rp = type(cfg.rope_parameters) == "table" and cfg.rope_parameters or {}
	local full = type(rp.full_attention) == "table" and rp.full_attention or {}
	local slide = type(rp.sliding_attention) == "table" and rp.sliding_attention or {}
	local hp = {
		d = cfg.hidden_size, n_layer = cfg.num_hidden_layers, n_head = cfg.num_attention_heads,
		n_ff = cfg.intermediate_size, eps = cfg.norm_eps or 1e-5,
		window = cfg.local_attention or 128, max_pos = cfg.max_position_embeddings or 8192,
		n_vocab = cfg.vocab_size,
		theta_global = full.rope_theta or cfg.global_rope_theta or 160000,
		theta_local = slide.rope_theta or cfg.local_rope_theta or 10000,
		global = {},
	}
	hp.head_dim = hp.d // hp.n_head
	local every = cfg.global_attn_every_n_layers or 1
	for i = 1, hp.n_layer do
		local t = type(cfg.layer_types) == "table" and cfg.layer_types[i]
		if t then
			hp.global[i] = t == "full_attention"
		else
			hp.global[i] = (i - 1) % every == 0
		end
	end
	return hp
end

local function lin(get, p)
	return { w = assert(get(p .. ".weight"), "missing tensor " .. p .. ".weight"), b = get(p .. ".bias") }
end

--- Collects the weights of an encoder.
-- get(name) returns a tensor or nil. Names are the Hugging Face names
-- after prefix, such as `layers.0.attn.Wqkv.weight`. Layer 0 may lack
-- `attn_norm`, as in ModernBERT, where it is the identity.
-- @tparam function get the tensor lookup
-- @tparam table hp the hyperparameters from `config`
-- @tparam[opt] string prefix the name prefix, such as `"model."`
-- @treturn table the weights
-- @raise when a tensor is missing
function M.weights(get, hp, prefix)
	prefix = prefix or ""
	local function g(name)
		return get(prefix .. name)
	end
	local w = {
		tok = assert(g("embeddings.tok_embeddings.weight"), "missing token embeddings"),
		emb_norm = lin(g, "embeddings.norm"), final_norm = lin(g, "final_norm"),
		layers = {},
	}
	for i = 0, hp.n_layer - 1 do
		local p = "layers." .. i .. "."
		w.layers[i + 1] = {
			attn_norm = g(p .. "attn_norm.weight") and lin(g, p .. "attn_norm") or nil,
			qkv = lin(g, p .. "attn.Wqkv"), o = lin(g, p .. "attn.Wo"),
			mlp_norm = lin(g, p .. "mlp_norm"),
			wi = lin(g, p .. "mlp.Wi"), wo = lin(g, p .. "mlp.Wo"),
		}
	end
	return w
end

--- Makes an encoder from weights.
-- @param be the lgml backend of the weights
-- @tparam table hp the hyperparameters from `config`
-- @tparam table w the weights from `weights`
-- @treturn ModernBERT the encoder
function M.new(be, hp, w)
	return setmetatable({ be = be, hp = hp, w = w, masks = {} }, M)
end

--- Loads a Hugging Face ModernBERT directory.
-- The directory holds config.json and model.safetensors. The tensor
-- names may have the `model.` prefix of ModernBertForMaskedLM.
-- @param be the lgml backend
-- @tparam string dir the model directory
-- @treturn ModernBERT the encoder; its `store` field holds the weights
-- @raise when a file or a tensor is missing
function M.load(be, dir)
	local hp = M.config(read_json(dir .. "/config.json"))
	local store = assert(be:load_safetensors(dir .. "/model.safetensors", { f32_1d = true }))
	local function get(name)
		return store:get(name)
	end
	local prefix = get("model.embeddings.tok_embeddings.weight") and "model." or ""
	local m = M.new(be, hp, M.weights(get, hp, prefix))
	m.store = store
	return m
end

--- An encoder, from `new` or `load`.
-- Fields: `be`, `hp`, `w` (weights) and, from `load`, `store`.
-- @type ModernBERT

--- Closes the weights that `load` opened, and the cached masks.
function M:close()
	for _, t in pairs(self.masks) do
		t:close()
	end
	self.masks = {}
	if self.store then
		self.store:close()
		self.store = nil
	end
end

-- The local attention mask for n tokens: 0 where |i - j| <= window / 2,
-- else -inf. Cached by n.
function M:local_mask(n)
	local m = self.masks[n]
	if m then
		return m
	end
	local half = self.hp.window // 2
	local neg, zero = string.pack("<f", -math.huge), string.pack("<f", 0)
	local rows = {}
	for i = 0, n - 1 do
		local lo, hi = math.max(0, i - half), math.min(n - 1, i + half)
		rows[i + 1] = neg:rep(lo) .. zero:rep(hi - lo + 1) .. neg:rep(n - 1 - hi)
	end
	m = self.be:tensor("f32", { n, n }, table.concat(rows))
	self.masks[n] = m
	return m
end

--- Makes the input leaves for a token sequence.
-- @tparam {int,...} ids token ids
-- @treturn table `{ n, ids, pos }`; close the leaves with `free_inputs`
-- @raise when the sequence is empty or longer than `max_pos`
function M:inputs(ids)
	local n = #ids
	assert(n >= 1 and n <= self.hp.max_pos, "sequence length out of range")
	local inp = { n = n, ids = self.be:tensor("i32", { n }, ids), pos = self.be:tensor("i32", { n }) }
	inp.pos:fill_range(0)
	return inp
end

--- Closes the leaves of `inputs`.
-- @tparam table inp the inputs
function M.free_inputs(inp)
	inp.ids:close()
	inp.pos:close()
end

local function linear(g, x, p)
	local y = g:mul_mat(p.w, x, true)
	if p.b then
		y = g:add(y, p.b)
	end
	return y
end

local function layer_norm(g, x, p, eps)
	local y = g:mul(g:norm(x, eps), p.w)
	if p.b then
		y = g:add(y, p.b)
	end
	return y
end

--- Bidirectional attention over one sequence.
-- q, k and v are (n, H, D). The mask is nil or an f32 (n, n) additive
-- mask. The scale is 1 / sqrt(D).
-- @param g the graph
-- @param q queries
-- @param k keys
-- @param v values
-- @param[opt] mask the mask
-- @treturn tensor (n, H * D)
function M.attend(g, q, k, v, mask)
	local shape = q:shape()
	local n, H, D = shape[1], shape[2], shape[3]
	local s = g:mul_mat(g:permute(k, { 2, 1, 3 }), g:permute(q, { 2, 1, 3 }), true)
	s = g:soft_max(s, mask, 1 / math.sqrt(D))
	local vt = g:cont(g:permute(v, { 2, 3, 1 }))
	local o = g:mul_mat(vt, s, true)
	return g:reshape(g:cont(g:permute(o, { 2, 1, 3 })), { n, H * D })
end

local function layer(self, g, x, L, i, inp, mask)
	local hp = self.hp
	local n, H, D = inp.n, hp.n_head, hp.head_dim
	local h = L.attn_norm and layer_norm(g, x, L.attn_norm, hp.eps) or x
	local qkv = linear(g, h, L.qkv)
	local row = lgml.row_size("f32", 3 * hp.d)
	local function part(j)
		return g:view(qkv, { n, H, D }, { row, lgml.row_size("f32", D) }, j * lgml.row_size("f32", hp.d))
	end
	local base = hp.global[i] and hp.theta_global or hp.theta_local
	local q = g:rope(g:cont(part(0)), inp.pos, { mode = "neox", base = base })
	local k = g:rope(g:cont(part(1)), inp.pos, { mode = "neox", base = base })
	local o = M.attend(g, q, k, part(2), not hp.global[i] and mask or nil)
	x = g:add(x, linear(g, o, L.o))
	h = linear(g, layer_norm(g, x, L.mlp_norm, hp.eps), L.wi)
	local frow = lgml.row_size("f32", 2 * hp.n_ff)
	local a = g:view(h, { n, hp.n_ff }, { frow })
	local gate = g:view(h, { n, hp.n_ff }, { frow }, lgml.row_size("f32", hp.n_ff))
	return g:add(x, linear(g, g:mul(g:gelu_erf(g:cont(a)), gate), L.wo))
end

--- Adds the encoder to a graph.
-- @param g the graph
-- @tparam table inp the inputs from `inputs`
-- @treturn tensor the last hidden state (n, d), after the final norm
function M:forward(g, inp)
	local hp, w = self.hp, self.w
	local mask
	for i = 1, hp.n_layer do
		if not hp.global[i] and inp.n > hp.window // 2 + 1 then
			mask = self:local_mask(inp.n)
			break
		end
	end
	local x = layer_norm(g, g:get_rows(w.tok, inp.ids), w.emb_norm, hp.eps)
	for i, L in ipairs(w.layers) do
		x = layer(self, g, x, L, i, inp, mask)
	end
	return layer_norm(g, x, w.final_norm, hp.eps)
end

--- Encodes one sequence.
-- @tparam {int,...} ids token ids
-- @treturn {number,...} the last hidden state, n * d values in row order
function M:encode(ids)
	local inp = self:inputs(ids)
	local g <close> = self.be:graph()
	local h = self:forward(g, inp)
	g:compute(h)
	local out = h:get()
	M.free_inputs(inp)
	return out
end

return M
