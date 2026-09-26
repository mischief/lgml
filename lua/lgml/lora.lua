--- LoRA adapters for `lgml.transformer` models.
-- An adapter holds a pair of f32 leaves for each weight W it changes:
-- A (rank, in) and B (out, rank). With the adapter applied, the model
-- computes W x + s B (A x), with s = scale * alpha / rank (or s = scale
-- when alpha is 0), as llama.cpp does. Weights are named as in the gguf
-- file, such as "blk.0.attn_q.weight".
--
--     local ad = lora.new(model, { rank = 8, alpha = 16 })
--     model:set_adapters({ ad })
--     ... train ad:params() ...
-- @module lgml.lora

local M = {}
local A = {}
A.__index = A

--- The default targets of `new`: attention and dense FFN projections.
M.TARGETS = { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" }

local next_id = 0

local function adapter(model, alpha, scale)
	next_id = next_id + 1
	return setmetatable({
		id = "lora" .. next_id, arch = model.hp.arch, be = model.be,
		alpha = alpha or 0, scale = scale or 1, names = {}, pairs = {},
	}, A)
end

local function add(ad, name, a, b)
	if not ad.pairs[name] then
		ad.names[#ad.names + 1] = name
	end
	ad.pairs[name] = { a = a, b = b }
end

--- Make a new adapter to train.
-- A gets uniform values in +-1 / sqrt(in), as PEFT does; B is zero, so
-- the new adapter does not change the model. Both are trainable (see
-- `tensor:requires_grad`).
-- @param model a model from `lgml.transformer.load`
-- @tparam[opt] table opts options:
-- opts.rank: the rank (default 8).
-- opts.alpha: alpha (default 2 * rank).
-- opts.targets: weight names after "blk.N." without ".weight", such as
-- "attn_q" (default `TARGETS`); "output" adds the output projection.
-- opts.layers: a list of layer indexes from 1 (default all layers).
-- opts.seed: the seed for A (default 1).
-- @treturn Adapter the adapter
-- @raise when no weight of the model matches the targets
function M.new(model, opts)
	opts = opts or {}
	local rank = opts.rank or 8
	local ad = adapter(model, opts.alpha or 2 * rank, 1)
	local be, seed = model.be, opts.seed or 1
	local function make(name, w)
		local sh = w:shape()
		if #sh ~= 2 then
			return
		end
		local a = be:tensor("f32", { rank, sh[2] })
		local bound = 1 / math.sqrt(sh[2])
		a:fill("uniform", -bound, bound, seed + #ad.names)
		local b = be:tensor("f32", { sh[1], rank })
		b:zero()
		a:requires_grad()
		b:requires_grad()
		add(ad, name, a, b)
	end
	local layers = opts.layers
	if not layers then
		layers = {}
		for i = 1, #model.layers do
			layers[i] = i
		end
	end
	for _, il in ipairs(layers) do
		for _, t in ipairs(opts.targets or M.TARGETS) do
			local name = ("blk.%d.%s.weight"):format(il - 1, t)
			local w = t ~= "output" and model.w:get(name)
			if w then
				make(name, w)
			end
		end
	end
	for _, t in ipairs(opts.targets or {}) do
		if t == "output" and model.output ~= model.tok_embd then
			make("output.weight", model.output)
		end
	end
	if #ad.names == 0 then
		error("lora: no weight of the model matches the targets")
	end
	return ad
end

--- A LoRA adapter, from `new`.
-- Fields: id (unique in this process), arch, alpha, scale (the user
-- scale; change it, then call `model:set_adapters` again), names (the
-- weight names in order) and pairs (name to { a, b }).
-- @type Adapter

--- The factor s on B (A x) for one weight.
-- @tparam string name the weight name
-- @treturn number scale * alpha / rank, or scale when alpha is 0
function A:scale_of(name)
	local rank = self.pairs[name].a:shape()[1]
	return (self.alpha ~= 0 and self.alpha / rank or 1) * self.scale
end

--- The trainable leaves: A and B of each weight, in name order.
-- @treturn {lgml.tensor,...} the leaves
function A:params()
	local list = {}
	for _, name in ipairs(self.names) do
		local p = self.pairs[name]
		list[#list + 1] = p.a
		list[#list + 1] = p.b
	end
	return list
end

--- Count the adapter values.
-- @treturn integer the element count of all A and B
function A:numel()
	local n = 0
	for _, t in ipairs(self:params()) do
		local c = 1
		for _, d in ipairs(t:shape()) do
			c = c * d
		end
		n = n + c
	end
	return n
end

--- Free the tensors of the adapter.
-- Remove it from models first (see `model:set_adapters`).
function A:close()
	for _, t in ipairs(self:params()) do
		t:close()
	end
	self.pairs, self.names = {}, {}
end

return M
