--- LoRA adapters for `lgml.transformer` models.
-- An adapter holds a pair of f32 leaves for each weight W it changes:
-- A (rank, in) and B (out, rank). With the adapter applied, the model
-- computes W x + s B (A x), with s = scale * alpha / rank (or s = scale
-- when alpha is 0), as llama.cpp does. Weights are named as in the gguf
-- file, such as "blk.0.attn_q.weight".
--
-- Files are llama.cpp LoRA ggufs: general.type "adapter", adapter.type
-- "lora", adapter.lora.alpha, and tensors "<weight>.lora_a" and
-- "<weight>.lora_b". `load` also reads PEFT adapters
-- (adapter_model.safetensors with adapter_config.json).
--
--     local ad = lora.new(model, { rank = 8, alpha = 16 })
--     model:set_adapters({ ad })
--     ... train ad:params() ...
--     ad:save("adapter.gguf")
-- @module lgml.lora
local lgml = require("lgml")
local json = require("lgml.json")
local writer = require("lgml.gguf_writer")

local M = {}
local A = {}
A.__index = A

--- The default targets of `new`: attention and dense FFN projections.
M.TARGETS = { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down" }

-- PEFT module names to gguf weight names after "blk.N.".
local PEFT = {
	["self_attn.q_proj"] = "attn_q", ["self_attn.k_proj"] = "attn_k",
	["self_attn.v_proj"] = "attn_v", ["self_attn.o_proj"] = "attn_output",
	["mlp.gate_proj"] = "ffn_gate", ["mlp.up_proj"] = "ffn_up", ["mlp.down_proj"] = "ffn_down",
}

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

-- An f32 leaf (rows, cols) with the values of t, which may have another
-- type. A rank 1 A has lost its outer size 1.
local function to_f32(be, t)
	local sh = t:shape()
	local out = be:tensor("f32", #sh == 1 and { 1, sh[1] } or sh)
	local g <close> = be:graph()
	g:compute(g:cpy(t, out))
	return out
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

local function load_gguf(model, path, ad)
	local info = assert(lgml.gguf(path))
	local meta = info.meta
	if meta["general.type"] ~= "adapter" or meta["adapter.type"] ~= "lora" then
		error("lora: " .. path .. ": not a LoRA adapter gguf")
	end
	if meta["general.architecture"] ~= model.hp.arch then
		error(("lora: %s: adapter for %s, model is %s"):format(path,
		    tostring(meta["general.architecture"]), model.hp.arch))
	end
	ad.alpha = meta["adapter.lora.alpha"] or 0
	local w <close> = assert(model.be:load(path))
	local parts = {}
	for _, t in ipairs(info.tensors) do
		local base, ab = t.name:match("^(.*)%.lora_([ab])$")
		if base then
			parts[base] = parts[base] or {}
			parts[base][ab] = t.name
			if not parts[base].seen then
				parts[base].seen = true
				parts[#parts + 1] = base
			end
		elseif not t.name:match("_norm%.weight$") then
			error("lora: " .. path .. ": tensor " .. t.name .. " is not a LoRA tensor")
		end
	end
	for _, base in ipairs(parts) do
		local p = parts[base]
		if not (p.a and p.b) then
			error("lora: " .. path .. ": " .. base .. " lacks lora_a or lora_b")
		end
		if base:match("token_embd") then
			error("lora: " .. path .. ": adapters on token_embd are not supported")
		end
		add(ad, base, to_f32(model.be, w:get(p.a)), to_f32(model.be, w:get(p.b)))
	end
end

-- Reorder the rows of a Q or K projection from Hugging Face order to
-- gguf order for llama, as convert_hf_to_gguf.py does. rows is (out, r).
local function permute_rows(be, t, heads)
	local sh = t:shape()
	local v, out = t:get(), {}
	local d = sh[1] // heads
	local r = sh[2]
	for h = 0, heads - 1 do
		for j = 0, d // 2 - 1 do
			for x = 0, 1 do
				local dst, src = h * d + j * 2 + x, h * d + x * (d // 2) + j
				for c = 1, r do
					out[dst * r + c] = v[src * r + c]
				end
			end
		end
	end
	local nt = be:tensor("f32", sh, out)
	t:close()
	return nt
end

local function load_peft(model, path, ad)
	local dir, file = path, path .. "/adapter_model.safetensors"
	if path:match("%.safetensors$") then
		dir, file = path:match("^(.*)/[^/]*$") or ".", path
	end
	local f = io.open(dir .. "/adapter_config.json")
	if not f then
		error("lora: " .. dir .. ": no adapter_config.json")
	end
	local cfg = json.decode(f:read("a"))
	f:close()
	if type(cfg) ~= "table" then
		error("lora: " .. dir .. ": bad adapter_config.json")
	end
	local r = cfg.r or 8
	ad.alpha = cfg.lora_alpha or r
	if cfg.use_rslora then
		ad.alpha = ad.alpha * math.sqrt(r)
	end
	local w <close> = model.be:load_safetensors(file)
	local hp = model.hp
	local seen = {}
	for _, name in ipairs(w:names()) do
		local n = name:gsub("^base_model%.model%.", "")
		local mod, ab = n:match("^(.*)%.lora_([AB])%.weight$")
		if not mod then
			error("lora: " .. file .. ": tensor " .. name .. " is not a LoRA tensor")
		end
		local il, rest = mod:match("^model%.layers%.(%d+)%.(.*)$")
		local base
		if il and PEFT[rest] then
			base = ("blk.%s.%s.weight"):format(il, PEFT[rest])
		elseif mod == "lm_head" then
			base = "output.weight"
		else
			error("lora: " .. file .. ": module " .. mod .. " is not supported")
		end
		local t = to_f32(model.be, w:get(name))
		if ab == "B" and hp.arch == "llama" and (rest == "self_attn.q_proj" or
		    rest == "self_attn.k_proj") then
			t = permute_rows(model.be, t, rest == "self_attn.q_proj" and hp.n_head or hp.n_head_kv)
		end
		seen[base] = seen[base] or {}
		seen[base][ab == "A" and "a" or "b"] = t
	end
	local bases = {}
	for base in pairs(seen) do
		bases[#bases + 1] = base
	end
	table.sort(bases)
	for _, base in ipairs(bases) do
		local p = seen[base]
		if not (p.a and p.b) then
			error("lora: " .. file .. ": " .. base .. " lacks lora_A or lora_B")
		end
		add(ad, base, p.a, p.b)
	end
end

--- Load an adapter.
-- path is a llama.cpp LoRA gguf, a PEFT adapter directory, or its
-- adapter_model.safetensors. The tensors become f32 leaves on the
-- backend of the model. PEFT alpha comes from lora_alpha, times
-- sqrt(r) with use_rslora, as llama.cpp's converter does.
-- @param model a model from `lgml.transformer.load`
-- @tparam string path the adapter
-- @tparam[opt] table opts options: opts.scale (default 1)
-- @treturn Adapter the adapter
-- @raise when the file is not an adapter for the architecture of the
-- model, or names an unsupported weight
function M.load(model, path, opts)
	opts = opts or {}
	local ad = adapter(model, 0, opts.scale or 1)
	local fh = io.open(path, "rb")
	local magic = fh and fh:read(4)
	if fh then
		fh:close()
	end
	if magic == "GGUF" then
		load_gguf(model, path, ad)
	else
		load_peft(model, path, ad)
	end
	return ad
end

--- Parse "file[:scale]", as the command line tools take adapters.
-- @tparam string spec the file, with an optional ":scale" suffix
-- @treturn string the file
-- @treturn number the scale (default 1)
function M.parse_spec(spec)
	local file, s = spec:match("^(.*):([-+]?[%d.]+[eE]?[-+]?%d*)$")
	if file and tonumber(s) then
		return file, tonumber(s)
	end
	return spec, 1
end

--- A LoRA adapter, from `new` or `load`.
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

--- Save the adapter as a llama.cpp LoRA gguf with f32 tensors.
-- @tparam string path the file
-- @tparam[opt] table meta more metadata: { key = string or number }
-- @treturn[1] boolean true
-- @return[2] nil
-- @treturn[2] string an error message
function A:save(path, meta)
	local kv = {
		{ "general.architecture", "string", self.arch },
		{ "general.type", "string", "adapter" },
		{ "adapter.type", "string", "lora" },
		{ "adapter.lora.alpha", "f32", self.alpha },
	}
	local keys = {}
	for k in pairs(meta or {}) do
		keys[#keys + 1] = k
	end
	table.sort(keys)
	for _, k in ipairs(keys) do
		local v = meta[k]
		kv[#kv + 1] = { k, type(v) == "number" and (math.type(v) == "integer" and "i64" or "f32")
		    or "string", type(v) == "number" and v or tostring(v) }
	end
	local ts = {}
	for _, name in ipairs(self.names) do
		local p = self.pairs[name]
		ts[#ts + 1] = { name .. ".lora_a", "f32", p.a:shape(), p.a:bytes() }
		ts[#ts + 1] = { name .. ".lora_b", "f32", p.b:shape(), p.b:bytes() }
	end
	return writer.write(path, kv, ts)
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
