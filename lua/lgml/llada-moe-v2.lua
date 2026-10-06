--- LLaDA-MoE-v2: a masked diffusion language model on a MoE backbone.
-- The layers are `lgml.transformer` qwen3moe layers with an ungated
-- shared expert and a scale on the routed weights. Attention is not
-- causal: each position sees every position, so no K or V can be kept
-- between passes. `denoise` runs the block sampler of the reference
-- README over one block of mask tokens after a prefix. A pass over more
-- rows than a chunk runs layer by layer over chunks of rows, so its
-- compute memory does not grow with the rows; see `plan`.
-- @module lgml.llada-moe-v2
local lgml = require("lgml")
local kv = require("lgml.kv")
local transformer = require("lgml.transformer")

local M = {}
M.__index = M

--- The sampler defaults of the reference README.
-- `threshold` false unmasks a scheduled count of tokens per step.
-- `prefix_cache` true is not in the reference; see `denoise`.
M.DEFAULTS = {
	steps = 64, block_length = 64, threshold = 1.0, minimal_topk = 1, temperature = 0,
	prefix_cache = false,
}

--- The default most rows of one pass graph; longer passes run in chunks.
M.CHUNK = 512

--- Token names that end a reply, besides the tokenizer end tokens.
M.END_TOKENS = { "<|role_end|>" }

--- Load a LLaDA-MoE-v2 gguf.
-- The mask token is `tokenizer.ggml.mask_token_id`, else
-- `<arch>.mask_token_id`. The block length default is
-- `diffusion.block_length`, else 64.
-- @tparam lgml.backend be the backend that holds the weights
-- @tparam string path the gguf file
-- @tparam[opt] table opts options: flash_attn, chunk (default `CHUNK`),
-- and placement and pinned as for `lgml.transformer.load`
-- @treturn Model the model
-- @raise when the file does not load, a tensor is missing, or the
-- metadata has no mask token
function M.load(be, path, opts)
	opts = opts or {}
	local tm = transformer.load(be, path, { placement = opts.placement, pinned = opts.pinned })
	local meta, hp = tm.meta, tm.hp
	local arch = meta["general.architecture"]
	hp.mask_id = meta["tokenizer.ggml.mask_token_id"] or meta[arch .. ".mask_token_id"]
	assert(hp.mask_id, path .. ": no mask token id")
	hp.canvas = meta["diffusion.block_length"] or M.DEFAULTS.block_length
	hp.n_ctx = hp.n_ctx or 4096
	return setmetatable({ be = be, tm = tm, hp = hp, meta = meta, flash_attn = opts.flash_attn,
		chunk = opts.chunk or M.CHUNK }, M)
end

--- A loaded model.
-- Fields: `be`, `tm` (the `lgml.transformer` model), `hp` (its config,
-- with `mask_id` and `canvas`), `meta`, `flash_attn` and `chunk`.
-- @type Model

--- Close the weights.
function M:close()
	self:drop_cache()
	self.tm:close()
end

--- Free the K and V rows that the prefix cache of `denoise` keeps.
function M:drop_cache()
	local c = self.pcache
	self.pcache = nil
	for _, part in ipairs(c and { c.k, c.v } or {}) do
		for _, t in pairs(part) do
			t:close()
		end
	end
end

--- Make the graph inputs of a pass over n tokens.
-- Logits come from rows r0 to r0 + C - 1 (0-based).
-- @tparam int n the token count
-- @tparam int r0 the first logits row
-- @tparam int C the logits row count
-- @treturn table inputs for `lgml.transformer.forward`; fill `ids`
-- before a compute and close them with `close_inputs`
function M:inputs(n, r0, C)
	local be = self.be
	local inp = {
		n = n, n_past = 0, n_kv = n, n_attn = n, n_swa_attn = n, lo = 0,
		flash_attn = self.flash_attn, logits = "rows", causal = false, deepstack = 0,
		ids = be:tensor("i32", { n }), pos = be:tensor("i32", { n }),
		logit_rows = be:tensor("i32", { C }),
	}
	inp.pos:fill_range(0)
	inp.logit_rows:fill_range(r0)
	return inp
end

--- Close the leaves of inputs from `inputs`.
-- @static
-- @tparam table inp the inputs
function M.close_inputs(inp)
	for _, t in pairs(inp) do
		if type(t) == "userdata" then
			t:close()
		end
	end
end

--- Build the sampling nodes for block logits.
-- x0 is the argmax of logits plus temp times Gumbel noise from the
-- uniform draws u (plain argmax when temp is 0). conf is the softmax
-- probability of x0 under the logits without noise.
-- @static
-- @tparam lgml.graph g the graph
-- @tparam lgml.tensor logits block logits (C, n_vocab)
-- @tparam number temp the temperature
-- @tparam[opt] lgml.tensor u uniform draws in (0, 1) (C, n_vocab)
-- @treturn lgml.tensor x0 (C), i32
-- @treturn lgml.tensor conf (C, 1, 1)
function M.sample_nodes(g, logits, temp, u)
	local C, V = logits:shape()[1], logits:shape()[2]
	local s = logits
	if temp > 0 then
		s = g:add(logits, g:scale(g:neg(g:log(g:neg(g:log(u)))), temp))
	end
	local x0 = g:argmax(s)
	local p3 = g:reshape(g:soft_max(logits), { C, V, 1 })
	return x0, g:get_rows(p3, g:reshape(x0, { C, 1 }))
end

--- Round to the nearest bfloat16, ties to even.
-- The reference compares the confidence in the bf16 of its logits.
-- @static
-- @tparam number x the value
-- @treturn number the rounded value
function M.bf16(x)
	local b = string.unpack("<I4", string.pack("<f", x))
	b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
	return (string.unpack("<f", string.pack("<I4", b & 0xFFFFFFFF)))
end

--- The scheduled unmask count of each step: get_num_transfer_tokens.
-- @static
-- @tparam int masked the mask count of the block
-- @tparam int steps the step count
-- @treturn {int,...} the count per step
function M.schedule(masked, steps)
	local base, rem, out = masked // steps, masked % steps, {}
	for i = 1, steps do
		out[i] = base + (i <= rem and 1 or 0)
	end
	return out
end

--- Pick the positions to unmask in one step.
-- The masked positions go in descending confidence, ties to the lower
-- position. Without a threshold the first `scheduled` of them go. With
-- one, those with confidence at or above it go, and at least the first
-- minimal_topk.
-- @static
-- @tparam {int,...} masked the masked positions
-- @tparam {number,...} conf the confidence by position
-- @tparam ?number threshold the threshold, or false
-- @tparam int minimal_topk the least count with a threshold
-- @tparam int scheduled the count without a threshold
-- @treturn {int,...} the positions, in pick order
function M.select(masked, conf, threshold, minimal_topk, scheduled)
	local k = threshold and #masked or scheduled
	if k == 0 then
		return {}
	end
	local order = table.move(masked, 1, #masked, 1, {})
	table.sort(order, function(a, b)
		if conf[a] ~= conf[b] then
			return conf[a] > conf[b]
		end
		return a < b
	end)
	local out = {}
	for j = 1, k do
		local i = order[j]
		if not threshold or j <= minimal_topk or conf[i] >= threshold then
			out[#out + 1] = i
		end
	end
	return out
end

--- Check sampler options and fill in the defaults.
-- @static
-- @tparam[opt] table o options: steps, block_length, threshold (a number,
-- or false), minimal_topk and temperature
-- @tparam[opt] table def the defaults (default `DEFAULTS`)
-- @treturn[1] table the options
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.options(o, def)
	o, def = o or {}, def or M.DEFAULTS
	local r = {}
	for k, v in pairs(def) do
		r[k] = v
	end
	for k, v in pairs(o) do
		if def[k] == nil then
			return nil, "unknown sampler option " .. tostring(k)
		end
		r[k] = v
	end
	for _, k in ipairs({ "steps", "block_length", "minimal_topk" }) do
		if math.type(r[k]) ~= "integer" or r[k] < 1 then
			return nil, k .. " must be a positive integer"
		end
	end
	if r.threshold ~= false and type(r.threshold) ~= "number" then
		return nil, "threshold must be a number or false"
	elseif type(r.temperature) ~= "number" or r.temperature < 0 then
		return nil, "temperature must be a number from 0"
	elseif r.threshold and r.steps * r.minimal_topk < r.block_length then
		return nil, "steps * minimal_topk must cover one block"
	elseif type(r.prefix_cache) ~= "boolean" then
		return nil, "prefix_cache must be a boolean"
	end
	return r
end

-- The K and V leaves by layer for a chunked pass of n rows: those of the
-- cache, or two scratch layers that the layers take in turn. Also the
-- cache fields that `lgml.kv` reads, and the scratch leaves.
local function kv_leaves(self, n, cache)
	if cache then
		self.tm:grow(cache, n)
		return cache.k, cache.v, cache, {}
	end
	local be, L1 = self.be, self.tm.layers[1]
	local ty = self.flash_attn and "f16" or "f32"
	local s, K, V = {}, {}, {}
	for i = 1, 4 do
		s[i] = be:tensor(ty, { n, L1.Hkv * L1.D })
	end
	for il = 1, #self.tm.layers do
		K[il], V[il] = s[1 + (il - 1) % 2], s[3 + (il - 1) % 2]
	end
	return K, V, { type_k = ty, type_v = ty, be = be }, s
end

-- A full pass of n rows in chunks of at most U rows, layer by layer, as
-- one graph per layer that runs each chunk in turn. The graph of layer
-- il reads the chunk rows of the residual h, attends to the K and V of
-- all n rows, writes the rows back and writes the K and V of layer
-- il + 1. Graph 1 makes the embeddings and the K and V of layer 1. The
-- graphs and the head graph p.g share one compute buffer.
local function chunked(self, n, U, r0, C, cache)
	local be, hp, tm = self.be, self.hp, self.tm
	local nc = (n + U - 1) // U
	U = (n + nc - 1) // nc
	local K, V, kc, scratch = kv_leaves(self, n, cache)
	local p = {
		n = n, U = U, nc = nc, graphs = {},
		h = be:tensor("f32", { n, hp.n_embd }), ids = be:tensor("i32", { U }),
		rows = be:tensor("i32", { U }), logit_rows = be:tensor("i32", { C }),
	}
	p.leaves = { p.h, p.ids, p.rows, p.logit_rows, table.unpack(scratch) }
	p.logit_rows:fill_range(r0)
	local NL = #tm.layers
	local function keys(g, il, x)
		local L = tm.layers[il]
		local _, k, v = transformer.qkv(g, hp, L, x, p.rows, nil, "kv")
		local w = L.Hkv * L.D
		k = g:reshape(kv.rotate(g, k, kv.rot_k(kc, L.D)), { U, w })
		v = g:reshape(kv.rotate(g, v, kv.rot_v(kc, L.D)), { U, w })
		return g:set_rows(K[il], k, p.rows), g:set_rows(V[il], v, p.rows)
	end
	local g = be:graph()
	local x = g:get_rows(tm.tok_embd, p.ids)
	if hp.embd_scale then
		x = g:scale(x, hp.embd_scale)
	end
	p.graphs[1] = { g = g, out = { g:set_rows(p.h, x, p.rows), keys(g, 1, x) } }
	for il, L in ipairs(tm.layers) do
		g = be:graph(nil, p.graphs[1].g)
		x = g:get_rows(p.h, p.rows)
		local q, _, _, gate = transformer.qkv(g, hp, L, x, p.rows, nil, "q")
		local rk, rv = kv.rot_k(kc, L.D), kv.rot_v(kc, L.D)
		local k, v = kv.views(g, kc, K[il], V[il], n, L.Hkv, L.D)
		local o = transformer.attend(g, kv.rotate(g, q, rk), k, v, nil, hp.attn_scale or 1 / math.sqrt(L.D),
		    self.flash_attn, rv)
		x = transformer.ffn(g, hp, L, transformer.attn_out(g, hp, L, o, gate, x))
		local out = { g:set_rows(p.h, x, p.rows) }
		if il < NL then
			out[2], out[3] = keys(g, il + 1, x)
		end
		p.graphs[il + 1] = { g = g, out = out }
	end
	p.g = be:graph(nil, p.graphs[1].g)
	p.logits = transformer.head(p.g, hp, tm.output_norm, tm.output, p.g:get_rows(p.h, p.logit_rows), tm.lora_out)
	return p
end

-- Run the layer graphs of a chunked pass on its n token ids. The last
-- chunk repeats its last row to fill U rows; the copies write the same
-- values.
local function run_layers(p, ids)
	local n, U, rows, cid = p.n, p.U, {}, {}
	for gi, gr in ipairs(p.graphs) do
		for c = 0, p.nc - 1 do
			for i = 1, U do
				local r = math.min(c * U + i, n)
				rows[i], cid[i] = r - 1, ids[r]
			end
			p.rows:set(rows)
			if gi == 1 then
				p.ids:set(cid)
			end
			gr.g:compute(table.unpack(gr.out))
		end
	end
end

-- Close a pass from `pass` or `chunked`.
local function close_pass(p)
	for _, gr in ipairs(p.graphs or {}) do
		for _, t in ipairs(gr.out) do
			t:close()
		end
		gr.g:close()
	end
	for _, t in ipairs({ p.x0, p.conf, p.logits }) do
		t:close()
	end
	p.g:close()
	if p.inp then
		M.close_inputs(p.inp)
	end
	for _, t in ipairs(p.leaves or {}) do
		t:close()
	end
end

-- One kind of pass over list, with its sampling nodes: chunked when it
-- is a full pass of more than U rows. With a cache, a full pass runs
-- [prefix | block] and keeps every K and V row; else the block runs
-- alone on the cached prefix rows. A pass with share takes the compute
-- buffer of that graph. p.run(ids) computes p.x0 and p.conf.
local function pass(self, list, P, C, U, cache, temp, u, share)
	local be, full = self.be, not cache or #list > C
	local p
	if full and #list > U then
		p = chunked(self, #list, U, P, C, cache)
		function p.run(ids)
			run_layers(p, ids)
			p.g:compute(p.x0, p.conf)
		end
	else
		local inp
		if not cache then
			inp = self:inputs(P + C, P, C)
		else
			cache.n_past = full and 0 or P
			inp = self.tm:inputs(list, cache, { flash_attn = self.flash_attn, causal = false, logits = "rows" })
			inp.logit_rows = be:tensor("i32", { C })
			inp.logit_rows:fill_range(full and P or 0)
		end
		local g = be:graph(nil, share)
		p = { g = g, inp = inp, logits = self.tm:forward(g, inp) }
		function p.run(ids)
			inp.ids:set(ids)
			g:compute(p.x0, p.conf)
		end
	end
	p.x0, p.conf = M.sample_nodes(p.g, p.logits, temp, u)
	return p
end

--- Run the sampler over one block of mask tokens after prefix.
-- Each step runs the whole sequence and unmasks positions by `select`.
-- The step loop ends when no mask is left, or after opts.steps steps.
-- With opts.prefix_cache, only the first step runs the whole sequence;
-- later steps run the block on the prefix K and V of that step, as the
-- prefix cache of Fast-dLLM does. This is faster on long prefixes and
-- not exact. The cache stays with the model; see `drop_cache`.
-- A full pass over more than opts.chunk rows runs in chunks; see `plan`.
-- @tparam {int,...} prefix the prompt and earlier blocks
-- @tparam[opt] table opts options: the sampler options of `options`,
-- seed (for the noise when temperature > 0), chunk (default the `chunk`
-- of the model) and on_step: on_step(step, block) returns false to stop
-- @treturn {int,...} the block; positions still masked hold mask_id
-- @treturn int the steps run
-- @raise on bad sampler options
function M:denoise(prefix, opts)
	opts = opts or {}
	local so = {}
	for k in pairs(M.DEFAULTS) do
		so[k] = opts[k]
	end
	local o = assert(M.options(so))
	local be, hp = self.be, self.hp
	local P, C, V, U = #prefix, o.block_length, hp.n_vocab, opts.chunk or self.chunk
	local ids = table.move(prefix, 1, P, 1, {})
	local block = {}
	for i = 1, C do
		block[i] = hp.mask_id
		ids[P + i] = hp.mask_id
	end
	local cache = false
	if o.prefix_cache then
		self.pcache = self.pcache or self.tm:cache(hp.n_ctx, { initial = P + C })
		cache = self.pcache
	end
	local u
	if o.temperature > 0 then
		u = be:tensor("f32", { C, V })
	end
	local full, part
	local sched = not o.threshold and M.schedule(C, o.steps)
	local seed, steps = math.tointeger(opts.seed) or 0, 0
	local ok, err = pcall(function()
		full = pass(self, ids, P, C, U, cache, o.temperature, u)
		for step = 1, o.steps do
			local masked = {}
			for i = 1, C do
				if block[i] == hp.mask_id then
					masked[#masked + 1] = i
				end
			end
			if #masked == 0 then
				break
			end
			local p = full
			if cache and step > 1 then
				part = part or pass(self, block, P, C, U, cache, o.temperature, u, full.g)
				p = part
			end
			if u then
				u:fill("uniform", 2 ^ -40, 1, (seed * 7919 + steps) & 0x7fffffffffffffff)
			end
			p.run(p == part and block or ids)
			local x0, cv = p.x0:get(), p.conf:get()
			steps = steps + 1
			local conf = {}
			for _, i in ipairs(masked) do
				conf[i] = M.bf16(cv[i])
			end
			for _, i in ipairs(M.select(masked, conf, o.threshold, o.minimal_topk, sched and sched[step])) do
				block[i] = math.tointeger(x0[i])
				ids[P + i] = block[i]
			end
			if opts.on_step and opts.on_step(steps, block) == false then
				break
			end
		end
	end)
	for _, p in ipairs({ full, part }) do
		close_pass(p)
	end
	if u then
		u:close()
	end
	if not ok then
		error(err, 0)
	end
	return block, steps
end

--- Run one pass and give the logits of some rows.
-- @tparam {int,...} ids the token ids
-- @tparam[opt] int r0 the first row, 0-based (default 0)
-- @tparam[opt] int C the row count (default all rows from r0)
-- @tparam[opt] int chunk the most rows of one graph (default the
-- `chunk` of the model)
-- @treturn {number,...} the logits, C rows of n_vocab values
function M:logits(ids, r0, C, chunk)
	r0 = r0 or 0
	C = C or #ids - r0
	if #ids > (chunk or self.chunk) then
		local p = chunked(self, #ids, chunk or self.chunk, r0, C)
		local ok, v = pcall(function()
			run_layers(p, ids)
			p.g:compute(p.logits)
			return p.logits:get()
		end)
		close_pass(p)
		if not ok then
			error(v, 0)
		end
		return v
	end
	local inp = self:inputs(#ids, r0, C)
	inp.ids:set(ids)
	local g = self.be:graph()
	local out = self.tm:forward(g, inp)
	g:compute(out)
	local v = out:get()
	g:close()
	M.close_inputs(inp)
	return v
end

-- The device bytes of the leaves of p.
local function leaf_bytes(p)
	local b = 0
	for _, t in ipairs(p.leaves or {}) do
		b = b + t:nbytes()
	end
	for _, t in pairs(p.inp or {}) do
		if type(t) == "userdata" then
			b = b + t:nbytes()
		end
	end
	return b
end

--- Plan the compute memory of the largest passes of `denoise`.
-- It builds them and runs nothing: a single graph of `chunk` rows, and
-- with n_ctx above that, a chunked pass of n_ctx rows. A chunked pass
-- keeps the residual of all rows (n_ctx x n_embd f32) and, without the
-- prefix cache, the K and V rows of two layers. With prefix_cache it
-- plans on a cache of n_ctx rows and the block pass too.
-- @tparam int n_ctx the most rows of a pass
-- @tparam[opt] table opts block_length (default `hp.canvas`) and
-- prefix_cache
-- @treturn int the device bytes, with the input leaves and noise draws
-- @treturn int the device bytes of a prefix cache of n_ctx rows, or 0
function M:plan(n_ctx, opts)
	opts = opts or {}
	local be, C, U = self.be, opts.block_length or self.hp.canvas, self.chunk
	local ids, cache, u, single, ch, part = {}, nil, nil, nil, nil, nil
	local function measure(g, ...)
		return (g:measure(...))
	end
	local ok, r, cb = pcall(function()
		if opts.prefix_cache then
			cache = self.tm:cache(n_ctx)
		end
		local n = math.min(n_ctx, math.max(U, C + 1))
		for i = 1, math.max(n, n_ctx) do
			ids[i] = 0
		end
		u = be:tensor("f32", { C, self.hp.n_vocab })
		single = pass(self, table.move(ids, 1, n, 1, {}), n - C, C, n, cache, 1, u)
		local best = measure(single.g, single.x0, single.conf) + leaf_bytes(single)
		if n_ctx > U then
			ch = pass(self, ids, n_ctx - C, C, U, cache, 1, u)
			local d = measure(ch.g, ch.x0, ch.conf)
			for _, gr in ipairs(ch.graphs) do
				d = math.max(d, measure(gr.g, table.unpack(gr.out)))
			end
			if cache then
				part = pass(self, table.move(ids, 1, C, 1, {}), n_ctx - C, C, U, cache, 1, u, ch.g)
				d = math.max(d, measure(part.g, part.x0, part.conf) + leaf_bytes(part))
			end
			best = math.max(best, d + leaf_bytes(ch))
		end
		return best + u:nbytes(), cache and transformer.cache_bytes(cache) or 0
	end)
	for _, p in ipairs({ single, ch, part }) do
		close_pass(p)
	end
	for _, t in ipairs({ u }) do
		t:close()
	end
	for _, side in ipairs(cache and { cache.k, cache.v } or {}) do
		for _, t in pairs(side) do
			t:close()
		end
	end
	if not ok then
		error(r, 0)
	end
	return r, cb
end

--- Find the tokens to keep from a block: those before the first token
-- in eog.
-- @static
-- @tparam {int,...} block token ids
-- @tparam table eog set of end-of-generation ids
-- @treturn int the count of tokens to keep
-- @treturn ?int the end token, if any
function M.trim(block, eog)
	for i, id in ipairs(block) do
		if eog[id] then
			return i - 1, id
		end
	end
	return #block, nil
end

return M
