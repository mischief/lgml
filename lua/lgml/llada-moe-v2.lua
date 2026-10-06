--- LLaDA-MoE-v2: a masked diffusion language model on a MoE backbone.
-- The layers are `lgml.transformer` qwen3moe layers with an ungated
-- shared expert and a scale on the routed weights. Attention is not
-- causal: each position sees every position, so no K or V can be kept
-- between passes. `denoise` runs the block sampler of the reference
-- README over one block of mask tokens after a prefix.
-- @module lgml.llada-moe-v2
local lgml = require("lgml")
local transformer = require("lgml.transformer")

local M = {}
M.__index = M

--- The sampler defaults of the reference README.
-- `threshold` false unmasks a scheduled count of tokens per step.
M.DEFAULTS = { steps = 64, block_length = 64, threshold = 1.0, minimal_topk = 1, temperature = 0 }

--- Token names that end a reply, besides the tokenizer end tokens.
M.END_TOKENS = { "<|role_end|>" }

--- Load a LLaDA-MoE-v2 gguf.
-- The mask token is `tokenizer.ggml.mask_token_id`, else
-- `<arch>.mask_token_id`. The block length default is
-- `diffusion.block_length`, else 64.
-- @tparam lgml.backend be the backend that holds the weights
-- @tparam string path the gguf file
-- @tparam[opt] table opts options: flash_attn, and placement and
-- pinned as for `lgml.transformer.load`
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
	return setmetatable({ be = be, tm = tm, hp = hp, meta = meta, flash_attn = opts.flash_attn }, M)
end

--- A loaded model.
-- Fields: `be`, `tm` (the `lgml.transformer` model), `hp` (its config,
-- with `mask_id` and `canvas`), `meta` and `flash_attn`.
-- @type Model

--- Close the weights.
function M:close()
	self.tm:close()
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
	end
	return r
end

--- Run the sampler over one block of mask tokens after prefix.
-- Each step runs the whole sequence and unmasks positions by `select`.
-- The step loop ends when no mask is left, or after opts.steps steps.
-- @tparam {int,...} prefix the prompt and earlier blocks
-- @tparam[opt] table opts options: the sampler options of `options`,
-- seed (for the noise when temperature > 0) and on_step: on_step(step,
-- block) returns false to stop
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
	local P, C, V = #prefix, o.block_length, hp.n_vocab
	local n = P + C
	local ids = table.move(prefix, 1, P, 1, {})
	local block = {}
	for i = 1, C do
		block[i] = hp.mask_id
		ids[P + i] = hp.mask_id
	end
	local inp = self:inputs(n, P, C)
	local u
	if o.temperature > 0 then
		u = be:tensor("f32", { C, V })
	end
	local g = be:graph()
	local x0n, confn = M.sample_nodes(g, self.tm:forward(g, inp), o.temperature, u)
	local sched = not o.threshold and M.schedule(C, o.steps)
	local seed, steps = math.tointeger(opts.seed) or 0, 0
	local ok, err = pcall(function()
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
			inp.ids:set(ids)
			if u then
				u:fill("uniform", 2 ^ -40, 1, (seed * 7919 + steps) & 0x7fffffffffffffff)
			end
			g:compute(x0n, confn)
			local x0, cv = x0n:get(), confn:get()
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
	g:close()
	M.close_inputs(inp)
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
-- @treturn {number,...} the logits, C rows of n_vocab values
function M:logits(ids, r0, C)
	r0 = r0 or 0
	C = C or #ids - r0
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
