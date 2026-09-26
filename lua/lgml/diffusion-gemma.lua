--- DiffusionGemma: block text diffusion on a Gemma-4 MoE backbone.
-- One forward runs over [prompt | canvas]. Prompt rows are causal and
-- never see the canvas. Canvas rows see the prompt and the whole
-- canvas. The input embedding and the per-layer output scale depend on
-- the region. The layers are `lgml.transformer` blocks.
-- @module lgml.diffusion-gemma
local lgml = require("lgml")
local mt19937 = require("lgml.mt19937")
local transformer = require("lgml.transformer")

local M = {}
M.__index = M

-- A layer takes about 70 graph tensors, so a graph of GRAPH_SIZE 16384
-- holds 128 layers. Every current model runs in one graph.
local LAYERS_PER_GRAPH = 128

local function hparams(meta)
	local hp = transformer.config(meta)
	local tokens = meta["tokenizer.ggml.tokens"]
	hp.canvas = meta["diffusion.canvas_length"]
	hp.mask_id = meta["tokenizer.ggml.mask_token_id"]
	hp.n_vocab = tokens and #tokens
	return hp
end

-- Entropy-bound sampler settings: gguf metadata, else the defaults of
-- the llama.cpp reference.
local function eb_defaults(meta)
	local function f(key, def)
		local v = meta["diffusion." .. key]
		return type(v) == "number" and v or def
	end
	return {
		steps = f("eb_max_steps", 48),
		t_min = f("eb_t_min", 0.4),
		t_max = f("eb_t_max", 0.8),
		entropy_bound = f("eb_entropy_bound", 0.1),
		stability = f("eb_stability_threshold", 1),
		confidence = f("eb_confidence_threshold", 0.005),
	}
end

--- Load a DiffusionGemma gguf.
-- The entropy-bound sampler defaults come from the gguf metadata, else
-- from the llama.cpp reference.
-- @tparam lgml.backend be the backend that holds the weights
-- @tparam string path the gguf file
-- @tparam[opt] table opts options:
-- opts.layers_per_graph: layers per compute graph (default 128).
-- opts.flash_attn: use flash attention.
-- opts.sc_type: type of the self-conditioning table, "q8_0" (the
-- default, like the reference) or "f16".
-- @treturn Model the model
-- @raise when the file does not load, a tensor is missing, or the
-- metadata lacks diffusion.canvas_length or a per-layer
-- sliding_window_pattern
function M.load(be, path, opts)
	opts = opts or {}
	local info = assert(lgml.gguf(path))
	local w = assert(be:load(path))
	local hp = hparams(info.meta)
	assert(hp.canvas and hp.canvas > 0, "diffusion-gemma: need diffusion.canvas_length")
	local function get(name)
		return assert(w:get(name), name)
	end
	local m = setmetatable({
		be = be, w = w, hp = hp, meta = info.meta, eb = eb_defaults(info.meta),
		layers_per_graph = opts.layers_per_graph or LAYERS_PER_GRAPH,
		flash_attn = opts.flash_attn, sc_type = opts.sc_type or "q8_0",
	}, M)
	m.tok_embd = get("token_embd.weight")
	hp.n_vocab = m.tok_embd:shape()[1]
	m.output = w:get("output.weight") or m.tok_embd
	m.output_norm = get("output_norm.weight")
	m.rope_freqs = w:get("rope_freqs.weight")
	m.sc = {
		pre_norm = w:get("self_cond_pre_norm.weight"),
		gate = w:get("self_cond_gate.weight"),
		up = w:get("self_cond_up.weight"),
		down = w:get("self_cond_down.weight"),
	}
	m.layers = {}
	for i = 0, hp.n_layer - 1 do
		local p = "blk." .. i .. "."
		local L = transformer.load_layer(w, hp, i + 1, m.rope_freqs)
		for _, k in ipairs({ "wk", "k_norm", "post_ffn_norm_1", "pre_ffn_norm_2",
		    "post_ffn_norm_2", "gate_inp", "router_scale", "down_exps", "out_scale" }) do
			assert(L[k], p .. k)
		end
		L.enc_out_scale = get(p .. "enc_layer_output_scale.weight")
		m.layers[i + 1] = L
	end
	return m
end

--- A loaded DiffusionGemma model.
-- Fields: `be`, `w` (weights), `hp` (hyperparameters, with `canvas`,
-- `mask_id` and `n_vocab`), `meta` (gguf metadata), `eb` (sampler
-- defaults), `layers`, `flash_attn` and `layers_per_graph`.
-- @type Model

--- Close the weights and the self-conditioning table.
function M:close()
	if self.sc_table then
		self.sc_table:close()
	end
	self.w:close()
end

--- Get the self-conditioning embedding table.
-- This is the embedding table transposed to (n_embd, n_vocab), built
-- once. Each row chunk goes through the host, so the ggml reference
-- quantizer rounds it, as in llama.cpp.
-- @treturn lgml.tensor the table, of type `sc_type`
function M:sc_embedding()
	if self.sc_table then
		return self.sc_table
	end
	-- TODO: slow, 7.9 s on the 26B Q4 model. About 3 GB of f32 goes
	-- through Lua strings to a one-thread quantize. Do the transpose and
	-- quantize in C with threads, and build the table at load.
	local be, E, V, ty = self.be, self.hp.n_embd, self.hp.n_vocab, self.sc_type
	local tab = be:tensor(ty, { E, V })
	local rowq = lgml.row_size(ty, V)
	local R = math.max(1, (1 << 26) // V)
	local chunk = 8192
	for e0 = 0, E - 1, R do
		local r = math.min(R, E - e0)
		local buf <close> = be:tensor("f32", { r, V })
		for v0 = 0, V - 1, chunk do
			local n = math.min(chunk, V - v0)
			local g <close> = be:graph()
			local ids <close> = be:tensor("i32", { n })
			ids:fill_range(v0)
			local part = g:view(g:get_rows(self.tok_embd, ids), { n, r }, { E * 4 }, e0 * 4)
			local dst = g:view(buf, { r, n }, { V * 4 }, v0 * 4)
			g:compute(g:cpy(g:transpose(part), dst))
		end
		local q <close> = be:tensor(ty, { r, V }, lgml.quantize(ty, buf:bytes(), V))
		local g <close> = be:graph()
		g:compute(g:cpy(q, g:view(tab, { r, V }, { rowq }, e0 * rowq)))
	end
	self.sc_table = tab
	return tab
end

--- Make a prompt KV cache.
-- The cache has per-layer K and V leaves (cap, Hkv * D). Rows 1 to n
-- hold tokens[1..n]. Close it with `close_cache`.
-- @tparam int cap row capacity
-- @tparam[opt="f16"] string type "f16" or "f32"
-- @treturn table cache: `cap`, `type`, `n`, `tokens`, `k`, `v`
function M:kv_cache(cap, type)
	type = type or "f16"
	local c = { cap = cap, type = type, n = 0, tokens = {}, k = {}, v = {} }
	for il, L in ipairs(self.layers) do
		c.k[il] = self.be:tensor(type, { cap, L.Hkv * L.D })
		c.v[il] = self.be:tensor(type, { cap, L.Hkv * L.D })
	end
	return c
end

--- Close the K and V leaves of a cache from `kv_cache`.
-- @static
-- @tparam table c the cache
function M.close_cache(c)
	for i = 1, #c.k do
		c.k[i]:close()
		c.v[i]:close()
	end
end

--- Make the graph inputs for ids.
-- The mode is "unified" (ids = [prompt | canvas], P prompt tokens),
-- "prefill" (prompt tokens at off, K and V go to cache) or "decode"
-- (canvas tokens after the P cached prompt tokens).
-- @tparam {int,...} ids token ids
-- @tparam table opts options:
-- opts.mode: "unified" (the default), "prefill" or "decode".
-- opts.P: prompt token count, for unified and decode.
-- opts.off: first prompt position, for prefill (default 0).
-- opts.cache: a cache from `kv_cache`, for prefill and decode.
-- opts.sc: self-conditioning `{ prev = logits leaf, temp_inv = n }` or nil.
-- @treturn table inputs; close them with `close_inputs`
-- @raise on an unknown mode
function M:inputs(ids, opts)
	local be, hp, n = self.be, self.hp, #ids
	local mode = opts.mode or "unified"
	local inp = {
		mode = mode, n = n, cache = opts.cache, sc = opts.sc,
		ids = be:tensor("i32", { n }, ids), pos = be:tensor("i32", { n }),
	}
	local mt = self.flash_attn and "f16" or "f32"
	local mo, mw
	if mode == "unified" then
		inp.P, inp.C, inp.n_kv, inp.p0 = opts.P, n - opts.P, n, 0
		mo = { canvas = opts.P }
		mw = { canvas = opts.P, window = hp.n_swa }
	elseif mode == "decode" then
		inp.P, inp.C, inp.n_kv, inp.p0 = opts.P, n, opts.P + n, opts.P
		mo = { n_past = opts.P, canvas = opts.P }
		mw = { n_past = opts.P, canvas = opts.P, window = hp.n_swa }
	elseif mode == "prefill" then
		inp.off, inp.n_kv, inp.p0 = opts.off or 0, (opts.off or 0) + n, opts.off or 0
		mo = { n_past = inp.off }
		mw = { n_past = inp.off, window = hp.n_swa }
	else
		error("diffusion-gemma: unknown mode " .. tostring(mode))
	end
	inp.pos:fill_range(inp.p0)
	inp.mask = be:tensor(mt, { n, inp.n_kv })
	inp.mask:fill_mask(mo)
	inp.mask_swa = be:tensor(mt, { n, inp.n_kv })
	inp.mask_swa:fill_mask(mw)
	return inp
end

--- Close the leaves of inputs from `inputs`.
-- @static
-- @tparam table inp the inputs
function M.close_inputs(inp)
	for _, k in ipairs({ "ids", "pos", "mask", "mask_swa", "hidden" }) do
		if inp[k] then
			inp[k]:close()
		end
	end
end

-- Rows [r0, r0 + n) of a matrix node.
local function rows(g, x, r0, n)
	local s = x:strides()
	return g:view(x, { n, x:shape()[2] }, { s[1] }, r0 * s[1])
end

--- Build the canvas input.
-- The result is rms_norm(embedding [+ self-conditioning]), with no weight.
-- @tparam lgml.graph g the graph
-- @tparam table inp inputs from `inputs`
-- @tparam lgml.tensor x the scaled canvas embeddings (C, n_embd)
-- @treturn lgml.tensor the canvas input node
function M:canvas_embed(g, inp, x)
	local hp, sc = self.hp, inp.sc
	if sc then
		local probs = g:soft_max(sc.prev, nil, sc.temp_inv)
		local soft = g:scale(g:mul_mat(self:sc_embedding(), probs), math.sqrt(hp.n_embd))
		local cur = g:mul(g:rms_norm(soft, hp.eps), self.sc.pre_norm)
		local gg = g:gelu(g:mul_mat(self.sc.gate, cur))
		local sig = g:mul_mat(self.sc.down, g:mul(gg, g:mul_mat(self.sc.up, cur)))
		x = g:add(g:cont(x), sig)
	end
	return g:rms_norm(x, hp.eps)
end

--- Build the region-aware input embedding.
-- @tparam lgml.graph g the graph
-- @tparam table inp inputs from `inputs`
-- @treturn lgml.tensor the embedding node (n, n_embd)
function M:embed(g, inp)
	local hp = self.hp
	local x = g:scale(g:get_rows(self.tok_embd, inp.ids), math.sqrt(hp.n_embd))
	if inp.mode == "prefill" then
		return x
	elseif inp.mode == "decode" then
		return self:canvas_embed(g, inp, x)
	elseif inp.P == 0 then
		return g:rms_norm(x, hp.eps)
	end
	local prompt = g:cont(rows(g, x, 0, inp.P))
	local canvas = self:canvas_embed(g, inp, rows(g, x, inp.P, inp.C))
	return g:concat(prompt, canvas, 1)
end

-- Keys or values for attention, (n_kv, Hkv, D) f32. Prefill writes the
-- new rows to the cache. Prefill and decode read the cached prefix.
local function kv_rows(g, inp, t, cache, fresh, Hkv, D)
	if not cache then
		return fresh
	end
	local n_prev = inp.mode == "decode" and inp.P or inp.off
	local row = lgml.row_size(cache.type, Hkv * D)
	local hd = lgml.row_size(cache.type, D)
	if inp.mode == "prefill" then
		local dst = g:view(t, { inp.n, Hkv, D }, { row, hd }, n_prev * row)
		inp.writes[#inp.writes + 1] = g:cpy(fresh, dst)
	end
	if n_prev == 0 then
		return fresh
	end
	local prev = g:view(t, { n_prev, Hkv, D }, { row, hd })
	if cache.type ~= "f32" then
		prev = g:cast(prev, "f32")
	end
	return g:concat(prev, fresh, 1)
end

local function attention(self, g, inp, il, q, k, v)
	local L = self.layers[il]
	local mask = L.swa and inp.mask_swa or inp.mask
	local c = inp.cache
	k = kv_rows(g, inp, c and c.k[il], c, k, L.Hkv, L.D)
	v = kv_rows(g, inp, c and c.v[il], c, v, L.Hkv, L.D)
	return transformer.attend(g, q, k, v, mask, self.hp.attn_scale, self.flash_attn)
end

--- Build one transformer block.
-- @tparam lgml.graph g the graph
-- @tparam table inp inputs from `inputs`
-- @tparam int il layer index, 1-based
-- @tparam lgml.tensor x the hidden state (n, n_embd)
-- @treturn lgml.tensor the new hidden state
function M:layer(g, inp, il, x)
	local hp, L = self.hp, self.layers[il]
	local q, k, v = transformer.qkv(g, hp, L, x, inp.pos)
	local o = attention(self, g, inp, il, q, k, v)
	local cur = transformer.ffn(g, hp, L, transformer.attn_out(g, hp, L, o, nil, x))

	if inp.mode == "prefill" then
		return g:mul(cur, L.enc_out_scale)
	elseif inp.mode == "decode" or inp.P == 0 then
		return g:mul(cur, L.out_scale)
	end
	local prompt = g:mul(g:cont(rows(g, cur, 0, inp.P)), L.enc_out_scale)
	local canvas = g:mul(g:cont(rows(g, cur, inp.P, inp.C)), L.out_scale)
	return g:concat(prompt, canvas, 1)
end

--- Build the output head.
-- @tparam lgml.graph g the graph
-- @tparam table inp inputs from `inputs`
-- @tparam lgml.tensor x the final hidden state
-- @treturn lgml.tensor the canvas logits (C, n_vocab), soft-capped
function M:head(g, inp, x)
	local hp = self.hp
	if inp.mode == "unified" and inp.P > 0 then
		x = rows(g, x, inp.P, inp.C)
	end
	return (transformer.head(g, hp, self.output_norm, self.output, x))
end

--- Build the whole forward in one graph.
-- Prefill cache writes go to `inp.writes`.
-- @tparam lgml.graph g the graph
-- @tparam table inp inputs from `inputs`
-- @treturn ?lgml.tensor the canvas logits, or nil for prefill
function M:forward(g, inp)
	inp.writes = {}
	local x = self:embed(g, inp)
	for il = 1, #self.layers do
		x = self:layer(g, inp, il, x)
	end
	if inp.mode == "prefill" then
		return nil
	end
	return self:head(g, inp, x)
end

--- Run the forward over one or more graphs.
-- A graph holds at most layers_per_graph layers and passes the hidden
-- state on in a leaf.
-- @tparam table inp inputs from `inputs`
-- @tparam[opt] function tail tail(g, logits) returns the output nodes
-- of the last graph; not used for prefill
-- @treturn lgml.graph the last graph; the caller closes it
-- @return the nodes from tail
function M:eval(inp, tail)
	local be, nl, per = self.be, #self.layers, self.layers_per_graph
	local il = 1
	while true do
		local g = be:graph()
		inp.writes = {}
		local x = il == 1 and self:embed(g, inp) or inp.hidden
		local last = math.min(nl, il + per - 1)
		for j = il, last do
			x = self:layer(g, inp, j, x)
		end
		if last == nl then
			local outs = {}
			if inp.mode ~= "prefill" then
				outs = { tail(g, self:head(g, inp, x)) }
			end
			local all = table.move(outs, 1, #outs, 1, {})
			table.move(inp.writes, 1, #inp.writes, #all + 1, all)
			g:compute(table.unpack(all))
			return g, table.unpack(outs)
		end
		inp.hidden = inp.hidden or be:tensor("f32", { inp.n, self.hp.n_embd })
		g:compute(g:cpy(x, inp.hidden), table.unpack(inp.writes))
		g:close()
		il = last + 1
	end
end

--- Make cache hold ids.
-- The cache keeps the rows of the longest shared prefix. The rest is
-- prefilled in chunks of ubatch tokens.
-- @tparam table cache a cache from `kv_cache`
-- @tparam {int,...} ids prompt token ids
-- @tparam[opt=512] int ubatch tokens per prefill graph
-- @treturn int the count of reused prefix rows
-- @raise when #ids exceeds the cache capacity
function M:prefill(cache, ids, ubatch)
	ubatch = ubatch or 512
	if #ids > cache.cap then
		error("diffusion-gemma: prompt of " .. #ids .. " tokens exceeds cache of " .. cache.cap)
	end
	local old, n = cache.tokens or {}, 0
	while n < cache.n and n < #ids and old[n + 1] == ids[n + 1] do
		n = n + 1
	end
	cache.n = n
	for s = n + 1, #ids, ubatch do
		local e = math.min(#ids, s + ubatch - 1)
		local inp = self:inputs(table.move(ids, s, e, 1, {}),
		    { mode = "prefill", off = s - 1, cache = cache })
		local g = self:eval(inp)
		g:close()
		M.close_inputs(inp)
		cache.n = e
	end
	cache.tokens = table.move(ids, 1, #ids, 1, {})
	return n
end

--- Build the sampling nodes for canvas logits.
-- Per canvas row: argmax, entropy of softmax(logits * temp_inv) and a
-- multinomial draw against u: the count of cumulative probabilities
-- below u.
-- @static
-- @tparam lgml.graph g the graph
-- @tparam lgml.tensor logits canvas logits (C, n_vocab)
-- @tparam number temp_inv inverse temperature
-- @tparam lgml.tensor u f32 uniform draws (C, 1)
-- @treturn lgml.tensor argmax (C)
-- @treturn lgml.tensor entropy (C)
-- @treturn lgml.tensor drawn token (C)
function M.sample_nodes(g, logits, temp_inv, u)
	local C = logits:shape()[1]
	local best = g:argmax(logits)
	local p = g:soft_max(logits, nil, temp_inv)
	local plogp = g:mul(p, g:log(g:clamp(p, 1e-37, 1)))
	local ent = g:reshape(g:neg(g:sum_rows(plogp)), { C })
	local below = g:step(g:neg(g:sub(g:cumsum(p), u)))
	local drawn = g:reshape(g:sum_rows(below), { C })
	return best, ent, drawn
end

-- Round to f32, so the schedule matches the float math of the reference.
local function f32(x)
	return (string.unpack("<f", string.pack("<f", x)))
end

--- Run the entropy-bound denoiser over one canvas after prompt.
-- Unset sampler options take their values from `eb`.
-- @tparam {int,...} prompt prompt token ids
-- @tparam[opt] table opts options:
-- opts.seed: mt19937 seed (default 0).
-- opts.steps, t_min, t_max, entropy_bound, stability, confidence:
-- sampler settings.
-- opts.kv_cache: true to prefill the prompt into a cache.
-- opts.cache_type: type of that cache.
-- opts.cache: a `kv_cache` kept by the caller.
-- opts.ubatch: prefill chunk size.
-- opts.self_cond: self-conditioning (default true).
-- opts.on_step: on_step(step, argmax); return false to stop.
-- @treturn {int,...} the argmax canvas
-- @treturn int the steps run
function M:denoise(prompt, opts)
	opts = opts or {}
	local be, hp = self.be, self.hp
	local o = {}
	for k, v in pairs(self.eb) do
		o[k] = opts[k] == nil and v or opts[k]
	end
	local P, C, V = #prompt, hp.canvas, hp.n_vocab
	local S = math.max(1, o.steps)
	local rng = mt19937.new(opts.seed or 0)
	local canvas = {}
	for i = 1, C do
		canvas[i] = rng:below(V)
	end
	local cache, own = opts.cache, false
	if opts.kv_cache and not cache then
		cache, own = self:kv_cache(P, opts.cache_type), true
	end
	if cache then
		self:prefill(cache, prompt, opts.ubatch)
	end
	local sc_on = opts.self_cond ~= false and self.sc.gate ~= nil
	local prev <close> = be:tensor("f32", { C, V })
	local u_t <close> = be:tensor("f32", { C, 1 })
	local ids = table.move(prompt, 1, P, 1, {})
	local argmax, prev_argmax = {}, nil
	local held, prev_temp_inv, steps = 0, 1, 0
	for cur = S, 1, -1 do
		local step = S - cur
		local t = f32(f32(o.t_min) + f32(f32(f32(o.t_max) - f32(o.t_min)) * f32(cur / S)))
		local temp_inv = f32(1 / t)
		local u, renoise = {}, {}
		for i = 1, C do
			u[i] = rng:uniform()
			renoise[i] = rng:below(V)
		end
		u_t:set(u)
		local sc = sc_on and step > 0 and { prev = prev, temp_inv = prev_temp_inv } or nil
		local inp
		if cache then
			inp = self:inputs(canvas, { mode = "decode", P = P, cache = cache, sc = sc })
		else
			table.move(canvas, 1, C, P + 1, ids)
			inp = self:inputs(ids, { mode = "unified", P = P, sc = sc })
		end
		local g, best, ent, drawn = self:eval(inp, function(g, logits)
			local b, e, d = M.sample_nodes(g, logits, temp_inv, u_t)
			return b, e, d, g:cpy(logits, prev)
		end)
		argmax = best:get()
		local entropy, drawn_v = ent:get(), drawn:get()
		g:close()
		M.close_inputs(inp)
		steps = steps + 1

		local order = {}
		for i = 1, C do
			order[i] = i
		end
		table.sort(order, function(a, b)
			return entropy[a] < entropy[b]
		end)
		local accepted, cum = {}, 0
		for _, i in ipairs(order) do
			cum = cum + entropy[i]
			if cum - entropy[i] <= o.entropy_bound then
				accepted[i] = true
			end
		end
		local esum = 0
		for i = 1, C do
			canvas[i] = accepted[i] and math.min(math.tointeger(drawn_v[i]), V - 1) or renoise[i]
			esum = f32(esum + entropy[i])
		end
		local same = prev_argmax ~= nil
		for i = 1, C do
			if not same or prev_argmax[i] ~= argmax[i] then
				same = false
				break
			end
		end
		held = same and held + 1 or 0
		prev_argmax, prev_temp_inv = argmax, temp_inv
		if opts.on_step and opts.on_step(step, argmax) == false then
			break
		end
		if held >= o.stability and f32(esum / C) < f32(o.confidence) then
			break
		end
	end
	if own then
		M.close_cache(cache)
	end
	return argmax, steps
end

--- Find the tokens to keep from a canvas.
-- The kept tokens are those before the first token in eog or the start
-- of a loop (a token that repeats at stride 1 or 2 for 6 or more steps).
-- @static
-- @tparam {int,...} canvas token ids
-- @tparam table eog set of end-of-generation ids
-- @treturn int the count of tokens to keep
-- @treturn ?int the end token, if any
function M.trim(canvas, eog)
	local n, cut = #canvas, #canvas
	for i = 1, n do
		if eog[canvas[i]] then
			cut = i - 1
			break
		end
	end
	for i = 1, cut - 1 do
		local loop = false
		for stride = 1, 2 do
			local run, j = 0, i
			while j + stride <= n and canvas[j] == canvas[j + stride] do
				run, j = run + 1, j + 1
			end
			if run >= 6 * stride then
				loop = true
				break
			end
		end
		if loop then
			cut = i - 1
			break
		end
	end
	local stop = cut < n and eog[canvas[cut + 1]] and canvas[cut + 1] or nil
	return cut, stop
end

return M
