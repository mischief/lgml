--- Embedding engine for gguf text models.
-- It serves `/v1/embeddings` through `lgml.server.openai`; the
-- interface is in `lgml.server.engine`. Requests run one at a time.
-- The embedding of an input is the normed hidden state of its last
-- token, scaled to length 1, as gguf `pooling_type` 3 (last) asks.
-- Short inputs of a request share passes, each input with its own
-- attention (see `lgml.transformer.embed_packed`). An input longer than
-- one pass runs alone through the KV cache, in chunks.
-- A text input gets the end token when the gguf sets
-- `tokenizer.ggml.add_eos_token`. Token id inputs run as they are.
-- @module lgml.embed_engine
local lgml = require("lgml")
local budget = require("lgml.budget")
local transformer = require("lgml.transformer")
local stream = require("lgml.server.stream")
local log = require("lgml.log")

local L = log.new("engine")

local M = {}
local E = {}
E.__index = E

-- Rows the KV cache gets at load; it grows on demand.
local KV_INITIAL = 4096
-- The most tokens per pass.
local UBATCH = 2048

--- gguf `pooling_type` values, by name.
-- NONE gives one vector per token; lgml serves only LAST.
M.pooling = { NONE = 0, MEAN = 1, CLS = 2, LAST = 3, RANK = 4 }

--- The pooling type of gguf metadata.
-- @tparam table meta the gguf metadata
-- @treturn ?integer the `<arch>.pooling_type` value, or nil
function M.pooling_of(meta)
	return math.tointeger(meta[tostring(meta["general.architecture"]) .. ".pooling_type"])
end

--- Loads a model and makes an engine.
-- A gguf with no `pooling_type` pools the last token.
-- @tparam table opts the options:
--
-- - `model`: gguf path
-- - `backend`: backend name (default: the best one)
-- - `threads`: CPU thread count (default: `lgml.threads()`)
-- - `n_ctx`: the most tokens of one input (default and at most: the
--   model's)
-- - `flash_attn`: false to use soft_max attention (default true)
-- - `cache_type_k`, `cache_type_v`: KV cache types from `lgml.kv`
--   (default q8_0 with flash attention when the head size fits, else f16)
-- - `kv_initial`: rows the KV cache gets at load (default 4096); it
--   grows when an input needs more, up to `n_ctx`. 0 gives all `n_ctx`
--   rows at load.
-- - `ubatch`: the most tokens per pass (default 2048)
-- - `id`: model id (default: the gguf `general.name`, or `"lgml"`)
-- @treturn Engine the engine
-- @raise when the backend, the model or its tokenizer does not load,
-- or the model pools other than the last token
function M.new(opts)
	local info = assert(lgml.gguf(opts.model))
	local meta = info.meta
	local arch = meta["general.architecture"]
	if not transformer.supports(arch) then
		error(("embeddings: architecture %s not supported"):format(tostring(arch)))
	end
	local pool = M.pooling_of(meta) or M.pooling.LAST
	if pool ~= M.pooling.LAST then
		error(("embeddings: pooling type %d not supported; lgml pools the last token"):format(pool))
	end
	local be = assert(lgml.backend(opts.backend, { threads = opts.threads }))
	local model = transformer.load(be, opts.model)
	local fa = opts.flash_attn ~= false
	local copts = { type_k = opts.cache_type_k or "f16", type_v = opts.cache_type_v or "f16" }
	if fa and not opts.cache_type_k and not opts.cache_type_v and
	    pcall(model.cache_size, model, { type = "q8_0" }) then
		copts = { type = "q8_0" }
	end
	local initial = opts.kv_initial or KV_INITIAL
	if initial > 0 then
		copts.initial = initial
	end
	local n_ctx = math.min(opts.n_ctx or model.hp.n_ctx, model.hp.n_ctx)
	local tk = assert(lgml.tokenizer(opts.model))
	return setmetatable({
		be = be, model = model, tk = tk, n_ctx = n_ctx, flash_attn = fa,
		cache = model:cache(n_ctx, copts), ubatch = opts.ubatch or UBATCH,
		packable = model:packable(),
		eos = meta["tokenizer.ggml.add_eos_token"] and tk:special().eos or nil,
		id = opts.id or meta["general.name"] or "lgml",
		busy = false, waiting = {},
	}, E)
end

--- An embedding engine. It implements `lgml.server.engine.Engine`.
-- @type Engine

--- Lists the one model of the engine.
-- @treturn {table,...} `{ { id, context_length = n_ctx } }`
function E:models()
	return { { id = self.id, context_length = self.n_ctx } }
end

--- Returns the properties: `{ n_ctx, total_slots = 1 }`.
-- @treturn table the properties
function E:props()
	return { n_ctx = self.n_ctx, total_slots = 1 }
end

function E:acquire()
	while self.busy do
		stream.await(function(done)
			self.waiting[#self.waiting + 1] = done
		end)
	end
	self.busy = true
end

function E:release()
	self.busy = false
	local w = table.remove(self.waiting, 1)
	if w then
		w()
	end
end

-- The token ids of each input, or nil and an error message.
function E:tokens(inputs)
	local out, nv = {}, self.tk:n_vocab()
	for i, x in ipairs(inputs) do
		local ids
		if type(x) == "string" then
			ids = self.tk:encode(x)
			if self.eos then
				ids[#ids + 1] = self.eos
			end
		elseif type(x) == "table" then
			ids = {}
			for j, id in ipairs(x) do
				id = math.tointeger(id)
				if not id or id < 0 or id >= nv then
					return nil, ("input %d: token %d is not a token id of the model (0 to %d)")
					    :format(i - 1, j - 1, nv - 1)
				end
				ids[j] = id
			end
		else
			return nil, ("input %d must be a string or a list of token ids"):format(i - 1)
		end
		if #ids == 0 then
			return nil, ("input %d is empty"):format(i - 1)
		elseif #ids > self.n_ctx then
			return nil, ("input %d has %d tokens; context is %d"):format(i - 1, #ids, self.n_ctx)
		end
		out[i] = ids
	end
	return out
end

-- v scaled to length 1, in place.
local function unit(v)
	local s = 0
	for _, x in ipairs(v) do
		s = s + x * x
	end
	s = s > 0 and 1 / math.sqrt(s) or 0
	for j = 1, #v do
		v[j] = v[j] * s
	end
	return v
end

-- The passes of a request: lists of input indexes. Inputs of at most
-- ubatch tokens fill packed passes of at most ubatch tokens; a new pass
-- starts when an input does not fit the last one. A longer input runs
-- alone through the cache, in chunks.
function E:passes(lists)
	local out, cur, n = {}, nil, 0
	for i, ids in ipairs(lists) do
		if not self.packable or #ids > self.ubatch then
			out[#out + 1] = { i, alone = true }
		else
			if not cur or n + #ids > self.ubatch then
				cur, n = {}, 0
				out[#out + 1] = cur
			end
			cur[#cur + 1], n = i, n + #ids
		end
	end
	return out
end

-- The embedding of each token list, scaled to length 1. Returns nil
-- when ctl cancels.
function E:run(lists, ctl)
	local data, opts = {}, { flash_attn = self.flash_attn, ubatch = self.ubatch }
	for _, p in ipairs(self:passes(lists)) do
		if p.alone then
			data[p[1]] = unit(self.model:embed(self.cache, lists[p[1]], opts))
		else
			local group = {}
			for k, i in ipairs(p) do
				group[k] = lists[i]
			end
			for k, v in ipairs(self.model:embed_packed(group, opts)) do
				data[p[k]] = unit(v)
			end
		end
		-- Let other messages run, such as a cancel.
		if coroutine.isyieldable() then
			stream.sleep(0)
		end
		if ctl and ctl.cancelled and ctl.cancelled() then
			return nil
		end
	end
	return data
end

--- Embeds the inputs of a request (see
-- `lgml.server.engine.Engine.embed`).
-- A bad input returns status 400, a cancel 499 and a Lua error 500.
-- @tparam table req the request: `input` is a list; each entry is a
-- string or a list of token ids
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table `{ data, usage = { prompt_tokens }, timings }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function E:embed(req, ctl)
	if type(req.input) ~= "table" or #req.input == 0 then
		return nil, "input must be a non-empty list", 400
	end
	local lists, err = self:tokens(req.input)
	if not lists then
		return nil, err, 400
	end
	local n = 0
	for _, ids in ipairs(lists) do
		n = n + #ids
	end
	self:acquire()
	local t0 = lgml.clock()
	local ok, data = pcall(self.run, self, lists, ctl)
	local t1 = lgml.clock()
	self:release()
	if not ok then
		return nil, tostring(data), budget.status(data, 500)
	elseif not data then
		return nil, "cancelled", 499
	end
	L:debug("embed", "inputs", #lists, "tokens", n, "ms", (t1 - t0) * 1000)
	return {
		data = data, usage = { prompt_tokens = n },
		timings = { prompt_n = n, prompt_ms = (t1 - t0) * 1000,
		    prompt_per_second = n / math.max(t1 - t0, 1e-9) },
	}
end

return M
