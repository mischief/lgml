--- Chat engine for block diffusion models (DiffusionGemma, LLaDA-MoE-v2).
-- It serves `lgml.server.openai` (see `lgml.server.engine`). Each block
-- is denoised, cut at the model's `trim`, streamed, and appended to the
-- prompt for the next block. One slot. A model with `kv_cache` keeps
-- the shared prompt prefix across blocks and requests.
-- @module lgml.diffusion_engine

-- The model module is `lgml.<general.architecture>`, with `load`,
-- `trim`, `Model:denoise` and `hp.canvas` (the block length). With
-- `options` it takes sampler options (see `lgml.llada-moe-v2`).
-- `END_TOKENS` names more tokens that end a reply.
local lgml = require("lgml")
local budget = require("lgml.budget")
local chat = require("lgml.chat")
local log = require("lgml.log")
local tokenizer = require("lgml.tokenizer")
local stream = require("lgml.server.stream")
local output = require("lgml.chat.output")

local M = {}
local E = {}
E.__index = E
local L = log.new("diffusion")

-- A block cut short by a loop starts the next block at the cut. After
-- this many such blocks in a row, generation stops.
local MAX_LOOP_CUTS = 3

-- The least context that `memory` cuts to.
local MIN_FIT_CTX = 8192

--- Loads a model and makes an engine.
-- @tparam table opts the options:
--
-- - `model`: gguf path
-- - `backend`: backend name (default: the best one)
-- - `threads`: CPU thread count (default: `lgml.threads()`)
-- - `n_ctx`: context size (default 4096, at most the model's)
-- - `flash_attn`: true for flash attention and an f16 KV cache
-- - `id`: model id (default: the gguf `general.name`, or `"lgml"`)
-- - `family`: a chat family module (default: from the gguf)
-- - `seed`: the seed when a request has none (default: random)
-- - `steps`: denoising steps per block
-- - `sampler`: sampler option defaults, for a model module with
--   `options` (`steps` above also sets one)
-- - `ubatch`: the most rows of one pass graph, for a model that runs
--   long passes in chunks (default: the model's)
-- @treturn Engine the engine
-- @raise when the backend, the model or its tokenizer does not load,
-- when no chat template fits the model, or on bad sampler options
function M.new(opts)
	local be = assert(lgml.backend(opts.backend, { threads = opts.threads }))
	local info = assert(lgml.gguf(opts.model))
	local arch = require("lgml." .. info.meta["general.architecture"])
	local model = arch.load(be, opts.model, { flash_attn = opts.flash_attn, chunk = opts.ubatch })
	local n_ctx = math.min(opts.n_ctx or 4096, model.hp.n_ctx)
	local e = setmetatable({
		be = be, model = model, arch = arch, family = opts.family or chat.load(info.meta),
		tk = assert(tokenizer.open(opts.model)),
		n_ctx = n_ctx, canvas = model.hp.canvas, seed = opts.seed, steps = opts.steps,
		id = opts.id or info.meta["general.name"] or "lgml",
		busy = false, waiting = {},
	}, E)
	if model.kv_cache then
		e.cache = model:kv_cache(n_ctx, opts.flash_attn and "f16" or "f32")
	end
	if arch.options then
		local def = {}
		for k, v in pairs(opts.sampler or {}) do
			def[k] = v
		end
		def.steps = def.steps or opts.steps
		def.block_length = def.block_length or model.hp.canvas
		e.sampler = assert(arch.options(def))
	end
	e.eog = {}
	local special = e.tk:special()
	for _, id in ipairs(special.eog or {}) do
		e.eog[id] = true
	end
	for _, name in ipairs(arch.END_TOKENS or {}) do
		local id = special.tokens and special.tokens[name]
		if id then
			e.eog[id] = true
		end
	end
	return e
end

--- A chat engine. It implements `lgml.server.engine.Engine`.
-- @type Engine

--- The memory plan of the engine, for `lgml.budget.report`.
-- A model with `plan` plans its largest passes at n_ctx with the
-- sampler defaults. When they do not fit the memory budget, the context
-- drops by a quarter, in steps of 256, down to 8192.
-- @treturn table `{ backend, compute, cache, cache_now }`: the device
-- bytes of the passes, and of a prefix cache at n_ctx (0 without one)
function E:memory()
	local model, so = self.model, self.sampler or {}
	if not model.plan then
		return { backend = self.be }
	end
	while true do
		local ok, c, cache = pcall(function()
			local c, cache = model:plan(self.n_ctx, { block_length = so.block_length, prefix_cache = so.prefix_cache })
			local fits, err = lgml.memory_check(self.be, c + cache, "the compute buffers")
			if not fits then
				error(err, 0)
			end
			return c, cache
		end)
		if ok then
			L:info("memory plan", "n_ctx", self.n_ctx, "chunk", model.chunk, "compute_mib", c // 1048576,
			    "cache_mib", cache // 1048576)
			return { backend = self.be, compute = c, cache = cache, cache_now = 0 }
		end
		local n = self.n_ctx * 3 // 4 // 256 * 256
		if not budget.is_budget(c) or n < MIN_FIT_CTX then
			error(c, 0)
		end
		L:warn("context cut to fit the memory budget", "n_ctx", n, "from", self.n_ctx)
		self.n_ctx = n
	end
end

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

local function thinking(req)
	local k = req.chat_template_kwargs
	if type(k) == "table" and k.enable_thinking ~= nil then
		return k.enable_thinking ~= false
	end
	return req.reasoning_effort ~= "none"
end

-- Request fields that the denoiser cannot use, with the value that
-- changes nothing, if any.
local NEUTRAL = {
	{ "temperature" }, { "top_p" }, { "top_k" }, { "min_p" }, { "grammar" }, { "lora" },
	{ "reasoning_budget" },
	{ "presence_penalty", 0 }, { "frequency_penalty", 0 }, { "repeat_penalty", 1 },
	{ "logprobs", false },
}

-- The first field of req that the engine cannot do, or nil. A model
-- with sampler options takes temperature and diffusion.
local function unsupported(req, options)
	for _, f in ipairs(NEUTRAL) do
		local v = req[f[1]]
		if v ~= nil and v ~= f[2] and not (options and f[1] == "temperature") then
			return f[1]
		end
	end
	if req.diffusion ~= nil and not options then
		return "diffusion"
	end
	local kw = req.chat_template_kwargs
	for k in pairs(type(kw) == "table" and kw or {}) do
		if k ~= "enable_thinking" then
			return "chat_template_kwargs." .. tostring(k)
		end
	end
	if type(req.logit_bias) == "table" and next(req.logit_bias) ~= nil then
		return "logit_bias"
	elseif req.tool_choice == "required" or type(req.tool_choice) == "table" then
		return "tool_choice"
	end
	return nil
end

--- Runs one chat completion (see `lgml.server.engine.Engine.chat`).
-- The prompt and one canvas must fit in the context. Generation stops
-- at an end token, at the token limit, at the context end, or after 3
-- blocks in a row that end in a loop. The request can set `seed`,
-- `max_tokens`, `max_completion_tokens`, `stop`, `tools`, `reasoning_effort =
-- "none"` and `chat_template_kwargs.enable_thinking`. The sampler
-- fields (temperature, `top_p`, `top_k`, `min_p`, the penalties,
-- `logit_bias` and `logprobs`), a grammar or JSON schema, `lora`, other
-- template kwargs, `reasoning_budget` and a `tool_choice` that forces a
-- call give status 400. A Lua error resets the cache and returns
-- status 500. `count_only`, `sink:prompt` and the `stop_sequence` of
-- the usage work as in `lgml.engine`.
-- @tparam table req the chat request
-- @tparam lgml.server.engine.Sink sink the reply
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function E:chat(req, sink)
	if type(req.messages) ~= "table" or #req.messages == 0 then
		return nil, "messages must be a non-empty array", 400
	end
	local bad = unsupported(req, self.sampler)
	if bad then
		return nil, ("%s: a diffusion model does not take it"):format(bad), 400
	end
	-- Sampler options: the engine's, then temperature and the fields of
	-- the request object diffusion.
	local so
	if self.sampler then
		local o = {}
		for k, v in pairs(self.sampler) do
			o[k] = v
		end
		o.temperature = req.temperature or o.temperature
		if req.diffusion ~= nil and type(req.diffusion) ~= "table" then
			return nil, "diffusion must be an object", 400
		end
		for k, v in pairs(req.diffusion or {}) do
			o[k] = v
		end
		local err
		so, err = self.arch.options(o)
		if not so then
			return nil, "diffusion: " .. err, 400
		end
	end
	self:acquire()
	local ok, a, b, c = pcall(self.run, self, req, sink, so)
	if not ok and self.cache then
		-- The cache may not match its token list now.
		self.cache.n, self.cache.tokens = 0, {}
	end
	self:release()
	if not ok then
		return nil, tostring(a), budget.status(a, 500)
	end
	return a, b, c
end

function E:run(req, sink, so)
	local C = so and so.block_length or self.canvas
	local text = self.family.render(req.messages, {
		tools = req.tools, add_generation_prompt = true, enable_thinking = thinking(req),
	})
	local prompt = self.tk:encode(text)
	if req.count_only then
		sink:finish("stop", { prompt_tokens = #prompt, completion_tokens = 0, cached_tokens = 0,
		    reasoning_tokens = 0 })
		return true
	end
	if #prompt + C > self.n_ctx then
		return nil, ("prompt (%d tokens) + canvas (%d tokens) exceeds the context (%d tokens)")
		    :format(#prompt, C, self.n_ctx), 400
	end
	local max = req.max_completion_tokens or req.max_tokens
	max = math.min(max or self.n_ctx, self.n_ctx - #prompt)
	local seed = math.tointeger(req.seed) or self.seed or math.random(0, 0x7fffffff)
	local out = output.new(sink, self.family.new_parser({ tools = req.tools or {} }),
	    { stop = req.stop, one_call = req.parallel_tool_calls == false })
	local dec = tokenizer.decoder(self.tk, { special = false, keep = self.family.preserved })

	local t0 = lgml.clock()
	local prefix = table.move(prompt, 1, #prompt, 1, {})
	local cached, cancelled, c = 0, false, self.cache
	while c and cached < c.n and cached < #prompt and c.tokens[cached + 1] == prompt[cached + 1] do
		cached = cached + 1
	end
	if sink.prompt and not sink:prompt(#prompt, cached) then
		return true
	end
	local dopts = {}
	for k, v in pairs(so or {}) do
		dopts[k] = v
	end
	dopts.seed, dopts.steps = seed, so and so.steps or self.steps
	dopts.kv_cache, dopts.cache = c ~= nil, c
	function dopts.on_step()
		stream.sleep(0)
		cancelled = sink:cancelled()
		return not cancelled
	end
	local n, cuts, reason, stopped = 0, 0, "length", false
	local steps, blocks = 0, 0
	while n < max and #prefix + C <= self.n_ctx do
		local canvas, s = self.model:denoise(prefix, dopts)
		if cancelled then
			return true
		end
		steps, blocks = steps + s, blocks + 1
		local cut, stop = self.model.trim(canvas, self.eog)
		cut = math.min(cut, max - n)
		for i = 1, cut do
			local r = out:token(dec:push(canvas[i]))
			if r == "stop" then
				n, stopped = n + i, true
				break
			elseif not r then
				return true
			end
		end
		if stopped then
			reason = "stop"
			break
		end
		n = n + cut
		if stop and n < max then
			n, reason = n + 1, "stop"
			break
		end
		cuts = cut < C and cuts + 1 or 0
		if cuts >= MAX_LOOP_CUTS then
			break
		end
		table.move(canvas, 1, cut, #prefix + 1, prefix)
	end
	if not stopped then
		local r = out:finish(dec:flush())
		if r == "stop" then
			reason = "stop"
		elseif not r then
			return true
		end
	end
	local t1 = lgml.clock()
	sink:finish(reason, {
		prompt_tokens = #prompt, completion_tokens = n, cached_tokens = cached,
		reasoning_tokens = out.reasoning_tokens, stop_sequence = out.stop_sequence,
	}, {
		cache_n = cached, prompt_n = #prompt - cached,
		predicted_n = n, predicted_ms = (t1 - t0) * 1000,
		predicted_per_second = n / math.max(t1 - t0, 1e-9),
		steps_n = steps, blocks_n = blocks,
	})
	return true
end

return M
