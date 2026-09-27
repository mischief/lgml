--- Chat engine for block diffusion models (DiffusionGemma).
-- It serves `lgml.server.openai`; the interface is in
-- `lgml.server.engine`. Each block of canvas_length tokens is denoised,
-- trimmed at an end token or a loop, streamed to the sink, and appended
-- to the prompt for the next block. One slot. The prompt KV cache keeps
-- the shared prefix across blocks and requests.
-- @module lgml.diffusion_engine
local lgml = require("lgml")
local budget = require("lgml.budget")
local chat = require("lgml.chat")
local tokenizer = require("lgml.tokenizer")
local stream = require("lgml.server.stream")
local output = require("lgml.chat.output")

local M = {}
local E = {}
E.__index = E

-- A block cut short by a loop starts the next block at the cut. After
-- this many such blocks in a row, generation stops.
local MAX_LOOP_CUTS = 3

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
-- @treturn Engine the engine
-- @raise when the backend, the model or its tokenizer does not load,
-- or when no chat template fits the model
function M.new(opts)
	local be = assert(lgml.backend(opts.backend, { threads = opts.threads }))
	local info = assert(lgml.gguf(opts.model))
	local arch = require("lgml." .. info.meta["general.architecture"])
	local model = arch.load(be, opts.model, { flash_attn = opts.flash_attn })
	local n_ctx = math.min(opts.n_ctx or 4096, model.hp.n_ctx)
	local e = setmetatable({
		be = be, model = model, family = opts.family or chat.load(info.meta),
		tk = assert(tokenizer.open(opts.model)),
		n_ctx = n_ctx, canvas = model.hp.canvas, seed = opts.seed, steps = opts.steps,
		id = opts.id or info.meta["general.name"] or "lgml",
		cache = model:kv_cache(n_ctx, opts.flash_attn and "f16" or "f32"),
		busy = false, waiting = {},
	}, E)
	e.eog = {}
	for _, id in ipairs(e.tk:special().eog or {}) do
		e.eog[id] = true
	end
	return e
end

--- A chat engine. It implements `lgml.server.engine.Engine`.
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

-- The first field of req that the engine cannot do, or nil.
local function unsupported(req)
	for _, f in ipairs(NEUTRAL) do
		local v = req[f[1]]
		if v ~= nil and v ~= f[2] then
			return f[1]
		end
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
	local bad = unsupported(req)
	if bad then
		return nil, ("%s: a diffusion model does not take it"):format(bad), 400
	end
	self:acquire()
	local ok, a, b, c = pcall(self.run, self, req, sink)
	if not ok then
		-- The cache may not match its token list now.
		self.cache.n, self.cache.tokens = 0, {}
	end
	self:release()
	if not ok then
		return nil, tostring(a), budget.status(a, 500)
	end
	return a, b, c
end

function E:run(req, sink)
	local C = self.canvas
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
	local dec = tokenizer.decoder(self.tk, { special = false })

	local t0 = lgml.clock()
	local prefix = table.move(prompt, 1, #prompt, 1, {})
	local cached, cancelled = 0, false
	while cached < self.cache.n and cached < #prompt and
	    self.cache.tokens[cached + 1] == prompt[cached + 1] do
		cached = cached + 1
	end
	if sink.prompt and not sink:prompt(#prompt, cached) then
		return true
	end
	local n, cuts, reason, stopped = 0, 0, "length", false
	while n < max and #prefix + C <= self.n_ctx do
		local canvas = self.model:denoise(prefix, {
			seed = seed, steps = self.steps, kv_cache = true, cache = self.cache,
			on_step = function()
				stream.sleep(0)
				cancelled = sink:cancelled()
				return not cancelled
			end,
		})
		if cancelled then
			return true
		end
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
	})
	return true
end

return M
