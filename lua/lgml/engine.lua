--- Chat engine for gguf text models.
-- It serves `lgml.server.openai`; the interface is in
-- `lgml.server.engine`. One slot: requests run one at a time. The slot
-- keeps its KV cache and token list, so a request that shares a prefix
-- computes only the rest.
-- @module lgml.engine
local lgml = require("lgml")
local chat = require("lgml.chat")
local sampler = require("lgml.sampler")
local tokenizer = require("lgml.tokenizer")
local stream = require("lgml.server.stream")
local replay = require("lgml.replay")
local transformer = require("lgml.transformer")
local prompt_cache = require("lgml.prompt_cache")

local M = {}
local E = {}
E.__index = E

local UBATCH = 512

-- Device memory kept free for compute buffers when the engine sizes the
-- context to fit.
local RESERVE = 1536 << 20

-- The largest context, in steps of 256, whose cache fits in the free
-- memory of the model's device. nil when the device does not report it.
local function fit_ctx(be, model, max, copts)
	if not model.cache_size then
		return nil
	end
	local free
	for _, d in ipairs(lgml.devices()) do
		if d.name == be:name() then
			free = d.free
		end
	end
	if not free then
		return nil
	end
	local per, fixed = model:cache_size(copts)
	if per == 0 then
		return max
	end
	local n = (free - RESERVE - fixed) // per // 256 * 256
	return math.min(max, math.max(n, 256))
end

--- Loads a model and makes an engine.
-- @tparam table opts the options:
--
-- - `model`: gguf path
-- - `backend`: backend name (default: the best one)
-- - `n_ctx`: context size (default: the model's, or less so the cache
--   fits in free device memory; at most the model's)
-- - `flash_attn`: false to use soft_max attention (default true)
-- - `cache_type_k`, `cache_type_v`: KV cache types from `lgml.kv`
--   (default q8_0 with flash attention when the head size fits, else f16)
-- - `id`: model id (default: the gguf `general.name`, or `"lgml"`)
-- - `cache_ram`: MiB of host RAM for the prompt store (default 8192; 0
--   turns it off, -1 means no limit); see `lgml.prompt_cache`
-- - `ctx_checkpoints`: the most state checkpoints per slot, for models
--   with recurrent layers or a sliding ring (default 8; 0 turns them off)
-- @treturn Engine the engine
-- @raise when the backend, the model or its tokenizer does not load,
-- or when no chat template fits the model
function M.new(opts)
	local be = assert(lgml.backend(opts.backend))
	local info = assert(lgml.gguf(opts.model))
	local family = chat.load(info.meta)
	local arch = info.meta["general.architecture"]
	local model = (transformer.supports(arch) and transformer or require("lgml." .. arch))
	    .load(be, opts.model)
	local fa = opts.flash_attn ~= false
	local copts = { type_k = opts.cache_type_k or "f16", type_v = opts.cache_type_v or "f16" }
	if fa and not opts.cache_type_k and not opts.cache_type_v and
	    pcall(model.cache_size, model, { type = "q8_0" }) then
		copts = { type = "q8_0" }
	end
	local n_ctx = math.min(opts.n_ctx or model.hp.n_ctx, model.hp.n_ctx)
	if not opts.n_ctx then
		n_ctx = fit_ctx(be, model, n_ctx, copts) or n_ctx
		if n_ctx < model.hp.n_ctx then
			io.stderr:write(("lgml: context %d of %d, to fit free device memory\n"):format(
			    n_ctx, model.hp.n_ctx))
		end
	end
	local cache = model:cache(n_ctx, copts)
	local e = setmetatable({
		be = be, model = model, family = family,
		tk = assert(tokenizer.open(opts.model)),
		n_ctx = n_ctx, flash_attn = fa,
		id = opts.id or info.meta["general.name"] or "lgml",
		cache = cache, tokens = {},
		busy = false, waiting = {},
	}, E)
	e.eog = {}
	for _, id in ipairs(e.tk:special().eog or {}) do
		e.eog[id] = true
	end
	if model.snapshot then
		local ram = opts.cache_ram or 8192
		e.pc = prompt_cache.new(model, cache, {
			ram = ram < 0 and math.huge or math.floor(ram * 1048576),
			checkpoints = opts.ctx_checkpoints or 8,
			special = e.tk:special().tokens,
		})
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

-- Wait for the slot. Requests queue in arrival order.
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

local function sampler_opts(req, think)
	local o = {}
	for _, k in ipairs({ "temperature", "top_p", "top_k", "min_p", "seed",
	    "presence_penalty", "frequency_penalty", "repeat_penalty" }) do
		if type(req[k]) == "number" then
			o[k] = req[k]
		end
	end
	return sampler.new(think and "qwen3_think" or "qwen3", o)
end

-- Run ids at cache.n_past. Returns the logits node, which stays valid
-- until the next pass: the cache keeps the graph for replay.
function E:eval(ids)
	return replay.run(self.model, self.cache, ids,
	    { logits = "last", flash_attn = self.flash_attn })
end

-- Keep the longest shared prefix of the cached tokens. At least one
-- prompt token must run to get logits.
function E:reuse(prompt)
	local n = 0
	local old = self.tokens
	while n < #old and n < #prompt - 1 and old[n + 1] == prompt[n + 1] do
		n = n + 1
	end
	-- The prompt cache can swap in a stored slot that shares more.
	if self.pc then
		n = self.pc:reuse(self, prompt, n)
		old = self.tokens
	end
	local want = n
	-- A sliding-window ring can lose the rows before n.
	if self.model.rewind and n < #old then
		n = self.model:rewind(self.cache, n)
	end
	-- A recurrent state cannot go back: keep all old tokens or none.
	if self.model.recurrent and n < #old then
		n = 0
	end
	if self.pc and n < want then
		n = self.pc:back(self, want)
	end
	self.cache.n_past = n
	for i = #old, n + 1, -1 do
		old[i] = nil
	end
	if self.pc then
		self.pc:trim(n)
	end
	return n
end

-- Prefill prompt[from..] in chunks. Returns the last logits node.
function E:prefill(prompt, from, sink)
	local logits
	local i = from
	-- Passes end at checkpoint positions.
	local stops, c = self.pc and self.pc:plan(prompt, from) or {}, 1
	while i <= #prompt do
		local j = math.min(#prompt, i + UBATCH - 1)
		if stops[c] and stops[c] <= j then
			j = stops[c]
		end
		logits = self:eval(table.move(prompt, i, j, 1, {}))
		table.move(prompt, i, j, #self.tokens + 1, self.tokens)
		if j == stops[c] then
			self.pc:checkpoint()
			c = c + 1
		end
		i = j + 1
		stream.sleep(0)
		if sink:cancelled() then
			return nil
		end
	end
	return logits
end

local function deliver(sink, ev, state)
	local t = ev.type
	if t == "reasoning" then
		return sink:reasoning(ev.text)
	elseif t == "content" then
		return sink:content(ev.text)
	elseif t == "tool_call_begin" then
		local idx = sink:tool_call(ev.name)
		state[ev.index] = idx
		return idx
	elseif t == "tool_call_args" then
		return sink:tool_arguments(state[ev.index], ev.text)
	end
	return true
end

local function deliver_all(sink, events, state)
	for _, ev in ipairs(events) do
		if not deliver(sink, ev, state) then
			return nil
		end
	end
	return true
end

--- Runs one chat completion (see `lgml.server.engine.Engine.chat`).
-- It renders the chat template, reuses the cached prefix, prefills and
-- samples. It stops at an end token, at the token limit or on cancel.
-- The request can set sampler settings, `seed`, `max_tokens`,
-- `max_completion_tokens`, `tools`, `reasoning_effort = "none"` and
-- `chat_template_kwargs` `enable_thinking`, `reasoning_effort` and
-- `preserve_thinking`. A Lua error resets the cache and returns status
-- 500.
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
	self:acquire()
	local ok, a, b, c = pcall(self.run, self, req, sink)
	if not ok then
		-- The cache may not match the token list now.
		self.tokens, self.cache.n_past = {}, 0
	end
	self:release()
	if not ok then
		return nil, tostring(a), 500
	end
	return a, b, c
end

function E:run(req, sink)
	local think = thinking(req)
	local kw = type(req.chat_template_kwargs) == "table" and req.chat_template_kwargs or {}
	local effort, keep = kw.reasoning_effort, kw.preserve_thinking
	local text = self.family.render(req.messages, {
		tools = req.tools, add_generation_prompt = true, enable_thinking = think,
		reasoning_effort = type(effort) == "string" and effort or nil,
		preserve_thinking = keep == false and false or nil,
	})
	local prompt = self.tk:encode(text)
	if #prompt >= self.n_ctx then
		return nil, ("prompt has %d tokens; context is %d"):format(#prompt, self.n_ctx), 400
	end
	local max = req.max_completion_tokens or req.max_tokens
	max = math.min(max or self.n_ctx, self.n_ctx - #prompt)
	local smp <close> = sampler_opts(req, think)
	local parser = self.family.new_parser({ tools = req.tools, enable_thinking = think })
	local dec = tokenizer.decoder(self.tk, { special = false })
	local state = {}

	local t0 = lgml.clock()
	local cached = self:reuse(prompt)
	local logits = self:prefill(prompt, cached + 1, sink)
	if not logits then
		return true
	end
	local t1 = lgml.clock()
	local n, reason = 0, "length"
	while n < max do
		local id = smp:sample(logits)
		smp:accept(id)
		n = n + 1
		if self.eog[id] then
			reason = "stop"
			break
		end
		if not deliver_all(sink, parser:feed(dec:push(id)), state) then
			return true
		end
		if n >= max then
			break
		end
		logits = self:eval({ id })
		self.tokens[#self.tokens + 1] = id
		stream.sleep(0)
		if sink:cancelled() then
			return true
		end
	end
	local rest = parser:feed(dec:flush())
	for _, ev in ipairs(parser:finish()) do
		rest[#rest + 1] = ev
	end
	if not deliver_all(sink, rest, state) then
		return true
	end
	local t2 = lgml.clock()
	sink:finish(reason, {
		prompt_tokens = #prompt, completion_tokens = n, cached_tokens = cached,
	}, {
		cache_n = cached, restored_n = self.pc and self.pc.restored or 0,
		prompt_n = #prompt - cached, prompt_ms = (t1 - t0) * 1000,
		predicted_n = n, predicted_ms = (t2 - t1) * 1000,
		predicted_per_second = n / math.max(t2 - t1, 1e-9),
	})
	return true
end

return M
