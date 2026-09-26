--- Chat engine for gguf text models.
-- It serves `lgml.server.openai`; the interface is in
-- `lgml.server.engine`. One slot: requests run one at a time. The slot
-- keeps its KV cache and token list, so a request that shares a prefix
-- computes only the rest.
-- With a vision encoder (`mmproj`), user messages can hold images. An
-- image takes one prompt position for each of its embeddings. In the
-- token list such a position is the string `"img:<sha256>:<i>"`, so
-- the cached prefix and the prompt store match only the same image.
-- At debug level a chat logs the rendered prompt, with a marker for each
-- image, the sampling settings and the prefill; at trace level, each
-- token (see `lgml.log`).
-- @module lgml.engine
local lgml = require("lgml")
local chat = require("lgml.chat")
local sampler = require("lgml.sampler")
local tokenizer = require("lgml.tokenizer")
local stream = require("lgml.server.stream")
local engine_api = require("lgml.server.engine")
local replay = require("lgml.replay")
local transformer = require("lgml.transformer")
local decide = require("lgml.decide")
local prompt_cache = require("lgml.prompt_cache")
local grammar = require("lgml.grammar")
local lora = require("lgml.lora")
local media = require("lgml.chat.media")
local output = require("lgml.chat.output")
local log = require("lgml.log")

local L = log.new("engine")

local M = {}
local E = {}
E.__index = E

local UBATCH = 512

-- Image embeddings kept in host memory, most recent first.
local EMBD_KEEP = 8

-- Device memory kept free for compute buffers when the engine sizes the
-- context to fit.
local RESERVE = 1536 << 20

-- Reasoning block tags by chat family, for constrained answers.
local THINK = {
	qwen3 = { "<think>", "</think>" },
	qwen35 = { "<think>", "</think>" },
	gemma4 = { "<|channel>thought", "<channel|>" },
}

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
-- - `threads`: CPU thread count (default: `lgml.threads()`)
-- - `n_ctx`: context size (default: the model's, or less so the cache
--   fits in free device memory; at most the model's)
-- - `flash_attn`: false to use soft_max attention (default true)
-- - `cache_type_k`, `cache_type_v`: KV cache types from `lgml.kv`
--   (default q8_0 with flash attention when the head size fits, else f16)
-- - `id`: model id (default: the gguf `general.name`, or `"lgml"`)
-- - `recipe`: a decision recipe for `decide`, from `lgml.decide.recipe`
--   or a recipe file path (default `lgml.decide.default_recipe`)
-- - `cache_ram`: MiB of host RAM for the prompt store (default 8192; 0
--   turns it off, -1 means no limit); see `lgml.prompt_cache`
-- - `ctx_checkpoints`: the most state checkpoints per slot, for models
--   with recurrent layers or a sliding ring (default 8; 0 turns them off)
-- - `lora`: LoRA adapters, a list of { path, scale } (scale default 1);
--   see `lgml.lora.load`. A request can change the scales with
--   `lora = { { id, scale }, ... }`, ids from 0, as llama-server takes.
-- - `mmproj`: a vision mmproj gguf for image input (default none); see
--   `lgml.vision`
-- - `image_min_tokens`, `image_max_tokens`: the token range of one
--   image (default: the encoder's)
-- @treturn Engine the engine
-- @raise when the backend, the model or its tokenizer does not load,
-- when no chat template fits the model, an adapter does not fit it, or
-- the mmproj does not fit it
function M.new(opts)
	local be = assert(lgml.backend(opts.backend, { threads = opts.threads }))
	local info = assert(lgml.gguf(opts.model))
	local family, family_name = chat.load(info.meta)
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
			L:info("context cut to fit free device memory", "n_ctx", n_ctx, "n_ctx_train", model.hp.n_ctx)
		end
	end
	local adapters = {}
	for i, spec in ipairs(opts.lora or {}) do
		local path, scale = spec.path or spec[1], spec.scale or spec[2] or 1
		adapters[i] = { ad = lora.load(model, path, { scale = scale }), path = path, scale = scale }
	end
	local cache = model:cache(n_ctx, copts)
	local e = setmetatable({
		be = be, model = model, family = family, family_name = family_name,
		tk = assert(tokenizer.open(opts.model)),
		n_ctx = n_ctx, flash_attn = fa,
		id = opts.id or info.meta["general.name"] or "lgml",
		cache = cache, tokens = {},
		busy = false, waiting = {}, adapters = adapters,
	}, E)
	if type(opts.recipe) == "string" then
		e.recipe = assert(decide.load_recipe(opts.recipe))
	else
		e.recipe = opts.recipe or decide.default_recipe()
	end
	e.eog = {}
	for _, id in ipairs(e.tk:special().eog or {}) do
		e.eog[id] = true
	end
	e:set_lora()
	if opts.mmproj then
		e:load_vision(opts)
	end
	if model.snapshot then
		local ram = opts.cache_ram or 8192
		e.pc = prompt_cache.new(model, cache, {
			key = e.lora_key,
			ram = ram < 0 and math.huge or math.floor(ram * 1048576),
			checkpoints = opts.ctx_checkpoints or 8,
			special = e.tk:special().tokens,
		})
	end
	return e
end

--- A chat engine. It implements `lgml.server.engine.Engine`.
-- @type Engine

-- Load the vision encoder of opts.mmproj on the model's backend.
function E:load_vision(opts)
	local vision = require("lgml.vision")
	local enc = vision.load(self.be, opts.mmproj, {
		min_tokens = opts.image_min_tokens, max_tokens = opts.image_max_tokens,
	})
	if enc.hp.n_out ~= self.model.hp.n_embd then
		error(("%s: image embeddings have %d values; the model takes %d"):format(opts.mmproj,
		    enc.hp.n_out, self.model.hp.n_embd))
	end
	local im = self.family.image
	if not im then
		error(("chat: the %s template has no image tokens"):format(self.family_name))
	end
	self.image_id = self.tk:special().tokens[im.token]
	if not self.image_id then
		error("tokenizer has no image token " .. im.token)
	end
	self.vision, self.image_marker = enc, im.begin .. im.token .. im.finish
	self.embd_store = {}
end

--- Lists the one model of the engine.
-- @treturn {table,...} `{ { id, context_length = n_ctx, vision } }`;
-- vision is true with an mmproj
function E:models()
	return { { id = self.id, context_length = self.n_ctx, vision = self.vision ~= nil } }
end

--- Returns the properties: `{ n_ctx, total_slots = 1, lora, modalities }`.
-- lora lists the adapters as llama-server does: `{ id, path, scale }`,
-- with the global scales.
-- modalities is `{ vision = true }` with an mmproj, else `{ vision = false }`.
-- @treturn table the properties
function E:props()
	local list = {}
	for i, a in ipairs(self.adapters) do
		list[i] = { id = i - 1, path = a.path, scale = a.scale }
	end
	return { n_ctx = self.n_ctx, total_slots = 1, lora = list,
	    modalities = { vision = self.vision ~= nil } }
end

--- Sets the global adapter scales (see
-- `lgml.server.engine.Engine.lora_adapters`). Adapters that the list
-- does not name get scale 0. The next request applies them.
-- @tparam {table,...} list `{ id, scale }` entries, ids from 0
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int 400
function E:lora_adapters(list)
	local scales, err = engine_api.lora_scales(list, #self.adapters)
	if not scales then
		return nil, err, 400
	end
	for i, a in ipairs(self.adapters) do
		a.scale = scales[i]
	end
	return true
end

--- Set the adapter scales for a request, as llama-server does.
-- With no list, the scales are the global ones. A list sets the scales
-- of the adapters it names; the others get 0. A new adapter set empties
-- the slot: its cache holds keys and values of the old set. The prompt
-- store keeps entries of each set apart.
-- @tparam[opt] {table,...} list `{ id, scale }` entries, ids from 0
-- @treturn[1] nil
-- @treturn[2] string an error message for a bad list
function E:set_lora(list)
	local scales = {}
	for i, a in ipairs(self.adapters) do
		scales[i] = a.scale
	end
	if list ~= nil then
		local err
		scales, err = engine_api.lora_scales(list, #self.adapters)
		if not scales then
			return err
		end
	end
	local key = table.concat(scales, ",")
	if key == self.lora_key then
		return nil
	end
	if self.pc then
		self.pc:save(self)
		self.pc:trim(-1)
		self.pc.key = key
	end
	local on = {}
	for i, a in ipairs(self.adapters) do
		a.ad.scale = scales[i]
		if scales[i] ~= 0 then
			on[#on + 1] = a.ad
		end
	end
	if self.model.set_adapters then
		self.model:set_adapters(on)
	elseif #on > 0 then
		error(self.model.hp.arch .. ": this model takes no adapters")
	end
	self.lora_key = key
	self.tokens, self.cache.n_past = {}, 0
	return nil
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

local function sampler_opts(req, think, extra)
	local o = extra or {}
	for _, k in ipairs({ "temperature", "top_p", "top_k", "min_p", "seed",
	    "presence_penalty", "frequency_penalty", "repeat_penalty" }) do
		if type(req[k]) == "number" then
			o[k] = req[k]
		end
	end
	return sampler.new(think and "qwen3_think" or "qwen3", o)
end

-- Run ids at cache.n_past. Returns the logits node, which stays valid
-- until the next pass: the cache keeps the graph for replay. rows is
-- "last" (the default) or "all". embd holds embeddings for the ids: a
-- non-causal pass reads them in place of the token embeddings.
function E:eval(ids, rows, embd)
	return replay.run(self.model, self.cache, ids, {
		logits = rows or "last", flash_attn = self.flash_attn,
		embd = embd, causal = embd == nil,
	})
end

-- The image key of a token list entry, or nil for a token id.
local function image_key(v)
	return type(v) == "string" and v:match("^(.*):%d+$") or nil
end

-- Move n back to the start of the image that holds position n + 1.
local function span_start(prompt, n)
	local key = image_key(prompt[n + 1])
	while key and n > 0 and image_key(prompt[n]) == key do
		n = n - 1
	end
	return n
end

-- Keep the longest shared prefix of the cached tokens. At least one
-- prompt token must run to get logits. An image runs whole or not at
-- all.
function E:reuse(prompt)
	local n = 0
	local old = self.tokens
	while n < #old and n < #prompt - 1 and old[n + 1] == prompt[n + 1] do
		n = n + 1
	end
	n = span_start(prompt, n)
	-- The prompt cache can swap in a stored slot that shares more.
	if self.pc then
		n = span_start(prompt, self.pc:reuse(self, prompt, n))
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

-- The embeddings of the image with key: from the host store, or from
-- the encoder with the prepared images of this request.
function E:image_embd(key)
	for i, e in ipairs(self.embd_store) do
		if e.key == key then
			table.insert(self.embd_store, 1, table.remove(self.embd_store, i))
			return e.buf
		end
	end
	local img = assert(self.images and self.images[key], "no image for " .. key)
	local t0 = lgml.clock()
	local buf = assert(self.vision:encode(img.prep))
	self.image_ms = (self.image_ms or 0) + (lgml.clock() - t0) * 1000
	table.insert(self.embd_store, 1, { key = key, buf = buf })
	while #self.embd_store > EMBD_KEEP do
		table.remove(self.embd_store).buf:close()
	end
	return buf
end

-- Prefill prompt[from..] in chunks. Returns the last logits node. An
-- image runs as one non-causal pass of its embeddings.
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
		local key = image_key(prompt[i])
		local embd
		if key then
			assert(prompt[i] == key .. ":1", "prefill starts inside an image")
			j = i
			while j < #prompt and image_key(prompt[j + 1]) == key do
				j = j + 1
			end
			embd = self:image_embd(key)
		else
			for k = i + 1, j do
				if image_key(prompt[k]) then
					j = k - 1
					break
				end
			end
		end
		local ids = table.move(prompt, i, j, 1, {})
		if embd then
			for k = 1, #ids do
				ids[k] = 0
			end
		end
		logits = self:eval(ids, "last", embd)
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

--- Runs one chat completion (see `lgml.server.engine.Engine.chat`).
-- It renders the chat template, reuses the cached prefix, prefills and
-- samples. It stops at an end token, at the token limit or on cancel.
-- The request can set sampler settings, `seed`, `max_tokens`,
-- `max_completion_tokens`, `stop` (see `lgml.chat.output`), `tools`,
-- `reasoning_effort = "none"` and
-- `chat_template_kwargs` `enable_thinking`, `reasoning_effort` and
-- `preserve_thinking`. `grammar` (GBNF) or `response_format` (see
-- `lgml.grammar.from_request`) constrain the answer; a reasoning block
-- before it stays free. A bad grammar returns status 400. A Lua error
-- resets the cache and returns status 500.
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
	local lerr = self:set_lora(req.lora)
	if lerr then
		self:release()
		return nil, lerr, 400
	end
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

--- Answers typed questions by logit readout (see
-- `lgml.server.engine.Engine.decide` and `lgml.decide`).
-- The request waits for the slot, as chat does. A Lua error resets the
-- cache and returns status 500. Images in `req.images` need an mmproj;
-- without one they give status 400.
-- @tparam table req the decision request
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table the result: `{ model, answers, usage }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function E:decide(req, ctl)
	self:acquire()
	local lerr = self:set_lora(req.lora)
	if lerr then
		self:release()
		return nil, lerr, 400
	end
	self.image_ms = 0
	local ok, a, b, c = pcall(decide.run, self, req, ctl)
	self:drop_images()
	if not ok then
		self.tokens, self.cache.n_past = {}, 0
	end
	self:release()
	if not ok then
		return nil, tostring(a), 500
	end
	return a, b, c
end

-- Sampler options for the grammar of req: {} when it has none, or nil
-- and an error. With thinking on, the grammar applies after the block.
function E:grammar_opts(req, think, prompt)
	local text, err = grammar.from_request(req)
	if not text then
		return err == nil and {} or nil, err
	end
	local root, tags = "root", THINK[self.family_name]
	if think and tags then
		local opened = prompt:gsub("%s+$", ""):sub(-#tags[1]) == tags[1]
		text, root = grammar.reasoning(text, tags[1], tags[2], opened)
	end
	return { grammar = text, grammar_root = root, tokenizer = self.tk }
end

-- Prepare the images of a request, each `{ bytes }`, for the encoder.
-- Returns their keys in order (`"img:<sha256>"`), or nil and an error
-- message. drop_images frees them.
function E:prepare_images(images)
	self.images = self.images or {}
	local keys = {}
	for k, img in ipairs(images) do
		local key = "img:" .. lgml.sha256(img.bytes)
		local have = self.images[key]
		if not have then
			local prep, err = self.vision:prepare(img.bytes)
			if not prep then
				return nil, ("image %d: %s"):format(k, err)
			end
			have = { prep = prep, n = self.vision:tokens(prep) }
			self.images[key] = have
		end
		if have.n > (self.cache.ubatch or have.n) then
			return nil, ("image %d has %d tokens; a pass takes %d"):format(k, have.n,
			    self.cache.ubatch)
		end
		keys[k] = key
	end
	return keys
end

-- Replace each image token of prompt with one entry for each embedding
-- of the next prepared image. Returns the new prompt, or nil and an
-- error message.
function E:expand_images(prompt, keys)
	local out, k = {}, 0
	for _, id in ipairs(prompt) do
		if id ~= self.image_id then
			out[#out + 1] = id
		else
			k = k + 1
			local key = keys[k]
			if not key then
				return nil, "the prompt holds more image tokens than images"
			end
			for i = 1, self.images[key].n do
				out[#out + 1] = key .. ":" .. i
			end
		end
	end
	if k ~= #keys then
		return nil, "the prompt holds fewer image tokens than images"
	end
	return out
end

-- Put the images of a request into prompt. Returns the new prompt, or
-- nil and an error message.
function E:add_images(prompt, images)
	self.images = {}
	local keys, err = self:prepare_images(images)
	if not keys then
		return nil, err
	end
	return self:expand_images(prompt, keys)
end

-- Free the prepared images of the last request.
function E:drop_images()
	for _, img in pairs(self.images or {}) do
		img.prep:close()
	end
	self.images = nil
end

-- The prompt text for logs: each image token becomes a marker with the
-- image number, key, size and token count.
function E:shown_prompt(text, images)
	if #images == 0 then
		return text
	end
	local tok = self.family.image.token
	local k = 0
	return (text:gsub(tok:gsub("%p", "%%%0"), function()
		k = k + 1
		local img = images[k]
		if not img then
			return tok
		end
		local key = "img:" .. lgml.sha256(img.bytes)
		local have = self.images and self.images[key]
		return ("<image %d %s %dB %d tokens>"):format(k, key:sub(1, 12), #img.bytes,
		    have and have.n or 0)
	end))
end

-- The first lines of a grammar, for logs.
local function first_lines(text, n)
	local out = {}
	for l in text:gmatch("[^\n]+") do
		out[#out + 1] = l
		if #out == n then
			break
		end
	end
	return table.concat(out, "\n")
end

function E:run(req, sink)
	self.image_ms = 0
	local ok, a, b, c = pcall(self.run_prompt, self, req, sink)
	self:drop_images()
	if not ok then
		error(a, 0)
	end
	return a, b, c
end

function E:run_prompt(req, sink)
	local think = thinking(req)
	local kw = type(req.chat_template_kwargs) == "table" and req.chat_template_kwargs or {}
	local effort, keep = kw.reasoning_effort, kw.preserve_thinking
	local messages, images = req.messages, {}
	if media.has_images(messages) then
		if not self.vision then
			return nil, "image input is not supported: start the server with --mmproj", 400
		end
		messages, images = media.extract(messages, self.image_marker)
		if not messages then
			return nil, images, 400
		end
	end
	local text = self.family.render(messages, {
		tools = req.tools, add_generation_prompt = true, enable_thinking = think,
		reasoning_effort = type(effort) == "string" and effort or nil,
		preserve_thinking = keep == false and false or nil,
	})
	local prompt = self.tk:encode(text)
	if self.vision then
		local err
		prompt, err = self:add_images(prompt, images)
		if not prompt then
			return nil, err, 400
		end
	end
	if log.want_prompt() then
		log.prompt(L, "chat", self:shown_prompt(text, images), "model", self.id, "tokens", #prompt,
		    "images", #images > 0 and #images or nil, "thinking", think)
	end
	if #prompt >= self.n_ctx then
		return nil, ("prompt has %d tokens; context is %d"):format(#prompt, self.n_ctx), 400
	end
	local max = req.max_completion_tokens or req.max_tokens
	max = math.min(max or self.n_ctx, self.n_ctx - #prompt)
	local gopts, gerr = self:grammar_opts(req, think, text)
	if not gopts then
		return nil, gerr, 400
	end
	local good, smp = pcall(sampler_opts, req, think, gopts)
	if not good then
		return nil, tostring(smp):gsub("^.-grammar: ", "grammar: "), 400
	end
	local _ <close> = smp
	if L:on(log.DEBUG) then
		local preset = think and "qwen3_think" or "qwen3"
		L:debug("sampling", "preset", preset, "settings", sampler.settings(preset, req),
		    "seed", req.seed or "random", "max_tokens", max, "thinking", think,
		    "grammar", gopts.grammar and first_lines(gopts.grammar, 6), "root", gopts.grammar_root)
	end
	local out = output.new(sink, self.family.new_parser({ tools = req.tools, enable_thinking = think }),
	    { stop = req.stop })
	local dec = tokenizer.decoder(self.tk, { special = false })

	local t0 = lgml.clock()
	local cached = self:reuse(prompt)
	L:debug("prefill", "tokens", #prompt, "cached", cached, "restored", self.pc and self.pc.restored or 0,
	    "compute", #prompt - cached)
	local logits = self:prefill(prompt, cached + 1, sink)
	if not logits then
		return true
	end
	local t1 = lgml.clock()
	local n, reason, stopped = 0, "length", false
	local tracing = log.level >= log.TRACE
	while n < max do
		local id = smp:sample(logits)
		smp:accept(id)
		n = n + 1
		if self.eog[id] then
			if tracing then
				L:trace("token", "n", n, "id", id, "eog", true)
			end
			reason = "stop"
			break
		end
		local piece = dec:push(id)
		if tracing then
			L:trace("token", "n", n, "id", id, "text", piece)
		end
		local r = out:token(piece)
		if r == "stop" then
			reason, stopped = "stop", true
			break
		elseif not r then
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
	-- After a stop string, the rest of the output does not go out.
	if not stopped then
		local r = out:finish(dec:flush())
		if r == "stop" then
			reason = "stop"
		elseif not r then
			return true
		end
	end
	local t2 = lgml.clock()
	sink:finish(reason, {
		prompt_tokens = #prompt, completion_tokens = n, cached_tokens = cached,
		reasoning_tokens = out.reasoning_tokens,
	}, {
		cache_n = cached, restored_n = self.pc and self.pc.restored or 0,
		prompt_n = #prompt - cached, prompt_ms = (t1 - t0) * 1000,
		prompt_per_token_ms = (t1 - t0) * 1000 / math.max(#prompt - cached, 1),
		prompt_per_second = (#prompt - cached) / math.max(t1 - t0, 1e-9),
		predicted_n = n, predicted_ms = (t2 - t1) * 1000,
		predicted_per_token_ms = (t2 - t1) * 1000 / math.max(n, 1),
		predicted_per_second = n / math.max(t2 - t1, 1e-9),
		images = #images, image_encode_ms = self.image_ms,
	})
	return true
end

return M
