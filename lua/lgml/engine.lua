--- Chat engine for gguf text models.
-- It serves `lgml.server.openai`; the interface is in
-- `lgml.server.engine`. The engine keeps a few slots. A slot is one
-- sequence of a shared KV pool, with its token list, so a request that
-- shares a prefix with a slot computes only the rest. The pool holds
-- n_ctx cells for all slots: a slot takes blocks of cells as its
-- sequence grows, and an idle slot gives its blocks up when a running
-- one needs them. Full blocks stay cached by a key of their prefix, so
-- any slot can take them up (see `lgml.kvpool`); evicted ones spill to
-- the RAM store. Requests have priorities: the pool keeps the cells of
-- high ones longest, and a running request of low priority waits for
-- a high one when the pool is full (see `E:preempt`).
-- With a vision encoder (`mmproj`), user messages can hold images. An
-- image takes one prompt position for each of its embeddings. In the
-- token list such a position is the string `"img:<sha256>:<i>"`, so
-- the cached prefix and the prompt store match only the same image.
-- At debug level a chat logs the rendered prompt, with a marker for each
-- image, the sampling settings and the prefill; at trace level, each
-- token (see `lgml.log`).
-- @module lgml.engine
local lgml = require("lgml")
local budget_mod = require("lgml.budget")
local chat = require("lgml.chat")
local sampler = require("lgml.sampler")
local tokenizer = require("lgml.tokenizer")
local stream = require("lgml.server.stream")
local uv = require("luv")
local engine_api = require("lgml.server.engine")
local replay = require("lgml.replay")
local transformer = require("lgml.transformer")
local decide = require("lgml.decide")
local prompt_cache = require("lgml.prompt_cache")
local kvpool = require("lgml.kvpool")
local grammar = require("lgml.grammar")
local lora = require("lgml.lora")
local media = require("lgml.chat.media")
local output = require("lgml.chat.output")
local placement = require("lgml.placement")
local xcache = require("lgml.xcache")
local log = require("lgml.log")
local spec = require("lgml.spec")

local L = log.new("engine")

local M = {}

-- Rows the full-attention KV cache gets at load; it grows on demand.
local KV_INITIAL = 4096

-- A prompt that shares fewer tokens with every slot takes the least
-- recently used slot instead of the one with the longest match.
local SLOT_MIN_SHARED = 32
local E = {}
E.__index = E

local UBATCH = 512
-- The prompt chunk sizes that ubatch "auto" tries, the largest first.
local AUTO_UBATCH = { 4096, 2048, 1024, 512 }
-- The default count of one token in a row that ends a reply.
local REPEAT_STOP = 256
-- The least milliseconds between two progress reports of a request.
local PROGRESS_MS = 1000
-- The smallest context that `reserve` cuts to before it gives up.
local MIN_FIT_CTX = 8192

-- Graphs of small passes that a speculating slot keeps, besides the
-- last one: decode steps and verify passes of a few lengths.
local SPEC_GRAPHS = 6

-- Image embeddings kept in host memory, most recent first.
local EMBD_KEEP = 8

-- Device memory kept free for compute buffers when the engine sizes the
-- context to fit.
local RESERVE = 1536 << 20

-- Device memory that an automatic expert cache leaves free, past the
-- memory guard margin and the planned growth of the engine.
local XCACHE_SLACK = 256 << 20

-- The expert cache options of the engine options, or nil without
-- moe_cache. Sizes in MiB become bytes; bytes is nil for "auto".
local function xcache_opts(opts)
	local v = opts.moe_cache
	if v == nil or v == false then
		return nil
	end
	local function mib(x, key)
		local n = tonumber(x)
		if not n or n < 0 then
			error(("engine: %s must be MiB, not %s"):format(key, tostring(x)), 0)
		end
		return math.floor(n * 1048576)
	end
	local x = {
		profile = opts.moe_cache_profile,
		halflife = opts.moe_cache_halflife,
		swap = opts.moe_cache_swap and mib(opts.moe_cache_swap, "moe_cache_swap"),
		max = opts.moe_cache_max and mib(opts.moe_cache_max, "moe_cache_max"),
		zerocopy = opts.moe_cache_zerocopy,
	}
	if v ~= true and v ~= "auto" then
		x.bytes = mib(v, "moe_cache")
	end
	local f = x.profile and io.open(x.profile, "r")
	if f then
		f:close()
	elseif x.profile then
		L:warn("expert profile not found; the cache starts empty", "path", x.profile)
		x.profile = nil
	end
	return x
end

-- The placement options with the routed experts of every layer on the
-- CPU, for an expert cache.
local function cache_placement(opts)
	local o = {}
	for k, v in pairs(opts) do
		o[k] = v
	end
	if opts.n_cpu_moe ~= nil then
		L:warn("moe_cache keeps every expert in host memory; n_cpu_moe ignored")
	end
	o.cpu_moe, o.n_cpu_moe = true, nil
	return o
end

-- Make the expert cache of x (see `xcache_opts`) on model.
local function new_xcache(model, x)
	local xc = xcache.new(model, {
		slots = x.slots, bytes = x.bytes, swap = x.swap, halflife = x.halflife,
		profile = x.profile, zerocopy = x.zerocopy,
	})
	L:info("expert cache", "slots", xc.n, "experts", xc.ne, "device_mib", xc.bytes // 1048576)
	return xc
end

-- The cancel test of a reply sink, as a plain function.
local function cancel_of(sink)
	return function()
		return sink:cancelled()
	end
end

-- The priority of a request: LOW for decisions, HIGH for a chat that
-- asks to keep its prefix (prompt_cache_retention "24h"), else NORMAL.
local function prio_of(req, decision)
	if decision then
		return kvpool.LOW
	end
	return req.prompt_cache_retention == "24h" and kvpool.HIGH or kvpool.NORMAL
end

-- Reasoning block tags by chat family, for constrained answers.
local THINK = {
	qwen3 = { "<think>", "</think>" },
	qwen35 = { "<think>", "</think>" },
	gemma4 = { "<|channel>thought", "<channel|>" },
}

-- The line that the engine adds to the reasoning at the reasoning
-- budget, before the end of the block.
local WRAP_UP = "\n\nConsidering the limited time by the user, I have to give the solution " ..
    "based on the thinking directly now."

-- The text that closes a reasoning block at the reasoning budget, as
-- the chat template renders the end of the block.
local THINK_END = {
	qwen3 = "\n</think>\n\n",
	qwen35 = "\n</think>\n\n",
	gemma4 = "<channel|>",
}

-- The largest context, in steps of 256, whose cache fits in the free
-- memory of the model's device, with the per-sequence part for n_slots
-- sequences. nil when the device does not report it.
local function fit_ctx(be, model, max, copts, n_slots)
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
	local n = (free - RESERVE - fixed * n_slots) // per // 256 * 256
	return math.min(max, math.max(n, 256))
end

-- True when the backend name, or the best backend without one, is a CPU.
local function on_cpu(name)
	if name then
		return name:upper():match("^CPU") ~= nil
	end
	for _, d in ipairs(lgml.devices()) do
		if d.type == "gpu" or d.type == "igpu" then
			return false
		end
	end
	return true
end

-- The device type of a backend: "gpu", "igpu", "cpu" or nil.
local function dev_type(be)
	for _, d in ipairs(lgml.devices()) do
		if d.name == be:name() then
			return d.type
		end
	end
end

-- The most host RAM, as a fraction, that pinned weights may take. The
-- driver maps pinned memory into a GPU address space of limited size
-- (on amdgpu, GTT: half of RAM). Past it, command submits fail and the
-- device is lost.
local PIN_MAX = 0.4

-- True when the weights that rules put on the CPU go to pinned memory:
-- opts.pinned when set, else on a discrete GPU when they fit PIN_MAX.
local function pin_host(be, rules, info, opts)
	if opts.pinned ~= nil then
		return opts.pinned
	end
	if dev_type(be) == "igpu" then
		-- RAM is shared, so pinned memory gives no speed, and it
		-- counts against the GPU memory limit.
		return false
	end
	if not rules or not transformer.supports(info.meta["general.architecture"]) then
		return true
	end
	local hp = transformer.config(info.meta)
	local _, bytes = placement.plan(rules, transformer.weight_tensors(info, hp), hp.n_layer)
	for _, d in ipairs(lgml.devices()) do
		if d.type == "cpu" and d.total and bytes > d.total * PIN_MAX then
			L:info("weights on the CPU too large to pin", "cpu_mib", bytes // 1048576,
			    "limit_mib", math.floor(d.total * PIN_MAX) // 1048576)
			return false
		end
	end
	return true
end

-- The cache options of a model shape: q8_0 with flash attention when
-- it fits and no type is set, else the types of opts (default f16).
local function shape_copts(shape, opts)
	if opts.flash_attn ~= false and not opts.cache_type_k and not opts.cache_type_v and
	    pcall(transformer.cache_size, shape, { type = "q8_0" }) then
		return { type = "q8_0" }
	end
	return { type_k = opts.cache_type_k or "f16", type_v = opts.cache_type_v or "f16" }
end

-- Set rules.n_cpu_moe to the fewest expert layers on the CPU so the
-- weights and the cache of the wanted context fit in free device memory.
-- The per-sequence part of the cache counts once per slot.
-- The MTP gguf that the speculative options name, or nil.
local function mtp_file(opts)
	local so = opts.speculative
	return type(so) == "table" and type(so.mtp) == "string" and so.mtp or nil
end

local function auto_moe(be, rules, info, opts, n_slots)
	local hp = transformer.config(info.meta)
	local free
	for _, d in ipairs(lgml.devices()) do
		if d.name == be:name() then
			free = d.free
		end
	end
	local n = 0
	if free then
		local shape = { hp = hp, layers = hp.layers }
		local per, fixed = transformer.cache_size(shape, shape_copts(shape, opts))
		local ctx = opts.n_ctx or math.min(hp.n_ctx, placement.AUTO_CTX)
		local mmproj = 0
		for _, extra in ipairs({ opts.mmproj or false, mtp_file(opts) or false }) do
			local st = extra and require("luv").fs_stat(extra)
			mmproj = mmproj + (st and st.size or 0)
		end
		n = placement.auto(rules, transformer.weight_tensors(info, hp), hp.n_layer, free - RESERVE - mmproj - fixed * n_slots - per * ctx)
	end
	L:info("n_cpu_moe auto", "n_cpu_moe", n, "free_mib", free and free // 1048576)
	rules.n_cpu_moe, rules.auto = n, nil
end

-- The slot count: opts.slots, else the default of lgml.kvpool for the
-- pool of the wanted context. lgml.estimate counts the same.
local function slot_count(info, opts)
	if opts.slots then
		return math.max(1, opts.slots)
	end
	local arch = info.meta["general.architecture"]
	if not transformer.supports(arch) then
		return kvpool.SEQS
	end
	local hp = transformer.config(info.meta)
	local shape = { hp = hp, layers = hp.layers }
	local ok, per, fixed = pcall(transformer.cache_size, shape, shape_copts(shape, opts))
	if not ok then
		return kvpool.SEQS
	end
	return kvpool.default_seqs(per * math.min(opts.n_ctx or hp.n_ctx, hp.n_ctx), fixed)
end

-- The sampler options of the gguf general.sampling keys, as llama.cpp
-- reads them, or nil when the file has none.
local GGUF_SAMPLING = {
	temp = "temperature", top_k = "top_k", top_p = "top_p", min_p = "min_p",
	penalty_repeat = "repeat_penalty", penalty_last_n = "repeat_last_n",
}
local function model_sampling(meta)
	local o
	for key, opt in pairs(GGUF_SAMPLING) do
		local v = meta["general.sampling." .. key]
		if type(v) == "number" then
			o = o or {}
			o[opt] = v
		end
	end
	return o
end

--- Loads a model and makes an engine.
-- @tparam table opts the options:
--
-- - `model`: gguf path
-- - `backend`: backend name (default: the best one)
-- - `threads`: CPU thread count (default: `lgml.threads()`)
-- - `n_ctx`: context size (default: the model's, or less so the cache
--   fits in free device memory; at most the model's). It is the size of
--   the KV pool that all slots share, and the most tokens of one request.
-- - `flash_attn`: false to use soft_max attention (default true)
-- - `cache_type_k`, `cache_type_v`: KV cache types from `lgml.kv`
--   (default q8_0 with flash attention when the head size fits, else f16)
-- - `id`: model id (default: the gguf `general.name`, or `"lgml"`)
-- - `recipe`: a decision recipe for `decide`, from `lgml.decide.recipe`
--   a recipe file path, or "jevify" (the default) or "nimble" (see
--   `lgml.decide.nimble_recipe`)
-- - `cache_ram`: MiB of host RAM for the prompt store and the state
--   index (default 8192; 0 turns them off, -1 means no limit); see
--   `lgml.prompt_cache`
-- - `kv_initial`: cells the KV pool gets at load (default 4096); the
--   pool doubles when its sequences need more, up to `n_ctx`. 0 gives
--   all `n_ctx` cells at load.
-- - `slots`: the most sequences the engine keeps. Each request takes
--   the slot that shares the longest prefix with its prompt, so clients
--   that take turns keep their caches; see `E:pick_slot`. Slots share
--   the pool, so a slot costs only its recurrent states and rings, not
--   n_ctx cells. The default is 4, or fewer when the states and rings
--   of 4 slots would take more than a quarter of the pool; see
--   `lgml.kvpool.default_seqs`. With two or more, requests run at the
--   same time in batched passes.
-- - `ctx_checkpoints`: the most state checkpoints per slot, for models
--   with recurrent layers or a sliding ring (default 8; 0 turns them off)
-- - `lora`: LoRA adapters, a list of { path, scale } (scale default 1);
--   see `lgml.lora.load`. A request can change the scales with
--   `lora = { { id, scale }, ... }`, ids from 0, as llama-server takes.
-- - `mmproj`: a vision mmproj gguf for image input (default none); see
--   `lgml.vision`
-- - `image_min_tokens`, `image_max_tokens`: the token range of one
--   image (default: the encoder's)
-- - `cpu_moe`, `n_cpu_moe`, `n_gpu_layers`, `override`: weights to keep
--   in host memory, as `lgml.placement.rules` takes; the ops on them run
--   on the CPU. `n_cpu_moe = "auto"` picks the fewest expert layers on
--   the CPU so the weights and the cache of `n_ctx` positions (default:
--   at most `lgml.placement.AUTO_CTX`) fit in free device memory. They
--   apply only on a GPU backend.
-- - `repeat_stop_tokens`: a reply that samples the same token this many
--   times in a row ends with reason `length` (default 256; 0 turns the
--   guard off). The finish timings then have `repeat_stop = 1`.
-- - `reasoning_budget_tokens`: the reasoning budget of a request
--   without `reasoning_budget` (default none); see `E:chat`
-- - `lazy_tool_grammar`: false turns off the grammar that keeps tool
--   calls to the request tools under tool_choice "auto" (default on)
-- - `ubatch`: the most prompt tokens per pass (default 512), or
--   `"auto"`: the largest of 4096, 2048, 1024 and 512 whose compute
--   buffers fit the memory budget, as `E:reserve` plans them. A model
--   without a KV pool takes 512 for auto.
-- - `speculative`: true, or a table, turns speculative decoding on by
--   default; a request can turn it on or off with `speculative`. The
--   table holds `max` (the most draft tokens, default 16), `adaptive`
--   (false drafts `max` tokens at each step, default true), and
--   `ngram_min`, `ngram_max` for the prompt lookup drafter (see
--   `lgml.spec.lookup`). `drafter`, a function of the request that
--   returns a drafter, replaces prompt lookup (see `lgml.spec`). `mtp`:
--   the path of an MTP gguf for a qwen4exp model, or false to leave the
--   MTP layer of the model file unused. A model with an MTP layer
--   drafts with it first (see `lgml.spec.mtp`).
-- - `moe_cache`: an adaptive expert cache (see `lgml.xcache`): every
--   routed expert stays in host memory, and device slots hold the most
--   used ones. A number is the MiB of the slots; `"auto"` or true fills
--   the device memory that is free after the KV cache of `n_ctx` and the
--   compute buffers, at most `moe_cache_max` MiB. `moe_cache_profile`
--   names a profile file for the first scores (see `XCache:save`);
--   `moe_cache_swap` is the most MiB moved into slots per pass;
--   `moe_cache_halflife` is the half-life of a pick in tokens. It
--   applies only on a GPU backend, and replaces `cpu_moe`/`n_cpu_moe`.
--   `moe_cache_zerocopy` keeps the slots fixed and lets the device read
--   the other experts from pinned host memory (see `XCache:zerocopy`).
-- @treturn Engine the engine
-- @raise when the backend, the model or its tokenizer does not load,
-- when no chat template fits the model, an adapter does not fit it, the
-- mmproj does not fit it, or a placement key is bad
function M.new(opts)
	local info = assert(lgml.gguf(opts.model))
	local family, family_name = chat.load(info.meta)
	local arch = info.meta["general.architecture"]
	local xopts = xcache_opts(opts)
	local rules = placement.rules(xopts and cache_placement(opts) or opts)
	if rules and (on_cpu(opts.backend) or not transformer.supports(arch)) then
		L:warn("weight placement needs a GPU backend and a transformer model; ignored",
		    "backend", opts.backend or "best", "arch", arch)
		rules, xopts = nil, nil
	end
	local be = assert(lgml.backend(opts.backend, { threads = opts.threads, cpu = rules ~= nil }))
	local n_slots = slot_count(info, opts)
	if rules and rules.auto then
		auto_moe(be, rules, info, opts, n_slots)
	end
	-- Speculation loads the MTP layer of a model that has one.
	local so = opts.speculative
	local mtp = (so == true or type(so) == "table") and arch == "qwen4exp" and
	    (mtp_file(opts) or (type(so) ~= "table" or so.mtp ~= false)) or nil
	local model = (transformer.supports(arch) and transformer or require("lgml." .. arch))
	    .load(be, opts.model, (rules or mtp) and {
		placement = rules,
		pinned = pin_host(be, rules, info, opts),
		mtp = mtp,
	    } or nil)
	if rules then
		local dev, host = model.w:size()
		L:info("weights placed", "device_mib", dev // 1048576, "cpu_mib", host // 1048576,
		    "n_cpu_moe", rules.n_cpu_moe, "n_gpu_layers", rules.n_gpu_layers)
	end
	if xopts and not model.hp.n_expert then
		L:warn("moe_cache needs a model with routed experts; ignored")
		xopts = nil
	end
	-- A cache of a fixed size comes before the KV cache; "auto" after it.
	if xopts then
		if not xopts.bytes then
			xopts.slots = 1
		end
		new_xcache(model, xopts)
	end
	local fa = opts.flash_attn ~= false
	local copts = { type_k = opts.cache_type_k or "f16", type_v = opts.cache_type_v or "f16" }
	if fa and not opts.cache_type_k and not opts.cache_type_v and
	    pcall(model.cache_size, model, { type = "q8_0" }) then
		copts = { type = "q8_0" }
	end
	local n_ctx = math.min(opts.n_ctx or model.hp.n_ctx, model.hp.n_ctx)
	if not opts.n_ctx then
		n_ctx = fit_ctx(be, model, n_ctx, copts, n_slots) or n_ctx
		if n_ctx < model.hp.n_ctx then
			L:info("context cut to fit free device memory", "n_ctx", n_ctx, "n_ctx_train", model.hp.n_ctx)
		end
	end
	local adapters = {}
	for i, spec in ipairs(opts.lora or {}) do
		local path, scale = spec.path or spec[1], spec.scale or spec[2] or 1
		adapters[i] = { ad = lora.load(model, path, { scale = scale }), path = path, scale = scale }
	end
	local initial = opts.kv_initial or KV_INITIAL
	if initial > 0 then
		copts.initial = initial
	end
	local auto = opts.ubatch == "auto"
	copts.ubatch = not auto and opts.ubatch or UBATCH
	-- The slots are sequences of one KV pool when the model has one.
	local pool = model.pool and model:pool(n_ctx, copts)
	local function new_cache()
		return pool and model:seq(pool, { n_ctx = n_ctx, ubatch = copts.ubatch }) or model:cache(n_ctx, copts)
	end
	local cache = new_cache()
	local slots = { { cache = cache, tokens = {}, ckpts = {}, used = 0 } }
	for i = 2, n_slots do
		slots[i] = { cache = new_cache(), tokens = {}, ckpts = {}, used = 0 }
	end
	local e = setmetatable({
		slots = slots, slot = slots[1], tick = 0, pending = {}, slot_waiting = {},
		-- With several slots, requests decode together: see E:decode_step.
		concurrent = #slots > 1 and model.batchable ~= nil and model:batchable(),
		be = be, model = model, family = family, family_name = family_name,
		tk = assert(tokenizer.open(opts.model)),
		n_ctx = n_ctx, flash_attn = fa, pool = pool, copts = copts,
		-- Recurrent states and rings need a checkpoint to start from
		-- shared blocks.
		fixed = model.recurrent or (cache.ring ~= nil and cache.ring < n_ctx),
		id = opts.id or info.meta["general.name"] or "lgml",
		sampling = model_sampling(info.meta),
		cache = cache, tokens = {},
		busy = false, waiting = {}, adapters = adapters, acts = {}, queued = 0,
		repeat_stop = opts.repeat_stop_tokens or REPEAT_STOP,
		ubatch = copts.ubatch, ubatch_auto = auto and pool ~= nil,
		reasoning_budget = opts.reasoning_budget_tokens,
		lazy_tool_grammar = opts.lazy_tool_grammar,
	}, E)
	if opts.recipe == "nimble" then
		e.recipe = decide.nimble_recipe()
	elseif opts.recipe == "jevify" then
		e.recipe = decide.default_recipe()
	elseif type(opts.recipe) == "string" then
		e.recipe = assert(decide.load_recipe(opts.recipe))
	else
		e.recipe = opts.recipe or decide.default_recipe()
	end
	local so = opts.speculative
	e.spec_on = so == true or type(so) == "table"
	so = type(so) == "table" and so or {}
	e.spec_opts = { max = so.max or 16, ngram_min = so.ngram_min, ngram_max = so.ngram_max,
	    drafter = so.drafter, adaptive = so.adaptive }
	if not so.drafter and model.mtp then
		e.spec_opts.drafter = function()
			return spec.mtp(model, { ngram_min = so.ngram_min, ngram_max = so.ngram_max, flash_attn = fa,
				max = so.max, min_p = so.min_p })
		end
	end
	e.spec_costs = spec.costs()
	e.eog = {}
	for _, id in ipairs(e.tk:special().eog or {}) do
		e.eog[id] = true
	end
	if pool then
		pool.pressure = function(_, want)
			e:pool_pressure(want)
		end
	end
	e:set_lora()
	if opts.mmproj then
		e:load_vision(opts)
	end
	-- The plans pick the chunk size before the prompt cache takes the
	-- first slot.
	if e.ubatch_auto then
		e:reserve()
	end
	if xopts and not xopts.bytes then
		e:fit_xcache(xopts)
	end
	if model.snapshot then
		local ram = opts.cache_ram or 8192
		e.pc = prompt_cache.new(model, cache, {
			key = e.lora_key,
			ram = ram < 0 and math.huge or math.floor(ram * 1048576),
			checkpoints = opts.ctx_checkpoints or 8,
			special = e.tk:special().tokens,
		})
		-- Blocks that the pool evicts spill to the RAM store.
		if pool and e.pc.ram > 0 then
			pool.alloc.on_evict = function(b, key, info)
				e:spill_block(b, key, info)
			end
		end
	end
	return e
end

--- A chat engine. It implements `lgml.server.engine.Engine`.
-- @type Engine

-- Size the expert cache to the free device memory after the planned
-- growth: the KV cache at n_ctx, the compute buffers and the margin.
-- The plans run with a cache of one slot, so they hold its graph.
function E:fit_xcache(x)
	local r = budget_mod.report(self)
	local free = budget_mod.free(self.be:name()) or 0
	local margin = lgml.memory_guard().margin
	local bytes = free - margin - math.max(0, r.used - r.now) - XCACHE_SLACK
	if x.max then
		bytes = math.min(bytes, x.max)
	end
	self.model.xcache:close()
	collectgarbage()
	collectgarbage()
	-- Under a memory limit the slots may not fit; then take fewer.
	x.slots, x.bytes = nil, math.max(0, bytes)
	while true do
		local ok, err = pcall(new_xcache, self.model, x)
		if ok then
			return
		elseif not budget_mod.is_budget(err) or x.bytes == 0 then
			error(err, 0)
		end
		x.bytes = x.bytes * 3 // 4 >> 20 << 20
	end
end

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

-- The device and CPU compute bytes of a pass of k pool sequences of n
-- tokens each, with its input leaves. The sequences share span cells
-- block by block, so each one reads a window of all span cells. Rings
-- hold ubatch more rows. It plans the graph and runs nothing; the blocks
-- go back to the pool.
function E:plan_pass(k, n, span, ubatch)
	local model, pool = self.model, self.pool
	local each = math.max(n, span // k // pool.block * pool.block)
	local caches, ids = {}, {}
	local ok, dev, cpu = pcall(function()
		for i = 1, k do
			caches[i] = model:seq(pool, { n_ctx = self.n_ctx, ubatch = ubatch })
			ids[i] = {}
			for j = 1, n do
				ids[i][j] = 0
			end
		end
		for cells = pool.block, each, pool.block do
			for _, c in ipairs(caches) do
				model:grow(c, cells)
			end
		end
		for _, c in ipairs(caches) do
			c.n_past = each - n
			if c.rs_pos then
				c.rs_pos = c.n_past
			end
		end
		local inp
		if k == 1 then
			inp = model:inputs(ids[1], caches[1], { logits = "last", flash_attn = self.flash_attn })
		else
			inp = model:batch_inputs(caches, ids, { flash_attn = self.flash_attn })
		end
		local g <close> = model.be:graph()
		local d, h = g:measure((model:forward(g, inp)))
		-- The input leaves, such as the masks, live with the graph.
		for _, part in ipairs({ inp, table.unpack(inp.batch or {}) }) do
			for _, t in pairs(part) do
				if type(t) == "userdata" then
					d = d + t:nbytes()
					t:close()
				end
			end
		end
		return d, h
	end)
	for _, c in ipairs(caches) do
		model:free_seq(c)
	end
	if not ok then
		error(dev, 0)
	end
	return dev, cpu
end

--- The device bytes of the largest compute buffers of the engine.
-- It plans prompt chunks of ubatch tokens over all n_ctx cells: one
-- alone plus, with batched slots, one per slot, as a dropped batch graph
-- lives until the collector takes its rows. With speculation on, the
-- kept graphs and the draft head's passes add `plan_spec`. An image of
-- the most tokens adds its encoder. The pool grows to n_ctx for the plans and shrinks
-- back, so a full cache that does not fit raises the budget error.
-- With ubatch "auto", the first call picks the chunk size; see `new`.
-- @treturn ?integer the bytes, or nil for an engine without a pool
function E:reserve()
	local pool = self.pool
	if not pool then
		return nil
	end
	while true do
		local ok, r = pcall(self.reserve_ctx, self)
		if ok then
			return r
		end
		local n = pool.n_ctx * 3 // 4 // 256 * 256
		if not budget_mod.is_budget(r) or n < MIN_FIT_CTX then
			error(r, 0)
		end
		L:warn("context cut to fit the memory budget", "n_ctx", n, "from", pool.n_ctx)
		self:cut_ctx(n)
	end
end

-- Lower the context of the engine to n cells, a multiple of 256. The
-- pool tensors must hold at most n cells.
function E:cut_ctx(n)
	local pool = self.pool
	pool.n_ctx = n
	pool.alloc.n = n // pool.block
	self.n_ctx = n
	for _, s in ipairs(self.slots) do
		s.cache.n_ctx = math.min(s.cache.n_ctx, n)
	end
end

-- The plans of `reserve` at the context of the engine now.
function E:reserve_ctx()
	local pool, model = self.pool, self.model
	local cells = pool.cap
	local name = self.be:name()
	local before = budget_mod.free(name)
	local list = { self.ubatch }
	if self.ubatch_auto then
		list = {}
		for _, u in ipairs(AUTO_UBATCH) do
			if u <= pool.n_ctx or u == UBATCH then
				list[#list + 1] = u
			end
		end
	end
	local ok, best, pick = pcall(function()
		model:pool_resize(pool, pool.n_ctx)
		for i, u in ipairs(list) do
			local good, r = pcall(self.plan_chunks, self, u)
			if good or i == #list or not budget_mod.is_budget(r) then
				if not good then
					error(r, 0)
				end
				return r + self:plan_spec(), u
			end
		end
	end)
	model:pool_resize(pool, cells)
	-- Later budget checks must see the cells that the shrink freed.
	-- Nodes of the plans still hold the old tensors until a collect.
	collectgarbage()
	collectgarbage()
	if before then
		budget_mod.settle(name, before - (64 << 20))
	end
	if not ok then
		error(best, 0)
	end
	if self.ubatch_auto then
		self:set_ubatch(pick)
	end
	if self.vision and self.vision.measure then
		best = best + self.vision:measure()
	end
	return best
end

-- The compute bytes of prompt chunks of u tokens over the full pool,
-- for `reserve_ctx`. Rings that grow for u count too. It raises the
-- budget error when they do not fit.
function E:plan_chunks(u)
	local pool, model = self.pool, self.model
	local ks, top = { 1 }, pool.n_ctx
	if self.concurrent then
		ks[2] = #self.slots
	end
	local r = 0
	for _, k in ipairs(ks) do
		local n = math.min(u, top // k // pool.block * pool.block)
		if n > 0 then
			r = r + self:plan_pass(k, n, top, u)
		end
	end
	local rings = 0
	if u ~= self.ubatch and model.hp.n_swa then
		local o = setmetatable({ ubatch = u }, { __index = self.copts })
		rings = (select(2, model:cache_size(o)) - select(2, model:cache_size(self.copts))) * #self.slots
	end
	-- The full pool is in place: the compute buffers must fit too.
	local fits, err = lgml.memory_check(self.be, r + rings, "the compute buffers")
	if not fits then
		error(err, 0)
	end
	return r
end

-- The compute bytes that speculation adds to the plan, and the bytes
-- of kept graphs it allows each cache (`self.spec_kept_bytes`). A
-- verify pass runs at most 2 * max + 2 rows with logits for each. The
-- slot keeps graphs up to twice that pass, the draft head's view of
-- the cache as many, and the head's own pass takes one more.
function E:plan_spec()
	if not self.spec_on then
		return 0
	end
	local n = 2 * self.spec_opts.max + 2
	local v = self:plan_pass(1, n, self.pool.n_ctx, self.ubatch) + n * self.tk:n_vocab() * 4
	self.spec_kept_bytes = 2 * v
	L:info("speculation plan", "verify_mib", v // 1048576, "kept_mib", 2 * v // 1048576)
	return 5 * v
end

-- Take u tokens per prompt pass from now on. With rings, the slots get
-- new sequences that hold u more rows; they must be empty.
function E:set_ubatch(u)
	self.ubatch_auto = false
	L:info("prompt chunk size", "ubatch", u, "auto", true)
	if u == self.ubatch then
		return
	end
	self.ubatch, self.copts.ubatch = u, u
	if not self.model.hp.n_swa then
		return
	end
	for _, s in ipairs(self.slots) do
		assert(#s.tokens == 0 and s.cache.n_past == 0, "ubatch: a slot is in use")
		self.model:free_seq(s.cache)
		s.cache = self.model:seq(self.pool, { n_ctx = self.n_ctx, ubatch = u })
	end
	self.cache = self.slot.cache
	self.fixed = self.model.recurrent or (self.cache.ring ~= nil and self.cache.ring < self.n_ctx)
	if self.pc then
		self.pc.cache = self.cache
	end
end

--- The memory plan of the engine, for `lgml.budget.report`.
-- @treturn table `{ backend, cache, cache_now, compute }`: the device
-- bytes of the KV cache at n_ctx cells and now, and the largest compute
-- buffers from `reserve`
function E:memory()
	-- The reserve can cut the context; the cache sizes come after it.
	local r = { backend = self.be, compute = self:reserve() }
	if self.model.cache_size then
		local per, fixed = self.model:cache_size(self.copts)
		local n = #self.slots
		-- A speculating hybrid model keeps a spare copy of one state.
		if self.spec_on and self.model.recurrent then
			n = n + 1
		end
		if self.pool then
			r.cache = per * self.pool.n_ctx + fixed * n
			r.cache_now = per * self.pool.cap + fixed * n
		else
			r.cache, r.cache_now = 0, 0
			for _, s in ipairs(self.slots) do
				r.cache = r.cache + per * s.cache.n_ctx + fixed
				r.cache_now = r.cache_now + per * (s.cache.cap or s.cache.n_ctx) + fixed
			end
		end
	end
	return r
end

--- Lists the one model of the engine.
-- @treturn {table,...} `{ { id, context_length = n_ctx, vision } }`;
-- vision is true with an mmproj
function E:models()
	return { { id = self.id, context_length = self.n_ctx, vision = self.vision ~= nil } }
end

--- Returns the properties: `{ n_ctx, total_slots, lora, modalities }`.
-- lora lists the adapters as llama-server does: `{ id, path, scale }`,
-- with the global scales.
-- modalities is `{ vision = true }` with an mmproj, else `{ vision = false }`.
-- @treturn table the properties
function E:props()
	local list = {}
	for i, a in ipairs(self.adapters) do
		list[i] = { id = i - 1, path = a.path, scale = a.scale }
	end
	return { n_ctx = self.n_ctx, total_slots = #self.slots, lora = list,
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
-- The adapter scales and key for a request's lora list, or nil and an
-- error message.
function E:lora_for(list)
	local scales = {}
	for i, a in ipairs(self.adapters) do
		scales[i] = a.scale
	end
	if list ~= nil then
		local err
		scales, err = engine_api.lora_scales(list, #self.adapters)
		if not scales then
			return nil, err
		end
	end
	return scales, table.concat(scales, ",")
end

function E:set_lora(list)
	local scales, key = self:lora_for(list)
	if not scales then
		return key
	end
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
	self.tokens = {}
	self:clear_cache(self.cache)
	return nil
end

-- Empty a slot's cache: no positions, and no pool blocks.
function E:clear_cache(cache)
	cache.n_past, cache.replay_keep = 0, nil
	if cache.pool then
		self.model:truncate(cache, 0)
	end
	if self.model.rewind then
		self.model:rewind(cache, 0)
	end
end

-- Empty an idle slot so the pool gets its blocks back. With a prompt
-- store, the slot goes there first, so a later request can restore it;
-- the store keeps only slots of the current adapter scales.
function E:evict_slot(s)
	local pc = self.pc
	local active = s == self.slot
	if active then
		s.tokens = self.tokens
		if pc then
			s.ckpts = pc.ckpts
		end
	end
	if s.key == self.lora_key then
		self:publish_tail(s.cache, s.tokens)
	end
	if pc and #s.tokens > 0 and s.key == self.lora_key then
		local cache, ckpts = pc.cache, pc.ckpts
		pc.cache, pc.ckpts = s.cache, s.ckpts
		pc:save({ tokens = s.tokens, prio = s.prio })
		pc.cache, pc.ckpts = cache, ckpts
	end
	if pc then
		pc:drop(s.ckpts)
	end
	L:debug("slot evicted", "index", self:slot_index(s), "tokens", #s.tokens, "blocks", #s.cache.blocks)
	s.tokens = {}
	if active then
		self.tokens = s.tokens
	end
	self:clear_cache(s.cache)
	self.evictions = (self.evictions or 0) + 1
end

-- The pool has no free block for want (a sequence that runs now):
-- empty idle slots, the lowest priority and then the least recently
-- used first, until a block is free. Slots that requests use keep
-- their blocks.
function E:pool_pressure(want)
	local list = {}
	for _, s in ipairs(self.slots) do
		if s.cache ~= want and not s.owner and #s.cache.blocks > 0 then
			list[#list + 1] = s
		end
	end
	table.sort(list, function(a, b)
		local pa, pb = a.prio or 1, b.prio or 1
		return pa < pb or (pa == pb and a.used < b.used)
	end)
	local alloc = self.pool.alloc
	for _, s in ipairs(list) do
		self:evict_slot(s)
		if alloc.used < alloc.cap then
			return
		end
	end
end

-- True when the pool can give the active slot room for n positions:
-- its free blocks and the blocks of idle slots.
function E:room_for(n)
	local cache = self.cache
	if not cache.pool then
		return true
	end
	local B, a = self.pool.block, self.pool.alloc
	local room = a.n - a.used
	for _, s in ipairs(self.slots) do
		if s ~= self.slot and not s.owner then
			room = room + #s.cache.blocks
		end
	end
	return (n + B - 1) // B - #cache.blocks <= room
end

-- Let the device go until a running request ends, then take it again
-- as `chat` does. Returns false without the lock when the sink is
-- cancelled.
function E:wait_room(st, req, sink)
	st.lock = false
	self:release()
	stream.await(function(done)
		self.slot_waiting[#self.slot_waiting + 1] = done
	end)
	local _, key = self:lora_for(req.lora)
	if not self:acquire_slot(key, cancel_of(sink)) then
		return false
	end
	st.lock = true
	self:set_lora(req.lora)
	return true
end

--- Let a running request go and take it up again later.
-- The pool is full, and other running requests hold it. The request
-- keeps a checkpoint of its state (models with recurrent states or
-- rings), lets its slot and cells go, and waits until another request
-- ends. Then it takes a slot again, finds its tokens so far in the pool,
-- the RAM store or the state index, and runs the rest. Its cells stay
-- cached, so it seldom runs much again.
-- @tparam table st the request state
-- @tparam table req the request
-- @tparam table slot the request's slot
-- @tparam {integer,...} rest the tokens that did not run: the sampled
-- token, or the rest of the prompt
-- @param sink the reply sink
-- @treturn table the slot the request owns now
-- @return the logits after the last token, or nil when the sink was
-- cancelled
function E:preempt(st, req, slot, rest, sink)
	local toks = table.move(slot.tokens, 1, #slot.tokens, 1, {})
	table.move(rest, 1, #rest, #toks + 1, toks)
	self:acquire()
	local ok, err = pcall(function()
		if self.pc and self.fixed then
			self:checkpoint_slot(slot)
		end
		self:publish_tail(slot.cache, slot.tokens)
	end)
	L:info("kv pool full; the request waits", "tokens", #toks, "running", self:owned() - 1,
	    "priority", slot.prio)
	self.preemptions = (self.preemptions or 0) + 1
	self.waits = self.waits or {}
	self.waits[slot.prio or 1] = (self.waits[slot.prio or 1] or 0) + 1
	if self.pc then
		if slot == self.slot then
			self.pc:trim(-1)
		else
			self.pc:drop(slot.ckpts)
		end
	end
	slot.tokens = {}
	self:clear_cache(slot.cache)
	if slot == self.slot then
		self.tokens = slot.tokens
	end
	slot.owner, st.slot = nil, nil
	self:release()
	if not ok then
		error(err, 0)
	end
	self:slot_freed()
	stream.await(function(done)
		self.slot_waiting[#self.slot_waiting + 1] = done
	end)
	local _, key = self:lora_for(req.lora)
	if not self:acquire_slot(key, cancel_of(sink)) then
		return nil, nil
	end
	st.lock = true
	self:set_lora(req.lora)
	self.req_prio = prio_of(req)
	local cached = self:reuse(toks)
	L:debug("request resumes", "tokens", #toks, "cached", cached)
	local logits = self:prefill(toks, cached + 1, sink)
	if not logits then
		return nil, nil
	end
	return self:own_slot(st), logits
end

-- Wait for the slot. Requests queue in arrival order. With a cancel
-- function, a cancelled request stops waiting: it passes a wake on to
-- the next request and returns false.
function E:acquire(cancelled)
	while self.busy do
		stream.await(function(done)
			self.waiting[#self.waiting + 1] = done
		end)
		if cancelled and cancelled() then
			if not self.busy then
				self:release()
			end
			return false
		end
	end
	self.busy = true
	return true
end

function E:release()
	self.busy = false
	local w = table.remove(self.waiting, 1)
	if w then
		w()
	end
end

-- The count of slots that decoding requests own.
function E:owned()
	local n = 0
	for _, s in ipairs(self.slots) do
		if s.owner then
			n = n + 1
		end
	end
	return n
end

-- Wake the requests that wait for a slot.
function E:slot_freed()
	local w = self.slot_waiting
	self.slot_waiting = {}
	for _, done in ipairs(w) do
		done()
	end
end

-- Take the device lock with a free slot, and with no other request
-- decoding when the adapter scales must change. Returns true with the
-- lock, or false without it when cancelled() turns true.
function E:acquire_slot(lora_key, cancelled)
	while true do
		if not self:acquire(cancelled) then
			return false
		end
		local owned = self:owned()
		if owned < #self.slots and (owned == 0 or lora_key == self.lora_key) then
			return true
		end
		self:release()
		stream.await(function(done)
			self.slot_waiting[#self.slot_waiting + 1] = done
		end)
		if cancelled and cancelled() then
			return false
		end
	end
end

-- Give up a request's slot and the device lock if it still holds it.
function E:end_request(st)
	if st.lock then
		st.lock = false
		self:release()
	end
	if st.slot then
		-- The next request of the slot starts with a new graph: this one
		-- only holds compute memory.
		replay.drop(st.slot.cache, true)
		st.slot.owner, st.slot.preempt = nil, nil
		st.slot = nil
		self:slot_freed()
	end
end

-- After a prefill: the request owns its slot and lets the device go, so
-- other requests prefill and decode alongside.
function E:own_slot(st)
	local slot = self.slot
	slot.tokens = self.tokens
	slot.owner = st
	st.slot = slot
	st.lock = false
	self:release()
	return slot
end

--- Decode one token for a slot that a request owns.
-- The step waits in a queue; one batched pass runs the queued steps of
-- all slots (`lgml.transformer.decode_batch`), then each request gets
-- its row of logits.
-- @tparam table slot the slot
-- @tparam integer id the token id
-- @return the logits node of this slot, valid until the next batch pass
function E:decode_step(slot, id)
	return self:batch_step(slot, { id })
end

--- Run tokens of a slot that a request owns in the next batched pass:
-- one decode token, or a chunk of a prompt.
-- @tparam table slot the slot
-- @tparam {integer,...} ids the token ids
-- @return the logits node of the slot's last token, valid until the next
-- batch pass
function E:batch_step(slot, ids)
	-- A request of higher priority needs the cells of this one.
	if slot.preempt then
		slot.preempt = nil
		error("kv pool full: a request of higher priority needs the cells", 0)
	end
	-- Alone: run at once, with no trip through the event loop.
	if #self.pending == 0 and not self.busy and self:owned() == 1 then
		self:acquire()
		-- One graph at a time holds compute memory: the batch one goes.
		if self.model.drop_batch then
			self.model:drop_batch(false)
		end
		local ok, logits = pcall(self.run_pass, self, slot.cache, ids,
		    { logits = "last", flash_attn = self.flash_attn })
		self:release()
		if not ok then
			error(logits, 0)
		end
		return logits
	end
	local entry = { slot = slot, ids = ids }
	self.pending[#self.pending + 1] = entry
	if not self.flush_scheduled then
		self.flush_scheduled = true
		local t = uv.new_timer()
		t:start(0, 0, function()
			t:close()
			coroutine.wrap(function()
				self:flush()
			end)()
		end)
	end
	stream.await(function(done)
		entry.done = done
	end)
	if entry.retry then
		-- A request of lower priority lets its cells go; then run again.
		stream.await(function(done)
			self.slot_waiting[#self.slot_waiting + 1] = done
		end)
		return self:batch_step(slot, ids)
	end
	if entry.err then
		error(entry.err, 0)
	end
	return entry.logits
end

-- A running request of lower priority than the one of slot, whose
-- cells can go: the lowest priority, the least recently started first.
function E:victim(slot)
	local best
	for _, s in ipairs(self.slots) do
		if s.owner and s ~= slot and not s.preempt and (s.prio or 1) < (slot.prio or 1) and
		    (not best or (s.prio or 1) < (best.prio or 1) or
		    ((s.prio or 1) == (best.prio or 1) and s.used < best.used)) then
			best = s
		end
	end
	return best
end

-- Run the queued decode steps as one batched pass.
function E:flush()
	self:acquire()
	self.flush_scheduled = false
	local list = self.pending
	self.pending = {}
	if #list == 0 then
		self:release()
		return
	end
	-- Take the pool blocks of each step first: a step that gets none
	-- fails alone. When a running request of lower priority can let its
	-- cells go, it does, and the step runs again after that.
	local run = {}
	for _, e in ipairs(list) do
		local c = e.slot.cache
		local good, err = pcall(self.model.grow, self.model, c, math.min(c.n_ctx, c.n_past + #e.ids))
		if good then
			run[#run + 1] = e
		else
			e.err = err
			local v = tostring(err):find("kv pool full", 1, true) and self:victim(e.slot)
			if v then
				v.preempt, e.retry = true, true
			end
		end
	end
	local caches, ids, kept = {}, {}, {}
	for _, e in ipairs(run) do
		if e.slot.preempt then
			e.slot.preempt = nil
			e.err = "kv pool full: a request of higher priority needs the cells"
		else
			kept[#kept + 1] = e
			caches[#kept], ids[#kept] = e.slot.cache, e.ids
		end
	end
	run = kept
	local all = list
	list = run
	local ok, rows = true, {}
	if #list == 1 and self.model.drop_batch then
		self.model:drop_batch(false)
	elseif #list > 1 then
		-- The requests of these slots read their last logits already.
		for _, c in ipairs(caches) do
			replay.drop(c, true)
		end
	end
	if #list == 1 then
		-- One request: its slot's own replayed graph is the fast path.
		ok, rows = pcall(function()
			return { self:run_pass(caches[1], ids[1], { logits = "last", flash_attn = self.flash_attn }) }
		end)
	elseif #list > 1 then
		ok, rows = pcall(self.model.run_batch, self.model, caches, ids,
		    { flash_attn = self.flash_attn })
		if not ok and budget_mod.is_budget(rows) then
			L:warn("a batch pass does not fit the memory budget; graphs go and it runs again",
			    "err", rows)
			self:relieve()
			ok, rows = pcall(self.model.run_batch, self.model, caches, ids,
			    { flash_attn = self.flash_attn })
		end
	end
	self:release()
	self.batch_passes = (self.batch_passes or 0) + 1
	self.batch_rows = (self.batch_rows or 0) + #list
	for _, e in ipairs(list) do
		self.batch_tokens = (self.batch_tokens or 0) + #e.ids
	end
	if L:on(log.TRACE) then
		L:trace("batch", "slots", #list)
	end
	for i, e in ipairs(list) do
		if ok then
			e.logits = rows[i]
		else
			e.err = rows
		end
	end
	for _, e in ipairs(all) do
		e.done()
	end
end

local function thinking(req)
	local k = req.chat_template_kwargs
	if type(k) == "table" and k.enable_thinking ~= nil then
		return k.enable_thinking ~= false
	end
	return req.reasoning_effort ~= "none"
end

-- The sampler options of a request over the preset: the model's gguf
-- values, then the request values, added to extra.
local function sampler_opts_table(req, model, extra)
	local o = extra or {}
	for k, v in pairs(model or {}) do
		o[k] = v
	end
	for _, k in ipairs({ "temperature", "top_p", "top_k", "min_p", "seed",
	    "presence_penalty", "frequency_penalty", "repeat_penalty" }) do
		if type(req[k]) == "number" then
			o[k] = req[k]
		end
	end
	return o
end

local function sampler_opts(req, think, extra, model)
	return sampler.new(think and "qwen3_think" or "qwen3", sampler_opts_table(req, model, extra))
end

-- Free memory that later passes can make again, for a pass on cache
-- that did not fit: the kept graphs of each slot, the graphs of idle
-- slots, the batch graph when no other request runs, and the outputs
-- that only the garbage collector still holds.
function E:relieve(cache)
	local alone = self:owned() <= 1
	for _, s in ipairs(self.slots) do
		replay.drop_kept(s.cache)
		if s.cache ~= cache and not s.owner then
			replay.drop(s.cache, true)
		end
	end
	if alone and self.model.drop_batch then
		self.model:drop_batch(false)
	end
	collectgarbage()
	collectgarbage()
	self.relieved = (self.relieved or 0) + 1
end

-- The smallest prompt chunk that a pass past the memory budget splits.
local MIN_SPLIT = 64

-- replay.run with one more try when the pass does not fit the memory
-- budget: `relieve` frees what it can first. A prompt chunk that still
-- does not fit runs in two halves; the result is that of the last.
function E:run_pass(cache, ids, opts)
	local ok, out = pcall(replay.run, self.model, cache, ids, opts)
	if ok or not budget_mod.is_budget(out) then
		if not ok then
			error(out, 0)
		end
		return out
	end
	L:warn("a pass does not fit the memory budget; graphs go and it runs again", "tokens", #ids,
	    "err", out)
	self:relieve(cache)
	ok, out = pcall(replay.run, self.model, cache, ids, opts)
	-- Smaller chunks need the same cells, so a cache growth stays an error.
	local cells = not ok and (tostring(out):find("the KV pool from", 1, true) or
	    tostring(out):find("the KV cache from", 1, true))
	if ok or not budget_mod.is_budget(out) or cells or #ids < 2 * MIN_SPLIT or opts.embd or
	    opts.logits == "all" then
		if not ok then
			error(out, 0)
		end
		return out
	end
	local half = #ids // 2
	L:warn("a prompt chunk does not fit the memory budget; it runs in halves", "tokens", #ids)
	self:run_pass(cache, table.move(ids, 1, half, 1, {}), opts)
	return self:run_pass(cache, table.move(ids, half + 1, #ids, 1, {}), opts)
end

-- Run ids at cache.n_past. Returns the logits node, which stays valid
-- until the next pass: the cache keeps the graph for replay. rows is
-- "last" (the default) or "all". embd holds embeddings for the ids: a
-- non-causal pass reads them in place of the token embeddings.
function E:eval(ids, rows, embd)
	return self:run_pass(self.cache, ids, {
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

--- Make the slot for a prompt the active one.
-- The slot with the longest shared prefix wins. When no slot shares
-- at least `SLOT_MIN_SHARED` tokens, an empty slot or the least recently
-- used one is taken. The active slot's cache and tokens are `self.cache`
-- and `self.tokens`; the prompt cache follows the active slot. A slot
-- made under other adapter scales starts empty.
-- @tparam {integer,...} prompt the token ids
-- @return the slot
function E:pick_slot(prompt)
	self.tick = self.tick + 1
	local cur = self.slot
	cur.tokens = self.tokens
	if self.pc then
		cur.ckpts = self.pc.ckpts
	end
	local best, best_n = cur, -1
	if #self.slots > 1 then
		best = nil
		for _, s in ipairs(self.slots) do
			local t, n = s == cur and self.tokens or s.tokens, 0
			while not s.owner and n < #t and n < #prompt and t[n + 1] == prompt[n + 1] do
				n = n + 1
			end
			if not s.owner and (n > best_n or (n == best_n and s == cur)) then
				best, best_n = s, n
			end
		end
		if best_n < math.min(SLOT_MIN_SHARED, #prompt) then
			-- An empty slot, else the one of the lowest priority, the
			-- least recently used first.
			best = nil
			for _, s in ipairs(self.slots) do
				if not s.owner and #s.tokens == 0 then
					best = s
					break
				end
				if not s.owner and (not best or (s.prio or 1) < (best.prio or 1) or
				    ((s.prio or 1) == (best.prio or 1) and s.used < best.used)) then
					best = s
				end
			end
		end
	end
	if best ~= cur then
		self.slot, self.cache, self.tokens = best, best.cache, best.tokens
		if self.pc then
			self.pc.cache, self.pc.ckpts = best.cache, best.ckpts
		end
		L:debug("slot", "index", self:slot_index(best), "shared", math.max(best_n, 0))
	end
	if best.key ~= self.lora_key then
		if self.pc then
			self.pc:trim(-1)
		end
		self.tokens, best.key = {}, self.lora_key
		self:clear_cache(self.cache)
	end
	best.used = self.tick
	-- The old content keeps its priority when it goes to the RAM store
	-- (see `lgml.prompt_cache.save`); the new one has the request's.
	self.prio = best.prio
	best.prio = self.req_prio or kvpool.NORMAL
	best.cache.prio = best.prio
	return best
end


-- The 1-based index of a slot.
function E:slot_index(slot)
	for i, s in ipairs(self.slots) do
		if s == slot then
			return i
		end
	end
end

-- Keep the longest shared prefix of the cached tokens. At least one
-- prompt token must run to get logits. An image runs whole or not at
-- all.
function E:reuse(prompt)
	self:pick_slot(prompt)
	local n = 0
	local old = self.tokens
	while n < #old and n < #prompt - 1 and old[n + 1] == prompt[n + 1] do
		n = n + 1
	end
	n = span_start(prompt, n)
	-- Cached pool blocks can give a longer prefix than the slot.
	n = span_start(prompt, self:attach(prompt, n))
	old = self.tokens
	-- The prompt cache can swap in a stored slot that shares more.
	if self.pc then
		n = span_start(prompt, self.pc:reuse(self, prompt, n))
		old = self.tokens
		-- Tokens from host memory: a checkpoint, or spilled blocks.
		if self.pc.restored == 0 then
			self.pc.restored = self.state_n or math.min(n, self.from_ram * self.pool.block)
		end
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
	if self.cache.pool then
		self.model:truncate(self.cache, n)
	end
	for i = #old, n + 1, -1 do
		old[i] = nil
	end
	if self.pc then
		self.pc:trim(n)
	end
	self:publish(self.cache, self.tokens)
	return n
end

-- The key before the first block of a sequence: blocks made under
-- other adapter scales do not match.
function E:root_key()
	return "lgml:" .. (self.lora_key or "")
end

-- Give keys to the full blocks of a pool sequence that have none, so
-- other sequences can share them. tokens are the sequence's tokens.
function E:publish(cache, tokens)
	if not cache.pool then
		return
	end
	local B, alloc, keys = cache.pool.block, cache.pool.alloc, cache.keys
	for i = #keys + 1, math.min(cache.n_past, #tokens) // B do
		local parent = keys[i - 1] or self:root_key()
		local key = kvpool.key(parent, tokens, (i - 1) * B + 1, B)
		keys[i] = key
		alloc:publish(cache.blocks[i], key, parent, i, table.move(tokens, (i - 1) * B + 1, i * B, 1, {}),
		    cache.prio)
	end
end

-- Copy the cells of a keyed block that the pool evicts to the RAM
-- store of the prompt cache.
function E:spill_block(b, key, info)
	local pool = self.pool
	self.pc:spill(key, info, transformer.block_bytes(pool), function(buf)
		transformer.copy_block_host(pool, b, buf, true)
	end)
end

-- Put a spilled block back in the pool for the active slot, after
-- block last. Returns the block, with one reference, or nil.
function E:unspill(key, last)
	local e = self.pc and self.pc:unspill(key)
	if not e then
		return nil
	end
	local ok, b = pcall(self.model.take_block, self.model, self.pool, self.cache, last and last + 1)
	if ok then
		transformer.copy_block_host(self.pool, b, e.buf, false)
		self.pool.alloc:publish(b, key, e.parent, e.depth, e.tokens,
		    math.max(e.prio or kvpool.NORMAL, self.cache.prio or kvpool.NORMAL))
		self.from_ram = (self.from_ram or 0) + 1
	end
	e.buf:close()
	return ok and b or nil
end

-- The cached blocks that hold the first full blocks of prompt, at most
-- max tokens, and their keys. Spilled blocks come back from the RAM
-- store. Each block holds one more reference until `drop_held`.
function E:cached_blocks(prompt, max)
	local B, alloc = self.pool.block, self.pool.alloc
	local blocks, keys = {}, {}
	local parent = self:root_key()
	while (#blocks + 1) * B <= max do
		local key = kvpool.key(parent, prompt, #blocks * B + 1, B)
		local b = alloc:lookup(key)
		if b then
			alloc:ref(b, self.cache.prio)
		else
			b = self:unspill(key, blocks[#blocks])
			if not b then
				break
			end
		end
		blocks[#blocks + 1], keys[#keys + 1] = b, key
		parent = key
	end
	return blocks, keys
end

-- Drop the references that `cached_blocks` holds.
function E:drop_held(blocks)
	for _, b in ipairs(blocks) do
		self.pool.alloc:unref(b)
	end
end

-- Keep the partial last block of a pool sequence cached when the
-- sequence lets it go, so a prompt that shares its tokens can copy it.
-- Only models with recurrent states or rings keep it: a checkpoint
-- there needs its cells. For other models it would crowd the pool with
-- the ends of one-off requests.
function E:publish_tail(cache, tokens)
	if not cache.pool or not self.fixed then
		return
	end
	self:publish(cache, tokens)
	local B = cache.pool.block
	local n = math.min(cache.n_past, #tokens)
	local i = n // B + 1
	if n % B ~= 0 and cache.blocks[i] and #cache.keys == i - 1 then
		cache.pool.alloc:publish_part(cache.blocks[i], cache.keys[i - 1] or self:root_key(), i,
		    table.move(tokens, (i - 1) * B + 1, n, 1, {}), cache.prio)
	end
end

-- Put the newest checkpoint of a slot in the state index of the prompt
-- cache, under the key of the pool block that ends at or before it.
function E:share_state(cache, tokens, ckpts)
	local ck = ckpts[#ckpts]
	if not cache.pool or not ck or ck.n ~= cache.n_past or ck.n > #tokens then
		return
	end
	self:publish(cache, tokens)
	local B = cache.pool.block
	local j = ck.n // B
	local parent = j == 0 and self:root_key() or cache.keys[j]
	if parent then
		self.pc:add_state(parent, table.move(tokens, j * B + 1, ck.n, 1, {}), ck, cache.prio)
	end
end

-- Start the active slot of a model with recurrent states or rings from
-- cached blocks and a checkpoint of the state index, when that gives
-- more than keep tokens. The cells from the last block edge to the
-- checkpoint come from a cached block that shares them. Returns the
-- tokens the slot holds, or nil. With save, the slot goes to the RAM
-- store before it starts over.
function E:attach_state(prompt, keep, save)
	if not self.pc or #self.pc.states == 0 then
		return nil
	end
	local blocks, keys = self:cached_blocks(prompt, #prompt - 1)
	local ok, n = pcall(self.attach_edge, self, prompt, keep, save, blocks, keys)
	self:drop_held(blocks)
	if not ok then
		error(n, 0)
	end
	return n
end

-- The search of `attach_state` over blocks, which the caller holds.
function E:attach_edge(prompt, keep, save, blocks, keys)
	local pc, cache = self.pc, self.cache
	local B, alloc = self.pool.block, self.pool.alloc
	local max = #prompt - 1
	for j = #blocks, 0, -1 do
		local parent = j == 0 and self:root_key() or keys[j]
		for _, s in ipairs(pc:find_states(parent, prompt, j * B + 1, max)) do
			if s.n <= keep then
				self.from_ram = 0
				return nil
			end
			local part, kid, got = s.n - j * B, nil, 0
			if part > 0 then
				kid, got = alloc:longest_kid(parent, prompt, j * B + 1, part)
			end
			if got >= part then
				if save then
					pc:save(self)
				end
				self.model:truncate(cache, 0)
				self.model:share(cache, table.move(blocks, 1, j, 1, {}), table.move(keys, 1, j, 1, {}))
				if part > 0 then
					alloc:ref(kid)
					local ok, err = pcall(self.model.grow, self.model, cache, s.n)
					if ok then
						self.model.copy_block(cache.pool, kid, cache.blocks[j + 1], part)
					end
					alloc:unref(kid)
					if not ok then
						error(err, 0)
					end
				end
				self.model:restore(cache, s.ck)
				pc:adopt(s)
				self.tokens = table.move(prompt, 1, s.n, 1, {})
				self.slot.tokens = self.tokens
				self.state_n = s.n
				alloc.stats.hits = alloc.stats.hits + 1
				alloc.stats.hit_blocks = alloc.stats.hit_blocks + j
				L:debug("shared blocks and state", "blocks", j, "copied", part, "state", s.n, "slot_kept", keep)
				return s.n
			end
		end
	end
	self.from_ram = 0
	return nil
end

-- Share cached pool blocks that hold a longer prefix of prompt than the
-- n tokens the active slot keeps. The slot then starts over on them. A
-- cached block that shares only part of its tokens with the prompt
-- gives a copy of those cells. A model with recurrent states or rings
-- also needs a checkpoint from the state index; see `attach_state`.
-- Returns the tokens the slot holds.
function E:attach(prompt, n)
	local cache = self.cache
	self.state_n, self.from_ram = nil, 0
	if not cache.pool then
		return n
	end
	-- The slot's last partial block may go now; keep it cached.
	if n < #self.tokens then
		self:publish_tail(cache, self.tokens)
	end
	if self.fixed then
		local keep = self.pc and self.pc:slot_keep(self, n) or 0
		local save = self.pc and keep * 2 < #self.tokens
		return self:attach_state(prompt, keep, save) or n
	end
	local B, alloc = cache.pool.block, cache.pool.alloc
	-- At least one prompt token must run.
	local max = #prompt - 1
	local blocks, keys = self:cached_blocks(prompt, max)
	local ok, got = pcall(self.attach_blocks, self, prompt, n, blocks, keys)
	self:drop_held(blocks)
	if not ok then
		error(got, 0)
	end
	return got
end

-- The rest of `attach` over blocks, which the caller holds.
function E:attach_blocks(prompt, n, blocks, keys)
	local cache = self.cache
	local B, alloc = cache.pool.block, cache.pool.alloc
	local max = #prompt - 1
	local m = #blocks * B
	local kid, part = alloc:longest_kid(keys[#keys] or self:root_key(), prompt, m + 1,
	    math.min(B - 1, max - m))
	if m + part <= n then
		self.from_ram = 0
		return n
	end
	-- A slot that keeps less than half its tokens goes to the RAM store
	-- first, as in `lgml.prompt_cache.reuse`.
	if self.pc and n * 2 < #self.tokens then
		self.pc:save(self)
	end
	self.model:truncate(cache, 0)
	self.model:share(cache, blocks, keys)
	if part > 0 then
		-- Hold the block while the copy takes a new one.
		alloc:ref(kid)
		local ok, err = pcall(self.model.grow, self.model, cache, m + part)
		if ok then
			self.model.copy_block(cache.pool, kid, cache.blocks[#blocks + 1], part)
			cache.n_past = m + part
		else
			part = 0
		end
		alloc:unref(kid)
		if not ok then
			L:debug("no block for a partial copy", "err", err)
		end
	end
	self.tokens = table.move(prompt, 1, m + part, 1, {})
	self.slot.tokens = self.tokens
	alloc.stats.hits = alloc.stats.hits + 1
	alloc.stats.hit_blocks = alloc.stats.hit_blocks + #blocks
	L:debug("shared blocks", "blocks", #blocks, "copied", part, "slot_kept", n)
	return m + part
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
-- Take a prompt cache checkpoint of a slot that is not the active one:
-- the prompt cache points at the slot for the call. Needs the lock.
function E:checkpoint_slot(slot)
	local pc = self.pc
	if slot == self.slot then
		pc:checkpoint()
		self:share_state(slot.cache, slot.tokens, pc.ckpts)
		return
	end
	local cache, ckpts = pc.cache, pc.ckpts
	pc.cache, pc.ckpts = slot.cache, slot.ckpts
	pc:checkpoint()
	slot.ckpts = pc.ckpts
	pc.cache, pc.ckpts = cache, ckpts
	self:share_state(slot.cache, slot.tokens, slot.ckpts)
end

-- The most temporary sequences that `fork_rows` runs at once.
local FORKS = 4

--- Run several prompts that start with the first n tokens of the active
-- slot, in batched passes. Each prompt gets a temporary pool sequence
-- on the slot's cells of those n tokens. A model with fixed state
-- also needs ck, a checkpoint of the slot at n. Needs the lock.
-- @tparam integer n the shared tokens
-- @tparam {{integer,...},...} prompts the prompts
-- @param[opt] ck the checkpoint at n
-- @param sink the sink; a cancelled sink stops the passes
-- @return the logits of the last token of each prompt, as tables, or
-- nil when cancelled
function E:fork_rows(n, prompts, ck, sink)
	local cache, model = self.cache, self.model
	assert(cache.pool and n <= cache.n_past, "fork_rows: the slot does not hold the shared tokens")
	assert(not self.fixed or (ck and ck.n == n), "fork_rows: a checkpoint at n is needed")
	self:publish(cache, self.tokens)
	local B = cache.pool.block
	local j, part = n // B, n % B
	local rows, seqs = {}, {}
	local ok, err = pcall(function()
		for first = 1, #prompts, FORKS do
			local last = math.min(#prompts, first + FORKS - 1)
			for i = first, last do
				local c = model:seq(cache.pool, { n_ctx = cache.n_ctx, ubatch = UBATCH })
				c.prio = kvpool.LOW
				seqs[i] = c
				model:share(c, table.move(cache.blocks, 1, j, 1, {}), table.move(cache.keys, 1, j, 1, {}))
				if part > 0 then
					model:grow(c, n)
					model.copy_block(cache.pool, cache.blocks[j + 1], c.blocks[j + 1], part)
				end
				c.n_past = n
				if ck then
					model:restore(c, ck)
				end
			end
			local at = {}
			while true do
				local cs, ids, who = {}, {}, {}
				for i = first, last do
					local p, k = prompts[i], at[i] or n
					if k < #p then
						who[#who + 1], cs[#cs + 1] = i, seqs[i]
						ids[#ids + 1] = table.move(p, k + 1, math.min(#p, k + UBATCH), 1, {})
					end
				end
				if #cs == 0 then
					break
				end
				local out = model:run_batch(cs, ids, { flash_attn = self.flash_attn })
				for x, i in ipairs(who) do
					at[i] = (at[i] or n) + #ids[x]
					if at[i] == #prompts[i] then
						rows[i] = out[x]:get()
					end
				end
				stream.sleep(0)
				if sink:cancelled() then
					return
				end
			end
			for i = first, last do
				model:free_seq(seqs[i])
				seqs[i] = nil
			end
		end
	end)
	for _, c in pairs(seqs) do
		model:free_seq(c)
	end
	if not ok then
		error(err, 0)
	end
	if #rows < #prompts then
		return nil
	end
	return rows
end

--- Prefill a prompt on a slot that a request owns, in batched passes.
-- Chunks of at most ubatch tokens join the decode steps and prompt
-- chunks of other requests in one pass. Checkpoints come at the prompt
-- cache's planned stops, as in `prefill`. When other running requests
-- hold the pool, the request waits for them; see `preempt`.
-- @tparam table st the request state
-- @tparam table req the request
-- @tparam table slot the slot
-- @tparam {integer,...} prompt the token ids
-- @tparam integer from the first token to run
-- @tparam {integer,...} stops the checkpoint positions, from the plan
-- @param sink the reply sink; a cancelled sink stops the prefill
-- @return the logits node of the last token, or nil when cancelled
-- @treturn table the slot the request owns at the end
function E:prefill_batched(st, req, slot, prompt, from, stops, sink)
	if sink:cancelled() then
		return nil, slot
	end
	local logits
	local i, c = from, 1
	self:note(sink, slot, "prompt", from - 1, #prompt)
	while i <= #prompt do
		local j = math.min(#prompt, i + self.ubatch - 1)
		if stops[c] and stops[c] <= j then
			j = stops[c]
		end
		local ids = table.move(prompt, i, j, 1, {})
		local ok, lg = pcall(self.batch_step, self, slot, ids)
		if not ok then
			if not tostring(lg):find("kv pool full", 1, true) or self:owned() < 2 then
				error(lg, 0)
			end
			local s, l = self:preempt(st, req, slot, table.move(prompt, i, #prompt, 1, {}), sink)
			return l, s
		end
		logits = lg
		table.move(prompt, i, j, #slot.tokens + 1, slot.tokens)
		self:publish(slot.cache, slot.tokens)
		self:note(sink, slot, "prompt", j, #prompt)
		if j == stops[c] then
			self:acquire()
			local good, err = pcall(self.checkpoint_slot, self, slot)
			self:release()
			if not good then
				error(err, 0)
			end
			c = c + 1
		end
		i = j + 1
		if sink:cancelled() then
			return nil, slot
		end
	end
	return logits, slot
end

function E:prefill(prompt, from, sink, extra)
	if sink:cancelled() then
		return nil
	end
	local logits
	local i = from
	-- Passes end at checkpoint positions.
	local stops, c = self.pc and self.pc:plan(prompt, from) or {}, 1
	-- A caller can ask for more, such as the end of a state that several
	-- prompts share. Only models with fixed state need them.
	if extra and self.pc and self.pc.fixed then
		for _, k in ipairs(extra) do
			if k >= from and k < #prompt then
				stops[#stops + 1] = k
			end
		end
		table.sort(stops)
		for k = #stops, 2, -1 do
			if stops[k] == stops[k - 1] then
				table.remove(stops, k)
			end
		end
	end
	if self.pc then
		self:share_state(self.cache, self.tokens, self.pc.ckpts)
	end
	self:note(sink, self.slot, "prompt", from - 1, #prompt)
	while i <= #prompt do
		local j = math.min(#prompt, i + self.ubatch - 1)
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
		self:publish(self.cache, self.tokens)
		self:note(sink, self.slot, "prompt", j, #prompt)
		if j == stops[c] then
			self.pc:checkpoint()
			self:share_state(self.cache, self.tokens, self.pc.ckpts)
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
-- `max_completion_tokens`, `stop` (see `lgml.chat.output`), `logprobs`
-- and `top_logprobs` (from the C `sampler:logprobs`), `logit_bias`, `tools`,
-- `tool_choice` (`"required"` or a named function forces a call with
-- the tool grammar of the chat family), `parallel_tool_calls`,
-- `reasoning_effort = "none"` and
-- `chat_template_kwargs` `enable_thinking`, `reasoning_effort` and
-- `preserve_thinking`. `grammar` (GBNF) or `response_format` (see
-- `lgml.grammar.from_request`) constrain the answer; a reasoning block
-- before it stays free. A bad grammar returns status 400. A Lua error
-- resets the cache and returns status 500. `reasoning_budget` (default:
-- the `reasoning_budget_tokens` of the engine) ends the reasoning block
-- after that many reasoning tokens: the engine adds a short wrap-up
-- line and the end tag of the block, then the model writes the answer.
-- These tokens count as reasoning. `count_only`
-- renders and tokenizes the prompt and ends with its size, with no
-- prefill. The sink gets `prompt` before the prefill, and the finish
-- usage has `stop_sequence` when a stop string ended the reply.
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
	local st = { lock = true }
	local got
	self.queued = self.queued + 1
	if self.concurrent then
		local scales, key = self:lora_for(req.lora)
		if not scales then
			self.queued = self.queued - 1
			return nil, key, 400
		end
		got = self:acquire_slot(key, cancel_of(sink))
	else
		got = self:acquire(cancel_of(sink))
	end
	self.queued = self.queued - 1
	if not got then
		return true
	end
	local lerr = self:set_lora(req.lora)
	if lerr then
		self:end_request(st)
		return nil, lerr, 400
	end
	self.req_prio = prio_of(req)
	local ok, a, b, c = pcall(self.run, self, req, sink, st)
	if not ok then
		-- The cache may not match the token list now.
		local slot = st.slot
		if slot then
			slot.tokens = {}
			self:clear_cache(slot.cache)
			if slot == self.slot then
				self.tokens = slot.tokens
			end
		else
			self.tokens = {}
			self:clear_cache(self.cache)
		end
	end
	self:end_request(st)
	if not ok then
		return nil, tostring(a), budget_mod.status(a, 500)
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
	local gone = ctl and ctl.cancelled
	local got
	if self.concurrent then
		local scales, key = self:lora_for(req.lora)
		if not scales then
			return nil, key, 400
		end
		got = self:acquire_slot(key, gone)
	else
		got = self:acquire(gone)
	end
	if not got then
		return nil, "cancelled", 499
	end
	local lerr = self:set_lora(req.lora)
	if lerr then
		self:release()
		return nil, lerr, 400
	end
	self.req_prio = prio_of(req, true)
	self.image_ms = 0
	local ok, a, b, c = pcall(decide.run, self, req, ctl)
	self:drop_images()
	if not ok then
		self.tokens = {}
		self:clear_cache(self.cache)
	end
	self:release()
	if not ok then
		return nil, tostring(a), budget_mod.status(a, 500)
	end
	return a, b, c
end

-- The tools that req forces a call to, and whether it allows more than
-- one call: all tools for tool_choice "required", one for a named
-- function. nil when the model may answer without a call.
local function forced_tools(req)
	local tc, tools = req.tool_choice, req.tools or {}
	if tc == "required" then
		return tools, req.parallel_tool_calls ~= false
	elseif type(tc) == "table" then
		local name = type(tc["function"]) == "table" and tc["function"].name
		for _, t in ipairs(tools) do
			if type(t["function"]) == "table" and t["function"].name == name then
				return { t }, false
			end
		end
	end
	return nil
end

-- Sampler options for the grammar of req: {} when it has none, or nil
-- and an error. A forced tool call takes the tool call grammar of the
-- chat family. With thinking on, the grammar applies after the block.
-- With tools and tool_choice "auto", and no grammar or response format,
-- the tool call grammar is lazy: it applies from a call opener to the
-- end of the call, outside the reasoning block.
function E:grammar_opts(req, think, prompt)
	local text, err, root
	local forced, parallel = forced_tools(req)
	local tags = THINK[self.family_name]
	local opened = think and tags and prompt:gsub("%s+$", ""):sub(-#tags[1]) == tags[1]
	if forced then
		if not self.family.tool_grammar then
			return nil, ("tool_choice: the %s chat template cannot force a tool call"):format(self.family_name)
		end
		text, root = self.family.tool_grammar(forced, { parallel = parallel })
		if not text then
			return nil, "tool_choice: " .. root
		end
	else
		text, err = grammar.from_request(req)
		if err then
			return nil, err
		elseif not text then
			return self:lazy_tool_opts(req, think and tags, opened)
		end
		root = "root"
	end
	if think and tags then
		text, root = grammar.reasoning(text, tags[1], tags[2], opened, root)
	end
	return { grammar = text, grammar_root = root, tokenizer = self.tk }
end

-- Sampler options for the lazy tool call grammar of req, or {} when it
-- has none. tags, when set, are the reasoning block tags; opened tells
-- that the prompt opened the block.
function E:lazy_tool_opts(req, tags, opened)
	local tools, tc = req.tools, req.tool_choice
	if not self.family.tool_grammar or self.lazy_tool_grammar == false or type(tools) ~= "table" or
	    #tools == 0 or (tc ~= nil and tc ~= "auto") then
		return {}
	end
	local text, root, triggers = self.family.tool_grammar(tools, { lazy = true })
	if not text then
		L:warn("no tool call grammar", "error", root)
		return {}
	end
	return { grammar = text, grammar_root = root, triggers = triggers, tokenizer = self.tk,
	    free_block = tags and { tags[1], tags[2], inside = opened } or nil }
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

-- The sampler form of a request logit_bias: token ids as integers.
-- Returns nil and an error message for an id that is not in the vocab.
function E:logit_bias(lb)
	if lb == nil then
		return nil
	end
	local out, nv = {}, self.tk:n_vocab()
	for k, v in pairs(lb) do
		local id = math.tointeger(tonumber(k))
		if not id or id < 0 or id >= nv then
			return nil, ("logit_bias: %s is not a token id of the model (0 to %d)"):format(tostring(k), nv - 1)
		end
		out[id] = v
	end
	return out
end

-- A logprobs entry of token id with log probability v.
local function lp_entry(tk, id, v)
	local piece = tk:token(id)
	return { token = tokenizer.clean(piece), logprob = math.max(v, -9999), bytes = { piece:byte(1, -1) } }
end

-- The logprobs entry of the token id that smp just sampled, with the k
-- most likely tokens.
function E:logprobs(smp, id, k)
	local v, top = smp:logprobs(id, k)
	local e = lp_entry(self.tk, id, v)
	e.top_logprobs = {}
	for i, t in ipairs(top) do
		e.top_logprobs[i] = lp_entry(self.tk, t[1], t[2])
	end
	return e
end

-- The first lines of a grammar, for logs.
-- A function of each sampled token id that returns true at the nth
-- same id in a row. nil when n is 0 or nil.
local function repeat_guard(n)
	if not n or n <= 0 then
		return nil
	end
	local last, run = nil, 0
	return function(id)
		run = id == last and run + 1 or 1
		last = id
		return run >= n
	end
end

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

--- The work of each slot, for `GET /status`.
-- See `lgml.server.engine.Engine.status`.
-- @treturn {table,...} one entry: `{ id, status, slots, queued }`
function E:status()
	local slots = {}
	for i = 1, #self.slots do
		slots[i] = { id = i - 1, state = "idle" }
	end
	for _, a in pairs(self.acts) do
		if a.slot and slots[a.slot] then
			slots[a.slot] = engine_api.slot_status(a)
		end
	end
	return { { id = self.id, status = "loaded", slots = slots, queued = self.queued } }
end

-- Note the progress of the request of sink on slot: state "prompt" with
-- done of total prompt tokens, or "answer" with done tokens made. A
-- sink with `progress` gets the record at each change of state or slot,
-- and at most once in PROGRESS_MS else.
function E:note(sink, slot, state, done, total)
	local a = self.acts[sink]
	if not a then
		return
	end
	local i = self:slot_index(slot)
	local changed = a.state ~= state or a.slot ~= i
	a.slot, a.state = i, state
	if state == "prompt" then
		a.prompt_done, a.prompt_tokens = done, total
		a.cached_tokens = a.cached_tokens or done
	else
		a.tokens = done
	end
	local now = uv.now()
	if sink.progress and (changed or now - a.sent >= PROGRESS_MS) then
		a.sent = now
		sink:progress(a)
	end
end

-- The speculation session of a request, or nil when it is off. The
-- request field `speculative` wins over the engine default. Models
-- without pool sequences or truncation decode plainly.
function E:spec_start(req, prompt)
	local want = req.speculative
	if want == nil then
		want = self.spec_on
	end
	if not want or not self.model.truncate then
		return nil
	end
	local o = self.spec_opts
	local drafter = o.drafter and o.drafter(req) or spec.lookup(o)
	local sp = spec.session(drafter, self.spec_costs, { max = o.max, recurrent = self.model.recurrent,
	    adaptive = o.adaptive })
	sp.seq = table.move(prompt, 1, #prompt, 1, {})
	-- A saved state from an earlier request does not fit this one.
	if self.states then
		self.states.cache = nil
	end
	return sp
end

--- Run one decode step with speculation, alone on the device.
-- pend are the tokens that the cache lacks: tokens that an earlier
-- verify pass kept but could not keep in a recurrent state, then the
-- last sampled token. The drafter adds guesses when they pay (see
-- `lgml.spec.Session.choose`). `verify` runs them all in one pass.
-- Other running requests decode in batched passes, with no guesses.
-- @return nil when the step must run as a plain decode step, else a
-- table: `ready` (the sampled tokens, each `{ id, logprobs }`) and
-- `redo` (kept tokens that the next pass must run again)
function E:spec_step(sp, slot, toks, pend, smp, lim)
	if sp.off or slot and (slot.preempt or #self.pending > 0 or self.busy or self:owned() ~= 1) then
		return nil
	end
	local model = self.model
	local cache = slot and slot.cache or self.cache
	cache.replay_keep, cache.replay_keep_bytes = SPEC_GRAPHS, self.spec_kept_bytes
	-- A pass takes at most ubatch tokens; a ring takes at most its own.
	local room = math.min(lim.k, cache.n_ctx - cache.n_past, cache.ubatch or self.ubatch) - #pend
	local k = sp:choose(room)
	local drafts = {}
	if k > 0 then
		local good, got = pcall(sp.drafter.draft, sp.drafter, { seq = sp.seq, logits = sp.logits,
		    row = sp.row, hidden = sp.hidden, hidden_row = sp.hidden_row, cache = cache }, k)
		if not good then
			self:spec_memory(sp, cache, got)
			return nil
		end
		drafts = got
		-- An end token stays out of a draft: it ends the reply anyway.
		for j = 1, #drafts do
			if j > k or self.eog[drafts[j]] then
				for x = #drafts, j, -1 do
					drafts[x] = nil
				end
				break
			end
		end
	end
	local ids = table.move(pend, 1, #pend, 1, {})
	table.move(drafts, 1, #drafts, #ids + 1, ids)
	if slot then
		self:acquire()
	end
	-- A full pool sends the step to the plain path, which can wait.
	local ok, res = true, nil
	if not cache.pool or pcall(model.grow, model, cache, math.min(cache.n_ctx, cache.n_past + #ids)) then
		if slot and model.drop_batch then
			model:drop_batch(false)
		end
		ok, res = pcall(self.verify, self, sp, cache, toks, ids, #pend, drafts, smp, lim)
		if not ok and budget_mod.is_budget(res) then
			ok, res = pcall(self.spec_memory, self, sp, cache, res)
		end
	end
	if slot then
		self:release()
	end
	if not ok then
		error(res, 0)
	end
	return res
end

-- A draft or verify pass did not fit the memory budget: speculation
-- stops for this request and lets its graphs go, and the step runs as a
-- plain decode step. Other errors go on.
function E:spec_memory(sp, cache, err)
	if not budget_mod.is_budget(err) then
		error(err, 0)
	end
	L:warn("speculation does not fit the memory budget; it stops for this request", "err", err)
	sp.off = true
	cache.replay_keep = nil
	self:relieve(cache)
	return nil
end

-- The verify pass of `spec_step`: run ids, the np pending tokens and
-- the drafts, sample each row from the last pending one on, and drop
-- the rejected positions from the cache. A recurrent state goes back
-- to its copy from before the pass; the kept tokens run again in the
-- next pass.
function E:verify(sp, cache, toks, ids, np, drafts, smp, lim)
	local model, nd = self.model, #drafts
	local n0 = cache.n_past
	local saved = nd > 0 and model.recurrent
	local t0 = lgml.clock()
	if saved then
		self.states = self.states or spec.state_copy(model)
		self.states.save(cache)
	end
	local opts = { logits = nd > 0 and "all" or "last", flash_attn = self.flash_attn }
	local finish
	if sp.drafter.hidden then
		local name = "l_out-" .. (model.hp.n_layer - 1)
		opts.tag = "hidden"
		finish = function(_, out, taps)
			return out, taps[name]
		end
	end
	local logits, hidden = replay.run(model, cache, ids, opts, finish)
	if cache.replay_hit and (nd > 0 or #ids == 1) then
		sp.costs:add(#ids, lgml.clock() - t0)
	end
	-- Rows: the last pending token, then one per draft.
	local first = nd > 0 and np or 1
	local ready, a = {}, 0
	for j = 1, nd + 1 do
		local row = first + j - 1
		local id = smp:sample(logits, row)
		smp:accept(id)
		sp.seq[#sp.seq + 1] = id
		ready[j] = { id, lim.lp and self:logprobs(smp, id, lim.top) or nil }
		if j > nd or id ~= drafts[j] then
			sp.row, sp.hidden_row = row, np + j - 1
			break
		end
		a = j
	end
	sp.logits, sp.hidden = logits, hidden
	sp:update(nd, a)
	local keep, redo = n0 + np + a, {}
	if a < nd and saved then
		self.states.restore(cache)
		if cache.pool then
			model:truncate(cache, n0)
		end
		redo = table.move(ids, 1, np + a, 1, {})
		return { ready = ready, redo = redo }
	elseif a < nd then
		if model.rewind and model:rewind(cache, keep) ~= keep then
			error("speculation: the ring lost the kept positions", 0)
		end
		cache.n_past = keep
		if cache.pool then
			model:truncate(cache, keep)
		end
	end
	table.move(ids, 1, np + a, #toks + 1, toks)
	return { ready = ready, redo = redo }
end

-- At the end of a reply, drop the tokens past keep that verify passes
-- left in the cache, so the slot holds what a plain decode leaves. A
-- recurrent state goes back to its saved copy, when it has one. The
-- slot lets its kept graphs go.
function E:spec_end(sp, slot, toks, keep)
	self.spec_drafted = (self.spec_drafted or 0) + sp.drafted
	self.spec_accepted = (self.spec_accepted or 0) + sp.accepted
	L:debug("speculation", "drafted", sp.drafted, "accepted", sp.accepted, "passes", sp.passes,
	    "rate", ("%.2f"):format(sp:rate()))
	local model = self.model
	local cache = slot and slot.cache or self.cache
	cache.replay_keep = nil
	replay.drop_kept(cache)
	if #toks <= keep then
		return
	end
	if slot then
		self:acquire()
	end
	local ok, err = pcall(function()
		local sc = self.states
		if model.recurrent then
			if not (sc and sc.cache == cache and sc.n <= keep) then
				return
			end
			sc.restore(cache)
			keep = sc.n
		elseif model.rewind and model:rewind(cache, keep) ~= keep then
			return
		end
		cache.n_past = math.min(cache.n_past, keep)
		if cache.pool then
			model:truncate(cache, keep)
		end
		for i = #toks, keep + 1, -1 do
			toks[i] = nil
		end
	end)
	if slot then
		self:release()
	end
	if not ok then
		error(err, 0)
	end
end

function E:run(req, sink, st)
	self.image_ms = 0
	self.acts[sink] = { request_id = log.request_id(), state = "wait", sent = 0 }
	local ok, a, b, c = pcall(self.run_prompt, self, req, sink, st)
	self.acts[sink] = nil
	self:drop_images()
	if not ok then
		error(a, 0)
	end
	return a, b, c
end

function E:run_prompt(req, sink, st)
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
	if req.count_only then
		sink:finish("stop", { prompt_tokens = #prompt, completion_tokens = 0, cached_tokens = 0,
		    reasoning_tokens = 0 })
		return true
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
	gopts.logit_bias, gerr = self:logit_bias(req.logit_bias)
	if gerr then
		return nil, gerr, 400
	end
	local want_lp, top_lp = req.logprobs == true, math.tointeger(req.top_logprobs) or 0
	local good, smp = pcall(sampler_opts, req, think, gopts, self.sampling)
	if not good and gopts.triggers then
		-- A tool schema that the grammar cannot take leaves calls free.
		L:warn("tool call grammar", "error", tostring(smp))
		gopts = { logit_bias = gopts.logit_bias }
		good, smp = pcall(sampler_opts, req, think, gopts, self.sampling)
	end
	if not good then
		return nil, tostring(smp):gsub("^.-grammar: ", "grammar: "), 400
	end
	local _ <close> = smp
	if L:on(log.DEBUG) then
		local preset = think and "qwen3_think" or "qwen3"
		L:debug("sampling", "preset", preset, "settings", sampler.settings(preset, sampler_opts_table(req, self.sampling)),
		    "seed", req.seed or "random", "max_tokens", max, "thinking", think,
		    "grammar", gopts.grammar and first_lines(gopts.grammar, 6), "root", gopts.grammar_root,
		    "triggers", gopts.triggers and #gopts.triggers)
	end
	-- With no tools, as with tool_choice "none", the parser reads no calls.
	local out = output.new(sink, self.family.new_parser({ tools = req.tools or {}, enable_thinking = think }),
	    { stop = req.stop, one_call = req.parallel_tool_calls == false })
	local dec = tokenizer.decoder(self.tk, { special = false })

	local t0 = lgml.clock()
	local cached = self:reuse(prompt)
	-- A prompt that the pool cannot hold next to the running requests
	-- waits for one of them to end.
	while st and self.concurrent and self:owned() > 0 and not self:room_for(#prompt + 1) do
		L:debug("pool full, prompt waits", "tokens", #prompt, "running", self:owned())
		if not self:wait_room(st, req, sink) then
			return true
		end
		cached = self:reuse(prompt)
	end
	L:debug("prefill", "tokens", #prompt, "cached", cached, "restored", self.pc and self.pc.restored or 0,
	    "compute", #prompt - cached)
	if self.pool and L:on(log.DEBUG) then
		local a = self.pool.alloc
		L:debug("kv pool", "cells", self.pool.n_ctx, "used", a.used * a.size, "cached", a.cached * a.size,
		    "hits", a.stats.hits, "hit_blocks", a.stats.hit_blocks, "evicted", a.stats.evicted,
		    "slot_evictions", self.evictions or 0)
	end
	if sink.prompt and not sink:prompt(#prompt, cached) then
		return true
	end
	-- Concurrent engines: the request owns its slot, and prefill and
	-- decode run in batched passes with other requests. A prompt with
	-- images prefills alone first.
	local logits, slot
	if st and self.concurrent and #images == 0 and cached < #prompt then
		local stops = self.pc and self.pc:plan(prompt, cached + 1) or {}
		if self.pc then
			self:share_state(self.cache, self.tokens, self.pc.ckpts)
		end
		slot = self:own_slot(st)
		logits, slot = self:prefill_batched(st, req, slot, prompt, cached + 1, stops, sink)
	else
		logits = self:prefill(prompt, cached + 1, sink)
		slot = logits and st and self.concurrent and self:own_slot(st)
	end
	if not logits then
		return true
	end
	local toks = slot and slot.tokens or self.tokens
	local t1 = lgml.clock()
	local n, reason, stopped = 0, "length", false
	local tracing = log.level >= log.TRACE
	-- At the reasoning budget, the tokens of the block end replace sampling.
	local budget = think and THINK_END[self.family_name] and
	    (math.tointeger(req.reasoning_budget) or self.reasoning_budget)
	local forced
	local guard, repeated = repeat_guard(self.repeat_stop), nil
	-- With speculation, a verify pass samples several tokens ahead: they
	-- wait in ready. redo holds kept tokens that a recurrent state lost
	-- at a rejection; the next pass runs them first.
	local sp = self:spec_start(req, prompt)
	local ready, ri, redo = {}, 0, {}
	local aborted = false
	while n < max do
		local id, lp
		if ri < #ready then
			ri = ri + 1
			id, lp = ready[ri][1], ready[ri][2]
		else
			if forced and #forced > 0 then
				id = table.remove(forced, 1)
			else
				id = smp:sample(logits)
			end
			smp:accept(id)
			if sp then
				sp.seq[#sp.seq + 1] = id
			end
		end
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
		if want_lp and not lp then
			lp = self:logprobs(smp, id, top_lp)
		end
		local r = out:token(piece, lp)
		if r == "stop" then
			reason, stopped = "stop", true
			break
		elseif not r then
			aborted = true
			break
		end
		self:note(sink, slot or self.slot, "answer", n)
		if budget and not forced and out:in_reasoning() and out.reasoning_tokens >= budget then
			forced = self.tk:encode(WRAP_UP .. THINK_END[self.family_name])
			L:debug("reasoning budget", "tokens", out.reasoning_tokens)
		end
		if guard and guard(id) then
			L:warn("the same token repeats; the reply ends", "id", id, "count", self.repeat_stop, "tokens", n)
			reason, repeated = "length", 1
			break
		end
		if n >= max then
			break
		end
		-- The pass of the last sampled token, after the redo tokens.
		if ri >= #ready then
			local pend = redo
			pend[#pend + 1] = id
			redo, ready, ri = {}, {}, 0
			local res
			if sp and not (forced and #forced > 0) then
				-- Guesses stop before the token limit and the reasoning
				-- budget: past them, the reply ends or forced tokens follow.
				local k = max - n - 1
				if budget and not forced then
					k = math.min(k, budget - out.reasoning_tokens - 1)
				end
				res = self:spec_step(sp, slot, toks, pend, smp, { k = k, lp = want_lp, top = top_lp })
			end
			if res then
				ready, redo = res.ready, res.redo
			else
				local good, lg
				if slot then
					good, lg = pcall(self.batch_step, self, slot, pend)
				else
					good, lg = pcall(self.eval, self, pend)
				end
				local resumed = false
				if not good then
					if not tostring(lg):find("kv pool full", 1, true) then
						error(lg, 0)
					end
					-- Other requests hold the rest of the pool: this one waits
					-- for them. Alone, the reply ends.
					if not slot or self:owned() < 2 then
						L:info("kv pool full; the reply ends", "tokens", n)
						break
					end
					slot, lg = self:preempt(st, req, slot, pend, sink)
					if not lg then
						aborted = true
						break
					end
					toks, resumed = slot.tokens, true
				end
				logits = lg
				if sp then
					sp.logits, sp.row, sp.hidden = lg, nil, nil
				end
				if not resumed then
					table.move(pend, 1, #pend, #toks + 1, toks)
				end
			end
			self:publish(slot and slot.cache or self.cache, toks)
			stream.sleep(0)
			if sink:cancelled() then
				aborted = true
				break
			end
		end
	end
	if sp then
		self:spec_end(sp, slot, toks, #prompt + n - 1)
	end
	if aborted then
		return true
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
		reasoning_tokens = out.reasoning_tokens, stop_sequence = out.stop_sequence,
	}, {
		cache_n = cached, restored_n = self.pc and self.pc.restored or 0,
		prompt_n = #prompt - cached, prompt_ms = (t1 - t0) * 1000,
		prompt_per_token_ms = (t1 - t0) * 1000 / math.max(#prompt - cached, 1),
		prompt_per_second = (#prompt - cached) / math.max(t1 - t0, 1e-9),
		predicted_n = n, predicted_ms = (t2 - t1) * 1000,
		predicted_per_token_ms = (t2 - t1) * 1000 / math.max(n, 1),
		predicted_per_second = n / math.max(t2 - t1, 1e-9),
		images = #images, image_encode_ms = self.image_ms, repeat_stop = repeated,
		draft_n = sp and sp.drafted or nil, draft_accepted = sp and sp.accepted or nil,
	})
	return true
end

return M
