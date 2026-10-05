--- Model id and memory from a config spec, before the model loads.
-- `lgml-server` uses it to count models against a memory budget. It
-- reads file sizes and gguf metadata; it does not load weights or
-- start a ggml backend.
-- @module lgml.estimate
local uv = require("luv")
local lgml = require("lgml")
local detect = require("lgml.detect")
local kvpool = require("lgml.kvpool")

local M = {}

local MiB = 1048576

--- Bytes added to each estimate for compute buffers: 1 GiB.
M.MARGIN = 1024 * MiB
--- Bytes added to a text encoder on its own backend, for its compute
-- buffers: 256 MiB.
M.TEXT_MARGIN = 256 * MiB
--- Bytes added to the CPU part of a chat model with weights on the CPU
-- (see `lgml.placement`), for its compute buffers: 256 MiB.
M.CPU_MARGIN = 256 * MiB

-- Default ids of engine modules that do not take them from a file.
local ENGINE_IDS = {
	["lgml.tts.orpheus"] = "orpheus",
	["lgml.tts.qwen3tts"] = "qwen3-tts",
	["lgml.tts.vibevoice"] = "vibevoice",
}

local function basename(path)
	return (path:gsub("/+$", ""):match("([^/]+)$") or path)
end

--- The bytes of a model path.
-- A file gives its size. A directory gives the sum of the sizes of its
-- regular files, not of its subdirectories.
-- @tparam string path a file or a directory
-- @treturn integer the bytes, 0 when path does not exist
function M.files(path)
	local st = path and uv.fs_stat(path)
	if not st then
		return 0
	elseif st.type ~= "directory" then
		return st.size
	end
	local n = 0
	local h = uv.fs_scandir(path)
	while h do
		local name = uv.fs_scandir_next(h)
		if not name then
			break
		end
		local s = uv.fs_stat(path .. "/" .. name)
		if s and s.type == "file" then
			n = n + s.size
		end
	end
	return n
end

local function gguf_meta(path)
	if type(path) ~= "string" or not path:match("%.gguf$") then
		return nil
	end
	local info = lgml.gguf(path)
	return info and info.meta
end

--- The memory kind of a backend: `"cpu"` or `"gpu"`.
-- A name that starts with "CPU" is a CPU backend. With no name, the
-- best backend is a GPU when the machine has one.
-- @tparam[opt] string backend the backend name
-- @tparam[opt] boolean gpu false when the machine has no GPU (default true)
-- @treturn string the kind
function M.kind(backend, gpu)
	if backend then
		return backend:upper():match("^CPU") and "cpu" or "gpu"
	end
	return gpu == false and "cpu" or "gpu"
end

--- The model id of a spec.
-- It is `spec.id`, else the gguf `general.name` for a chat model, else
-- the default id of the engine module, else the DiT file name for
-- `lgml.image.engine`, else the base name of the model path without
-- `.gguf`. It matches the id that the engine picks when it has none.
-- @tparam table spec the model spec
-- @treturn string the id
function M.id(spec)
	if spec.id then
		return spec.id
	end
	local e = spec.engine
	if ENGINE_IDS[e] then
		return ENGINE_IDS[e]
	elseif e == "lgml.image.engine" then
		local h = uv.fs_scandir(spec.model)
		while h do
			local name = uv.fs_scandir_next(h)
			if not name then
				break
			elseif name:lower():match("%.gguf$") and name:lower():find("image") then
				return (name:gsub("%.gguf$", ""))
			end
		end
	end
	local meta = not e and gguf_meta(spec.model)
	if meta then
		return meta["general.name"] or "lgml"
	end
	return (basename(spec.model):gsub("%.gguf$", ""))
end

-- The config of a chat gguf for the transformer, or nil.
local function config(meta)
	local transformer = require("lgml.transformer")
	if not transformer.supports(meta["general.architecture"]) then
		return nil
	end
	local ok, hp = pcall(transformer.config, meta)
	return ok and hp or nil
end

-- KV cache bytes of a chat gguf: per position and fixed on the device,
-- the trained context, then per position and fixed on the CPU. cpu is
-- the set of layers, from 0, whose cache is on the CPU. nil for other
-- models.
local function cache_size(meta, spec, cpu)
	local transformer = require("lgml.transformer")
	local hp = config(meta)
	if not hp then
		return nil
	end
	local layers = {}
	for i, d in ipairs(hp.layers) do
		layers[i] = setmetatable({ cpu = cpu and cpu[i - 1] or nil }, { __index = d })
	end
	local shape = { hp = hp, layers = layers }
	local fa = spec.flash_attn ~= false
	local copts = { type_k = spec.cache_type_k or "f16", type_v = spec.cache_type_v or "f16" }
	if fa and not spec.cache_type_k and not spec.cache_type_v and
	    pcall(transformer.cache_size, shape, { type = "q8_0" }) then
		copts = { type = "q8_0" }
	end
	local per, fixed, cper, cfixed = transformer.cache_size(shape, copts)
	return per, fixed, hp.n_ctx, cper, cfixed
end

-- The slot count of a spec: its slots key, else the default of
-- lgml.kvpool for a pool of n positions of per bytes and slots with
-- fixed bytes each, as in lgml.engine.
local function slot_count(spec, per, fixed, n)
	local s = math.tointeger(tonumber(spec.slots))
	if s then
		return math.max(1, s)
	end
	return kvpool.default_seqs(per * n, fixed)
end

-- Place the weights of a chat model on a GPU by the placement keys of
-- spec. Sets r.n_cpu_moe for "auto", moves bytes from r.files to
-- r.extra.cpu, and returns the set of layers whose cache is on the CPU.
local function place(r, spec, meta, budget)
	local placement = require("lgml.placement")
	local pspec = spec
	if spec.moe_cache then
		-- An expert cache keeps every routed expert in host memory.
		pspec = { cpu_moe = true }
		for k, v in pairs(spec) do
			if k ~= "cpu_moe" and k ~= "n_cpu_moe" then
				pspec[k] = v
			end
		end
	end
	local rules = placement.rules(pspec)
	local hp = rules and config(meta)
	if not hp then
		return nil
	end
	local tensors = lgml.gguf(spec.model).tensors
	if rules.auto then
		local n = 0
		if budget and budget < math.huge then
			local per, fixed = cache_size(meta, spec)
			local ctx = math.tointeger(tonumber(spec.ctx)) or math.min(hp.n_ctx, placement.AUTO_CTX)
			local slots = slot_count(spec, per, fixed, math.tointeger(tonumber(spec.ctx)) or hp.n_ctx)
			local mmproj = spec.mmproj and M.files(spec.mmproj) or 0
			n = placement.auto(rules, tensors, hp.n_layer,
			    budget - mmproj - M.MARGIN - fixed * slots - per * ctx)
		end
		rules.n_cpu_moe, rules.auto = n, nil
		r.n_cpu_moe = n
	end
	local set, bytes, layers = placement.plan(rules, tensors, hp.n_layer)
	if not set then
		return nil
	end
	r.files = r.files - bytes
	r.extra.cpu = (r.extra.cpu or 0) + bytes + M.CPU_MARGIN
	return layers
end

--- The memory estimate of a spec.
-- It is the bytes of the model files and the mmproj, plus the KV cache
-- of a chat model, plus `MARGIN`. The KV cache of a chat model is one
-- pool of `ctx` cells that its slots share, plus the rings and
-- recurrent states of each slot (`slots`, default from
-- `lgml.kvpool.default_seqs`). An image model (`lgml.image.engine`)
-- with a `text_backend` of another kind than its `backend` has two
-- parts: the text encoder file plus `TEXT_MARGIN` goes in `extra`, under
-- the kind of `text_backend`, and the other files in `files`. For a
-- chat model with no `ctx`, the context is the trained one, or less so
-- that the model fits budget alone, in steps of 256 and at least 256.
-- The cache types follow `lgml.engine`: q8_0 with flash attention when
-- it fits, else f16.
-- A chat model on a GPU with placement keys (see `lgml.placement`) has
-- a cpu part: its weights on the CPU, the KV cache of its layers on the
-- CPU, and `CPU_MARGIN`. With `n_cpu_moe = "auto"` and a budget, the
-- estimate picks the count so the weights and the cache of `ctx`
-- positions (default: at most `lgml.placement.AUTO_CTX`) fit; the
-- result gives it as n_cpu_moe.
-- With `moe_cache`, every routed expert goes to the CPU part, and the
-- cache slots count on the GPU: `moe_cache` MiB, or for "auto" under a
-- budget, the rest of the budget, given as moe_cache_max.
-- @tparam table spec the model spec: `model`, `engine`, `stt`, `mmproj`,
-- `ctx`, `slots`, `flash_attn`, `cache_type_k`, `cache_type_v`, `backend`,
-- `text_backend`, `cpu_moe`, `n_cpu_moe`, `n_gpu_layers`, `override`,
-- `moe_cache` and `moe_cache_max`
-- @tparam[opt] number budget the memory budget in bytes of the kind of
-- `backend`
-- @tparam[opt] boolean gpu false when the machine has no GPU (see `kind`)
-- @treturn table `{ bytes, files, kv, margin, n_ctx, slots, kind, extra,
-- n_cpu_moe, moe_cache, moe_cache_max }`; moe_cache is the bytes of the
-- expert cache, and moe_cache_max its cap in bytes for "auto"; bytes is the part on the kind of `backend`; n_ctx is the
-- context for the worker, or nil when the estimate does not know it;
-- slots is the slot count of a chat model;
-- extra is `{ [kind] = bytes }` for the parts on other kinds
-- @raise on bad placement keys
function M.model(spec, budget, gpu)
	local files = M.files(spec.model) + (spec.mmproj and M.files(spec.mmproj) or 0)
	local kind = M.kind(spec.backend, gpu)
	local r = { files = files, kv = 0, margin = M.MARGIN, kind = kind, extra = {} }
	if spec.engine == "lgml.image.engine" and spec.text_backend and
	    M.kind(spec.text_backend, gpu) ~= kind then
		local _, llm = detect.image_files(spec.model)
		local text = M.files(llm)
		r.files = files - text
		r.extra[M.kind(spec.text_backend, gpu)] = text + M.TEXT_MARGIN
	end
	local meta = not spec.engine and not spec.stt and gguf_meta(spec.model)
	if meta and not detect.is_whisper(spec.model) then
		local cpu = kind == "gpu" and place(r, spec, meta, budget) or nil
		local per, fixed, n_train, cper, cfixed = cache_size(meta, spec, cpu)
		if per then
			local n = math.tointeger(tonumber(spec.ctx)) or n_train
			-- The slots share one pool of n cells; each slot has its own
			-- rings and recurrent states.
			local slots = slot_count(spec, per + cper, fixed + cfixed, n)
			if not spec.ctx and budget and budget < math.huge and per > 0 then
				local fit = (budget - r.files - fixed * slots - M.MARGIN) // per // 256 * 256
				n = math.min(n, math.max(math.tointeger(fit) or 256, 256))
			end
			r.n_ctx = n
			r.slots = slots
			r.kv = per * n + fixed * slots
			if cpu then
				r.extra.cpu = r.extra.cpu + cper * n + cfixed * slots
			end
		end
	end
	if spec.moe_cache and kind == "gpu" and r.extra.cpu then
		local mib = tonumber(spec.moe_cache)
		local bytes = mib and math.floor(mib * 1048576)
		if not bytes and budget and budget < math.huge then
			bytes = math.max(0, budget - r.files - r.kv - r.margin)
			local max = tonumber(spec.moe_cache_max)
			if max then
				bytes = math.min(bytes, math.floor(max * 1048576))
			end
			r.moe_cache_max = bytes
		end
		r.moe_cache = bytes or 0
	end
	r.bytes = r.files + r.kv + r.margin + (r.moe_cache or 0)
	return r
end

return M
