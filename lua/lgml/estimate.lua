--- Model id and memory from a config spec, before the model loads.
-- `lgml-server` uses it to count models against a memory budget. It
-- reads file sizes and gguf metadata; it does not load weights or
-- start a ggml backend.
-- @module lgml.estimate
local uv = require("luv")
local lgml = require("lgml")
local detect = require("lgml.detect")

local M = {}

local MiB = 1048576

--- Bytes added to each estimate for compute buffers: 1 GiB.
M.MARGIN = 1024 * MiB
--- Bytes added to a text encoder on its own backend, for its compute
-- buffers: 256 MiB.
M.TEXT_MARGIN = 256 * MiB

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

-- KV cache bytes of a chat gguf: per position, fixed, and the trained
-- context. nil for other models.
local function cache_size(meta, spec)
	local transformer = require("lgml.transformer")
	local arch = meta["general.architecture"]
	if not transformer.supports(arch) then
		return nil
	end
	local ok, hp = pcall(transformer.config, meta)
	if not ok then
		return nil
	end
	local shape = { hp = hp, layers = hp.layers }
	local fa = spec.flash_attn ~= false
	local copts = { type_k = spec.cache_type_k or "f16", type_v = spec.cache_type_v or "f16" }
	if fa and not spec.cache_type_k and not spec.cache_type_v and
	    pcall(transformer.cache_size, shape, { type = "q8_0" }) then
		copts = { type = "q8_0" }
	end
	local per, fixed = transformer.cache_size(shape, copts)
	return per, fixed, hp.n_ctx
end

--- The memory estimate of a spec.
-- It is the bytes of the model files and the mmproj, plus the KV cache
-- of a chat model, plus `MARGIN`. An image model (`lgml.image.engine`)
-- with a `text_backend` of another kind than its `backend` has two
-- parts: the text encoder file plus `TEXT_MARGIN` goes in `extra`, under
-- the kind of `text_backend`, and the other files in `files`. For a
-- chat model with no `ctx`, the context is the trained one, or less so
-- that the model fits budget alone, in steps of 256 and at least 256.
-- The cache types follow `lgml.engine`: q8_0 with flash attention when
-- it fits, else f16.
-- @tparam table spec the model spec: `model`, `engine`, `stt`, `mmproj`,
-- `ctx`, `flash_attn`, `cache_type_k`, `cache_type_v`, `backend` and
-- `text_backend`
-- @tparam[opt] number budget the memory budget in bytes of the kind of
-- `backend`
-- @tparam[opt] boolean gpu false when the machine has no GPU (see `kind`)
-- @treturn table `{ bytes, files, kv, margin, n_ctx, kind, extra }`; bytes
-- is the part on the kind of `backend`; n_ctx is the context for the
-- worker, or nil when the estimate does not know it; extra is
-- `{ [kind] = bytes }` for the parts on other kinds
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
		local per, fixed, n_train = cache_size(meta, spec)
		if per then
			local n = math.tointeger(tonumber(spec.ctx)) or n_train
			if not spec.ctx and budget and budget < math.huge and per > 0 then
				local fit = (budget - files - fixed - M.MARGIN) // per // 256 * 256
				n = math.min(n, math.max(math.tointeger(fit) or 256, 256))
			end
			r.n_ctx = n
			r.kv = per * n + fixed
		end
	end
	r.bytes = r.files + r.kv + r.margin
	return r
end

return M
