--- The model specs of lgml-server: checks, worker arguments, and the
-- engines of a config file or of the command line options. A bad spec
-- or config raises an error with the message.
-- @module lgml.server.models
local log = require("lgml.log")

local M = {}

local L = log.new("server")
local MiB = 1048576

local function die(msg)
	error(msg, 0)
end

-- Adapters of a spec as { path, scale } entries. spec.lora is
-- "file[:scale]", or a list of those or of { path, scale }.
local function adapters(spec)
	local list = type(spec.lora) == "table" and spec.lora or { spec.lora }
	local out = {}
	for i, a in ipairs(list) do
		local path, scale
		if type(a) == "table" then
			path, scale = a.path or a[1], a.scale or a[2] or 1
		elseif type(a) == "string" then
			path, scale = a:match("^(.*):([^:/]+)$")
			if not (path and tonumber(scale)) then
				path, scale = a, 1
			end
		end
		scale = tonumber(scale)
		if type(path) ~= "string" or not scale then
			die(("bad lora entry %d: %s"):format(i, tostring(a)))
		end
		out[i] = { path = path, scale = scale }
	end
	return out
end

--- Makes one engine from a model spec.
-- A spec is `{ model, id, backend, threads, ctx, flash_attn, stt,
-- inproc, ... }` (see the lgml-server usage). With `inproc`, the engine
-- runs in this process; else `o.start` starts its worker.
-- @tparam table spec the model spec
-- @tparam table o `start(spec, args, modes, name, options)`: starts a
-- worker with the lgml-worker args and gives its engine; name is the
-- prompt store name; `managed(spec, modes, lora, start)`: adds the spec
-- to a pool of models that load on demand, and gives the pool model;
-- `names`: a set that gets the prompt store name
-- @return the engine
-- @raise for a bad spec
function M.engine(spec, o)
	local managed = o.managed
	local n_ctx = spec.ctx and tonumber(spec.ctx)
	if spec.ctx and not n_ctx then
		die("bad context size " .. tostring(spec.ctx))
	end
	local ram, ckpt = spec.cache_ram and tonumber(spec.cache_ram),
	    spec.ctx_checkpoints and math.tointeger(tonumber(spec.ctx_checkpoints))
	if spec.cache_ram and not ram then
		die("bad cache RAM size " .. tostring(spec.cache_ram))
	end
	if spec.ctx_checkpoints and not (ckpt and ckpt >= 0) then
		die("bad checkpoint count " .. tostring(spec.ctx_checkpoints))
	end
	local kvi = spec.kv_initial and math.tointeger(tonumber(spec.kv_initial))
	if spec.kv_initial and not (kvi and kvi >= 0) then
		die("bad KV initial size " .. tostring(spec.kv_initial))
	end
	local sp = spec.speculative
	if type(sp) ~= "boolean" and sp ~= nil then
		sp = math.tointeger(tonumber(sp))
		if not (sp and sp >= 0 and sp <= 64) then
			die("bad speculative setting " .. tostring(spec.speculative))
		end
	end
	if sp == true then
		sp = 16
	end
	if spec.spec_draft ~= nil and not math.tointeger(spec.spec_draft) then
		die("bad spec_draft " .. tostring(spec.spec_draft))
	end
	if spec.spec_min_p ~= nil and not (tonumber(spec.spec_min_p) and spec.spec_min_p >= 0 and spec.spec_min_p <= 1) then
		die("bad spec_min_p " .. tostring(spec.spec_min_p))
	end
	local nslots = spec.slots and math.tointeger(tonumber(spec.slots))
	if spec.slots and not (nslots and nslots >= 1 and nslots <= 64) then
		die("bad slot count " .. tostring(spec.slots))
	end
	local rst = spec.repeat_stop_tokens and math.tointeger(tonumber(spec.repeat_stop_tokens))
	if spec.repeat_stop_tokens and not (rst and rst >= 0) then
		die("bad repeat_stop_tokens " .. tostring(spec.repeat_stop_tokens))
	end
	local rbt = spec.reasoning_budget_tokens and math.tointeger(tonumber(spec.reasoning_budget_tokens))
	if spec.reasoning_budget_tokens and not (rbt and rbt >= 0) then
		die("bad reasoning_budget_tokens " .. tostring(spec.reasoning_budget_tokens))
	end
	if rbt == 0 then
		rbt = nil
	end
	local ub = spec.ubatch
	if ub ~= nil and ub ~= "auto" then
		ub = math.tointeger(tonumber(ub))
		if not (ub and ub >= 1 and ub <= 65536) then
			die("bad ubatch " .. tostring(spec.ubatch) .. ": an integer from 1, or auto")
		end
	end
	local threads = spec.threads and math.tointeger(tonumber(spec.threads))
	if spec.threads and not (threads and threads >= 1 and threads <= 1024) then
		die("bad thread count " .. tostring(spec.threads))
	end
	local detect = require("lgml.detect")
	local kind = not spec.engine and detect.engine_of(spec)
	local stt = kind == "lgml.stt_engine"
	local embed = kind == "lgml.embed_engine"
	local diff = kind == "lgml.diffusion_engine"
	local okm, modes = pcall(detect.modes, spec)
	if not okm then
		die(modes)
	end
	local lora = adapters(spec)
	if #lora > 0 and (stt or embed or diff or spec.engine) then
		die("lora needs a chat model")
	end
	if spec.diffusion ~= nil and (type(spec.diffusion) ~= "table" or not diff) then
		die("diffusion needs a table and a block diffusion model")
	end
	if spec.sampling ~= nil and (stt or embed or diff or spec.engine) then
		die("sampling needs a chat model")
	end
	if spec.max_tokens ~= nil and (not math.tointeger(spec.max_tokens) or spec.max_tokens < 1 or
	    stt or embed or diff or spec.engine) then
		die("max_tokens needs a chat model and a count above 0")
	end
	local oks, serr = pcall(require("lgml.engine").check_sampling, spec.sampling)
	if not oks then
		die(serr)
	end
	local imin, imax = spec.image_min_tokens and math.tointeger(tonumber(spec.image_min_tokens)),
	    spec.image_max_tokens and math.tointeger(tonumber(spec.image_max_tokens))
	if (spec.image_min_tokens and not (imin and imin > 0)) or
	    (spec.image_max_tokens and not (imax and imax >= (imin or 1))) then
		die("bad image token range")
	end
	if spec.text_backend and spec.engine ~= "lgml.image.engine" then
		die("text_backend needs engine lgml.image.engine")
	end
	local side = spec.image_max_side and math.tointeger(tonumber(spec.image_max_side))
	if spec.image_max_side and (spec.engine ~= "lgml.image.engine" or not side or side < 16 or
	    side > 2048 or side % 16 ~= 0) then
		die("image_max_side needs engine lgml.image.engine and a multiple of 16 up to 2048")
	end
	local okp, rules = pcall(require("lgml.placement").rules, spec)
	if not okp then
		die(rules)
	elseif (rules or spec.moe_cache) and (stt or embed or diff or spec.engine) then
		die("cpu_moe, n_cpu_moe, n_gpu_layers, override and moe_cache need a chat model")
	end
	if spec.inproc and diff then
		return require("lgml.diffusion_engine").new({
			model = spec.model, backend = spec.backend, threads = threads, n_ctx = n_ctx,
			flash_attn = spec.flash_attn ~= false, id = spec.id, sampler = spec.diffusion,
		})
	elseif spec.inproc and spec.engine then
		return require(spec.engine).new({
			model = spec.model, backend = spec.backend, text_backend = spec.text_backend,
			threads = threads, id = spec.id, n_ctx = n_ctx, max_side = side,
		})
	elseif spec.inproc and stt then
		return require("lgml.stt_engine").new({
			model = spec.model, backend = spec.backend, threads = threads, id = spec.id,
		})
	elseif spec.inproc and embed then
		return require("lgml.embed_engine").new({
			model = spec.model, backend = spec.backend, threads = threads, n_ctx = n_ctx,
			flash_attn = spec.flash_attn ~= false, id = spec.id,
			cache_type_k = spec.cache_type_k, cache_type_v = spec.cache_type_v, kv_initial = kvi,
		})
	elseif spec.inproc then
		return require("lgml.engine").new({
			model = spec.model, backend = spec.backend, threads = threads, n_ctx = n_ctx,
			flash_attn = spec.flash_attn ~= false, id = spec.id,
			cache_type_k = spec.cache_type_k, cache_type_v = spec.cache_type_v,
			recipe = spec.recipe, cache_ram = ram, ctx_checkpoints = ckpt, lora = lora,
			sampling = spec.sampling, max_tokens = spec.max_tokens,
			kv_initial = kvi, slots = nslots, repeat_stop_tokens = rst,
			reasoning_budget_tokens = rbt, ubatch = ub,
			speculative = sp and sp > 0 and { max = sp, mtp = spec.spec_mtp, draft = spec.spec_draft,
				min_p = spec.spec_min_p } or nil,
			mmproj = spec.mmproj, image_min_tokens = imin, image_max_tokens = imax,
			cpu_moe = spec.cpu_moe, n_cpu_moe = spec.n_cpu_moe, n_gpu_layers = spec.n_gpu_layers,
			override = spec.override, moe_cache = spec.moe_cache,
			moe_cache_max = spec.moe_cache_max, moe_cache_profile = spec.moe_cache_profile,
			moe_cache_swap = spec.moe_cache_swap, moe_cache_halflife = spec.moe_cache_halflife,
			moe_cache_zerocopy = spec.moe_cache_zerocopy,
		})
	end
	local wargs = { "-m", spec.model }
	for flag, v in pairs({
		["-b"] = spec.backend, ["-t"] = threads, ["-c"] = spec.ctx, ["--alias"] = spec.id,
		["--text-backend"] = spec.text_backend,
		["--ctk"] = spec.cache_type_k, ["--ctv"] = spec.cache_type_v,
		["--recipe"] = spec.recipe, ["--cache-ram"] = spec.cache_ram,
		["--ctx-checkpoints"] = spec.ctx_checkpoints, ["--mmproj"] = spec.mmproj,
		["--kv-initial"] = kvi, ["--slots"] = nslots, ["--repeat-stop"] = rst,
		["--reasoning-budget"] = rbt, ["--ubatch"] = ub, ["--max-tokens"] = spec.max_tokens,
		["--spec"] = sp and sp > 0 and sp or nil,
		["--spec-mtp"] = sp and sp > 0 and spec.spec_mtp or nil,
		["--spec-draft"] = sp and sp > 0 and spec.spec_draft or nil,
		["--spec-min-p"] = sp and sp > 0 and spec.spec_min_p or nil,
		["--image-min-tokens"] = imin, ["--image-max-tokens"] = imax,
		["--image-max-side"] = side, ["--mem-margin"] = spec.mem_margin,
		["--moe-cache-max"] = spec.moe_cache_max, ["--moe-cache-profile"] = spec.moe_cache_profile,
		["--moe-cache-swap"] = spec.moe_cache_swap,
		["--moe-cache-halflife"] = spec.moe_cache_halflife,
	}) do
		table.move({ flag, tostring(v) }, 1, 2, #wargs + 1, wargs)
	end
	if spec.flash_attn == false then
		wargs[#wargs + 1] = "--no-fa"
	end
	for _, a in ipairs(lora) do
		table.move({ "--lora", ("%s:%.17g"):format(a.path, a.scale) }, 1, 2, #wargs + 1, wargs)
	end
	if rules then
		for _, o in ipairs(rules.override) do
			table.move({ "--override-tensor", o.pattern .. (o.cpu and "=CPU" or "=GPU") }, 1, 2,
			    #wargs + 1, wargs)
		end
		if rules.n_cpu_moe == math.huge then
			wargs[#wargs + 1] = "--cpu-moe"
		elseif rules.n_cpu_moe or rules.auto then
			table.move({ "--n-cpu-moe", tostring(rules.n_cpu_moe or "auto") }, 1, 2, #wargs + 1, wargs)
		end
		if rules.n_gpu_layers then
			table.move({ "--n-gpu-layers", tostring(rules.n_gpu_layers) }, 1, 2, #wargs + 1, wargs)
		end
	end
	if spec.moe_cache then
		local v = spec.moe_cache == true and "auto" or tostring(spec.moe_cache)
		table.move({ "--moe-cache", v }, 1, 2, #wargs + 1, wargs)
		if spec.moe_cache_zerocopy then
			wargs[#wargs + 1] = "--moe-cache-zerocopy"
		end
	end
	if stt then
		wargs[#wargs + 1] = "--stt"
	end
	if spec.engine then
		table.move({ "--engine", spec.engine }, 1, 2, #wargs + 1, wargs)
	end
	if spec.sampling then
		table.move({ "--sampling", require("lgml.json").encode(spec.sampling) }, 1, 2, #wargs + 1, wargs)
	end
	if spec.diffusion then
		table.move({ "--diffusion", require("lgml.json").encode(spec.diffusion) }, 1, 2, #wargs + 1, wargs)
	end
	local sname = require("lgml.server.fdstore").state_name(spec.id or spec.model)
	if o.names then
		o.names[sname] = true
	end
	local function start(so)
		return o.start(spec, wargs, modes, sname, so)
	end
	if managed then
		return managed(spec, modes, lora, start)
	end
	return start({})
end

--- Device memory kept out of the default budget: 1024 MiB.
M.VRAM_RESERVE = 1024 * MiB

--- Reads a number key of a config or spec.
-- @tparam table cfg the config or spec
-- @tparam string key the key
-- @param[opt] flag the command line value, which wins over the key
-- @treturn ?number the number, or nil when neither is set
-- @raise when the value is not a number
function M.number(cfg, key, flag)
	local v = flag or cfg[key]
	if v ~= nil and not tonumber(v) then
		die(("bad %s %s"):format(key, tostring(v)))
	end
	return v and tonumber(v)
end

-- A pool of models that load on demand. The gpu budget is vram, else
-- the free memory of the first GPU at startup minus VRAM_RESERVE.
local function managed_engines(path, cfg, opts, o)
	local estimate = require("lgml.estimate")
	local seen = {}
	for _, spec in ipairs(cfg.models) do
		spec.id = estimate.id(spec)
		if seen[spec.id] then
			die(("%s: two models have the id %s"):format(path, spec.id))
		end
		seen[spec.id] = true
	end
	local vram = M.number(cfg, "vram", opts.vram)
	local ram = M.number(cfg, "ram")
	local idle = M.number(cfg, "idle", opts.idle)
	local wait = M.number(cfg, "load_wait")
	local gpu
	for _, d in ipairs(o.devices and o.devices() or {}) do
		if not gpu and (d.type == "gpu" or d.type == "igpu") then
			gpu = d
		end
	end
	local budget = {
		gpu = vram and vram * MiB or (gpu and math.max(gpu.free - M.VRAM_RESERVE, 0)) or math.huge,
		cpu = ram and ram * MiB or math.huge,
	}
	local pool = require("lgml.server.pool").new({ budget = budget, idle = idle, load_wait = wait })
	L:info("budget", "gpu", budget.gpu < math.huge and ("%d MiB"):format(budget.gpu // MiB) or "none",
	    "cpu", budget.cpu < math.huge and ("%d MiB"):format(budget.cpu // MiB) or "none",
	    "idle_unload", (idle or 0) > 0 and ("after %g s"):format(idle) or "off")
	local preload = {}
	local kinds, ests = {}, {}
	local function managed(spec, modes, lora, start)
		local kind, est = kinds[spec], ests[spec]
		local mem = M.number(spec, "mem")
		local list = {}
		for i, a in ipairs(lora) do
			list[i] = { id = i - 1, path = a.path, scale = a.scale }
		end
		local m = pool:add({
			id = spec.id, modes = modes, kind = kind,
			size = mem and mem * MiB or est.bytes, fixed = mem ~= nil,
			files = est.files, margin = est.margin, extra = est.extra,
			context_length = est.n_ctx, n_ctx_train = est.n_ctx_train,
			idle = M.number(spec, "idle"),
			props = { n_ctx = est.n_ctx or 0, total_slots = 1, lora = list,
			    modalities = { vision = spec.mmproj ~= nil } },
			start = function(on_exit)
				return start({ restart = false, on_exit = on_exit })
			end,
		})
		local extra = {}
		for k, bytes in pairs(est.extra) do
			extra[#extra + 1] = ("%d MiB %s"):format(bytes // MiB, k)
		end
		L:info("model", "model", m.id, "kind", kind, "size", ("%d MiB"):format(m.size // MiB),
		    "from", mem and "mem" or "estimate", "extra", #extra > 0 and table.concat(extra, ", ") or nil,
		    "preload", spec.preload and true or nil)
		if spec.preload then
			preload[#preload + 1] = m
		end
		return m
	end
	local engines = {}
	for i, spec in ipairs(cfg.models) do
		local kind = estimate.kind(spec.backend, gpu ~= nil)
		local oke, est = pcall(estimate.model, spec, budget[kind], gpu ~= nil)
		if not oke then
			die(("%s: %s"):format(spec.id, est))
		end
		-- The worker takes the count that the estimate picked.
		if est.n_cpu_moe then
			spec.n_cpu_moe = est.n_cpu_moe
		end
		if est.moe_cache_max then
			spec.moe_cache_max = est.moe_cache_max / MiB
		end
		-- Under a budget, the worker gets the context of the estimate, not
		-- one that fits the free memory at its load.
		if not spec.ctx and est.n_ctx and budget[kind] < math.huge then
			spec.ctx = est.n_ctx
		end
		kinds[spec], ests[spec] = kind, est
		engines[i] = M.engine(spec, { managed = managed, start = o.start, names = o.names })
	end
	pool:preload(preload)
	return engines
end

--- Loads a config file: Lua that returns `{ models = { spec, ... } }`.
-- @tparam string path the file
-- @treturn table the config
-- @raise when the file does not load or has no models
function M.load(path)
	local chunk, err = loadfile(path, "t", {})
	if not chunk then
		die(err)
	end
	local ok, cfg = pcall(chunk)
	if not ok or type(cfg) ~= "table" or type(cfg.models) ~= "table" or #cfg.models == 0 then
		die(path .. ": must return { models = { ... } }")
	end
	return cfg
end

-- The router of the models of a config.
local function config_engine(path, cfg, opts, o)
	local inproc = opts.inproc
	for i, spec in ipairs(cfg.models) do
		if type(spec.model) ~= "string" then
			die(("%s: models[%d].model must be a path"):format(path, i))
		end
		-- The threads of a spec, else --threads, else the threads key.
		if spec.threads == nil then
			spec.threads = opts.t or cfg.threads
		end
		if spec.mem_margin == nil then
			spec.mem_margin = cfg.mem_margin
		end
		if spec.repeat_stop_tokens == nil then
			spec.repeat_stop_tokens = cfg.repeat_stop_tokens
		end
		local mm = spec.mem_margin
		if mm ~= nil and not (tonumber(mm) and tonumber(mm) >= 0) then
			die(("%s: bad mem_margin %s"):format(path, tostring(mm)))
		end
	end
	local engines = {}
	if inproc then
		for i, spec in ipairs(cfg.models) do
			spec.inproc = true
			engines[i] = M.engine(spec, o)
		end
	else
		engines = managed_engines(path, cfg, opts, o)
	end
	return require("lgml.server.router").new(engines), cfg
end

--- Makes the engine of the lgml-server options.
-- @tparam table opts the parsed options: `fake`, `C` (the config path),
-- or `m` with the model flags
-- @tparam table cfg the config of `C`, else an empty table
-- @tparam table o the worker hooks of `engine`, and `devices()`: the
-- ggml devices, for the default GPU budget
-- @return the engine
-- @raise for a bad config, spec or option
function M.make(opts, cfg, o)
	if opts.fake then
		return require("lgml.server.fake_engine").new({ delay = 30 })
	end
	if opts.C then
		return config_engine(opts.C, cfg, opts, o)
	end
	if opts.m then
		return M.engine({
			model = opts.m, id = opts.alias, backend = opts.b, threads = opts.t, ctx = opts.c,
			text_backend = opts["text-backend"],
			flash_attn = not opts["no-fa"], stt = opts.stt, inproc = opts.inproc,
			cache_type_k = opts.ctk, cache_type_v = opts.ctv, recipe = opts.recipe,
			cache_ram = opts["cache-ram"], ctx_checkpoints = opts["ctx-checkpoints"],
			kv_initial = opts["kv-initial"], slots = opts.parallel,
			repeat_stop_tokens = opts["repeat-stop"], reasoning_budget_tokens = opts["reasoning-budget"],
			ubatch = opts.ubatch,
			speculative = opts.spec, spec_mtp = opts["spec-mtp"],
			engine = opts.engine, lora = opts.lora, mmproj = opts.mmproj,
			image_min_tokens = opts["image-min-tokens"],
			image_max_tokens = opts["image-max-tokens"],
			cpu_moe = opts["cpu-moe"], n_cpu_moe = opts["n-cpu-moe"],
			n_gpu_layers = opts["n-gpu-layers"], override = opts["override-tensor"],
			moe_cache = opts["moe-cache"], moe_cache_profile = opts["moe-cache-profile"],
		}, o)
	end
	die("no engine: give -m model, -C config or --fake")
end


return M
