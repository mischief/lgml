--- Models that load on demand and unload by least recent use.
-- A pool holds models of several workers. Each model starts its worker
-- on the first request, and the requests that arrive during the load
-- wait for the same load. Each model has a size in bytes, and each
-- memory kind (`"gpu"` or `"cpu"`) has a budget. A model with parts on
-- more than one kind, such as an image model with its text encoder on
-- the CPU, or a chat model with weights in host memory (see
-- `lgml.placement`), counts each part against the budget of its kind. When a load
-- does not fit, the pool unloads idle models of that kind, least
-- recently used first. A model with requests in flight or a load under
-- way stays.
-- When the idle models do not free enough room, the load waits for
-- requests to end, for at most `load_wait` seconds, and then fails
-- with status 503.
--
-- Loads and unloads log at info level, budget checks at debug (see
-- `lgml.log`). Each model is an engine (see `lgml.server.engine`); a router
-- (`lgml.server.router`) picks among them by id and mode. Loads of one
-- kind run one at a time, so the worker can measure what its load took.
-- @module lgml.server.pool
local uv = require("luv")
local stream = require("lgml.server.stream")
local metrics = require("lgml.server.metrics")
local engine_api = require("lgml.server.engine")
local budget_mod = require("lgml.budget")
local log = require("lgml.log")

local L = log.new("pool")

local M = {}
local P = {}
P.__index = P
local Model = {}
Model.__index = Model

local MiB = 1048576

-- The reason of a load that stopped because its client went away.
local CANCELLED = "the client closed the request"

local function now()
	return uv.hrtime() / 1e9
end

local function mib(bytes)
	return ("%.0f MiB"):format(bytes / MiB)
end

-- Wake each function in list with the arguments, then empty it.
local function wake(list, ...)
	local fns = table.move(list, 1, #list, 1, {})
	for i = #list, 1, -1 do
		list[i] = nil
	end
	for _, f in ipairs(fns) do
		f(...)
	end
end

-- The memory parts of a model, as gauges.
local PARTS = {
	{ "weights", "weights and other leaves of the loaded model" },
	{ "cache", "KV cache of the loaded model at its full context" },
	{ "compute", "largest compute buffers of the loaded model" },
	{ "overhead", "driver memory of the worker beyond its buffers" },
}

-- Sets the gauges of model m; all 0 when it is not loaded.
local function gauges(m)
	local loaded = m.state == "loaded" and m.engine ~= nil
	metrics.set("model_loaded_bytes", "device or host bytes of the loaded model", m.id,
	    loaded and m.size or 0)
	for _, p in ipairs(PARTS) do
		local v = loaded and m.parts and m.parts[p[1]] or 0
		metrics.set("model_" .. p[1] .. "_bytes", p[2], m.id, v)
	end
end

--- Makes a pool.
-- @tparam[opt] table opts the options:
--
-- - `budget`: `{ gpu = bytes, cpu = bytes }`, the memory for each kind
--   (default: no limit)
-- - `idle`: seconds without requests before a model unloads; 0 or nil
--   turns it off (default off)
-- - `load_wait`: the most seconds a load waits for requests of other
--   models to end, when it needs their room (default 60)
-- @treturn Pool the pool
function M.new(opts)
	opts = opts or {}
	return setmetatable({
		budget = opts.budget or {}, idle = opts.idle, load_wait = opts.load_wait or 60,
		models = {}, lock = {}, events = {},
	}, P)
end

--- A pool of models.
-- @type Pool

--- Adds a model to the pool. It does not load it.
-- @tparam table o the model:
--
-- - `id`: the model id
-- - `modes`: the mode names (see `lgml.detect.MODES`)
-- - `kind`: `"gpu"` or `"cpu"`, the budget it counts against
-- - `size`: the estimated bytes that the model takes when loaded
-- - `files`: the bytes of its files; a measure below it does not count
--   (default 0)
-- - `margin`: bytes added to a measured size, for compute buffers
--   (default 0)
-- - `extra`: `{ [kind] = bytes }`, parts of the model that count against
--   the budgets of other kinds, such as a text encoder on the CPU of a
--   GPU model; the measure of the worker does not change them
-- - `fixed`: true to keep `size` and not use the measure of the worker
-- - `context_length`: the context size, when known before the load
-- - `props`: the props to give while the model is not loaded
-- - `idle`: seconds without requests before it unloads (default: the
--   pool's; 0 turns it off)
-- - `start`: a function(on_exit) that starts the worker and returns a
--   `lgml.server.remote_engine`; on_exit is its `on_exit` option
-- @treturn Model the model
function P:add(o)
	local m = setmetatable({
		pool = self, id = o.id, mode_list = o.modes, kind = o.kind or "gpu",
		estimate = o.size or 0, size = o.size or 0, files = o.files or 0,
		margin = o.margin or 0, extra = o.extra or {}, fixed = o.fixed,
		context_length = o.context_length, static_props = o.props, idle = o.idle,
		start = o.start,
		state = "unloaded", inflight = 0, wanted = 0, last = 0, waiters = {}, counted = false,
	}, Model)
	self.models[#self.models + 1] = m
	gauges(m)
	return m
end

--- Finds a model by id.
-- @tparam string id the model id
-- @treturn ?Model the model
function P:get(id)
	for _, m in ipairs(self.models) do
		if m.id == id then
			return m
		end
	end
end

--- The bytes counted against the budget of a kind.
-- Loaded models, loads under way and models whose worker has not yet
-- exited count.
-- @tparam string kind `"gpu"` or `"cpu"`
-- @treturn integer the bytes
function P:used(kind)
	local n = 0
	for _, m in ipairs(self.models) do
		if m.counted then
			n = n + m:need(kind)
		end
	end
	return n
end

-- How often a wait checks that its client is still there, in ms.
local GONE_CHECK_MS = 250

-- Waits until wake calls back, the deadline passes, or gone() turns
-- true. Returns what the callback gave, or false.
local function await(add, deadline, gone)
	local timers = {}
	local r = table.pack(stream.await(function(done)
		add(done)
		if deadline then
			local t = uv.new_timer()
			t:start(math.max(0, math.floor((deadline - now()) * 1000)), 0, function()
				done(false)
			end)
			timers[#timers + 1] = t
		end
		if gone then
			local t = uv.new_timer()
			t:start(GONE_CHECK_MS, GONE_CHECK_MS, function()
				if gone() then
					done(false)
				end
			end)
			timers[#timers + 1] = t
		end
	end))
	for _, t in ipairs(timers) do
		t:close()
	end
	return table.unpack(r, 1, r.n)
end

-- Wait for a pool event (a request end, an unload, a load end), the
-- deadline, or gone() to turn true. Returns false at the deadline.
function P:wait_event(deadline, gone)
	return await(function(done)
		self.events[#self.events + 1] = function()
			done(true)
		end
	end, deadline, gone)
end

function P:notify()
	wake(self.events)
end

-- Take the load locks of the kinds of m, in name order; loads of one
-- kind run one at a time. Returns false when gone() turns true first.
function P:take_lock(m, gone)
	for _, kind in ipairs(m:kinds()) do
		while self.lock[kind] do
			if gone and gone() then
				return false
			end
			self:wait_event(nil, gone)
		end
		self.lock[kind] = m
	end
	return true
end

-- Drops the load locks that m holds.
function P:drop_lock(m)
	for _, kind in ipairs(m:kinds()) do
		if self.lock[kind] == m then
			self.lock[kind] = nil
		end
	end
	self:notify()
end

-- The idle loaded models of a kind, least recently used first.
function P:idle_models(kind, except)
	local list = {}
	for _, m in ipairs(self.models) do
		if m ~= except and m:need(kind) > 0 and m.state == "loaded" and m.inflight == 0 and
		    m.wanted == 0 then
			list[#list + 1] = m
		end
	end
	table.sort(list, function(a, b)
		return a.last < b.last
	end)
	return list
end

-- Makes room for the part of m of one kind. Room of other kinds that
-- earlier calls made stays, since m holds their load locks. Returns
-- true, or nil, the reason and a status: 507 when m can never fit, 503
-- after the wait, 499 when gone() turns true. It then unloads nothing.
function P:reserve_kind(m, kind, deadline, gone)
	local budget = self.budget[kind] or math.huge
	local need = m:need(kind)
	L:debug("reserve", "model", m.id, "kind", kind, "need", mib(need), "budget", mib(budget),
	    "used", mib(self:used(kind)))
	if need > budget then
		return nil, ("model %s needs %s%s; the %s budget is %s"):format(m.id, mib(need),
		    m:parts_text(), kind, mib(budget)), budget_mod.STATUS
	end
	while true do
		local used = self:used(kind)
		if used + need <= budget then
			return true
		end
		if gone and gone() then
			return nil, CANCELLED, 499
		end
		local idle = self:idle_models(kind, m)
		local room = budget - used
		for _, o in ipairs(idle) do
			room = room + o:need(kind)
		end
		if room >= need then
			local o = idle[1]
			L:info("unloading", "model", o.id, "why", "least recently used", "for", m.id,
			    "kind", kind, "frees", mib(o:need(kind)), "need", mib(need), "budget", mib(budget),
			    "used", mib(used))
			o:unload()
		else
			local busy = {}
			for _, o in ipairs(self.models) do
				if o ~= m and o:need(kind) > 0 and o.counted and
				    (o.inflight > 0 or o.wanted > 0 or o.state ~= "loaded") then
					busy[#busy + 1] = o.id
				end
			end
			local why = ("model %s needs %s; the %s budget is %s, %s in use"):format(m.id,
			    mib(need), kind, mib(budget), mib(used))
			if #busy == 0 then
				return nil, why, budget_mod.STATUS
			end
			L:debug("waiting for room", "model", m.id, "kind", kind, "busy", table.concat(busy, ","))
			if gone and gone() then
				return nil, CANCELLED, 499
			end
			if now() >= deadline or not self:wait_event(deadline, gone) then
				if gone and gone() then
					return nil, CANCELLED, 499
				end
				return nil, ("%s; models with requests in flight: %s; waited %g s"):format(why,
				    table.concat(busy, ", "), self.load_wait), 503
			end
		end
	end
end

-- Makes room for m in the budget of each of its kinds and counts it. It
-- holds the load locks of m. Returns true, or nil, the reason and a
-- status (see reserve_kind).
function P:reserve(m, gone)
	local deadline = now() + self.load_wait
	for _, kind in ipairs(m:kinds()) do
		local ok, err, status = self:reserve_kind(m, kind, deadline, gone)
		if not ok then
			return nil, err, status
		end
	end
	m.counted = true
	return true
end

--- Loads the models with preload, in order.
-- It runs in a new coroutine and returns at once.
-- @tparam {Model,...} list the models to load
function P:preload(list)
	coroutine.wrap(function()
		for _, m in ipairs(list) do
			local ok, err = m:load()
			if not ok then
				L:error("preload failed", "model", m.id, "err", err)
			end
		end
	end)()
end

--- Stops the worker of each model. No model loads after this.
function P:stop()
	self.stopping = true
	for _, m in ipairs(self.models) do
		m:stop()
	end
end

--- A model of the pool. It implements `lgml.server.engine.Engine`.
-- Each request method loads the model when it is not loaded, and
-- counts the request as in flight while it runs. The field `state` is
-- `"unloaded"`, `"loading"`, `"loaded"` or `"unloading"`. A model whose
-- worker crashed lists as `"failed"`, and requests fail until a load
-- that `load` without hold starts, such as `POST /models/load`.
-- @type Model

--- Lists the one model with its load state.
-- It does not load the model.
-- @treturn {table,...} `{ { id, context_length, capabilities, status,
-- bytes, memory, vision } }`; status is the state, bytes the size counted
-- against the budget, and memory the parts that the worker measured:
-- `{ weights, cache, compute, overhead }` in bytes, while loaded
function Model:models()
	local ctx = self.context_length
	local h = self.state == "loaded" and self.engine and self.engine.hello
	local p = self.parts
	local vision
	for _, x in ipairs(self.mode_list) do
		vision = vision or x == "multimodal"
	end
	if h and h.models and h.models[1] then
		ctx = h.models[1].context_length or ctx
	end
	return { {
		id = self.id, context_length = ctx, capabilities = self.mode_list,
		status = self.failed and self.state == "unloaded" and "failed" or self.state,
		bytes = self.size, vision = vision,
		memory = self.state == "loaded" and p and { weights = p.weights, cache = p.cache,
		    compute = p.compute, overhead = p.overhead } or nil,
	} }
end

--- Lists the modes of the model.
-- @treturn {string,...} the mode names
function Model:modes()
	return self.mode_list
end

--- Returns the properties. It does not load the model.
-- The `lora` field has the global scales that `lora_adapters` set.
-- @treturn table the worker's props when loaded, else the static props
function Model:props()
	local p
	if self.state == "loaded" and self.engine and self.engine.hello then
		p = self.engine.hello.props
	else
		p = self.static_props or { n_ctx = self.context_length or 0, total_slots = 1 }
	end
	if not self.lora then
		return p
	end
	local t = {}
	for k, v in pairs(p) do
		t[k] = v
	end
	t.lora = engine_api.lora_view(p.lora, self.lora)
	return t
end

--- Sets the global LoRA scales (see `lgml.server.engine`).
-- It does not load the model. The model keeps them across loads, and
-- sends them with each chat and decide request that has no `lora`.
-- @tparam {table,...} list `{ id, scale }` entries
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int 400
function Model:lora_adapters(list)
	local scales, err = engine_api.lora_scales(list, #(self:props().lora or {}))
	if not scales then
		return nil, err, 400
	end
	self.lora = scales
	return true
end

function Model:set_state(s)
	self.state = s
	self.pool:notify()
end

--- The bytes of the model that count against the budget of a kind.
-- @tparam string kind `"gpu"` or `"cpu"`
-- @treturn integer the bytes
function Model:need(kind)
	return (self.kind == kind and self.size or 0) + (self.extra[kind] or 0)
end

--- The kinds whose budgets the model counts against, in name order.
-- @treturn {string,...} the kinds
function Model:kinds()
	local list = { self.kind }
	for kind, bytes in pairs(self.extra) do
		if kind ~= self.kind and bytes > 0 then
			list[#list + 1] = kind
		end
	end
	table.sort(list)
	return list
end

-- The measured parts for a message, as " (weights 1 MiB, ...)", or "".
function Model:parts_text()
	local p = self.parts
	if not p then
		return ""
	end
	return (" (weights %s, cache %s, compute %s, overhead %s)"):format(mib(p.weights),
	    mib(p.cache), mib(p.compute), mib(p.overhead))
end

-- The size for a log line, with the part of each kind when it has more
-- than one.
function Model:describe()
	local kinds = self:kinds()
	if #kinds == 1 then
		return mib(self:need(kinds[1]))
	end
	local parts = {}
	for i, kind in ipairs(kinds) do
		parts[i] = mib(self:need(kind)) .. " " .. kind
	end
	return table.concat(parts, ", ")
end

-- Start the idle timer when the model has nothing in flight.
function Model:arm_idle()
	local idle = self.idle
	if idle == nil then
		idle = self.pool.idle
	end
	if not idle or idle <= 0 or self.state ~= "loaded" or self.inflight > 0 then
		return
	end
	self.idle_timer = self.idle_timer or uv.new_timer()
	self.idle_timer:start(math.floor(idle * 1000), 0, function()
		if self.state == "loaded" and self.inflight == 0 and now() - self.last >= idle - 0.001 then
			coroutine.wrap(function()
				L:info("unloading", "model", self.id, "why", "idle", "idle_s", idle)
				self:unload()
			end)()
		end
	end)
end

-- The worker exited. A loaded model goes back to unloaded.
function Model:exited(eng, code, sig)
	if self.engine ~= eng or self.state ~= "loaded" then
		return
	end
	if (code or 0) ~= 0 or (sig or 0) ~= 0 then
		-- A crash, such as a GPU out of memory, would repeat on the next
		-- request: requests fail until a load on purpose.
		self.failed = ("the worker of model %s crashed (code %s, signal %s); see the log, then POST /models/load to load it again")
		    :format(self.id, code, sig)
		L:error("worker crashed; model failed", "model", self.id, "code", code, "signal", sig)
	else
		L:warn("worker exited; unloaded", "model", self.id)
	end
	self.engine, self.counted = nil, false
	metrics.add("model_unloads_total", "model unloads", self.id, 1)
	self:set_state("unloaded")
	gauges(self)
end

--- Loads the model, or waits for a load under way.
-- It yields the calling coroutine. With hold, the model counts one
-- request in flight on success; call `release` after it.
-- @tparam[opt] boolean hold true to hold the model for a request
-- @tparam[opt] function gone returns true when the client went away;
-- the wait then ends, and a load not started yet does not start
-- @treturn[1] lgml.server.remote_engine.Remote the engine
-- @treturn[2] nil
-- @treturn[2] string the reason of a failed load
-- @treturn[2] int 503; 507 when the model does not fit the memory
-- budget or the device; 499 when the client went away
function Model:load(hold, gone)
	-- A model with a caller in here is not idle: a load wakes other
	-- waiters first, and they must not unload it before this caller runs.
	self.wanted = self.wanted + 1
	local eng, err, code = self:load_loop(hold, gone)
	self.wanted = self.wanted - 1
	if self.wanted == 0 then
		self.pool:notify()
	end
	return eng, err, code
end

function Model:load_loop(hold, gone)
	while true do
		if self.pool.stopping then
			return nil, "server is stopping", 503
		elseif gone and gone() then
			return nil, CANCELLED, 499
		elseif hold and self.failed and self.state == "unloaded" then
			return nil, self.failed, 503
		elseif self.state == "loaded" then
			if hold then
				self.inflight = self.inflight + 1
				self.last = now()
				if self.idle_timer then
					self.idle_timer:stop()
				end
			end
			return self.engine
		elseif self.state == "loading" or self.state == "unloading" then
			local ok, err, status = await(function(done)
				self.waiters[#self.waiters + 1] = done
			end, nil, gone)
			if not ok and err then
				return nil, err, status or 503
			end
		else
			-- A load on purpose (no hold) clears a crash.
			if not hold then
				self.failed = nil
			end
			local ok, err, status = self:do_load(gone)
			if not ok then
				return nil, err, status
			end
		end
	end
end

-- The part of the device delta that a sum may miss before a warning:
-- 256 MiB, or a tenth of the sum.
local DELTA_SLACK = 256 * MiB

-- Takes the memory report of a loaded worker. An exact report sets the
-- size to its sum; a report with only a device delta sets it to the
-- delta, at least the file size, plus the margin.
function Model:measure(mem)
	local used = mem and tonumber(mem.used)
	if not used then
		return
	end
	self.parts = mem.exact and mem or nil
	if mem.exact and mem.delta and mem.now and
	    math.abs(mem.delta - mem.now) > math.max(DELTA_SLACK, mem.now // 10) then
		L:warn("device delta differs from the buffer sum", "model", self.id,
		    "delta", mib(mem.delta), "held", mib(mem.now))
	end
	if self.fixed then
		return
	end
	if mem.exact then
		-- Without its worst-case graphs, the compute part is a guess.
		self.measured = used + (mem.reserved and 0 or self.margin)
	else
		self.measured = math.max(used, self.files) + self.margin
	end
	self.size = self.measured
	return used
end

-- Counts the measured size of a loaded model against the budgets: a
-- larger size than the one reserved unloads idle models, or fails.
-- Returns true, or nil, the reason and a status.
function Model:recount(gone)
	local pool = self.pool
	self.counted = false
	local ok, err, status = pool:reserve(self, gone)
	self.counted = true
	return ok, err, status
end

-- Runs one load. Wakes the waiters with true, or with false, the reason
-- and the status. Returns the same.
function Model:do_load(gone)
	local pool = self.pool
	self:set_state("loading")
	local t0 = now()
	local ok, err, status = pool:take_lock(self, gone), CANCELLED, 499
	-- A failed load keeps its measure out: start from the estimate.
	if ok and not self.fixed then
		self.size = self.measured or self.estimate
	end
	if ok then
		ok, err, status = pool:reserve(self, gone)
	end
	if err == CANCELLED and not ok then
		-- Other waiters retry the load themselves.
		self.counted = false
		pool:drop_lock(self)
		self:set_state("unloaded")
		L:info("load cancelled", "model", self.id)
		wake(self.waiters, false)
		return nil, err, 499
	end
	local eng
	if ok then
		L:info("loading", "model", self.id, "size", self:describe(),
		    "from", self.measured and "measure" or (self.fixed and "mem" or "estimate"))
		ok, eng = pcall(self.start, function(code, sig)
			self:exited(eng, code, sig)
		end)
		if not ok then
			err, status = tostring(eng), 503
		else
			ok, err = eng:ready()
			status = budget_mod.status(err, 503)
			if ok then
				local before = self.size
				self:measure(eng.hello.mem)
				if self.size > before then
					ok, err, status = self:recount(gone)
				end
			end
			if not ok then
				eng:shutdown()
			end
		end
	end
	if not ok then
		self.counted = false
		pool:drop_lock(self)
		self:set_state("unloaded")
		metrics.add("model_load_failures_total", "model loads that failed", self.id, 1)
		err = ("model %s did not load: %s"):format(self.id, tostring(err))
		L:error("load failed", "model", self.id, "err", err, "status", status)
		wake(self.waiters, false, err, status)
		return nil, err, status
	end
	local secs = now() - t0
	local mem = eng.hello.mem or {}
	L:info("loaded", "model", self.id, "s", secs, "size", self:describe(),
	    "estimate", mib(self.estimate), "measured", mem.used and mib(mem.used) or nil,
	    "weights", mem.exact and mib(mem.weights) or nil, "cache", mem.exact and mib(mem.cache) or nil,
	    "compute", mem.exact and mib(mem.compute) or nil,
	    "overhead", mem.exact and mib(mem.overhead) or nil)
	self.engine, self.last = eng, now()
	self.state = "loaded"
	metrics.add("model_loads_total", "model loads", self.id, 1)
	metrics.add("model_load_seconds_total", "time spent loading models", self.id, secs)
	gauges(self)
	pool:drop_lock(self)
	wake(self.waiters, true)
	self:arm_idle()
	return true
end

--- Ends a request that `load(true)` held.
function Model:release()
	self.inflight = self.inflight - 1
	self.last = now()
	self:arm_idle()
	self.pool:notify()
end

--- Unloads the model and waits until its worker exits.
-- It yields the calling coroutine.
-- @treturn[1] boolean true, also when the model was not loaded
-- @treturn[2] nil
-- @treturn[2] string the reason: requests in flight, or a load under way
-- @treturn[2] int 409
function Model:unload()
	if self.state == "unloaded" then
		return true
	elseif self.state == "unloading" then
		stream.await(function(done)
			self.waiters[#self.waiters + 1] = done
		end)
		return true
	elseif self.state == "loading" then
		return nil, ("model %s is loading"):format(self.id), 409
	elseif self.inflight > 0 then
		return nil, ("model %s has %d requests in flight"):format(self.id, self.inflight), 409
	end
	if self.idle_timer then
		self.idle_timer:stop()
	end
	local eng = self.engine
	self:set_state("unloading")
	eng:shutdown()
	self.engine, self.counted = nil, false
	metrics.add("model_unloads_total", "model unloads", self.id, 1)
	self:set_state("unloaded")
	gauges(self)
	wake(self.waiters, false)
	return true
end

--- Stops the worker without waiting.
function Model:stop()
	if self.idle_timer then
		self.idle_timer:stop()
		self.idle_timer:close()
		self.idle_timer = nil
	end
	if self.engine then
		self.engine:stop()
	end
end

-- Hold the model for a request, run method on its engine, release.
-- gone tells if the client went away while the model loads.
local function run(self, gone, method, ...)
	local eng, msg, status = self:load(true, gone)
	if not eng then
		return nil, msg, status
	end
	local r = table.pack(pcall(eng[method], eng, ...))
	self:release()
	if not r[1] then
		error(r[2], 0)
	end
	return table.unpack(r, 2, r.n)
end

--- Runs a chat completion. See `lgml.server.engine`.
-- @tparam table req the chat request
-- @tparam lgml.server.engine.Sink sink the reply
-- @return the results of `Engine:chat`, or nil, a message and 503
-- when the model does not load
function Model:chat(req, sink)
	return run(self, sink.cancelled and function() return sink:cancelled() end, "chat", engine_api.with_lora(req, self.lora), sink)
end

--- Transcribes speech. See `lgml.server.engine`.
-- @tparam table req the request
-- @tparam string audio the file bytes
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:transcribe`, or nil, a message and 503
function Model:transcribe(req, audio, ctl)
	return run(self, ctl and ctl.cancelled, "transcribe", req, audio, ctl)
end

--- Makes speech. See `lgml.server.engine`.
-- @tparam table req the request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:speak`, or nil, a message and 503
function Model:speak(req, ctl)
	return run(self, ctl and ctl.cancelled, "speak", req, ctl)
end

--- Answers typed questions. See `lgml.server.engine`.
-- @tparam table req the decision request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:decide`, or nil, a message and 503
function Model:decide(req, ctl)
	return run(self, ctl and ctl.cancelled, "decide", engine_api.with_lora(req, self.lora), ctl)
end

--- Makes images. See `lgml.server.engine`.
-- @tparam table req the image request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:generate`, or nil, a message and 503
function Model:generate(req, ctl)
	return run(self, ctl and ctl.cancelled, "generate", req, ctl)
end

--- Embeds text. See `lgml.server.engine`.
-- @tparam table req the embedding request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:embed`, or nil, a message and 503
function Model:embed(req, ctl)
	return run(self, ctl and ctl.cancelled, "embed", req, ctl)
end

return M
