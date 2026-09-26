--- Models that load on demand and unload by least recent use.
-- A pool holds models of several workers. Each model starts its worker
-- on the first request, and the requests that arrive during the load
-- wait for the same load. Each model has a size in bytes, and each
-- memory kind (`"gpu"` or `"cpu"`) has a budget. A model with parts on
-- more than one kind, such as an image model with its text encoder on
-- the CPU, counts each part against the budget of its kind. When a load
-- does not fit, the pool unloads idle models of that kind, least
-- recently used first. A model with requests in flight or a load under
-- way stays.
-- When the idle models do not free enough room, the load waits for
-- requests to end, for at most `load_wait` seconds, and then fails
-- with status 503.
--
-- Each model is an engine (see `lgml.server.engine`); a router
-- (`lgml.server.router`) picks among them by id and mode. Loads of one
-- kind run one at a time, so the worker can measure what its load took.
-- @module lgml.server.pool
local uv = require("luv")
local stream = require("lgml.server.stream")
local metrics = require("lgml.server.metrics")

local M = {}
local P = {}
P.__index = P
local Model = {}
Model.__index = Model

local MiB = 1048576

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

--- Makes a pool.
-- @tparam[opt] table opts the options:
--
-- - `budget`: `{ gpu = bytes, cpu = bytes }`, the memory for each kind
--   (default: no limit)
-- - `idle`: seconds without requests before a model unloads; 0 or nil
--   turns it off (default off)
-- - `load_wait`: the most seconds a load waits for requests of other
--   models to end, when it needs their room (default 60)
-- - `log`: a function for messages (default: write to stderr)
-- @treturn Pool the pool
function M.new(opts)
	opts = opts or {}
	return setmetatable({
		budget = opts.budget or {}, idle = opts.idle, load_wait = opts.load_wait or 60,
		models = {}, lock = {}, events = {},
		log = opts.log or function(msg)
			io.stderr:write("lgml-server: ", msg, "\n")
		end,
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
		state = "unloaded", inflight = 0, last = 0, waiters = {}, counted = false,
	}, Model)
	self.models[#self.models + 1] = m
	metrics.set("model_loaded_bytes", "device or host bytes of the loaded model", m.id, 0)
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

-- Wait for a pool event (a request end, an unload, a load end) or the
-- deadline. Returns false at the deadline.
function P:wait_event(deadline)
	local t
	local ok = stream.await(function(done)
		self.events[#self.events + 1] = function()
			done(true)
		end
		if deadline then
			t = uv.new_timer()
			t:start(math.max(0, math.floor((deadline - now()) * 1000)), 0, function()
				done(false)
			end)
		end
	end)
	if t then
		t:close()
	end
	return ok
end

function P:notify()
	wake(self.events)
end

-- Take the load locks of the kinds of m, in name order; loads of one
-- kind run one at a time.
function P:take_lock(m)
	for _, kind in ipairs(m:kinds()) do
		while self.lock[kind] do
			self:wait_event()
		end
		self.lock[kind] = true
	end
end

function P:drop_lock(m)
	for _, kind in ipairs(m:kinds()) do
		self.lock[kind] = nil
	end
	self:notify()
end

-- The idle loaded models of a kind, least recently used first.
function P:idle_models(kind, except)
	local list = {}
	for _, m in ipairs(self.models) do
		if m ~= except and m:need(kind) > 0 and m.state == "loaded" and m.inflight == 0 then
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
-- true, or nil and the reason.
function P:reserve_kind(m, kind, deadline)
	local budget = self.budget[kind] or math.huge
	local need = m:need(kind)
	if need > budget then
		return nil, ("model %s needs %s; the %s budget is %s"):format(m.id, mib(need),
		    kind, mib(budget))
	end
	while true do
		local used = self:used(kind)
		if used + need <= budget then
			return true
		end
		local idle = self:idle_models(kind, m)
		local room = budget - used
		for _, o in ipairs(idle) do
			room = room + o:need(kind)
		end
		if room >= need then
			local o = idle[1]
			self.log(("unloading %s (least recently used) to load %s"):format(o.id, m.id))
			o:unload()
		else
			local busy = {}
			for _, o in ipairs(self.models) do
				if o ~= m and o:need(kind) > 0 and o.counted and
				    (o.inflight > 0 or o.state ~= "loaded") then
					busy[#busy + 1] = o.id
				end
			end
			local why = ("model %s needs %s; the %s budget is %s, %s in use"):format(m.id,
			    mib(need), kind, mib(budget), mib(used))
			if #busy == 0 then
				return nil, why
			elseif now() >= deadline or not self:wait_event(deadline) then
				return nil, ("%s; models with requests in flight: %s; waited %g s"):format(why,
				    table.concat(busy, ", "), self.load_wait)
			end
		end
	end
end

-- Makes room for m in the budget of each of its kinds and counts it. It
-- holds the load locks of m. Returns true, or nil and the reason.
function P:reserve(m)
	local deadline = now() + self.load_wait
	for _, kind in ipairs(m:kinds()) do
		local ok, err = self:reserve_kind(m, kind, deadline)
		if not ok then
			return nil, err
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
				self.log(("preload %s: %s"):format(m.id, err))
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
-- `"unloaded"`, `"loading"`, `"loaded"` or `"unloading"`.
-- @type Model

--- Lists the one model with its load state.
-- It does not load the model.
-- @treturn {table,...} `{ { id, context_length, capabilities, status,
-- bytes, vision } }`; status is the state, and bytes the size counted
-- against the budget
function Model:models()
	local ctx = self.context_length
	local h = self.state == "loaded" and self.engine and self.engine.hello
	local vision
	for _, x in ipairs(self.mode_list) do
		vision = vision or x == "multimodal"
	end
	if h and h.models and h.models[1] then
		ctx = h.models[1].context_length or ctx
	end
	return { {
		id = self.id, context_length = ctx, capabilities = self.mode_list, status = self.state,
		bytes = self.size, vision = vision,
	} }
end

--- Lists the modes of the model.
-- @treturn {string,...} the mode names
function Model:modes()
	return self.mode_list
end

--- Returns the properties. It does not load the model.
-- @treturn table the worker's props when loaded, else the static props
function Model:props()
	if self.state == "loaded" and self.engine and self.engine.hello then
		return self.engine.hello.props
	end
	return self.static_props or { n_ctx = self.context_length or 0, total_slots = 1 }
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
				self.pool.log(("unloading %s: idle for %g s"):format(self.id, idle))
				self:unload()
			end)()
		end
	end)
end

-- The worker exited. A loaded model goes back to unloaded.
function Model:exited(eng)
	if self.engine ~= eng or self.state ~= "loaded" then
		return
	end
	self.pool.log(("model %s: worker exited; unloaded"):format(self.id))
	self.engine, self.counted = nil, false
	metrics.add("model_unloads_total", "model unloads", self.id, 1)
	metrics.set("model_loaded_bytes", "device or host bytes of the loaded model", self.id, 0)
	self:set_state("unloaded")
end

--- Loads the model, or waits for a load under way.
-- It yields the calling coroutine. With hold, the model counts one
-- request in flight on success; call `release` after it.
-- @tparam[opt] boolean hold true to hold the model for a request
-- @treturn[1] lgml.server.remote_engine.Remote the engine
-- @treturn[2] nil
-- @treturn[2] string the reason of a failed load
-- @treturn[2] int 503
function Model:load(hold)
	while true do
		if self.pool.stopping then
			return nil, "server is stopping", 503
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
			local ok, err = stream.await(function(done)
				self.waiters[#self.waiters + 1] = done
			end)
			if not ok and err then
				return nil, err, 503
			end
		else
			local ok, err = self:do_load()
			if not ok then
				return nil, err, 503
			end
		end
	end
end

-- Runs one load. Wakes the waiters with true, or with false and the
-- reason.
function Model:do_load()
	local pool = self.pool
	self:set_state("loading")
	local t0 = now()
	pool:take_lock(self)
	-- A failed load keeps its measure out: start from the estimate.
	if not self.fixed then
		self.size = self.measured or self.estimate
	end
	local ok, err = pool:reserve(self)
	local eng
	if ok then
		pool.log(("loading %s (%s)"):format(self.id, self:describe()))
		ok, eng = pcall(self.start, function()
			self:exited(eng)
		end)
		if not ok then
			err = tostring(eng)
		else
			ok, err = eng:ready()
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
		pool.log(err)
		wake(self.waiters, false, err)
		return nil, err
	end
	local used = eng.hello.mem and tonumber(eng.hello.mem.used)
	if used and not self.fixed then
		self.measured = math.max(used, self.files) + self.margin
		self.size = self.measured
	end
	local secs = now() - t0
	pool.log(("loaded %s in %.2f s (%s)"):format(self.id, secs, self:describe()))
	self.engine, self.last = eng, now()
	self.state = "loaded"
	metrics.add("model_loads_total", "model loads", self.id, 1)
	metrics.add("model_load_seconds_total", "time spent loading models", self.id, secs)
	metrics.set("model_loaded_bytes", "device or host bytes of the loaded model", self.id, self.size)
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
	metrics.set("model_loaded_bytes", "device or host bytes of the loaded model", self.id, 0)
	self:set_state("unloaded")
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
local function run(self, method, ...)
	local eng, msg, status = self:load(true)
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
	return run(self, "chat", req, sink)
end

--- Transcribes speech. See `lgml.server.engine`.
-- @tparam table req the request
-- @tparam string audio the file bytes
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:transcribe`, or nil, a message and 503
function Model:transcribe(req, audio, ctl)
	return run(self, "transcribe", req, audio, ctl)
end

--- Makes speech. See `lgml.server.engine`.
-- @tparam table req the request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:speak`, or nil, a message and 503
function Model:speak(req, ctl)
	return run(self, "speak", req, ctl)
end

--- Answers typed questions. See `lgml.server.engine`.
-- @tparam table req the decision request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:decide`, or nil, a message and 503
function Model:decide(req, ctl)
	return run(self, "decide", req, ctl)
end

--- Makes images. See `lgml.server.engine`.
-- @tparam table req the image request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:generate`, or nil, a message and 503
function Model:generate(req, ctl)
	return run(self, "generate", req, ctl)
end

return M
