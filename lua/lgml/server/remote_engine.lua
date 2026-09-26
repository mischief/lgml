--- An engine that runs in a worker process (`lgml-worker`).
-- It speaks imsg over a socketpair; the messages are in `lgml.ipc`.
-- When the worker dies, open jobs fail with 503 and a new worker starts
-- after one second. The front polls for client cancel every 250 ms.
-- @module lgml.server.remote_engine
local uv = require("luv")
local ipc = require("lgml.ipc")
local stream = require("lgml.server.stream")
local detect = require("lgml.detect")

local M = {}
local R = {}
R.__index = R

local CANCEL_POLL_MS = 250
local RESTART_MS = 1000

--- Starts a worker and makes an engine for it.
-- The worker gets `--fd 3` and args, and the `LUA_PATH` and `LUA_CPATH`
-- of this process.
-- @tparam table opts the options:
--
-- - `lua`: the Lua interpreter path
-- - `script`: the worker script path
-- - `args`: a list of worker arguments
-- - `log`: a function for messages (default: write to stderr)
-- - `modes`: the mode names of the model (default: from the worker's
--   `HELLO`)
-- @treturn Remote the engine
-- @raise when the worker does not start
function M.new(opts)
	local r = setmetatable({
		opts = opts, jobs = {}, next_id = 1, hello = nil, waiting = {},
		log = opts.log or function(msg)
			io.stderr:write("lgml-server: ", msg, "\n")
		end,
	}, R)
	r:start()
	return r
end

local function environ(extra)
	local env = {}
	for k, v in pairs(uv.os_environ()) do
		if extra[k] == nil then
			env[#env + 1] = k .. "=" .. v
		end
	end
	for k, v in pairs(extra) do
		env[#env + 1] = k .. "=" .. v
	end
	return env
end

--- A worker engine. It implements `lgml.server.engine.Engine`.
-- It has `chat`, `transcribe`, `speak`, `decide` and `generate`. The
-- worker gives the methods that its engine has in `ready().caps`.
-- @type Remote

function R:start()
	local o = self.opts
	local pair = assert(uv.socketpair("stream", 0, { nonblock = true }, {}))
	local args = { o.script, "--fd", "3" }
	table.move(o.args, 1, #o.args, #args + 1, args)
	local proc, pid = uv.spawn(o.lua, {
		args = args, stdio = { 0, 1, 2, pair[2] },
		env = environ({ LUA_PATH = package.path, LUA_CPATH = package.cpath }),
	}, function(code, sig)
		self:exited(code, sig)
	end)
	if not proc then
		error("spawn worker: " .. tostring(pid))
	end
	self.proc = proc
	-- The child has its end now.
	uv.fs_close(pair[2])
	self.ch = ipc.channel(pair[1], function(t, id, v)
		self:message(t, id, v)
	end, function(err)
		self:lost(err)
	end)
end

function R:exited(code, sig)
	self.proc:close()
	self.proc = nil
	self.log(("worker exited, code %s signal %s"):format(code, sig))
	if self.ch then
		self.ch:close("exit")
	end
	if not self.stopping then
		local t = uv.new_timer()
		t:start(RESTART_MS, 0, function()
			t:close()
			self:start()
		end)
	end
end

function R:lost(err)
	self.ch = nil
	for id, job in pairs(self.jobs) do
		self.jobs[id] = nil
		job:push({ t = ipc.DONE, v = { msg = "worker died: " .. tostring(err), status = 503 } })
	end
end

function R:message(t, id, v)
	if t == ipc.HELLO then
		self.hello = v
		for _, w in ipairs(self.waiting) do
			w()
		end
		self.waiting = {}
		return
	end
	local job = self.jobs[id]
	if job then
		if t == ipc.DONE then
			self.jobs[id] = nil
		end
		job:push({ t = t, v = v })
	end
end

--- Waits until the worker has sent `HELLO`.
-- It yields in a coroutine. Outside a coroutine, it runs the luv loop.
-- @treturn table the `HELLO` payload: `{ models, props, caps }`
function R:ready()
	while not self.hello do
		if coroutine.isyieldable() then
			stream.await(function(done)
				self.waiting[#self.waiting + 1] = done
			end)
		else
			uv.run("once")
		end
	end
	return self.hello
end

--- Lists the models of the worker engine.
-- @treturn {table,...} the model list
function R:models()
	return self:ready().models
end

--- Lists the modes of the worker's model.
-- @treturn {string,...} `opts.modes`, else the modes from the worker's
-- `HELLO` (see `lgml.detect.modes_of`)
function R:modes()
	if self.opts.modes then
		return self.opts.modes
	end
	local h = self:ready()
	local m = h.models[1]
	return detect.modes_of(h.caps or {}, m and m.vision)
end

--- Returns the properties of the worker engine.
-- @treturn table the properties
function R:props()
	return self:ready().props
end

--- Sends SIGTERM to the worker. No new worker starts.
function R:stop()
	self.stopping = true
	if self.proc then
		self.proc:kill("sigterm")
	end
end

local Job = {}
Job.__index = Job

function Job:push(ev)
	self.q[#self.q + 1] = ev
	if self.wake then
		local w = self.wake
		self.wake = nil
		w()
	end
end

-- Wait for events or the cancel poll timer.
function Job:wait()
	if #self.q > 0 then
		return
	end
	local t = uv.new_timer()
	stream.await(function(done)
		self.wake = done
		t:start(CANCEL_POLL_MS, 0, function()
			if self.wake == done then
				self.wake = nil
				done()
			end
		end)
	end)
	t:close()
end

local function forward(sink, map, v)
	local k = v.k
	if k == "reasoning" then
		return sink:reasoning(v.text)
	elseif k == "content" then
		return sink:content(v.text)
	elseif k == "tool_call" then
		local idx = sink:tool_call(v.name, v.id)
		map[v.index] = idx
		return idx
	elseif k == "tool_args" then
		return sink:tool_arguments(map[v.index], v.text)
	end
	return true
end

--- Runs a chat job on the worker.
-- A cancel sends `CANCEL` and returns true after `DONE`.
-- @tparam table req the chat request
-- @tparam lgml.server.engine.Sink sink the reply
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function R:chat(req, sink)
	self:ready()
	if not self.ch then
		return nil, "worker not running", 503
	end
	local id = self.next_id
	self.next_id = id + 1
	local job = setmetatable({ q = {} }, Job)
	self.jobs[id] = job
	self.ch:send(ipc.JOB, id, req)
	local map, cancelled = {}, false
	local function cancel()
		if not cancelled then
			cancelled = true
			if self.ch then
				self.ch:send(ipc.CANCEL, id)
			end
		end
	end
	while true do
		job:wait()
		if sink:cancelled() then
			cancel()
		end
		while #job.q > 0 do
			local ev = table.remove(job.q, 1)
			if ev.t == ipc.DONE then
				local d = ev.v
				if d.ok and d.finish and not cancelled then
					sink:finish(d.finish.reason, d.finish.usage, d.finish.timings)
				end
				if d.ok or cancelled then
					return true
				end
				return nil, d.msg, d.status
			elseif not cancelled and not forward(sink, map, ev.v) then
				cancel()
			end
		end
	end
end

-- Send a transcription job: the request as JSON, then the raw audio.
function R:send_transcribe(id, req, audio)
	local ok, err = self.ch:send(ipc.TRANSCRIBE, id, req)
	if ok then
		ok, err = self.ch:send(ipc.AUDIO, id, audio)
	end
	if not ok then
		self.jobs[id] = nil
		return nil, "send to worker: " .. tostring(err), err == "message too large" and 413 or 503
	end
	return true
end

--- Runs a transcription job on the worker (see `lgml.stt_engine`).
-- Audio too large for one message gives 413.
-- @tparam table req the request
-- @tparam string audio the file bytes
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table `{ text, language, duration, segments }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function R:transcribe(req, audio, ctl)
	local hello = self:ready()
	if hello.caps and not hello.caps.transcribe then
		return nil, "model does not support transcription", 400
	end
	if not self.ch then
		return nil, "worker not running", 503
	end
	local id = self.next_id
	self.next_id = id + 1
	local job = setmetatable({ q = {} }, Job)
	self.jobs[id] = job
	local ok, msg, status = self:send_transcribe(id, req, audio)
	if not ok then
		return nil, msg, status
	end
	local cancelled = false
	while true do
		job:wait()
		if not cancelled and ctl and ctl.cancelled and ctl.cancelled() then
			cancelled = true
			if self.ch then
				self.ch:send(ipc.CANCEL, id)
			end
		end
		while #job.q > 0 do
			local ev = table.remove(job.q, 1)
			if ev.t == ipc.DONE then
				if ev.v.ok then
					return ev.v.result
				end
				return nil, ev.v.msg, ev.v.status
			end
		end
	end
end

--- Runs a speech job on the worker.
-- @tparam table req the request
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table `{ wav = bytes, sample_rate }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function R:speak(req, ctl)
	local hello = self:ready()
	if hello.caps and not hello.caps.speak then
		return nil, "model does not support speech", 400
	end
	if not self.ch then
		return nil, "worker not running", 503
	end
	local id = self.next_id
	self.next_id = id + 1
	local job = setmetatable({ q = {} }, Job)
	self.jobs[id] = job
	local ok, err = self.ch:send(ipc.SPEAK, id, req)
	if not ok then
		self.jobs[id] = nil
		return nil, "send to worker: " .. tostring(err), 503
	end
	local wav, cancelled = nil, false
	while true do
		job:wait()
		if not cancelled and ctl and ctl.cancelled and ctl.cancelled() then
			cancelled = true
			if self.ch then
				self.ch:send(ipc.CANCEL, id)
			end
		end
		while #job.q > 0 do
			local ev = table.remove(job.q, 1)
			if ev.t == ipc.AUDIO then
				wav = ev.v
			elseif ev.t == ipc.DONE then
				if ev.v.ok and wav then
					local r = ev.v.result or {}
					r.wav = wav
					return r
				end
				return nil, ev.v.msg or "no audio", ev.v.status
			end
		end
	end
end

--- Runs a decide job on the worker.
-- @tparam table req the decision request
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table the result of `Engine:decide`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function R:decide(req, ctl)
	local hello = self:ready()
	if hello.caps and not hello.caps.decide then
		return nil, "model does not support decisions", 400
	end
	if not self.ch then
		return nil, "worker not running", 503
	end
	local id = self.next_id
	self.next_id = id + 1
	local job = setmetatable({ q = {} }, Job)
	self.jobs[id] = job
	local ok, err = self.ch:send(ipc.DECIDE, id, req)
	if not ok then
		self.jobs[id] = nil
		return nil, "send to worker: " .. tostring(err), 503
	end
	local cancelled = false
	while true do
		job:wait()
		if not cancelled and ctl and ctl.cancelled and ctl.cancelled() then
			cancelled = true
			if self.ch then
				self.ch:send(ipc.CANCEL, id)
			end
		end
		while #job.q > 0 do
			local ev = table.remove(job.q, 1)
			if ev.t == ipc.DONE then
				if ev.v.ok then
					return ev.v.result
				end
				return nil, ev.v.msg, ev.v.status
			end
		end
	end
end

--- Runs an image job on the worker.
-- The input image of an edit goes as an `IMAGE` message; each result
-- image comes back as one. Messages too large give 413.
-- @tparam table req the image request
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table the result of `Engine:generate`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function R:generate(req, ctl)
	local hello = self:ready()
	if hello.caps and not hello.caps.generate then
		return nil, "model does not support images", 400
	end
	if not self.ch then
		return nil, "worker not running", 503
	end
	local id = self.next_id
	self.next_id = id + 1
	local job = setmetatable({ q = {} }, Job)
	self.jobs[id] = job
	local body = {}
	for k, v in pairs(req) do
		body[k] = v
	end
	body.image, body.has_image = nil, req.image ~= nil or nil
	local ok, err = self.ch:send(ipc.GENERATE, id, body)
	if ok and req.image then
		ok, err = self.ch:send(ipc.IMAGE, id, req.image)
	end
	if not ok then
		self.jobs[id] = nil
		return nil, "send to worker: " .. tostring(err), err == "message too large" and 413 or 503
	end
	local images, cancelled = {}, false
	while true do
		job:wait()
		if not cancelled and ctl and ctl.cancelled and ctl.cancelled() then
			cancelled = true
			if self.ch then
				self.ch:send(ipc.CANCEL, id)
			end
		end
		while #job.q > 0 do
			local ev = table.remove(job.q, 1)
			if ev.t == ipc.IMAGE then
				images[#images + 1] = ev.v
			elseif ev.t == ipc.DONE then
				if ev.v.ok then
					local r = ev.v.result or {}
					r.images = images
					return r
				end
				return nil, ev.v.msg, ev.v.status
			end
		end
	end
end

return M
