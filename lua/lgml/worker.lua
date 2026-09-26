--- Worker process: owns one device and runs engine jobs for the front.
-- Each job runs in its own coroutine. The engine yields between steps,
-- so `JOB` and `CANCEL` messages arrive while other jobs run. The
-- messages are in `lgml.ipc`. A chat job gets a sink that sends each
-- piece of output as a `DELTA` message.
-- @module lgml.worker
local uv = require("luv")
local ipc = require("lgml.ipc")

local M = {}

-- A sink that sends each delta to the front. Tool call indexes are
-- local to the job; the front maps them to its own.
local function proxy(ch, id)
	local s = { ncalls = 0, stopped = false }

	local function delta(v)
		if s.stopped then
			return nil, "cancelled"
		end
		if not ch:send(ipc.DELTA, id, v) then
			s.stopped = true
			return nil, "cancelled"
		end
		return true
	end

	function s.reasoning(_, text)
		return delta({ k = "reasoning", text = text })
	end

	function s.content(_, text)
		return delta({ k = "content", text = text })
	end

	function s.tool_call(_, name, cid)
		local index = s.ncalls
		s.ncalls = index + 1
		if not delta({ k = "tool_call", name = name, id = cid, index = index }) then
			return nil, "cancelled"
		end
		return index
	end

	function s.tool_arguments(_, index, text)
		return delta({ k = "tool_args", index = index, text = text })
	end

	function s.finish(_, reason, usage, timings)
		s.done = { reason = reason, usage = usage, timings = timings }
	end

	function s.cancelled()
		return s.stopped
	end

	return s
end

local function caps(engine)
	return {
		chat = engine.chat ~= nil, transcribe = engine.transcribe ~= nil,
		speak = engine.speak ~= nil,
	}
end

--- Serves the engine on fd until the front closes it.
-- It sends `HELLO`, then runs the luv loop. It returns when the channel
-- closes. A job for a method that the engine does not have ends with
-- status 400.
-- @tparam int fd the imsg socket to the front
-- @tparam lgml.server.engine.Engine engine the engine
function M.serve(fd, engine)
	local jobs, pending = {}, {}
	local ch
	local function chat(id, req)
		local sink = proxy(ch, id)
		jobs[id] = sink
		coroutine.wrap(function()
			local ok, msg, status
			if engine.chat then
				ok, msg, status = engine:chat(req, sink)
			else
				msg, status = "model does not support chat", 400
			end
			jobs[id] = nil
			ch:send(ipc.DONE, id, {
				ok = ok == true, msg = msg, status = status, finish = sink.done,
			})
		end)()
	end
	-- A TRANSCRIBE request waits for its AUDIO message.
	local function transcribe(id, audio)
		local req = pending[id]
		pending[id] = nil
		if not req then
			return
		end
		local ctl = { stopped = false }
		function ctl.cancelled()
			return ctl.stopped
		end
		jobs[id] = ctl
		coroutine.wrap(function()
			local r, msg, status
			if engine.transcribe then
				r, msg, status = engine:transcribe(req, audio, ctl)
			else
				msg, status = "model does not support transcription", 400
			end
			jobs[id] = nil
			ch:send(ipc.DONE, id, { ok = r ~= nil, msg = msg, status = status, result = r })
		end)()
	end
	-- r.audio is an lgml audio object, or r.wav holds WAV bytes.
	local function speak(id, req)
		local ctl = { stopped = false }
		function ctl.cancelled()
			return ctl.stopped
		end
		jobs[id] = ctl
		coroutine.wrap(function()
			local r, msg, status
			if engine.speak then
				r, msg, status = engine:speak(req, ctl)
			else
				msg, status = "model does not support speech", 400
			end
			jobs[id] = nil
			local result
			if r then
				local wav = r.wav or r.audio:wav(1, "s16")
				result = { sample_rate = r.sample_rate or (r.audio and r.audio:rate()) }
				local ok, err = ch:send(ipc.AUDIO, id, wav)
				if not ok then
					r, msg, status = nil, "send audio: " .. tostring(err), 500
				end
			end
			ch:send(ipc.DONE, id, { ok = r ~= nil, msg = msg, status = status, result = result })
		end)()
	end
	ch = ipc.channel(fd, function(t, id, v)
		if t == ipc.JOB then
			chat(id, v)
		elseif t == ipc.TRANSCRIBE then
			pending[id] = v
		elseif t == ipc.AUDIO then
			transcribe(id, v)
		elseif t == ipc.SPEAK then
			speak(id, v)
		elseif t == ipc.CANCEL and jobs[id] then
			jobs[id].stopped = true
		end
	end, function()
		uv.stop()
	end)
	ch:send(ipc.HELLO, 0, { models = engine:models(), props = engine:props(), caps = caps(engine) })
	uv.run()
end

return M
