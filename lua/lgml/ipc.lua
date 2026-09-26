--- imsg channel between the front and a worker, driven by luv.
-- Payloads are JSON, except for `AUDIO`, which carries raw bytes. One
-- poll handle per fd watches reads, and writes while the output queue
-- is not empty.
--
-- Each message has a type, a job id and a payload. The worker sends
-- `HELLO` once, with id 0. Then the front starts jobs with `JOB`,
-- `TRANSCRIBE` or `SPEAK`, each with a new id. The worker answers each
-- job with zero or more `DELTA` or `AUDIO` messages and one `DONE`.
-- The front sends `CANCEL` to stop a job; the job still ends with
-- `DONE`.
-- @module lgml.ipc
local uv = require("luv")
local imsg = require("imsg")
local json = require("lgml.json")

local M = {}

--- Worker to front: the worker is ready.
-- Payload: `{ models, props, caps }`. models and props are the results
-- of `Engine:models` and `Engine:props`. caps is `{ chat, transcribe,
-- speak }`, with true for each method the engine has.
M.HELLO = 1
--- Front to worker: start a chat job.
-- Payload: the chat request for `Engine:chat`.
M.JOB = 2
--- Front to worker: stop the job with this id.
-- Payload: an empty object.
M.CANCEL = 3
--- Worker to front: one piece of chat output.
-- Payload: `{ k, text, name, index, id }`. k is `"reasoning"` or
-- `"content"` with text, `"tool_call"` with name, index and an
-- optional id, or `"tool_args"` with index and text. Tool call indexes
-- start at 0 in each job.
M.DELTA = 4
--- Worker to front: the job ended.
-- Payload: `{ ok, msg, status, finish, result }`. ok is true on
-- success. msg and status give the error otherwise. finish is the
-- `{ reason, usage, timings }` of `Sink:finish` for a chat job. result
-- is the transcription for a transcribe job, or `{ sample_rate }` for
-- a speech job.
M.DONE = 5
--- Front to worker: start a transcription job.
-- Payload: the request for `Engine:transcribe`. An `AUDIO` message with
-- the same id follows. The job starts when it arrives.
M.TRANSCRIBE = 6
--- Raw audio bytes for a job, in either direction.
-- Front to worker: the uploaded file for `TRANSCRIBE`. Worker to front:
-- the WAV bytes for `SPEAK`, before `DONE`.
M.AUDIO = 7
--- Front to worker: start a speech job.
-- Payload: the request for `Engine:speak`. The worker sends `AUDIO`,
-- then `DONE`.
M.SPEAK = 8

-- These types carry raw bytes, not JSON.
local RAW = { [M.AUDIO] = true }

--- The largest imsg message in bytes: 256 MiB.
M.MAXSIZE = 256 << 20

local C = {}
C.__index = C

--- Opens a channel on a socket.
-- on_message(type, id, value) runs for each message. value is a string
-- for `AUDIO` and the decoded JSON for the other types. on_close(err)
-- runs once, at EOF or error.
-- @tparam int fd the socket file descriptor; the channel owns it
-- @tparam function on_message the message callback
-- @tparam[opt] function on_close the close callback
-- @treturn Channel the channel
function M.channel(fd, on_message, on_close)
	local ch = setmetatable({
		fd = fd, buf = imsg.new(fd), on_message = on_message,
		on_close = on_close, poll = uv.new_poll(fd), events = "",
	}, C)
	ch.buf:set_maxsize(M.MAXSIZE)
	ch:watch("r")
	return ch
end

--- An imsg channel.
-- @type Channel

function C:watch(events)
	if self.events ~= events then
		self.events = events
		self.poll:start(events, function(err, ev)
			self:ready(err, ev)
		end)
	end
end

function C:ready(err, ev)
	if err then
		return self:close(err)
	end
	if ev:find("w") then
		local ok, e = pcall(self.buf.write, self.buf)
		if not ok then
			return self:close(e)
		end
	end
	if ev:find("r") then
		local ok, r = pcall(self.buf.read, self.buf)
		if not ok then
			return self:close(r)
		end
		self:drain()
		if r == nil then
			return self:close("eof")
		end
	end
	if self.buf then
		self:watch(self.buf:queuelen() > 0 and "rw" or "r")
	end
end

function C:drain()
	while self.buf do
		local ok, m = pcall(self.buf.get, self.buf)
		if not ok then
			return self:close(m)
		elseif not m then
			return
		end
		local t = m:type()
		local v = RAW[t] and m:data() or json.decode(m:data())
		self.on_message(t, m:id(), v)
	end
end

--- Queues a message and starts writing.
-- Raw types take a string; the others take a value to encode as JSON.
-- A write error closes the channel.
-- @tparam int t the message type
-- @tparam int id the job id
-- @tparam[opt] any value the payload; nil sends an empty object
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string `"closed"`, `"message too large"` or an error
-- @raise when value cannot be encoded as JSON
function C:send(t, id, value)
	if not self.buf then
		return nil, "closed"
	end
	local s = RAW[t] and value or assert(json.encode(value or json.object()))
	if #s > M.MAXSIZE - 64 then
		return nil, "message too large"
	end
	self.buf:compose(t, id, 0, -1, s)
	local ok, e = pcall(self.buf.write, self.buf)
	if not ok then
		self:close(e)
		return nil, e
	end
	self:watch(self.buf:queuelen() > 0 and "rw" or "r")
	return true
end

--- Closes the channel and its fd.
-- It stops the poll handle and calls on_close(err). A second call does
-- nothing.
-- @tparam[opt] any err the reason passed to on_close
function C:close(err)
	if not self.buf then
		return
	end
	self.poll:stop()
	self.poll:close()
	self.buf:close(true)
	self.buf = nil
	if self.on_close then
		self.on_close(err)
	end
end

return M
