--- A luv TCP handle seen from one coroutine.
-- Reads queue in the background, so the owner sees EOF and resets
-- while it does other work. `Stream:read` and `Stream:write` yield the
-- calling coroutine until luv calls back. Reading stops while 1 MiB
-- waits unread.
-- @module lgml.server.stream

local uv = require("luv")

local M = {}
local S = {}
S.__index = S

-- Stop reading when this many bytes wait unread.
local HIGH_WATER = 1 << 20

local function resume(co, ...)
	local ok, err = coroutine.resume(co, ...)
	if not ok then
		require("lgml.log").new("stream"):error("coroutine error", "err", debug.traceback(co, err))
	end
end

--- Yields until fn's callback runs.
-- fn gets a function to call when done. When fn calls it at once,
-- await does not yield.
-- @function await
-- @tparam function fn starts the work
-- @return the arguments passed to the done function
-- @raise when not called from a coroutine
local function await(fn)
	local co = assert(coroutine.running(), "await needs a coroutine")
	local done, res = false, nil
	local waiting = false
	fn(function(...)
		if done then
			return
		end
		done, res = true, table.pack(...)
		if waiting then
			resume(co)
		end
	end)
	if not done then
		waiting = true
		coroutine.yield()
	end
	return table.unpack(res, 1, res.n)
end
M.await = await

--- Yields the running coroutine for ms milliseconds.
-- @tparam int ms the time
-- @raise when not called from a coroutine
function M.sleep(ms)
	local t = uv.new_timer()
	await(function(done)
		t:start(ms, 0, done)
	end)
	t:close()
end

--- Wraps a TCP handle and starts reading.
-- @tparam userdata tcp a connected luv TCP handle
-- @treturn Stream the stream
function M.new(tcp)
	local s = setmetatable({
		tcp = tcp, q = {}, qhead = 1, qtail = 0, qbytes = 0,
		eof = false, err = nil, closed = false, broken = false,
		reading = false, wake = nil,
	}, S)
	s:start()
	return s
end

--- A TCP stream.
-- @type Stream

function S:start()
	if self.reading or self.eof or self.closed then
		return
	end
	self.reading = true
	self.tcp:read_start(function(err, data)
		self:on_read(err, data)
	end)
end

function S:on_read(err, data)
	if err or not data then
		self.eof, self.err = true, err
		self.reading = false
		self.tcp:read_stop()
	else
		self.qtail = self.qtail + 1
		self.q[self.qtail] = data
		self.qbytes = self.qbytes + #data
		if self.qbytes > HIGH_WATER then
			self.reading = false
			self.tcp:read_stop()
		end
	end
	if self.wake then
		self.wake()
	end
end

local function pop(self)
	local h = self.qhead
	local data = self.q[h]
	self.q[h] = nil
	self.qhead = h + 1
	self.qbytes = self.qbytes - #data
	if not self.reading and self.qbytes < HIGH_WATER then
		self:start()
	end
	return data
end

--- Returns the next chunk.
-- @tparam[opt] int timeout the time to wait in milliseconds
-- @treturn[1] string the chunk
-- @treturn[2] nil
-- @treturn[2] string `"eof"`, `"timeout"` or an error
function S:read(timeout)
	if self.qhead <= self.qtail then
		return pop(self)
	end
	if self.eof or self.closed then
		return nil, self.err or "eof"
	end
	local timer = timeout and uv.new_timer()
	local why = await(function(done)
		self.wake = done
		if timer then
			timer:start(timeout, 0, function()
				done("timeout")
			end)
		end
	end)
	self.wake = nil
	if timer then
		timer:close()
	end
	if why == "timeout" then
		return nil, "timeout"
	end
	return self:read()
end

--- Writes data and waits until the kernel has it.
-- @tparam string data the bytes
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
function S:write(data)
	if self.closed or self.broken then
		return nil, "closed"
	end
	local err = await(function(done)
		local ok, e = self.tcp:write(data, done)
		if not ok then
			done(e or "write failed")
		end
	end)
	if err then
		self.broken = true
		return nil, err
	end
	return true
end

--- Tells if the peer is gone: EOF, a reset or a failed write.
-- @treturn boolean true when the peer is gone
function S:gone()
	return self.eof or self.broken or self.closed
end

-- Reads and drops input until EOF or ms pass, then closes. A close
-- with unread input sends RST, and RST can destroy the reply in flight.
local function linger(tcp, eof, ms)
	local timer = uv.new_timer()
	local function finish()
		if not timer:is_closing() then
			timer:close()
		end
		if not tcp:is_closing() then
			tcp:close()
		end
	end
	if eof then
		return finish()
	end
	timer:start(ms, 0, finish)
	tcp:read_stop()
	tcp:read_start(function(err, data)
		if err or not data then
			finish()
		end
	end)
end

--- Sends FIN after queued writes, then closes the handle.
-- After the FIN, it reads and drops input until EOF or for up to 2
-- seconds. Unread input at close sends RST, which can destroy the reply.
-- After a reset or a failed write, it closes at once.
function S:close()
	if self.closed then
		return
	end
	self.closed = true
	local tcp = self.tcp
	if self.wake then
		self.wake("closed")
	end
	if tcp:is_closing() then
		return
	end
	if self.broken or self.err then
		tcp:close()
		return
	end
	local eof = self.eof
	local ok = tcp:shutdown(function(err)
		if err then
			tcp:close()
		elseif not tcp:is_closing() then
			linger(tcp, eof, 2000)
		end
	end)
	if not ok then
		tcp:close()
	end
end

--- Closes at once, dropping queued writes.
function S:abort()
	self.closed = true
	if self.wake then
		self.wake("closed")
	end
	if not self.tcp:is_closing() then
		self.tcp:close()
	end
end

return M
