--- HTTP/1.1 server on luv. Each connection runs in its own coroutine.
-- Handlers get (req, res) and may yield on sockets, timers or jobs. req
-- comes from `lgml.server.http.read_request`. res is a
-- `lgml.server.http.Response`.
-- @module lgml.server.httpd

local uv = require("luv")
local http = require("lgml.server.http")
local stream = require("lgml.server.stream")

local M = {}
local Srv = {}
Srv.__index = Srv

local function log(...)
	io.stderr:write("lgml-server: ", string.format(...), "\n")
end

--- Makes a server.
-- @tparam[opt] table opts the options:
--
-- - `limits`: limits to change (see `lgml.server.http.limits`)
-- - `log`: a printf-style function (default: write to stderr)
-- - `cors`: an origin, such as `"*"`, to allow cross-origin requests
--   from; every response then has CORS headers, and OPTIONS on a known
--   path answers the preflight (default: no CORS)
-- @treturn Server the server
function M.new(opts)
	opts = opts or {}
	local lim = {}
	for k, v in pairs(http.limits) do
		lim[k] = (opts.limits and opts.limits[k]) or v
	end
	local extra
	if opts.cors then
		extra = {
			["Access-Control-Allow-Origin"] = opts.cors,
			["Access-Control-Expose-Headers"] = "*",
		}
		if opts.cors ~= "*" then
			extra.Vary = "Origin"
		end
	end
	return setmetatable({
		routes = {}, limits = lim, conns = {}, extra = extra,
		log = opts.log or log, closing = false,
	}, Srv)
end

--- An HTTP server.
-- After `listen`, the fields `host` and `port` hold the address.
-- @type Server

--- Adds a handler for method and exact path.
-- A GET handler also serves HEAD.
-- @tparam string method the method, such as `"GET"`
-- @tparam string path the path, with no query
-- @tparam function fn the handler: fn(req, res)
function Srv:route(method, path, fn)
	self.routes[path] = self.routes[path] or {}
	self.routes[path][method] = fn
end

local function allowed(methods)
	local t = {}
	for m in pairs(methods) do
		t[#t + 1] = m
	end
	table.sort(t)
	return table.concat(t, ", ")
end

--- Runs the handler for a request.
-- It sends 404 for an unknown path and 405 for an unknown method. A
-- handler error is logged. It gives a 500 when no response started,
-- and closes the connection.
-- @tparam table req the request
-- @tparam lgml.server.http.Response res the response
function Srv:dispatch(req, res)
	local methods = self.routes[req.path]
	if not methods then
		return res:error(404, "no route for " .. req.path)
	end
	local fn = methods[req.method]
	if not fn and req.method == "HEAD" then
		fn = methods.GET
	end
	if not fn and req.method == "OPTIONS" and self.extra then
		local m = allowed(methods)
		return res:send(204, "", {
			["Access-Control-Allow-Methods"] = methods.GET and m .. ", HEAD, OPTIONS" or m .. ", OPTIONS",
			["Access-Control-Allow-Headers"] = req.headers["access-control-request-headers"] or "*",
			["Access-Control-Max-Age"] = 600,
		})
	end
	if not fn then
		return res:json(405, http.error_object(405, req.method ..
		    " not allowed on " .. req.path), { Allow = allowed(methods) })
	end
	local ok, err = xpcall(fn, debug.traceback, req, res)
	if not ok then
		self.log("%s %s: %s", req.method, req.path, err)
		if not res.started then
			res:error(500, "internal error")
		end
		res.close = true
	end
end

-- Runs one request on conn. Returns true to keep the connection.
local function serve_one(self, conn)
	local req, status, msg = http.read_request(conn, self.limits)
	if not req then
		if status then
			local res = http.response(conn, nil)
			res:error(status, msg)
		end
		return false
	end
	local res = http.response(conn, req)
	res.extra = self.extra
	if self.closing then
		res.close = true
	end
	self:dispatch(req, res)
	if not res.started then
		res:error(500, "handler sent no response")
	elseif not res.done then
		res.close = true
	end
	return not (res.close or conn.s:gone() or self.closing)
end

local function serve(self, tcp)
	local conn = { s = stream.new(tcp), buf = "", idle = false }
	self.conns[conn] = true
	while serve_one(self, conn) do
	end
	self.conns[conn] = nil
	conn.s:close()
end

local sigpipe

-- Ignores SIGPIPE, so a write to a reset socket fails with EPIPE.
local function ignore_sigpipe()
	if sigpipe or not uv.new_signal then
		return
	end
	sigpipe = uv.new_signal()
	sigpipe:start("sigpipe", function() end)
	sigpipe:unref()
end

function Srv:on_connection(err)
	if err then
		self.log("accept: %s", err)
		return
	end
	local tcp = uv.new_tcp()
	if not self.tcp:accept(tcp) then
		tcp:close()
		return
	end
	tcp:nodelay(true)
	local co = coroutine.create(serve)
	local ok, e = coroutine.resume(co, self, tcp)
	if not ok then
		self.log("connection: %s", debug.traceback(co, e))
	end
end

--- Listens on host and port.
-- It also ignores SIGPIPE, so a write to a reset socket fails with EPIPE.
-- @tparam string host the address
-- @tparam int port the port; 0 picks a free port
-- @treturn[1] int the port
-- @treturn[2] nil
-- @treturn[2] string an error message
function Srv:listen(host, port)
	ignore_sigpipe()
	local tcp = uv.new_tcp()
	local ok, err = tcp:bind(host, port)
	if ok then
		ok, err = tcp:listen(128, function(e)
			self:on_connection(e)
		end)
	end
	if not ok then
		tcp:close()
		return nil, err
	end
	self.tcp = tcp
	local addr = tcp:getsockname()
	self.host, self.port = addr.ip, addr.port
	return addr.port
end

--- Stops accepting.
-- Idle connections close now. Busy ones close after their current
-- response.
function Srv:close()
	self.closing = true
	if self.tcp and not self.tcp:is_closing() then
		self.tcp:close()
	end
	for conn in pairs(self.conns) do
		if conn.idle then
			conn.s:close()
		end
	end
end

--- Coroutine helpers for handlers.
-- @section helpers

--- Yields the running coroutine for ms milliseconds.
-- This is `lgml.server.stream.sleep`.
-- @function sleep
-- @tparam int ms the time
M.sleep = stream.sleep
--- Yields until fn's callback runs. This is `lgml.server.stream.await`.
-- @function await
-- @tparam function fn starts the work; it gets a done function
-- @return the arguments passed to the done function
M.await = stream.await

return M
