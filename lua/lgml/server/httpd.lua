--- HTTP/1.1 server on luv. Each connection runs in its own coroutine.
-- Handlers get (req, res) and may yield on sockets, timers or jobs. req
-- comes from `lgml.server.http.read_request`. res is a
-- `lgml.server.http.Response`.
-- Each request gets an id. The reply has it in the header
-- `x-lgml-request-id`, and each log line of the request has it as `rid`
-- (see `lgml.log`). At info level each request logs one `access` line;
-- at debug level the request line and body come first, with blobs elided.
-- @module lgml.server.httpd

local uv = require("luv")
local http = require("lgml.server.http")
local stream = require("lgml.server.stream")
local log = require("lgml.log")

local L = log.new("http")

local M = {}
local Srv = {}
Srv.__index = Srv

--- Makes a server.
-- @tparam[opt] table opts the options:
--
-- - `limits`: limits to change (see `lgml.server.http.limits`)
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
		routes = {}, limits = lim, conns = {}, extra = extra, closing = false,
	}, Srv)
end

--- An HTTP server.
-- After `listen`, the fields `host` and `port` hold the address.
-- @type Server

--- Adds a handler for method and path.
-- A GET handler also serves HEAD. A path segment `{name}` is a
-- parameter: it matches one segment, and the handler finds its value
-- in `req.params.name`. An exact path wins over a path with
-- parameters.
-- @tparam string method the method, such as `"GET"`
-- @tparam string path the path, with no query
-- @tparam function fn the handler: fn(req, res)
function Srv:route(method, path, fn)
	if not self.routes[path] and path:find("{", 1, true) then
		local names = {}
		local pat = "^" .. path:gsub("[%^%$%(%)%%%.%[%]%*%+%-%?]", "%%%0"):gsub("{([%w_]+)}", function(name)
			names[#names + 1] = name
			return "([^/]+)"
		end) .. "$"
		self.templates = self.templates or {}
		self.templates[#self.templates + 1] = { path = path, pattern = pat, names = names }
	end
	self.routes[path] = self.routes[path] or {}
	self.routes[path][method] = fn
end

-- The methods for a request path: those of the exact path, else those
-- of the first template that matches it. A template sets req.params.
local function methods_for(self, req)
	local methods = self.routes[req.path]
	if methods then
		return methods
	end
	for _, t in ipairs(self.templates or {}) do
		local caps = { req.path:match(t.pattern) }
		if #caps > 0 then
			req.params = {}
			for i, name in ipairs(t.names) do
				req.params[name] = http.unescape(caps[i])
			end
			return self.routes[t.path]
		end
	end
	return nil
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
-- handler error is logged (see `lgml.log`). It gives a 500 when no response started,
-- and closes the connection.
-- @tparam table req the request
-- @tparam lgml.server.http.Response res the response
function Srv:dispatch(req, res)
	local methods = methods_for(self, req)
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
		L:error("handler error", "method", req.method, "path", req.path, "err", err)
		if not res.started then
			res:error(500, "internal error")
		end
		res.close = true
	end
end

-- Access line fields from the request notes, in this order; other
-- notes follow by name.
local NOTE_ORDER = {
	"model", "prompt_tokens", "cached_tokens", "completion_tokens", "tok_s", "images",
	"questions", "steps", "audio_s", "finish", "tool_calls",
}
local NOTE_SET = {}
for _, k in ipairs(NOTE_ORDER) do
	NOTE_SET[k] = true
end

-- The largest request body a debug line shows.
local BODY_MAX = 64 * 1024

local function log_request(req)
	local ctype = req.headers["content-type"] or ""
	local body
	if #req.body > 0 and (ctype:find("json", 1, true) or req.body:find("^%s*[{%[]")) then
		body = log.full(log.cut(log.elide(req.body), BODY_MAX))
	end
	L:debug("request", "method", req.method, "path", req.path,
	    "query", req.query ~= "" and req.query or nil, "type", ctype ~= "" and ctype or nil,
	    "bytes", #req.body, "body", body)
end

-- One info line per request: the request, the reply and the notes.
local function log_access(req, res, t0)
	local notes = log.notes()
	local f = {
		"method", req and req.method, "path", req and req.path, "status", res.status,
		"ms", math.floor((uv.hrtime() - t0) / 1e5 + 0.5) / 10, "in", req and #req.body,
		"out", res.bytes_out or 0,
	}
	local n = 12
	local function add(k, v)
		f[n + 1], f[n + 2], n = k, v, n + 2
	end
	for _, k in ipairs(NOTE_ORDER) do
		add(k, notes[k])
	end
	local rest = {}
	for k in pairs(notes) do
		if not NOTE_SET[k] then
			rest[#rest + 1] = k
		end
	end
	table.sort(rest)
	for _, k in ipairs(rest) do
		add(k, notes[k])
	end
	add("err", res.err)
	L:info("access", table.unpack(f, 1, n))
end

-- Runs one request on conn. Returns true to keep the connection. Each
-- request gets an id, bound to this coroutine while it runs.
local function serve_one(self, conn)
	local req, status, msg = http.read_request(conn, self.limits)
	local t0 = uv.hrtime()
	if not req then
		if status then
			local res = http.response(conn, nil)
			res.rid = log.new_id()
			log.bind(res.rid)
			res:error(status, msg)
			log_access(nil, res, t0)
			log.bind(nil)
		end
		return false
	end
	local res = http.response(conn, req)
	res.extra = self.extra
	res.rid = log.new_id()
	log.bind(res.rid)
	if self.closing then
		res.close = true
	end
	if L:on(log.DEBUG) then
		log_request(req)
	end
	self:dispatch(req, res)
	if not res.started then
		res:error(500, "handler sent no response")
	elseif not res.done then
		res.close = true
	end
	log_access(req, res, t0)
	log.bind(nil)
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
		L:error("accept", "err", err)
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
		L:error("connection", "err", debug.traceback(co, e))
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
