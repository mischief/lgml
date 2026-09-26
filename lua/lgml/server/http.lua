--- HTTP/1.1 request reader and response writer.
-- A connection is `{ s = stream, buf = string }`. s is a
-- `lgml.server.stream.Stream`. buf holds bytes read past the last
-- request, which is how pipelined requests survive.
-- @module lgml.server.http

local json = require("lgml.json")
local pico = require("picohttpparser").new()

local M = {}

--- Default limits for `read_request`.
-- @table limits
-- @field max_header header block size in bytes (64 KiB)
-- @field max_body body size in bytes (64 MiB)
-- @field idle_ms wait for a new request (60 s)
-- @field header_ms wait for the rest of a header block (30 s)
-- @field body_ms wait for each piece of a body (60 s)
M.limits = {
	max_header = 64 * 1024,
	max_body = 64 * 1024 * 1024,
	idle_ms = 60 * 1000,
	header_ms = 30 * 1000,
	body_ms = 60 * 1000,
}

--- Reason phrases by status code.
M.reasons = {
	[100] = "Continue", [200] = "OK", [204] = "No Content",
	[400] = "Bad Request", [401] = "Unauthorized", [403] = "Forbidden",
	[404] = "Not Found", [405] = "Method Not Allowed",
	[408] = "Request Timeout", [409] = "Conflict", [411] = "Length Required",
	[413] = "Content Too Large", [415] = "Unsupported Media Type",
	[422] = "Unprocessable Content", [429] = "Too Many Requests",
	[431] = "Request Header Fields Too Large",
	-- nginx uses 499 when the client closes the connection first.
	[499] = "Client Closed Request",
	[500] = "Internal Server Error", [501] = "Not Implemented",
	[503] = "Service Unavailable", [505] = "HTTP Version Not Supported",
}

local ERROR_TYPES = {
	[401] = "authentication_error", [403] = "permission_error",
	[404] = "not_found_error", [429] = "rate_limit_error",
}

--- Returns the OpenAI error object for status and message.
-- @tparam int status the HTTP status
-- @tparam[opt] string message the message (default: the reason phrase)
-- @tparam[opt] string etype the error type (default: from status)
-- @tparam[opt] string param the bad parameter
-- @tparam[opt] string code the error code
-- @treturn table `{ error = { message, type, param, code } }`
function M.error_object(status, message, etype, param, code)
	return {
		error = {
			message = message or M.reasons[status] or "error",
			type = etype or ERROR_TYPES[status] or (status >= 500 and
			    "server_error" or "invalid_request_error"),
			param = param or json.null,
			code = code or json.null,
		},
	}
end

--- The message of an error body, for logs.
-- It reads the OpenAI `error.message`, or a FastAPI `detail` (a string,
-- or a list whose first entry has `msg`).
-- @tparam table value the error body
-- @treturn ?string the message
function M.error_message(value)
	local e, d = value.error, value.detail
	if type(e) == "table" and type(e.message) == "string" then
		return e.message
	elseif type(d) == "string" then
		return d
	elseif type(d) == "table" and type(d[1]) == "table" and d[1].msg then
		return tostring(d[1].msg)
	end
end

-- request reading

local function pdecode(s)
	return (s:gsub("+", " "):gsub("%%(%x%x)", function(h)
		return string.char(tonumber(h, 16))
	end))
end

--- Decodes the `%XX` escapes of a path segment.
-- @tparam string s the segment
-- @treturn string the decoded text
function M.unescape(s)
	return (s:gsub("%%(%x%x)", function(h)
		return string.char(tonumber(h, 16))
	end))
end

--- Decodes a query string into a table. The last value of a key wins.
-- @tparam ?string q the query string
-- @treturn table the keys and values
function M.parse_query(q)
	local t = {}
	for pair in (q or ""):gmatch("[^&]+") do
		local k, v = pair:match("^([^=]*)=?(.*)$")
		t[pdecode(k)] = pdecode(v)
	end
	return t
end

-- Parses the request line and headers with picohttpparser. head ends
-- with the blank line. Header names are lowercase; repeats join with ", ".
local function parse_head(head)
	local r, h = {}, {}
	if pico:parseRequest(r, h, head) ~= #head then
		if head:match("^%u+ %S+ HTTP/[02-9]") then
			return nil, 505, "HTTP/1.x only"
		end
		return nil, 400, "bad request"
	end
	local path, query = r.path:match("^([^?#]*)%??([^#]*)")
	local req = {
		method = r.method, target = r.path, path = path, query = query,
		version = ("1.%d"):format(math.tointeger(r.minor_version) or 0),
		headers = {},
	}
	for name, v in pairs(h) do
		local k = name:lower()
		local val = type(v) == "table" and table.concat(v, ", ") or v
		local old = req.headers[k]
		req.headers[k] = old and (old .. ", " .. val) or val
	end
	return req
end

local function keep_alive(req)
	local c = (req.headers.connection or ""):lower()
	if req.version == "1.0" then
		return c:find("keep%-alive") ~= nil
	end
	return not c:find("close")
end

-- Reads until the end of the header block. Returns the head text, or
-- nil plus a status (nil for a clean close before any byte).
local function read_head(conn, lim)
	local from = 1
	while true do
		conn.buf = conn.buf:gsub("^\r?\n", "")
		local e = conn.buf:find("\r\n\r\n", from, true)
		if e and e > lim.max_header then
			return nil, 431, "request header too large"
		elseif e then
			local head = conn.buf:sub(1, e + 1)
			conn.buf = conn.buf:sub(e + 4)
			return head
		end
		if #conn.buf > lim.max_header then
			return nil, 431, "request header too large"
		end
		from = math.max(1, #conn.buf - 3)
		local empty = #conn.buf == 0
		conn.idle = empty
		local data, err = conn.s:read(empty and lim.idle_ms or lim.header_ms)
		conn.idle = false
		if not data then
			if empty then
				return nil
			end
			return nil, err == "timeout" and 408 or 400,
			    "incomplete request header"
		end
		conn.buf = conn.buf .. data
	end
end

local function body_length(req, lim)
	local h = req.headers
	if h["transfer-encoding"] then
		return nil, 411, "chunked request bodies are not supported"
	end
	local cl = h["content-length"]
	if not cl then
		return 0
	end
	if not cl:match("^%d+$") or #cl > 15 then
		return nil, 400, "bad content-length"
	end
	local n = tonumber(cl)
	if n > lim.max_body then
		return nil, 413, "request body too large"
	end
	return n
end

local function read_body(conn, n, lim)
	local parts, have = { conn.buf }, #conn.buf
	while have < n do
		local data, err = conn.s:read(lim.body_ms)
		if not data then
			return nil, err == "timeout" and 408 or 400,
			    "incomplete request body"
		end
		parts[#parts + 1] = data
		have = have + #data
	end
	local all = table.concat(parts)
	conn.buf = all:sub(n + 1)
	return all:sub(1, n)
end

--- Reads one request.
-- A nil status means the peer closed or went idle. It sends
-- `100 Continue` when the client asks for it. Chunked bodies give 411.
-- @tparam table conn the connection
-- @tparam[opt] table lim the limits (default `limits`)
-- @treturn[1] table the request: `method`, `target`, `path`, `query`,
-- `version` (`"1.0"` or `"1.1"`), `headers` (lowercase names; repeats
-- join with ", "), `keep_alive` and `body`
-- @treturn[2] nil
-- @treturn[2] ?int an HTTP status
-- @treturn[2] string a message
function M.read_request(conn, lim)
	lim = lim or M.limits
	local head, status, msg = read_head(conn, lim)
	if not head then
		return nil, status, msg
	end
	local req
	req, status, msg = parse_head(head .. "\r\n")
	if not req then
		return nil, status, msg
	end
	req.keep_alive = keep_alive(req)
	local n
	n, status, msg = body_length(req, lim)
	if not n then
		return nil, status, msg
	end
	local expect = (req.headers.expect or ""):lower()
	if n > #conn.buf and expect == "100-continue" and req.version == "1.1" then
		conn.s:write("HTTP/1.1 100 Continue\r\n\r\n")
	end
	req.body, status, msg = read_body(conn, n, lim)
	if not req.body then
		return nil, status, msg
	end
	return req
end

-- response writing

local R = {}
R.__index = R

--- Makes a response for a request.
-- @tparam table conn the connection
-- @tparam ?table req the request; nil for a request that did not parse
-- @treturn Response the response
function M.response(conn, req)
	return setmetatable({
		conn = conn, req = req, s = conn.s,
		started = false, done = false, chunked = false,
		close = not (req and req.keep_alive),
	}, R)
end

--- An HTTP response.
-- The fields `started` and `done` track the state. Set the field
-- `close` to true to close the connection after the response.
-- After it starts, `status` holds the status and `bytes_out` the body
-- bytes sent so far. A JSON reply with status 400 or more sets `err` to
-- its message. When the field `rid` is set, each response has the header
-- `x-lgml-request-id` with it.
-- @type Response

local function header_block(res, status, headers)
	local out = {
		("HTTP/1.1 %d %s\r\n"):format(status,
		    M.reasons[status] or "Unknown"),
		"Server: lgml\r\n",
		os.date("!Date: %a, %d %b %Y %H:%M:%S GMT\r\n"),
	}
	for k, v in pairs(headers or {}) do
		out[#out + 1] = k .. ": " .. tostring(v) .. "\r\n"
	end
	-- Headers for every response, such as CORS headers.
	for k, v in pairs(res.extra or {}) do
		if not (headers and headers[k]) then
			out[#out + 1] = k .. ": " .. tostring(v) .. "\r\n"
		end
	end
	if res.rid then
		out[#out + 1] = "x-lgml-request-id: " .. res.rid .. "\r\n"
	end
	if res.close then
		out[#out + 1] = "Connection: close\r\n"
	elseif res.req and res.req.version == "1.0" then
		out[#out + 1] = "Connection: keep-alive\r\n"
	end
	out[#out + 1] = "\r\n"
	return table.concat(out)
end

--- Sends a complete response.
-- A HEAD request gets no body.
-- @tparam int status the HTTP status
-- @tparam[opt] string body the body
-- @tparam[opt] table headers more headers; the table is changed
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @raise when the response has started
function R:send(status, body, headers)
	assert(not self.started, "response already started")
	body = body or ""
	headers = headers or {}
	headers["Content-Length"] = #body
	self.started, self.done = true, true
	self.status = status
	local head = header_block(self, status, headers)
	if self.req and self.req.method == "HEAD" then
		body = ""
	end
	self.bytes_out = (self.bytes_out or 0) + #body
	return self.s:write(head .. body)
end

--- Sends value as a JSON response.
-- @tparam int status the HTTP status
-- @param value the value
-- @tparam[opt] table headers more headers; the table is changed
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @raise when the response has started, or value cannot be encoded
function R:json(status, value, headers)
	if status >= 400 and type(value) == "table" then
		self.err = M.error_message(value)
	end
	headers = headers or {}
	headers["Content-Type"] = "application/json"
	return self:send(status, assert(json.encode(value)) .. "\n", headers)
end

--- Sends an OpenAI-shaped error.
-- The arguments are those of `error_object`.
-- @tparam int status the HTTP status
-- @tparam[opt] string message the message
-- @tparam[opt] string etype the error type
-- @tparam[opt] string param the bad parameter
-- @tparam[opt] string code the error code
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
function R:error(status, message, etype, param, code)
	return self:json(status, M.error_object(status, message, etype,
	    param, code))
end

--- Starts a streamed response.
-- It uses chunked encoding on HTTP/1.1 and a close-delimited body on
-- HTTP/1.0.
-- @tparam int status the HTTP status
-- @tparam[opt] table headers more headers; the table is changed
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @raise when the response has started
function R:start(status, headers)
	assert(not self.started, "response already started")
	headers = headers or {}
	if self.req and self.req.version == "1.1" then
		self.chunked = true
		headers["Transfer-Encoding"] = "chunked"
	else
		self.close = true
	end
	self.status = status
	self.started = true
	return self.s:write(header_block(self, status, headers))
end

--- Writes one piece of a streamed body. Each call flushes.
-- @tparam string data the bytes; an empty string writes nothing
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @raise when the response is not streaming
function R:write(data)
	assert(self.started and not self.done, "response not streaming")
	if #data == 0 then
		return true
	end
	self.bytes_out = (self.bytes_out or 0) + #data
	if self.chunked then
		data = ("%x\r\n%s\r\n"):format(#data, data)
	end
	return self.s:write(data)
end

--- Ends a streamed body. A second call does nothing.
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
function R:finish()
	if self.done then
		return true
	end
	self.done = true
	if self.chunked then
		return self.s:write("0\r\n\r\n")
	end
	return true
end

return M
