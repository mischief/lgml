-- In-process HTTP requests to a server's routes, without a socket.
local uv = require("luv")
local json = require("lgml.json")
local http = require("lgml.server.http")

local M = {}

--- Runs one request on srv and returns the status, the headers
-- (lowercase names) and the body text. body is a table (sent as JSON)
-- or text.
function M.request(srv, method, target, body, headers)
	local out = {}
	local s = {
		write = function(_, data)
			out[#out + 1] = data
			return true
		end,
		gone = function()
			return false
		end,
	}
	local text = type(body) == "table" and assert(json.encode(body)) or body or ""
	local path, query = target:match("^([^?]*)%??(.*)$")
	local hs = {}
	for k, v in pairs(headers or {}) do
		hs[k:lower()] = v
	end
	local req = { method = method, target = target, path = path, query = query, version = "1.0",
		body = text, headers = hs }
	local res = http.response({ s = s }, req)
	res.rid = "0000abcd"
	local done
	coroutine.wrap(function()
		srv:dispatch(req, res)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	local raw = table.concat(out)
	local head, rest = raw:match("^(.-)\r\n\r\n(.*)$")
	local h = {}
	for k, v in head:gmatch("\r\n([^:\r\n]+): ([^\r\n]*)") do
		h[k:lower()] = v
	end
	return tonumber(head:match("^HTTP/1%.1 (%d+)")), h, rest
end

--- As request, with the body decoded as JSON.
function M.json(srv, method, target, body, headers)
	local status, h, text = M.request(srv, method, target, body, headers)
	return status, json.decode(text), h, text
end

--- The named events of an event stream: each is `{ name, data }` with
-- data decoded. The second result is true when each event is exactly
-- an `event:` line, one `data:` line and a blank line.
function M.events(text)
	local list, framed, pos = {}, true, 1
	while pos <= #text do
		local name, data, nxt = text:match("^event: ([^\n]+)\ndata: ([^\n]*)\n\n()", pos)
		if not name then
			framed = false
			break
		end
		list[#list + 1] = { name = name, data = json.decode(data) }
		pos = nxt
	end
	return list, framed
end

return M
