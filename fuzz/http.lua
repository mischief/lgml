-- HTTP request heads, query strings and multipart bodies from one
-- connection buffer. Bad input must give nil and a status.
local http = require("lgml.server.http")
local multipart = require("lgml.server.multipart")

local stream = {
	read = function()
		return nil, "eof"
	end,
	write = function()
		return true
	end,
}

return function(s)
	local conn = { s = stream, buf = s }
	for _ = 1, 8 do
		local req = http.read_request(conn)
		if not req then
			break
		end
		http.parse_query(req.query)
		multipart.from_request(req)
	end
end
