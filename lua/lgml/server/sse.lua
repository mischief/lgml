--- Server-sent events on a streamed HTTP response.
-- Each event is one write, so each event reaches the client at once.
-- @module lgml.server.sse

local json = require("lgml.json")

local M = {}
local W = {}
W.__index = W

--- Starts the event stream on res.
-- It sends status 200 with the event stream headers.
-- @tparam lgml.server.http.Response res the response
-- @tparam[opt] table headers more headers; the table is changed
-- @treturn[1] Writer the writer
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.start(res, headers)
	headers = headers or {}
	headers["Content-Type"] = "text/event-stream"
	headers["Cache-Control"] = "no-cache"
	headers["X-Accel-Buffering"] = "no"
	local ok, err = res:start(200, headers)
	if not ok then
		return nil, err
	end
	return setmetatable({ res = res }, W)
end

--- An event stream writer.
-- @type Writer

--- Sends value as JSON in one `data:` event.
-- @param value the value
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @raise when value cannot be encoded as JSON
function W:send(value)
	return self.res:write("data: " .. assert(json.encode(value)) .. "\n\n")
end

--- Sends a comment line. Clients ignore it.
-- @tparam string text the comment
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
function W:comment(text)
	return self.res:write(": " .. text .. "\n\n")
end

--- Sends the `[DONE]` event and ends the body.
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
function W:done()
	local ok, err = self.res:write("data: [DONE]\n\n")
	if not ok then
		return nil, err
	end
	return self.res:finish()
end

return M
