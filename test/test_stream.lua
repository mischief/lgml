-- lgml.server.stream: a write to a peer that does not read ends at the
-- write deadline. Output is TAP.
local uv = require("luv")
local stream = require("lgml.server.stream")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	failed = failed or not cond
end

stream.WRITE_TIMEOUT_MS = 300
local server = uv.new_tcp()
server:bind("127.0.0.1", 0)
local port = server:getsockname().port
local result
server:listen(1, function()
	local c = uv.new_tcp()
	server:accept(c)
	coroutine.wrap(function()
		local s = stream.new(c)
		local chunk = string.rep("x", 1 << 20)
		local t0 = uv.now()
		local r, err
		for _ = 1, 256 do
			r, err = s:write(chunk)
			if not r then
				break
			end
		end
		result = { r = r, err = err, ms = uv.now() - t0, gone = s:gone() }
		c:close()
	end)()
end)
-- A client that connects and never reads.
local client = uv.new_tcp()
client:connect("127.0.0.1", port, function() end)
local t = uv.new_timer()
t:start(10000, 0, function() uv.stop() end)
while not result and uv.run("once") do
end
t:close()
ok(result and result.r == nil and result.err == "write timeout",
    "a write to a peer that does not read ends with write timeout")
ok(result and result.gone, "the stream is gone after the timeout")
ok(result and result.ms < 5000, ("it ends soon (%d ms)"):format(result and result.ms or -1))
client:close()
server:close()
print("1.." .. n)
os.exit(failed and 1 or 0)
