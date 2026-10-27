-- The file store of the server (lgml.server.fdstore) with a fake
-- service manager, and a file sent over an ipc channel. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "../lua/?.lua;" .. package.path
local uv = require("luv")
local ipc = require("lgml.ipc")
local fdstore = require("lgml.server.fdstore")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = failed + 1
	end
end

local function file(bytes)
	local path = os.tmpname()
	local fd = assert(uv.fs_open(path, "w+", 384))
	os.remove(path)
	assert(uv.fs_write(fd, ("x"):rep(bytes), 0))
	return fd
end

local function open(fd)
	return uv.fs_fstat(fd) ~= nil
end

-- A service manager that passed a listener, a store and a stray file.
local sent = {}
local passed = { { fd = file(10), name = "listen" }, { fd = file(300), name = "pc-a" },
	{ fd = file(1), name = "other" } }
local sd = {
	sd_notify = function(state, fds)
		sent[#sent + 1] = { state = state, n = fds and #fds or 0 }
		return true
	end,
	sd_listen_fds = function()
		return passed
	end,
}

do
	local s = fdstore.new({ sd = sd, ram = 1000 })
	ok(s.listen == passed[1].fd, "the passed listener")
	ok(s.files["pc-a"] and s.files["pc-a"].bytes == 300, "the passed store, with its size")
	ok(not open(passed[3].fd) and sent[1].state == "FDSTOREREMOVE=1\nFDNAME=other",
		"a file of another name is dropped")
	s:put("pc-b", file(400), 400)
	ok(sent[#sent].state == "FDSTORE=1\nFDNAME=pc-b" and sent[#sent].n == 1, "put stores the file")
	local c = file(500)
	s:put("pc-c", c, 500)
	ok(not s.files["pc-a"] and s.files["pc-b"] and s.files["pc-c"], "past the limit the oldest goes")
	local fd = s:take("pc-c")
	ok(fd == c and not s.files["pc-c"] and sent[#sent].state == "FDSTOREREMOVE=1\nFDNAME=pc-c",
		"take gives the file and removes it from the manager")
	uv.fs_close(fd)
	ok(s:take("pc-c") == nil, "a second take gets nothing")
	local b = s.files["pc-b"].fd
	s:prune({ ["pc-x"] = true })
	ok(not s.files["pc-b"] and not open(b), "prune drops the stores of other models")
	local k = #sent
	s:keep_listen(passed[1].fd)
	ok(#sent == k, "the passed listener is kept as it is")
	s:keep_listen(99)
	ok(sent[#sent].state == "FDSTORE=1\nFDNAME=listen" and s.listen == 99, "a new listener goes to the manager")
	s:ready()
	ok(sent[#sent].state == "READY=1", "ready")
	uv.fs_close(passed[1].fd)
	ok(fdstore.state_name("qwen3.8-27b") == "pc-qwen3.8-27b" and fdstore.state_name("a:b/c") == "pc-a_b_c",
		"store names")
end

-- A file goes from worker to front with STATE.
do
	local pair = assert(uv.socketpair("stream", 0, { nonblock = true }, { nonblock = true }))
	local got
	local front = ipc.channel(pair[1], function(t, id, v, fd)
		got = { t = t, id = id, v = v, fd = fd }
	end)
	local worker = ipc.channel(pair[2], function() end)
	local fd = file(64)
	ok(worker:send(ipc.STATE, 0, { bytes = 64 }, fd), "send with a file")
	local t = uv.new_timer()
	t:start(2000, 0, function() end)
	while not got and t:get_due_in() > 0 do
		uv.run("once")
	end
	t:close()
	ok(got and got.t == ipc.STATE and got.v.bytes == 64 and got.fd, "the file arrives")
	ok(got and got.fd and uv.fs_fstat(got.fd).size == 64, "it is the same file")
	if got and got.fd then
		uv.fs_close(got.fd)
	end
	worker:close()
	front:close()
end

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
