--- Files that outlive a worker or the server: the prompt stores that
-- workers save as they exit (see `lgml.ipc.STATE`), and the listening
-- socket.
-- This process holds each store file until the next worker of the model
-- takes it. Under systemd with `FileDescriptorStoreMax=`, the service
-- manager holds a copy too, and gives the files back to the next server
-- process (see `lgml.sd_listen_fds`). Without the `lgml` module, or
-- without systemd, the files live only in this process.
-- @module lgml.server.fdstore
local uv = require("luv")
local log = require("lgml.log")

local L = log.new("fdstore")

local M = {}
local S = {}
S.__index = S

--- The name of the listening socket.
M.LISTEN = "listen"

--- The store name of a model's prompt store.
-- @tparam string id the model id
-- @treturn string the name: `"pc-"` and the id, with characters other
-- than letters, digits, `.`, `-` and `_` as `_`
function M.state_name(id)
	return "pc-" .. tostring(id):gsub("[^%w%.%-_]", "_"):sub(1, 200)
end

--- Makes a store and takes the files that the service manager passed.
-- @tparam[opt] table opts options:
--
-- - `ram`: the most bytes of prompt store files to hold (default 4 GiB);
--   past it the oldest go
-- - `sd`: the functions `sd_notify` and `sd_listen_fds` (default: the
--   `lgml` module's, when it has them)
-- @treturn Store the store
function M.new(opts)
	opts = opts or {}
	local sd = opts.sd
	if not sd then
		local ok, lgml = pcall(require, "lgml")
		sd = ok and lgml.sd_notify and lgml or {}
	end
	local s = setmetatable({ sd = sd, ram = opts.ram or 4 << 30, files = {}, tick = 0, listen = nil }, S)
	for _, e in ipairs(sd.sd_listen_fds and sd.sd_listen_fds() or {}) do
		if e.name == M.LISTEN and not s.listen then
			s.listen = e.fd
		elseif e.name:match("^pc%-") and not s.files[e.name] then
			local st = uv.fs_fstat(e.fd)
			s:hold(e.name, e.fd, st and st.size or 0)
		else
			L:info("dropped a stored file", "name", e.name)
			s:notify("FDSTOREREMOVE=1\nFDNAME=" .. e.name)
			uv.fs_close(e.fd)
		end
	end
	return s
end

--- A store of files.
-- Fields: `listen`, the listening socket that the service manager
-- passed, or nil.
-- @type Store

function S:notify(state, fds)
	if self.sd.sd_notify then
		local ok, err = self.sd.sd_notify(state, fds)
		if ok == nil then
			L:warn("sd_notify", "err", err)
		end
		return ok
	end
	return false
end

function S:hold(name, fd, bytes)
	self.tick = self.tick + 1
	self.files[name] = { fd = fd, bytes = bytes, tick = self.tick }
	-- The oldest files go past the byte limit.
	while true do
		local total, old = 0, nil
		for n, f in pairs(self.files) do
			total = total + f.bytes
			if n ~= name and (not old or f.tick < self.files[old].tick) then
				old = n
			end
		end
		if total <= self.ram or not old then
			break
		end
		L:info("dropped a prompt store", "name", old, "mib", self.files[old].bytes // 1048576)
		self:drop(old)
	end
end

--- Tells the service manager that the server is ready.
function S:ready()
	self:notify("READY=1")
end

--- Gives the listening socket to the service manager to keep.
-- This process keeps its own fd.
-- @tparam int fd the socket
function S:keep_listen(fd)
	if self.listen ~= fd then
		self:notify("FDSTOREREMOVE=1\nFDNAME=" .. M.LISTEN)
		self:notify("FDSTORE=1\nFDNAME=" .. M.LISTEN, { fd })
		self.listen = fd
	end
end

--- Holds a file under a name, in place of an older one.
-- The store owns fd.
-- @tparam string name the name (see `state_name`)
-- @tparam int fd the file
-- @tparam int bytes its size
function S:put(name, fd, bytes)
	self:drop(name)
	self:notify("FDSTORE=1\nFDNAME=" .. name, { fd })
	self:hold(name, fd, bytes)
end

--- Takes a file out of the store. The caller owns it.
-- @tparam string name the name
-- @treturn ?int the file, or nil
function S:take(name)
	local f = self.files[name]
	if not f then
		return nil
	end
	self.files[name] = nil
	self:notify("FDSTOREREMOVE=1\nFDNAME=" .. name)
	return f.fd
end

--- Closes the file of a name, if there is one.
-- @tparam string name the name
function S:drop(name)
	local fd = self:take(name)
	if fd then
		uv.fs_close(fd)
	end
end

--- Closes the files whose names are not in keep.
-- @tparam table keep a set of names
function S:prune(keep)
	for name in pairs(self.files) do
		if not keep[name] then
			L:info("dropped a prompt store", "name", name)
			self:drop(name)
		end
	end
end

return M
