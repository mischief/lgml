--- A bounded in-memory store of Responses API responses.
-- Each entry holds one response: its input items, its output items,
-- the Response object and the id of the response before it. A chain
-- of `previous_response_id` links is read back entry by entry. The
-- store drops the least recently used entries when it holds too many
-- or too many bytes, and drops entries older than the age limit.
-- @module lgml.server.response_store
local json = require("lgml.json")

local M = {}

--- Default limits.
-- @table LIMITS
-- @field entries the most entries (1024)
-- @field bytes the most bytes of JSON (256 MiB)
-- @field age the most seconds since an entry was made (86400)
M.LIMITS = { entries = 1024, bytes = 256 << 20, age = 86400 }

--- Makes a store.
-- @tparam[opt] table limits limits to change, as `LIMITS`
-- @tparam[opt] function clock returns the time in seconds (default os.time)
-- @treturn Store the store
function M.new(limits, clock)
	local lim = {}
	for k, v in pairs(M.LIMITS) do
		lim[k] = limits and limits[k] or v
	end
	return setmetatable({ lim = lim, map = {}, count = 0, bytes = 0, tick = 0, clock = clock or os.time },
		{ __index = M.Store })
end

--- A store.
-- @type Store
M.Store = {}

local function remove(self, id)
	local e = self.map[id]
	if e then
		self.map[id] = nil
		self.count = self.count - 1
		self.bytes = self.bytes - e.bytes
	end
end

local function expired(self, e)
	return self.clock() - e.created > self.lim.age
end

-- Drops old entries, then the least recently used ones over the limits.
local function trim(self)
	for id, e in pairs(self.map) do
		if expired(self, e) then
			remove(self, id)
		end
	end
	while self.count > self.lim.entries or self.bytes > self.lim.bytes do
		local lru
		for _, e in pairs(self.map) do
			if not lru or e.used < lru.used then
				lru = e
			end
		end
		remove(self, lru.id)
	end
end

--- Adds an entry.
-- @tparam table e `{ id, prev, input, output, response }`; input and
-- output are item lists, response the Response object
function M.Store:put(e)
	remove(self, e.id)
	e.bytes = #assert(json.encode(e.input)) + #assert(json.encode(e.response))
	e.created = self.clock()
	self.tick = self.tick + 1
	e.used = self.tick
	self.map[e.id] = e
	self.count = self.count + 1
	self.bytes = self.bytes + e.bytes
	trim(self)
end

--- Returns an entry and marks it used.
-- @tparam string id the response id
-- @treturn ?table the entry, or nil when it is unknown or too old
function M.Store:get(id)
	local e = self.map[id]
	if e and expired(self, e) then
		remove(self, id)
		return nil
	end
	if e then
		self.tick = self.tick + 1
		e.used = self.tick
	end
	return e
end

--- Removes an entry.
-- @tparam string id the response id
-- @treturn boolean true when the store had it
function M.Store:delete(id)
	local had = self:get(id) ~= nil
	remove(self, id)
	return had
end

--- The chain of entries that ends at id, oldest first.
-- @tparam string id the last response id
-- @treturn[1] {table,...} the entries
-- @treturn[2] nil
-- @treturn[2] string the id that is not in the store
function M.Store:chain(id)
	local list, seen = {}, {}
	while id do
		if seen[id] then
			break
		end
		seen[id] = true
		local e = self:get(id)
		if not e then
			return nil, id
		end
		table.insert(list, 1, e)
		id = e.prev
	end
	return list
end

--- Finds a stored input or output item by its id.
-- @tparam string id the item id
-- @treturn ?table the item
function M.Store:item(id)
	for _, e in pairs(self.map) do
		if not expired(self, e) then
			for _, list in ipairs({ e.input, e.output }) do
				for _, it in ipairs(list) do
					if it.id == id then
						return it
					end
				end
			end
		end
	end
	return nil
end

return M
