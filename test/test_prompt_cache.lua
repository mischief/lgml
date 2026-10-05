-- The prompt cache (lgml.prompt_cache) with a fake model: buffer
-- references across the store, the state index and the slot. Output
-- is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "../lua/?.lua;" .. package.path
local prompt_cache = require("lgml.prompt_cache")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = failed + 1
	end
end

-- A model whose snapshots are size bytes each. restore fails on a
-- closed buffer, as the backend does.
local function fake(size)
	local m = { recurrent = true }
	function m:snapshot(cache)
		local buf = {}
		function buf:close()
			self.closed = true
		end
		return { n = cache.n_past, bytes = size, buf = buf }
	end
	function m:snapshot_bytes()
		return size
	end
	function m:restore(cache, s)
		assert(not s.buf.closed, "hostbuf closed")
		cache.n_past = s.n
	end
	return m
end

local function tokens(k)
	local t = {}
	for i = 1, k do
		t[i] = i
	end
	return t
end

-- A state index entry that the slot does not hold, and a save that
-- needs its room: the slot then starts from the entry.
local function start_from_state(keep)
	local cache = { n_past = 40, n_ctx = 1024 }
	local pc = prompt_cache.new(fake(60), cache, { ram = 100, checkpoints = 4 })
	pc:checkpoint()
	pc:add_state("root", tokens(40), pc.ckpts[1])
	pc:trim(-1)
	local s = pc:find_states("root", tokens(50), 1, 49)[1]
	local eng = { tokens = tokens(64) }
	cache.n_past = 64
	pc:save(eng, keep and s or nil)
	return pc, cache, s
end

do
	local pc, cache, s = start_from_state(true)
	ok(s and not s.ck.buf.closed, "save keeps the state entry it starts from")
	ok(#pc.store == 0, "save without room does not go to the store")
	ok(pcall(pc.model.restore, pc.model, cache, s.ck), "restore of the kept entry")
	pc:adopt(s)
	ok(pc.ckpts[1] == s.ck and s.ck.refs == 2, "adopt holds the entry checkpoint")
	pc:clear()
	ok(s.ck.buf.closed, "clear frees the checkpoint")
end

do
	local pc, _, s = start_from_state(false)
	ok(s.ck.buf.closed and #pc.states == 0 and #pc.store == 1,
		"save without keep evicts the state entry")
	pc:clear()
end

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
