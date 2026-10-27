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

-- dump and undump through a file, with real host buffers: a
-- checkpoint that two entries share is written once.
do
	local lgml = require("lgml")
	local uv = require("luv")
	local function snap(n, byte, size)
		local buf = lgml.hostbuf(size)
		buf:set(0, string.char(byte):rep(size))
		return { n = n, bytes = size, full = true, buf = buf }
	end
	local path = os.tmpname()
	local fd = assert(lgml.memfd and lgml.memfd("test") or uv.fs_open(path, "w+", 384))
	local pc = prompt_cache.new(fake(0), { n_past = 0 }, { ram = 1 << 20 })
	local ck = snap(10, 7, 100)
	ck.refs = 2
	pc.store = {
		{ tokens = tokens(40), ckpts = { ck }, bytes = 400, snap = snap(40, 1, 300), used = 1, key = "a", refs = 1 },
		{ tokens = tokens(50), ckpts = { ck }, bytes = 600, snap = snap(50, 2, 500), used = 2, key = "a" },
	}
	local n = pc:dump(fd, "fp1")
	ok(n and n == 4096 + 900, "dump writes the shared checkpoint once")
	local q = prompt_cache.new(fake(0), { n_past = 0 }, { ram = 1 << 20 })
	ok(select(2, q:undump(fd, "fp2")) == "another model, setting or code", "undump refuses another fingerprint")
	ok(q:undump(fd, "fp1") == 2 and #q.store == 2, "undump reads the entries")
	local e1, e2 = q.store[1], q.store[2]
	ok(#e1.tokens == 40 and e1.snap.buf:bytes(299, 1) == "\1" and e2.snap.buf:bytes(0, 1) == "\2",
		"entries keep their tokens and bytes")
	ok(e1.ckpts[1] == e2.ckpts[1] and e1.ckpts[1].refs == 2 and e1.ckpts[1].buf:bytes(0, 1) == "\7",
		"the shared checkpoint stays shared")
	ok(e1.used < e2.used and q.bytes == 1000, "the order of use and the byte count")
	local small = prompt_cache.new(fake(0), { n_past = 0 }, { ram = 700 })
	ok(small:undump(fd, "fp1") == 1 and #small.store[1].tokens == 50, "the newest entry that fits goes in")
	q:clear()
	small:clear()
	uv.fs_close(fd)
	os.remove(path)
end

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
