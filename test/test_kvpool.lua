-- KV pool: several sequences share one buffer of cells, and a mask per
-- sequence keeps them apart. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local tiny = require("tiny")
local transformer = require("lgml.transformer")
local replay = require("lgml.replay")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = failed + 1
	end
end

local be <close> = assert(lgml.backend("CPU"))

-- The mask of one sequence shows the cells of its positions only.
do
	local t = be:tensor("f32", { 3, 8 })
	-- Block size 2: positions 0-1 in block 3 (cells 6, 7), 2-3 in block 1
	-- (cells 2, 3). The window starts at cell 0.
	t:fill_blocks({ n_past = 1, block = 2, blocks = { 3, 1 }, lo = 0 })
	local v = t:get()
	local want = {
		-- query at position 1: cells 6, 7
		{ [7] = true, [8] = true },
		-- position 2: cells 6, 7, 2
		{ [7] = true, [8] = true, [3] = true },
		-- position 3: cells 6, 7, 2, 3
		{ [7] = true, [8] = true, [3] = true, [4] = true },
	}
	local good = true
	for r = 1, 3 do
		for c = 1, 8 do
			local x = v[(r - 1) * 8 + c]
			good = good and ((want[r][c] and x == 0) or (not want[r][c] and x == -math.huge))
		end
	end
	ok(good, "fill_blocks: each query sees the cells of its earlier positions only")
	t:fill_blocks({ n_past = 1, block = 2, blocks = { 3, 1 }, lo = 0, causal = false })
	v = t:get()
	ok(v[3] == 0 and v[4] == 0, "fill_blocks: a non-causal pass sees its later positions")
	ok(not pcall(t.fill_blocks, t, { n_past = 1, block = 2, blocks = { 3 } }),
	    "fill_blocks: too few blocks is an error")
	ok(not pcall(t.fill_blocks, t, { n_past = 1, block = 2, blocks = { 3, 1 }, lo = 4 }),
	    "fill_blocks: a cell outside the window is an error")
	t:close()
end

local path = tiny.tmp(".gguf")
tiny.write(path, "qwen3", { n_ctx = 2048 })
local m = transformer.load(be, path)

local function maxdiff(a, b)
	local d = #a == #b and 0 or math.huge
	for i = 1, #a do
		d = math.max(d, math.abs(a[i] - (b[i] or math.huge)))
	end
	return d
end

local function argmax(v)
	local b = 1
	for i = 2, #v do
		if v[i] > v[b] then
			b = i
		end
	end
	return b - 1
end

local function ids_of(seed, count)
	local r, out = tiny.rng(seed), {}
	for i = 1, count do
		out[i] = 1 + math.floor(r() * 255)
	end
	return out
end

for _, fa in ipairs({ false, true }) do
	local tag = fa and "flash_attn: " or ""
	local o = { logits = "last", flash_attn = fa }
	local pa, pb = ids_of(1, 150), ids_of(2, 150)

	-- A alone on its own cache.
	local alone = m:cache(1024, { type = "f16" })
	local want = {}
	for s = 1, #pa, 50 do
		want[#want + 1] = replay.run(m, alone, table.move(pa, s, s + 49, 1, {}), o):get()
	end
	local next_a = argmax(want[#want])
	want[#want + 1] = replay.run(m, alone, { next_a }, o):get()

	-- A and B on one pool, in passes of 50 tokens that take turns, so
	-- their blocks interleave.
	local pool = m:pool(1024, { type = "f16", block = 64 })
	local a, b = m:seq(pool), m:seq(pool)
	local got = {}
	for s = 1, #pa, 50 do
		got[#got + 1] = replay.run(m, a, table.move(pa, s, s + 49, 1, {}), o):get()
		replay.run(m, b, table.move(pb, s, s + 49, 1, {}), o)
	end
	local inter = false
	for i = 2, #a.blocks do
		inter = inter or a.blocks[i] ~= a.blocks[i - 1] + 1
	end
	ok(inter, tag .. "the blocks of two sequences interleave (" .. table.concat(a.blocks, ",") .. ")")
	local err = 0
	for i = 1, #got do
		err = math.max(err, maxdiff(got[i], want[i]))
	end
	ok(err == 0, ("%sa lone pass on the pool equals its own cache (max error %.3g)"):format(tag, err))

	-- Write large values into every cell that A does not own; A's next
	-- step does not change. On the CPU, flash attention over 512 or
	-- more cells splits the cells among threads, which changes the
	-- rounding; so the step compares with the same step before the
	-- write, and with the lone cache within a tolerance.
	local before = replay.run(m, a, { next_a }, o):get()
	a.n_past = a.n_past - 1
	local tol = fa and 1e-2 or 0
	local mine = {}
	for _, blk in ipairs(a.blocks) do
		mine[blk] = true
	end
	local row = pool.k[1]:nbytes() // pool.k[1]:shape()[1]
	local junk = lgml.hostbuf(64 * row)
	local jt = be:tensor("f16", { 64, row // 2 })
	jt:fill("uniform", -60, 60, 99)
	jt:to_host(junk)
	for blk = 0, pool.cap // 64 - 1 do
		if not mine[blk] then
			for il in pairs(pool.k) do
				pool.k[il]:from_host(junk, 0, blk * 64 * row, 64 * row)
				pool.v[il]:from_host(junk, 0, blk * 64 * row, 64 * row)
			end
		end
	end
	jt:close()
	junk:close()
	local step = replay.run(m, a, { next_a }, o):get()
	local d = maxdiff(step, before)
	ok(d == 0, ("%sthe mask hides the cells of other sequences (max error %.3g)"):format(tag, d))
	d = maxdiff(step, want[#want])
	ok(d <= tol, ("%sthe step matches the lone cache (max error %.3g)"):format(tag, d))

	-- A snapshot of a pool sequence restores into a plain cache, and a
	-- plain snapshot into a pool sequence.
	local snap = m:snapshot(a, true)
	local plain = m:cache(1024, { type = "f16" })
	m:restore(plain, snap)
	local c = m:seq(pool)
	local snap2 = m:snapshot(alone, true)
	m:restore(c, snap2)
	local x = replay.run(m, plain, { 7 }, o):get()
	local y = replay.run(m, c, { 7 }, o):get()
	local z = replay.run(m, alone, { 7 }, o):get()
	ok(maxdiff(x, z) <= tol and maxdiff(y, z) <= tol,
	    ("%sfull snapshots move between pool sequences and plain caches (max error %.3g, %.3g)")
	    :format(tag, maxdiff(x, z), maxdiff(y, z)))
	snap.buf:close()
	snap2.buf:close()

	-- Truncate gives the blocks back; a new sequence takes the lowest.
	local used = pool.alloc.used
	local nb = #b.blocks
	local low = math.huge
	for _, blk in ipairs(b.blocks) do
		low = math.min(low, blk)
	end
	m:truncate(b, 0)
	ok(#b.blocks == 0 and b.n_past == 0 and pool.alloc.used == used - nb,
	    tag .. "truncate gives back the blocks")
	local d2 = m:seq(pool)
	m:grow(d2, 1)
	ok(d2.blocks[1] == low, tag .. "a new sequence takes the lowest free block")
	for _, s in ipairs({ a, b, c, d2 }) do
		replay.drop(s)
	end
	replay.drop(alone)
	replay.drop(plain)
end

-- A pool that starts small grows and keeps its cells; the pool gives
-- the same tokens as a cache with all rows.
do
	local pool = m:pool(2048, { type = "f16", initial = 256 })
	ok(pool.cap == 256, "the pool starts with 256 cells")
	local a, b = m:seq(pool), m:seq(pool)
	local pa, pb = ids_of(3, 200), ids_of(4, 200)
	local ta = m:step(a, pa)
	local tb = m:step(b, pb)
	ok(pool.cap == 512, "the pool doubled when its free blocks ran out (" .. pool.cap .. ")")
	local c1, c2 = m:cache(2048, { type = "f16" }), m:cache(2048, { type = "f16" })
	ok(ta == m:step(c1, pa) and tb == m:step(c2, pb), "the grown pool keeps the cells of both sequences")
	local full = m:pool(256, { type = "f16" })
	local s = m:seq(full)
	m:step(s, ids_of(5, 250))
	local s2 = m:seq(full)
	local good, err = pcall(m.step, m, s2, ids_of(6, 20))
	ok(not good and tostring(err):find("kv pool full"), "a full pool is an error: " .. tostring(err))
	local called
	full.pressure = function(p, want)
		called = want == s2
		m:truncate(s, 0)
	end
	ok(pcall(m.step, m, s2, ids_of(6, 20)) and called, "a full pool calls pressure, which can free blocks")
	for _, x in ipairs({ a, b, s, s2 }) do
		replay.drop(x)
	end
end

m:close()
os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
