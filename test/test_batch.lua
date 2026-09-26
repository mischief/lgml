-- Batched decode: several slots in one pass give the same tokens as
-- each slot alone. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local tiny = require("tiny")
local transformer = require("lgml.transformer")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = failed + 1
	end
end

local path = tiny.tmp(".gguf")
tiny.write(path, "qwen3", { n_ctx = 2048 })
local be <close> = assert(lgml.backend("CPU"))
local m = transformer.load(be, path)
local vocab = m.hp.n_vocab

-- The argmax of each row of an (S, vocab) logits node.
local function argmax_rows(node, S)
	local v = node:get()
	local out = {}
	for s = 1, S do
		local best, bi = -math.huge, 0
		for j = 1, vocab do
			local x = v[(s - 1) * vocab + j]
			if x > best then
				best, bi = x, j - 1
			end
		end
		out[s] = bi
	end
	return out
end

local prompts = {
	{ 10, 20, 30, 40, 50 },
	{ 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88 },
	{ 7 },
}
local steps = 40

-- Each slot alone.
local want = {}
for i, p in ipairs(prompts) do
	local c = m:cache(2048, { type = "f16", initial = 256 })
	local id = m:step(c, p)
	want[i] = { id }
	for _ = 2, steps do
		id = m:step(c, { id })
		want[i][#want[i] + 1] = id
	end
end

-- All slots in one pass per step; prefill runs alone per slot.
local caches, ids, got = {}, {}, {}
for i, p in ipairs(prompts) do
	caches[i] = m:cache(2048, { type = "f16", initial = 256 })
	ids[i] = m:step(caches[i], p)
	got[i] = { ids[i] }
end
for _ = 2, steps do
	local next_ids = argmax_rows(m:decode_batch(caches, ids), #caches)
	for i = 1, #caches do
		got[i][#got[i] + 1] = next_ids[i]
	end
	ids = next_ids
end
for i = 1, #prompts do
	local same = #got[i] == #want[i]
	for j = 1, #want[i] do
		same = same and got[i][j] == want[i][j]
	end
	ok(same, ("slot %d: batched tokens match the slot alone"):format(i))
	ok(caches[i].n_past == #prompts[i] + steps - 1, ("slot %d: the cache advanced"):format(i))
end

-- Steps with the same spans replay one graph.
local g1 = m.batch_replay.g
m:decode_batch(caches, ids)
ok(m.batch_replay.g == g1, "a step with the same spans replays the batch graph")

-- A slot set in another order is another graph and still correct.
local rev = { caches[3], caches[1] }
local r = argmax_rows(m:decode_batch(rev, { ids[3], ids[1] }), 2)
ok(#r == 2 and m.batch_replay.g ~= g1, "a new slot set builds a new graph")
m:close()
os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
