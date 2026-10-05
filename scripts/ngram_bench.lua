-- Time lgml.ngram reads of a table on disk.
-- usage: lua ngram_bench.lua table.gguf meta.gguf [mode] [tokens] [seed] [threads] [batch] [align]
-- Token ids are random, so each seed reads rows that are likely not in
-- the page cache. Pass 1 reads them cold; pass 2 reads the same rows
-- again. batch 1 is decode, one call per token; a larger batch is
-- prefill, that many tokens per call.
local lgml = require("lgml")

local path, meta = arg[1], arg[2]
local mode = arg[3] or "direct"
local ntok = tonumber(arg[4] or "256")
local seed = tonumber(arg[5] or "1")
local threads = tonumber(arg[6] or "8")
local batch = tonumber(arg[7] or "1")
local align = tonumber(arg[8] or "0")
assert(path and meta, "usage: ngram_bench.lua table.gguf meta.gguf [mode] [tokens] [seed] [threads] [batch] [align]")

local r <close> = assert(lgml.ngram.open(path, { meta = meta, mode = mode, threads = threads, align = align }))
local info = r:info()
math.randomseed(seed)
local ids = {}
for i = 1, ntok do
	ids[i] = math.random(0, 248000)
end
local out = lgml.hostbuf(batch * info.dim * 4)

local function pass()
	r:stats(true)
	local t0 = lgml.clock()
	local worst = 0
	for i = 1, ntok, batch do
		local t1 = lgml.clock()
		local chunk = { table.unpack(ids, i, math.min(i + batch - 1, ntok)) }
		r:embed(chunk, { table.unpack(ids, math.max(1, i - 2), i - 1) }, out)
		worst = math.max(worst, lgml.clock() - t1)
	end
	local wall = lgml.clock() - t0
	local st = r:stats()
	return st, wall, worst
end

print(string.format("mode %s%s, align %d, threads %d, %d tokens, batch %d, seed %d",
	info.mode, info.fell_back and " (fell back)" or "", info.align, threads, ntok, batch, seed))
for p = 1, 2 do
	local st, wall, worst = pass()
	print(string.format("pass %d: %.1f rows/tok, %.1f unique/tok, %.1f reads/tok, " ..
		"%.0f bytes/tok, %.3f ms/tok (reads %.3f), worst call %.3f ms",
		p, st.rows / ntok, st.unique / ntok, st.reads / ntok, st.bytes / ntok,
		wall * 1e3 / ntok, st.read_sec * 1e3 / ntok, worst * 1e3))
end
