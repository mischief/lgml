-- Time prompt processing and token generation of a model, as tokens/s.
-- usage: lua bench.lua model.gguf [backend] [pp] [tg] [reps]
-- Token ids are random. Each rep fills a new cache with pp tokens in one
-- pass, then decodes tg tokens one at a time. Prints the median rep.
local lgml = require("lgml")
local transformer = require("lgml.transformer")
local replay = require("lgml.replay")

local path = assert(arg[1], "usage: bench.lua model.gguf [backend] [pp] [tg] [reps]")
local bname = arg[2]
local pp, tg, reps = tonumber(arg[3] or "512"), tonumber(arg[4] or "128"), tonumber(arg[5] or "3")

local be = assert(lgml.backend(bname))
local t0 = lgml.clock()
local model = transformer.load(be, path)
local load = lgml.clock() - t0
local nv = model.hp.n_vocab
math.randomseed(1)
local function ids(n)
	local t = {}
	for i = 1, n do
		t[i] = math.random(100, nv - 1)
	end
	return t
end

local function median(t)
	table.sort(t)
	return t[(#t + 1) // 2]
end
local pps, tgs = {}, {}
-- rep 0 warms up: it builds the pipelines and graphs
for rep = 0, reps do
	local cache = model:cache(pp + tg + 8)
	local p = ids(pp)
	local t1 = lgml.clock()
	replay.run(model, cache, p, { logits = "last" })
	local t2 = lgml.clock()
	for _, id in ipairs(ids(tg)) do
		replay.run(model, cache, { id }, { logits = "last" })
	end
	local t3 = lgml.clock()
	if rep > 0 then
		pps[#pps + 1] = pp / (t2 - t1)
		tgs[#tgs + 1] = tg / (t3 - t2)
	end
end
print(("%s  %s  load %.1fs  pp%d %.1f t/s  tg%d %.2f t/s"):format(path:match("[^/]*$"), be:name(),
    load, pp, median(pps), tg, median(tgs)))
