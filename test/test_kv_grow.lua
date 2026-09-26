-- A cache that starts small and grows gives the same logits as a cache
-- with all rows at once. Output is TAP.
local here = (arg[0]:match("^(.*/)") or "./")
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local tiny = require("tiny")
local transformer = require("lgml.transformer")
local gen = require("lgml.generate")

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

local prompt = {}
local r = tiny.rng(7)
for i = 1, 700 do
	prompt[i] = 1 + math.floor(r() * 255)
end

for _, ty in ipairs({ "f16", "f32" }) do
	local copts = { type = ty }
	local fa = ty == "f16"
	local full = m:cache(2048, copts)
	local want = gen.greedy(m, full, prompt, 24, { flash_attn = fa })
	local small = m:cache(2048, { type = ty, initial = 256 })
	ok(small.cap == 256, ty .. ": the cache starts with 256 rows")
	local before = transformer.cache_bytes(small)
	local got = gen.greedy(m, small, prompt, 24, { flash_attn = fa })
	ok(small.cap == 768, ty .. ": the cache grew to the 768 rows the prompt needs (got " .. small.cap .. ")")
	ok(transformer.cache_bytes(small) > before and transformer.cache_bytes(small) < transformer.cache_bytes(full),
	    ty .. ": the grown cache is larger than at first and smaller than the full one")
	local same = #got == #want
	for i = 1, #want do
		same = same and got[i] == want[i]
	end
	ok(same, ty .. ": greedy tokens match the full cache")
end

-- Growing in small steps during decode keeps the earlier rows.
local c = m:cache(2048, { type = "f16", initial = 256 })
local a = gen.greedy(m, c, { 5, 6, 7 }, 300)
local c2 = m:cache(2048, { type = "f16" })
local b = gen.greedy(m, c2, { 5, 6, 7 }, 300)
local same = #a == #b
for i = 1, #b do
	same = same and a[i] == b[i]
end
ok(same and c.cap == 512, "decode past the first rows grows the cache and keeps the tokens")

-- A full snapshot restores into a smaller cache by growing it.
local src = m:cache(2048, { type = "f16" })
gen.greedy(m, src, prompt, 1)
local snap = m:snapshot(src, true)
local dst = m:cache(2048, { type = "f16", initial = 256 })
m:restore(dst, snap)
ok(dst.cap >= snap.n and dst.n_past == snap.n, "restore grows the cache to the snapshot rows")
local x = gen.greedy(m, src, { 9 }, 8)
local y = gen.greedy(m, dst, { 9 }, 8)
same = #x == #y
for i = 1, #x do
	same = same and x[i] == y[i]
end
ok(same, "the restored cache continues like the source")
snap.buf:close()

-- A pass past n_ctx is still an error.
local full = m:cache(256, { type = "f16", initial = 256 })
ok(not pcall(m.step, m, full, prompt), "a pass past n_ctx is an error")
m:close()
os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
