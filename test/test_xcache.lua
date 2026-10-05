-- The expert cache (lgml.xcache): slot state and plans, and tiny MoE
-- models whose logits with a cache of any size equal those of one
-- backend, with slot changes between passes and batched decode.
-- The KV cache is f32: f16 rounding would grow small sum-order
-- differences. LGML_DEVICE picks the device (default CPU, with a second
-- CPU backend for the experts). Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local tiny = require("tiny")
local transformer = require("lgml.transformer")
local placement = require("lgml.placement")
local replay = require("lgml.replay")
local xcache = require("lgml.xcache")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = failed + 1
	end
end

local DEV = os.getenv("LGML_DEVICE") or "CPU"
-- On a GPU, the CPU and the device quantize the activations of q8_0
-- weights in other ways, and a close router pick can flip.
local TOLS = DEV == "CPU" and { split = 1e-4, fused = 1e-4, q8_0 = 1e-4 }
    or { split = 2e-3, fused = 2e-3, q8_0 = 0.3 }

-- Slot state.
do
	local st = lgml.expert_cache(2, 6)
	st:resize(1, 3)
	local plan = st:plan(10)
	ok(#plan == 3, "a plan fills every empty slot")
	for _, c in ipairs(plan) do
		st:assign(c[1], c[2], c[3])
	end
	local map, hit, ex = st:tables(1)
	ok(#map == 6 and #hit == 6 and #ex == 3, "tables have one entry per expert and per slot")
	local held = 0
	for e = 1, 6 do
		held = held + hit[e]
		if hit[e] == 1 then
			ok(ex[map[e] + 1] == e - 1, ("expert %d is in slot %d"):format(e - 1, map[e]))
		end
	end
	ok(held == 3, "three experts are in slots")
	-- Scores: expert 5 is hot; it replaces the coldest held expert.
	st:seed(1, { 1, 2, 3, 0, 0, 40 })
	plan = st:plan(10, 1, 0.25)
	ok(#plan == 1 and plan[1][3] == 5, "a hot expert gets a slot")
	ok(st:assign(1, plan[1][2], 5) >= 0, "assign gives the old expert")
	ok(#st:plan(10, 1, 0.25) == 0, "the margin keeps the slots stable")
	st:seed(1, { 100, 100, 100, 100, 100, 100 })
	ok(#st:plan(1, 1, 0) <= 1, "a plan takes at most n changes")
	local s = st:scores(1)
	ok(#s == 6, "scores have one entry per expert")
	ok(not pcall(st.assign, st, 1, 3, 0), "a slot out of range is an error")
	ok(not pcall(st.resize, st, 3, 1), "a layer out of range is an error")
	st:resize(2, 6)
	ok(#st:plan(100) == 6, "a layer with a slot for each expert fills too")
end

-- set_slice copies one expert.
do
	local cb <close> = assert(lgml.backend("CPU"))
	local a = cb:tensor("f32", { 3, 2, 4 })
	a:fill("range", 0, 1)
	local b = cb:tensor("f32", { 2, 2, 4 })
	b:zero()
	b:set_slice(1, a, 2)
	local v = b:get()
	ok(v[1] == 0 and v[9] == 16 and v[16] == 23, "set_slice copies slice 2 to slice 1")
	ok(not pcall(b.set_slice, b, 2, a, 0), "a slice out of range is an error")
end

-- Logits of ids, one pass per token, through replay.
local function rows(m, ids)
	local c = m:cache(128, { type = "f32" })
	local out = {}
	for i, id in ipairs(ids) do
		out[i] = replay.run(m, c, { id }, { logits = "last" }):get()
	end
	return out
end

-- The last logits of one pass of all ids.
local function batch_row(m, ids)
	local c = m:cache(128, { type = "f32" })
	return { replay.run(m, c, ids, { logits = "last" }):get() }
end

local function maxdiff(a, b)
	if #a ~= #b then
		return math.huge
	end
	local d = 0
	for i = 1, #a do
		for j = 1, #a[i] do
			d = math.max(d, math.abs(a[i][j] - (b[i][j] or math.huge)))
		end
	end
	return d
end

local ids = {}
for i = 1, 24 do
	ids[i] = (i * 37) % 250 + 1
end

local NE = 16
for _, kind in ipairs({ "split", "fused", "q8_0" }) do
	local path = tiny.tmp(".gguf")
	tiny.write(path, "qwen3moe", { n_ctx = 512, quant = kind == "q8_0" and "q8_0" or nil,
	    moe = { n = NE, used = 4, ff = 32, fused = kind == "fused" } })
	local TOL = TOLS[kind]
	local one <close> = assert(lgml.backend(DEV))
	local want = rows(transformer.load(one, path), ids)
	local want_batch = batch_row(transformer.load(one, path), ids)
	local be <close> = assert(lgml.backend(DEV, { cpu = true }))
	local rules = placement.rules({ cpu_moe = true })
	for _, slots in ipairs({ 0, 1, 5, NE - 1, NE }) do
		local tag = ("%s, %d slots: "):format(kind, slots)
		local m = transformer.load(be, path, { placement = rules })
		local xc = xcache.new(m, { slots = slots, swap = 1 << 30, halflife = 3, margin = 0 })
		ok(xc.n == slots and m.layers[1].xc ~= nil, tag .. "the cache is on each layer")
		local d = maxdiff(want, rows(m, ids))
		ok(d < TOL, ("%slogits equal one backend (max error %.3g)"):format(tag, d))
		local h, mi = xc:stats()
		if slots > 0 and slots < NE then
			ok(h > 0 and mi > 0 and h + mi == #ids * 4 * #m.layers,
			    ("%scounts every pick (%d held, %d not)"):format(tag, h, mi))
			ok(xc.moved > xc.bytes // 2, tag .. "experts moved between passes")
		end
		-- A batch rounds in other ways on a GPU: the device computes
		-- the expert matmuls in f16 tiles.
		d = maxdiff(want_batch, batch_row(m, ids))
		ok(d < (DEV == "CPU" and TOL or 1e-2),
		    ("%sa batch pass equals one backend (max error %.3g)"):format(tag, d))
		xc:close()
		ok(m.layers[1].xc == nil and m.before_pass == nil, tag .. "close detaches the cache")
	end
	-- The rest of the model on the CPU too: the router runs there.
	do
		local m = transformer.load(be, path, { placement = placement.rules({ cpu_moe = true,
		    n_gpu_layers = 0 }) })
		local xc = xcache.new(m, { slots = 5, swap = 1 << 30, halflife = 3, margin = 0 })
		local d = maxdiff(want, rows(m, ids))
		ok(xc.layers[1].router_cpu and d < TOL,
		    ("%s, router on the CPU: logits equal one backend (max error %.3g)"):format(kind, d))
	end
	-- Slots that change before every pass, at most one expert each.
	do
		local m = transformer.load(be, path, { placement = rules })
		local xc = xcache.new(m, { slots = 6, swap = 1, halflife = 1, margin = 0 })
		xc.swap = xc.expert_bytes
		local d = maxdiff(want, rows(m, ids))
		ok(d < TOL, ("%s, one change per pass: logits equal one backend (max error %.3g)")
		    :format(kind, d))
	end
	-- Batched decode of three sequences across the slots.
	do
		local prompts = { { 10, 20, 30, 40 }, { 99, 98, 97 }, { 7 } }
		local steps = 8
		local function run(m)
			local pool = m:pool(3 * 256, { type = "f32" })
			local caches, cur, out = {}, {}, {}
			for i, p in ipairs(prompts) do
				caches[i] = m:seq(pool, { n_ctx = 128 })
				cur[i] = m:step(caches[i], p)
				out[i] = {}
			end
			for _ = 1, steps do
				local rs = m:decode_batch(caches, cur)
				for i, r in ipairs(rs) do
					local v = r:get()
					out[i][#out[i] + 1] = v
					local best, bi = -math.huge, 0
					for j = 1, #v do
						if v[j] > best then
							best, bi = v[j], j - 1
						end
					end
					cur[i] = bi
				end
			end
			return out
		end
		local static = transformer.load(be, path, { placement = rules })
		local base = run(static)
		local m = transformer.load(be, path, { placement = rules })
		local xc = xcache.new(m, { slots = 5, swap = 1 << 30, halflife = 2, margin = 0 })
		local got = run(m)
		local d = 0
		for i = 1, #prompts do
			d = math.max(d, maxdiff(base[i], got[i]))
		end
		ok(d < TOL, ("%s, batched decode: logits equal the static placement (max error %.3g)")
		    :format(kind, d))
		ok(select(3, xc:stats()) > 0, kind .. ", batched decode: some picks hit the cache")
	end
	-- A profile seeds the scores, and the slots take its hot experts.
	do
		local m = transformer.load(be, path, { placement = rules })
		local xc = xcache.new(m, { slots = 3, swap = 0 })
		rows(m, ids)
		local file = tiny.tmp(".prof")
		xc:save(file)
		local prof = xc:profile()
		xc:close()
		local m2 = transformer.load(be, path, { placement = rules })
		local xc2 = xcache.new(m2, { slots = 3, profile = file })
		local s = prof[1]
		local order = {}
		for e = 1, NE do
			order[e] = e - 1
		end
		table.sort(order, function(a, b)
			return s[a + 1] > s[b + 1] or (s[a + 1] == s[b + 1] and a < b)
		end)
		local _, hit = xc2.state:tables(1)
		ok(hit[order[1] + 1] == 1 and hit[order[2] + 1] == 1 and hit[order[3] + 1] == 1,
		    kind .. ": a profile puts its hot experts in the slots")
		local d = maxdiff(want, rows(m2, ids))
		ok(d < TOL, ("%s, from a profile: logits equal one backend (max error %.3g)"):format(kind, d))
		os.remove(file)
	end
	os.remove(path)
end

-- A cache needs experts on the CPU.
do
	local path = tiny.tmp(".gguf")
	tiny.write(path, "qwen3moe", { moe = { n = 4, used = 2, ff = 16 } })
	local be <close> = assert(lgml.backend(DEV, { cpu = true }))
	local m = transformer.load(be, path)
	ok(not pcall(xcache.new, m, { slots = 1 }), "no cache for a model with experts on the device")
	local mc = transformer.load(be, path, { placement = placement.rules({ cpu_moe = true }) })
	local layers, sum = xcache.layers(mc)
	ok(layers[1] and layers[2] and sum > 0, "layers lists the layers with experts on the CPU")
	ok(xcache.slots_for(mc, sum * 3 + 1024) == 1 and xcache.slots_for(mc, sum * 4 + 1024) == 4,
	    "slots_for divides the bytes by the layers, less a zero slot per pick")
	ok(xcache.slots_for(mc, 1 << 40) == 4, "slots_for gives at most the expert count")
	os.remove(path)
end

-- The estimate of a server: experts on the CPU part, slots on the GPU.
do
	local estimate = require("lgml.estimate")
	local path = tiny.tmp(".gguf")
	tiny.write(path, "qwen3moe", { n_ctx = 512, moe = { n = NE, used = 4, ff = 32 } })
	local plain = estimate.model({ model = path, backend = "Vulkan0", ctx = 256 }, math.huge, true)
	local r = estimate.model({ model = path, backend = "Vulkan0", ctx = 256, moe_cache = 0.5 },
	    math.huge, true)
	ok(r.moe_cache == 524288 and r.extra.cpu > 0 and r.files < plain.files and
	    r.bytes == r.files + r.kv + r.margin + r.moe_cache,
	    "estimate: experts on the CPU, the slots on the GPU")
	local budget = plain.bytes + (1 << 20)
	r = estimate.model({ model = path, backend = "Vulkan0", ctx = 256, moe_cache = "auto",
	    moe_cache_max = 0.75 }, budget, true)
	ok(r.moe_cache_max == 786432 and r.bytes <= budget,
	    "estimate: auto takes the rest of the budget, at most the max")
	r = estimate.model({ model = path, backend = "Vulkan0", ctx = 256, moe_cache = "auto" },
	    r.files + r.kv + r.margin + 4096, true)
	ok(r.moe_cache_max == 4096, "estimate: auto under a budget gives its cap")
	os.remove(path)
end

-- The engine on a GPU: a cache of a fixed size, and one that fills the
-- free device memory up to moe_cache_max.
if DEV ~= "CPU" then
	local uv = require("luv")
	local engine = require("lgml.engine")
	local budget = require("lgml.budget")
	local path = tiny.tmp(".gguf")
	tiny.write(path, "qwen3moe", { n_ctx = 512, moe = { n = NE, used = 4, ff = 32 } })
	for _, v in ipairs({ 0.5, "auto" }) do
		local tag = ("engine, moe_cache %s: "):format(v)
		local e = engine.new({ model = path, backend = DEV, n_ctx = 256, slots = 1,
		    moe_cache = v, moe_cache_max = 0.25 })
		local xc = e.model.xcache
		ok(xc and xc.n > 0 and e.model.layers[1].down_exps:backend() == e.be:cpu(),
		    tag .. "experts in host memory, slots on the device")
		ok(xc and xc.bytes <= (v == "auto" and 0.25 or v) * 1048576,
		    tag .. ("the slots fit the size (%d bytes)"):format(xc and xc.bytes or -1))
		local r = budget.report(e)
		ok(r.weights >= xc.bytes, tag .. "the report counts the slots as weights")
		local res, done = nil, false
		local sink = {
			content = function() return true end,
			finish = function(_, _, usage) res = usage end,
			cancelled = function() return false end,
		}
		sink.reasoning, sink.tool_arguments = sink.content, sink.content
		coroutine.wrap(function()
			e:chat({ messages = { { role = "user", content = "hello there" } }, max_tokens = 8,
			    top_k = 1 }, sink)
			done = true
		end)()
		while not done do
			uv.run("once")
		end
		local h, mi = xc:stats()
		ok(res and res.completion_tokens >= 1 and h + mi > 0,
		    tag .. ("a chat runs through the cache (%d of %d picks held)"):format(h, h + mi))
	end
	os.remove(path)
end

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
