-- Compute memory of the chat engine over many requests: replaced graphs
-- give their buffers back at once, kept graphs stay within the plan, and
-- a pass past the memory budget frees graphs and runs again instead of
-- failing. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local uv = require("luv")
local lgml = require("lgml")
local engine = require("lgml.engine")
local gguf = require("gguf")
local tiny = require("tiny")
local q4t = require("q4_tiny")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function cpu()
	return lgml.memory().CPU or { weights = 0, tensors = 0, compute = 0, peak = 0, peak_compute = 0 }
end

local function chat(e, req)
	local r = {}
	r.sink = {
		content = function() return true end,
		reasoning = function() return true end,
		tool_call = function() return 0 end,
		tool_arguments = function() return true end,
		finish = function(_, reason, usage, timings) r.reason, r.usage, r.timings = reason, usage, timings end,
		cancelled = function() return false end,
	}
	local done
	coroutine.wrap(function()
		r.res, r.err, r.status = e:chat(req, r.sink)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	return r
end

-- A prompt of k words that repeat, so prompt lookup finds drafts.
local WORDS = { "alpha", "beta", "gamma", "delta", "local", "function", "end", "return", "x", "=" }
local function prompt(rng, k)
	local t = {}
	for j = 1, k do
		t[j] = WORDS[1 + math.floor(rng() * #WORDS)]
	end
	return table.concat(t, " ")
end

local function req(content, max)
	return { messages = { { role = "user", content = content } }, max_tokens = max, temperature = 0,
		chat_template_kwargs = { enable_thinking = false } }
end

-- 1. Requests of many prompt lengths, with speculation on: the compute
-- memory after each one stays within the plan of the engine, and so
-- does its peak. Each prompt length gives new graph shapes.
local qpath, mpath = tiny.tmp(".gguf"), tiny.tmp(".gguf")
tiny.write(qpath, "qwen3", { n_ctx = 8192 })
local kv, ts = q4t.build({ n_ctx = 8192 })
kv[#kv + 1] = { "tokenizer.chat_template", gguf.STRING,
	"<|im_start|> <tool_call> <function= <think> tool | tojson sysns reasoning_effort" }
gguf.write(mpath, kv, ts)
for _, m in ipairs({ { "qwen3", qpath }, { "qwen4exp mtp", mpath } }) do
	collectgarbage()
	collectgarbage()
	local e = engine.new({ model = m[2], backend = "CPU", n_ctx = 8192, slots = 1, kv_initial = 256,
		cache_ram = 0, speculative = { max = 3 } })
	local plan = e:memory().compute
	collectgarbage()
	collectgarbage()
	local base = cpu().compute
	lgml.memory_reset()
	local rng = tiny.rng(5)
	local worst, drafted = 0, 0
	for _ = 1, 10 do
		local r = chat(e, req(prompt(rng, 5 + math.floor(rng() * 1200)), 20 + math.floor(rng() * 40)))
		drafted = drafted + (r.timings and r.timings.draft_n or 0)
		worst = math.max(worst, cpu().compute - base)
	end
	local peak = cpu().peak_compute - base
	ok(e.spec_kept_bytes and e.spec_kept_bytes > 0, m[1] .. ": the plan counts kept graphs")
	ok(worst <= plan and peak <= plan and drafted > 0,
	    ("%s: compute stays within the plan (after a request %.1f MiB, peak %.1f, plan %.1f; %d drafted)")
	    :format(m[1], worst / 1048576, peak / 1048576, plan / 1048576, drafted))
end

-- 2. A tight budget: the kept graphs of a slot take memory that a new
-- prompt pass needs. The pass frees them and runs; the request does
-- not fail. A twin engine with no kept graphs measures what the
-- request needs.
do
	local function setup(keep)
		-- Without flash attention, a pass of 64 rows over a long prefix
		-- takes a large buffer.
		local e = engine.new({ model = qpath, backend = "CPU", n_ctx = 8192, slots = 1, kv_initial = 4096,
			cache_ram = 0, flash_attn = false })
		chat(e, req(prompt(tiny.rng(9), 1500), 10))
		if keep then
			local cache = e.slot.cache
			cache.replay_keep = 6
			local replay = require("lgml.replay")
			-- Graphs of six shapes, as a speculating slot keeps them.
			for k = 59, 64 do
				local ids = {}
				for x = 1, k do
					ids[x] = 100 + x
				end
				replay.run(e.model, cache, ids, { logits = "all", flash_attn = e.flash_attn })
				cache.n_past = cache.n_past - k
			end
		end
		collectgarbage()
		collectgarbage()
		return e
	end
	local function total()
		local c = cpu()
		return c.weights + c.tensors + c.compute
	end
	local next_prompt = prompt(tiny.rng(10), 900)
	local twin = setup(false)
	local before = total()
	lgml.memory_reset()
	chat(twin, req(next_prompt, 10))
	local need = cpu().peak - before
	local e = setup(true)
	local kept = require("lgml.replay").kept_bytes(e.slot.cache)
	local old = lgml.memory_guard({ limit = total() + need - kept // 2 })
	local r = chat(e, req(next_prompt, 10))
	lgml.memory_guard(old)
	ok(kept > 0 and r.res == true and (e.relieved or 0) > 0,
	    ("a pass past the budget frees %.2f MiB of kept graphs and runs (%s, relieved %d)"):format(
	    kept / 1048576, tostring(r.err or r.reason), e.relieved or 0))
end

-- 3. A draft or verify pass that does not fit stops speculation for the
-- request; the reply goes on with plain decode steps and matches the
-- reply without speculation.
do
	local budget = require("lgml.budget")
	local mk
	local e = engine.new({ model = qpath, backend = "CPU", n_ctx = 8192, slots = 1, kv_initial = 256,
		cache_ram = 0, flash_attn = false,
		speculative = { max = 4, adaptive = false, drafter = function() return mk() end } })
	local text = ("alpha beta gamma delta "):rep(30)
	mk = function() return { draft = function() return {} end } end
	local plain = chat(e, req(text, 40))
	local calls = 0
	mk = function()
		return { draft = function()
			calls = calls + 1
			error(budget.PREFIX .. " a tensor needs 1 MiB more on CPU", 0)
		end }
	end
	local r = chat(e, req(text, 40))
	ok(r.res == true and r.usage.completion_tokens == plain.usage.completion_tokens and calls == 1,
	    ("a draft past the budget stops speculation (%s, %d draft calls)"):format(tostring(r.err or r.reason), calls))
	-- A verify pass past the budget: the drafter sets a tight guard, and
	-- relieve lifts it, as freed graphs would.
	local relieve, old = e.relieve, nil
	function e.relieve(self, cache)
		if old then
			lgml.memory_guard(old)
			old = nil
		end
		return relieve(self, cache)
	end
	calls = 0
	mk = function()
		return { draft = function(_, ctx, k)
			calls = calls + 1
			old = lgml.memory_guard({ limit = cpu().weights + cpu().tensors + cpu().compute + 1 })
			return table.move(ctx.seq, #ctx.seq - k + 1, #ctx.seq, 1, {})
		end }
	end
	r = chat(e, req(text, 40))
	e.relieve = relieve
	ok(r.res == true and r.usage.completion_tokens == plain.usage.completion_tokens and calls == 1 and
	    old == nil, ("a verify pass past the budget stops speculation (%s)"):format(tostring(r.err or r.reason)))
end

os.remove(qpath)
os.remove(mpath)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
