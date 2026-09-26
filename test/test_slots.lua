-- Engine slots: clients that take turns keep their caches on the device.
-- Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local uv = require("luv")
local engine = require("lgml.engine")
local tiny = require("tiny")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local path = tiny.tmp(".gguf")
tiny.write(path, "qwen3", { n_ctx = 2048 })

local function sink()
	local r = { text = {} }
	r.sink = {
		content = function(_, t) r.text[#r.text + 1] = t return true end,
		reasoning = function() return true end,
		tool_call = function() return 0 end,
		tool_arguments = function() return true end,
		finish = function(_, reason, usage) r.reason, r.usage = reason, usage end,
		cancelled = function() return false end,
	}
	return r
end

local function chat(e, words)
	local req = { messages = { { role = "user", content = words } }, max_tokens = 4,
		temperature = 0, chat_template_kwargs = { enable_thinking = false } }
	local r = sink()
	local done
	coroutine.wrap(function()
		r.res, r.err = e:chat(req, r.sink)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	return r
end

-- Two long, different system-free prompts; each client resends its own
-- prompt with one more word, as a chat client does each turn.
local a = ("apple banana cherry date elder fig grape "):rep(12)
local b = ("zebra yak xray walrus vole urchin tiger "):rep(12)

local function run(slots)
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = slots,
		cache_ram = 0, kv_initial = 256 })
	chat(e, a)
	chat(e, b)
	local ra = chat(e, a .. "more")
	local rb = chat(e, b .. "more")
	return e, ra.usage.cached_tokens, rb.usage.cached_tokens, ra, rb
end

local e1, a1, b1 = run(1)
ok(a1 < 20 and b1 < 20, ("one slot, no RAM store: turns do not reuse (cached %d, %d)"):format(a1, b1))
local e2, a2, b2, ra, rb = run(2)
ok(a2 > 60 and b2 > 60, ("two slots: each client reuses its own prefix (cached %d, %d)"):format(a2, b2))
ok(e2:props().total_slots == 2, "props report two slots")
ok(e2.slots[1].cache ~= e2.slots[2].cache, "each slot has its own cache")

-- The same answers as one slot with the prompt cache: slots change
-- what is cached, not what is computed.
local e3 = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 256 })
chat(e3, a)
local want = table.concat(chat(e3, a .. "more").text)
ok(table.concat(ra.text) == want, "a slot gives the same tokens as a fresh cache")

-- A third client with no shared prefix takes the least recently used slot.
local rc = chat(e2, ("one two three four five six seven "):rep(12))
ok(rc.res, "a third client runs")
-- Client a was the least recently used, so c took a's slot; b kept its own.
local rb2 = chat(e2, b .. "more again")
local ra2 = chat(e2, a .. "more again")
ok(rb2.usage.cached_tokens > 60, ("b kept its slot (cached %d)"):format(rb2.usage.cached_tokens))
ok(ra2.usage.cached_tokens < 20, ("a lost its slot to c (cached %d)"):format(ra2.usage.cached_tokens))

os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
