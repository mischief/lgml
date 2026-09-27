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
ok(a1 > 400 and b1 > 400 and e1.pc.restored == 0,
    ("one slot, no RAM store: the cached pool blocks keep both clients (cached %d, %d)"):format(a1, b1))
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
-- Client a was the least recently used, so c took a's slot; b kept its
-- own. The blocks of a stay cached in the pool.
local rb2 = chat(e2, b .. "more again")
local ra2 = chat(e2, a .. "more again")
ok(rb2.usage.cached_tokens > 60, ("b kept its slot (cached %d)"):format(rb2.usage.cached_tokens))
ok(ra2.usage.cached_tokens > 400 and e2.pc.restored == 0,
    ("a lost its slot to c, and gets its cached blocks back (cached %d)"):format(ra2.usage.cached_tokens))

-- Concurrent requests decode together in batched passes and give the
-- same tokens as one at a time.
local function chat_async(e, words, n_tok, res)
	local req = { messages = { { role = "user", content = words } }, max_tokens = n_tok,
		temperature = 0, chat_template_kwargs = { enable_thinking = false } }
	local r = sink()
	coroutine.wrap(function()
		r.res, r.err = e:chat(req, r.sink)
		res[#res + 1] = r
	end)()
	return r
end
local ec = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 3, kv_initial = 256 })
local seq = {}
for i, w in ipairs({ a, b, "short prompt" }) do
	local one = {}
	local r = chat_async(ec, w .. " seq", 40, one)
	while #one < 1 do
		uv.run("once")
	end
	seq[i] = table.concat(r.text)
end
local fresh = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 3, kv_initial = 256 })
local done = {}
local rs = {}
for i, w in ipairs({ a, b, "short prompt" }) do
	rs[i] = chat_async(fresh, w .. " seq", 40, done)
end
while #done < 3 do
	uv.run("once")
end
local same = true
for i = 1, 3 do
	same = same and rs[i].res and table.concat(rs[i].text) == seq[i]
end
ok(same, "three concurrent chats give the same text as one at a time")
ok((fresh.batch_rows or 0) > (fresh.batch_passes or 0),
    ("decode steps shared passes: %d rows in %d passes"):format(fresh.batch_rows or 0, fresh.batch_passes or 0))
ok(fresh:owned() == 0, "all slots are free after the requests")

-- The slots share one pool of n_ctx cells: more slots add no cells.
local transformer = require("lgml.transformer")
local p1 = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 1, kv_initial = 0 })
local p4 = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 4, kv_initial = 0 })
ok(p4.pool.n_ctx == 2048 and transformer.cache_bytes(p4.pool) == transformer.cache_bytes(p1.pool),
    ("four slots take the cells of one (%d bytes)"):format(transformer.cache_bytes(p4.pool)))
ok(engine.new({ model = path, backend = "CPU", n_ctx = 2048 }):props().total_slots == 4,
    "an engine keeps four slots by default")

-- Pool pressure: 768 cells hold one of the two clients. The idle one
-- gives its blocks up and goes to the RAM store; its next turn restores
-- it from there.
local ep = engine.new({ model = path, backend = "CPU", n_ctx = 768, slots = 3, kv_initial = 256 })
chat(ep, a)
chat(ep, b)
ok((ep.evictions or 0) == 1, ("b takes the blocks of idle a (%d evictions)"):format(ep.evictions or 0))
local pa = chat(ep, a .. "more")
ok(pa.usage.cached_tokens > 400 and ep.pc.restored == pa.usage.cached_tokens,
    ("a comes back from the RAM store (cached %d, restored %d)"):format(pa.usage.cached_tokens, ep.pc.restored))
ok(table.concat(pa.text) == want, "a restored slot gives the same tokens as a fresh cache")
local en = engine.new({ model = path, backend = "CPU", n_ctx = 768, slots = 3, kv_initial = 256, cache_ram = 0 })
chat(en, a)
chat(en, b)
local na = chat(en, a .. "more")
ok(na.usage.cached_tokens > 0 and na.usage.cached_tokens < 400 and na.usage.cached_tokens % 64 == 0,
    ("without a RAM store, an evicted slot keeps the first blocks; the pool took its last ones (cached %d)")
    :format(na.usage.cached_tokens))

-- Evicted blocks spill to host RAM and come back on a hit. The RAM
-- budget holds a few blocks, not a whole slot, so only blocks spill.
local transformer_m = require("lgml.transformer")
local er = engine.new({ model = path, backend = "CPU", n_ctx = 768, slots = 2, kv_initial = 256 })
er.pc.ram = 4 * transformer_m.block_bytes(er.pool)
chat(er, a)
chat(er, b)
local st = er.pc.stats
ok(st.saved == 0 and st.spilled >= 3, ("b evicts blocks of a; they spill to RAM (%d spilled)"):format(st.spilled))
local ra3 = chat(er, a .. "more")
ok(ra3.usage.cached_tokens == 448 and st.unspilled >= 3 and er.pc.restored == st.unspilled * 64,
    ("a finds its 7 blocks in the pool and in RAM (cached %d, %d from RAM)")
    :format(ra3.usage.cached_tokens, er.pc.restored))
ok(table.concat(ra3.text) == want, "blocks back from RAM give the same tokens as a fresh cache")

-- A prompt that does not fit next to a running request waits for it;
-- both give the same text as one at a time.
local function pair(e, words, n_tok)
	local done, rs = {}, {}
	for i, w in ipairs(words) do
		rs[i] = chat_async(e, w, n_tok[i], done)
	end
	while #done < #words do
		uv.run("once")
	end
	return rs
end
local solo = {}
for i, w in ipairs({ a .. " x", b .. " y" }) do
	local one = {}
	local r = chat_async(engine.new({ model = path, backend = "CPU", n_ctx = 768, slots = 1 }), w, 30, one)
	while #one < 1 do
		uv.run("once")
	end
	solo[i] = table.concat(r.text)
end
local ew = engine.new({ model = path, backend = "CPU", n_ctx = 768, slots = 2, kv_initial = 256 })
local rw = pair(ew, { a .. " x", b .. " y" }, { 30, 30 })
ok(rw[1].res and rw[2].res and table.concat(rw[1].text) == solo[1] and table.concat(rw[2].text) == solo[2],
    "a prompt that does not fit waits for the running request")
ok(ew:owned() == 0 and ew.pool.alloc.used <= ew.pool.alloc.n, "all slots are free after the requests")

-- Two running requests that fill the pool: the one that gets no block
-- lets its cells go and waits for the other, then takes them up again.
-- Both replies are whole and equal the replies alone.
local short = ("kiwi lime mango "):rep(12)
local half = ("apple banana cherry date elder fig grape "):rep(6)
local alone = {}
for i, w in ipairs({ half, short }) do
	local one = {}
	local r = chat_async(engine.new({ model = path, backend = "CPU", n_ctx = 768, slots = 1 }), w, 400, one)
	while #one < 1 do
		uv.run("once")
	end
	alone[i] = table.concat(r.text)
end
local ef = engine.new({ model = path, backend = "CPU", n_ctx = 768, slots = 2, kv_initial = 256 })
local rf = pair(ef, { half, short }, { 400, 400 })
local n1, n2 = rf[1].usage.completion_tokens, rf[2].usage.completion_tokens
ok(rf[1].res and rf[2].res and n1 == 400 and n2 == 400 and (ef.preemptions or 0) >= 1,
    ("a full pool makes a request wait, not end (%d, %d tokens, %d waits)"):format(n1, n2, ef.preemptions or 0))
-- Under pressure, cells come back out of position order, which changes
-- the rounding; a tiny random model is near ties, so only the start of
-- the replies must match.
local function same_len(x, y)
	local k = 0
	while k < #x and x:byte(k + 1) == y:byte(k + 1) do
		k = k + 1
	end
	return k
end
local s1, s2 = same_len(table.concat(rf[1].text), alone[1]), same_len(table.concat(rf[2].text), alone[2])
ok(s1 >= 200 and s2 >= 200, ("the replies start as the replies alone (%d, %d chars)"):format(s1, s2))
ok(ef:owned() == 0, "all slots are free after the requests")

-- The same two requests, the short one with high priority: the other
-- one lets its cells go for it, not the other way.
local kvp = require("lgml.kvpool")
local eh = engine.new({ model = path, backend = "CPU", n_ctx = 768, slots = 2, kv_initial = 256 })
local done_h, rh = {}, {}
for i, it in ipairs({ { half, nil }, { short, "24h" } }) do
	local r = sink()
	rh[i] = r
	coroutine.wrap(function()
		r.res = eh:chat({ messages = { { role = "user", content = it[1] } }, max_tokens = 400, temperature = 0,
			prompt_cache_retention = it[2], chat_template_kwargs = { enable_thinking = false } }, r.sink)
		done_h[#done_h + 1] = true
	end)()
end
while #done_h < 2 do
	uv.run("once")
end
local w = eh.waits or {}
ok(rh[1].res and rh[2].res and rh[1].usage.completion_tokens == 400 and rh[2].usage.completion_tokens == 400 and
    (w[kvp.NORMAL] or 0) >= 1 and (w[kvp.HIGH] or 0) == 0,
    ("the normal request waits for the high one (%d normal, %d high waits)"):format(w[kvp.NORMAL] or 0,
    w[kvp.HIGH] or 0))

-- Chats with messages, one at a time. retention is the
-- prompt_cache_retention field.
local function chat_msgs(e, messages, n_tok, retention)
	local r = sink()
	local done
	coroutine.wrap(function()
		r.res, r.err = e:chat({ messages = messages, max_tokens = n_tok or 4, temperature = 0,
			prompt_cache_retention = retention, chat_template_kwargs = { enable_thinking = false } }, r.sink)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	return r
end

-- Two clients with one long system prompt share its blocks: the pool
-- holds them once. The second one starts while the first one runs, so
-- it takes the other slot.
local sys = { role = "system", content = ("Follow the rules of the house and answer in short words. "):rep(4) }
local es = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 2, cache_ram = 0 })
local x1 = chat_msgs(es, { sys, { role = "user", content = "first question about fruit" } })
local y1 = sink()
local x1b = sink()
do
	local done = 0
	for _, it in ipairs({ { x1b, "first question about fruit, again" }, { y1, "another question about the sea" } }) do
		coroutine.wrap(function()
			it[1].res = es:chat({ messages = { sys, { role = "user", content = it[2] } }, max_tokens = 4,
				temperature = 0, chat_template_kwargs = { enable_thinking = false } }, it[1].sink)
			done = done + 1
		end)()
	end
	while done < 2 do
		uv.run("once")
	end
end
local bx, by = es.slots[1].cache.blocks, es.slots[2].cache.blocks
local same = 0
while bx[same + 1] and bx[same + 1] == by[same + 1] do
	same = same + 1
end
ok(same >= 3 and bx[same + 1] ~= by[same + 1] and es.pool.alloc:refs_of(bx[1]) == 2,
    ("two slots share the blocks of the system prompt (%s / %s)"):format(table.concat(bx, ","), table.concat(by, ",")))
ok(y1.usage.cached_tokens >= 192, ("the second client reuses them (cached %d)"):format(y1.usage.cached_tokens))
local fresh_e = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 1, cache_ram = 0 })
ok(table.concat(y1.text) == table.concat(chat_msgs(fresh_e, { sys,
    { role = "user", content = "another question about the sea" } }).text), "shared blocks give the same tokens")

-- A prompt that shares part of a block gets a copy of those cells, and
-- the first client's cells do not change.
local cut = sys.content:sub(1, 150)
local z1 = chat_msgs(es, { { role = "system", content = cut .. " but the rest differs" },
	{ role = "user", content = "first question about fruit" } })
local bz = es.slot.cache.blocks
local k = z1.usage.cached_tokens // 64
ok(k >= 1 and bz[k] == bx[k] and bz[k + 1] ~= bx[k + 1] and z1.usage.cached_tokens % 64 ~= 0,
    ("a partial block is copied (cached %d, blocks %s / %s)"):format(z1.usage.cached_tokens,
    table.concat(bz, ","), table.concat(bx, ",")))
local x2 = chat_msgs(es, { sys, { role = "user", content = "first question about fruit" } })
ok(table.concat(x2.text) == table.concat(x1.text), "the copy leaves the shared cells as they were")

-- One long chat and many short one-off requests on four slots: the
-- long prefix stays in the pool, and its next turn finds it there.
local el = engine.new({ model = path, backend = "CPU", n_ctx = 2048, slots = 4, cache_ram = 0 })
local long = { { role = "user", content = ("a long story about the river and the hills "):rep(20) } }
local l1 = chat_msgs(el, long, 8)
for i = 1, 10 do
	chat_msgs(el, { { role = "user", content = ("short request number %d, %s"):format(i, ("xyz"):rep(i)) } })
end
long[2] = { role = "assistant", content = table.concat(l1.text) }
long[3] = { role = "user", content = "and then?" }
local l2 = chat_msgs(el, long, 8)
ok(l2.usage.cached_tokens >= l1.usage.prompt_tokens - 64,
    ("after 10 short requests the long chat finds its prefix (cached %d of %d)"):format(
    l2.usage.cached_tokens, l2.usage.prompt_tokens))

-- Priorities: a chat that asks to keep its prefix (retention "24h")
-- keeps it while other requests fill the pool; a plain chat loses its
-- last blocks to them.
local function keeps(retention)
	local ek = engine.new({ model = path, backend = "CPU", n_ctx = 1024, slots = 4, cache_ram = 0 })
	local m1 = { { role = "user", content = a } }
	chat_msgs(ek, m1, 4, retention)
	for i = 1, 6 do
		chat_msgs(ek, { { role = "user", content = ("other request %d "):format(i) .. ("qrs tuv "):rep(20) } })
	end
	return chat_msgs(ek, m1, 4, retention).usage.cached_tokens, ek
end
local plain = keeps(nil)
local kept, ek = keeps("24h")
ok(kept > 400 and plain < kept, ("a high priority chat keeps its prefix (cached %d; %d without)"):format(kept, plain))
ok(ek.slots[1].prio == require("lgml.kvpool").HIGH or ek.slot.prio == require("lgml.kvpool").HIGH,
    "the slot of the chat has high priority")


-- A request that is cancelled while it waits for a slot leaves the
-- queue at its turn. It computes nothing, and the next request runs.
do
	local ew = engine.new({ model = path, backend = "CPU", n_ctx = 1024, slots = 1, kv_initial = 256 })
	local runs, run = 0, ew.run
	ew.run = function(...)
		runs = runs + 1
		return run(...)
	end
	local fin = {}
	local first = chat_async(ew, a, 20, fin)
	local gone = chat_async(ew, b, 20, fin)
	local stop = false
	gone.sink.cancelled = function() return stop end
	local after = chat_async(ew, "short prompt after", 4, fin)
	stop = true
	while #fin < 3 do
		uv.run("once")
	end
	ok(gone.res == true and runs == 2 and #gone.text == 0,
	    "a cancelled request that waits for a slot computes nothing")
	ok(first.usage and after.usage and after.reason and ew:owned() == 0 and not ew.busy,
	    "the requests before and after it run to the end")
end
os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
