-- Prompt chunk sizes of the gguf chat engine with a tiny qwen3 model of
-- random weights: chunks of 64, 512 and the auto size give the same
-- tokens, alone and in batched slots, and auto falls back to a smaller
-- size when the compute buffers do not fit. TAP output.
local lgml = require("lgml")
local uv = require("luv")
local engine = require("lgml.engine")
local budget = require("lgml.budget")
local translate = require("lgml.server.translate")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
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
tiny.write(path, "qwen3", { n_ctx = 1024 })

local function new(o)
	o.model, o.backend, o.n_ctx, o.id = path, "CPU", 1024, "tiny"
	return engine.new(o)
end

-- The greedy reply to a prompt of about 700 tokens, and the chunks
-- that the prompt took.
local function reply(e, word)
	local sink = translate.blocks({ s = { gone = function() return false end } })
	local done
	coroutine.wrap(function()
		e:chat({ messages = { { role = "user", content = (word or "ab "):rep(230) } }, max_tokens = 8,
			temperature = 0, reasoning_effort = "none" }, sink)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	local out = {}
	for _, b in ipairs(sink.list) do
		out[#out + 1] = translate.text(b)
	end
	return table.concat(out), sink.finished.usage
end

local small, base, auto = new({ ubatch = 64, slots = 1 }), new({ slots = 1 }), new({ ubatch = "auto", slots = 1 })
ok(small.ubatch == 64 and base.ubatch == 512 and auto.ubatch == 1024,
    ("the chunk sizes: 64, 512 by default, %d for auto with 1024 cells"):format(auto.ubatch))
local a, u = reply(small)
local b = reply(base)
local c = reply(auto)
ok(u.prompt_tokens > 512 and #a > 0 and a == b and b == c,
    ("chunks of 64, 512 and 1024 give the same tokens for %d prompt tokens"):format(u.prompt_tokens))
-- Two slots decode in batched passes; the prompt chunks join them.
local two = new({ ubatch = 128, slots = 2 })
ok(two.concurrent, "two slots run in batched passes")
local d = reply(two)
ok(d == a, "batched chunks of 128 give the same tokens")
-- The planned compute grows with the chunk size.
local r64, r512, r1024 = small:reserve(), base:reserve(), auto:reserve()
ok(r64 < r512 and r512 < r1024, ("the plans grow with the chunk: %d < %d < %d"):format(r64, r512, r1024))
-- A budget between the plans of 512 and 1024 makes auto pick 512.
local check = lgml.memory_check
local limit = (r512 + r1024) // 2
lgml.memory_check = function(_, bytes)
	if bytes > limit then
		return nil, budget.PREFIX .. " the compute buffers do not fit"
	end
	return true
end
local fit_ok, fit = pcall(new, { ubatch = "auto", slots = 1 })
lgml.memory_check = check
ok(fit_ok and fit.ubatch == 512 and fit:reserve() == r512,
    ("auto falls back to 512 under a smaller budget (%s)"):format(fit_ok and fit.ubatch or tostring(fit)))
ok(fit_ok and reply(fit) == a, "the fallback gives the same tokens")
os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
