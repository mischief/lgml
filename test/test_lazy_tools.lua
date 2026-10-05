-- Lazy tool call grammars on tiny models, hermes (qwen3) and XML
-- (qwen35) calls: under tool_choice "auto", logit biases push the model
-- to a call opener, then to an undeclared name; calls keep to the
-- declared tools, and other text stays free. No flash attention, so
-- speculative and plain decoding see the same logits. TAP output.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local uv = require("luv")
local engine = require("lgml.engine")
local gguf = require("gguf")
local json = require("lgml.json")
local tiny = require("tiny")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local WEATHER = { type = "function", ["function"] = { name = "get_weather", parameters = { type = "object",
	properties = { city = { type = "string", enum = { "Paris", "Rome" } } }, required = { "city" } } } }
local TIME = { type = "function", ["function"] = { name = "get_time",
	parameters = { type = "object", properties = {}, additionalProperties = false } } }
local DECLARED = { get_weather = true, get_time = true }

-- The drafter of the next request.
local next_drafter
local function test_drafter()
	return next_drafter
end

-- A drafter that keeps the sequence it sees and guesses nothing.
local function recorder()
	return { draft = function(self, ctx)
		self.seq = ctx.seq
		return {}
	end }
end

-- A drafter that guesses the reference sequence ref but puts bad in
-- every third guess: bad is a token that a call does not allow.
local function oracle(ref, bad)
	local d = { calls = 0 }
	function d:draft(ctx, k)
		local out = {}
		for j = 1, k do
			local id = ref[#ctx.seq + j]
			if not id then
				break
			end
			self.calls = self.calls + 1
			out[j] = self.calls % 3 == 0 and bad or id
		end
		return out
	end
	return d
end

local function run(e, req)
	local r = { content = {}, reasoning = {}, tools = {} }
	local sink = {}
	function sink.content(_, t)
		r.content[#r.content + 1] = t
		return true
	end
	function sink.reasoning(_, t)
		r.reasoning[#r.reasoning + 1] = t
		return true
	end
	function sink.tool_call(_, name)
		r.tools[#r.tools + 1] = { name = name, args = {} }
		return #r.tools - 1
	end
	function sink.tool_arguments(_, i, t)
		local a = r.tools[i + 1].args
		a[#a + 1] = t
		return true
	end
	function sink.finish(_, reason, usage, timings)
		r.reason, r.usage, r.timings = reason, usage, timings
	end
	function sink.cancelled()
		return false
	end
	local done
	coroutine.wrap(function()
		r.res, r.err = e:chat(req, sink)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	r.text = table.concat(r.content)
	r.think = table.concat(r.reasoning)
	local calls = {}
	for _, t in ipairs(r.tools) do
		calls[#calls + 1] = t.name .. table.concat(t.args)
	end
	r.calls = table.concat(calls, ";")
	return r
end

-- True when r has at least min calls, each to a declared tool.
local function declared(r, min)
	for _, t in ipairs(r.tools) do
		if not DECLARED[t.name] then
			return false
		end
	end
	return #r.tools >= (min or 1)
end

local function family(name, kv)
	local path = tiny.tmp(".gguf")
	tiny.write(path, "qwen3", { n_ctx = 4096, tokens = { "<tool_call>", "</tool_call>" }, kv = kv })
	local function mk(extra)
		local o = { model = path, backend = "CPU", n_ctx = 4096, flash_attn = false,
			speculative = { max = 4, adaptive = false, drafter = test_drafter } }
		for k, v in pairs(extra or {}) do
			o[k] = v
		end
		return engine.new(o)
	end
	local e, free = mk(), mk({ lazy_tool_grammar = false })
	ok(e.family_name == name, "family " .. e.family_name)
	local function id(t)
		for i = 0, e.tk:n_vocab() - 1 do
			if e.tk:token(i) == t then
				return i
			end
		end
		error("no token " .. t)
	end
	local CALL, X, A = id("<tool_call>"), string.byte("x"), string.byte("A")
	-- "A" first, then the opener; "x" starts no declared name; "/" ends
	-- a qwen35 call before any parameter. The presence penalty takes
	-- each token once.
	local bias = { [tostring(A)] = 120, [tostring(CALL)] = 100, [tostring(X)] = 70,
		[tostring(string.byte("/"))] = 60 }
	local function req(extra)
		local r = { messages = { { role = "user", content = "Hi there" } }, max_tokens = 200, temperature = 0,
			chat_template_kwargs = { enable_thinking = false }, tools = { WEATHER, TIME },
			logit_bias = bias, presence_penalty = 150, speculative = false }
		for k, v in pairs(extra or {}) do
			r[k] = v
		end
		return r
	end
	local tag = name .. ": "
	-- Without the presence penalty, "A" would repeat.
	local calls = { [tostring(CALL)] = 100, [tostring(X)] = 70, [tostring(string.byte("/"))] = 60 }

	local f = run(free, req())
	ok(f.res and #f.tools == 0 and f.text:find("<tool_call>x", 1, true),
	    tag .. ("without the grammar the bias spoils the call: %q"):format(f.text))
	local r = run(e, req())
	ok(r.res and declared(r, 1), tag .. "the call goes to a declared tool: " .. r.calls)
	ok(r.text:sub(1, 1) == "A", tag .. ("text before the call is free: %q"):format(r.text))
	ok(not r.text:find("<tool_call>", 1, true), tag .. "no call text in the content")
	if name == "qwen35" then
		ok(#r.tools == 1 and #r.text > 1, tag .. ("text after the call is free: %q"):format(r.text))
	end
	local none = run(e, req({ tool_choice = "none" }))
	ok(none.text == f.text, tag .. "tool_choice none: no grammar")
	local one = run(e, req({ presence_penalty = 0, logit_bias = calls, max_tokens = 400,
		parallel_tool_calls = false }))
	ok(#one.tools == 1 and declared(one), tag .. "one call without parallel calls: " .. one.calls)
	local two = run(e, req({ presence_penalty = 0, logit_bias = calls, max_tokens = 400 }))
	ok(declared(two, 2), tag .. ("parallel calls: %d, all declared"):format(#two.tools))
	local fmt = run(e, req({ response_format = { type = "json_object" }, max_tokens = 60 }))
	ok(fmt.res and #fmt.tools == 0 and fmt.text:match("^{"),
	    tag .. ("a response format wins over the tool grammar: %q"):format(fmt.text:sub(1, 20)))

	-- With thinking, a trigger in the block does nothing.
	local tb = { [tostring(id("<think>"))] = 130, [tostring(id("</think>"))] = 50 }
	for k, v in pairs(bias) do
		tb[k] = v
	end
	local th = run(e, req({ chat_template_kwargs = { enable_thinking = true }, logit_bias = tb,
		max_tokens = 60 }))
	ok(th.think:find("<tool_call>x", 1, true), tag .. ("the reasoning stays free: %q"):format(th.think))

	-- Speculative decoding: the same greedy reply when bad guesses are
	-- rejected in and out of calls.
	local base = req({ presence_penalty = 0, logit_bias = calls, max_tokens = 300 })
	local plain = run(e, base)
	local rec = recorder()
	next_drafter = rec
	base.speculative = true
	run(e, base)
	next_drafter = oracle(rec.seq, X)
	local sp = run(e, base)
	local t = sp.timings or {}
	ok(sp.res and sp.text == plain.text and sp.calls == plain.calls and
	    sp.usage.completion_tokens == plain.usage.completion_tokens and
	    (t.draft_accepted or 0) > 0 and t.draft_accepted < t.draft_n,
	    tag .. ("speculative reply matches (%d calls, %s of %s guesses kept)"):format(#plain.tools,
	    t.draft_accepted, t.draft_n))
	if name == "qwen3" then
		local args = json.decode(table.concat(r.tools[1].args))
		ok(args and (r.tools[1].name == "get_time" or args.city == "Paris" or args.city == "Rome"),
		    tag .. "the arguments follow the schema")
	end
	os.remove(path)
end

family("qwen3")
family("qwen35", { { "tokenizer.chat_template", gguf.STRING,
	"<|im_start|> {{- tool | tojson }} <think> <tool_call>\n<function=" } })
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
