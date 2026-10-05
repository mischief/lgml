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
	properties = { city = { type = "string", enum = { "Paris", "Rome" } }, days = { type = "integer", enum = { 1, 2 } } },
	required = { "days" } } } }
-- Typed parameters for qwen35 calls; values other than strings follow
-- the schema.
local TYPED = { type = "function", ["function"] = { name = "set", parameters = { type = "object",
	properties = { on = { type = "boolean" }, n = { type = "integer", maximum = 99 },
		o = { type = "object", properties = { k = { type = "integer", maximum = 9 } }, required = { "k" },
			additionalProperties = false },
		a = { type = "array", items = { type = "boolean" }, maxItems = 2 } },
	required = { "on", "n", "o", "a" } } } }
local TIME = { type = "function", ["function"] = { name = "get_time",
	parameters = { type = "object", properties = {}, additionalProperties = false } } }
local DECLARED = { get_weather = true, get_time = true, set = true }

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
	local r = { content = {}, reasoning = {}, tools = {}, order = {} }
	local sink = {}
	function sink.content(_, t)
		r.content[#r.content + 1] = t
		r.order[#r.order + 1] = "c"
		return true
	end
	function sink.reasoning(_, t)
		r.reasoning[#r.reasoning + 1] = t
		return true
	end
	function sink.tool_call(_, name)
		r.tools[#r.tools + 1] = { name = name, args = {} }
		r.order[#r.order + 1] = "t"
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
	r.args = r.tools[1] and json.decode(table.concat(r.tools[1].args))
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

-- A tiny model with the chat template kv and extra tokens: the engine
-- with the grammar, one without it, and the token id of a text.
local function setup(kv, tokens)
	local path = tiny.tmp(".gguf")
	tiny.write(path, "qwen3", { n_ctx = 4096, tokens = tokens, kv = kv })
	local function mk(extra)
		local o = { model = path, backend = "CPU", n_ctx = 4096, flash_attn = false,
			speculative = { max = 4, adaptive = false, drafter = test_drafter } }
		for k, v in pairs(extra or {}) do
			o[k] = v
		end
		return engine.new(o)
	end
	local e = mk()
	local function id(t)
		for i = 0, e.tk:n_vocab() - 1 do
			if e.tk:token(i) == t then
				return tostring(i)
			end
		end
		error("no token " .. t)
	end
	local free = mk({ lazy_tool_grammar = false })
	os.remove(path)
	return e, free, id
end

local X, A, SLASH = tostring(string.byte("x")), tostring(string.byte("A")), tostring(string.byte("/"))

-- A request with tools; extra fields override.
local function req(bias, extra)
	local r = { messages = { { role = "user", content = "Hi there" } }, max_tokens = 200, temperature = 0,
		chat_template_kwargs = { enable_thinking = false }, tools = { WEATHER, TIME },
		logit_bias = bias, presence_penalty = 150, speculative = false }
	for k, v in pairs(extra or {}) do
		r[k] = v
	end
	return r
end

-- Speculative decoding gives the greedy reply of plain decoding when
-- every third guess is "x", in and out of calls.
local function same_spec(e, base, what)
	base.speculative = false
	local plain = run(e, base)
	local rec = recorder()
	next_drafter = rec
	base.speculative = true
	run(e, base)
	next_drafter = oracle(rec.seq, tonumber(X))
	local sp = run(e, base)
	local t = sp.timings or {}
	ok(sp.res and sp.text == plain.text and sp.calls == plain.calls and
	    sp.usage.completion_tokens == plain.usage.completion_tokens and
	    (t.draft_accepted or 0) > 0 and t.draft_accepted < t.draft_n,
	    ("%s: speculative reply matches (%d calls, %s of %s guesses kept)"):format(what,
	    #plain.tools, t.draft_accepted, t.draft_n))
end

local function family(name, kv)
	local e, free, id = setup(kv, { "<tool_call>", "</tool_call>" })
	ok(e.family_name == name, "family " .. e.family_name)
	local CALL = id("<tool_call>")
	-- "A" first, then the opener; "x" starts no declared name; "/"
	-- ends a qwen35 call early. The presence penalty takes each token
	-- once.
	local bias = { [A] = 120, [CALL] = 100, [X] = 70, [SLASH] = 60 }
	-- Without the presence penalty, "A" would repeat.
	local calls = { [CALL] = 100, [X] = 70, [SLASH] = 60 }
	local tag = name .. ": "

	local f = run(free, req(bias))
	ok(f.res and #f.tools == 0 and f.text:find("<tool_call>x", 1, true),
	    tag .. ("without the grammar the bias spoils the call: %q"):format(f.text))
	local r = run(e, req(bias, { max_tokens = 600 }))
	ok(r.res and declared(r, 1), tag .. "the call goes to a declared tool: " .. r.calls)
	ok(r.text:sub(1, 1) == "A" and r.order[1] == "c", tag .. ("text before the call is free: %q"):format(r.text))
	ok(r.reason == "length" or not r.text:find("<tool_call>", 1, true), tag .. "no call text in the content")
	ok(table.concat(r.order):find("tc"), tag .. "text after a call is free")
	local none = run(e, req(bias, { tool_choice = "none" }))
	ok(none.text == f.text, tag .. "tool_choice none: no grammar")
	local one = run(e, req(calls, { presence_penalty = 0, max_tokens = 400, parallel_tool_calls = false }))
	ok(#one.tools == 1 and declared(one), tag .. "one call without parallel calls: " .. one.calls)
	local two = run(e, req(calls, { presence_penalty = 0, max_tokens = 400 }))
	ok(declared(two, 2), tag .. ("parallel calls: %d, all declared"):format(#two.tools))
	local fmt = run(e, req(bias, { response_format = { type = "json_object" }, max_tokens = 60 }))
	ok(fmt.res and #fmt.tools == 0 and fmt.text:match("^{"),
	    tag .. ("a response format wins over the tool grammar: %q"):format(fmt.text:sub(1, 20)))

	-- With thinking, a trigger in the block does nothing.
	local tb = { [id("<think>")] = 130, [id("</think>")] = 50 }
	for k, v in pairs(bias) do
		tb[k] = v
	end
	local th = run(e, req(tb, { chat_template_kwargs = { enable_thinking = true }, max_tokens = 60 }))
	ok(th.think:find("<tool_call>x", 1, true), tag .. ("the reasoning stays free: %q"):format(th.think))

	same_spec(e, req(calls, { presence_penalty = 0, max_tokens = 300 }), tag .. "calls")
	local args = r.args
	ok(args and (r.tools[1].name == "get_time" or args.days == 1 or args.days == 2),
	    tag .. "the arguments follow the schema")
	if name ~= "qwen35" then
		return
	end

	-- Typed values come back as JSON of their types.
	local typed = { [CALL] = 100 }
	local topts = { tools = { TYPED }, presence_penalty = 0, max_tokens = 300, parallel_tool_calls = false }
	local tr = run(e, req(typed, topts))
	args = tr.args
	ok(args and tr.tools[1].name == "set" and type(args.on) == "boolean" and
	    math.type(args.n) == "integer" and type(args.o) == "table" and math.type(args.o.k) == "integer" and
	    type(args.a) == "table" and #args.a <= 2 and (args.a[1] == nil or type(args.a[1]) == "boolean"),
	    tag .. "typed values: " .. tr.calls)
	local tf = run(free, req(typed, topts))
	ok(not tf.args, tag .. ("typed values: the free model writes no good call: %q"):format(tf.text:sub(1, 40)))
	topts.parallel_tool_calls = nil
	same_spec(e, req(typed, topts), tag .. "typed calls")
end

-- Qwen-Coder calls without <tool_call>, and string values. The vocab
-- has the opener of a bare call, a value and the value close as tokens,
-- so biases can lead the model through them.
local function bare(kv)
	local e, free, id = setup(kv, { "<tool_call>", "</tool_call>", "<function=get_time>", "Ċa<b>Ċc",
		"Ċ</parameter>Ċ" })
	local tag = "qwen35 bare: "
	local b = { [A] = 120, [id("<function=get_time>")] = 100, [X] = 70, [SLASH] = 60 }
	local f = run(free, req(b))
	ok(#f.tools == 0 and f.text:find("<function=get_time>x", 1, true),
	    tag .. ("without the grammar the call is spoiled: %q"):format(f.text:sub(1, 40)))
	local r = run(e, req(b))
	ok(r.res and #r.tools >= 1 and declared(r) and r.tools[1].name == "get_time" and r.order[1] == "c" and
	    r.text:sub(1, 1) == "A" and not r.text:find("<function", 1, true),
	    tag .. ("the call is a tool call: %s"):format(r.calls))
	local nb = { [id("<function=get_time>")] = 100, [X] = 70, [SLASH] = 60 }
	same_spec(e, req(nb, { presence_penalty = 0, max_tokens = 300 }), tag .. "calls")

	local STR = { type = "function", ["function"] = { name = "echo", parameters = { type = "object",
		properties = { text = { type = "string" } }, required = { "text" } } } }
	DECLARED.echo = true
	local s = run(e, req({ [id("<tool_call>")] = 100, [id("\na<b>\nc")] = 90, [id("\n</parameter>\n")] = 80 },
		{ tools = { STR }, parallel_tool_calls = false }))
	ok(s.args and s.args.text == "a<b>\nc", tag .. ("a string value keeps new lines and <: %s"):format(s.calls))
end

local QWEN35 = { { "tokenizer.chat_template", gguf.STRING,
	"<|im_start|> {{- tool | tojson }} <think> <tool_call>\n<function=" } }
family("qwen3")
family("qwen35", QWEN35)
bare(QWEN35)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
