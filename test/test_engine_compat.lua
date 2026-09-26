-- OpenAI request features of the gguf chat engine on a tiny qwen3 gguf
-- with random weights: stop strings, logprobs, logit_bias, tool_choice
-- and the tool grammars, and the parser output. TAP output.
local uv = require("luv")
local engine = require("lgml.engine")
local output = require("lgml.chat.output")
local hermes = require("lgml.chat.hermes")
local json = require("lgml.json")
local lgml = require("lgml")
local tool_grammar = require("lgml.chat.tool_grammar")

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

-- A sink that records the reply.
local function recorder()
	local r = { content = {}, reasoning = {}, tools = {}, pieces = {}, lp = {} }
	local sink = {}
	function sink.content(_, text, lp)
		r.content[#r.content + 1] = text
		r.pieces[#r.pieces + 1] = text
		for _, e in ipairs(lp or {}) do
			r.lp[#r.lp + 1] = e
		end
		return true
	end
	function sink.reasoning(_, text)
		r.reasoning[#r.reasoning + 1] = text
		return true
	end
	function sink.tool_call(_, name)
		r.tools[#r.tools + 1] = { name = name, args = {} }
		return #r.tools - 1
	end
	function sink.tool_arguments(_, i, text)
		local t = r.tools[i + 1].args
		t[#t + 1] = text
		return true
	end
	function sink.finish(_, reason, usage)
		r.reason, r.usage = reason, usage
	end
	function sink.cancelled()
		return false
	end
	r.sink = sink
	return r
end

local function text(parts)
	return table.concat(parts)
end

-- Output with stop strings on a hermes parser, fed one byte at a time.
local function test_output()
	local function run(s, stop)
		local r = recorder()
		local out = output.new(r.sink, hermes.new_parser({ tools = {} }), { stop = stop })
		local res = true
		for i = 1, #s do
			res = out:token(s:sub(i, i))
			if res ~= true then
				break
			end
		end
		if res == true then
			res = out:finish()
		end
		return r, res, out
	end
	local r, res = run("Hello END world", { "END" })
	ok(text(r.content) == "Hello " and res == "stop", "stop ends the content before the stop string")
	local leak = false
	for _, p in ipairs(r.pieces) do
		leak = leak or p:find("E", 1, true) ~= nil
	end
	ok(not leak, "no piece of the stop string goes out")
	r, res = run("<think>a END b</think>Hi END", { "END" })
	ok(text(r.reasoning) == "a END b" and text(r.content) == "Hi " and res == "stop",
	    "stop applies to the content, not to the reasoning")
	r, res = run("Hi EN", { "END", "xyz" })
	ok(text(r.content) == "Hi EN" and res == true, "a held partial stop string goes out at the end")
	r, res = run("a<b>c", { "<b>", "a<" })
	ok(text(r.content) == "" and res == "stop", "the first stop string in the text wins")
	r, res = run('Hi <tool_call>{"name": "f", "arguments": {"x": "END"}}</tool_call>', { "END" })
	ok(r.reason == nil and #r.tools == 0 and text(r.content):find("END", 1, true) == nil,
	    "tool call text of an unknown tool is content, and stop applies")
	local r2 = recorder()
	local out = output.new(r2.sink, hermes.new_parser({ tools = { "f" } }), { stop = { "END" } })
	out:token('Hi <tool_call>{"name": "f", "arguments": {"x": "END"}}</tool_call>')
	ok(out:finish() == true and #r2.tools == 1 and text(r2.tools[1].args) == '{"x": "END"}',
	    "stop does not apply to tool call arguments")
	local r3 = recorder()
	out = output.new(r3.sink, hermes.new_parser({ tools = { "f", "g" } }), { one_call = true })
	local call = '<tool_call>{"name": "%s", "arguments": {}}</tool_call>\n'
	local res = out:token(call:format("f") .. call:format("g"))
	ok(res == "stop" and #r3.tools == 1 and r3.tools[1].name == "f", "one_call ends the reply after a call")
	local r4 = recorder()
	out = output.new(r4.sink, hermes.new_parser({ tools = {} }))
	out:token(call:format("f"))
	ok(out:finish() == true and #r4.tools == 0, "with no tools a call is content")
end

local WEATHER = { type = "function", ["function"] = { name = "get_weather", parameters = {
	type = "object", properties = { city = { enum = { "Paris", "Rome" } }, days = { type = "integer" } },
	required = { "city" } } } }
local TIME = { type = "function", ["function"] = { name = "get_time" } }

local function test_tool_grammars()
	local function matches(text, root, s)
		local g = assert(lgml.grammar(text, { root = root, lax = true }))
		local r = g:accept(s) and g:can_end()
		g:close()
		return r == true
	end
	local t, root = assert(tool_grammar.hermes({ WEATHER, TIME }, { parallel = true }))
	local call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
	ok(matches(t, root, call), "hermes grammar takes a call")
	ok(matches(t, root, call .. "\n" .. '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'),
	    "hermes grammar takes two calls with parallel")
	ok(not matches(t, root, call:gsub("Paris", "Oslo")), "hermes arguments follow the schema")
	ok(not matches(t, root, call:gsub("get_weather", "get_other")), "hermes grammar takes the tools only")
	ok(not matches(t, root, "Hello"), "hermes grammar takes no text")
	t, root = tool_grammar.hermes({ WEATHER })
	ok(not matches(t, root, call .. "\n" .. call), "one call without parallel")
	t, root = tool_grammar.qwen35({ WEATHER, TIME }, { parallel = true })
	local x = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
	ok(matches(t, root, x), "qwen35 grammar takes a call")
	ok(matches(t, root, "<tool_call>\n<function=get_time>\n<parameter=any>\n1\n</parameter>\n</function>\n" ..
	    "</tool_call>"), "qwen35: a tool with no schema takes any parameter")
	ok(not matches(t, root, x:gsub("=city", "=town")), "qwen35 parameter names follow the schema")
	t, root = tool_grammar.gemma4({ WEATHER })
	ok(matches(t, root, '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'),
	    "gemma4 grammar takes a call")
	ok(not matches(t, root, '<|tool_call>call:get_time{}<tool_call|>'), "gemma4 grammar takes the tools only")
end

local path = tiny.tmp(".gguf")
tiny.write(path, "qwen3", { n_ctx = 1024 })
local e = engine.new({ model = path, backend = "CPU", n_ctx = 1024 })

local function chat(extra)
	local req = { messages = { { role = "user", content = "Hi there" } }, max_tokens = 40,
		temperature = 0, chat_template_kwargs = { enable_thinking = false } }
	for k, v in pairs(extra or {}) do
		req[k] = v
	end
	local r = recorder()
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

local function test_stop()
	local base = chat()
	local c = text(base.content)
	ok(base.res and #c > 0, ("the tiny model writes content: %q"):format(c))
	-- A stop string of two ASCII bytes from inside the content.
	local stop
	for i = 3, #c - 1 do
		local s = c:sub(i, i + 1)
		if s:match("^[%w%p ][%w%p ]$") then
			stop = s
			break
		end
	end
	if not stop then
		ok(true, "# skip: no ASCII pair in the content")
		return
	end
	local at = c:find(stop, 1, true)
	local r = chat({ stop = { "\0never", stop } })
	ok(r.res and text(r.content) == c:sub(1, at - 1) and r.reason == "stop",
	    ("stop %q ends the reply: %q"):format(stop, text(r.content)))
	ok(r.usage.completion_tokens < base.usage.completion_tokens, "the stopped reply has fewer tokens")
	local again = chat()
	ok(text(again.content) == c, "the next request is not cut")
end

local function test_logprobs()
	local base = chat()
	local r = chat({ logprobs = true, top_logprobs = 3 })
	local c = text(r.content)
	ok(c == text(base.content) and #r.lp > 0, ("logprobs do not change the reply; %d entries"):format(#r.lp))
	local bytes, sorted, greedy = {}, true, true
	for _, e in ipairs(r.lp) do
		bytes[#bytes + 1] = string.char(table.unpack(e.bytes))
		local t = e.top_logprobs
		sorted = sorted and #t == 3 and e.logprob <= 0 and t[1].logprob >= t[2].logprob and
		    t[2].logprob >= t[3].logprob
		greedy = greedy and t[1].token == e.token and math.abs(t[1].logprob - e.logprob) < 1e-6
	end
	ok(sorted, "each entry has 3 alternatives, most likely first")
	ok(greedy, "at temperature 0 the token is the most likely one")
	ok(require("lgml.tokenizer").clean(table.concat(bytes)):find(c, 1, true) ~= nil,
	    "the token bytes hold the content")
	local plain = chat({ logprobs = false })
	ok(#plain.lp == 0, "no entries without logprobs")
end

local function test_logit_bias()
	local r = chat({ logit_bias = { ["65"] = 100 } })
	ok(text(r.content):match("^AAAA") ~= nil, "logit_bias 100 forces a token: " .. text(r.content):sub(1, 10))
	local base = chat({ logprobs = true })
	local first = base.lp[1]
	local id = e.tk:encode(first.token)[1]
	local b = chat({ logprobs = true, logit_bias = { [tostring(id)] = -100 } })
	ok(b.lp[1] and b.lp[1].token ~= first.token, "logit_bias -100 bans a token")
	local bad = chat({ logit_bias = { ["999999"] = 1 } })
	ok(bad.res == nil and bad.status == 400 and bad.err:find("logit_bias"), "a token id past the vocab gives 400")
end

local function test_tool_choice()
	local tools = { WEATHER, TIME }
	local base = chat({ tools = tools })
	ok(base.res and #base.tools == 0, "auto: the random model writes no call")
	local r = chat({ tools = tools, tool_choice = "required", max_tokens = 200 })
	local args = r.tools[1] and json.decode(text(r.tools[1].args))
	ok(r.res and #r.tools >= 1 and text(r.content) == "", ("required forces a call: %d calls, content %q")
	    :format(#r.tools, text(r.content)))
	ok(r.tools[1] and (r.tools[1].name == "get_time" or (args and (args.city == "Paris" or
	    args.city == "Rome"))), "the forced call follows the schema")
	r = chat({ tools = tools, tool_choice = { type = "function", ["function"] = { name = "get_time" } },
		max_tokens = 200 })
	ok(#r.tools == 1 and r.tools[1].name == "get_time", "a named function forces one call to it")
	r = chat({ tools = tools, tool_choice = "required", parallel_tool_calls = false, max_tokens = 400,
		temperature = 1, seed = 3 })
	ok(#r.tools == 1, "required without parallel calls makes one call")
	-- Biases on the tags and on "<" lead the random model through a
	-- short reasoning block to the call, not into white space.
	local bias = { ["60"] = 10 }
	for id = 0, e.tk:n_vocab() - 1 do
		local t = e.tk:token(id)
		if t == "<think>" or t == "</think>" then
			bias[tostring(id)] = 20
		end
	end
	r = chat({ tools = tools, tool_choice = "required", max_tokens = 200, logit_bias = bias,
		chat_template_kwargs = { enable_thinking = true } })
	ok(#r.tools >= 1, ("required with thinking calls after the reasoning (%d reasoning tokens)")
	    :format(r.usage and r.usage.reasoning_tokens or -1))
	r = chat({ tools = tools, tool_choice = "none" })
	ok(r.res and #r.tools == 0, "none: no calls")
end

test_output()
test_tool_grammars()
test_stop()
test_logprobs()
test_logit_bias()
test_tool_choice()
os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
