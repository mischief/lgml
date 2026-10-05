-- The Anthropic Messages and OpenAI Responses APIs on the gguf chat
-- engine with a tiny qwen3 model of random weights: stop sequences,
-- forced tool calls, thinking on and off, the reasoning budget, the
-- repeat guard and token counts. TAP output.
local json = require("lgml.json")
local engine = require("lgml.engine")
local httpd = require("lgml.server.httpd")
local openai = require("lgml.server.openai")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local tiny = require("tiny")
local inproc = require("inproc")

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
local e = engine.new({ model = path, backend = "CPU", n_ctx = 1024, id = "tiny" })
local srv = httpd.new()
openai.install(srv, e, { key = ("t"):rep(32) })

local function token_id(t)
	for id = 0, e.tk:n_vocab() - 1 do
		if e.tk:token(id) == t then
			return id
		end
	end
	error("no token " .. t)
end

-- Biases on the think tags lead the random model into a long block.
local THINK = { [tostring(token_id("<think>"))] = 100 }

local HEADERS = { ["anthropic-version"] = "2023-06-01" }

-- The line that the engine adds at the reasoning budget.
local WRAP_UP = "Considering the limited time by the user"
-- Its tokens with the end of the block; the tiny vocabulary spells
-- most of it in bytes.
local WRAP_N = #e.tk:encode("\n\n" .. WRAP_UP .. ", I have to give the solution based on the " ..
    "thinking directly now.\n</think>\n\n")

local function messages(extra)
	local b = { model = "tiny", max_tokens = 24, temperature = 0, thinking = { type = "disabled" },
		messages = { { role = "user", content = "Hi there" } } }
	for k, v in pairs(extra or {}) do
		b[k] = v
	end
	return inproc.json(srv, "POST", "/v1/messages", b, HEADERS)
end

local function responses(extra)
	local b = { model = "tiny", max_output_tokens = 24, temperature = 0, reasoning = { effort = "none" },
		input = "Hi there" }
	for k, v in pairs(extra or {}) do
		b[k] = v
	end
	return inproc.json(srv, "POST", "/v1/responses", b)
end

local function text_of(b)
	for _, c in ipairs(b.content) do
		if c.type == "text" then
			return c.text
		end
	end
	return ""
end

local function output_text(b)
	for _, it in ipairs(b.output) do
		if it.type == "message" then
			return it.content[1].text
		end
	end
	return ""
end

local WEATHER = { name = "get_weather", input_schema = { type = "object",
	properties = { city = { enum = { "Paris", "Rome" } } }, required = { "city" } } }
local TIME = { name = "get_time", input_schema = { type = "object" } }

local function test_messages()
	local status, base = messages()
	local c = text_of(base)
	ok(status == 200 and #c > 0 and base.stop_reason == "max_tokens" and base.usage.output_tokens == 24,
	    ("messages: the model writes until max_tokens: %q"):format(c))
	local stop = c:match("[%w%p ][%w%p ]", 3)
	if stop then
		local _, b = messages({ stop_sequences = { "\0never", stop } })
		ok(b.stop_reason == "stop_sequence" and b.stop_sequence == stop and text_of(b) == c:sub(1, c:find(stop, 1,
		    true) - 1), ("messages: the stop sequence %q ends the text"):format(stop))
	end
	local _, b = messages({ tools = { WEATHER, TIME }, max_tokens = 200,
		tool_choice = { type = "any", disable_parallel_tool_use = true } })
	local call = b.content[#b.content]
	ok(b.stop_reason == "tool_use" and call.type == "tool_use" and call.id:match("^toolu_") and
	    (call.name == "get_time" or call.input.city == "Paris" or call.input.city == "Rome"),
	    "messages: any forces a call that follows the schema")
	_, b = messages({ tools = { WEATHER, TIME }, tool_choice = { type = "tool", name = "get_time" }, max_tokens = 200 })
	local calls = 0
	for _, x in ipairs(b.content) do
		calls = calls + (x.type == "tool_use" and 1 or 0)
	end
	ok(calls == 1 and b.content[#b.content].name == "get_time", "messages: tool forces one call to it")
	_, b = messages({ thinking = { type = "adaptive" }, max_tokens = 30 })
	local on = b.usage.input_tokens + b.usage.cache_read_input_tokens
	local off = base.usage.input_tokens + base.usage.cache_read_input_tokens
	ok(on < off and base.content[1].type == "text" and base.usage.output_tokens_details.thinking_tokens == 0,
	    ("messages: thinking off adds the empty think block (%d < %d tokens) and gives no thinking"):format(on, off))
	-- The budget: the engine closes the block after 1024 tokens; with
	-- max_tokens 30 the reply ends first, so compare with the chat API.
	local _, n1 = inproc.json(srv, "POST", "/v1/messages/count_tokens", { model = "tiny",
		thinking = { type = "disabled" }, messages = { { role = "user", content = "Hi there" } } }, HEADERS)
	ok(n1.input_tokens == base.usage.input_tokens + base.usage.cache_read_input_tokens,
	    ("count_tokens matches the prompt of a reply: %d"):format(n1.input_tokens))
end

local function test_budget()
	-- A chat request with logit_bias drives the model into reasoning;
	-- Messages has no logit_bias, so drive the engine directly with the
	-- chat request that a Messages budget makes.
	local anthropic = require("lgml.server.anthropic")
	local translate = require("lgml.server.translate")
	local ctx = translate.context()
	local chat = assert(anthropic.translate(json.decode(json.encode({ model = "tiny", max_tokens = 40 + WRAP_N,
		temperature = 0, thinking = { type = "enabled", budget_tokens = 1024 },
		messages = { { role = "user", content = "Hi there" } } })), ctx))
	ok(chat.reasoning_budget == 1024 and chat.chat_template_kwargs.enable_thinking == true,
	    "a Messages budget becomes the reasoning budget")
	chat.reasoning_budget, chat.logit_bias = 6, THINK
	local sink = translate.blocks({ s = { gone = function() return false end } })
	local done
	coroutine.wrap(function()
		e:chat(chat, sink)
		done = true
	end)()
	while not done do
		require("luv").run("once")
	end
	local u = sink.finished.usage
	local wrap = WRAP_N
	ok(u.reasoning_tokens > wrap and u.reasoning_tokens < 20 + wrap and sink.list[1].kind == "reasoning" and
	    translate.text(sink.list[1]):find(WRAP_UP, 1, true) and sink.list[2] and sink.list[2].kind == "text",
	    ("the budget ends the reasoning block with the wrap-up line and the text follows (%d reasoning tokens)")
	    :format(u.reasoning_tokens))
	-- The chat field, with a schema for the answer after the block. The
	-- grammar masks the think tag there; a smaller bias on "{" starts
	-- the object, as the random model would write blanks forever.
	local bias = { [tostring(token_id("<think>"))] = 100, [tostring(token_id("{"))] = 50 }
	local schema = { type = "object", properties = { city = { enum = { "Paris", "Rome" } } },
		required = { "city" }, additionalProperties = false }
	local _, c = inproc.json(srv, "POST", "/v1/chat/completions", { model = "tiny", max_tokens = 200 + WRAP_N,
		temperature = 0, logit_bias = bias, reasoning_budget_tokens = 6,
		response_format = { type = "json_schema", json_schema = { name = "c", schema = schema } },
		messages = { { role = "user", content = "Hi there" } } })
	local m = c.choices[1].message
	local r = c.usage.completion_tokens_details.reasoning_tokens
	local v = json.decode(m.content or "")
	ok(r > wrap and r < 20 + wrap and m.reasoning_content:find(WRAP_UP, 1, true) and
	    type(v) == "table" and (v.city == "Paris" or v.city == "Rome"),
	    ("chat: reasoning_budget_tokens ends the block and the schema answer follows (%d, %q)")
	    :format(r, tostring(m.content)))
	-- The model default applies to a request without a budget.
	e.reasoning_budget = 6
	_, c = inproc.json(srv, "POST", "/v1/chat/completions", { model = "tiny", max_tokens = 40 + WRAP_N,
		temperature = 0, logit_bias = THINK, messages = { { role = "user", content = "Hi there" } } })
	e.reasoning_budget = nil
	r = c.usage.completion_tokens_details.reasoning_tokens
	ok(r > wrap and r < 20 + wrap and #c.choices[1].message.content > 0,
	    ("the model default budget applies (%d reasoning tokens)"):format(r))
	-- Responses has no logit_bias: a wrapper engine adds it.
	local think = setmetatable({}, { __index = e })
	function think.chat(_, req, s)
		req.logit_bias = THINK
		return e:chat(req, s)
	end
	local s2 = httpd.new()
	openai.install(s2, think, { key = ("t"):rep(32) })
	local _, b = inproc.json(s2, "POST", "/v1/responses", { model = "tiny", max_output_tokens = 40 + WRAP_N,
		temperature = 0, reasoning_budget_tokens = 6, input = "Hi there" })
	r = b.usage.output_tokens_details.reasoning_tokens
	ok(r > wrap and r < 20 + wrap and b.output[1].type == "reasoning" and b.output[2] and
	    b.output[2].type == "message", ("responses: reasoning_budget_tokens ends the block (%d)"):format(r))
end

local function test_responses()
	local status, base = responses()
	local c = output_text(base)
	ok(status == 200 and #c > 0 and base.status == "incomplete" and base.usage.output_tokens == 24,
	    ("responses: the model writes until max_output_tokens: %q"):format(c))
	local tools = { { type = "function", name = "get_weather", parameters = WEATHER.input_schema },
		{ type = "function", name = "get_time", parameters = TIME.input_schema } }
	local _, b = responses({ tools = tools, tool_choice = "required", max_output_tokens = 200,
		parallel_tool_calls = false })
	local call = b.output[#b.output]
	ok(b.status == "completed" and call.type == "function_call" and call.call_id:match("^call_") and
	    json.decode(call.arguments) ~= nil, "responses: required forces a call")
	_, b = responses({ tools = tools, tool_choice = { type = "function", name = "get_time" },
		max_output_tokens = 200, parallel_tool_calls = false })
	ok(b.output[#b.output].name == "get_time", "responses: a named function")
	local _, b2 = responses({ tools = tools, tool_choice = "none" })
	ok(b2.output[#b2.output].type == "message", "responses: none gives text")
	local _, on = responses({ reasoning = { effort = "high" }, max_output_tokens = 10 })
	ok(on.status == "incomplete" and on.usage.input_tokens < base.usage.input_tokens,
	    "responses: effort none adds the empty think block of the template; high does not")
	local _, chained = responses({ previous_response_id = base.id, input = "More." })
	ok(chained.status and chained.usage.input_tokens > base.usage.input_tokens,
	    "responses: previous_response_id adds the earlier turn to the prompt")
	ok(chained.usage.input_tokens_details.cached_tokens > 0,
	    ("responses: the chained prompt reuses the cache (%d tokens)"):format(
	    chained.usage.input_tokens_details.cached_tokens))
	local _, cnt = inproc.json(srv, "POST", "/v1/responses/input_tokens", { model = "tiny", input = "Hi there",
		reasoning = { effort = "none" } })
	ok(cnt.input_tokens == base.usage.input_tokens, "input_tokens matches the prompt of a reply")
end

-- A bias on one token makes the model repeat it; the guard ends each
-- reply with the length reason of its API.
local function test_repeat()
	local metrics = require("lgml.server.metrics")
	local one = { [tostring(token_id("a"))] = 100 }
	local rep = setmetatable({}, { __index = e })
	function rep.chat(_, req, sink)
		req.logit_bias = one
		return e:chat(req, sink)
	end
	local s2 = httpd.new()
	openai.install(s2, rep, { key = ("t"):rep(32) })
	e.repeat_stop = 5
	metrics.reset()
	local _, c = inproc.json(s2, "POST", "/v1/chat/completions", { model = "tiny", max_tokens = 24,
		temperature = 0, reasoning_effort = "none", messages = { { role = "user", content = "Hi there" } } })
	ok(c.choices[1].finish_reason == "length" and c.usage.completion_tokens == 5,
	    ("repeat guard: chat ends with length after 5 tokens (%d)"):format(c.usage.completion_tokens))
	local _, m = inproc.json(s2, "POST", "/v1/messages", { model = "tiny", max_tokens = 24, temperature = 0,
		thinking = { type = "disabled" }, messages = { { role = "user", content = "Hi there" } } }, HEADERS)
	ok(m.stop_reason == "max_tokens" and m.usage.output_tokens == 5, "repeat guard: messages ends with max_tokens")
	local _, r = inproc.json(s2, "POST", "/v1/responses", { model = "tiny", max_output_tokens = 24,
		temperature = 0, reasoning = { effort = "none" }, input = "Hi there" })
	ok(r.status == "incomplete" and r.incomplete_details.reason == "max_output_tokens" and
	    r.usage.output_tokens == 5, "repeat guard: responses is incomplete with max_output_tokens")
	ok(metrics.render():find('lgml_repeat_stops_total{model="tiny"} 3', 1, true),
	    "repeat guard: /metrics counts the stops")
	e.repeat_stop = 0
	_, c = inproc.json(s2, "POST", "/v1/chat/completions", { model = "tiny", max_tokens = 24,
		temperature = 0, reasoning_effort = "none", messages = { { role = "user", content = "Hi there" } } })
	ok(c.usage.completion_tokens == 24, "repeat guard: 0 turns it off")
	e.repeat_stop = 256
end

test_messages()
test_budget()
test_repeat()
test_responses()
os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
