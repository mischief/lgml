-- The Anthropic Messages and OpenAI Responses APIs on the gguf chat
-- engine with a tiny qwen3 model of random weights: stop sequences,
-- forced tool calls, thinking on and off, the reasoning budget and
-- token counts. TAP output.
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
	local chat = assert(anthropic.translate(json.decode(json.encode({ model = "tiny", max_tokens = 40,
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
	ok(u.reasoning_tokens >= 6 and u.reasoning_tokens < 20 and sink.list[1].kind == "reasoning" and
	    sink.list[2] and sink.list[2].kind == "text",
	    ("the budget ends the reasoning block and the text follows (%d reasoning tokens)"):format(u.reasoning_tokens))
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

test_messages()
test_budget()
test_responses()
os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
