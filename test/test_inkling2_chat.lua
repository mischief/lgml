-- The Inkling 2 chat family: render gives the bytes of the gguf Jinja
-- template (rendered with jinja2), and the parser splits reasoning,
-- content and tool calls. Output is TAP.
local json = require("lgml.json")
local chat = require("lgml.chat")
local f = require("lgml.chat.inkling2")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

-- Requests and the prompts the template renders for them.
local CASES = {
	{ req = [==[{"messages": [{"role": "user", "content": "Hi there"}]}]==],
		want = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi there<|end_message|>" },
	{ req = [==[{"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}], "reasoning_effort": "low"}]==],
		want = "<|message_system|><|content_text|>Be terse.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>Hi<|end_message|>" },
	{ req = [==[{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"}, {"role": "user", "content": "Weather in Paris? ünï"}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}], "reasoning_effort": "none"}]==],
		want = "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Get weather\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Hi<|end_message|><|message_model|><|content_thinking|>greet<|end_message|><|message_model|><|content_text|>Hello!<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>Weather in Paris? ünï<|end_message|>" },
	{ req = [==[{"messages": [{"role": "user", "content": "Weather?"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"days": 3, "city": "Paris"}}}]}, {"role": "tool", "tool_call_id": "c1", "content": "sunny"}, {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}], "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}]==],
		want = "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"}},\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Weather?<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":3}}<|end_message|><|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>sunny<|end_message|><|message_user|><|content_text|>part one<|end_message|><|message_user|><|content_text|>part two<|end_message|>" },
	{ req = [==[{"messages": [{"role": "system", "content": "only system"}], "reasoning_effort": "xhigh"}]==],
		want = "<|message_system|><|content_text|>only system<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|>" },
}

for i, c in ipairs(CASES) do
	local req = json.decode(c.req)
	local got = f.render(req.messages, { tools = req.tools, reasoning_effort = req.reasoning_effort })
	ok(got == c.want, "render case " .. i)
	if got ~= c.want then
		print("# got  " .. got)
		print("# want " .. c.want)
	end
end

ok(chat.detect({ ["general.architecture"] = "inkling2", ["tokenizer.chat_template"] = "x<|message_model|>" }) ==
    "inkling2", "detect by template")
ok(f.render({ { role = "user", content = "hi" } }, { enable_thinking = false }):find("Thinking effort level: 0<", 1, true),
    "thinking off is effort 0")

-- An image marker from lgml.chat.media gets a block of its own.
local marker = f.image.begin .. f.image.token .. f.image.finish
local p = f.render({ { role = "user", content = marker .. "What is this?" } })
ok(p:find("<|message_user|><|content_image|><|unused_200054|><|end_message|>" ..
    "<|message_user|><|content_text|>What is this?<|end_message|>", 1, true), "image block")

-- Events of text fed in chunks of size k, as one string per event.
local function events(text, opts, k)
	local pr = f.new_parser(opts)
	local out = {}
	local function take(list)
		for _, e in ipairs(list) do
			local last = out[#out]
			local s = e.type .. ":" .. (e.text or e.name or "")
			if last and e.text and last.type == e.type then
				last.s = last.s .. e.text
			else
				out[#out + 1] = { type = e.type, s = s }
			end
		end
	end
	for i = 1, #text, k do
		take(pr:feed(text:sub(i, i + k - 1)))
	end
	take(pr:finish())
	local r = {}
	for i, e in ipairs(out) do
		r[i] = e.s
	end
	return table.concat(r, " | ")
end

local T1 = "<|message_model|><|content_thinking|>Let me think.<|end_message|>" ..
    "<|message_model|><|content_text|>Paris.<|end_message|><|content_model_end_sampling|>"
local T2 = "<|message_model|><|content_thinking|>tool<|end_message|><|message_model|>get_weather" ..
    '<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Oslo"}}<|end_message|>' ..
    "<|content_model_end_sampling|>"
for _, k in ipairs({ 1, 5, 1000 }) do
	ok(events(T1, {}, k) == "reasoning:Let me think. | content:Paris.", "reasoning and content, chunks of " .. k)
	ok(events(T2, { tools = { "get_weather" } }, k) ==
	    'reasoning:tool | tool_call_begin:get_weather | tool_call_args:{"city":"Oslo"} | tool_call_end:',
	    "tool call, chunks of " .. k)
end
ok(events(T2, { tools = { "other" } }, 7):find("content:", 1, true) ~= nil, "a call to an unknown tool is content")
ok(events("plain text", {}, 3) == "content:plain text", "text with no header")

print("1.." .. n)
os.exit(failed and 1 or 0)
