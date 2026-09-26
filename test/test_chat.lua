-- Chat template and tool-call parser tests. Output is TAP.
-- Fixtures in test/chat/ (Qwen3), test/chat/gemma4/ and
-- test/chat/qwen35/ hold llama-server /apply-template requests (.json)
-- and the prompt it returned (.txt).
local dir = (arg and arg[0] or ""):match("^(.*)/[^/]*$") or "."
package.path = dir .. "/../lua/?.lua;" .. dir .. "/../lua/?/init.lua;" .. package.path

local chat = require("lgml.chat")
local qwen3 = require("lgml.chat.qwen3")
local gemma4 = require("lgml.chat.gemma4")
local qwen35 = require("lgml.chat.qwen35")
local tojson = require("lgml.chat.tojson")
local media = require("lgml.chat.media")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	return cond
end

-- Small JSON decoder for fixtures. Objects keep key order in __order.
local decode
do
	local s, pos
	local function ws()
		pos = s:find("[^ \t\r\n]", pos) or #s + 1
	end
	local function str()
		local out = {}
		pos = pos + 1
		while true do
			local c = s:sub(pos, pos)
			if c == '"' then
				pos = pos + 1
				return table.concat(out)
			elseif c == "\\" then
				local e = s:sub(pos + 1, pos + 1)
				if e == "u" then
					out[#out + 1] = utf8.char(tonumber(s:sub(pos + 2, pos + 5), 16))
					pos = pos + 6
				else
					local map = { b = "\b", f = "\f", n = "\n", r = "\r", t = "\t" }
					out[#out + 1] = map[e] or e
					pos = pos + 2
				end
			else
				out[#out + 1] = c
				pos = pos + 1
			end
		end
	end
	local value
	local function list(close, item)
		pos = pos + 1
		ws()
		if s:sub(pos, pos) == close then
			pos = pos + 1
			return
		end
		repeat
			ws()
			item()
			ws()
			local c = s:sub(pos, pos)
			pos = pos + 1
		until c == close
	end
	function value()
		ws()
		local c = s:sub(pos, pos)
		if c == "{" then
			local t, order = {}, {}
			list("}", function()
				local k = str()
				ws()
				pos = pos + 1
				t[k] = value()
				order[#order + 1] = k
			end)
			return setmetatable(t, { __order = order, __jsontype = "object" })
		elseif c == "[" then
			local t = {}
			list("]", function()
				t[#t + 1] = value()
			end)
			return setmetatable(t, { __jsontype = "array" })
		elseif c == '"' then
			return str()
		end
		local lit = s:match("^[%w%.%+%-]+", pos)
		pos = pos + #lit
		if lit == "true" or lit == "false" then
			return lit == "true"
		elseif lit == "null" then
			return tojson.null
		end
		return math.tointeger(tonumber(lit)) or tonumber(lit)
	end
	function decode(text)
		s, pos = text, 1
		return value()
	end
end

local function slurp(path)
	local f = assert(io.open(path, "rb"))
	local d = f:read("a")
	f:close()
	return d
end

local function fixtures(sub)
	local names = {}
	local p = assert(io.popen('ls "' .. dir .. "/" .. sub .. '"'))
	for line in p:lines() do
		local base = line:match("^(.*)%.json$")
		if base then
			names[#names + 1] = base
		end
	end
	p:close()
	table.sort(names)
	return names
end

-- Show the first differing byte on stderr.
local function diff(name, got, want)
	local i = 1
	while i <= #want and got:byte(i) == want:byte(i) do
		i = i + 1
	end
	io.stderr:write(string.format("%s: differs at byte %d\n want: %q\n got:  %q\n",
		name, i, want:sub(i, i + 60), got:sub(i, i + 60)))
end

local function test_fixtures()
	local names = fixtures("chat")
	ok(#names >= 12, "at least 12 fixtures")
	for _, name in ipairs(names) do
		local req = decode(slurp(dir .. "/chat/" .. name .. ".json"))
		local kw = req.chat_template_kwargs or {}
		local got = qwen3.render(req.messages, {
			tools = req.tools,
			add_generation_prompt = req.add_generation_prompt ~= false,
			enable_thinking = kw.enable_thinking,
			continue_final_message = req.continue_final_message,
		})
		local want = slurp(dir .. "/chat/" .. name .. ".txt")
		if not ok(got == want, "render " .. name) then
			diff(name, got, want)
		end
	end
end

-- A final assistant message is continued, as llama-server does.
local function test_gemma4_fixtures()
	local names = fixtures("chat/gemma4")
	ok(#names >= 30, "at least 30 gemma4 fixtures")
	for _, name in ipairs(names) do
		local req = decode(slurp(dir .. "/chat/gemma4/" .. name .. ".json"))
		local kw = req.chat_template_kwargs or {}
		local last = req.messages[#req.messages]
		local got = gemma4.render(req.messages, {
			tools = req.tools,
			add_generation_prompt = req.add_generation_prompt ~= false,
			enable_thinking = kw.enable_thinking,
			continue_final_message = req.continue_final_message or last.role == "assistant",
		})
		local want = slurp(dir .. "/chat/gemma4/" .. name .. ".txt")
		if not ok(got == want, "gemma4 render " .. name) then
			diff(name, got, want)
		end
	end
end

-- Fixtures in qwen35/ come from Ornith-1.5 and in qwen35/effort/ from
-- Qwen3.8. A final assistant message is continued.
local function test_qwen35_fixtures(sub, variant)
	local names = fixtures(sub)
	ok(#names >= 30, "at least 30 fixtures in " .. sub)
	for _, name in ipairs(names) do
		local req = decode(slurp(dir .. "/" .. sub .. "/" .. name .. ".json"))
		local kw = req.chat_template_kwargs or {}
		local last = req.messages[#req.messages]
		local got = qwen35.render(req.messages, {
			tools = req.tools,
			variant = variant,
			add_generation_prompt = req.add_generation_prompt ~= false,
			enable_thinking = kw.enable_thinking,
			reasoning_effort = kw.reasoning_effort,
			preserve_thinking = kw.preserve_thinking,
			continue_final_message = req.continue_final_message or last.role == "assistant",
		})
		local want = slurp(dir .. "/" .. sub .. "/" .. name .. ".txt")
		if not ok(got == want, sub .. " render " .. name) then
			diff(name, got, want)
		end
	end
end

local function test_tojson()
	local floats = {
		{ 1.0, "1.0" }, { 2.5, "2.5" }, { 1e16, "1e+16" }, { 1e15, "1000000000000000.0" },
		{ 1e-5, "1e-05" }, { 0.0001, "0.0001" }, { -0.0, "-0.0" }, { 1 / 3, "0.3333333333333333" },
		{ 1.5e300, "1.5e+300" }, { 123456.0, "123456.0" }, { 5e-324, "5e-324" },
	}
	local good = true
	for _, f in ipairs(floats) do
		if tojson.encode(f[1]) ~= f[2] then
			io.stderr:write("float ", f[2], " got ", tojson.encode(f[1]), "\n")
			good = false
		end
	end
	ok(good, "tojson floats match Python repr")
	local t = setmetatable({ b = 1, a = { 1, "x" }, c = tojson.null }, { __order = { "c", "b" } })
	ok(tojson.encode(t) == '{"c": null, "b": 1, "a": [1, "x"]}', "tojson key order")
	ok(tojson.encode("a\"\\\n\1\127é") == '"a\\"\\\\\\n\\u0001\127é"', "tojson string escapes")
	ok(tojson.encode(setmetatable({}, { __jsontype = "array" })) == "[]", "tojson empty array")
end

local function test_detect()
	local tpl = "<|im_start|>... <tool_call> ..."
	local m, name = chat.load({ ["general.architecture"] = "qwen3", ["tokenizer.chat_template"] = tpl })
	ok(m == qwen3 and name == "qwen3", "detect qwen3")
	local r, err = chat.detect({ ["general.architecture"] = "llama" })
	ok(r == nil and err:find("llama", 1, true), "detect other arch gives error")
	ok(not pcall(chat.load, { ["general.architecture"] = "gemma3" }), "load other arch raises")
	local g = "{{- '<|tool_call>call:' + function['name'] + '{' -}}"
	m, name = chat.load({ ["general.architecture"] = "diffusion-gemma", ["tokenizer.chat_template"] = g })
	ok(m == gemma4 and name == "gemma4", "detect gemma4 by template")
	ok(chat.detect({ ["general.architecture"] = "gemma4" }) == "gemma4", "detect gemma4 by arch")
	local q = "<|im_start|> {{- tool | tojson }} <think> <tool_call>\n<function="
	m, name = chat.load({ ["general.architecture"] = "qwen35moe", ["tokenizer.chat_template"] = q })
	local u = { { role = "user", content = "hi" } }
	ok(name == "qwen35" and m.render(u, {}) == qwen35.render(u, {}), "detect qwen35")
	m = chat.load({ ["general.architecture"] = "qwen35",
		["tokenizer.chat_template"] = q .. " sysns reasoning_effort" })
	ok(m.render(u, {}) == qwen35.render(u, { variant = "effort" }), "detect qwen35 effort variant")
	r = chat.detect({ ["general.architecture"] = "qwen3moe",
		["tokenizer.chat_template"] = "<|im_start|> <tool_call>\n<function=" })
	ok(r == nil, "Qwen3-Coder template is not qwen3")
end

-- Merge adjacent text events so split points do not matter.
local function coalesce(evs)
	local out = {}
	for _, e in ipairs(evs) do
		local last = out[#out]
		if e.text and last and last.text and last.type == e.type and last.index == e.index then
			last.text = last.text .. e.text
		else
			out[#out + 1] = { type = e.type, index = e.index, name = e.name, text = e.text, error = e.error }
		end
	end
	return out
end

local function run(pieces, opts, new)
	local p = (new or qwen3.new_parser)(opts)
	local all = {}
	for _, s in ipairs(pieces) do
		local evs = p:feed(s)
		table.move(evs, 1, #evs, #all + 1, all)
	end
	for _, e in ipairs(p:finish()) do
		all[#all + 1] = e
	end
	return all
end

local function show(evs)
	local parts = {}
	for _, e in ipairs(evs) do
		parts[#parts + 1] = string.format("%s(%s,%s,%q,%s)", e.type, e.index, e.name,
			e.text or "", e.error)
	end
	return table.concat(parts, " ")
end

local function same(a, b)
	return show(a) == show(b)
end

-- Feed s split at every offset, as single bytes, and in random pieces.
local function fuzz(s, opts, want, new)
	local base = run
	local function run(pieces, o)
		return base(pieces, o, new)
	end
	local ref = coalesce(run({ s }, opts))
	local good = same(ref, want)
	if not good then
		io.stderr:write(" want: ", show(want), "\n got:  ", show(ref), "\n")
	end
	for i = 0, #s do
		good = good and same(coalesce(run({ s:sub(1, i), s:sub(i + 1) }, opts)), ref)
	end
	local bytes = {}
	for i = 1, #s do
		bytes[i] = s:sub(i, i)
	end
	local evs = run(bytes, opts)
	for _, e in ipairs(evs) do
		good = good and (not e.text or utf8.len(e.text) ~= nil)
	end
	good = good and same(coalesce(evs), ref)
	math.randomseed(#s)
	for _ = 1, 50 do
		local pieces, i = {}, 1
		while i <= #s do
			local k = math.random(1, 7)
			pieces[#pieces + 1] = s:sub(i, i + k - 1)
			i = i + k
		end
		good = good and same(coalesce(run(pieces, opts)), ref)
	end
	return good
end

local function C(t)
	return { type = "content", text = t }
end
local function R(t)
	return { type = "reasoning", text = t }
end
local function B(i, name)
	return { type = "tool_call_begin", index = i, name = name }
end
local function A(i, t)
	return { type = "tool_call_args", index = i, text = t }
end
local function E(i, err)
	return { type = "tool_call_end", index = i, error = err }
end

local BAD = "invalid tool call"

local parser_cases = {
	{
		"think, content, two calls",
		"<think>\nLet me see.\n</think>\n\nSure.\n<tool_call>\n"
			.. '{"name": "get_weather", "arguments": {"city": "Zürich", "n": [1, -2.5e3, true, null], "s": "a\\"}\\u00e9"}}\n'
			.. '</tool_call>\n<tool_call>\n{"name": "now", "arguments": {}}\n</tool_call>',
		nil,
		{ R("Let me see."), C("Sure."), B(0, "get_weather"),
			A(0, '{"city": "Zürich", "n": [1, -2.5e3, true, null], "s": "a\\"}\\u00e9"}'), E(0),
			B(1, "now"), A(1, "{}"), E(1) },
	},
	{
		"prompt opened think",
		"I should answer.\n</think>\n\nThe answer is 4. 😀",
		{ in_think = true },
		{ R("I should answer."), C("The answer is 4. 😀") },
	},
	{
		"plain content with angle brackets",
		"a < b <to <think> x </tool_call> <tool_cal",
		nil,
		{ C("a < b <to <think> x </tool_call> <tool_cal") },
	},
	{
		"broken head falls back to content",
		"Hi.\n<tool_call>\nnot json\n</tool_call>\nBye.",
		nil,
		{ C("Hi.\n<tool_call>\nnot json\n</tool_call>\nBye.") },
	},
	{
		"unknown tool name falls back to content",
		'<tool_call>\n{"name": "rm", "arguments": {}}\n</tool_call>',
		{ tools = { "now" } },
		{ C('<tool_call>\n{"name": "rm", "arguments": {}}\n</tool_call>') },
	},
	{
		"bad arguments end the call with an error",
		'<tool_call>\n{"name": "now", "arguments": {"a": 1x}}\n</tool_call>',
		nil,
		{ B(0, "now"), A(0, '{"a": 1'), E(0, BAD), C("x}}\n</tool_call>") },
	},
	{
		"missing close tag after brace",
		'<tool_call>\n{"name": "now", "arguments": {"a": [0.5]}}',
		nil,
		{ B(0, "now"), A(0, '{"a": [0.5]}'), E(0) },
	},
	{
		"truncated arguments",
		'<tool_call>\n{"name": "now", "arguments": {"a": "b',
		nil,
		{ B(0, "now"), A(0, '{"a": "b'), E(0, BAD) },
	},
	{
		"truncated head",
		'text <tool_call>\n{"name": "no',
		nil,
		{ C('text <tool_call>\n{"name": "no') },
	},
	{
		"content after call and empty think",
		'<think>\n\n</think>\n\n<tool_call>\n{"name":"now","arguments":{"k":"v"}}\n</tool_call>\nDone.',
		nil,
		{ B(0, "now"), A(0, '{"k":"v"}'), E(0), C("Done.") },
	},
	{
		"unclosed think",
		"<think>\nstill thinking <tool_call> here",
		nil,
		{ R("still thinking <tool_call> here") },
	},
}

local function test_parser()
	for _, c in ipairs(parser_cases) do
		ok(fuzz(c[2], c[3], c[4]), "parser " .. c[1])
	end
end

local Q = '<|"|>'

local gemma4_cases = {
	{
		"thought, content, two calls",
		"<|channel>thought\nLet me see.<channel|>Sure.<|tool_call>call:get_weather{city:" .. Q
			.. "Zürich" .. Q .. ",n:[1,-2.5e3,true,null],s:" .. Q .. 'a"}\\<|' .. Q
			.. ",o:{k:{}}}<tool_call|><|tool_call>call:now{}<tool_call|>",
		nil,
		{ R("Let me see."), C("Sure."), B(0, "get_weather"),
			A(0, '{"city":"Zürich","n":[1,-2.5e3,true,null],"s":"a\\"}\\\\<|","o":{"k":{}}}'),
			E(0), B(1, "now"), A(1, "{}"), E(1) },
	},
	{
		"turn header, empty thought",
		"<|turn>model\n<|channel>thought\n<channel|>The answer is 4. 😀",
		nil,
		{ C("The answer is 4. 😀") },
	},
	{
		"prompt opened thought",
		"I should answer.\n<channel|>\nDone",
		{ in_think = true },
		{ R("I should answer."), C("Done") },
	},
	{
		"plain content with tag-like text",
		"a < b <|chan <|tool <channel <|\"|> x",
		nil,
		{ C('a < b <|chan <|tool <channel <|"|> x') },
	},
	{
		"stray markers are dropped",
		"one<channel|>two<|channel>three<|channel><|channel>thought x<channel|>four",
		nil,
		{ C("onetwothree"), R("x"), C("four") },
	},
	{
		"unknown tool name falls back to content",
		"<|tool_call>call:rm{}<tool_call|>",
		{ tools = { "now" } },
		{ C("<|tool_call>call:rm{}<tool_call|>") },
	},
	{
		"broken head falls back to content",
		"Hi.<|tool_call>cal now{}<tool_call|> Bye.",
		nil,
		{ C("Hi.<|tool_call>cal now{}<tool_call|> Bye.") },
	},
	{
		"bad arguments end the call with an error",
		"<|tool_call>call:now{a:1x}<tool_call|>",
		nil,
		{ B(0, "now"), A(0, '{"a":'), E(0, BAD), C("}<tool_call|>") },
	},
	{
		"missing close tag after brace",
		"<|tool_call>call:now{a:[0.5, " .. Q .. "b" .. Q .. "]}",
		nil,
		{ B(0, "now"), A(0, '{"a":[0.5,"b"]}'), E(0) },
	},
	{
		"truncated string",
		"<|tool_call>call:now{a:" .. Q .. "b<|\"",
		nil,
		{ B(0, "now"), A(0, '{"a":"b'), E(0, BAD) },
	},
	{
		"truncated head",
		"text <|tool_call>call:no",
		nil,
		{ C("text <|tool_call>call:no") },
	},
	{
		"content after call, key escapes",
		"<|tool_call>call:now{a b\\:" .. Q .. "x\ny\1" .. Q .. "}<tool_call|>\nDone.",
		nil,
		{ B(0, "now"), A(0, '{"a b\\\\":"x\\ny\\u0001"}'), E(0), C("Done.") },
	},
	{
		"unclosed thought",
		"<|channel>thought\nstill thinking <|tool_call> here",
		nil,
		{ R("still thinking <|tool_call> here") },
	},
}

local function test_gemma4_parser()
	for _, c in ipairs(gemma4_cases) do
		ok(fuzz(c[2], c[3], c[4], gemma4.new_parser), "gemma4 parser " .. c[1])
	end
end

local QT = { { type = "function", ["function"] = { name = "get_weather", parameters = {
	type = "object", ["$defs"] = { city = { type = "string" } },
	properties = { city = { ["$ref"] = "#/$defs/city" }, days = { type = "integer" },
		opts = { type = "object" }, unit = { anyOf = { { type = "null" }, { enum = { "c", "f" } } } } },
	required = { "city" } } } },
	{ type = "function", ["function"] = { name = "now" } } }
local QOFF = { tools = QT, enable_thinking = false }

local function qcall(name, params)
	local s = "<tool_call>\n<function=" .. name .. ">\n"
	for _, p in ipairs(params) do
		s = s .. "<parameter=" .. p[1] .. ">\n" .. p[2] .. "\n</parameter>\n"
	end
	return s .. "</function>\n</tool_call>"
end

local qwen35_cases = {
	{
		"reasoning, content, two calls, typed values",
		"Plan it.\n</think>\n\nSure.\n" .. qcall("get_weather", { { "city", 'Zürich "x"' },
			{ "days", "3" }, { "opts", '{"a": [1, true]}' }, { "unit", "c" } }) .. "\n" .. qcall("now", {}),
		{ tools = QT },
		{ R("Plan it."), C("Sure."), B(0, "get_weather"),
			A(0, '{"city":"Zürich \\"x\\"","days":3,"opts":{"a": [1, true]},"unit":"c"}'), E(0),
			B(1, "now"), A(1, "{}"), E(1) },
	},
	{
		"thinking off, tag without newline is content",
		"Hi <tool_call> there 😀",
		{ enable_thinking = false },
		{ C("Hi <tool_call> there 😀") },
	},
	{
		"think block in output",
		"<think>\nr\n</think>\n\nok",
		{ in_think = false },
		{ R("r"), C("ok") },
	},
	{
		"unknown tool name falls back to content",
		qcall("rm", {}),
		QOFF,
		{ C(qcall("rm", {})) },
	},
	{
		"unknown parameter ends the call with an error",
		qcall("now", { { "x", "1" } }),
		QOFF,
		{ B(0, "now"), A(0, "{"), E(0, BAD), C(">\n1\n</parameter>\n</function>\n</tool_call>") },
	},
	{
		"bad JSON for an integer parameter",
		qcall("get_weather", { { "city", "A" }, { "days", "3 " } }),
		QOFF,
		{ B(0, "get_weather"), A(0, '{"city":"A","days":3'), E(0, BAD),
			C(" \n</parameter>\n</function>\n</tool_call>") },
	},
	{
		"missing required parameter",
		qcall("get_weather", { { "days", "3" } }),
		QOFF,
		{ B(0, "get_weather"), A(0, '{"days":3}'), E(0, BAD) },
	},
	{
		"string value with close-like text",
		qcall("get_weather", { { "city", "a\n</parameter\nb</parameter>\n\n" } }),
		QOFF,
		{ B(0, "get_weather"), A(0, '{"city":"a\\n</parameter\\nb</parameter>\\n\\n"}'), E(0) },
	},
	{
		"no newline after the call tag",
		"<tool_call><function=now>\n</function>\n</tool_call>",
		QOFF,
		{ C("<tool_call><function=now>\n</function>\n</tool_call>") },
	},
	{
		"truncated value",
		"<tool_call>\n<function=get_weather>\n<parameter=city>\nPar",
		QOFF,
		{ B(0, "get_weather"), A(0, '{"city":"Par'), E(0, BAD) },
	},
	{
		"missing close tag after function",
		"<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
		QOFF,
		{ B(0, "get_weather"), A(0, '{"city":"Paris"}'), E(0) },
	},
	{
		"tools without schemas",
		qcall("f", { { "a", "[1, 2]" }, { "b", "hello world" }, { "c", "" } }),
		{ tools = { "f" }, enable_thinking = false },
		{ B(0, "f"), A(0, '{"a":[1, 2],"b":"hello world","c":""}'), E(0) },
	},
	{
		"space around tags, content after call",
		"<tool_call>\n  <function=now>\n\n</function>\n \n</tool_call>\nDone.",
		QOFF,
		{ B(0, "now"), A(0, "{}"), E(0), C("Done.") },
	},
	{
		"unclosed think",
		"still thinking <tool_call>\n<function=now>",
		{ tools = QT },
		{ R("still thinking <tool_call>\n<function=now>") },
	},
}

local function test_qwen35_parser()
	for _, c in ipairs(qwen35_cases) do
		ok(fuzz(c[2], c[3], c[4], qwen35.new_parser), "qwen35 parser " .. c[1])
	end
end

local function test_qwen35_render()
	local m = { { role = "user", content = "q" }, { role = "system", content = "late" } }
	ok(not pcall(qwen35.render, m, { variant = "effort" }), "qwen35 effort: late system raises")
	ok(pcall(qwen35.render, m, {}), "qwen35: late system is dropped")
	local u = { { role = "user", content = "q" } }
	ok(not pcall(qwen35.render, u, { variant = "effort", reasoning_effort = "max" }),
		"qwen35 effort: bad reasoning effort raises")
	local bad = { { role = "user", content = "q" }, { role = "assistant", content = "",
		tool_calls = { { type = "function", ["function"] = { name = "f", arguments = "{bad" } } } },
		{ role = "user", content = "q2" } }
	ok(not pcall(qwen35.render, bad, {}), "qwen35: arguments that are not JSON raise")
end

-- Image content parts: data URLs, and the llama-server text joins.
local function test_media()
	local png = "\137PNG\r\n\26\n"
	local bytes, mime = media.data_url("data:image/png;base64,iVBORw0KGgo=")
	ok(bytes == png and mime == "image/png", "media: data URL decodes")
	bytes, mime = media.data_url("DATA:Image/JPEG;charset=x;BASE64,iVBO Rw0K\nGgo=")
	ok(bytes == png and mime == "image/jpeg", "media: scheme, type and base64 flag ignore case")
	local cases = {
		{ "https://example.com/cat.png", "http and https URLs are not supported" },
		{ "http://x/y", "not supported" },
		{ "file:///etc/passwd", "must be a data URL" },
		{ "cat.png", "must be a data URL" },
		{ "data:image/png;base64", "no comma" },
		{ "data:text/plain;base64,aGk=", "image type" },
		{ "data:;base64,aGk=", "image type" },
		{ "data:image/png,rawbytes", "must be base64" },
		{ "data:image/png;base64,a!==", "base64" },
		{ "data:image/png;base64,", "no data" },
	}
	for _, c in ipairs(cases) do
		local b, err = media.data_url(c[1])
		ok(b == nil and err and err:find(c[2], 1, true), "media: " .. c[1] .. " gives an error")
	end
	ok(select(2, media.data_url(42)):find("string", 1, true), "media: a URL that is not a string")

	local url = "data:image/png;base64,iVBORw0KGgo="
	local msgs = {
		{ role = "system", content = "sys" },
		{ role = "user", content = {
			{ type = "text", text = "a" }, { type = "text", text = "b" },
			{ type = "image_url", image_url = { url = url } },
			{ type = "text", text = "c" }, { type = "image_url", image_url = url },
			{ type = "image_url", image_url = { url = url } }, { type = "text", text = "" },
			{ type = "text", text = "d" }, { type = "input_audio" } } },
		{ role = "user", content = { { type = "text", text = "only text" } } },
	}
	ok(media.has_images(msgs) and not media.has_images({ msgs[1], msgs[3] }),
	    "media: has_images finds image parts")
	local out, images = media.extract(msgs, "<M>")
	ok(out[2].content == "a\nb<M>c<M><M>\nd" and #images == 3 and images[1].bytes == png,
	    "media: text joins as llama-server: " .. tostring(out[2].content))
	ok(out[1] == msgs[1] and out[3] == msgs[3] and msgs[2].content[1].text == "a",
	    "media: other messages and the input stay as they are")
	local lead = media.extract({ { role = "user", content = {
		{ type = "text", text = "" }, { type = "text", text = "x" } , { type = "image_url",
		image_url = { url = url } } } } }, "<M>")
	ok(lead[1].content == "x<M>", "media: no newline after empty text")
	local bad, err = media.extract({ { role = "user", content = {
		{ type = "text", text = "x" }, { type = "image_url", image_url = { url = 5 } } } } }, "<M>")
	ok(bad == nil and err == "messages[0].content[1]: image_url.url must be a string",
	    "media: a bad part names its place: " .. tostring(err))
	bad, err = media.extract({ { role = "user", content = {
		{ type = "image_url", image_url = { url = "https://x/y.png" } } } } }, "<M>")
	ok(bad == nil and err:find("^messages%[0%]%.content%[0%]: image url must be a data URL"),
	    "media: an http URL is an error with its place")

	local im = gemma4.image
	local rendered = gemma4.render(media.extract({ { role = "user", content = {
		{ type = "image_url", image_url = { url = url } },
		{ type = "text", text = " What is this? " } } } }, im.begin .. im.token .. im.finish),
		{ add_generation_prompt = true, enable_thinking = false })
	ok(rendered == "<|turn>user\n<|image><|image|><image|> What is this?<turn|>\n<|turn>model\n",
	    "gemma4: an image renders as llama-server: " .. rendered)
end

test_tojson()
test_detect()
test_media()
test_fixtures()
test_gemma4_fixtures()
test_qwen35_fixtures("chat/qwen35", nil)
test_qwen35_fixtures("chat/qwen35/effort", "effort")
test_parser()
test_gemma4_parser()
test_qwen35_parser()
test_qwen35_render()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
