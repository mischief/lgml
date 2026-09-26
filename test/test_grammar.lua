-- GBNF grammar tests on the CPU: parse errors, string matches, and
-- token masks with a tiny vocab. Output is TAP.
local lgml = require("lgml")

local n = 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		os.exit(1)
	end
end

-- Parse errors

local bad = {
	{ "root ::= foo", "undefined rule 'foo'" },
	{ "root ::= \"a", "unexpected end" },
	{ "root ::= [a-", "unexpected end" },
	{ "root ::= (\"a\"", "expecting '%)'" },
	{ "root ::= \"a\" )", "expecting newline" },
	{ "root = \"a\"", "expecting ::=" },
	{ "root ::= \"\\q\"", "unknown escape" },
	{ "root ::= \"\\x4\"", "expecting hex" },
	{ "root ::= *", "repetition" },
	{ "root ::= \"a\"{2,1}", "repetition" },
	{ "root ::= \"a\"{3000}", "too many repetitions" },
	{ "root ::= \"a\"{1234567890}", "number too large" },
	{ "root ::= \"a\"{2", "expecting ',' or '}'" },
	{ "root ::= \"a\"{2,3", "expecting '}'" },
	{ "root ::= <foo>", "no vocab" },
	{ "root ::= <[1]", "expecting ']>'" },
	{ "x ::= \"a\"", "no rule 'root'" },
	{ "", "no rule 'root'" },
	{ "root ::= \"a\" | root \"a\"", "left recursion" },
	{ "root ::= asdf\nasdf ::= \"a\" | foo \"b\"\nfoo ::= \"c\" | asdf \"d\" | \"e\"",
	    "left recursion" },
	{ "root ::= asdf\nasdf ::= \"a\" | foo \"b\"\nfoo ::= \"c\" | empty asdf \"d\" | \"e\"\n" ..
	    "empty ::= \"blah\" | ", "left recursion" },
	{ "root ::= " .. ("(" ):rep(100) .. "\"a\"" .. (")"):rep(100), "nest too deep" },
	{ "root ::= (\"a\"{0,50}){0,50}", "repetition" },
	{ "root ::= \"a\"\0\"b\"", "NUL" },
	{ "root ::= a\na ::= b\nb ::= \"x\" a", nil },
}
for _, c in ipairs(bad) do
	local g, err = lgml.grammar(c[1])
	local src = c[1]:sub(1, 40):gsub("%c", " ")
	if c[2] then
		ok(g == nil and err:find(c[2]) ~= nil,
		    ("parse error '%s': %s"):format(src, tostring(err)))
	else
		ok(g ~= nil, ("parses '%s'"):format(src))
	end
end

ok(lgml.grammar("root ::= <foo>", { lax = true }) ~= nil, "lax token names parse")
ok(lgml.grammar("foobar ::= \"x\"", { root = "foobar" }) ~= nil, "custom root")
ok(lgml.grammar("root ::= \"x\"", { root = "nope" }) == nil, "missing custom root")

-- Deep but legal right recursion stays in bounds.
do
	local g = assert(lgml.grammar("root ::= \"(\" root \")\" | \"x\""))
	local r = g:accept(("("):rep(2000))
	ok(r == false, "stack depth is bounded")
end

-- A wide alternation is bounded by the stack limit.
do
	local alts = {}
	for i = 1, 5000 do
		alts[i] = ("\"%d\""):format(i)
	end
	local g, err = lgml.grammar("root ::= x\nx ::= " .. table.concat(alts, " | "))
	ok(g == nil and err:find("too many parse stacks"), "stack count is bounded")
end

-- String matching, from llama.cpp test-grammar-integration.

local function matches(text, s)
	local g = assert(lgml.grammar(text))
	return g:accept(s) and g:can_end()
end

local cases = {
	{ "simple grammar", [[
		root ::= expr
		expr ::= term ("+" term)*
		term ::= number
		number ::= [0-9]+]],
	    { "42", "1+2+3+4+5", "123+456" },
	    { "+", "/ 3", "1+2+3+4+5+", "12a45" } },
	{ "medium complexity grammar", [[
		root ::= expression
		expression ::= term ws (("+"|"-") ws term)*
		term ::= factor ws (("*"|"/") ws factor)*
		factor ::= number | variable | "(" expression ")" | function-call
		number ::= [0-9]+
		variable ::= [a-zA-Z_][a-zA-Z0-9_]*
		function-call ::= variable ws "(" (expression ("," ws expression)*)? ")"
		ws ::= [ \t\n\r]?]],
	    { "42", "1*2*3*4*5", "x", "x+10", "x1+y2", "(a+b)*(c-d)", "func()",
	      "func(x,y+2)", "a*(b+c)-d/e", "f(g(x),h(y,z))", "x + 10",
	      "(a + b) * (c - d)", "func(x, y + 2)", "f(g(x), h(y, z))",
	      "123*456*789-123/456+789*123" },
	    { "+", "/ 3x", "x + + y", "a * / b", "func(,)", "func(x y)", "(a + b",
	      "x + y)", "42 +", "func(", "f(g(x), h(y, z)" } },
	{ "special characters", [[root ::= ... "abc" ...]],
	    { "abcabcabc", "aaaabcccc", "🔵🟠✅abc❌🟠🔵" },
	    { "aaabcccc", "aaaaabcccc", "aaaabccc", "aaaabccccc", "🔵🟠✅❌abc❌✅🟠🔵",
	      "🔵🟠abc🟠🔵" } },
	{ "* quantifier", [[root ::= "a"*]],
	    { "", "a", "aaaaa", ("a"):rep(60) }, { "b", "ab", "aab", "ba", ("a"):rep(60) .. "b" } },
	{ "+ quantifier", [[root ::= "a"+]], { "a", "aaaaa" }, { "", "b", "ab", "ba" } },
	{ "? quantifier", [[root ::= "a"?]], { "", "a" }, { "b", "ab", "aa", "ba" } },
	{ "mixed quantifiers", [[
		root ::= cons+ vowel* cons? (vowel cons)*
		vowel ::= [aeiouy]
		cons ::= [bcdfghjklmnpqrstvwxyz]
	]], { "yes", "no", "noyes", "crwth", "four", "bryyyy" },
	    { "yess", "yesno", "forty", "catyyy" } },
	{ "exact repetition", [[root ::= [ab]{4}]], { "aaaa", "bbbb", "abab" }, { "a", "b", "aaaaa" } },
	{ "min repetition", [[root ::= [ab]{4,}]], { "aaaa", "aaaaab", "ababab" }, { "", "aba" } },
	{ "max repetition", [[root ::= [ab]{0,4}]], { "", "a", "aaab" }, { "aaaaa" } },
	{ "min / max repetition", [[root ::= ("0x" [A-F0-9]{2} " "?){3,5}]],
	    { "0xFF 0x12 0xAB", "0xFF 0x12 0xAB 0x00 0x00" },
	    { "", "0xFF", "0xFF 0x12", "0xFF 0x12 0xAB 0x00 0x00 0x00" } },
	{ "nested stars", [[root ::= ( [x]* )*]], { "", "x", "xx" }, { "y", "yy" } },
	{ "negated classes and escapes", [[
		root ::= [^a-c\n] "\x41\u00e4\U0001F600" [\]-] "\"" # comment
	]], { "dA\u{e4}\u{1f600}]\"", "\u{e9}A\u{e4}\u{1f600}-\"" },
	    { "bA\u{e4}\u{1f600}]\"", "\nA\u{e4}\u{1f600}]\"", "dA\u{e4}\u{1f600}x\"" } },
	{ "comments and newlines in groups", "root ::= (\n  \"a\" # one\n | \"b\"\n)\r\n",
	    { "a", "b" }, { "", "ab" } },
}
for _, c in ipairs(cases) do
	for _, s in ipairs(c[3]) do
		ok(matches(c[2], s), c[1] .. ": match " .. s:sub(1, 30):gsub("%c", " "))
	end
	for _, s in ipairs(c[4]) do
		ok(not matches(c[2], s), c[1] .. ": no match " .. s:sub(1, 30):gsub("%c", " "))
	end
end

-- Char by char: state stays after a bad char; partial UTF-8 waits.
do
	local g = assert(lgml.grammar([[root ::= "{" [0-9]+ "}"]]))
	ok(g:accept("{") and not g:can_end(), "accept one char")
	ok(g:accept("1") and g:accept("2"), "accept digits")
	local r, err = g:accept("x")
	ok(r == false and err == "input does not fit", "reject keeps the state")
	ok(g:accept("}") and g:can_end(), "complete after a reject")
	ok(not g:accept("}"), "nothing after the end")

	local u = assert(lgml.grammar([[root ::= "ä" [€-€] "!"]]))
	ok(u:accept("\xc3") and u:accept("\xa4"), "split two-byte char")
	ok(u:accept("\xe2") and u:accept("\x82") and u:accept("\xac!"), "split three-byte char")
	ok(u:can_end(), "split chars complete")
	local v = assert(lgml.grammar([[root ::= "ä"]]))
	ok(not v:accept("\xa4"), "stray continuation byte")
	ok(not v:accept("\xc3\x41"), "bad continuation byte")
	ok(not v:accept("\xe2"), "partial that cannot become the char")
end

-- Token masks with a tiny vocab.

local function vocab(pieces, eog)
	local t = {}
	function t.n_vocab()
		return #pieces
	end
	function t.token(_, id)
		return pieces[id + 1], "normal"
	end
	function t.special()
		return { eog = eog or {} }
	end
	return t
end

local function set(ids)
	return table.concat(ids, ",")
end

local be <close> = assert(lgml.backend("CPU"))

do
	--          0    1     2     3    4    5    6       7       8     9      10
	local p = { "{", "\"", "a", "ab", "}", "", "\xc3", "\xa4", "ä}", "</s>", "\0x" }
	local tk = vocab(p, { 9 })
	local text = [[root ::= "{" "\"" [aä]* "\"" "}"]]
	local s = lgml.sampler({ grammar = text, tokenizer = tk, temperature = 0 })
	ok(set(s:allowed()) == "0", "start: only {")
	s:accept(0)
	ok(set(s:allowed()) == "1", "then a quote")
	s:accept(1)
	ok(set(s:allowed()) == "1,2,6", "in string: quote, a, first byte of ä")
	s:accept(6)
	ok(set(s:allowed()) == "7", "partial UTF-8: only the second byte")
	s:accept(7)
	s:accept(1)
	ok(set(s:allowed()) == "4", "close brace")
	ok(not s:can_end(), "cannot end before the brace")
	s:accept(4)
	ok(s:can_end() and set(s:allowed()) == "9", "only end of generation at the end")
	s:accept(9)
	ok(not pcall(s.accept, s, 0), "accept of a token that does not fit raises")

	-- The sampler picks the allowed token although another has the top logit.
	local g = lgml.sampler({ grammar = text, tokenizer = tk, temperature = 0 })
	local x <close> = be:tensor("f32", { 1, #p }, { 1, 2, 9, 3, 4, 0, 0, 0, 0, 20, 30 })
	ok(g:sample(x) == 0, "greedy picks the only fit")
	g:accept(0)
	g:accept(1)
	ok(g:sample(x) == 2, "greedy picks the best fit")
	local r = lgml.sampler({ grammar = text, tokenizer = tk, temperature = 1, top_k = 3,
	    top_p = 1, min_p = 0, seed = 7 })
	r:accept(0)
	r:accept(1)
	local seen = {}
	local y <close> = be:tensor("f32", { 1, #p }, { 1, 3, 3, 0, 0, 0, 3, 0, 0, 20, 30 })
	for _ = 1, 200 do
		seen[r:sample(y)] = true
	end
	ok(seen[1] and seen[2] and not seen[9] and not seen[10] and not seen[4],
	    "sampling keeps to the grammar")
	local ids = {}
	for id in pairs(seen) do
		ids[#ids + 1] = id
	end
	table.sort(ids)
	ok(set(ids) == "1,2,6", "sampled set " .. set(ids))

	-- Logits may be longer than the vocab; the tail never fits.
	local long <close> = be:tensor("f32", { 1, #p + 2 }, { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50 })
	local q = lgml.sampler({ grammar = text, tokenizer = tk, temperature = 0 })
	ok(q:sample(long) == 0, "ids past the vocab are masked")
end

-- Token elements.
do
	local p = { "<a>", "x", "<b>", "y", "</s>" }
	local tk = vocab(p, { 4 })
	local s = lgml.sampler({ grammar = "root ::= <[0]> (!<[2]>)* <b>", tokenizer = tk })
	ok(set(s:allowed()) == "0", "token start")
	s:accept(0)
	ok(set(s:allowed()) == "0,1,2,3", "any token but <b>, or <b>")
	s:accept(1)
	s:accept(3)
	s:accept(2)
	ok(set(s:allowed()) == "4", "end after <b>")
	ok(select(2, pcall(lgml.sampler, { grammar = "root ::= <zz>", tokenizer = tk }))
	    :find("unknown token"), "unknown token name")
	ok(lgml.grammar("root ::= <b>", { tokenizer = tk }) ~= nil, "token names with a tokenizer")
end

-- The vocab table is built once per tokenizer.
do
	local calls = 0
	local tk = vocab({ "a", "b" })
	local tok = tk.token
	tk.token = function(...)
		calls = calls + 1
		return tok(...)
	end
	lgml.sampler({ grammar = "root ::= \"a\"", tokenizer = tk })
	lgml.sampler({ grammar = "root ::= \"b\"", tokenizer = tk })
	ok(calls == 2, "vocab cached per tokenizer")
end

-- Errors
ok(not pcall(lgml.sampler, { grammar = "root ::= \"a\"" }), "grammar needs a tokenizer")
ok(not pcall(lgml.sampler, { grammar = "root ::= x", tokenizer = vocab({ "a" }) }),
    "bad grammar raises")
do
	local s = lgml.sampler({ grammar = "root ::= \"q\"", tokenizer = vocab({ "a", "b" }) })
	local x <close> = be:tensor("f32", { 1, 2 }, { 1, 2 })
	local good, err = pcall(s.sample, s, x)
	ok(not good and tostring(err):find("no token fits"), "no token fits is an error")
	ok(not pcall(lgml.sampler().allowed, lgml.sampler()), "allowed needs a grammar")
end

-- lgml.grammar: requests and the reasoning wrapper.
do
	local G = require("lgml.grammar")
	ok(G.from_request({}) == nil, "no grammar in a plain request")
	ok(G.from_request({ grammar = "root ::= \"a\"" }) == "root ::= \"a\"", "raw grammar")
	ok(G.from_request({ response_format = { type = "text" } }) == nil, "text format")
	local obj = G.from_request({ response_format = { type = "json_object" } })
	ok(obj and G.check(obj), "json_object grammar")
	local js = G.from_request({ response_format = { type = "json_schema",
	    json_schema = { schema = { type = "integer" } } } })
	ok(js and js:find("root ::=") and G.check(js), "json_schema grammar")
	local _, err, param = G.from_request({ grammar = "x", json_schema = { type = "integer" } })
	ok(err and param == "grammar", "grammar and schema together")
	_, err, param = G.from_request({ response_format = { type = "xml" } })
	ok(err and param == "response_format", "unknown format")
	_, err = G.from_request({ response_format = { type = "json_schema", json_schema = {} } })
	ok(err, "json_schema without schema")
	_, err = G.from_request({ response_format = { type = "json_schema",
	    json_schema = { schema = { type = "nope" } } } })
	ok(err and err:find("Unrecognized"), "bad schema")
	ok(not G.check("root ::= x"), "check finds errors")
	ok(G.check("root ::= <think> \"a\""), "check skips token names")

	local function wrapped(s, opened, close)
		local text, root = G.reasoning('root ::= "{" [a-z]* "}"', "<think>", close or "</think>", opened)
		local g = assert(lgml.grammar(text, { root = root }))
		return g:accept(s) and g:can_end()
	end
	ok(wrapped("{abc}"), "reasoning: answer alone")
	ok(wrapped("<think>\nhmm </thi <</think>\n\n{x}"), "reasoning: block, then answer")
	ok(not wrapped("<think>a</think> x </think>{a}"), "reasoning: the first close ends it")
	ok(not wrapped("<think>{a}"), "reasoning: open block cannot end")
	ok(wrapped("x < </think>{a}", true), "reasoning: opened in the prompt")
	ok(not wrapped("{a}", true), "reasoning: an opened block must close first")
	ok(wrapped("<think>aaab{a}", false, "aab"), "reasoning: overlapping close tag")
	ok(not wrapped("<think>aaba{a}", false, "aab"), "reasoning: text after an overlapping close")
end

-- The OpenAI request check turns formats into GBNF.
do
	local openai = require("lgml.server.openai")
	local json = require("lgml.json")
	local raw = [[{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema",]] ..
	    [["json_schema":{"name":"x","schema":{"type":"object","properties":{"zed":{"type":"integer"},]] ..
	    [["alpha":{"type":"string"}},"required":["zed","alpha"]}}}}]]
	local b = json.decode(raw)
	ok(openai.check_chat(b, raw) and b.response_format == nil and type(b.grammar) == "string",
	    "check_chat: schema becomes a grammar")
	local zi, ai = b.grammar:find("zed%-kv"), b.grammar:find("alpha%-kv")
	local root = b.grammar:match("root ::= ([^\n]*)")
	ok(root and root:find("zed%-kv") < root:find("alpha%-kv") and zi and ai,
	    "check_chat: properties keep their order")
	local _, err, param = openai.check_chat({ messages = json.array({ { role = "user", content = "x" } }),
	    grammar = "root ::= undefined-rule" })
	ok(err and err:find("undefined rule") and param == "grammar", "check_chat: bad grammar")
	_, err, param = openai.check_chat({ messages = json.array({ { role = "user", content = "x" } }),
	    grammar = "root ::= \"a\"", response_format = { type = "json_object" } })
	ok(err and param == "grammar", "check_chat: grammar and schema together")
	_, err, param = openai.check_chat({ messages = json.array({ { role = "user", content = "x" } }),
	    response_format = { type = "json_schema", json_schema = { schema = { type = "nope" } } } })
	ok(err and param == "response_format", "check_chat: bad schema")
	local t = { messages = json.array({ { role = "user", content = "x" } }), response_format = { type = "text" } }
	ok(openai.check_chat(t) and t.grammar == nil, "check_chat: text format")
end

print("1.." .. n)
