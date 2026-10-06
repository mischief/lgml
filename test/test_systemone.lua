-- Tests for /v1/systemone: the wire checks, the Jevify prompt and
-- recipe, and the route on the fake engine. TAP output. With
-- LGML_SYSTEMONE_MODEL set to a chat gguf, it also asks the model
-- (backend LGML_SYSTEMONE_BACKEND, default the best one).

local uv = require("luv")
local json = require("lgml.json")
local so = require("lgml.systemone")
local decide = require("lgml.decide")
local server = require("lgml.server.httpd")
local openai = require("lgml.server.openai")
local router = require("lgml.server.router")
local fake = require("lgml.server.fake_engine")
local stream = require("lgml.server.stream")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function near(a, b, eps)
	return math.abs(a - b) <= (eps or 1e-9)
end

-- The smoke questions of the task, in Jev's wire format.
local CAPITAL = [[{"type": "choice",
 "instructions": "Which of these is the capital of France?",
 "criteria": {"Paris": null, "Berlin": null, "Madrid": null, "Rome": null}}]]
local NOUL = [[{"type": "noul", "instructions": "Is the review positive?"}]]
local SCORE = [[{"type": "score", "instructions": "How positive is the review?",
 "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]}]]

local function body(state, questions)
	local t = {}
	for _, q in ipairs(questions) do
		t[#t + 1] = json.encode_string(q[1]) .. ": " .. q[2]
	end
	return '{"state": ' .. json.encode_string(state) .. ', "questions": {' ..
	    table.concat(t, ", ") .. "}}"
end

-- Python random and shuffle

local function test_random()
	ok(decide.py_random(42) == 0.6394267984578837, "random.Random(42).random()")
	ok(table.concat(decide.shuffle(10, 42), ",") == "7,3,2,8,5,6,9,4,0,1",
	    "random.Random(42).shuffle(range(10))")
end

-- wire

local function test_json()
	local v = assert(so.decode('{"b": 1, "a": [1, 2.0, {}, []], "c": "x\\u00e9\\ud83d\\ude00\\n", "d": null}'))
	ok(table.concat(getmetatable(v).__order, ",") == "b,a,c,d", "decode keeps key order")
	ok(math.type(v.a[1]) == "integer" and math.type(v.a[2]) == "float", "decode keeps int and float")
	ok(v.c == "xé😀\n" and v.d == json.null, "decode escapes and null")
	ok(so.dumps(v) == '{"b": 1, "a": [1, 2.0, {}, []], "c": "xé😀\\n", "d": null}',
	    "dumps as Python json.dumps")
	ok(so.dumps(v, 2) == '{\n  "b": 1,\n  "a": [\n    1,\n    2.0,\n    {},\n    []\n  ],\n' ..
	    '  "c": "xé😀\\n",\n  "d": null\n}', "dumps with indent 2")
	ok(so.describe(json.null) == "" and so.describe("t") == "t" and
	    so.describe(assert(so.decode('{"k": [1]}'))) == '{"k": [1]}', "describe")
	local dup = assert(so.decode('{"a": 1, "b": 2, "a": 3}'))
	ok(dup.a == 3 and getmetatable(dup).__dups == "a" and #getmetatable(dup).__order == 2,
	    "repeated key: first place, last value")
	ok(so.decode('{"a": 1,}') == nil and so.decode("[01]") == nil and so.decode("") == nil,
	    "decode rejects bad JSON")
end

local function check_fails(text, loc, name)
	local r, e = so.check(text)
	local d = e and e.detail and e.detail[1]
	ok(r == nil and d and table.concat(d.loc, ".") == loc, name .. " (" .. (d and d.msg or "passed") .. ")")
end

local function test_check()
	local r, b = so.check(body("Paris is the capital of France.", {
		{ "capital", CAPITAL }, { "positive", NOUL }, { "rating", SCORE } }))
	ok(r and r.state == "Paris is the capital of France." and #r.questions == 3, "check accepts")
	ok(r.questions[1].id == "capital" and r.questions[1].options[4].key == "Rome" and
	    r.questions[1].options[1].text == "", "choice options in order")
	ok(r.questions[3].levels[5] == "Very positive" and r.model == nil, "score levels")
	ok(b.questions.rating.criteria[1] == "Very negative", "decoded body")
	r = assert(so.check('{"state": {"z": 1, "a": "b"}, "model": "m", ' ..
	    '"questions": {"q": {"type": "noul", "criteria": {"true": "yes!", "false": ""}}}}'))
	ok(r.state == '{\n  "z": 1,\n  "a": "b"\n}' and r.model == "m", "JSON state as indented text")
	ok(r.questions[1].yes == "yes!" and r.questions[1].no == nil, "noul criteria, empty is absent")

	check_fails("{bad", "body", "bad JSON")
	check_fails('{"questions": {"q": ' .. NOUL .. '}}', "body.state", "state required")
	check_fails('{"state": 5, "questions": {"q": ' .. NOUL .. '}}', "body.state", "state type")
	check_fails('{"state": "s", "questions": {}}', "body.questions", "no questions")
	check_fails('{"state": "s", "questions": {"q": ' .. NOUL .. '}, "x": 1}', "body.x", "extra field")
	check_fails('{"state": "s", "questions": {"q": ' .. NOUL .. ', "q": ' .. NOUL .. '}}',
	    "body.questions", "repeated question id")
	check_fails('{"state": "s", "questions": {"q": {"type": "maybe"}}}', "body.questions.q.type",
	    "unknown type")
	check_fails('{"state": "s", "questions": {"q": {"criteria": {}}}}', "body.questions.q.type",
	    "missing type")
	check_fails('{"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": null}}}}',
	    "body.questions.q.criteria", "one option")
	local many = {}
	for i = 1, 256 do
		many[i] = ('"o%d": null'):format(i)
	end
	check_fails('{"state": "s", "questions": {"q": {"type": "choice", "criteria": {' ..
	    table.concat(many, ", ") .. '}}}}', "body.questions.q.criteria", "256 options")
	ok(so.check('{"state": "s", "questions": {"q": {"type": "choice", "criteria": {' ..
	    table.concat(many, ", ", 1, 255) .. '}}}}') ~= nil, "255 options pass")
	check_fails('{"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": null, "a": "x"}}}}',
	    "body.questions.q.criteria", "repeated option")
	check_fails('{"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": null, "": null}}}}',
	    "body.questions.q.criteria", "empty option name")
	check_fails('{"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": 1, "b": null}}}}',
	    "body.questions.q.criteria.a", "number option text")
	check_fails('{"state": "s", "questions": {"q": {"type": "score", "criteria": ["a"]}}}',
	    "body.questions.q.criteria", "one level")
	local levels = {}
	for i = 1, 27 do
		levels[i] = ('"%d"'):format(i - 1)
	end
	check_fails('{"state": "s", "questions": {"q": {"type": "score", "criteria": [' ..
	    table.concat(levels, ",") .. ']}}}', "body.questions.q.criteria", "27 levels")
	ok(so.check('{"state": "s", "questions": {"q": {"type": "score", "criteria": [' ..
	    table.concat(levels, ",", 1, 26) .. ']}}}') ~= nil, "26 levels pass")
	local qs = {}
	for i = 1, 65 do
		qs[i] = ('"q%d": %s'):format(i, NOUL)
	end
	check_fails('{"state": "s", "questions": {' .. table.concat(qs, ", ") .. '}}', "body.questions",
	    "65 questions")
	ok(so.check('{"state": "s", "keep_alive": "5m", "questions": {"q": ' .. NOUL .. '}}') ~= nil,
	    "keep_alive passes")
	check_fails('{"state": "s", "keep_alive": true, "questions": {"q": ' .. NOUL .. '}}',
	    "body.keep_alive", "keep_alive type")
	check_fails('{"state": "s", "questions": {"q": {"type": "score", "criteria": ["a", null]}}}',
	    "body.questions.q.criteria.1", "null level")
	check_fails('{"state": "s", "questions": {"q": {"type": "noul", "criteria": {"maybe": "x"}}}}',
	    "body.questions.q.criteria.maybe", "noul criteria field")
	check_fails('{"state": "s", "questions": {"q": {"type": "noul", "instructions": 3}}}',
	    "body.questions.q.instructions", "instructions type")
	check_fails('{"state": "s", "questions": {"q": ' .. NOUL .. '}, "lgml": {"raw": 1}}',
	    "body.lgml.raw", "lgml.raw type")
end

-- A 1x1 PNG as a data URL, and a JPEG header.
local PNG64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
local PNG_URL = "data:image/png;base64," .. PNG64
local JPEG_URL = "data:image/jpeg;base64," .. require("lgml.base64").encode("\255\216\255\224 jfif")

local function with_images(list)
	return '{"state": "s", "questions": {"q": ' .. NOUL .. '}, "attachments": ' .. list .. "}"
end

local function test_attachments()
	local r = assert(so.check(body("s", { { "q", NOUL } })))
	ok(r.images == nil, "no attachments, no images")
	r = assert(so.check(with_images("[]")))
	ok(r.images == nil, "an empty list gives no images")
	r = assert(so.check(with_images("null")))
	ok(r.images == nil, "null attachments give no images")
	r = assert(so.check(with_images(("[%q, %q]"):format(PNG_URL, JPEG_URL))))
	ok(#r.images == 2 and r.images[1].mime == "image/png" and r.images[1].bytes:sub(2, 4) == "PNG" and
	    r.images[2].mime == "image/jpeg", "data URL strings, in order")
	r = assert(so.check(with_images(('[{"type": "image_url", "image_url": {"url": %q}}, ' ..
	    '{"type": "image_url", "image_url": %q}]'):format(JPEG_URL, PNG_URL))))
	ok(#r.images == 2 and r.images[1].mime == "image/jpeg" and r.images[2].mime == "image/png",
	    "OpenAI image_url parts")
	local _, e, status = so.check(with_images(("[%q]"):format("https://example.com/a.png")))
	ok(status == 400 and table.concat(e.detail[1].loc, ".") == "body.attachments.0" and
	    e.detail[1].msg:find("data URL", 1, true), "an https URL gives 400")
	_, e, status = so.check(with_images(('[{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]')))
	ok(status == 400, "an http URL in a part gives 400")
	-- OpenJev forms: an images field, and image_url parts in a state of
	-- chat messages. Order: attachments, images, then the state.
	r = assert(so.check(('{"state": "s", "questions": {"q": %s}, "images": [%q]}'):format(NOUL, JPEG_URL)))
	ok(#r.images == 1 and r.images[1].mime == "image/jpeg", "the images field")
	-- Clef and Jev send bare base64.
	r = assert(so.check(('{"state": "s", "questions": {"q": %s}, "images": [%q, %q]}'):format(NOUL, PNG64,
	    JPEG_URL:match(",(.*)$"))))
	ok(#r.images == 2 and r.images[1].mime == "image/png" and r.images[2].mime == "image/jpeg" and
	    r.images[2].bytes:sub(1, 3) == "\255\216\255", "bare base64 images")
	check_fails(('{"state": "s", "questions": {"q": %s}, "images": ["aGVsbG8="]}'):format(NOUL),
	    "body.images.0", "bare base64 that is not an image")
	r = assert(so.check(('{"state": {"messages": [{"role": "user", "content": [' ..
	    '{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": %q}}]}]}, ' ..
	    '"questions": {"q": %s}, "attachments": [%q], "images": [%q]}'):format(PNG_URL, NOUL, JPEG_URL, JPEG_URL)))
	ok(#r.images == 3 and r.images[1].mime == "image/jpeg" and r.images[3].mime == "image/png" and
	    r.state:find('"look"', 1, true) and not r.state:find("image_url", 1, true) and
	    r.state_compact == '{"messages":[{"role":"user","content":[{"type":"text","text":"look"}]}]}',
	    "image parts of a chat state go to the images, last")
	r = assert(so.check(('{"state": [{"role": "user", "content": [{"type": "image_url", "image_url": %q}]}], ' ..
	    '"questions": {"q": %s}}'):format(PNG_URL, NOUL)))
	ok(#r.images == 1 and r.state_compact == '[{"role":"user","content":[]}]', "a state that is an array of messages")
	local many = {}
	for i = 1, so.MAX_ATTACHMENTS do
		many[i] = ("%q"):format(PNG_URL)
	end
	check_fails(('{"state": "s", "questions": {"q": %s}, "attachments": [%s], "images": [%q]}'):format(
	    NOUL, table.concat(many, ","), PNG_URL), "body.attachments", "the image limit counts every source")
	check_fails(with_images("{}"), "body.attachments", "attachments must be an array")
	check_fails(with_images("[5]"), "body.attachments.0", "a number is not an attachment")
	check_fails(with_images('[{"type": "text", "text": "x"}]'), "body.attachments.0.type", "a text part")
	check_fails(with_images('[{"type": "image_url", "image_url": {}}]'),
	    "body.attachments.0.image_url.url", "a part without url")
	check_fails(with_images(("[%q, %q]"):format(PNG_URL, "data:text/plain;base64,aGk=")),
	    "body.attachments.1", "a text data URL")
	check_fails(with_images(("[%q]"):format("data:image/gif;base64,R0lGODlh")),
	    "body.attachments.0", "a GIF")
	check_fails(with_images(("[%q]"):format("data:image/png;base64,aGVsbG8=")),
	    "body.attachments.0", "bytes that are not PNG")
	check_fails(with_images(("[%q]"):format("data:image/png;base64,***")),
	    "body.attachments.0", "bad base64")
	check_fails(with_images(("[%q]"):format("/tmp/a.png")), "body.attachments.0", "a path")
	local many = {}
	for i = 1, so.MAX_ATTACHMENTS + 1 do
		many[i] = ("%q"):format(PNG_URL)
	end
	check_fails(with_images("[" .. table.concat(many, ", ") .. "]"), "body.attachments", "too many images")
	local huge = "data:image/png;base64," .. string.rep("A", so.MAX_ATTACHMENT_BYTES // 3 * 4 + 12)
	check_fails(with_images(("[%q]"):format(huge)), "body.attachments.0", "an image too large")
end

local function test_stats()
	local r = so.round({ 0.333333, 0.333333, 0.333334 })
	ok(r[1] == 0.33 and r[2] == 0.33 and r[3] == 0.34, "largest remainder rounding")
	r = so.round({ 0.004, 0.004, 0.992 })
	ok(r[1] + r[2] + r[3] == 1.0 and r[3] == 0.99, "rounding sums to 1")
	r = so.round({ 0.12345, 0.87655 }, 4)
	ok(r[1] == 0.1235 or r[1] == 0.1234, "rounding to 4 places")
	ok(near(so.choice_confidence({ 0.7, 0.1, 0.1, 0.1 }), 0.6), "choice confidence")
	ok(so.choice_confidence({ 0.25, 0.25, 0.25, 0.25 }) == 0, "uniform choice confidence 0")
	ok(so.score_confidence({ 0, 1, 0 }) == 1 and near(so.score_confidence({ 0.5, 0, 0.5 }), 0),
	    "score confidence")
	-- TypeSafe's rule: mean distance to the mode 0.3, uniform 2/3.
	ok(near(so.score_confidence({ 0.1, 0.7, 0.2 }), 0.55, 1e-12), "score confidence by distance to the mode")
	ok(near(so.score_expectation({ 0.1, 0.2, 0.7 }), 1.6), "score expectation")
end

-- prompt and recipe

-- The nimble format on Ollama's decision guide example: the user turn
-- holds the context and schema as Go writes them, then the field.
local function test_nimble()
	local r = assert(so.check([[{"model": "nimble",
	 "state": {"ticket": "I was charged twice. Please refund the extra payment."},
	 "questions": {
	  "refund": {"type": "noul", "instructions": "Is the customer requesting a refund?",
	   "criteria": {"false": "No refund is requested", "true": "The customer requests a refund"}},
	  "urgency": {"type": "score", "instructions": "How urgently does this ticket need a response?",
	   "criteria": ["Routine: no time pressure", "Soon: a customer is inconvenienced",
	    "Immediate: a critical service is unavailable"]}}}]]))
	local family = { render = function(msgs, opts)
		return msgs[1].content .. "|" .. msgs[2].content .. "|" .. tostring(opts.enable_thinking)
	end }
	local rds = decide.render_nimble(family, r.state_compact, r.questions)
	local data = '{"context":"{\\"ticket\\":\\"I was charged twice. Please refund the extra payment.\\"}",' ..
	    '"schema":[{"name":"refund","description":"Is the customer requesting a refund?","choices":[' ..
	    '{"code":"A","value":false,"description":"No refund is requested"},' ..
	    '{"code":"B","value":true,"description":"The customer requests a refund"}]},' ..
	    '{"name":"urgency","description":"How urgently does this ticket need a response?","choices":[' ..
	    '{"code":"A","value":"0","description":"Routine: no time pressure"},' ..
	    '{"code":"B","value":"1","description":"Soon: a customer is inconvenienced"},' ..
	    '{"code":"C","value":"2","description":"Immediate: a critical service is unavailable"}]}]}'
	ok(rds[1].text == decide.NIMBLE_SYSTEM_PROMPT .. "|" .. data .. '\n\nRequested field: "refund"|false',
	    "nimble: the prompt of the first field")
	ok(rds[2].text:find('Requested field: "urgency"|', 1, true) ~= nil and
	    table.concat(rds[2].candidates) == "ABC" and table.concat(rds[1].keys, ",") == "0,1",
	    "nimble: codes and keys")
	local esc = decide.render_nimble(family, "a<b & \"c\"\n", { r.questions[1] })[1].text
	ok(esc:find('"context":"a\\u003cb \\u0026 \\"c\\"\\n"', 1, true) ~= nil, "nimble: Go escapes")
	ok(near(so.entropy_confidence({ 0.0125, 0.9781, 0.0093 }), 0.8906, 5e-4),
	    "nimble: entropy confidence of the docs example")
	local out = json.decode(so.reply(r, json.decode([[{"questions": {
	  "refund": {}, "urgency": {"criteria": ["a", "b", "c"]}}}]]), {
		format = "nimble", answers = {
			refund = { yes = 0.99891234 },
			urgency = { probabilities = { 0.25, 0.5, 0.25 } },
		} }))
	ok(out.answers.refund.noul == 0.99891234, "nimble: noul not rounded")
	ok(near(out.answers.urgency.score, 1) and out.answers.urgency.legend["1"] == "b" and
	    near(out.answers.urgency.confidence, 1 - 1.0397207708 / math.log(3), 1e-6), "nimble: score answer")
end

local function test_render()
	local r = assert(so.check(body("Paris is nice.", { { "q", [[{"type": "choice",
	 "instructions": "  Which of these is the capital of France?\n",
	 "criteria": {"Paris": "Paris", "Berlin": null, "Madrid": "Spain's capital", "Rome": {"country": "Italy"}}}]] } })))
	local rd = decide.render(r.state, r.questions[1], { identifiers = { "A", "B", "C", "D", "E" } })
	ok(rd.prefix == "State:\nParis is nice.\n\nQuestion: Which of these is the capital of France?\n" ..
	    "Options:\nA. Paris\nB. Berlin\nC. Madrid — Spain's capital\nD. Rome — {\"country\": \"Italy\"}\n" ..
	    "Allowed answers: A, B, C, D\nAnswer: ", "choice prompt")
	ok(table.concat(rd.candidates, ",") == "A,B,C,D" and table.concat(rd.keys, ",") ==
	    "Paris,Berlin,Madrid,Rome", "choice candidates and keys")
	local order = decide.shuffle(4, 1001)
	rd = decide.render(r.state, r.questions[1], { identifiers = { "A", "B", "C", "D" }, seed = 1001 })
	ok(rd.keys[1] == r.questions[1].options[order[1] + 1].key and
	    rd.prefix:find("\nA. " .. rd.keys[1], 1, true), "shuffled order")
	rd = decide.render(r.state, r.questions[1], { identifiers = { "A" } })
	ok(rd.candidates[4] == "D", "too few identifiers give letters")

	r = assert(so.check(body("Great movie.", { { "s", SCORE },
	    { "n", [[{"type": "noul", "criteria": {"true": "it praises", "false": {"k": "pans"}}}]] } })))
	rd = decide.render(r.state, r.questions[1])
	ok(rd.prefix == "State:\nGreat movie.\n\nQuestion: How positive is the review?\n" ..
	    "Levels (ordered from lowest to highest):\n0. Very negative\n1. Negative\n2. Neutral\n" ..
	    "3. Positive\n4. Very positive\nAllowed answers: 0, 1, 2, 3, 4\nAnswer: ", "score prompt")
	rd = decide.render(r.state, r.questions[2])
	ok(rd.prefix == "State:\nGreat movie.\n\nQuestion: Is the statement true?\n" ..
	    "Answer yes if: it praises\nAnswer no if: {\"k\": \"pans\"}\nAllowed answers: yes, no\nAnswer: ",
	    "noul prompt")
	ok(table.concat(rd.keys, ",") == "1,0" and rd.candidates[1] == "yes", "noul keys")
	rd = decide.render(r.state, r.questions[2], { content_free = true })
	ok(rd.prefix:find("^State:\nN/A\n\n"), "content-free state")

	local long = string.rep("é", 20000)
	ok(decide.render_state(long) == long, "a long state is not cut")
	ok(decide.strip("\u{3000} a b\t\n") == "a b", "strip")

	local family = { render = function(msgs, o)
		return ("<%s|%s|%s|%s>"):format(msgs[1].content, msgs[2].content,
		    tostring(o.add_generation_prompt), tostring(o.enable_thinking))
	end }
	ok(decide.chat_prompt(family, "State:\nx\n\nQuestion: q\nAnswer: ") ==
	    "<" .. decide.SYSTEM_PROMPT .. "|State:\nx\n\nQuestion: q|true|false>Answer: ", "chat prompt")
end

-- A tokenizer with one token per byte, except that a space merges with
-- the letters after it and "\n\n" is one token, as in BPE.
local function toy_tokenizer()
	local ids, names = {}, {}
	local function id(s)
		if not ids[s] then
			names[#names + 1] = s
			ids[s] = #names - 1
		end
		return ids[s]
	end
	local tk = {}
	function tk.encode(_, s)
		local out, i = {}, 1
		while i <= #s do
			local piece = s:match("^ %a+", i) or s:match("^\n\n", i) or s:sub(i, i)
			out[#out + 1] = id(piece)
			i = i + #piece
		end
		return out
	end
	function tk.name(i)
		return names[i + 1]
	end
	return tk
end

local function test_tokenize()
	local tk = toy_tokenizer()
	local text = "sys\n\nAnswer: "
	local p, c = decide.tokenize(tk, text, { "A", "BC" })
	local pt = tk:encode(text)
	ok(#p == #pt - 1 and #c[1] == 1 and tk.name(c[1][1]) == " A" and tk.name(c[2][1]) == " BC",
	    "letters take the cue space")
	p, c = decide.tokenize(tk, text, { "0", "12" })
	ok(#p == #pt and tk.name(c[1][1]) == "0" and #c[2] == 2, "digits follow the whole prompt")
	ok(#decide.identifiers(tk) == 255 and decide.identifiers(tk)[27] == "AA", "identifiers")
end

local function test_finalize()
	local rc = decide.default_recipe()
	local q = { type = "choice", options = { { key = "a" }, { key = "b" } } }
	local runs = {
		{ keys = { "a", "b" }, logscores = { math.log(0.8), math.log(0.2) } },
		{ keys = { "b", "a" }, logscores = { math.log(0.4), math.log(0.6) } },
	}
	local a = decide.finalize(q, runs, nil, rc)
	ok(near(a.probabilities[1], 0.7) and near(a.probabilities[2], 0.3), "two orders average")
	rc = assert(decide.recipe(rc, { permutations = { choice = 1 }, temperature = { choice = 2 } }))
	a = decide.finalize(q, runs, nil, rc)
	ok(near(a.probabilities[1], 2 / 3), "temperature on one order")
	local nq = { type = "noul" }
	local nr = { { keys = { "1", "0" }, logscores = { 0.0, 0.0 } } }
	rc = assert(decide.recipe(decide.default_recipe(), { recipe = { bias = { noul = math.log(3) } } }))
	ok(near(decide.finalize(nq, nr, nil, rc).yes, 0.75), "noul bias (recipe.json form)")
	rc = assert(decide.recipe(decide.default_recipe(), { prior_weight = 1 }))
	local prior = { keys = { "1", "0" }, logscores = { math.log(0.75), math.log(0.25) } }
	ok(near(decide.finalize(nq, nr, prior, rc).yes, 0.25), "prior correction")
	ok(decide.recipe(rc, { temperature = { choice = 0 } }) == nil and
	    decide.recipe(rc, { mode = "label" }) == nil, "bad recipes fail")

	-- Fitted temperatures from the gguf: by type and option count
	-- bucket; a temperature in the recipe wins.
	local fit = decide.model_temperatures({ ["general.architecture"] = "x",
		["x.decision.temperature.choice.2"] = 2.0, ["x.decision.temperature.choice"] = 4.0,
		["x.decision.temperature.noul"] = "0.5", ["x.other"] = 9 })
	ok(fit and fit["choice.2"] == 2 and fit.choice == 4 and fit.noul == 0.5, "model temperatures")
	rc = assert(decide.recipe(decide.default_recipe(), { permutations = { choice = 1 } }))
	rc.fitted = fit
	ok(near(decide.finalize(q, runs, nil, rc).probabilities[1], 2 / 3), "fitted temperature of the bucket")
	local q3 = { type = "choice", options = { { key = "a" }, { key = "b" }, { key = "c" } } }
	local r3 = { { keys = { "a", "b", "c" }, logscores = { math.log(0.8), math.log(0.1), math.log(0.1) } } }
	local p = decide.finalize(q3, r3, nil, rc).probabilities
	ok(near(p[1], 0.8 ^ 0.25 / (0.8 ^ 0.25 + 2 * 0.1 ^ 0.25)), "fitted temperature of the type")
	rc = assert(decide.recipe(rc, { temperature = { choice = 1 } }))
	rc.fitted = fit
	ok(near(decide.finalize(q, runs, nil, rc).probabilities[1], 0.8), "a recipe temperature wins")
end

-- HTTP

local function post(port, path, text)
	local tcp = uv.new_tcp()
	assert(not stream.await(function(done)
		tcp:connect("127.0.0.1", port, done)
	end))
	local s = stream.new(tcp)
	s:write(("POST %s HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Length: %d\r\n\r\n%s")
	    :format(path, #text, text))
	local buf = ""
	while true do
		local d = s:read(5000)
		if not d then
			break
		end
		buf = buf .. d
	end
	s:close()
	local status = tonumber(buf:match("^HTTP/1%.1 (%d+)"))
	return status, buf:match("\r\n\r\n(.*)$"), buf
end

local function sum(t)
	local s = 0
	for _, v in pairs(t) do
		s = s + v
	end
	return s
end

local function test_http(port, bare_port)
	local text = body("Paris is the capital of France. The review was glowing and positive.", {
		{ "capital", CAPITAL }, { "positive", NOUL }, { "rating", SCORE } })
	local status, rb, raw = post(port, "/v1/systemone", text)
	local b = json.decode(rb or "")
	ok(status == 200 and b and b.model == "fake", "systemone answers")
	local c = b.answers.capital
	ok(c.type == "choice" and c.choice == "Paris" and c.probabilities.Paris > 0.9, "Paris wins")
	ok(near(sum(c.probabilities), 1, 1e-9) and near(c.confidence,
	    (c.probabilities.Paris - 0.25) / 0.75, 1e-4), "choice sums to 1, Jev confidence")
	ok(rb:find('"probabilities": {"Paris": [%d.]+, "Berlin": [%d.]+, "Madrid"'), "options in request order")
	ok(rb:find('^{"model": "fake", "answers": {"capital": {"type": "choice", "choice": "Paris", "confidence"'),
	    "field order as Jev")
	local p = b.answers.positive
	ok(p.type == "noul" and p.noul > 0.5 and p.confidence == nil, "noul is P(yes) only")
	local s = b.answers.rating
	ok(s.type == "score" and s.legend["3"] == "Positive" and near(sum(s.probabilities), 1, 1e-9) and
	    s.score > 2 and s.confidence >= 0 and s.confidence <= 1, "score with legend")
	ok(math.type(b.usage.input_tokens) == "integer" and b.usage.output_tokens == 11, "usage")
	ok(raw:find("x%-jevify%-latency%-ms: [%d.]+\r\n"), "latency header")
	io.stderr:write(rb, "\n")

	status, rb = post(port, "/systemone", body("x", { { "q", NOUL } }))
	ok(status == 200 and json.decode(rb).answers.q.noul, "route without /v1")
	status, rb = post(port, "/v1/systemone", '{"state": "s", "questions": {"q": {"type": "score", "criteria": ["a"]}}}')
	b = json.decode(rb)
	ok(status == 422 and b.detail[1].loc[3] == "q" and b.detail[1].type == "value_error", "bad question gives 422")
	status, rb = post(port, "/v1/systemone", "{nope")
	ok(status == 422 and json.decode(rb).detail[1].type == "json_invalid", "bad JSON gives 422")
	status, rb = post(port, "/v1/systemone", body("fail", { { "q", NOUL } }))
	ok(status == 500 and json.decode(rb).detail == "fake failure", "engine error")
	status, rb = post(port, "/v1/systemone", '{"state": "x", "questions": {"q": ' .. NOUL ..
	    '}, "lgml": {"recipe": {"mode": "label"}}}')
	b = json.decode(rb)
	local d = b and type(b.detail) == "table" and b.detail[1]
	ok(status == 422 and d and d.loc[2] == "lgml" and d.loc[3] == "recipe" and
	    d.msg == "recipe mode must be index" and d.type == "value_error", "bad recipe gives a FastAPI detail list")
	status, rb, raw = post(port, "/v1/systemone", '{"state": "x", "questions": {"q": ' .. NOUL ..
	    '}, "attachments": ["' .. PNG_URL .. '", "' .. JPEG_URL .. '"]}')
	ok(status == 200 and raw:find("x%-lgml%-images: 2\r\n"), "attachments reach the engine; images header")
	status, rb = post(port, "/v1/systemone", '{"state": "x", "questions": {"q": ' .. NOUL ..
	    '}, "attachments": ["https://example.com/a.png"]}')
	ok(status == 400 and json.decode(rb).detail[1].loc[2] == "attachments", "http image URL gives 400")
	status = post(bare_port, "/v1/systemone", body("x", { { "q", NOUL } }))
	ok(status == 404, "engine without decide gives 404")
end

local function test_router()
	local a = fake.new({ model = "a", delay = 0 })
	local b = fake.new({ model = "b", delay = 0 })
	local chat_only = { chat = function() return true end }
	function chat_only.models() return { { id = "c", context_length = 1 } } end
	local r = router.new({ chat_only, a, b })
	ok(r:pick("b", "decide") == b and r:pick("jevify-latest", "decide") == a and
	    r:pick(nil, "decide") == a, "router picks a decide engine")
	local req = assert(so.check(body("x", { { "q", NOUL } })))
	req.model = "b"
	ok(r:decide(req).model == "b", "router decides by model")
end

-- A real model, when LGML_SYSTEMONE_MODEL names one.
local function test_model(path)
	local eng = require("lgml.engine").new({ model = path, backend = os.getenv("LGML_SYSTEMONE_BACKEND"),
		n_ctx = 4096 })
	local function ask(state, qs)
		local req, b = assert(so.check(body(state, qs)))
		local r, msg = eng:decide(req, {})
		assert(r, msg)
		local text = so.reply(req, b, r)
		io.stderr:write(text, "\n")
		return json.decode(text), r
	end
	local capital = [[{"type": "choice", "instructions": "Which of these is the capital of France?",
	 "criteria": {"Berlin": null, "Madrid": null, "Paris": null, "Rome": null}}]]
	local b = ask("A geography quiz.", { { "capital", capital } })
	local c = b.answers.capital
	ok(c.choice == "Paris" and c.probabilities.Paris >= 0.8, "model: Paris wins")
	b = ask("The battery lasted two days on a single charge and the screen is gorgeous.",
	    { { "positive", NOUL }, { "rating", SCORE } })
	ok(b.answers.positive.noul > 0.5, "model: positive review is yes")
	ok(b.answers.rating.score > 2, "model: positive review scores high")
	ok(b.usage.input_tokens > 0 and b.usage.output_tokens == 2 + 5, "model: usage")
	local cached = #eng.tokens
	b = ask("The battery lasted two days on a single charge and the screen is gorgeous.",
	    { { "positive", NOUL } })
	ok(cached > 0 and b.answers.positive.noul > 0.5, "model: repeat request")
end

local function main()
	test_random()
	test_json()
	test_check()
	test_attachments()
	test_stats()
	test_nimble()
	test_render()
	test_tokenize()
	test_finalize()
	local eng = fake.new({ delay = 1 })
	local srv = server.new({ log = function() end })
	openai.install(srv, eng)
	local bare = setmetatable({ decide = false }, { __index = eng })
	local srv2 = server.new({ log = function() end })
	openai.install(srv2, bare)
	test_http(assert(srv:listen("127.0.0.1", 0)), assert(srv2:listen("127.0.0.1", 0)))
	srv:close()
	srv2:close()
	test_router()
	local path = os.getenv("LGML_SYSTEMONE_MODEL")
	if path and path ~= "" then
		test_model(path)
	end
	print("1.." .. n)
	uv.stop()
end

local co = coroutine.create(main)
local good, err = coroutine.resume(co)
if not good then
	error(debug.traceback(co, err))
end
uv.run()
if coroutine.status(co) ~= "dead" then
	print("not ok - main did not finish")
	failed = failed + 1
end
os.exit(failed == 0 and 0 or 1)
