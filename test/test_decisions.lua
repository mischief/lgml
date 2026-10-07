-- lgml.decisions: OpenAI Decisions requests to System One and back.
-- Output is TAP.
local json = require("lgml.json")
local decisions = require("lgml.decisions")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
	return cond
end

local function close(a, b)
	return math.abs(a - b) < 1e-9
end

-- 1. A request of all three types becomes a System One request.
local req, infos = decisions.check([[{
  "model": "gpt-6-luna",
  "input": [{"role": "user", "content": [
    {"type": "input_text", "text": "Inspect the product."},
    {"type": "input_text", "text": "It came in a box."}]}],
  "questions": [
    {"type": "predicate", "name": "damage", "instructions": "Is it damaged?"},
    {"type": "choice", "name": "dept", "instructions": "Which department?",
     "choices": [{"value": "billing", "description": "Payments."}, {"value": 7}]},
    {"type": "score", "instructions": "How severe?",
     "levels": [{"label": "Cosmetic", "description": "Looks only."}, {"label": "Blocked"}]}]}]])
ok(req ~= nil, "check: a valid request passes")
ok(req.model == "gpt-6-luna" and req.state == "Inspect the product.\n\nIt came in a box.",
    "check: model and the text parts joined")
ok(#req.questions == 3 and req.questions[1].type == "noul" and req.questions[2].type == "choice"
    and req.questions[3].type == "score", "check: predicate is noul; choice and score keep their type")
local opts = req.questions[2].options
ok(#opts == 2 and opts[1].key == "billing" and opts[1].text:find("Payments", 1, true)
    and opts[2].key == "7", "check: choice values are option keys, descriptions their text")
ok(#req.questions[3].levels == 2 and req.questions[3].levels[1]:find("Cosmetic: Looks only.", 1, true),
    "check: a level is its label and description")
ok(infos[1].name == "damage" and infos[3].name == nil and infos[2].values[2] == 7,
    "check: names and original choice values are kept")

-- 2. The reply has OpenAI's answer shapes and Jev's confidence.
local text = decisions.reply(req, infos, {
	answers = {
		q1 = { yes = 0.923 },
		q2 = { probabilities = { 0.95, 0.05 } },
		q3 = { probabilities = { 0.3, 0.7 } },
	},
	usage = { input_tokens = 42 },
}, "dec_test")
local r = json.decode(text)
ok(r.id == "dec_test" and r.object == "decision" and r.model == "gpt-6-luna"
    and r.usage.input_tokens == 42, "reply: id, object, model and usage")
local a = r.answers
ok(a[1].type == "predicate" and a[1].name == "damage" and close(a[1].probability, 0.92),
    "reply: predicate probability, rounded to 2 places")
ok(a[2].type == "choice" and a[2].choice == "billing" and a[2].probabilities[2].value == 7
    and close(a[2].confidence, 0.9), "reply: choice, values as sent, confidence (p - 1/K) / (1 - 1/K)")
ok(a[3].type == "score" and a[3].name == nil and close(a[3].score, 0.7)
    and a[3].probabilities[2].label == "Blocked" and a[3].probabilities[2].value == 1,
    "reply: score is the mean level index; levels carry index and label")

-- 3. OpenAI's published examples give its published confidences.
local function conf(kind, probs)
	local body = kind == "choice"
	    and [[{"input":"x","questions":[{"type":"choice","instructions":"?",
	          "choices":[{"value":"a"},{"value":"b"},{"value":"c"},{"value":"d"}]}]}]]
	    or [[{"input":"x","questions":[{"type":"score","instructions":"?",
	          "levels":[{"label":"a"},{"label":"b"},{"label":"c"}]}]}]]
	local q, i = decisions.check(body)
	return json.decode(decisions.reply(q, i, { answers = { q1 = { probabilities = probs } } }, "d"))
	    .answers[1]
end
ok(close(conf("choice", { 0.95, 0.02, 0.01, 0.02 }).confidence, 0.9333),
    "example: choice billing 0.95 gives confidence 0.93")
local s = conf("score", { 0.1, 0.7, 0.2 })
ok(close(s.score, 1.1) and close(s.confidence, 0.55), "example: score 1.1 with confidence 0.55")

-- 4. Errors name the OpenAI parameter.
local function err(body)
	local q, msg, status, param = decisions.check(body)
	return q == nil and status, param, msg
end
local function bad(body, want, name)
	local status, param = err(body)
	ok(status == 400 and param == want, name .. " (" .. tostring(param) .. ")")
end
bad([[{"questions":[{"type":"predicate","instructions":"?"}]}]], "input", "error: missing input")
bad([[{"input":[{"role":"user","content":[{"type":"input_image","image_url":"https://x/a.png"}]}],
       "questions":[{"type":"predicate","instructions":"?"}]}]],
    "input[0].content[0].image_url", "error: a hosted image URL")
bad([[{"input":"x","questions":[{"type":"rank","instructions":"?"}]}]], "questions[0].type",
    "error: an unknown question type")
bad([[{"input":"x","questions":[{"type":"predicate","name":"a","instructions":"?"},
       {"type":"predicate","name":"a","instructions":"?"}]}]], "questions[1].name", "error: a repeated name")
bad([[{"input":"x","questions":[{"type":"predicate","instructions":"?"},
       {"type":"choice","instructions":"?","choices":[{"value":"a"}]}]}]], "questions[1].choices",
    "error: one choice, from the System One check")
bad([[{"input":"x","temperature":0,"questions":[{"type":"predicate","instructions":"?"}]}]],
    "temperature", "error: an unknown parameter")

print("1.." .. n)
os.exit(failed and 1 or 0)
