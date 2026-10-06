-- usage: lua5.4 scripts/decide_head_check.lua model.gguf [backend [request.json]]
-- Answers hand-made /v1/systemone requests with a Kev or Clef gguf and
-- prints each reply and its latency. It checks that probabilities sum
-- to 1 and that the answers are sensible. With a request file, it
-- answers that request only and prints the unrounded engine answers.
-- Run it under `flock -x /tmp/lgml.lock`.
local uv = require("luv")
local lgml = require("lgml")
local json = require("lgml.json")
local so = require("lgml.systemone")
local engine = require("lgml.engine")

local path = assert(arg[1], "usage: decide_head_check.lua model.gguf [backend]")
local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

-- The request of the llama.cpp PR that added /v1/systemone.
local SUPPORT = [[{
 "state": {"message": "Hi, I was charged twice for my order #4471 and I want a refund.",
  "plan": "pro", "order": {"id": 4471, "items": ["phone case", "charger"]}},
 "questions": {
  "intent": {"type": "choice", "instructions": "What does the customer want?",
   "criteria": {"refund": "wants money back", "cancel": "wants to cancel an order",
    "track": "wants to know where an order is", "other": "anything else"}},
  "urgent": {"type": "noul", "instructions": "Does this need a human within the hour?"},
  "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
   "criteria": ["calm", "mildly annoyed", "annoyed", "angry"]},
  "refund": {"type": "noul", "instructions": "Is a refund requested?",
   "criteria": {"true": "money back is asked", "false": "no money back is asked"}},
  "team": {"type": "choice", "instructions": "Which team?",
   "criteria": {"billing": null, "shipping": null, "technical": null, "sales": null, "legal": null,
    "returns": null, "fraud": null, "accounts": null, "retention": null, "other": null}}}}]]

local CAPITAL = [[{"state": "A geography quiz.",
 "questions": {"capital": {"type": "choice", "instructions": "Which of these is the capital of France?",
  "criteria": {"Berlin": null, "Madrid": null, "Paris": null, "Rome": null}}}}]]

local REVIEW = [[{"state": "The battery lasted two days on a single charge and the screen is gorgeous.",
 "questions": {"positive": {"type": "noul", "instructions": "Is the review positive?"},
  "rating": {"type": "score", "instructions": "How positive is the review?",
   "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]}}}]]

local NEGATIVE = [[{"state": "Arrived broken, support never answered, and the refund took six weeks.",
 "questions": {"positive": {"type": "noul", "instructions": "Is the review positive?"},
  "rating": {"type": "score", "instructions": "How positive is the review?",
   "criteria": ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]}}}]]

local backend = arg[2] ~= "" and arg[2] or nil
-- CTK=f16 and UB=n in the environment set the KV cache type and ubatch.
local ctk = os.getenv("CTK")
local e = engine.new({ model = path, backend = backend, n_ctx = 8192, cache_type_k = ctk, cache_type_v = ctk,
	ubatch = tonumber(os.getenv("UB") or "") })
print("model", e.id, "recipe", e.recipe.prompt)

local function ask(text)
	local req, body = assert(so.check(text))
	req.raw = true
	local t0 = lgml.clock()
	local r, msg = e:decide(req, {})
	local ms = (lgml.clock() - t0) * 1000
	assert(r, msg)
	-- The engine probabilities before rounding sum to 1.
	for _, q in ipairs(req.questions) do
		local a = r.answers[q.id]
		if a.probabilities then
			local s = 0
			for _, p in ipairs(a.probabilities) do
				s = s + p
			end
			ok(math.abs(s - 1) < 1e-6, q.id .. ": probabilities sum to 1")
		else
			ok(a.yes >= 0 and a.yes <= 1, q.id .. ": noul is a probability")
		end
	end
	local reply = so.reply(req, body, r)
	print(("%.1f ms, %d input tokens, %d computed, %d cached"):format(ms, r.usage.input_tokens,
	    r.timings.prompt_n, r.timings.cache_n))
	print(reply)
	return json.decode(reply).answers, r
end

local function main()
	if arg[3] then
		local f = assert(io.open(arg[3]))
		local _, r = ask(f:read("a"))
		f:close()
		print("engine answers", json.encode(r.answers))
		print("1.." .. n)
		return
	end
	local a = ask(SUPPORT)
	ok(a.intent.choice == "refund", "support: intent is refund")
	ok(a.refund.noul > 0.5, "support: a refund is requested")
	-- Without descriptions the team is a close call; billing is a top one.
	ok(a.team.probabilities.billing >= 0.1, "support: billing is likely")
	a = ask(SUPPORT)
	ok(a.intent.choice == "refund", "support again (cached state)")
	a = ask(CAPITAL)
	ok(a.capital.choice == "Paris", "capital: Paris")
	a = ask(REVIEW)
	ok(a.positive.noul > 0.5 and a.rating.score > 2, "review: positive")
	a = ask(NEGATIVE)
	ok(a.positive.noul < 0.5 and a.rating.score < 2, "review: negative")
	print("1.." .. n)
end

local done
coroutine.wrap(function()
	local good, err = xpcall(main, debug.traceback)
	if not good then
		print("not ok - " .. tostring(err))
		failed = failed + 1
	end
	done = true
end)()
while not done do
	uv.run("once")
end
os.exit(failed == 0 and 0 or 1)
