--- OpenAI's Decisions API (`POST /v1/decisions`) on the System One
-- engine. `check` turns a request into a System One body and checks
-- it with `lgml.systemone.check`; `reply` turns the engine result into
-- OpenAI's answers. The question types map as predicate to noul,
-- choice to choice and score to score. The confidence rules are Jev's
-- (`lgml.systemone`); they give OpenAI's published examples exactly.
-- @module lgml.decisions
local json = require("lgml.json")
local tojson = require("lgml.chat.tojson")
local systemone = require("lgml.systemone")

local M = {}

--- The model name of a reply when the request names none.
M.DEFAULT_MODEL = "gpt-6-luna"

local Bad = {}

-- Raise a request error: status 400 with an OpenAI parameter path.
local function bad(param, msg)
	error(setmetatable({ param = param, msg = msg }, Bad), 0)
end

local function is_object(v)
	local mt = type(v) == "table" and getmetatable(v)
	return mt ~= nil and mt ~= false and mt.__jsontype == "object"
end

local function is_array(v)
	return type(v) == "table" and not is_object(v)
end

local function ordered(t, keys)
	return setmetatable(t, { __jsontype = "object", __order = keys })
end

local function only(t, allowed, param)
	for _, k in ipairs(getmetatable(t).__order) do
		if not allowed[k] then
			bad(param and (param .. "." .. k) or k, ("unknown parameter: %s"):format(k))
		end
	end
end

-- The text and images of one message content: a string, or parts.
local function content(c, param, texts, images)
	if type(c) == "string" then
		texts[#texts + 1] = c
		return
	end
	if not is_array(c) then
		bad(param, "must be a string or an array of content parts")
	end
	for i, p in ipairs(c) do
		local here = ("%s[%d]"):format(param, i - 1)
		if not is_object(p) then
			bad(here, "must be an object")
		elseif p.type == "input_text" then
			if type(p.text) ~= "string" then
				bad(here .. ".text", "must be a string")
			end
			texts[#texts + 1] = p.text
		elseif p.type == "input_image" then
			if p.file_id ~= nil then
				bad(here .. ".file_id", "file_id inputs are not supported; use a base64 data URL")
			end
			local url = p.image_url
			if is_object(url) then
				url = url.url
			end
			if type(url) ~= "string" then
				bad(here .. ".image_url", "must be a data URL string")
			elseif not url:match("^data:") then
				bad(here .. ".image_url", "images must be inline base64 data URLs")
			end
			images[#images + 1] = url
		else
			bad(here .. ".type", ("unknown content part type: %s"):format(tostring(p.type)))
		end
	end
end

-- The state text and image URLs of the input.
local function input(v)
	local texts, images = {}, json.array()
	if type(v) == "string" then
		return v, images
	elseif not is_array(v) or #v == 0 then
		bad("input", "must be a string or a non-empty array of messages")
	end
	for i, m in ipairs(v) do
		local here = ("input[%d]"):format(i - 1)
		if not is_object(m) then
			bad(here, "must be a message object")
		end
		only(m, { role = true, content = true, type = true }, here)
		if m.role ~= nil and m.role ~= "user" then
			bad(here .. ".role", "must be user")
		end
		content(m.content, here .. ".content", texts, images)
	end
	return table.concat(texts, "\n\n"), images
end

-- The text of a choice or level: the label, then the description.
local function describe(label, desc)
	if desc == nil or desc == json.null or desc == "" then
		return label
	end
	return label .. ": " .. tostring(desc)
end

-- One question as a System One question; info keeps what the reply needs.
local function question(q, i)
	local here = ("questions[%d]"):format(i - 1)
	if not is_object(q) then
		bad(here, "must be an object")
	end
	only(q, { type = true, name = true, instructions = true, choices = true, levels = true }, here)
	if q.name ~= nil and (type(q.name) ~= "string" or q.name == "") then
		bad(here .. ".name", "must be a non-empty string")
	end
	if type(q.instructions) ~= "string" then
		bad(here .. ".instructions", "must be a string")
	end
	local info = { type = q.type, name = q.name }
	if q.type == "predicate" then
		return ordered({ type = "noul", instructions = q.instructions }, { "type", "instructions" }), info
	elseif q.type == "choice" then
		if not is_array(q.choices) then
			bad(here .. ".choices", "must be an array of choices")
		end
		local keys, crit, seen = {}, {}, {}
		info.values = {}
		for j, c in ipairs(q.choices) do
			local at = ("%s.choices[%d]"):format(here, j - 1)
			if not is_object(c) then
				bad(at, "must be an object")
			end
			local v = c.value
			if type(v) ~= "string" and math.type(v) == nil then
				bad(at .. ".value", "must be a string or a number")
			end
			local key = tostring(v)
			if seen[key] then
				bad(at .. ".value", ("duplicate choice value %q"):format(key))
			end
			seen[key] = true
			keys[j], crit[key], info.values[j] = key, describe(key, c.description), v
		end
		return ordered({ type = "choice", instructions = q.instructions, criteria = ordered(crit, keys) },
		    { "type", "instructions", "criteria" }), info
	elseif q.type == "score" then
		if not is_array(q.levels) then
			bad(here .. ".levels", "must be an array of levels")
		end
		local crit = json.array()
		info.labels = {}
		for j, l in ipairs(q.levels) do
			local at = ("%s.levels[%d]"):format(here, j - 1)
			if not is_object(l) or type(l.label) ~= "string" then
				bad(at .. ".label", "must be a string")
			end
			crit[j], info.labels[j] = describe(l.label, l.description), l.label
		end
		return ordered({ type = "score", instructions = q.instructions, criteria = crit },
		    { "type", "instructions", "criteria" }), info
	end
	bad(here .. ".type", ("unknown question type: %s"):format(tostring(q.type)))
end

-- The System One body of a request, and the per-question info.
local function translate(b)
	if not is_object(b) then
		bad(nil, "request body must be a JSON object")
	end
	only(b, { model = true, input = true, questions = true, lgml = true })
	if b.input == nil then
		bad("input", "field required")
	elseif b.model ~= nil and type(b.model) ~= "string" then
		bad("model", "must be a string")
	elseif not is_array(b.questions) or #b.questions == 0 then
		bad("questions", "must be a non-empty array of questions")
	end
	local state, images = input(b.input)
	local keys, qs, infos, names = {}, {}, {}, {}
	for i, q in ipairs(b.questions) do
		local sq, info = question(q, i)
		if info.name then
			if names[info.name] then
				bad(("questions[%d].name"):format(i - 1), ("duplicate question name %q"):format(info.name))
			end
			names[info.name] = true
		end
		-- System One ids are object keys; a unique one per position.
		keys[i] = "q" .. i
		qs[keys[i]], infos[i] = sq, info
	end
	local order = { "model", "state", "questions" }
	local body = ordered({ model = b.model, state = state, questions = ordered(qs, keys) }, order)
	if #images > 0 then
		body.images = images
		order[#order + 1] = "images"
	end
	if b.lgml ~= nil then
		body.lgml = b.lgml
		order[#order + 1] = "lgml"
	end
	return body, infos
end

-- The OpenAI parameter of a System One error location: question key
-- qN is questions[N-1], and its criteria are the choices or levels.
local function param_of(loc, infos)
	local i = loc[2] == "questions" and tonumber((loc[3] or ""):match("^q(%d+)$"))
	if not i then
		return table.concat(loc, ".", 2)
	end
	local p = ("questions[%d]"):format(i - 1)
	if loc[4] == "criteria" then
		p = p .. (infos[i] and infos[i].type == "score" and ".levels" or ".choices")
		if loc[5] ~= nil then
			p = p .. ("[%s]"):format(loc[5])
		end
	elseif loc[4] ~= nil then
		p = p .. "." .. loc[4]
	end
	return p
end

--- Reads and checks a request body.
-- @tparam string text the body
-- @treturn[1] table the System One engine request (see `lgml.systemone.check`)
-- @treturn[1] table the info of each question: `type`, `name`, and
-- `values` (choice) or `labels` (score)
-- @treturn[2] nil
-- @treturn[2] string the error message
-- @treturn[2] int the status
-- @treturn[2] ?string the parameter at fault
function M.check(text)
	local b, err = systemone.decode(text)
	if b == nil then
		return nil, "invalid JSON: " .. err, 400
	end
	local ok, body, infos = pcall(translate, b)
	if not ok then
		if getmetatable(body) == Bad then
			return nil, body.msg, 400, body.param
		end
		error(body, 0)
	end
	local req, ebody = systemone.check(tojson.encode(body))
	if not req then
		local d = ebody.detail and ebody.detail[1] or {}
		return nil, d.msg or "invalid request", 400, d.loc and param_of(d.loc, infos)
	end
	return req, infos
end

local function round(x, places)
	local s = 10 ^ places
	return math.floor(x * s + 0.5) / s
end

-- One answer. Probabilities are rounded to systemone.PLACES places,
-- score and confidence to 4, as Jev does.
local function answer(q, info, a)
	local out, keys
	if info.type == "predicate" then
		out = { type = "predicate", name = info.name,
			probability = round(math.min(1, math.max(0, a.yes)), systemone.PLACES) }
		keys = { "type", "name", "probability" }
		return ordered(out, keys)
	end
	local p = systemone.round(a.probabilities)
	local probs = json.array()
	if info.type == "choice" then
		local best = 1
		for i, v in ipairs(info.values) do
			probs[i] = ordered({ value = v, probability = p[i] }, { "value", "probability" })
			if p[i] > p[best] then
				best = i
			end
		end
		out = { type = "choice", name = info.name, choice = info.values[best], probabilities = probs,
			confidence = round(systemone.choice_confidence(p), 4) }
		keys = { "type", "name", "choice", "probabilities", "confidence" }
	else
		for i, l in ipairs(info.labels) do
			probs[i] = ordered({ value = i - 1, label = l, probability = p[i] }, { "value", "label", "probability" })
		end
		out = { type = "score", name = info.name, score = round(systemone.score_expectation(p), 4),
			probabilities = probs, confidence = round(systemone.score_confidence(p), 4) }
		keys = { "type", "name", "score", "probabilities", "confidence" }
	end
	return ordered(out, keys)
end

--- Makes the response body from an engine result.
-- @tparam table req the engine request from `check`
-- @tparam {table,...} infos the question info from `check`
-- @tparam table result the result of `Engine:decide`
-- @tparam string id the decision id
-- @treturn string the JSON text
-- @raise when the result does not fit the request
function M.reply(req, infos, result, id)
	local answers = json.array()
	for i, q in ipairs(req.questions) do
		local a = result.answers[q.id]
		assert(a, "engine gave no answer for question " .. i)
		answers[i] = answer(q, infos[i], a)
	end
	local u = result.usage or {}
	local input_tokens = math.tointeger(u.input_tokens) or 0
	return tojson.encode(ordered({
		id = id, object = "decision", created_at = os.time(),
		model = req.model or M.DEFAULT_MODEL,
		answers = answers,
		usage = ordered({ input_tokens = input_tokens, total_tokens = input_tokens },
		    { "input_tokens", "total_tokens" }),
	}, { "id", "object", "created_at", "model", "answers", "usage" }))
end

return M
