--- The System One wire format of Jev (`POST /v1/systemone`).
-- A request has a `state` and a map of typed questions: `noul`,
-- `choice` and `score`. `check` reads a request body and makes the
-- request for `Engine:decide`. `reply` makes the response body from the
-- engine result. The rules follow Jevify (`jevify/wire.py`) and its
-- notes on the Jev 1.13.0 API (`docs/JEV_CONTRACT.md`).
-- `attachments` (images of the state) is an lgml extension; see `check`.
-- @module lgml.systemone
local json = require("lgml.json")
local tojson = require("lgml.chat.tojson")
local media = require("lgml.chat.media")

local M = {}

--- Most images in `attachments` (16).
M.MAX_ATTACHMENTS = 16
--- Largest image in bytes after base64 decode (20 MiB).
M.MAX_ATTACHMENT_BYTES = 20 << 20

--- Most options of a choice question (255).
M.MAX_CHOICE_OPTIONS = 255
--- Fewest levels of a score question (2).
M.MIN_SCORE_LEVELS = 2
--- Most levels of a score question (26, as Ollama allows).
M.MAX_SCORE_LEVELS = 26
--- Most questions of a request (64, as Ollama allows).
M.MAX_QUESTIONS = 64
--- The model name of a request with no `model`.
M.DEFAULT_MODEL = "jevify-latest"
--- Decimal places of reported probabilities: Jev rounds to 0.01.
M.PLACES = 2

-- ordered JSON

local function object(keys)
	return setmetatable({}, { __jsontype = "object", __order = keys })
end

local function is_object(v)
	local mt = type(v) == "table" and getmetatable(v)
	return mt ~= nil and mt ~= false and mt.__jsontype == "object"
end

local function is_array(v)
	return type(v) == "table" and json.isarray(v)
end

local ESC = { ['"'] = '"', ["\\"] = "\\", ["/"] = "/", b = "\b", f = "\f",
	n = "\n", r = "\r", t = "\t" }

local D = {}
D.__index = D

function D:fail(msg)
	error({ json = ("%s at byte %d"):format(msg, self.i) }, 0)
end

function D:ws()
	self.i = self.s:find("[^ \t\r\n]", self.i) or #self.s + 1
end

function D:string()
	local s, i, parts = self.s, self.i + 1, {}
	while true do
		local j = s:find("[\0-\31\"\\]", i)
		if not j then
			self:fail("unterminated string")
		end
		parts[#parts + 1] = s:sub(i, j - 1)
		local c = s:sub(j, j)
		if c == '"' then
			self.i = j + 1
			return table.concat(parts)
		elseif c ~= "\\" then
			self.i = j
			self:fail("control character in string")
		end
		local e = s:sub(j + 1, j + 1)
		if ESC[e] then
			parts[#parts + 1] = ESC[e]
			i = j + 2
		elseif e == "u" then
			local cp = tonumber(s:match("^%x%x%x%x", j + 2) or "", 16)
			if not cp then
				self.i = j
				self:fail("bad \\u escape")
			end
			i = j + 6
			if cp >= 0xd800 and cp <= 0xdbff then
				local lo = tonumber(s:match("^\\u(%x%x%x%x)", i) or "", 16)
				if lo and lo >= 0xdc00 and lo <= 0xdfff then
					cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00)
					i = i + 6
				else
					cp = 0xfffd
				end
			elseif cp >= 0xdc00 and cp <= 0xdfff then
				cp = 0xfffd
			end
			parts[#parts + 1] = utf8.char(cp)
		else
			self.i = j
			self:fail("bad escape")
		end
	end
end

function D:number()
	local s, i = self.s, self.i
	local int = s:match("^-?0", i) or s:match("^-?[1-9]%d*", i)
	if not int then
		self:fail("bad number")
	end
	local j = i + #int
	local frac = s:match("^%.%d+", j) or ""
	j = j + #frac
	local exp = s:match("^[eE][-+]?%d+", j) or ""
	j = j + #exp
	self.i = j
	local text = s:sub(i, j - 1)
	if frac == "" and exp == "" then
		local v = math.tointeger(tonumber(text))
		if v and tostring(v) == text:gsub("^%-0$", "0") then
			return v
		end
	end
	return tonumber(text) + 0.0
end

function D:value(depth)
	if depth > json.max_depth then
		self:fail("nesting too deep")
	end
	self:ws()
	local s, i = self.s, self.i
	local c = s:sub(i, i)
	if c == "{" then
		local keys, seen = {}, {}
		local t = object(keys)
		self.i = i + 1
		self:ws()
		if s:sub(self.i, self.i) == "}" then
			self.i = self.i + 1
			return t
		end
		while true do
			self:ws()
			if s:sub(self.i, self.i) ~= '"' then
				self:fail("expected a key")
			end
			local k = self:string()
			self:ws()
			if s:sub(self.i, self.i) ~= ":" then
				self:fail("expected ':'")
			end
			self.i = self.i + 1
			local v = self:value(depth + 1)
			if seen[k] then
				getmetatable(t).__dups = getmetatable(t).__dups or k
			else
				seen[k] = true
				keys[#keys + 1] = k
			end
			-- Python keeps the first position and the last value.
			t[k] = v
			self:ws()
			c = s:sub(self.i, self.i)
			self.i = self.i + 1
			if c == "}" then
				return t
			elseif c ~= "," then
				self.i = self.i - 1
				self:fail("expected ',' or '}'")
			end
		end
	elseif c == "[" then
		local t = json.array()
		self.i = i + 1
		self:ws()
		if s:sub(self.i, self.i) == "]" then
			self.i = self.i + 1
			return t
		end
		while true do
			t[#t + 1] = self:value(depth + 1)
			self:ws()
			c = s:sub(self.i, self.i)
			self.i = self.i + 1
			if c == "]" then
				return t
			elseif c ~= "," then
				self.i = self.i - 1
				self:fail("expected ',' or ']'")
			end
		end
	elseif c == '"' then
		return self:string()
	elseif s:find("^true", i) then
		self.i = i + 4
		return true
	elseif s:find("^false", i) then
		self.i = i + 5
		return false
	elseif s:find("^null", i) then
		self.i = i + 4
		return json.null
	end
	return self:number()
end

--- Decodes JSON and keeps the key order of objects.
-- An object has a metatable with `__jsontype = "object"` and
-- `__order`, the keys in text order, as `lgml.chat.tojson` reads. A
-- repeated key keeps its first position and its last value, as in
-- Python; `__dups` in the metatable holds the first repeated key.
-- Arrays carry the `lgml.json` array mark. Numbers without a fraction
-- or exponent are integers; others are floats.
-- @tparam string s the JSON text
-- @treturn[1] any the value
-- @treturn[2] nil
-- @treturn[2] string the error message
function M.decode(s)
	if not utf8.len(s) then
		return nil, "body is not valid UTF-8"
	end
	local d = setmetatable({ s = s, i = 1 }, D)
	local ok, v = pcall(function()
		local v = d:value(1)
		d:ws()
		if d.i <= #s then
			d:fail("trailing data")
		end
		return v
	end)
	if not ok then
		if type(v) == "table" and v.json then
			return nil, v.json
		end
		error(v, 0)
	end
	return v
end

--- Encodes a value as Python `json.dumps(v, ensure_ascii=False)` does.
-- With indent, items go on their own lines, as `json.dumps(v,
-- indent=n)` gives. Object keys follow `__order`.
-- @param v the value
-- @tparam[opt] int indent spaces per level; nil for one line
-- @treturn string the JSON text
function M.dumps(v, indent)
	if not indent then
		return tojson.encode(v)
	end
	local function enc(x, pad)
		if type(x) ~= "table" or x == json.null then
			return tojson.encode(x)
		end
		local inner = pad .. string.rep(" ", indent)
		local parts = {}
		if tojson.kind(x) == "array" then
			for i = 1, #x do
				parts[i] = inner .. enc(x[i], inner)
			end
			if #parts == 0 then
				return "[]"
			end
			return "[\n" .. table.concat(parts, ",\n") .. "\n" .. pad .. "]"
		end
		for i, k in ipairs(tojson.keys(x)) do
			parts[i] = inner .. tojson.string(tostring(k)) .. ": " .. enc(x[k], inner)
		end
		if #parts == 0 then
			return "{}"
		end
		return "{\n" .. table.concat(parts, ",\n") .. "\n" .. pad .. "}"
	end
	return enc(v, "")
end

--- Returns a JSON value as compact text with no spaces, as Go's
-- json.Compact gives it. Keys keep their order.
-- @param v the value
-- @treturn string the JSON text
function M.compact(v)
	if type(v) ~= "table" or v == json.null then
		return tojson.encode(v)
	end
	local parts = {}
	if tojson.kind(v) == "array" then
		for i = 1, #v do
			parts[i] = M.compact(v[i])
		end
		return "[" .. table.concat(parts, ",") .. "]"
	end
	for i, k in ipairs(tojson.keys(v)) do
		parts[i] = tojson.string(tostring(k)) .. ":" .. M.compact(v[k])
	end
	return "{" .. table.concat(parts, ",") .. "}"
end

--- Returns the prompt text of a JSON value, as Jevify's `_describe`.
-- nil and null give `""`, a string gives itself, and other values give
-- their one-line JSON.
-- @param v the value
-- @treturn string the text
function M.describe(v)
	if v == nil or v == json.null then
		return ""
	elseif type(v) == "string" then
		return v
	end
	return M.dumps(v)
end

-- Python truth: "", {} and [] are false.
local function truthy(v)
	if v == nil or v == json.null or v == "" then
		return false
	elseif type(v) == "table" then
		return next(v) ~= nil
	end
	return v ~= false
end

-- A plain copy with no order metatables, for engines and IPC.
local function plain(v)
	if type(v) ~= "table" or v == json.null then
		return v
	end
	local t = is_array(v) and json.array() or {}
	for k, x in pairs(v) do
		t[k] = plain(x)
	end
	return t
end

-- request checks

local Bad = {}

local function bad(loc, msg)
	error(setmetatable({ loc = loc, msg = msg }, Bad), 0)
end

local function at(base, ...)
	local t = table.move(base, 1, #base, 1, {})
	for _, k in ipairs({ ... }) do
		t[#t + 1] = k
	end
	return t
end

-- JSONContent in jevify.wire: a string, an object or an array.
local function content(v, loc, nullable)
	if (nullable and (v == nil or v == json.null)) or type(v) == "string" or
	    is_object(v) or is_array(v) then
		return
	end
	bad(loc, "must be a string, an object or an array" .. (nullable and " or null" or ""))
end

local function only(t, allowed, loc)
	for _, k in ipairs(getmetatable(t).__order) do
		if not allowed[k] then
			bad(at(loc, k), "extra field not permitted")
		end
	end
end

local QFIELDS = { type = true, instructions = true, criteria = true }

local function check_noul(q, loc, out)
	local c = q.criteria
	if c ~= nil and c ~= json.null then
		if not is_object(c) then
			bad(at(loc, "criteria"), "must be an object or null")
		end
		only(c, { ["true"] = true, ["false"] = true }, at(loc, "criteria"))
		content(c["true"], at(loc, "criteria", "true"), true)
		content(c["false"], at(loc, "criteria", "false"), true)
		out.yes = truthy(c["true"]) and M.describe(c["true"]) or nil
		out.no = truthy(c["false"]) and M.describe(c["false"]) or nil
	end
end

local function check_choice(q, loc, out)
	local c = q.criteria
	if not is_object(c) then
		bad(at(loc, "criteria"), "must be an object of options")
	end
	local mt = getmetatable(c)
	local keys = mt.__order
	if mt.__dups then
		bad(at(loc, "criteria"), ("duplicate option %q"):format(mt.__dups))
	elseif #keys < 2 or #keys > M.MAX_CHOICE_OPTIONS then
		bad(at(loc, "criteria"), ("choice needs 2..%d options, got %d"):format(
		    M.MAX_CHOICE_OPTIONS, #keys))
	end
	out.options = json.array()
	for i, k in ipairs(keys) do
		if k == "" then
			bad(at(loc, "criteria"), "choice option names must be non-empty")
		end
		content(c[k], at(loc, "criteria", k), true)
		out.options[i] = { key = k, text = M.describe(c[k]), null = c[k] == json.null or nil }
	end
end

local function check_score(q, loc, out)
	local c = q.criteria
	if not is_array(c) then
		bad(at(loc, "criteria"), "must be an array of levels")
	elseif #c < M.MIN_SCORE_LEVELS or #c > M.MAX_SCORE_LEVELS then
		bad(at(loc, "criteria"), ("score needs %d..%d levels, got %d"):format(
		    M.MIN_SCORE_LEVELS, M.MAX_SCORE_LEVELS, #c))
	end
	out.levels = json.array()
	for i, v in ipairs(c) do
		content(v, at(loc, "criteria", i - 1))
		out.levels[i] = M.describe(v)
	end
end

local CHECK = { noul = check_noul, choice = check_choice, score = check_score }

local function check_question(id, q)
	local loc = { "body", "questions", id }
	if not is_object(q) then
		bad(loc, "must be an object")
	end
	local kind = q.type
	if not CHECK[kind] then
		bad(at(loc, "type"), ("unknown question type: %s"):format(
		    type(kind) == "string" and ("%q"):format(kind) or "none"))
	end
	only(q, QFIELDS, loc)
	if kind == "score" or kind == "choice" then
		if q.criteria == nil then
			bad(at(loc, "criteria"), "field required")
		end
	end
	content(q.instructions, at(loc, "instructions"), true)
	local out = { id = id, type = kind, instructions = M.describe(q.instructions) }
	if type(q.instructions) == "table" then
		out.instructions_compact = M.compact(q.instructions)
	end
	CHECK[kind](q, loc, out)
	return out
end

local LGML_FIELDS = { raw = true, recipe = true }

local function check_lgml(x, out)
	if not is_object(x) then
		bad({ "body", "lgml" }, "must be an object")
	end
	only(x, LGML_FIELDS, { "body", "lgml" })
	if x.raw ~= nil and type(x.raw) ~= "boolean" then
		bad({ "body", "lgml", "raw" }, "must be a boolean")
	elseif x.recipe ~= nil and not is_object(x.recipe) then
		bad({ "body", "lgml", "recipe" }, "must be an object")
	end
	out.raw = x.raw
	out.recipe = x.recipe and plain(x.recipe)
end

-- Raised for an http or https image URL; the route answers 400.
local Unsupported = {}

local PNG_SIG = "\137PNG\r\n\26\n"

-- The URL of one attachment: a string, or an image_url part.
local function attachment_url(v, loc)
	if type(v) == "string" then
		return v
	elseif is_object(v) then
		if v.type ~= "image_url" then
			bad(at(loc, "type"), 'must be "image_url"')
		end
		local u = v.image_url
		if is_object(u) then
			u = u.url
		end
		if type(u) ~= "string" then
			bad(at(loc, "image_url", "url"), "must be a string")
		end
		return u
	end
	bad(loc, "must be a data URL string or an image_url part")
end

local function check_attachments(list)
	local loc = { "body", "attachments" }
	if not is_array(list) then
		bad(loc, "must be an array of images")
	elseif #list > M.MAX_ATTACHMENTS then
		bad(loc, ("at most %d images, got %d"):format(M.MAX_ATTACHMENTS, #list))
	end
	local out = {}
	for i, v in ipairs(list) do
		local here = at(loc, i - 1)
		local url = attachment_url(v, here)
		local scheme = (url:match("^(%a[%w+.-]*):") or ""):lower()
		if scheme == "http" or scheme == "https" then
			error(setmetatable({ loc = here,
				msg = "image url must be a data URL; http and https URLs are not supported" },
			    Unsupported), 0)
		end
		local head = url:match("^[^,]*")
		if #url - #head > M.MAX_ATTACHMENT_BYTES // 3 * 4 + 8 then
			bad(here, ("image larger than %d bytes"):format(M.MAX_ATTACHMENT_BYTES))
		end
		local bytes, mime = media.data_url(url)
		if not bytes then
			bad(here, mime)
		end
		local png, jpeg = bytes:sub(1, 8) == PNG_SIG, bytes:sub(1, 3) == "\255\216\255"
		if not (mime == "image/png" or mime == "image/jpeg" or mime == "image/jpg") then
			bad(here, "image must be image/png or image/jpeg, got " .. mime)
		elseif not (png or jpeg) then
			bad(here, "data is not a PNG or JPEG file")
		elseif #bytes > M.MAX_ATTACHMENT_BYTES then
			bad(here, ("image larger than %d bytes"):format(M.MAX_ATTACHMENT_BYTES))
		end
		out[i] = { bytes = bytes, mime = png and "image/png" or "image/jpeg" }
	end
	return out
end

local TOP = { state = true, model = true, questions = true, lgml = true, attachments = true,
	keep_alive = true }

local function check_body(b)
	if not is_object(b) then
		bad({ "body" }, "request body must be a JSON object")
	end
	only(b, TOP, { "body" })
	if b.state == nil then
		bad({ "body", "state" }, "field required")
	end
	content(b.state, { "body", "state" })
	if b.model ~= nil and type(b.model) ~= "string" then
		bad({ "body", "model" }, "must be a string")
	end
	local qs = b.questions
	if not is_object(qs) then
		bad({ "body", "questions" }, "must be an object of questions")
	end
	local mt = getmetatable(qs)
	if #mt.__order == 0 then
		bad({ "body", "questions" }, "at least one question is required")
	elseif mt.__dups then
		bad({ "body", "questions" }, ("duplicate question %q"):format(mt.__dups))
	elseif #mt.__order > M.MAX_QUESTIONS then
		bad({ "body", "questions" }, ("at most %d questions, got %d"):format(M.MAX_QUESTIONS, #mt.__order))
	end
	-- keep_alive is Ollama's; the pool unloads by its own idle time.
	local ka = b.keep_alive
	if ka ~= nil and ka ~= json.null and type(ka) ~= "string" and type(ka) ~= "number" then
		bad({ "body", "keep_alive" }, "must be a duration string or a number of seconds")
	end
	local req = {
		model = b.model,
		state = type(b.state) == "string" and b.state or M.dumps(b.state, 2),
		questions = json.array(),
	}
	if type(b.state) == "table" then
		req.state_compact = M.compact(b.state)
	end
	for i, id in ipairs(mt.__order) do
		req.questions[i] = check_question(id, qs[id])
	end
	if b.lgml ~= nil then
		check_lgml(b.lgml, req)
	end
	if b.attachments ~= nil and b.attachments ~= json.null then
		local images = check_attachments(b.attachments)
		req.images = #images > 0 and images or nil
	end
	return req
end

--- Reads and checks a request body.
-- The checks are those of `jevify.wire.SystemOneRequest`: unknown
-- fields, question types, 2 to 255 choice options with non-empty
-- names, and string, object or array content. Score levels (2 to 26),
-- questions (at most 64) and `keep_alive` follow Ollama's rules. lgml
-- also rejects a repeated question id or option name, which Python
-- would drop in silence.
--
-- lgml extension to the Jev contract: `attachments`, the images of the
-- state. It is an array of up to `MAX_ATTACHMENTS` items. An item is a
-- data URL string (`data:image/png;base64,...` or `image/jpeg`), as
-- the Jev-like wrapper for vision models sends, or an OpenAI part
-- `{ "type": "image_url", "image_url": { "url": ... } }`. Each image is
-- a PNG or JPEG file of at most `MAX_ATTACHMENT_BYTES`. An http or
-- https URL gives status 400; other bad items give 422.
--
-- The engine request is:
--
-- - `model`: the model name, or nil
-- - `state`: the state text; a JSON state is its `json.dumps(indent=2)`
-- - `questions`: an array of `{ id, type, instructions }`, in request
--   order. A choice has `options`, an array of `{ key, text }`. A score
--   has `levels`, an array of texts. A noul can have `yes` and `no`,
--   the texts of `criteria.true` and `criteria.false`. Every text is
--   `describe` of the JSON value.
-- - `raw`, `recipe`: the lgml extension object `lgml` of the body
-- - `images`: the attachments in order, each `{ bytes, mime }`; nil
--   when there are none
-- @tparam string text the body
-- @treturn[1] table the engine request
-- @treturn[1] table the decoded body, with key order
-- @treturn[2] nil
-- @treturn[2] table the error body: `{ detail = { { loc, msg, type } } }`
-- @treturn[2] int the status: 422, or 400 for an http or https image URL
function M.check(text)
	local b, err = M.decode(text)
	if b == nil then
		return nil, M.error_body({ "body" }, "invalid JSON: " .. err, "json_invalid"), 422
	end
	local ok, r = pcall(check_body, b)
	if ok then
		return r, b
	elseif getmetatable(r) == Bad then
		return nil, M.error_body(r.loc, r.msg), 422
	elseif getmetatable(r) == Unsupported then
		return nil, M.error_body(r.loc, r.msg), 400
	end
	error(r, 0)
end

--- Makes an error body as FastAPI gives for a failed check.
-- @tparam {string,...} loc the path to the bad field
-- @tparam string msg the message
-- @tparam[opt] string etype the error type (default `"value_error"`)
-- @treturn table `{ detail = { { loc, msg, type } } }`
function M.error_body(loc, msg, etype)
	return { detail = json.array({ {
		loc = json.array(loc), msg = msg, type = etype or "value_error",
	} }) }
end

--- The field sets that `check` accepts, for `lgml.server.openapi`.
-- Each set maps a field name to true. Read only.
-- @table fields
-- @field request the top-level fields of a request
-- @field question the fields of a question
-- @field lgml the fields of the lgml extension object
-- @field types the question types
M.fields = { request = TOP, question = QFIELDS, lgml = LGML_FIELDS, types = {} }
for k in pairs(CHECK) do
	M.fields.types[k] = true
end

-- statistics

--- Rounds a distribution so that it sums to exactly 1.
-- It uses the largest remainder method, as `jevify.wire.round_probabilities`.
-- @tparam {number,...} values the probabilities
-- @tparam[opt] int places decimal places (default `PLACES`)
-- @treturn {number,...} the rounded values
function M.round(values, places)
	local scale = 10 ^ (places or M.PLACES)
	local raw, total = {}, 0
	for i, v in ipairs(values) do
		raw[i] = math.max(v, 0) * scale
		total = total + raw[i]
	end
	if total == 0 then
		total = 1
	end
	local floors, order, sum = {}, {}, 0
	for i = 1, #raw do
		raw[i] = raw[i] * scale / total
		floors[i] = math.floor(raw[i])
		sum = sum + floors[i]
		order[i] = i
	end
	-- Python sorts stably: equal remainders keep index order.
	table.sort(order, function(a, b)
		local ra, rb = raw[a] - floors[a], raw[b] - floors[b]
		if ra ~= rb then
			return ra > rb
		end
		return a < b
	end)
	for k = 1, math.floor(scale - sum + 0.5) do
		local i = order[k]
		if i then
			floors[i] = floors[i] + 1
		end
	end
	for i = 1, #floors do
		floors[i] = floors[i] / scale
	end
	return floors
end

local function clip01(x)
	return math.min(1.0, math.max(0.0, x))
end

--- Returns the choice confidence `(p_max - 1/K) / (1 - 1/K)`.
-- Jevify measured this rule on every Jev answer.
-- @tparam {number,...} p the probabilities
-- @treturn number the confidence, from 0 to 1
function M.choice_confidence(p)
	local k = #p
	if k < 2 then
		return 1.0
	end
	local max = math.max(table.unpack(p))
	return clip01((max - 1 / k) / (1 - 1 / k))
end

--- Returns the expected level index, Jev's `score`.
-- @tparam {number,...} p the level probabilities, lowest first
-- @treturn number the sum of `i * p[i]`, with levels from 0
function M.score_expectation(p)
	local s = 0.0
	for i, v in ipairs(p) do
		s = s + (i - 1) * v
	end
	return s
end

--- Returns the score confidence `1 - std / std_max`.
-- Jev's own rule is unknown; this is Jevify's. std is the standard
-- deviation of the level index and std_max is `(K - 1) / 2`.
-- @tparam {number,...} p the level probabilities, lowest first
-- @treturn number the confidence, from 0 to 1
function M.score_confidence(p)
	local k = #p
	if k < 2 then
		return 1.0
	end
	local mu, var = M.score_expectation(p), 0.0
	for i, v in ipairs(p) do
		var = var + v * (i - 1 - mu) ^ 2
	end
	return clip01(1 - math.sqrt(math.max(var, 0)) / ((k - 1) / 2))
end

-- Python round(x, n) for the fields that Jevify rounds to 4 places.
local function round_to(x, places)
	local s = 10 ^ places
	return math.floor(x * s + 0.5) / s
end

local function ordered(t, keys)
	return setmetatable(t, { __jsontype = "object", __order = keys })
end

--- Returns Ollama's confidence: `1 - H(p) / ln N`, with H the entropy.
-- @tparam {number,...} p the probabilities
-- @treturn number the confidence, from 0 to 1
function M.entropy_confidence(p)
	local h = 0.0
	for _, v in ipairs(p) do
		if v > 0 then
			h = h - v * math.log(v)
		end
	end
	return clip01(1 - h / math.log(#p))
end

-- An answer of the nimble format, as Ollama gives it: no rounding and
-- the entropy confidence.
local function answer_nimble(q, src, a)
	if q.type == "noul" then
		return ordered({ type = "noul", noul = clip01(a.yes) }, { "type", "noul" })
	end
	local keys = {}
	if q.type == "choice" then
		for i, o in ipairs(q.options) do
			keys[i] = o.key
		end
	else
		for i = 1, #q.levels do
			keys[i] = tostring(i - 1)
		end
	end
	local p = a.probabilities
	local probs, best = ordered({}, keys), 1
	for i, k in ipairs(keys) do
		probs[k] = p[i]
		if p[i] > p[best] then
			best = i
		end
	end
	local conf = M.entropy_confidence(p)
	if q.type == "choice" then
		return ordered({ type = "choice", choice = keys[best], probabilities = probs, confidence = conf },
		    { "type", "choice", "probabilities", "confidence" })
	end
	local legend = ordered({}, keys)
	for i, k in ipairs(keys) do
		legend[k] = src.criteria[i]
	end
	return ordered({ type = "score", score = M.score_expectation(p), legend = legend,
		probabilities = probs, confidence = conf },
	    { "type", "score", "legend", "probabilities", "confidence" })
end

local function answer(q, src, a, format)
	if format == "nimble" then
		local out = answer_nimble(q, src, a)
		out.lgml = a.lgml
		return out
	end
	local out
	if q.type == "noul" then
		local p = round_to(clip01(a.yes), M.PLACES)
		out = ordered({ type = "noul", noul = p }, { "type", "noul" })
	else
		local keys = {}
		if q.type == "choice" then
			for i, o in ipairs(q.options) do
				keys[i] = o.key
			end
		else
			for i = 1, #q.levels do
				keys[i] = tostring(i - 1)
			end
		end
		assert(#a.probabilities == #keys, "engine gave the wrong number of probabilities")
		local vals = M.round(a.probabilities)
		local probs = ordered({}, keys)
		local best = 1
		for i, k in ipairs(keys) do
			probs[k] = vals[i]
			if vals[i] > vals[best] then
				best = i
			end
		end
		if q.type == "choice" then
			out = ordered({
				type = "choice", choice = keys[best],
				confidence = round_to(M.choice_confidence(vals), 4),
				probabilities = probs,
			}, { "type", "choice", "confidence", "probabilities" })
		else
			local legend = ordered({}, keys)
			for i, k in ipairs(keys) do
				legend[k] = src.criteria[i]
			end
			out = ordered({
				type = "score", score = round_to(M.score_expectation(vals), 4),
				confidence = round_to(M.score_confidence(vals), 4),
				legend = legend, probabilities = probs,
			}, { "type", "score", "confidence", "legend", "probabilities" })
		end
	end
	if a.lgml then
		out.lgml = a.lgml
	end
	return out
end

--- Makes the response body from an engine result.
-- Probabilities are rounded to `PLACES`; score and confidence to 4
-- places, as Jevify does. A noul gives `P(yes)` and no confidence.
-- @tparam table req the engine request from `check`
-- @tparam table body the decoded body from `check`
-- @tparam table result the result of `Engine:decide`
-- @treturn string the JSON text
-- @raise when the result does not fit the request
function M.reply(req, body, result)
	local ids, answers = {}, {}
	for i, q in ipairs(req.questions) do
		ids[i] = q.id
		local a = result.answers[q.id]
		assert(a, "engine gave no answer for question " .. q.id)
		answers[q.id] = answer(q, body.questions[q.id], a, result.format)
	end
	local u = result.usage or {}
	local out = ordered({
		model = result.model or req.model or M.DEFAULT_MODEL,
		answers = ordered(answers, ids),
		usage = ordered({
			input_tokens = math.tointeger(u.input_tokens) or 0,
			output_tokens = math.tointeger(u.output_tokens) or 0,
		}, { "input_tokens", "output_tokens" }),
	}, { "model", "answers", "usage" })
	return tojson.encode(out)
end

return M
