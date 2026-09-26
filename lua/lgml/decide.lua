--- Typed decisions by logit readout, as Jevify's Tier 0 engine does.
-- A question is not answered by generation. The engine renders a
-- prompt that ends with an answer cue, runs it once and reads the
-- log-probabilities of the allowed answers at the last position. A
-- recipe then maps these log-scores to probabilities. The prompt is
-- Jevify's `jevify` format (`jevify/engine/template.py`), with the
-- state first so that questions on one state share a prefix. The
-- recipe is that of `jevify/engine/predict.py`. The request and result
-- are those of `lgml.server.engine.Engine.decide`.
-- @module lgml.decide
local json = require("lgml.json")
local mt19937 = require("lgml.mt19937")

local M = {}

--- The system prompt of the chat turn.
M.SYSTEM_PROMPT = "You are a System One model: you read a state and answer one question " ..
    "about it with exactly one of the allowed answers. Do not explain."
--- The answer cue. Candidates follow it with no space.
M.CUE = "Answer: "
--- The state of the content-free prompt for prior correction.
M.CONTENT_FREE_STATE = "N/A"
--- Longest state in characters; the rest is cut.
M.MAX_STATE_CHARS = 12000
--- The name of the prompt format.
M.PROMPT_FORMAT = "jevify"
--- Seed of the first extra option order; order i uses `SEED + i`.
M.SEED = 1000

local PRIMS = { "choice", "score", "noul" }

-- Python random.Random(seed): MT19937 seeded by init_by_array.

local function py_random(seed)
	local N, MASK = 624, 0xffffffff
	local key = {}
	seed = math.abs(seed)
	repeat
		key[#key + 1] = seed & MASK
		seed = seed >> 32
	until seed == 0
	local g = mt19937.new(19650218)
	local s = g.s
	local i, j = 1, 0
	for _ = 1, math.max(N, #key) do
		local p = s[i - 1]
		s[i] = ((s[i] ~ ((p ~ (p >> 30)) * 1664525)) + key[j + 1] + j) & MASK
		i, j = i + 1, j + 1
		if i >= N then
			s[0], i = s[N - 1], 1
		end
		if j >= #key then
			j = 0
		end
	end
	for _ = 1, N - 1 do
		local p = s[i - 1]
		s[i] = ((s[i] ~ ((p ~ (p >> 30)) * 1566083941)) - i) & MASK
		i = i + 1
		if i >= N then
			s[0], i = s[N - 1], 1
		end
	end
	s[0] = 0x80000000
	g.i = N
	return g
end

local function randbelow(g, n)
	local k = 0
	while (1 << k) <= n do
		k = k + 1
	end
	local r = g:next() >> (32 - k)
	while r >= n do
		r = g:next() >> (32 - k)
	end
	return r
end

--- Returns `random.Random(seed).random()` of Python, for tests.
-- @tparam int seed the seed
-- @tparam[opt=1] int n the draw number
-- @treturn number the float of draw n
function M.py_random(seed, n)
	local g, x = py_random(seed), nil
	for _ = 1, n or 1 do
		local a, b = g:next() >> 5, g:next() >> 6
		x = (a * 67108864.0 + b) / 9007199254740992.0
	end
	return x
end

--- Shuffles `0..n-1` as `random.Random(seed).shuffle` of Python.
-- @tparam int n the length
-- @tparam int seed the seed
-- @treturn {int,...} the order, 0-based values
function M.shuffle(n, seed)
	local x = {}
	for i = 1, n do
		x[i] = i - 1
	end
	local g = py_random(seed)
	for i = n - 1, 1, -1 do
		local j = randbelow(g, i + 1)
		x[i + 1], x[j + 1] = x[j + 1], x[i + 1]
	end
	return x
end

-- text

-- Python str.isspace code points.
local SPACE = { [0x85] = true, [0xa0] = true, [0x1680] = true, [0x2028] = true,
	[0x2029] = true, [0x202f] = true, [0x205f] = true, [0x3000] = true }

local function space(cp)
	return (cp >= 9 and cp <= 13) or (cp >= 0x1c and cp <= 0x20) or SPACE[cp] or
	    (cp >= 0x2000 and cp <= 0x200a)
end

--- Strips white space at both ends, as Python `str.strip()`.
-- @tparam string s UTF-8 text
-- @treturn string the text
function M.strip(s)
	local cps = {}
	for _, c in utf8.codes(s) do
		cps[#cps + 1] = c
	end
	local a, b = 1, #cps
	while a <= b and space(cps[a]) do
		a = a + 1
	end
	while b >= a and space(cps[b]) do
		b = b - 1
	end
	if a > b then
		return ""
	end
	return s:sub(utf8.offset(s, a), (utf8.offset(s, b + 1) or #s + 1) - 1)
end

--- Returns the state text as the prompt shows it.
-- A state longer than `MAX_STATE_CHARS` code points is cut and ends
-- with `"\n…[truncated]"`.
-- @tparam string text the state text
-- @treturn string the text
function M.render_state(text)
	local cut = utf8.offset(text, M.MAX_STATE_CHARS + 1)
	if cut and cut <= #text then
		return text:sub(1, cut - 1) .. "\n…[truncated]"
	end
	return text
end

--- Returns identifiers for k options when none fit the tokenizer.
-- @tparam int k the option count
-- @treturn {string,...} `A..Z` for k up to 26, else `1..k`
function M.default_identifiers(k)
	local out = {}
	for i = 1, k do
		out[i] = k <= 26 and string.char(64 + i) or tostring(i)
	end
	return out
end

local function common(a, b, ia, ib)
	local n = 0
	ia, ib = ia or 0, ib or 0
	while a[ia + n + 1] ~= nil and a[ia + n + 1] == b[ib + n + 1] do
		n = n + 1
	end
	return n
end

--- Finds option identifiers that are one token after the cue.
-- The candidates are `A..Z`, then `AA..ZZ`, up to 255, as Jevify's
-- `HFScorer.identifiers`. With fewer than 26, it gives the numbers
-- `1..255`.
-- @param tk the tokenizer, from `lgml.tokenizer.open`
-- @treturn {string,...} the identifiers
function M.identifiers(tk)
	local probe = "Allowed answers: A, B\n" .. M.CUE
	local raw = { add_bos = false }
	local base = tk:encode(probe, raw)
	local ok = {}
	local function try(c)
		local ids = tk:encode(probe .. c, raw)
		if #ids - common(base, ids) == 1 then
			ok[#ok + 1] = c
		end
		return #ok >= 255
	end
	for a = 65, 90 do
		if try(string.char(a)) then
			return ok
		end
	end
	for a = 65, 90 do
		for b = 65, 90 do
			if try(string.char(a, b)) then
				return ok
			end
		end
	end
	if #ok < 26 then
		return M.default_identifiers(255)
	end
	return ok
end

-- prompt

local function assemble(state, lines, images)
	local t = { "State:\n" .. (images or "") .. M.render_state(state), "" }
	table.move(lines, 1, #lines, #t + 1, t)
	t[#t + 1] = M.CUE
	return table.concat(t, "\n")
end

--- Renders one question as Jevify's `render` with the state first.
-- The result has `prefix` (the text up to and with the cue),
-- `candidates` (the answer texts to score), `keys` (the answer key of
-- each candidate: option name, level index or `"1"` and `"0"`) and
-- `mode` (`"index"`, `"digit"` or `"yesno"`).
-- @tparam string state the state text
-- @tparam table q a question of the engine request
-- @tparam[opt] table opts `seed`: shuffle the options with this seed;
-- `content_free`: use `CONTENT_FREE_STATE` and no images;
-- `identifiers`: the option identifiers; `images`: the image markers,
-- which go before the state text
-- @treturn table the rendering
function M.render(state, q, opts)
	opts = opts or {}
	local images = opts.images
	if opts.content_free then
		state, images = M.CONTENT_FREE_STATE, nil
	end
	local function with_state(lines)
		return assemble(state, lines, images)
	end
	local instr = M.strip(q.instructions or "")
	local function question(default)
		return "Question: " .. (instr ~= "" and instr or default)
	end
	if q.type == "noul" then
		local lines = { question("Is the statement true?") }
		if q.yes then
			lines[#lines + 1] = "Answer yes if: " .. q.yes
		end
		if q.no then
			lines[#lines + 1] = "Answer no if: " .. q.no
		end
		lines[#lines + 1] = "Allowed answers: yes, no"
		return { prefix = with_state(lines), candidates = { "yes", "no" },
			keys = { "1", "0" }, mode = "yesno" }
	elseif q.type == "score" then
		local lines = { question("Which level applies?"), "Levels (ordered from lowest to highest):" }
		local keys = {}
		for i, d in ipairs(q.levels) do
			keys[i] = tostring(i - 1)
			lines[#lines + 1] = ("%d. %s"):format(i - 1, d)
		end
		lines[#lines + 1] = "Allowed answers: " .. table.concat(keys, ", ")
		return { prefix = with_state(lines), candidates = keys, keys = keys, mode = "digit" }
	end
	local k = #q.options
	local order = opts.seed and M.shuffle(k, opts.seed) or {}
	if not opts.seed then
		for i = 1, k do
			order[i] = i - 1
		end
	end
	local ids = opts.identifiers
	if not ids or #ids < k then
		ids = M.default_identifiers(k)
	end
	local lines = { question("Which option applies?"), "Options:" }
	local cands, keys = {}, {}
	for i = 1, k do
		local o = q.options[order[i] + 1]
		cands[i], keys[i] = ids[i], o.key
		local desc = o.text
		lines[#lines + 1] = ids[i] .. ". " .. o.key ..
		    ((desc ~= "" and desc ~= o.key) and (" — " .. desc) or "")
	end
	lines[#lines + 1] = "Allowed answers: " .. table.concat(cands, ", ")
	return { prefix = with_state(lines), candidates = cands, keys = keys, mode = "index" }
end

--- Wraps a rendered prefix in the chat template, as Jevify's `to_chat`.
-- The system turn is `SYSTEM_PROMPT`. The user turn is the prefix
-- without its last line. The assistant turn is open, with thinking
-- off, and the cue follows it.
-- @tparam table family a chat family from `lgml.chat.load`
-- @tparam string prefix the prefix from `render`
-- @treturn string the prompt text
function M.chat_prompt(family, prefix)
	local tail = "\n" .. M.CUE
	assert(prefix:sub(-#tail) == tail, "prefix must end with the cue")
	local user = prefix:sub(1, #prefix - #tail)
	return family.render({
		{ role = "system", content = M.SYSTEM_PROMPT },
		{ role = "user", content = user },
	}, { add_generation_prompt = true, enable_thinking = false }) .. M.CUE
end

--- Tokenizes a prompt and its candidates jointly, as Jevify's `tokenize`.
-- Each candidate is tokenized with the prompt tail after the last line
-- break, so that a merge across the cue stays as the model sees it.
-- The candidate tokens start after the tokens that all joint encodings
-- share with the prompt. With a candidate `"A"` after `"Answer: "`, the
-- space leaves the prefix and `" A"` is one candidate token.
-- @param tk the tokenizer
-- @tparam string text the prompt
-- @tparam {string,...} cands the candidates
-- @treturn {int,...} the prefix ids
-- @treturn {{int,...},...} the candidate ids, one or more each
function M.tokenize(tk, text, cands)
	local raw = { add_bos = false }
	local p = tk:encode(text)
	local pos, tail, tail_ids = #text, text, {}
	for _ = 1, 4 do
		local cut = 1
		for i = pos - 1, 1, -1 do
			if text:byte(i) == 10 then
				cut = i + 1
				break
			end
		end
		tail = text:sub(cut)
		tail_ids = tk:encode(tail, raw)
		if #tail_ids > 0 and common(p, tail_ids, #p - #tail_ids) == #tail_ids then
			break
		end
		if cut == 1 then
			break
		end
		pos = cut - 1
	end
	local fast = #tail_ids > 0 and common(p, tail_ids, #p - #tail_ids) == #tail_ids
	-- Each joint encoding is p[1..base] followed by rest.
	local fulls = {}
	for i, c in ipairs(cands) do
		local ids = fast and tk:encode(tail .. c, raw)
		local n = ids and common(tail_ids, ids) or 0
		if fast and n > 0 then
			fulls[i] = { base = #p - #tail_ids + n, rest = table.move(ids, n + 1, #ids, 1, {}) }
		else
			fulls[i] = { base = 0, rest = tk:encode(text .. c) }
		end
	end
	local n_common = #p
	local short = false
	for _, f in ipairs(fulls) do
		local n = f.base + common(p, f.rest, f.base)
		n_common = math.min(n_common, n)
		f.len = f.base + #f.rest
	end
	for _, f in ipairs(fulls) do
		short = short or f.len == n_common
	end
	if n_common == #p and short then
		n_common = n_common - 1
	end
	local prefix = table.move(p, 1, n_common, 1, {})
	local out = {}
	for i, f in ipairs(fulls) do
		local c = {}
		for j = n_common + 1, f.len do
			c[#c + 1] = j <= f.base and p[j] or f.rest[j - f.base]
		end
		assert(#c > 0, "every candidate must add a token")
		out[i] = c
	end
	return prefix, out
end

-- recipe

--- Returns the default recipe.
-- The temperatures are 1, the noul bias is 0 and there is no prior
-- correction. Choice averages 2 option orders, as every Tier 0 recipe
-- that Jevify fits on jev-bench does.
-- @treturn table the recipe
function M.default_recipe()
	return {
		mode = "index",
		permutations = { choice = 2, score = 1, noul = 1 },
		temperature = { choice = 1.0, score = 1.0, noul = 1.0 },
		prior_weight = { choice = 0.0, score = 0.0, noul = 0.0 },
		bias = { noul = 0.0 },
		temp_k_slope = { choice = 0.0, score = 0.0, noul = 0.0 },
	}
end

local function per_prim(v, name, check)
	if type(v) == "number" then
		local t = {}
		for _, p in ipairs(PRIMS) do
			t[p] = v
		end
		v = t
	end
	if type(v) ~= "table" then
		return nil, name .. " must be a number or an object"
	end
	for k, x in pairs(v) do
		if type(x) ~= "number" or not check(x) then
			return nil, ("%s.%s is not valid"):format(name, tostring(k))
		end
	end
	return v
end

local CHECKS = {
	permutations = function(x) return math.tointeger(x) and x >= 1 and x <= 8 end,
	temperature = function(x) return x > 0 end,
	prior_weight = function(x) return x >= 0 end,
	bias = function(x) return x == x end,
	temp_k_slope = function(x) return x == x end,
}

--- Merges recipe fields over a base recipe.
-- over has the fields of Jevify's `Recipe` or its `recipe.json`:
-- `permutations`, `temperature`, `prior_weight`, `bias` and
-- `temp_k_slope`, each a number or a table by primitive. A table with a
-- `recipe` field is read as `recipe.json`. `mode` must be `"index"`,
-- `state_last` false and `prompt` `"jevify"` when given.
-- @tparam table base the base recipe
-- @tparam[opt] table over the fields to change
-- @treturn[1] table a new recipe
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.recipe(base, over)
	local r = {}
	for k, v in pairs(base) do
		r[k] = v
		if type(v) == "table" then
			r[k] = {}
			for p, x in pairs(v) do
				r[k][p] = x
			end
		end
	end
	if over == nil then
		return r
	end
	if type(over.recipe) == "table" then
		over = over.recipe
	end
	if over.mode ~= nil and over.mode ~= "index" then
		return nil, "recipe mode must be index"
	elseif over.state_last ~= nil and over.state_last ~= false then
		return nil, "recipe state_last must be false"
	elseif over.prompt ~= nil and over.prompt ~= M.PROMPT_FORMAT then
		return nil, "recipe prompt must be " .. M.PROMPT_FORMAT
	end
	for name, check in pairs(CHECKS) do
		if over[name] ~= nil then
			local v, err = per_prim(over[name], name, check)
			if not v then
				return nil, err
			end
			for p, x in pairs(v) do
				r[name][p] = x
			end
		end
	end
	return r
end

--- Reads a recipe file: a Jevify `recipe.json` or `jevify_config.json`.
-- @tparam string path the file path
-- @treturn[1] table the recipe over `default_recipe`
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.load_recipe(path)
	local f, err = io.open(path, "rb")
	if not f then
		return nil, err
	end
	local text = f:read("a")
	f:close()
	local v, jerr = json.decode(text)
	if type(v) ~= "table" then
		return nil, path .. ": " .. tostring(jerr or "not a JSON object")
	end
	local r, rerr = M.recipe(M.default_recipe(), v)
	if not r then
		return nil, path .. ": " .. rerr
	end
	return r
end

local function temp_for(r, prim, k)
	local t = r.temperature[prim] or 1.0
	local slope = r.temp_k_slope[prim] or 0.0
	if slope ~= 0 and k > 2 then
		t = t + slope * math.log(k / 2, 2)
	end
	return math.max(t, 0.05)
end

local function softmax(s, t)
	t = t or 1.0
	local m = -math.huge
	for _, v in ipairs(s) do
		m = math.max(m, v)
	end
	local ex, z = {}, 0.0
	for i, v in ipairs(s) do
		ex[i] = math.exp((v - m) / t)
		z = z + ex[i]
	end
	for i = 1, #ex do
		ex[i] = ex[i] / z
	end
	return ex
end

M.softmax = softmax

--- Applies a recipe to raw log-scores, as Jevify's `finalize`.
-- Each run is `{ keys, logscores }` for one option order. The noul bias
-- adds to the log-score of `"1"`. Prior correction subtracts `w * log
-- p0`, with p0 the content-free distribution, aligned by position for
-- choice and by key otherwise. Each run gets a softmax at the
-- temperature, and the runs average by key.
-- @tparam table q a question of the engine request
-- @tparam {table,...} runs the runs, presented order first
-- @tparam[opt] table prior the content-free run `{ keys, logscores }`
-- @tparam table recipe the recipe
-- @treturn table `{ yes = P(yes) }` for noul, else `{ probabilities }`
-- in option or level order
function M.finalize(q, runs, prior, recipe)
	local prim = q.type
	local t = temp_for(recipe, prim, #runs[1].keys)
	local pw = recipe.prior_weight[prim] or 0.0
	local bias = prim == "noul" and recipe.bias.noul or 0.0
	local acc, n = {}, 0
	for r = 1, math.min(#runs, math.max(1, recipe.permutations[prim] or 1)) do
		local keys, ls = runs[r].keys, table.move(runs[r].logscores, 1, #runs[r].logscores, 1, {})
		if bias ~= 0 then
			for i, k in ipairs(keys) do
				if k == "1" then
					ls[i] = ls[i] + bias
				end
			end
		end
		if prior and pw > 0 then
			local p0 = {}
			if prim == "choice" then
				p0 = table.move(prior.logscores, 1, #ls, 1, {})
			else
				local by = {}
				for i, k in ipairs(prior.keys) do
					by[k] = prior.logscores[i]
				end
				for i, k in ipairs(keys) do
					p0[i] = by[k]
				end
			end
			p0 = softmax(p0)
			for i = 1, #ls do
				ls[i] = ls[i] - pw * math.log(math.max(p0[i], 1e-12))
			end
		end
		for i, p in ipairs(softmax(ls, t)) do
			acc[keys[i]] = (acc[keys[i]] or 0.0) + p
		end
		n = n + 1
	end
	if prim == "noul" then
		return { yes = acc["1"] / n }
	end
	local probs = {}
	if prim == "choice" then
		for i, o in ipairs(q.options) do
			probs[i] = acc[o.key] / n
		end
	else
		for i = 1, #q.levels do
			probs[i] = acc[tostring(i - 1)] / n
		end
	end
	return { probabilities = probs }
end

-- scoring on a chat engine

local function logsumexp(row, from, n)
	local m = -math.huge
	for i = from, from + n - 1 do
		if row[i] > m then
			m = row[i]
		end
	end
	local z = 0.0
	for i = from, from + n - 1 do
		z = z + math.exp(row[i] - m)
	end
	return m + math.log(z)
end

local function sink_of(ctl)
	return {
		cached = 0, computed = 0,
		cancelled = function()
			return ctl and ctl.cancelled and ctl.cancelled() or false
		end,
	}
end

-- Runs ids so that the engine holds exactly ids. Returns the last
-- logits node, or nil on cancel.
local function advance(e, ids, sink)
	local cached = e:reuse(ids)
	sink.cached, sink.computed = sink.cached + cached, sink.computed + #ids - cached
	return e:prefill(ids, cached + 1, sink)
end

-- Log P(candidate | prefix) for each candidate. Candidates of one token
-- read the last row. Longer ones run their tokens after the prefix.
local function score(e, p, cands, sink)
	local logits = advance(e, p, sink)
	if not logits then
		return nil
	end
	local row = logits:get()
	local nv = #row
	local lse = logsumexp(row, 1, nv)
	local out = {}
	for i, c in ipairs(cands) do
		out[i] = row[c[1] + 1] - lse
	end
	for i, c in ipairs(cands) do
		if #c > 1 then
			local ids = table.move(p, 1, #p, 1, {})
			ids[#ids + 1] = c[1]
			local n = e:reuse(ids)
			if n < #p and not e:prefill(p, n + 1, sink) then
				return nil
			end
			local span = table.move(c, 1, #c - 1, 1, {})
			local rows = e:eval(span, "all"):get()
			table.move(span, 1, #span, #e.tokens + 1, e.tokens)
			local v = #rows // #span
			for j = 2, #c do
				local from = (j - 2) * v + 1
				out[i] = out[i] + rows[from + c[j]] - logsumexp(rows, from, v)
			end
			if sink:cancelled() then
				return nil
			end
		end
	end
	return out
end

-- Scores one rendering. Returns the log-scores and the token counts.
-- keys are the prepared images of the prompt, in order.
local function run_rendering(e, rd, sink, keys)
	local text = M.chat_prompt(e.family, rd.prefix)
	local p, cands = M.tokenize(e.tk, text, rd.candidates)
	if e.vision then
		local err
		p, err = e:expand_images(p, keys or {})
		if not p then
			error({ status = 400, msg = err })
		end
	end
	if #p + 8 >= e.n_ctx then
		error({ status = 400, msg = ("prompt has %d tokens; context is %d"):format(#p, e.n_ctx) })
	end
	local ls = score(e, p, cands, sink)
	if not ls then
		return nil
	end
	local nout = 0
	for _, c in ipairs(cands) do
		nout = nout + #c
	end
	return { keys = rd.keys, logscores = ls }, #p, nout
end

local PRIOR_CACHE = 256

-- The content-free run of a question, cached by prompt text.
local function prior_run(e, q, ids, sink)
	local rd = M.render("", q, { content_free = true, identifiers = ids })
	e.decide_priors = e.decide_priors or { n = 0, runs = {} }
	local c = e.decide_priors
	local run = c.runs[rd.prefix]
	if run then
		return run
	end
	run = run_rendering(e, rd, sink)
	if run then
		if c.n >= PRIOR_CACHE then
			c.n, c.runs = 0, {}
		end
		c.n = c.n + 1
		c.runs[rd.prefix] = run
	end
	return run
end

--- Answers the questions of a request on a chat engine.
-- e is an `lgml.engine` engine that holds the slot. It uses `e.tk`,
-- `e.family`, `e.n_ctx`, `e.tokens`, `e.reuse`, `e.prefill` and
-- `e.eval`, and the recipe `e.recipe`. Questions run in request order;
-- the state comes first in each prompt, so a question reuses the KV
-- rows of the state.
--
-- The images of `req.images` go at the start of the state, in order,
-- before the state text. So every question and option order on one
-- state shares the image rows through the cache. For images, e also
-- needs `e.vision`, `e.image_marker`, `e.prepare_images` and
-- `e.expand_images`. The content-free prompt of prior correction has
-- no images: the prior measures the bias of the question and its
-- answers alone, and one prior per question serves every state.
-- @param e the engine
-- @tparam table req the engine request (see `lgml.systemone.check`)
-- @tparam[opt] table ctl the cancel control
-- @treturn[1] table the result: `{ model, answers, usage, timings }`.
-- timings has `images`, `image_encode_ms` (encoder time of this
-- request), `prompt_n` (prompt positions computed) and `cache_n`
-- (positions reused from the cache).
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function M.run(e, req, ctl)
	local recipe, err = M.recipe(e.recipe or M.default_recipe(), req.recipe)
	if not recipe then
		return nil, err, 422
	end
	local images = req.images or {}
	local keys, markers
	if #images > 0 then
		if not e.vision then
			return nil, "image input is not supported: start the server with --mmproj", 400
		end
		local kerr
		keys, kerr = e:prepare_images(images)
		if not keys then
			return nil, kerr, 400
		end
		markers = string.rep(e.image_marker, #images)
	end
	e.decide_ids = e.decide_ids or M.identifiers(e.tk)
	local ids = e.decide_ids
	local sink = sink_of(ctl)
	local answers, nin, nout = {}, 0, 0
	local ok, res = pcall(function()
		for _, q in ipairs(req.questions) do
			local n = q.type == "choice" and recipe.permutations.choice or 1
			local runs = {}
			for i = 1, n do
				local rd = M.render(req.state, q, {
					identifiers = ids, seed = i > 1 and M.SEED + i - 1 or nil,
					images = markers,
				})
				local run, a, b = run_rendering(e, rd, sink, keys)
				if not run then
					return "cancelled"
				end
				runs[i], nin, nout = run, nin + a, nout + b
			end
			local prior
			if (recipe.prior_weight[q.type] or 0) > 0 then
				prior = prior_run(e, q, ids, sink)
				if not prior then
					return "cancelled"
				end
			end
			local a = M.finalize(q, runs, prior, recipe)
			if req.raw then
				a.lgml = { runs = runs, prior = prior }
			end
			answers[q.id] = a
		end
	end)
	if not ok then
		if type(res) == "table" and res.status then
			return nil, res.msg, res.status
		end
		error(res, 0)
	elseif res == "cancelled" then
		return nil, "cancelled", 499
	end
	return {
		model = e.id, answers = answers,
		usage = { input_tokens = nin, output_tokens = nout },
		timings = {
			images = #images, image_encode_ms = e.image_ms or 0,
			prompt_n = sink.computed, cache_n = sink.cached,
		},
	}
end

return M
