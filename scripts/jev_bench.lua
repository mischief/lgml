-- jev-bench evaluator for a running lgml-server (/v1/systemone).
-- usage: lua5.4 scripts/jev_bench.lua [-u url] [-d dir] [-n test] [-v val]
--            [-c config,...] [-r recipe.json] [-o out.json] [-s seed]
-- dir has data/<config>/{test,validation}.jsonl of Praveenrajus/jev-bench.
-- -v seeded validation rows (default 50) fit the Tier 0 recipe (-r skips
-- it); -n seeded test rows (default 200) are scored. "raw": T=1, one order.

local uv = require("luv")
local json = require("lgml.json")
local so = require("lgml.systemone")
local decide = require("lgml.decide")
local stream = require("lgml.server.stream")
local getopt = require("lgml.getopt")

local USAGE = "usage: jev_bench.lua [-u url] [-d dir] [-n test] [-v val] " ..
    "[-c config,...] [-r recipe.json] [-o out.json] [-s seed]\n"

local CONFIGS = {
	"arc_challenge", "banking77", "chaosnli", "clinc150", "go_emotions", "ledgar",
	"massive", "mmlu", "mnli", "helpsteer2_helpfulness", "helpsteer2_verbosity",
	"measuring_hate_speech", "sst5", "stsb", "yelp5", "boolq", "civil_comments",
	"fever_evidence", "paws", "sms_spam", "strategyqa_closed", "strategyqa_grounded",
}
local PRIMS = { "choice", "score", "noul" }
-- Each request gets at most this long; closing the connection cancels it.
local LIMIT_MS = 60000

-- HTTP client with one keep-alive connection.

local Client = {}
Client.__index = Client

local function client(url)
	local host, port = url:match("^http://([^:/]+):?(%d*)")
	assert(host, "url must be http://host[:port]")
	return setmetatable({ host = host, port = tonumber(port) or 80, buf = "" }, Client)
end

function Client:connect()
	local tcp = uv.new_tcp()
	local err = stream.await(function(done)
		tcp:connect(self.host, self.port, done)
	end)
	assert(not err, err)
	self.s, self.buf = stream.new(tcp), ""
end

function Client:fill()
	local left = self.deadline - uv.now()
	local d = left > 0 and self.s:read(left)
	if not d then
		return nil
	end
	self.buf = self.buf .. d
	return true
end

function Client:once(text)
	self.deadline = uv.now() + LIMIT_MS
	if not self.s then
		self:connect()
	end
	self.s:write(("POST /v1/systemone HTTP/1.1\r\nHost: %s\r\nContent-Type: application/json\r\n" ..
	    "Content-Length: %d\r\n\r\n%s"):format(self.host, #text, text))
	while not self.buf:find("\r\n\r\n", 1, true) do
		if not self:fill() then
			return nil
		end
	end
	local head, rest = self.buf:match("^(.-)\r\n\r\n(.*)$")
	local status = tonumber(head:match("^HTTP/1%.%d (%d+)"))
	local len = tonumber(head:lower():match("\r\ncontent%-length: *(%d+)"))
	while #rest < len do
		if not self:fill() then
			return nil
		end
		rest = self.buf:match("^.-\r\n\r\n(.*)$")
	end
	self.buf = rest:sub(len + 1)
	if head:lower():find("\r\nconnection: *close") then
		self.s:close()
		self.s = nil
	end
	return status, rest:sub(1, len)
end

-- Posts a body; on a lost connection, reconnects and tries once more.
-- Returns nil when the time limit ends.
function Client:post(text)
	local status, body = self:once(text)
	if not status and uv.now() >= self.deadline then
		if self.s then
			self.s:close()
		end
		self.s = nil
		return nil
	elseif not status then
		if self.s then
			self.s:close()
		end
		self.s = nil
		status, body = self:once(text)
	end
	return status, body
end

-- data

-- A sample of limit rows, the same for a given seed.
local function rows(path, limit, seed)
	local f = io.open(path, "rb")
	if not f then
		return nil
	end
	local lines = {}
	for line in f:lines() do
		if line:find("%S") then
			lines[#lines + 1] = line
		end
	end
	f:close()
	local out = {}
	for i, j in ipairs(decide.shuffle(#lines, seed)) do
		if i > limit then
			break
		end
		out[i] = assert(json.decode(lines[j + 1]))
	end
	return out
end

local function request_body(row, extra)
	return '{"state": ' .. row.state .. ', "questions": {"q": ' .. row.question .. '}, "lgml": ' ..
	    assert(json.encode(extra)) .. "}"
end

-- The option keys of a question in request order, and the label index.
local function label_of(row, q)
	local keys = {}
	if q.type == "choice" then
		for i, o in ipairs(q.options) do
			keys[i] = o.key
		end
	elseif q.type == "score" then
		for i = 1, #q.levels do
			keys[i] = tostring(i - 1)
		end
	else
		keys = { "0", "1" }
	end
	for i, k in ipairs(keys) do
		if k == tostring(row.label) then
			return i
		end
	end
	error("label " .. tostring(row.label) .. " not in options of " .. row.id)
end

-- metrics, as jevify.bench.metrics

local function metrics(items)
	local nb = 15
	local bins = {}
	for b = 1, nb do
		bins[b] = { n = 0, acc = 0, conf = 0 }
	end
	local n, acc, brier, nll = #items, 0, 0, 0
	for _, it in ipairs(items) do
		local p, y = it.p, it.y
		local conf, correct, ptrue
		if it.noul then
			local py = p[1]
			correct = ((py >= 0.5) and 2 or 1) == y
			conf = math.max(py, 1 - py)
			ptrue = y == 2 and py or 1 - py
			brier = brier + (py - (y - 1)) ^ 2
		else
			local best = 1
			for i = 2, #p do
				if p[i] > p[best] then
					best = i
				end
			end
			correct, conf, ptrue = best == y, p[best], p[y]
			for i, v in ipairs(p) do
				brier = brier + (v - (i == y and 1 or 0)) ^ 2
			end
		end
		acc = acc + (correct and 1 or 0)
		nll = nll - math.log(math.max(ptrue, 1e-12))
		-- Bin 1 is [0, 1/15]; bin b is ((b-1)/15, b/15].
		local b = math.max(1, math.min(nb, math.ceil(conf * nb - 1e-12)))
		bins[b].n = bins[b].n + 1
		bins[b].acc = bins[b].acc + (correct and 1 or 0)
		bins[b].conf = bins[b].conf + conf
	end
	local ece = 0
	for _, b in ipairs(bins) do
		if b.n > 0 then
			ece = ece + b.n / n * math.abs(b.acc / b.n - b.conf / b.n)
		end
	end
	return { n = n, acc = acc / n, ece = ece, brier = brier / n, nll = nll / n }
end

-- recipe fit, as jevify.runners.recipe.fit_recipe

local function nll_of(ls, ys, t)
	local s = 0
	for i, l in ipairs(ls) do
		s = s - math.log(math.max(decide.softmax(l, t)[ys[i]], 1e-12))
	end
	return s / math.max(#ls, 1)
end

local function fit_temperature(ls, ys)
	local a, b = math.log(0.05), math.log(20.0)
	local phi = (math.sqrt(5) - 1) / 2
	local c, d = b - phi * (b - a), a + phi * (b - a)
	local fc, fd = nll_of(ls, ys, math.exp(c)), nll_of(ls, ys, math.exp(d))
	for _ = 1, 60 do
		if fc < fd then
			b, d, fd = d, c, fc
			c = b - phi * (b - a)
			fc = nll_of(ls, ys, math.exp(c))
		else
			a, c, fc = c, d, fd
			d = a + phi * (b - a)
			fd = nll_of(ls, ys, math.exp(d))
		end
	end
	return math.exp((a + b) / 2)
end

local function fit_temperature_and_bias(ls, ys)
	local best = { math.huge, 1.0, 0.0 }
	for bi = -40, 40 do
		local bias = bi / 10
		local shifted = {}
		for i, l in ipairs(ls) do
			shifted[i] = { l[1], l[2] + bias }
		end
		local t = fit_temperature(shifted, ys)
		local v = nll_of(shifted, ys, t)
		if v < best[1] then
			best = { v, t, bias }
		end
	end
	return best[2], best[3], best[1]
end

local function unit_recipe(perm, pw)
	return assert(decide.recipe(decide.default_recipe(), {
		permutations = { choice = perm, score = 1, noul = 1 }, prior_weight = pw,
	}))
end

-- Log probabilities in option order under a recipe at temperature 1.
local function logprobs(it, rc)
	local a = decide.finalize(it.q, it.runs, it.prior, rc)
	local p = it.q.type == "noul" and { 1 - a.yes, a.yes } or a.probabilities
	local out = {}
	for i, v in ipairs(p) do
		out[i] = math.log(math.max(v, 1e-12))
	end
	return out
end

local function fit(items)
	local chosen, log = {}, {}
	for _, perm in ipairs({ 1, 2 }) do
		for _, pw in ipairs({ 0.0, 1.0 }) do
			local rc = unit_recipe(perm, pw)
			for _, prim in ipairs(PRIMS) do
				local ls, ys = {}, {}
				for _, it in ipairs(items) do
					if it.q.type == prim then
						ls[#ls + 1] = logprobs(it, rc)
						ys[#ys + 1] = it.y
					end
				end
				if #ys > 0 then
					local t, bias, v = nil, 0.0, nil
					if prim == "noul" then
						t, bias, v = fit_temperature_and_bias(ls, ys)
					else
						t = fit_temperature(ls, ys)
						v = nll_of(ls, ys, t)
					end
					log[#log + 1] = ("%s perm %d prior %g: T %.4f bias %.1f nll %.4f (raw %.4f, n %d)")
					    :format(prim, perm, pw, t, bias, v, nll_of(ls, ys, 1.0), #ys)
					if not chosen[prim] or v < chosen[prim].nll then
						chosen[prim] = { nll = v, perm = perm, pw = pw, t = t, bias = bias }
					end
				end
			end
		end
	end
	local r = {
		mode = "index", chat = true, state_last = false,
		permutations = { choice = 1, score = 1, noul = 1 },
		prior_weight = { choice = 0.0, score = 0.0, noul = 0.0 },
		temperature = { choice = 1.0, score = 1.0, noul = 1.0 },
		bias = { noul = 0.0 },
	}
	for prim, c in pairs(chosen) do
		r.permutations[prim] = prim == "choice" and c.perm or 1
		r.prior_weight[prim] = c.pw
		r.temperature[prim] = math.floor(c.t * 1e4 + 0.5) / 1e4
		if prim == "noul" then
			r.bias.noul = c.bias
		end
	end
	return r, log
end

-- runs

local function ask(cl, row, extra)
	local text = request_body(row, extra)
	local req = assert(so.check(text), "bad record " .. row.id)
	local status, body = cl:post(text)
	if not status then
		io.stderr:write(("\n%s: no answer in %d s, skipped\n"):format(row.id, LIMIT_MS // 1000))
		return nil
	elseif status ~= 200 then
		error(("%s: status %d: %s"):format(row.id, status, body))
	end
	local b = assert(json.decode(body))
	local q = req.questions[1]
	local a = b.answers.q
	local it = { q = q, y = label_of(row, q), noul = q.type == "noul", usage = b.usage }
	if a.lgml then
		it.runs, it.prior = a.lgml.runs, a.lgml.prior
	end
	if q.type == "noul" then
		it.p = { a.noul }
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
		it.p = {}
		for i, k in ipairs(keys) do
			it.p[i] = a.probabilities[k]
		end
	end
	return it
end

local function progress(what, cfg, i, total, t0)
	if i % 25 == 0 or i == total then
		io.stderr:write(("\r%s %s %d/%d  %.1f s"):format(what, cfg, i, total, uv.hrtime() / 1e9 - t0))
		if i == total then
			io.stderr:write("\n")
		end
	end
end

local function main(args)
	local opts, rest = getopt.parse(args, "u:d:n:v:c:r:o:s:h", {})
	if not opts or #rest > 0 or opts.h then
		io.stderr:write(USAGE)
		os.exit(opts and opts.h and 0 or 2)
	end
	local dir = opts.d or "build/jev-bench"
	local ntest, nval = tonumber(opts.n or 200), tonumber(opts.v or 50)
	local seed = math.tointeger(tonumber(opts.s or 20260920))
	local configs = CONFIGS
	if opts.c then
		configs = {}
		for c in opts.c:gmatch("[^,]+") do
			configs[#configs + 1] = c
		end
	end
	local cl = client(opts.u or "http://127.0.0.1:8080")
	local t0 = uv.hrtime() / 1e9
	local recipe, log
	local skipped = 0
	if opts.r then
		local f = assert(io.open(opts.r, "rb"))
		recipe = assert(json.decode(f:read("a")))
		f:close()
		recipe = recipe.recipe or recipe
	else
		local items = {}
		local extra = { raw = true, recipe = { permutations = { choice = 2 }, prior_weight = 1,
			temperature = 1, bias = { noul = 0 } } }
		for _, cfg in ipairs(configs) do
			local vs = rows(("%s/data/%s/validation.jsonl"):format(dir, cfg), nval, seed) or {}
			for i, row in ipairs(vs) do
				local it = ask(cl, row, extra)
				if it then
					items[#items + 1] = it
				else
					skipped = skipped + 1
				end
				progress("fit", cfg, i, #vs, t0)
			end
		end
		recipe, log = fit(items)
		io.stderr:write(table.concat(log, "\n"), "\n")
	end
	io.stdout:write("recipe: ", so.dumps(recipe), "\n")
	local results, tokens = {}, 0
	local sent = { raw = true, recipe = recipe }
	for _, cfg in ipairs(configs) do
		local ts = rows(("%s/data/%s/test.jsonl"):format(dir, cfg), ntest, seed)
		if ts and #ts > 0 then
			local fitted, raw, prim = {}, {}, nil
			local unit = unit_recipe(1, 0)
			for i, row in ipairs(ts) do
				local it = ask(cl, row, sent)
				if it then
					prim = it.q.type
					tokens = tokens + it.usage.input_tokens
					fitted[#fitted + 1] = it
					local p = {}
					for j, v in ipairs(logprobs(it, unit)) do
						p[j] = math.exp(v)
					end
					raw[#raw + 1] = { p = it.noul and { p[2] } or p, y = it.y, noul = it.noul }
				else
					skipped = skipped + 1
				end
				progress("test", cfg, i, #ts, t0)
			end
			if #fitted > 0 then
				results[#results + 1] = { cfg = cfg, prim = prim, m = metrics(fitted), raw = metrics(raw) }
			end
		end
	end
	local out = {}
	local function say(s)
		io.stdout:write(s, "\n")
		out[#out + 1] = s
	end
	say("| config | prim | n | acc | ECE | Brier | NLL | raw acc | raw ECE |")
	say("|---|---|---|---|---|---|---|---|---|")
	local sums = { all = { n = 0, acc = 0, ece = 0, brier = 0, racc = 0, rece = 0 } }
	for _, p in ipairs(PRIMS) do
		sums[p] = { n = 0, acc = 0, ece = 0, brier = 0, racc = 0, rece = 0 }
	end
	for _, r in ipairs(results) do
		local m = r.m
		say(("| %s | %s | %d | %.3f | %.3f | %.3f | %.2f | %.3f | %.3f |"):format(
		    r.cfg, r.prim, m.n, m.acc, m.ece, m.brier, m.nll, r.raw.acc, r.raw.ece))
		for _, k in ipairs({ "all", r.prim }) do
			local s = sums[k]
			s.n, s.acc, s.ece, s.brier = s.n + 1, s.acc + m.acc, s.ece + m.ece, s.brier + m.brier
			s.racc, s.rece = s.racc + r.raw.acc, s.rece + r.raw.ece
		end
	end
	for _, k in ipairs({ "all", "choice", "score", "noul" }) do
		local s = sums[k]
		if s.n > 0 then
			say(("| **macro %s** | | %d configs | %.3f | %.3f | %.3f | | %.3f | %.3f |"):format(
			    k, s.n, s.acc / s.n, s.ece / s.n, s.brier / s.n, s.racc / s.n, s.rece / s.n))
		end
	end
	say(("%d input tokens, %d records skipped, %.0f s"):format(tokens, skipped,
	    uv.hrtime() / 1e9 - t0))
	if opts.o then
		local f = assert(io.open(opts.o, "wb"))
		f:write(assert(json.encode({ recipe = recipe, fit = log, table = out })), "\n")
		f:close()
	end
	if cl.s then
		cl.s:close()
	end
end

local co = coroutine.create(main)
local good, err = coroutine.resume(co, arg)
if not good then
	error(debug.traceback(co, err))
end
uv.run()
