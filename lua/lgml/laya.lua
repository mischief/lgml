--- Laya typed decisions on lgml: a ModernBERT encoder and a decision head.
-- Laya (convaiinnovations/laya) answers Jev-style typed questions
-- (`noul`, `choice`, `score`) in one encoder pass each. Each option gets
-- a `[MASK]` marker token in the prompt. The head scores the hidden
-- state at each marker, and a softmax over the options, with a
-- temperature by question type and option count, gives the answer.
-- This module follows the reference `rl_common.py` and `rl_agent_api.py`
-- of the model repository: prompt layout, marker positions, head and
-- calibration.
-- @module lgml.laya
local lgml = require("lgml")
local json = require("lgml.json")
local modernbert = require("lgml.modernbert")

local M = {}
M.__index = M

--- Question type ids, as the head's type embedding counts them.
M.QTYPES = { choice = 0, score = 1, noul = 2 }

--- Most tokens of one option text after its marker (48).
M.OPTION_TOKENS = 48

local NOUL_NO = "no, the statement does not hold"
local NOUL_YES = "yes, the statement holds"

local function read_json(path)
	local f = assert(io.open(path, "rb"))
	local s = f:read("a")
	f:close()
	return assert(json.decode(s))
end

--- The option texts of a question, in answer order.
-- q is a question of `lgml.systemone.check`. A choice option is its
-- name, or `name: description`. A score level is `level i: text`, from
-- 0. A noul has `false: ...` then `true: ...`, so the second
-- probability is P(yes).
-- @tparam table q the question
-- @treturn {string,...} the option texts
function M.options(q)
	local out = {}
	if q.type == "choice" then
		for i, o in ipairs(q.options) do
			out[i] = (o.text == nil or o.text == "") and o.key or o.key .. ": " .. o.text
		end
	elseif q.type == "score" then
		for i, l in ipairs(q.levels) do
			out[i] = ("level %d: %s"):format(i - 1, l)
		end
	else
		out[1] = "false: " .. (q.no or NOUL_NO)
		out[2] = "true: " .. (q.yes or NOUL_YES)
	end
	return out
end

--- The temperature bucket of a question type and option count.
-- @tparam string t the question type
-- @tparam int k the option count
-- @treturn string such as `"choice:3-5"`
function M.bucket(t, k)
	local size = k <= 2 and "2" or k <= 5 and "3-5" or k <= 10 and "6-10" or "11+"
	return t .. ":" .. size
end

--- The calibration temperature of a question.
-- It is `temperature_by_options[bucket]` of the config, else
-- `temperature[type]`, else 1.
-- @tparam table cfg the decoded rl_agent_config.json
-- @tparam string t the question type
-- @tparam int k the option count
-- @treturn number the temperature
function M.temperature(cfg, t, k)
	local by = type(cfg.temperature_by_options) == "table" and cfg.temperature_by_options or {}
	local per = type(cfg.temperature) == "table" and cfg.temperature or {}
	return by[M.bucket(t, k)] or per[M.QTYPES[t] + 1] or 1.0
end

--- Softmax of logits divided by a temperature.
-- @tparam {number,...} z the logits
-- @tparam[opt=1] number t the temperature
-- @treturn {number,...} the probabilities
function M.softmax(z, t)
	t = t or 1
	local max, p, sum = -math.huge, {}, 0
	for _, v in ipairs(z) do
		max = math.max(max, v / t)
	end
	for i, v in ipairs(z) do
		p[i] = math.exp(v / t - max)
		sum = sum + p[i]
	end
	for i = 1, #p do
		p[i] = p[i] / sum
	end
	return p
end

--- Laya's confidence: 1 minus the normalized entropy.
-- @tparam {number,...} p the probabilities
-- @treturn number the confidence, from 0 to 1
function M.confidence(p)
	local k = #p
	if k < 2 then
		return 1.0
	end
	local ent = 0
	for _, v in ipairs(p) do
		v = math.min(math.max(v, 1e-12), 1)
		ent = ent - v * math.log(v)
	end
	return 1 - ent / math.log(k)
end

local function slice(t, n)
	local out = {}
	for i = 1, math.min(#t, math.max(n, 0)) do
		out[i] = t[i]
	end
	return out
end

local function append(dst, src)
	table.move(src, 1, #src, #dst + 1, dst)
	return dst
end

local function lin(get, p)
	return { w = assert(get(p .. ".weight"), "missing tensor " .. p .. ".weight"), b = get(p .. ".bias") }
end

--- Makes a model from its parts.
-- get(name) returns a tensor or nil, with the names of the Laya
-- safetensors file: `encoder.*` for ModernBERT, `type_emb`, `head.*`,
-- `scorer.*` and `act_head.*`.
-- @param be the lgml backend of the tensors
-- @tparam table cfg the decoded rl_agent_config.json
-- @tparam table ecfg the decoded encoder config.json
-- @param tk the tokenizer (`lgml.tokenizer`) of tokenizer.json
-- @tparam function get the tensor lookup
-- @treturn Laya the model
-- @raise when a tensor or a special token is missing
function M.new(be, cfg, ecfg, tk, get)
	local hp = modernbert.config(ecfg)
	local enc = modernbert.new(be, hp, modernbert.weights(get, hp, "encoder."))
	local head = {}
	for i = 0, (cfg.head_layers or 2) - 1 do
		local p = "head.layers." .. i .. "."
		head[i + 1] = {
			norm1 = lin(get, p .. "norm1"), norm2 = lin(get, p .. "norm2"),
			qkv = { w = assert(get(p .. "self_attn.in_proj_weight"), "missing head attention"),
				b = get(p .. "self_attn.in_proj_bias") },
			o = lin(get, p .. "self_attn.out_proj"),
			ff1 = lin(get, p .. "linear1"), ff2 = lin(get, p .. "linear2"),
		}
	end
	local sp = tk:special().tokens
	local m = setmetatable({
		be = be, cfg = cfg, hp = hp, enc = enc, tk = tk, head = head,
		type_emb = assert(get("type_emb.weight"), "missing type_emb"),
		scorer = { norm = lin(get, "scorer.0"), fc1 = lin(get, "scorer.1"), fc2 = lin(get, "scorer.3") },
		cls = assert(sp["[CLS]"], "tokenizer has no [CLS]"),
		sep = assert(sp["[SEP]"], "tokenizer has no [SEP]"),
		mask = assert(sp["[MASK]"], "tokenizer has no [MASK]"),
		max_len = cfg.max_len or 512, head_max_len = cfg.head_max_len or 192,
	}, M)
	-- The act head is small; it runs on the host.
	local a0, a2 = get("act_head.0.weight"), get("act_head.2.weight")
	if a0 and a2 then
		m.act = {
			w1 = a0:get(), b1 = assert(get("act_head.0.bias")):get(), n1 = a0:shape()[1],
			w2 = a2:get(), b2 = assert(get("act_head.2.bias")):get(), n2 = a2:shape()[1],
		}
	end
	return m
end

--- Loads a Laya model directory.
-- The directory holds rl_agent_config.json, encoder/config.json,
-- tokenizer/tokenizer.json and model.safetensors, as in the root of
-- convaiinnovations/laya and its typed-decisions subfolder.
-- @param be the lgml backend
-- @tparam string dir the model directory
-- @treturn Laya the model; its `store` field holds the weights
-- @raise when a file or a tensor is missing
function M.load(be, dir)
	dir = dir:gsub("/+$", "")
	local cfg = read_json(dir .. "/rl_agent_config.json")
	local ecfg = read_json(dir .. "/encoder/config.json")
	local tk = assert(lgml.tokenizer(dir .. "/tokenizer/tokenizer.json"))
	local store = assert(be:load_safetensors(dir .. "/model.safetensors", { f32_1d = true }))
	local m = M.new(be, cfg, ecfg, tk, function(name)
		return store:get(name)
	end)
	m.store, m.dir = store, dir
	return m
end

--- True when dir looks like a Laya model directory.
-- @tparam string dir the directory
-- @treturn boolean
function M.is_model(dir)
	local f = io.open(dir .. "/rl_agent_config.json", "rb")
	if f then
		f:close()
	end
	return f ~= nil
end

--- A Laya model, from `new` or `load`.
-- Fields: `be`, `cfg` (rl_agent_config.json), `hp` (encoder
-- hyperparameters), `enc` (`lgml.modernbert`), `tk`, `max_len`,
-- `head_max_len`, and from `load`, `store` and `dir`.
-- @type Laya

--- Closes the weights and the tokenizer.
function M:close()
	self.enc:close()
	if self.store then
		self.store:close()
		self.store = nil
	end
	self.tk:close()
end

local function encode(self, s)
	return self.tk:encode(s, { add_bos = false })
end

--- Builds the token sequence of one question.
-- The layout is `[CLS] <type> question: <instructions> [SEP] [MASK]
-- opt1 [MASK] opt2 ... [SEP] <state> [SEP]`. `[MASK]` in the texts
-- becomes a space. The options share `head_max_len` tokens; the state
-- gets the rest of `max_len`, and its end is cut.
-- @tparam string state the state text
-- @tparam table q the question, as `lgml.systemone.check` gives it
-- @treturn {int,...} the token ids
-- @treturn {int,...} the marker positions, from 0, one per option that fits
-- @treturn {string,...} the option texts
function M:sequence(state, q)
	local function plain(s)
		return (s:gsub("%[MASK%]", " "))
	end
	local opts = M.options(q)
	local head = encode(self, ("%s question: %s"):format(q.type, plain(q.instructions or "")))
	local opt_ids, total = {}, 0
	for i, o in ipairs(opts) do
		opt_ids[i] = append({ self.mask }, slice(encode(self, " " .. plain(o)), M.OPTION_TOKENS))
		total = total + #opt_ids[i]
	end
	local budget = self.head_max_len - total
	if budget < 16 then
		local per = math.max(4, (self.head_max_len - 16) // math.max(1, #opt_ids))
		total = 0
		for i, o in ipairs(opt_ids) do
			opt_ids[i] = slice(o, per)
			total = total + #opt_ids[i]
		end
		budget = self.head_max_len - total
	end
	local ids = append(append({ self.cls }, slice(head, math.max(8, budget))), { self.sep })
	local markers = {}
	for _, o in ipairs(opt_ids) do
		markers[#markers + 1] = #ids
		append(ids, o)
	end
	ids[#ids + 1] = self.sep
	local room = math.max(0, self.max_len - #ids - 1)
	append(ids, slice(encode(self, plain(state)), room))
	ids[#ids + 1] = self.sep
	ids = slice(ids, self.max_len)
	local kept = {}
	for _, p in ipairs(markers) do
		if p < self.max_len then
			kept[#kept + 1] = p
		end
	end
	return ids, kept, opts
end

local function linear(g, x, p)
	local y = g:mul_mat(p.w, x, true)
	if p.b then
		y = g:add(y, p.b)
	end
	return y
end

local function layer_norm(g, x, p, eps)
	local y = g:mul(g:norm(x, eps), p.w)
	if p.b then
		y = g:add(y, p.b)
	end
	return y
end

-- One PyTorch TransformerEncoderLayer with norm_first and ReLU.
local function head_layer(self, g, x, L, n)
	local d = self.hp.d
	local H = math.max(1, d // 64)
	local D = d // H
	local qkv = linear(g, layer_norm(g, x, L.norm1, 1e-5), L.qkv)
	local row = lgml.row_size("f32", 3 * d)
	local function part(j)
		return g:view(qkv, { n, H, D }, { row, lgml.row_size("f32", D) }, j * lgml.row_size("f32", d))
	end
	local o = modernbert.attend(g, g:cont(part(0)), g:cont(part(1)), part(2))
	x = g:add(x, linear(g, o, L.o))
	local f = linear(g, g:relu(linear(g, layer_norm(g, x, L.norm2, 1e-5), L.ff1)), L.ff2)
	return g:add(x, f)
end

--- Adds one question to a graph.
-- @param g the graph
-- @tparam table s the sequence: `ids`, `markers` (from 0) and `type`
-- @treturn tensor the marker logits (k, 1)
-- @treturn tensor the hidden state of the first token (1, d), for the act head
-- @treturn table the leaves to close after the compute
function M:forward(g, s)
	local be = self.be
	local inp = self.enc:inputs(s.ids)
	local qt = be:tensor("i32", { 1 }, { M.QTYPES[s.type] })
	local mk = be:tensor("i32", { #s.markers }, s.markers)
	local n = inp.n
	local h = self.enc:forward(g, inp)
	h = g:add(h, g:get_rows(self.type_emb, qt))
	for _, L in ipairs(self.head) do
		h = head_layer(self, g, h, L, n)
	end
	local m = g:get_rows(h, mk)
	local sc = self.scorer
	local z = linear(g, g:gelu_erf(linear(g, layer_norm(g, m, sc.norm, 1e-5), sc.fc1)), sc.fc2)
	local pooled = g:cont(g:view(h, { 1, self.hp.d }, nil, 0))
	return z, pooled, { inp.ids, inp.pos, qt, mk }
end

local function gelu(x)
	-- Exact GELU with erf; Lua has no erf, so use the Abramowitz-Stegun 7.1.26 form.
	local z = math.abs(x) / math.sqrt(2)
	local t = 1 / (1 + 0.3275911 * z)
	local y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
	    0.254829592) * t * math.exp(-z * z)
	local erf = x >= 0 and y or -y
	return 0.5 * x * (1 + erf)
end

--- The act head: the probability to act rather than escalate.
-- Its input is the pooled state and features of the raw answer
-- distribution: top probability, top margin, normalized entropy and
-- option count / 255.
-- @tparam {number,...} pooled the first hidden state after the head layers
-- @tparam {number,...} z the raw marker logits
-- @treturn ?number the probability of the first act, nil without an act head
function M:act_probability(pooled, z)
	local a = self.act
	if not a then
		return nil
	end
	local p = M.softmax(z)
	local k = math.max(#p, 2)
	local sorted = table.move(p, 1, #p, 1, {})
	table.sort(sorted, function(x, y)
		return x > y
	end)
	local ent = 0
	for _, v in ipairs(p) do
		ent = ent - v * math.log(math.max(v, 1e-9))
	end
	local x = table.move(pooled, 1, #pooled, 1, {})
	append(x, { sorted[1], sorted[1] - (sorted[2] or 0), ent / math.log(k), k / 255 })
	local nin = #x
	local h = {}
	for j = 1, a.n1 do
		local s, base = a.b1[j], (j - 1) * nin
		for i = 1, nin do
			s = s + a.w1[base + i] * x[i]
		end
		h[j] = gelu(s)
	end
	local out = {}
	for j = 1, a.n2 do
		local s, base = a.b2[j], (j - 1) * a.n1
		for i = 1, a.n1 do
			s = s + a.w2[base + i] * h[i]
		end
		out[j] = s
	end
	return M.softmax(out)[1]
end

--- Answers the questions about a state, in one graph.
-- Each result has `type`, `probabilities` (calibrated, in option order),
-- `logits` (raw), `temperature`, `confidence` (Laya's rule),
-- `act_probability` and `n_tokens`.
-- @tparam string state the state text
-- @tparam {table,...} questions the questions, as `lgml.systemone.check` gives them
-- @treturn[1] {table,...} the results, in question order
-- @treturn[2] nil
-- @treturn[2] string an error message, when the options do not fit
function M:decide(state, questions)
	local seqs = {}
	for i, q in ipairs(questions) do
		local ids, markers, opts = self:sequence(state, q)
		if #markers ~= #opts then
			return nil, ("question %s: options do not fit in head_max_len=%d tokens"):format(
			    q.id or i, self.head_max_len)
		end
		seqs[i] = { ids = ids, markers = markers, type = q.type }
	end
	local g <close> = self.be:graph()
	local outs, leaves = {}, {}
	for i, s in ipairs(seqs) do
		local z, pooled, lv = self:forward(g, s)
		outs[#outs + 1] = z
		outs[#outs + 1] = pooled
		seqs[i].z, seqs[i].pooled = z, pooled
		append(leaves, lv)
	end
	g:compute(table.unpack(outs))
	local res = {}
	for i, s in ipairs(seqs) do
		local z = s.z:get()
		local t = M.temperature(self.cfg, s.type, #z)
		local p = M.softmax(z, t)
		res[i] = {
			type = s.type, probabilities = p, logits = z, temperature = t,
			confidence = M.confidence(p), n_tokens = #s.ids,
			act_probability = self:act_probability(s.pooled:get(), z),
		}
	end
	for _, t in ipairs(leaves) do
		t:close()
	end
	return res
end

return M
