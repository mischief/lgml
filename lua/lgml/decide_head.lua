--- Decision heads on hidden states, for System One models whose
-- answers are not token logits. The weights follow the llama.cpp
-- conversion, and the head math runs in a ggml graph on the model
-- backend.
--
-- Kev (`<arch>.decision.type = "kev"`, tensor `cls.output`) is a
-- pointer head. Each question is a row that continues the shared
-- state: `<|fim_prefix|>` state `<|fim_middle|>` instructions, then
-- `<|box_start|>` option `<|box_end|>` per option, then
-- `<|fim_suffix|>`. Two projections of the normed hidden states score
-- each `<|box_end|>` against the last token as `q . k / sqrt(d)`. The
-- rows share the state through the slot cache.
--
-- Clef (arch `clef`) is a joint schema head. One prompt holds the state
-- and all questions. A small transformer reads the hidden states of all
-- tokens, the span means of each question and option, and the output
-- embeddings of the option tokens. It gives one logit per option.
-- With a vision encoder, images go before the state; the head reads
-- their hidden states too.
-- @module lgml.decide_head
local lgml = require("lgml")
local json = require("lgml.json")
local tojson = require("lgml.chat.tojson")
local systemone = require("lgml.systemone")
local decide = require("lgml.decide")
local log = require("lgml.log")
local stream = require("lgml.server.stream")

local L = log.new("decide")

local M = {}

--- The special tokens of the Kev prompt: state, question, option start,
-- option end and decide.
M.KEV_SPECIAL = { "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>" }
--- The system prompt of the Clef prompt.
M.CLEF_SYSTEM_PROMPT = "Read the complete state and schema. Decide every field jointly. Each answer " ..
    "must be exactly one of that field's allowed options."
--- The most tokens of a Clef prompt; a longer state is cut to fit.
M.CLEF_MAX_TOKENS = 16384
--- The most images of a Clef request, as Jev allows.
M.CLEF_MAX_IMAGES = 4
--- The default descriptions of the Clef noul options.
M.CLEF_NOUL = {
	["true"] = "The proposition is true or the answer is yes.",
	["false"] = "The proposition is false or the answer is no.",
}

local RAW = { add_bos = false }
local QTYPE = { noul = 0, choice = 1, score = 2 }

--- Returns the head kind that gguf metadata names.
-- @tparam table meta the gguf metadata
-- @treturn ?string `"kev"`, `"clef"` or nil
function M.kind(meta)
	local t = meta[tostring(meta["general.architecture"]) .. ".decision.type"]
	if t == "kev" or t == "clef" then
		return t
	end
end

-- request values

local function is_null(v)
	return v == nil or v == json.null
end

local function ordered(keys)
	return setmetatable({}, { __jsontype = "object", __order = keys })
end

-- The raw criteria of a question: the decoded `criteria_compact`, else
-- values made from the texts of the engine request.
local function criteria(q)
	if q.criteria_compact then
		return assert(systemone.decode(q.criteria_compact))
	end
	if q.type == "choice" then
		local keys = {}
		for i, o in ipairs(q.options) do
			keys[i] = o.key
		end
		local c = ordered(keys)
		for _, o in ipairs(q.options) do
			c[o.key] = o.null and json.null or o.text
		end
		return c
	elseif q.type == "score" then
		return json.array(table.move(q.levels, 1, #q.levels, 1, {}))
	end
	local keys = {}
	keys[#keys + 1] = q.yes and "true" or nil
	keys[#keys + 1] = q.no and "false" or nil
	if #keys == 0 then
		return nil
	end
	local c = ordered(keys)
	c["true"], c["false"] = q.yes, q.no
	return c
end

local function instructions(q)
	if q.instructions_compact then
		return assert(systemone.decode(q.instructions_compact))
	end
	return q.instructions
end

local function state_value(req)
	if req.state_compact then
		return assert(systemone.decode(req.state_compact))
	end
	return req.state
end

-- Kev prompt

local function lstrip(s)
	return (s:gsub("^%s+", ""))
end

--- Flattens a JSON value into text, as Kev's `render` does: objects
-- become `key: value` lines and arrays `- item` lines, nested ones
-- indented by two spaces per level. Booleans are `True` and `False`.
-- @param v the value, with key order (see `lgml.systemone.decode`)
-- @tparam[opt=0] int indent the level
-- @treturn string the text
function M.kev_text(v, indent)
	indent = indent or 0
	if is_null(v) then
		return ""
	elseif type(v) == "string" then
		return v
	elseif type(v) == "boolean" then
		return v and "True" or "False"
	elseif type(v) == "number" then
		return math.type(v) == "integer" and tostring(v) or tojson.float(v)
	end
	local pad = string.rep("  ", indent)
	local parts = {}
	if tojson.kind(v) == "array" then
		for i = 1, #v do
			parts[i] = pad .. "- " .. lstrip(M.kev_text(v[i], indent + 1))
		end
	else
		for i, k in ipairs(tojson.keys(v)) do
			local x = v[k]
			if type(x) == "table" and x ~= json.null then
				parts[i] = pad .. k .. ":\n" .. M.kev_text(x, indent + 1)
			else
				parts[i] = pad .. k .. ": " .. M.kev_text(x)
			end
		end
	end
	return table.concat(parts, "\n")
end

-- Caller text can never make a control token: `<|x|>` becomes `<¦x¦>`.
local function user_tokens(tk, text)
	return tk:encode((text:gsub("<|([%w_]+)|>", "<\u{a6}%1\u{a6}>")), RAW)
end

local function option_text(name, desc)
	if is_null(desc) or desc == "" then
		return name
	end
	return name .. ": " .. M.kev_text(desc)
end

--- Returns the option texts of a question, as Kev's `to_record`: noul
-- gives `no` then `yes`, each with its description; choice gives
-- `name` or `name: description` in request order; score gives the
-- level texts.
-- @tparam table q a question of the engine request
-- @treturn {string,...} the option texts
function M.kev_options(q)
	local c = criteria(q)
	local out = {}
	if q.type == "noul" then
		out[1] = option_text("no", c and c["false"])
		out[2] = option_text("yes", c and c["true"])
	elseif q.type == "choice" then
		for i, k in ipairs(tojson.keys(c)) do
			out[i] = option_text(k, c[k])
		end
	else
		for i = 1, #c do
			out[i] = M.kev_text(c[i])
		end
	end
	return out
end

--- Renders the Kev rows of a request.
-- Each row is the state, then one question. The pieces are tokenized
-- one by one, as Kev's `encode` does.
-- @param tk the tokenizer
-- @tparam table req the engine request (see `lgml.systemone.check`)
-- @treturn table `{ shared, rows }`: shared is the count of tokens that
-- all rows share; each row has `ids`, `ends` (the 1-based positions of
-- the option end tokens) and `first` (the position of the first option
-- start token)
function M.render_kev(tk, req)
	local sp = tk:special().tokens
	local s = {}
	for i, name in ipairs(M.KEV_SPECIAL) do
		s[i] = sp[name]
		if not s[i] then
			error({ status = 400, msg = "kev: the vocabulary has no " .. name })
		end
	end
	local head = { s[1] }
	local st = user_tokens(tk, M.kev_text(state_value(req)))
	table.move(st, 1, #st, 2, head)
	head[#head + 1] = s[2]
	local rows = {}
	for i, q in ipairs(req.questions) do
		local ids = table.move(head, 1, #head, 1, {})
		local it = user_tokens(tk, M.kev_text(instructions(q)))
		table.move(it, 1, #it, #ids + 1, ids)
		local row = { ids = ids, ends = {}, first = #ids + 1 }
		for _, text in ipairs(M.kev_options(q)) do
			ids[#ids + 1] = s[3]
			local ot = user_tokens(tk, text)
			table.move(ot, 1, #ot, #ids + 1, ids)
			ids[#ids + 1] = s[4]
			row.ends[#row.ends + 1] = #ids
		end
		ids[#ids + 1] = s[5]
		rows[i] = row
	end
	local shared = #head
	if #rows > 1 then
		shared = math.huge
		local a = rows[1].ids
		for k = 2, #rows do
			local b, n = rows[k].ids, 0
			while a[n + 1] ~= nil and a[n + 1] == b[n + 1] do
				n = n + 1
			end
			shared = math.min(shared, n)
		end
	end
	return { shared = shared, rows = rows }
end

-- Clef prompt

-- Compact JSON with sorted keys, as Python json.dumps(v,
-- ensure_ascii=False, separators=(",", ":"), sort_keys=True).
local function sorted_json(v)
	if type(v) ~= "table" or v == json.null then
		return tojson.encode(v)
	end
	local parts = {}
	if tojson.kind(v) == "array" then
		for i = 1, #v do
			parts[i] = sorted_json(v[i])
		end
		return "[" .. table.concat(parts, ",") .. "]"
	end
	local keys = {}
	for i, k in ipairs(tojson.keys(v)) do
		keys[i] = tostring(k)
	end
	table.sort(keys)
	for i, k in ipairs(keys) do
		parts[i] = tojson.string(k) .. ":" .. sorted_json(v[k])
	end
	return "{" .. table.concat(parts, ",") .. "}"
end

--- Returns the Clef text of a JSON value: a string as it is, other
-- values as compact JSON with sorted keys.
-- @param v the value
-- @treturn string the text
function M.clef_text(v)
	if type(v) == "string" then
		return v
	end
	return sorted_json(v)
end

--- Returns the options of a question in Clef order, as
-- `question_options` of the model code: noul gives `true` then
-- `false`, with default descriptions that the criteria replace; choice
-- gives the options sorted by name; score gives the levels.
-- @tparam table q a question of the engine request
-- @treturn {{string, any},...} the key and raw description of each
-- option; nil for no description
function M.clef_options(q)
	local c = criteria(q)
	local out = {}
	if q.type == "noul" then
		local d = { ["true"] = M.CLEF_NOUL["true"], ["false"] = M.CLEF_NOUL["false"] }
		for _, k in ipairs(c and tojson.keys(c) or {}) do
			if d[k] then
				d[k] = c[k]
			end
		end
		out[1] = { "true", d["true"] }
		out[2] = { "false", d["false"] }
	elseif q.type == "choice" then
		local names = tojson.keys(c)
		local keys = table.move(names, 1, #names, 1, {})
		table.sort(keys)
		for i, k in ipairs(keys) do
			out[i] = { k, c[k] }
		end
	else
		for i = 1, #c do
			out[i] = { tostring(i - 1), c[i] }
		end
	end
	for _, o in ipairs(out) do
		if o[2] == json.null then
			o[2] = nil
		end
	end
	return out
end

--- Renders the Clef prompt of a request, as `encode_record` of the
-- model code. The pieces are tokenized one by one. The questions are in
-- request order. A state longer than the room under max tokens is cut.
-- @param tk the tokenizer
-- @tparam table req the engine request
-- @tparam[opt] int max the most tokens (default `CLEF_MAX_TOKENS`)
-- @tparam[opt] table media entries after "STATE:\n", before the state:
-- the image entries and their newline (see `clef_media`)
-- @treturn table `{ ids, questions }`: each question has `type` (0
-- noul, 1 choice, 2 score), `span` `{ first, last }` (1-based, last
-- included) and `options`, each `{ key, span }`
-- @raise status 400 when the schema alone does not fit
function M.render_clef(tk, req, max, media)
	max = max or M.CLEF_MAX_TOKENS
	local function enc(text)
		return tk:encode(text, RAW)
	end
	local schema = enc("\n\nSCHEMA FIELDS:\n")
	local function add(text)
		local t = enc(text)
		local a = #schema + 1
		table.move(t, 1, #t, a, schema)
		return { a, #schema }
	end
	local qs = {}
	for i, q in ipairs(req.questions) do
		add(("\nFIELD %d\nID: %s\nTYPE: %s\nINSTRUCTION: "):format(i, q.id, q.type))
		local instr = instructions(q)
		if is_null(instr) or instr == "" then
			instr = q.id
		end
		local out = { id = q.id, type = QTYPE[q.type], span = add(M.clef_text(instr)), options = {} }
		add("\nALLOWED OPTIONS:\n")
		for j, o in ipairs(M.clef_options(q)) do
			add(("OPTION %d: "):format(j))
			local sem = '{"option_id":' .. tojson.string(o[1]) .. "}"
			if o[2] ~= nil then
				sem = '{"description":' .. sorted_json(o[2]) .. ',"option_id":' .. tojson.string(o[1]) .. "}"
			end
			out.options[j] = { key = o[1], span = add(sem) }
			add("\n")
		end
		add("END FIELD\n")
		for _, sp in ipairs({ out.span, table.unpack(out.options) }) do
			sp = sp.span or sp
			if sp[2] < sp[1] then
				error({ status = 400, msg = ("question %s: a question or option has no tokens"):format(q.id) })
			end
		end
		qs[i] = out
	end
	local prefix = enc("<|im_start|>system\n" .. M.CLEF_SYSTEM_PROMPT .. "<|im_end|>\n<|im_start|>user\nSTATE:\n")
	table.move(media or {}, 1, #(media or {}), #prefix + 1, prefix)
	local suffix = enc("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
	local fixed = #prefix + #schema + #suffix
	if fixed > max then
		error({ status = 400, msg = ("the schema needs %d tokens; the most is %d"):format(fixed, max) })
	end
	local st = enc(M.clef_text(state_value(req)))
	local ids = prefix
	table.move(st, 1, math.min(#st, max - fixed), #ids + 1, ids)
	local off = #ids
	table.move(schema, 1, #schema, #ids + 1, ids)
	table.move(suffix, 1, #suffix, #ids + 1, ids)
	for _, q in ipairs(qs) do
		q.span = { q.span[1] + off, q.span[2] + off }
		for _, o in ipairs(q.options) do
			o.span = { o.span[1] + off, o.span[2] + off }
		end
	end
	return { ids = ids, questions = qs }
end

-- weights

local function need(w, name, kind)
	local t = w:get(name)
	if not t then
		error(("%s: the model file has no %s; it needs a gguf with the decision head"):format(kind, name), 0)
	end
	return t
end

local function norm_of(w, p, kind)
	return { w = need(w, p .. ".weight", kind), b = need(w, p .. ".bias", kind) }
end

local function attn_of(w, p, kind)
	local a = {}
	for _, x in ipairs({ "q", "k", "v", "o" }) do
		a["w" .. x] = need(w, p .. "_" .. x .. ".weight", kind)
		a["b" .. x] = need(w, p .. "_" .. x .. ".bias", kind)
	end
	return a
end

--- Loads the head of a model.
-- @param model a model from `lgml.transformer.load`
-- @tparam string kind `"kev"` or `"clef"`
-- @treturn table the head weights and sizes
-- @raise when the model file lacks a weight of the head
function M.load(model, kind)
	local w, meta = model.w, model.meta or {}
	local arch = tostring(meta["general.architecture"])
	if kind == "kev" then
		local h = { kind = kind, w = need(w, "cls.output.weight", kind), b = w:get("cls.output.bias") }
		h.dim = h.w:shape()[1] // 2
		return h
	end
	local function key(k)
		local v = meta[arch .. ".decision." .. k]
		if v == nil then
			error(("clef: the model file has no %s.decision.%s"):format(arch, k), 0)
		end
		return v
	end
	local h = {
		kind = kind, n_routing = key("routing_block_count"), n_joint = key("block_count"),
		n_head = key("head_count"), eps = meta[arch .. ".attention.layer_norm_epsilon"] or 1e-5,
		hidden_norm = norm_of(w, "decision.hidden_norm", kind),
		option_summary_norm = norm_of(w, "decision.option_summary_norm", kind),
		field_norm = norm_of(w, "decision.field_norm", kind),
		option_norm = norm_of(w, "decision.option_norm", kind),
		type_embd = need(w, "token_types.weight", kind),
		scorer = need(w, "decision.scorer.weight", kind),
		scorer_b = need(w, "decision.scorer.bias", kind),
		scorer_out = need(w, "decision.scorer_out.weight", kind),
		scorer_out_b = need(w, "decision.scorer_out.bias", kind),
		output = model.output,
		layers = {},
	}
	for _, p in ipairs({ "memory", "question", "option_question", "global", "option_context", "option_lexical" }) do
		h["proj_" .. p] = need(w, "decision.proj_" .. p .. ".weight", kind)
	end
	-- The scales are the values used at inference: prior, joint, gate.
	h.scales = need(w, "decision.scales", kind):get()
	h.width = h.proj_memory:shape()[1]
	for i = 1, h.n_routing + h.n_joint do
		local p = "dec.blk." .. (i - 1) .. "."
		local l = {
			cross_norm = norm_of(w, p .. "cross_attn_norm", kind),
			cross = attn_of(w, p .. "cross_attn", kind),
			ffn_norm = norm_of(w, p .. "ffn_norm", kind),
			up = need(w, p .. "ffn_up.weight", kind), up_b = need(w, p .. "ffn_up.bias", kind),
			down = need(w, p .. "ffn_down.weight", kind), down_b = need(w, p .. "ffn_down.bias", kind),
		}
		if i <= h.n_routing then
			l.cross_norm_kv = norm_of(w, p .. "cross_attn_norm_kv", kind)
		else
			l.self_norm = norm_of(w, p .. "attn_norm", kind)
			l.self = attn_of(w, p .. "attn", kind)
		end
		h.layers[i] = l
	end
	return h
end

-- graphs

local function lin(g, w, b, x)
	local y = g:mul_mat(w, x)
	return b and g:add(y, b) or y
end

local function layer_norm(g, x, n, eps)
	return g:add(g:mul(g:norm(x, eps), n.w), n.b)
end

-- Multi-head attention of queries q (n, W) over kv (m, W), no mask, as
-- torch.nn.MultiheadAttention.
local function mha(g, h, a, q, kv)
	local W, H = h.width, h.n_head
	local d = W // H
	local n, m = q:shape()[1], kv:shape()[1]
	local Q = g:cont(g:permute(g:reshape(lin(g, a.wq, a.bq, q), { n, H, d }), { 2, 1, 3 }))
	local K = g:cont(g:permute(g:reshape(lin(g, a.wk, a.bk, kv), { m, H, d }), { 2, 1, 3 }))
	local V = g:reshape(lin(g, a.wv, a.bv, kv), { m, H, d })
	V = g:cont(g:transpose(g:permute(V, { 2, 1, 3 })))
	local s = g:soft_max(g:mul_mat(K, Q), nil, 1 / math.sqrt(d))
	local o = g:mul_mat(V, s)
	o = g:reshape(g:cont(g:permute(o, { 2, 1, 3 })), { n, W })
	return lin(g, a.wo, a.bo, o)
end

local function ffn(g, l, x, eps)
	local y = layer_norm(g, x, l.ffn_norm, eps)
	y = g:gelu_erf(lin(g, l.up, l.up_b, y))
	return lin(g, l.down, l.down_b, y)
end

--- Builds the Clef head on the backbone outputs.
-- t holds the inputs: `hidden` (N, n_embd), the normed hidden states of
-- the prompt; `pool_q` (Q, N) and `pool_o` (O, N), the span means of
-- the questions and options; `lex_ids` (T), the option tokens, and
-- `pool_lex` (O, T); `owner` (O), the question of each option from 0;
-- `qtype` (Q); and `mask` (Q, O), 0 where an option belongs to the
-- question and -inf elsewhere.
-- @param g the graph
-- @tparam table h the head from `load`
-- @tparam table t the inputs
-- @return the logits (O, 1)
function M.clef_graph(g, h, t)
	local eps = h.eps
	local x = layer_norm(g, t.hidden, h.hidden_norm, eps)
	local N, E = x:shape()[1], x:shape()[2]
	local mem = g:mul_mat(h.proj_memory, x)
	local glob = g:view(x, { 1, E }, nil, (N - 1) * lgml.row_size("f32", E))
	local xt = g:cont(g:transpose(x))
	local question = g:mul_mat(xt, t.pool_q)
	local ctx = g:mul_mat(xt, t.pool_o)
	local lex = g:get_rows(h.output, t.lex_ids)
	lex = g:mul_mat(g:cont(g:transpose(lex)), t.pool_lex)

	-- One query per option; the options read the prompt.
	local opts = g:add(g:mul_mat(h.proj_option_context, ctx), g:mul_mat(h.proj_option_lexical, lex))
	opts = g:add(opts, g:get_rows(g:mul_mat(h.proj_option_question, question), t.owner))
	for i = 1, h.n_routing do
		local l = h.layers[i]
		local a = mha(g, h, l.cross, layer_norm(g, opts, l.cross_norm, eps), layer_norm(g, mem, l.cross_norm_kv, eps))
		opts = g:add(opts, a)
		opts = g:add(opts, ffn(g, l, opts, eps))
	end

	-- One vector per question: its text, a summary of its options, the
	-- end of the prompt and its type.
	local fields = g:mul_mat(h.proj_question, question)
	local wts = g:soft_max(g:mul_mat(opts, fields), t.mask, 1 / math.sqrt(h.width))
	local summary = g:mul_mat(g:cont(g:transpose(opts)), wts)
	fields = g:add(fields, layer_norm(g, summary, h.option_summary_norm, eps))
	fields = g:add(fields, g:mul_mat(h.proj_global, glob))
	fields = g:add(fields, g:get_rows(h.type_embd, t.qtype))
	for i = h.n_routing + 1, #h.layers do
		local l = h.layers[i]
		local c = layer_norm(g, fields, l.self_norm, eps)
		fields = g:add(fields, mha(g, h, l.self, c, c))
		c = layer_norm(g, fields, l.cross_norm, eps)
		fields = g:add(fields, mha(g, h, l.cross, c, mem))
		fields = g:add(fields, ffn(g, l, fields, eps))
	end
	fields = layer_norm(g, fields, h.field_norm, eps)

	-- Prior: the output embeddings of an option against its question
	-- and the end of the prompt. Joint: the option against its field.
	local tiny = 1e-12
	local anchor = g:get_rows(g:l2_norm(g:add(question, glob), tiny), t.owner)
	local prior = g:sum_rows(g:mul(g:l2_norm(lex, tiny), anchor))
	local o = layer_norm(g, opts, h.option_norm, eps)
	local f = g:get_rows(fields, t.owner)
	local cos = g:sum_rows(g:mul(g:l2_norm(f, tiny), g:l2_norm(o, tiny)))
	local feat = g:concat(g:concat(g:concat(f, o, 2), g:mul(f, o), 2), g:abs(g:sub(f, o)), 2)
	local res = g:gelu_erf(lin(g, h.scorer, h.scorer_b, feat))
	-- The output row is a vector (W) in the file, often bf16.
	local wo = h.scorer_out
	if wo:type() ~= "f32" then
		wo = g:cast(wo, "f32")
	end
	res = g:add(g:sum_rows(g:mul(res, g:reshape(wo, { 1, h.width }))), h.scorer_out_b)
	local joint = g:add(g:scale(cos, h.scales[2]), res)
	return g:add(g:scale(prior, h.scales[1]), g:scale(joint, h.scales[3]))
end

-- f32 row bytes: zeros, then n copies of w from column a (1-based).
local F32_ZERO = string.pack("<f", 0)

local function span_row(n, a, b, w)
	local v = string.pack("<f", w)
	return F32_ZERO:rep(a - 1) .. v:rep(b - a + 1) .. F32_ZERO:rep(n - b)
end

--- Makes the input data of `clef_graph` for a rendering.
-- @tparam table rd the rendering from `render_clef`
-- @treturn table shapes and data: `pool_q`, `pool_o`, `lex_ids`,
-- `pool_lex`, `owner`, `qtype`, `mask`, each `{ type, shape, data }`
function M.clef_inputs(rd)
	local N = #rd.ids
	local pq, po, lex, plex, owner, qtype = {}, {}, {}, {}, {}, {}
	local spans = {}
	for qi, q in ipairs(rd.questions) do
		pq[qi] = span_row(N, q.span[1], q.span[2], 1 / (q.span[2] - q.span[1] + 1))
		qtype[qi] = q.type
		for _, o in ipairs(q.options) do
			local a, b = o.span[1], o.span[2]
			po[#po + 1] = span_row(N, a, b, 1 / (b - a + 1))
			owner[#owner + 1] = qi - 1
			spans[#spans + 1] = { #lex + 1, #lex + b - a + 1, b - a + 1 }
			table.move(rd.ids, a, b, #lex + 1, lex)
		end
	end
	for i, s in ipairs(spans) do
		plex[i] = span_row(#lex, s[1], s[2], 1 / s[3])
	end
	local Q, O = #pq, #po
	local neg = string.pack("<f", -math.huge)
	local mask = {}
	for qi = 1, Q do
		local row = {}
		for oi = 1, O do
			row[oi] = owner[oi] == qi - 1 and F32_ZERO or neg
		end
		mask[qi] = table.concat(row)
	end
	return {
		pool_q = { "f32", { Q, N }, table.concat(pq) },
		pool_o = { "f32", { O, N }, table.concat(po) },
		lex_ids = { "i32", { #lex }, lex },
		pool_lex = { "f32", { O, #lex }, table.concat(plex) },
		owner = { "i32", { O }, owner },
		qtype = { "i32", { Q }, qtype },
		mask = { "f32", { Q, O }, table.concat(mask) },
	}
end

-- running

local function sink_of(ctl)
	return {
		cached = 0, computed = 0,
		cancelled = function()
			return ctl and ctl.cancelled and ctl.cancelled() or false
		end,
	}
end

local function softmax(s, t)
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

-- A pass of ids on the slot that gives the head projection [q | k] of
-- the rows (0-based in ids).
local function kev_pass(e, h, ids, rows)
	local model = e.model
	local m = #rows
	local function finish(g, x, _, inp)
		inp.kev_rows = model.be:tensor("i32", { m })
		local y = g:get_rows(x, inp.kev_rows)
		y = g:mul(g:rms_norm(y, model.hp.eps), model.output_norm)
		return lin(g, h.w, h.b, y)
	end
	local opts = {
		hidden = model.hp.n_layer, flash_attn = e.flash_attn, tag = "kev:" .. m,
		fill = function(inp)
			inp.kev_rows:set(rows)
		end,
	}
	return e:run_pass(e.cache, ids, opts, finish):get()
end

-- Runs one Kev row. Returns the logits of its options, or nil when
-- cancelled.
local function kev_row(e, h, row, shared, sink)
	local p, first = row.ids, row.first
	-- The hidden states of the options must be computed now: keep at
	-- most the tokens before the first option. The prefill ends after
	-- its start token, so the shared state can get a checkpoint.
	local n = e:reuse(table.move(p, 1, first, 1, {}))
	sink.cached = sink.cached + n
	local stops = shared > n and shared < first and { shared } or nil
	if not e:prefill(table.move(p, 1, first, 1, {}), n + 1, sink, stops) then
		return nil
	end
	sink.computed = sink.computed + first - n
	local want = {}
	for k, pos in ipairs(row.ends) do
		want[pos] = k
	end
	want[#p] = 0
	local D = 2 * h.dim
	local vec = {}
	local i = first + 1
	while i <= #p do
		local j = math.min(#p, i + e.ubatch - 1)
		local ids = table.move(p, i, j, 1, {})
		local rows, who = {}, {}
		for pos = i, j do
			if want[pos] then
				rows[#rows + 1], who[#who + 1] = pos - i, want[pos]
			end
		end
		if #rows == 0 then
			e:eval(ids, "last")
		else
			local flat = kev_pass(e, h, ids, rows)
			for r, k in ipairs(who) do
				vec[k] = table.move(flat, (r - 1) * D + 1, r * D, 1, {})
			end
		end
		table.move(ids, 1, #ids, #e.tokens + 1, e.tokens)
		e:publish(e.cache, e.tokens)
		sink.computed = sink.computed + #ids
		i = j + 1
		stream.sleep(0)
		if sink:cancelled() then
			return nil
		end
	end
	local q, d = vec[0], h.dim
	local scale = 1 / math.sqrt(d)
	local out = {}
	for k = 1, #row.ends do
		local kv, s = vec[k], 0.0
		for c = 1, d do
			s = s + q[c] * kv[d + c]
		end
		out[k] = s * scale
	end
	return out
end

local function run_kev(e, req, sink, recipe)
	local h = e.decide_head
	local rd = M.render_kev(e.tk, req)
	local answers, nin = {}, 0
	for qi, q in ipairs(req.questions) do
		local row = rd.rows[qi]
		if #row.ids + 8 >= e.n_ctx then
			error({ status = 400, msg = ("prompt has %d tokens; context is %d"):format(#row.ids, e.n_ctx) })
		end
		if log.want_prompt() then
			log.prompt(L, "decide", e.tk:decode(row.ids), "model", e.id, "question", q.id)
		end
		local logits = kev_row(e, h, row, rd.shared, sink)
		if not logits then
			return nil
		end
		nin = nin + #row.ids
		local t = decide.temperature(recipe, q.type, #logits)
		local p = softmax(logits, t)
		L:debug("answer", "question", q.id, "type", q.type, "logits", logits, "temperature", t, "p", p)
		local a = q.type == "noul" and { yes = p[2] } or { probabilities = p }
		if req.raw then
			a.lgml = { logits = logits, temperature = t }
		end
		answers[q.id] = a
	end
	return answers, nin
end

-- Runs the Clef prompt ids from position 0 on the slot of e, which holds
-- its lock. Gives the normed hidden states of all tokens in a host
-- buffer, or nil when the sink is cancelled. Also M.clef_hidden.
local function clef_hidden(e, ids, sink)
	local model = e.model
	local E = model.hp.n_embd
	local row = lgml.row_size("f32", E)
	local buf = lgml.hostbuf(#ids * row)
	-- The head reads every token, so none comes from the cache.
	e:reuse({ ids[1] })
	local function finish(g, x)
		return g:mul(g:rms_norm(x, model.hp.eps), model.output_norm)
	end
	local delta = e.vision and e.vision.mrope and e.pos_deltas(ids)
	local i = 1
	while i <= #ids do
		local j = math.min(#ids, i + e.ubatch - 1)
		local key = e.image_key(ids[i])
		local opts = { hidden = model.hp.n_layer, flash_attn = e.flash_attn, tag = "clef" }
		if delta then
			e.cache.pos_delta = delta(i - 1)
		end
		if key then
			-- Image entries run on their embeddings; see `Engine:prefill`.
			local last = e.vision.causal and j or #ids
			j = i
			while j < last and e.image_key(ids[j + 1]) == key do
				j = j + 1
			end
			for k, v in pairs(e:image_opts(ids, i, j, e.cache.pos_delta or 0)) do
				opts[k] = v
			end
		else
			for k = i + 1, j do
				if e.image_key(ids[k]) then
					j = k - 1
					break
				end
			end
		end
		local part = table.move(ids, i, j, 1, {})
		local run = part
		if key then
			run = {}
			for k = 1, #part do
				run[k] = 0
			end
		end
		local out = e:run_pass(e.cache, run, opts, finish)
		out:to_host(buf, (i - 1) * row)
		table.move(part, 1, #part, #e.tokens + 1, e.tokens)
		e:publish(e.cache, e.tokens)
		sink.computed = sink.computed + #part
		i = j + 1
		stream.sleep(0)
		if sink:cancelled() then
			return nil
		end
	end
	return buf
end
M.clef_hidden = clef_hidden

--- Makes the Clef media entries of the prepared images of a request:
-- for each image, the begin token, one entry per embedding and the end
-- token; then a newline, as `IMAGE_PLACEHOLDER` of the model code.
-- @param e the engine, with a vision encoder
-- @tparam {string,...} keys the image keys, from `Engine:prepare_images`
-- @treturn table the entries
function M.clef_media(e, keys)
	local out = {}
	for _, key in ipairs(keys) do
		out[#out + 1] = e.image_tokens[1]
		for i = 1, e.images[key].n do
			out[#out + 1] = key .. ":" .. i
		end
		out[#out + 1] = e.image_tokens[2]
	end
	for _, id in ipairs(e.tk:encode("\n", RAW)) do
		out[#out + 1] = id
	end
	return out
end

--- Computes the Clef logits of a rendering from the hidden states.
-- @param be the backend
-- @tparam table h the head from `load`
-- @tparam table rd the rendering from `render_clef`
-- @param buf a host buffer with the normed hidden states, f32 rows
-- @tparam int n_embd the hidden size
-- @treturn {{number,...},...} the logits of each question, in its
-- option order
function M.clef_logits(be, h, rd, buf, n_embd)
	local N = #rd.ids
	local leaves = {}
	local function leaf(spec)
		local t = be:tensor(spec[1], spec[2], spec[3])
		leaves[#leaves + 1] = t
		return t
	end
	local t = {}
	for k, spec in pairs(M.clef_inputs(rd)) do
		t[k] = leaf(spec)
	end
	t.hidden = leaf({ "f32", { N, n_embd } })
	t.hidden:from_host(buf, 0, 0, N * lgml.row_size("f32", n_embd))
	local g = be:graph()
	local ok, flat = pcall(function()
		local out = M.clef_graph(g, h, t)
		g:compute(out)
		return out:get()
	end)
	g:close()
	for _, x in ipairs(leaves) do
		x:close()
	end
	if not ok then
		error(flat, 0)
	end
	local out, k = {}, 0
	for qi, q in ipairs(rd.questions) do
		out[qi] = {}
		for j = 1, #q.options do
			out[qi][j] = flat[k + j]
		end
		k = k + #q.options
	end
	return out
end

-- The prompt text for logs: an image becomes `<image key n tokens>`.
local function shown(e, ids)
	local parts, run, i = {}, {}, 1
	while i <= #ids do
		local key = type(ids[i]) == "string" and e.image_key(ids[i])
		if key then
			local j = i
			while e.image_key(ids[j + 1] or 0) == key do
				j = j + 1
			end
			parts[#parts + 1] = e.tk:decode(run)
			parts[#parts + 1] = ("<image %s %d tokens>"):format(key:sub(1, 12), j - i + 1)
			run, i = {}, j + 1
		else
			run[#run + 1] = ids[i]
			i = i + 1
		end
	end
	parts[#parts + 1] = e.tk:decode(run)
	return table.concat(parts)
end

local function run_clef(e, req, sink, recipe)
	local h = e.decide_head
	local media
	if req.images and #req.images > 0 then
		local keys, err = e:prepare_images(req.images)
		if not keys then
			error({ status = 400, msg = err })
		end
		media = M.clef_media(e, keys)
	end
	local rd = M.render_clef(e.tk, req, nil, media)
	local ids = rd.ids
	if #ids + 8 >= e.n_ctx then
		error({ status = 400, msg = ("prompt has %d tokens; context is %d"):format(#ids, e.n_ctx) })
	end
	if log.want_prompt() then
		log.prompt(L, "decide", shown(e, ids), "model", e.id, "images", media and #req.images)
	end
	local buf = clef_hidden(e, ids, sink)
	if not buf then
		return nil
	end
	local logits = M.clef_logits(e.be, h, rd, buf, e.model.hp.n_embd)
	local answers = {}
	for qi, q in ipairs(req.questions) do
		local cq, ls = rd.questions[qi], logits[qi]
		local t = decide.temperature(recipe, q.type, #ls)
		local p = softmax(ls, t)
		L:debug("answer", "question", q.id, "type", q.type, "logits", ls, "temperature", t, "p", p)
		local a
		if q.type == "noul" then
			a = { yes = p[1] }
		elseif q.type == "choice" then
			local by = {}
			for j, o in ipairs(cq.options) do
				by[o.key] = p[j]
			end
			a = { probabilities = {} }
			for j, o in ipairs(q.options) do
				a.probabilities[j] = by[o.key]
			end
		else
			a = { probabilities = p }
		end
		if req.raw then
			local keys = {}
			for j, o in ipairs(cq.options) do
				keys[j] = o.key
			end
			a.lgml = { keys = keys, logits = ls, temperature = t }
		end
		answers[q.id] = a
	end
	return answers, #ids
end

--- Answers the questions of a request with the decision head of the
-- engine (see `lgml.decide.run`, which calls this for the recipes of
-- `lgml.decide.HEADS`). The model must have the head of the recipe;
-- `e.decide_head` holds it, from `load`. Clef takes up to
-- `CLEF_MAX_IMAGES` images with a vision encoder; Kev images give 400.
-- @param e the engine
-- @tparam table req the engine request
-- @tparam[opt] table ctl the cancel control
-- @tparam table recipe the recipe; only its temperatures apply
-- @treturn[1] table the result, as `lgml.decide.run` gives it
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function M.run(e, req, ctl, recipe)
	local h = e.decide_head
	if not h or h.kind ~= recipe.prompt then
		return nil, ("the model has no %s decision head"):format(recipe.prompt), 400
	end
	local n_img = req.images and #req.images or 0
	if n_img > 0 and recipe.prompt ~= "clef" then
		return nil, recipe.prompt .. ": image input is not supported", 400
	elseif n_img > 0 and not e.vision then
		return nil, "image input is not supported: start the server with --mmproj", 400
	elseif n_img > M.CLEF_MAX_IMAGES then
		return nil, ("clef: at most %d images, got %d"):format(M.CLEF_MAX_IMAGES, n_img), 400
	end
	local sink = sink_of(ctl)
	local run = recipe.prompt == "kev" and run_kev or run_clef
	local ok, answers, nin = pcall(run, e, req, sink, recipe)
	if not ok then
		if type(answers) == "table" and answers.status then
			return nil, answers.msg, answers.status
		end
		error(answers, 0)
	elseif not answers then
		return nil, "cancelled", 499
	end
	return {
		model = e.id, answers = answers,
		usage = { input_tokens = nin, output_tokens = 0 },
		format = recipe.prompt,
		timings = { images = n_img, image_encode_ms = e.image_ms or 0, prompt_n = sink.computed,
			cache_n = sink.cached },
	}
end

return M
