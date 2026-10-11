--- LoRA fine-tuning of `lgml.transformer` models on chat data or text.
-- Chat samples are OpenAI chat messages, rendered with the chat
-- template of the model; the loss is the cross-entropy of the next
-- token, on the tokens of assistant turns only. Text samples are plain
-- text in chunks of the context size, with the loss on every token, for
-- continued pretraining. A micro-batch packs whole samples
-- into one pass; a block-diagonal causal mask keeps them apart, and
-- positions start at 0 in each sample. The gradients of a step add up
-- over its micro-batches, each weighted by its share of the loss
-- tokens, so a step minimizes the mean over all its tokens.
--
-- Training needs ops that have gradients in ggml: dense attention
-- models (llama, qwen3) with soft_max attention, and gated delta net
-- hybrids (qwen3next, qwen35). A hybrid takes one sample per
-- micro-batch. `bin/lgml-finetune` is the command line front end.
--
-- Direct Preference Optimization (DPO) trains on pairs of answers to one
-- prompt, a chosen and a rejected one. The reference policy is the model
-- with the adapter as it is before the first step.
-- @module lgml.finetune
local lgml = require("lgml")
local json = require("lgml.json")
local optim = require("lgml.optim")
local transformer = require("lgml.transformer")

local M = {}

--- Check that the model can train.
-- @param model a model from `lgml.transformer.load`
-- @raise when a part of the model has no gradient in ggml
function M.check(model)
	local hp = model.hp
	if hp.n_expert then
		error(hp.arch .. ": routed experts (mul_mat_id) have no gradient in ggml", 0)
	elseif hp.gemma or hp.n_swa then
		error(hp.arch .. ": gelu, soft capping and sliding windows have no gradient in ggml", 0)
	end
end

--- Read a JSONL dataset.
-- Each line is an object: { messages = OpenAI chat messages } or
-- { text = a string }. Blank lines are skipped. With format "chat" a
-- file may mix both kinds; with "text" every line holds text; with
-- "dpo" every line holds { messages, chosen, rejected }.
-- @tparam string path the file
-- @tparam[opt="chat"] string format "chat", "text" or "dpo"
-- @treturn {table,...} the records, as the lines hold them
-- @raise on a file error or a bad line
function M.read_jsonl(path, format)
	local text_only = format == "text"
	local f = assert(io.open(path))
	local out, ln = {}, 0
	for line in f:lines() do
		ln = ln + 1
		if line:match("%S") and format == "dpo" then
			local v, err = json.decode(line)
			if type(v) ~= "table" or type(v.messages) ~= "table" or #v.messages == 0 or
			    type(v.chosen) ~= "string" or type(v.rejected) ~= "string" then
				f:close()
				error(("%s:%d: need an object with messages, chosen and rejected (%s)"):format(path,
				    ln, err or "none found"))
			end
			out[#out + 1] = { messages = v.messages, chosen = v.chosen, rejected = v.rejected,
			    source = type(v.source) == "string" and v.source or nil }
		elseif line:match("%S") then
			local v, err = json.decode(line)
			local chat_ok = type(v) == "table" and type(v.messages) == "table" and #v.messages > 0
			local text_ok = type(v) == "table" and type(v.text) == "string" and v.text ~= ""
			if not text_ok and (text_only or not chat_ok) then
				f:close()
				error(("%s:%d: need an object with %s (%s)"):format(path, ln,
				    text_only and "text" or "messages or text", err or "none found"))
			end
			local rec = text_ok and { text = v.text } or { messages = v.messages }
			if type(v.source) == "string" then
				rec.source = v.source
			end
			out[#out + 1] = rec
		end
	end
	f:close()
	return out
end

--- Tokenize plain text into samples of at most ctx tokens.
-- Every token trains. Each chunk starts with BOS when the model wants
-- one; the first token of a chunk is context only.
-- @param tk the tokenizer, from `lgml.tokenizer.open`
-- @tparam string text the text
-- @tparam integer ctx the most tokens in a sample, at least 2
-- @treturn {table,...} the samples: each has ids and train
function M.encode_text(tk, text, ctx)
	local sp = tk:special()
	local bos = sp.add_bos and sp.bos or nil
	local ids = tk:encode(text, { add_bos = false })
	local per = ctx - (bos and 1 or 0)
	assert(per >= 2, "encode_text: ctx too small")
	local out = {}
	for i = 1, #ids, per do
		local s = { ids = {}, train = {} }
		if bos then
			s.ids[1], s.train[1] = bos, false
		end
		for j = i, math.min(#ids, i + per - 1) do
			s.ids[#s.ids + 1] = ids[j]
			s.train[#s.train + 1] = true
		end
		if #s.ids >= 2 then
			out[#out + 1] = s
		end
	end
	return out
end

-- Length of the common prefix of a and b.
local function common(a, b)
	local n = math.min(#a, #b)
	local i = 0
	while i < n and a:byte(i + 1) == b:byte(i + 1) do
		i = i + 1
	end
	return i
end

--- Tokenize one conversation and mark the tokens that train.
-- The text of assistant turn i is what the template adds to the
-- messages before it, past their generation prompt: the answer and the
-- end of turn. Some templates render an earlier turn in another form
-- than a final one (Qwen3 drops its think block); such a turn gets a
-- sample of its own that ends with it, as the model sees it when it
-- writes that turn.
-- @param tk the tokenizer, from `lgml.tokenizer.open`
-- @param family the chat family module, from `lgml.chat.load`
-- @tparam {table,...} messages the conversation
-- @tparam[opt] table opts options: enable_thinking (default false),
-- tools (OpenAI tools list), last (true: only the last message trains,
-- in one sample that ends with it)
-- @treturn {table,...} the samples: each has ids and train (a boolean
-- for each id)
function M.encode(tk, family, messages, opts)
	opts = opts or {}
	local function render(n, gen)
		return family.render(table.move(messages, 1, n, 1, {}), {
			enable_thinking = opts.enable_thinking or false, tools = opts.tools,
			add_generation_prompt = gen,
		})
	end
	local sp = tk:special()
	-- The sample of messages 1..n that trains the turns in want, or
	-- every turn that fits when want is nil. Also the turns that do not fit.
	local function sample(n, want)
		local full = render(n, false)
		local spans, rest = {}, {}
		for i = 1, n do
			if messages[i].role == "assistant" and (not want or want == i) then
				local upto = render(i, false)
				local start = common(render(i - 1, true), upto) + 1
				if full:sub(1, #upto) == upto and start <= #upto and
				    (#spans == 0 or start > spans[#spans][2]) then
					spans[#spans + 1] = { start, #upto }
				else
					rest[#rest + 1] = i
				end
			end
		end
		local s = { ids = {}, train = {} }
		local function push(text, train)
			if text == "" then
				return
			end
			for _, id in ipairs(tk:encode(text, { add_bos = false })) do
				s.ids[#s.ids + 1] = id
				s.train[#s.train + 1] = train
			end
		end
		if sp.add_bos and sp.bos then
			s.ids[1], s.train[1] = sp.bos, false
		end
		local pos = 1
		for _, span in ipairs(spans) do
			push(full:sub(pos, span[1] - 1), false)
			push(full:sub(span[1], span[2]), true)
			pos = span[2] + 1
		end
		push(full:sub(pos), false)
		return #spans > 0 and s or nil, rest
	end
	if opts.last then
		return { (sample(#messages, #messages)) }
	end
	local out = {}
	local s, rest = sample(#messages)
	out[#out + 1] = s
	for _, i in ipairs(rest) do
		out[#out + 1] = (sample(i, i))
	end
	return out
end

-- Loss tokens of a sample: next-token targets at trained positions.
local function targets_of(s)
	local n = 0
	for i = 2, #s.ids do
		if s.train[i] then
			n = n + 1
		end
	end
	return n
end

--- Pack samples into micro-batches.
-- Samples keep their order. A sample longer than ctx loses its end.
-- Samples without loss tokens are dropped.
-- @tparam {table,...} samples samples from `encode`
-- @tparam integer ctx the most tokens in a micro-batch
-- @tparam[opt] integer batch the most samples in a micro-batch (default
-- no limit)
-- @treturn {table,...} micro-batches: { samples, n (tokens), m (loss
-- tokens) }
-- @treturn integer the count of samples that lost their end
function M.pack(samples, ctx, batch)
	local out, cur, trimmed = {}, nil, 0
	for _, s in ipairs(samples) do
		if #s.ids > ctx then
			s = { ids = table.move(s.ids, 1, ctx, 1, {}), train = table.move(s.train, 1, ctx, 1, {}) }
			trimmed = trimmed + 1
		end
		local m = targets_of(s)
		if m > 0 then
			if not cur or cur.n + #s.ids > ctx or (batch and #cur.samples >= batch) then
				cur = { samples = {}, n = 0, m = 0 }
				out[#out + 1] = cur
			end
			cur.samples[#cur.samples + 1] = s
			cur.n, cur.m = cur.n + #s.ids, cur.m + m
			-- A micro-batch has a source when all its samples share it.
			if #cur.samples == 1 then
				cur.source = s.source
			elseif cur.source ~= s.source then
				cur.source = nil
			end
		end
	end
	return out, trimmed
end

local NEG, ZERO = string.pack("<f", -math.huge), string.pack("<f", 0)

-- Graph inputs for a micro-batch, as `transformer:inputs` makes them,
-- with a block-diagonal causal mask. Also the rows and targets of the
-- loss tokens.
local function inputs(model, mb)
	local be, n = model.be, mb.n
	local ids, pos, rows, targets, mask = {}, {}, {}, {}, {}
	for _, s in ipairs(mb.samples) do
		local off = #ids
		for i, id in ipairs(s.ids) do
			ids[off + i], pos[off + i] = id, i - 1
			mask[off + i] = NEG:rep(off) .. ZERO:rep(i) .. NEG:rep(n - off - i)
			if i > 1 and s.train[i] then
				rows[#rows + 1], targets[#targets + 1] = off + i - 2, id
			end
		end
	end
	local inp = {
		n = n, n_attn = n, n_swa_attn = n, flash_attn = false, logits = "all",
		hidden = model.hp.n_layer,
		ids = be:tensor("i32", { n }, ids), pos = be:tensor("i32", { n }, pos),
		mask = be:tensor("f32", { n, n }, table.concat(mask)),
		rs = model:zero_states(),
	}
	return inp, rows, targets
end

-- The loss of the head on hidden state x, times weight for the gradients.
local function head_loss(g, model, x, sel, labels, weight)
	local hp = model.hp
	local logits = transformer.head(g, hp, model.output_norm, model.output, g:get_rows(x, sel),
	    model.lora_out)
	local loss = g:cross_entropy_loss(logits, labels)
	return weight ~= 1 and g:scale(loss, weight) or loss, loss, logits
end

-- With its outputs closed, a graph frees its compute buffers when it
-- closes, not when the collector finds the nodes.
local function close_all(list)
	for _, t in pairs(list) do
		t:close()
	end
end

-- The backward pass one layer at a time. A forward pass keeps the
-- input of each layer; then the head and each layer, last first, run
-- again in a graph of their own and pass the gradient of their input
-- down. Device memory holds the activations of one layer.
local function loss_recompute(model, inp, sel, labels, weight)
	local be, hp = model.be, model.hp
	local NL, H = #model.layers, {}
	do
		local g <close> = be:graph(4096 + 2048 * NL)
		local x, taps = model:forward(g, inp)
		local outs = {}
		for il = 1, NL + 1 do
			H[il] = be:tensor("f32", { inp.n, hp.n_embd })
			outs[il] = g:cpy(il == 1 and taps.embd or taps["l_out-" .. (il - 2)], H[il])
		end
		g:compute(table.unpack(outs))
		outs[#outs + 1] = x
		close_all(outs)
	end
	local v
	do
		H[NL + 1]:requires_grad()
		local g <close> = be:graph(4096)
		local obj, loss, logits = head_loss(g, model, H[NL + 1], sel, labels, weight)
		obj:backward()
		g:compute(obj, loss)
		v = loss:get()[1]
		close_all({ obj, loss, logits })
	end
	for il = NL, 1, -1 do
		H[il]:requires_grad()
		local g <close> = be:graph(4096 + 2048)
		local li = setmetatable({ embd = H[il], deepstack = 0, from = il, hidden = il }, { __index = inp })
		local x = model:forward(g, li)
		-- Rows first: a GPU sums a whole tensor in one workgroup.
		local obj = g:sum(g:sum_rows(g:mul(x, H[il + 1]:grad())))
		obj:backward()
		g:compute(obj)
		close_all({ obj, x })
		-- A gradient is a leaf of its own: close it too, or it holds
		-- device memory until the collector runs.
		close_all({ H[il + 1]:grad(), H[il + 1] })
	end
	close_all({ H[1]:grad(), H[1] })
	return v
end

--- Compute the loss of one micro-batch.
-- With backward, the gradients of the loss times weight add to the
-- gradients of the trainable leaves.
-- @param model the model, with its adapters set
-- @tparam table mb a micro-batch from `pack`
-- @tparam[opt=1] number weight the factor on the loss for the gradients
-- @tparam[opt] boolean backward true to run the backward pass
-- @tparam[opt] boolean recompute true to run the backward pass one layer
-- at a time, recomputing each layer: more time, much less device memory
-- @treturn number the mean loss over the loss tokens
function M.loss(model, mb, weight, backward, recompute)
	local be, hp = model.be, model.hp
	weight = weight or 1
	-- The recurrent state would carry from one sample to the next.
	assert(not model.recurrent or #mb.samples == 1,
	    "a recurrent model takes one sample per micro-batch")
	local inp, rows, targets = inputs(model, mb)
	local sel = be:tensor("i32", { #rows }, rows)
	local labels = be:tensor("f32", { #rows, hp.n_vocab })
	labels:fill_onehot(targets)
	local v
	if backward and recompute then
		v = loss_recompute(model, inp, sel, labels, weight)
	else
		local g <close> = be:graph(4096 + 2048 * hp.n_layer)
		local x = model:forward(g, inp)
		local obj, loss, logits = head_loss(g, model, x, sel, labels, weight)
		if backward then
			obj:backward()
		end
		g:compute(obj, loss)
		v = loss:get()[1]
		close_all({ obj, loss, x, logits })
	end
	close_all({ inp.ids, inp.pos, inp.mask, sel, labels })
	if inp.rs then
		transformer.close_states(inp.rs)
	end
	return v
end

--- The mean loss over micro-batches, weighted by loss tokens.
-- @param model the model
-- @tparam {table,...} mbs micro-batches from `pack`
-- @treturn number the loss, or nil with no micro-batches
-- @treturn ?table the loss of each source, for micro-batches whose
-- samples share a `source`; nil when none has one
function M.evaluate(model, mbs)
	local sum, m, by = 0, 0, {}
	for _, mb in ipairs(mbs) do
		local l = M.loss(model, mb)
		sum = sum + l * mb.m
		m = m + mb.m
		if mb.source then
			local e = by[mb.source] or { sum = 0, m = 0 }
			e.sum, e.m = e.sum + l * mb.m, e.m + mb.m
			by[mb.source] = e
		end
	end
	local sources = {}
	for k, e in pairs(by) do
		sources[k] = e.sum / e.m
	end
	return m > 0 and sum / m or nil, next(sources) and sources or nil
end

--- Tokenize a preference pair.
-- Each side is the prompt plus an assistant turn with its answer; only
-- the answer and its end of turn train.
-- @param tk the tokenizer
-- @param family the chat family module
-- @tparam table rec a record from `read_jsonl` with format "dpo"
-- @tparam[opt] table opts options of `encode`
-- @treturn ?table { chosen, rejected, source }, with samples as `encode`
-- makes them; nil when a side has no answer tokens
function M.encode_pair(tk, family, rec, opts)
	opts = opts or {}
	local function side(answer)
		local msgs = table.move(rec.messages, 1, #rec.messages, 1, {})
		msgs[#msgs + 1] = { role = "assistant", content = answer }
		return M.encode(tk, family, msgs, { enable_thinking = opts.enable_thinking, tools = opts.tools,
		    last = true })[1]
	end
	local c, r = side(rec.chosen), side(rec.rejected)
	return c and r and { chosen = c, rejected = r, source = rec.source } or nil
end

--- Pack preference pairs: one micro-batch for each side.
-- A side longer than ctx loses its end; a pair with a side that keeps
-- no loss tokens is dropped.
-- @tparam {table,...} list pairs from `encode_pair`
-- @tparam integer ctx the most tokens in a side
-- @treturn {table,...} pairs: { c, r (micro-batches), n (tokens), source }
-- @treturn integer the count of sides that lost their end
function M.pack_pairs(list, ctx)
	local out, trimmed = {}, 0
	for _, p in ipairs(list) do
		local c, tc = M.pack({ p.chosen }, ctx)
		local r, tr = M.pack({ p.rejected }, ctx)
		trimmed = trimmed + tc + tr
		if c[1] and r[1] then
			out[#out + 1] = { c = c[1], r = r[1], n = c[1].n + r[1].n, source = p.source }
		end
	end
	return out, trimmed
end

-- The summed log-prob of the loss tokens of a micro-batch: `loss`
-- returns the mean cross-entropy over its mb.m tokens.
local function logp(model, mb)
	return -M.loss(model, mb) * mb.m
end

--- Compute the reference log-probs of pairs, forward only.
-- Sets ref_c and ref_r of each pair under the model as it is now.
-- @param model the model, with its adapters set
-- @tparam {table,...} list pairs from `pack_pairs`
function M.reference(model, list)
	for _, p in ipairs(list) do
		p.ref_c, p.ref_r = logp(model, p.c), logp(model, p.r)
	end
end

-- -log sigmoid(z) and 1 - sigmoid(z), without overflow.
local function neg_log_sigmoid(z)
	return math.max(-z, 0) + math.log(1 + math.exp(-math.abs(z)))
end

local function one_minus_sigmoid(z)
	local e = math.exp(-math.abs(z))
	return z >= 0 and e / (1 + e) or 1 / (1 + e)
end

--- Compute the DPO loss of one pair.
-- z = beta ((logp_c - ref_c) - (logp_r - ref_r)), and the loss is
-- -log sigmoid(z). The backward pass runs `loss` on each side with a
-- weight that makes its gradient the gradient of the DPO loss.
-- @param model the model, with its adapters set
-- @tparam table pair a pair from `pack_pairs`, after `reference`
-- @tparam number beta the DPO beta
-- @tparam[opt=1] number weight the factor on the loss for the gradients
-- @tparam[opt] boolean backward true to run the backward pass
-- @tparam[opt] boolean recompute see `loss`
-- @treturn number the loss
-- @treturn number z
function M.dpo_loss(model, pair, beta, weight, backward, recompute)
	local z = beta * ((logp(model, pair.c) - pair.ref_c) - (logp(model, pair.r) - pair.ref_r))
	if backward then
		-- d loss / d logp_c = -beta (1 - sigmoid z), and logp_c is -m_c
		-- times the mean cross-entropy of the chosen side. The rejected
		-- side has the opposite sign.
		local f = beta * one_minus_sigmoid(z) * (weight or 1)
		M.loss(model, pair.c, f * pair.c.m, true, recompute)
		M.loss(model, pair.r, -f * pair.r.m, true, recompute)
	end
	return neg_log_sigmoid(z), z
end

--- The mean DPO loss, reward margin and accuracy over pairs.
-- @param model the model
-- @tparam {table,...} list pairs from `pack_pairs`, after `reference`
-- @tparam number beta the DPO beta
-- @treturn number the loss, or nil with no pairs
-- @treturn number the mean reward margin z / beta
-- @treturn number the share of pairs with z > 0
-- @treturn ?table the loss of each source; nil when no pair has one
function M.dpo_evaluate(model, list, beta)
	if #list == 0 then
		return nil
	end
	local loss, margin, acc, by = 0, 0, 0, {}
	for _, p in ipairs(list) do
		local l, z = M.dpo_loss(model, p, beta)
		loss, margin, acc = loss + l, margin + z / beta, acc + (z > 0 and 1 or 0)
		if p.source then
			local e = by[p.source] or { sum = 0, m = 0 }
			e.sum, e.m = e.sum + l, e.m + 1
			by[p.source] = e
		end
	end
	local sources = {}
	for k, e in pairs(by) do
		sources[k] = e.sum / e.m
	end
	return loss / #list, margin / #list, acc / #list, next(sources) and sources or nil
end

-- Deterministic shuffle.
local function shuffle(list, seed)
	local x = seed
	for i = #list, 2, -1 do
		x = (x * 6364136223846793005 + 1442695040888963407) & 0x7fffffffffffffff
		local j = (x >> 20) % i + 1
		list[i], list[j] = list[j], list[i]
	end
end

--- Train an adapter.
-- Each epoch shuffles the micro-batches and runs them in steps of
-- `accum`. The learning rate rises linearly over the warmup steps, then
-- stays, or with schedule "cosine" falls to min_lr at the last step.
-- opts fields:
--
-- - `model`: the model, from `lgml.transformer.load`
-- - `adapter`: the adapter, from `lgml.lora`; it is applied to the model
-- - `train`, `eval`: micro-batches from `pack` (eval may be empty), or
--   pairs from `pack_pairs` with dpo
-- - `dpo`: true to train on pairs with the DPO loss, `beta` (0.1) its
--   beta; the reference log-probs come from the model before the first
--   step, and a step averages the loss over its pairs
-- - `epochs` (1), `accum` (1), `lr` (1e-4), `warmup` (0), `weight_decay`
--   (0), `optimizer` ("adamw" or "sgd"), `seed` (1), `schedule`
--   ("constant" or "cosine"), `min_lr` (0)
-- - `shuffle`: false keeps the order of train in every epoch, for a
--   curriculum (default true: a new order each epoch)
-- - `steps`: the most optimizer steps; the run stops there, and the
--   schedule ends there (default: every micro-batch of every epoch)
-- - `recompute`: run the backward pass one layer at a time (see `loss`)
-- - `clip`: the largest global gradient norm (default: no clipping)
-- - `eval_every`: steps between evals, and one before the first step
--   (default: at the end of each epoch)
-- - `save_every`, `save`: `save(step)` runs every save_every steps
-- - `log`: `log(event)` gets { kind = "step", step, epoch, loss, lr,
--   tokens, seconds, grad_norm, skipped } and { kind = "eval", step, loss,
--   sources }; sources maps each record source to its loss; with dpo
--   both also have margin (the mean of z / beta) and accuracy (the share
--   of pairs with z > 0)
--
-- A step whose loss or gradient norm is not finite makes no update, and
-- its event has skipped = true. MAX_SKIPS skips in a row raise an error.
-- @tparam table opts the options
-- @treturn table { steps, loss (of the last step), eval (the last eval
-- loss, or nil) }
local MAX_SKIPS = 8

local function finite(x)
	return x == x and x ~= math.huge and x ~= -math.huge
end

function M.train(opts)
	local model, ad = opts.model, opts.adapter
	M.check(model)
	model:set_adapters({ ad })
	local make = opts.optimizer == "sgd" and optim.sgd or optim.adamw
	local opt = make(ad:params(), { lr = opts.lr or 1e-4, weight_decay = opts.weight_decay or 0 })
	local lr, warmup, accum = opt.lr, opts.warmup or 0, opts.accum or 1
	local total = ((#opts.train + accum - 1) // accum) * (opts.epochs or 1)
	if opts.steps then
		total = math.min(total, opts.steps)
	end
	local min_lr = opts.min_lr or 0
	-- Linear warmup, then constant or a cosine down to min_lr at the end.
	local function lr_at(step)
		if warmup > 0 and step <= warmup then
			return lr * step / warmup
		elseif opts.schedule ~= "cosine" or total <= warmup then
			return lr
		end
		local f = (step - warmup) / (total - warmup)
		return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * f))
	end
	local log = opts.log or function() end
	local step, last, ev, skips = 0, nil, nil, 0
	local beta = opts.beta or 0.1
	if opts.dpo then
		M.reference(model, opts.train)
		M.reference(model, opts.eval or {})
	end
	local function eval()
		if opts.eval and #opts.eval > 0 then
			local margin, acc, sources
			if opts.dpo then
				ev, margin, acc, sources = M.dpo_evaluate(model, opts.eval, beta)
			else
				ev, sources = M.evaluate(model, opts.eval)
			end
			log({ kind = "eval", step = step, loss = ev, margin = margin, accuracy = acc,
			    sources = sources })
		end
	end
	if opts.eval_every and opts.eval_every > 0 then
		eval()
	end
	for epoch = 1, opts.epochs or 1 do
		local order = table.move(opts.train, 1, #opts.train, 1, {})
		if opts.shuffle ~= false then
			shuffle(order, (opts.seed or 1) + epoch)
		end
		for i = 1, #order, accum do
			if step >= total then
				break
			end
			local group = table.move(order, i, math.min(#order, i + accum - 1), 1, {})
			local m, tokens = 0, 0
			for _, mb in ipairs(group) do
				m, tokens = m + (opts.dpo and 1 or mb.m), tokens + mb.n
			end
			local t0 = lgml.clock()
			opt:zero_grad()
			local sum, margin, acc = 0, nil, nil
			if opts.dpo then
				margin, acc = 0, 0
				for _, p in ipairs(group) do
					local l, z = M.dpo_loss(model, p, beta, 1 / m, true, opts.recompute)
					sum, margin, acc = sum + l, margin + z / beta / m, acc + (z > 0 and 1 or 0) / m
				end
			else
				for _, mb in ipairs(group) do
					sum = sum + M.loss(model, mb, mb.m / m, true, opts.recompute) * mb.m
				end
			end
			step = step + 1
			opt.lr = lr_at(step)
			local gnorm = opts.clip and opt:clip(opts.clip) or nil
			local skip = not finite(sum) or (gnorm ~= nil and not finite(gnorm))
			if skip then
				skips = skips + 1
				if skips >= MAX_SKIPS then
					error(("finetune: %d steps in a row have a loss or gradient that is not finite"):format(skips))
				end
			else
				skips = 0
				opt:step()
				last = sum / m
			end
			log({ kind = "step", step = step, epoch = epoch, loss = sum / m, lr = opt.lr,
			    tokens = tokens, seconds = lgml.clock() - t0, grad_norm = gnorm, skipped = skip,
			    margin = margin, accuracy = acc })
			if opts.eval_every and opts.eval_every > 0 and step % opts.eval_every == 0 then
				eval()
			end
			if opts.save and opts.save_every and opts.save_every > 0 and
			    step % opts.save_every == 0 then
				opts.save(step)
			end
		end
		if not opts.eval_every then
			eval()
		end
		if step >= total then
			break
		end
	end
	opt:close()
	return { steps = step, loss = last, eval = ev }
end

return M
