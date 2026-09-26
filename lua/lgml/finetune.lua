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
-- models (llama, qwen3) with soft_max attention. `bin/lgml-finetune`
-- is the command line front end.
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
	if model.recurrent then
		error(hp.arch .. ": gated delta net layers have no gradient in ggml; " ..
		    "training needs a dense attention model such as llama or qwen3")
	elseif hp.n_expert then
		error(hp.arch .. ": routed experts (mul_mat_id) have no gradient in ggml")
	elseif hp.gemma or hp.n_swa then
		error(hp.arch .. ": gelu, soft capping and sliding windows have no gradient in ggml")
	end
end

--- Read a JSONL dataset.
-- Each line is an object: { messages = OpenAI chat messages } or
-- { text = a string }. Blank lines are skipped. With format "chat" a
-- file may mix both kinds; with "text" every line holds text.
-- @tparam string path the file
-- @tparam[opt="chat"] string format "chat" or "text"
-- @treturn {table,...} the records: { messages } or { text }
-- @raise on a file error or a bad line
function M.read_jsonl(path, format)
	local text_only = format == "text"
	local f = assert(io.open(path))
	local out, ln = {}, 0
	for line in f:lines() do
		ln = ln + 1
		if line:match("%S") then
			local v, err = json.decode(line)
			local chat_ok = type(v) == "table" and type(v.messages) == "table" and #v.messages > 0
			local text_ok = type(v) == "table" and type(v.text) == "string" and v.text ~= ""
			if not text_ok and (text_only or not chat_ok) then
				f:close()
				error(("%s:%d: need an object with %s (%s)"):format(path, ln,
				    text_only and "text" or "messages or text", err or "none found"))
			end
			out[#out + 1] = text_ok and { text = v.text } or { messages = v.messages }
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
-- tools (OpenAI tools list)
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
-- @treturn integer the count of samples cut to ctx
function M.pack(samples, ctx, batch)
	local out, cur, cut = {}, nil, 0
	for _, s in ipairs(samples) do
		if #s.ids > ctx then
			s = { ids = table.move(s.ids, 1, ctx, 1, {}), train = table.move(s.train, 1, ctx, 1, {}) }
			cut = cut + 1
		end
		local m = targets_of(s)
		if m > 0 then
			if not cur or cur.n + #s.ids > ctx or (batch and #cur.samples >= batch) then
				cur = { samples = {}, n = 0, m = 0 }
				out[#out + 1] = cur
			end
			cur.samples[#cur.samples + 1] = s
			cur.n, cur.m = cur.n + #s.ids, cur.m + m
		end
	end
	return out, cut
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
	}
	return inp, rows, targets
end

--- Compute the loss of one micro-batch.
-- With backward, the gradients of the loss times weight add to the
-- gradients of the trainable leaves.
-- @param model the model, with its adapters set
-- @tparam table mb a micro-batch from `pack`
-- @tparam[opt=1] number weight the factor on the loss for the gradients
-- @tparam[opt] boolean backward true to run the backward pass
-- @treturn number the mean loss over the loss tokens
function M.loss(model, mb, weight, backward)
	local be, hp = model.be, model.hp
	local inp, rows, targets = inputs(model, mb)
	local sel = be:tensor("i32", { #rows }, rows)
	local labels = be:tensor("f32", { #rows, hp.n_vocab })
	labels:fill_onehot(targets)
	local g <close> = be:graph(4096 + 2048 * hp.n_layer)
	local x = model:forward(g, inp)
	local logits = transformer.head(g, hp, model.output_norm, model.output, g:get_rows(x, sel),
	    model.lora_out)
	local loss = g:cross_entropy_loss(logits, labels)
	local obj = (weight or 1) ~= 1 and g:scale(loss, weight) or loss
	if backward then
		obj:backward()
	end
	g:compute(obj, loss)
	local v = loss:get()[1]
	for _, t in ipairs({ inp.ids, inp.pos, inp.mask, sel, labels }) do
		t:close()
	end
	return v
end

--- The mean loss over micro-batches, weighted by loss tokens.
-- @param model the model
-- @tparam {table,...} mbs micro-batches from `pack`
-- @treturn number the loss, or nil with no micro-batches
function M.evaluate(model, mbs)
	local sum, m = 0, 0
	for _, mb in ipairs(mbs) do
		sum = sum + M.loss(model, mb) * mb.m
		m = m + mb.m
	end
	return m > 0 and sum / m or nil
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
-- stays. opts fields:
--
-- - `model`: the model, from `lgml.transformer.load`
-- - `adapter`: the adapter, from `lgml.lora`; it is applied to the model
-- - `train`, `eval`: micro-batches from `pack` (eval may be empty)
-- - `epochs` (1), `accum` (1), `lr` (1e-4), `warmup` (0), `weight_decay`
--   (0), `optimizer` ("adamw" or "sgd"), `seed` (1)
-- - `eval_every`: steps between evals (default: at the end of each epoch)
-- - `save_every`, `save`: `save(step)` runs every save_every steps
-- - `log`: `log(event)` gets { kind = "step", step, epoch, loss, lr,
--   tokens, seconds } and { kind = "eval", step, loss }
-- @tparam table opts the options
-- @treturn table { steps, loss (of the last step), eval (the last eval
-- loss, or nil) }
function M.train(opts)
	local model, ad = opts.model, opts.adapter
	M.check(model)
	model:set_adapters({ ad })
	local make = opts.optimizer == "sgd" and optim.sgd or optim.adamw
	local opt = make(ad:params(), { lr = opts.lr or 1e-4, weight_decay = opts.weight_decay or 0 })
	local lr, warmup, accum = opt.lr, opts.warmup or 0, opts.accum or 1
	local log = opts.log or function() end
	local step, last, ev = 0, nil, nil
	local function eval()
		if opts.eval and #opts.eval > 0 then
			ev = M.evaluate(model, opts.eval)
			log({ kind = "eval", step = step, loss = ev })
		end
	end
	for epoch = 1, opts.epochs or 1 do
		local order = table.move(opts.train, 1, #opts.train, 1, {})
		shuffle(order, (opts.seed or 1) + epoch)
		for i = 1, #order, accum do
			local group = table.move(order, i, math.min(#order, i + accum - 1), 1, {})
			local m, tokens = 0, 0
			for _, mb in ipairs(group) do
				m, tokens = m + mb.m, tokens + mb.n
			end
			local t0 = lgml.clock()
			opt:zero_grad()
			local sum = 0
			for _, mb in ipairs(group) do
				sum = sum + M.loss(model, mb, mb.m / m, true) * mb.m
			end
			step = step + 1
			opt.lr = warmup > 0 and step <= warmup and lr * step / warmup or lr
			opt:step()
			last = sum / m
			log({ kind = "step", step = step, epoch = epoch, loss = last, lr = opt.lr,
			    tokens = tokens, seconds = lgml.clock() - t0 })
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
	end
	opt:close()
	return { steps = step, loss = last, eval = ev }
end

return M
