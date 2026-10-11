--- Group Relative Policy Optimization of LoRA adapters.
-- For each prompt the model writes K answers with the adapter it trains.
-- A reward function scores them; a KL penalty to a frozen reference
-- adapter comes off each reward. The advantage of an answer is its reward
-- against the mean and spread of its group, and the step raises the
-- log-prob of answers above the mean and lowers the others.
-- @module lgml.grpo
local lgml = require("lgml")
local finetune = require("lgml.finetune")
local optim = require("lgml.optim")
local replay = require("lgml.replay")
local sampler = require("lgml.sampler")

local M = {}

--- Write one answer to a conversation with the model as it is now.
-- @param model the model, with its adapters set
-- @param tk the tokenizer
-- @param family the chat family module
-- @tparam {table,...} messages the prompt conversation
-- @tparam[opt] table opts temperature (0.8), seed (1), max_new (40),
-- enable_thinking (false), and other `lgml.sampler` options
-- @treturn string the answer text
-- @treturn {integer,...} its token ids, without the end token
-- @treturn table the training sample: the prompt and the answer tokens as
-- sampled, with the end token when one came; only the answer trains
function M.generate(model, tk, family, messages, opts)
	opts = opts or {}
	local text = family.render(messages, { add_generation_prompt = true,
	    enable_thinking = opts.enable_thinking or false })
	local ids = tk:encode(text, { add_bos = false })
	local max = opts.max_new or 40
	local eog = {}
	for _, id in ipairs(tk:special().eog or {}) do
		eog[id] = true
	end
	local so = { temperature = opts.temperature or 0.8, seed = opts.seed or 1 }
	for _, k in ipairs({ "top_k", "top_p", "min_p" }) do
		so[k] = opts[k]
	end
	local smp = sampler.new("default", so)
	local cache = model:cache(#ids + max + 8)
	local logits = replay.run(model, cache, ids, { logits = "last" })
	local out, last = {}, nil
	for _ = 1, max do
		local id = smp:sample(logits)
		smp:accept(id)
		if eog[id] then
			last = id
			break
		end
		out[#out + 1] = id
		logits = replay.run(model, cache, { id }, { logits = "last" })
	end
	-- The sample holds the tokens as sampled: a round trip through the
	-- chat template could tokenize the answer another way.
	local s = { ids = table.move(ids, 1, #ids, 1, {}), train = {} }
	for i = 1, #ids do
		s.train[i] = false
	end
	for _, id in ipairs(out) do
		s.ids[#s.ids + 1], s.train[#s.train + 1] = id, true
	end
	if last then
		s.ids[#s.ids + 1], s.train[#s.train + 1] = last, true
	end
	return tk:decode(out), out, s
end

local function logp(model, mb)
	return -finetune.loss(model, mb) * mb.m
end

--- Advantages of a group: (r - mean) / (std + eps), all 0 when the
-- rewards are equal.
-- @tparam {number,...} r the rewards
-- @treturn {number,...} the advantages
-- @treturn number the standard deviation of r
function M.advantages(r)
	local n, mean = #r, 0
	for _, v in ipairs(r) do
		mean = mean + v / n
	end
	local var = 0
	for _, v in ipairs(r) do
		var = var + (v - mean) ^ 2 / n
	end
	local sd = math.sqrt(var)
	local a = {}
	for i, v in ipairs(r) do
		a[i] = sd > 1e-6 and (v - mean) / (sd + 1e-4) or 0
	end
	return a, sd
end

--- Train an adapter with GRPO.
-- opts: `model`, `tk`, `family`; `adapter` (it trains) and `ref` (a frozen
-- copy of its start, for the KL penalty); `prompts` (records with
-- `messages` and what the reward needs); `reward(answer, record, ids)` ->
-- number, parts; `samples` (8) answers a prompt; `per_step` (4) prompts a
-- step; `kl` (0.05) on logp - logp_ref summed over the answer;
-- `temperature` (0.8); `max_new` (40); `steps`, `lr` (2e-6), `warmup`,
-- `weight_decay`, `clip`, `recompute`, `seed`; `log(event)` with step,
-- reward, kl, parse, length, spread, seconds, grad_norm, skipped;
-- `dump(record)` for every answer (with its micro-batch, mb); `save(step)`,
-- `save_every`.
--
-- The mean cross-entropy L of answer i has weight A_i m_i / (P K T): the
-- summed log-prob over a constant (T = max_new), so a long answer gets no
-- smaller push per token than a short one.
-- @tparam table opts the options
-- @treturn table { steps }
function M.train(opts)
	local model, tk, family, ad, ref = opts.model, opts.tk, opts.family, opts.adapter, opts.ref
	finetune.check(model)
	local K, P = opts.samples or 8, opts.per_step or 4
	local kl, T = opts.kl or 0.05, opts.max_new or 40
	local opt = optim.adamw(ad:params(), { lr = opts.lr or 2e-6, weight_decay = opts.weight_decay or 0 })
	local lr, warmup = opt.lr, opts.warmup or 0
	local log, dump = opts.log or function() end, opts.dump or function() end
	local steps = opts.steps or #opts.prompts // P
	local seed = opts.seed or 1
	local pi = 0
	for step = 1, steps do
		local t0 = lgml.clock()
		opt.lr = warmup > 0 and step <= warmup and lr * step / warmup or lr
		local groups = {}
		local rsum, klsum, parse, len, sdsum, n = 0, 0, 0, 0, 0, 0
		for _ = 1, P do
			pi = pi % #opts.prompts + 1
			local rec = opts.prompts[pi]
			local g = { rec = rec, items = {} }
			model:set_adapters({ ad })
			for k = 1, K do
				local text, ids, s = M.generate(model, tk, family, rec.messages, {
					temperature = opts.temperature, max_new = T, seed = seed * 1000003 + step * 1009 + k * 31 + pi })
				local mb = #ids > 0 and finetune.pack({ s }, #s.ids)[1]
				if mb then
					local r, parts = opts.reward(text, rec, ids)
					g.items[#g.items + 1] = { text = text, n = #ids, mb = mb, r = r, parts = parts }
				end
			end
			for _, it in ipairs(g.items) do
				it.logp = logp(model, it.mb)
			end
			model:set_adapters({ ref })
			for _, it in ipairs(g.items) do
				it.logp_ref = logp(model, it.mb)
			end
			model:set_adapters({ ad })
			local rr = {}
			for i, it in ipairs(g.items) do
				it.kl = it.logp - it.logp_ref
				rr[i] = it.r - kl * it.kl
			end
			local a, sd = M.advantages(rr)
			for i, it in ipairs(g.items) do
				it.a = a[i]
				rsum, klsum, len, n = rsum + it.r, klsum + it.kl, len + it.n, n + 1
				parse = parse + ((it.parts and it.parts.parse) and 1 or 0)
				dump({ step = step, prompt = rec, answer = it.text, reward = it.r, parts = it.parts,
				    logp = it.logp, logp_ref = it.logp_ref, advantage = it.a, mb = it.mb })
			end
			sdsum = sdsum + sd
			groups[#groups + 1] = g
		end
		opt:zero_grad()
		local any = false
		for _, g in ipairs(groups) do
			for _, it in ipairs(g.items) do
				if it.a ~= 0 then
					finetune.loss(model, it.mb, it.a * it.mb.m / (P * K * T), true, opts.recompute)
					any = true
				end
			end
		end
		local gnorm = any and opts.clip and opt:clip(opts.clip) or nil
		local skip = not any or (gnorm ~= nil and gnorm ~= gnorm)
		if not skip then
			opt:step()
		end
		groups = nil
		collectgarbage()
		log({ kind = "step", step = step, reward = rsum / n, kl = klsum / n, parse = parse / n,
		    length = len / n, spread = sdsum / P, lr = opt.lr, seconds = lgml.clock() - t0,
		    grad_norm = gnorm, skipped = skip })
		if opts.save and opts.save_every and opts.save_every > 0 and step % opts.save_every == 0 then
			opts.save(step)
		end
	end
	opt:close()
	return { steps = steps }
end

return M
