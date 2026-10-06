--- Speculative decoding: drafters, the draft length choice and state
-- copies for rollback. The engine runs the last sampled token and the
-- drafter's guesses in one verify pass, then samples each row in turn;
-- a guess stays while it equals the sampled token (see `lgml.engine`).
--
-- A drafter is a table with `draft(self, ctx, k)`, which returns at
-- most k token ids, maybe none. ctx holds `seq` (the prompt and the
-- sampled tokens; the last one is not in the cache yet; a non-integer
-- entry is an image position), `logits` and `row` (the node and row
-- that gave the last token), and `cache` (the slot's cache). A drafter
-- with `hidden = true` also gets `hidden` and `hidden_row`: the state
-- after the last layer, before the output norm, at that row. The
-- engine then runs each decode step through the verify path.
-- An optional `feedback(self, k, a)` hears that a of k guesses stayed;
-- a model-native draft head (MTP) drops its rejected positions there.
-- @module lgml.spec
local M = {}

-- Bits per token id in an n-gram key: three ids fit in 63 bits.
local ID_BITS = 21
local ID_MAX = (1 << ID_BITS) - 1

-- The key of the n-gram that ends at position p of seq, or nil when an
-- entry is not a token id that fits.
local function gram_key(seq, p, n)
	local key = 0
	for i = p - n + 1, p do
		local id = seq[i]
		if math.type(id) ~= "integer" or id < 0 or id > ID_MAX then
			return nil
		end
		key = (key << ID_BITS) | id
	end
	return key
end

local Lookup = {}
Lookup.__index = Lookup

--- A prompt lookup drafter.
-- It finds the last n tokens of the sequence at an earlier place in
-- the prompt or the output, longest n first, and guesses the tokens
-- that followed there. A match that runs into the end of the sequence
-- repeats with its period, so a loop gives full drafts. An index of the
-- last place of each n-gram keeps each draft fast.
-- @tparam[opt] table opts `ngram_min` (default 2) and `ngram_max`
-- (default 3, at most 3): the n-gram sizes
-- @treturn table the drafter
function M.lookup(opts)
	opts = opts or {}
	local lo, hi = opts.ngram_min or 2, math.min(opts.ngram_max or 3, 3)
	assert(lo >= 1 and lo <= hi, "lookup: bad n-gram sizes")
	local maps = {}
	for n = lo, hi do
		maps[n] = {}
	end
	return setmetatable({ lo = lo, hi = hi, maps = maps, done = 0 }, Lookup)
end

function Lookup:draft(ctx, k)
	local seq = ctx.seq
	local L = #seq
	-- Index the n-grams that end before the last token.
	for p = self.done + 1, L - 1 do
		for n = self.lo, math.min(self.hi, p) do
			local key = gram_key(seq, p, n)
			if key then
				self.maps[n][key] = p
			end
		end
	end
	self.done = math.max(self.done, L - 1)
	for n = math.min(self.hi, L), self.lo, -1 do
		local key = gram_key(seq, L, n)
		local p = key and self.maps[n][key]
		if p then
			local out = {}
			for j = 1, k do
				local q = p + j
				while q > L do
					q = q - (L - p)
				end
				local id = seq[q]
				if math.type(id) ~= "integer" then
					break
				end
				out[j] = id
			end
			return out
		end
	end
	return {}
end

local Mtp = {}
Mtp.__index = Mtp

--- The default draft gate of the MTP head: 0.5. On Qwen3.8-Flash-Next
-- with sampling and thinking, ungated drafts made decode slower than no
-- drafts; this gate made it about 20% faster than no drafts.
M.MTP_MIN_P = 0.5

--- An MTP drafter: the model's own draft head (see
-- `lgml.qwen4exp.mtp_draft`), then prompt lookup when the head cannot
-- draft. The head takes the state of the trunk at the row that gave
-- the last token, so the engine runs each decode step through the
-- verify path.
-- @param model the model, with mtp
-- @tparam[opt] table opts `max` (the most guesses of the head, default
-- 3), `min_p` (a guess whose probability is below it ends the draft;
-- default `MTP_MIN_P`, 0 drafts all), `flash_attn`, and the options of
-- `lookup`
-- @treturn table the drafter
function M.mtp(model, opts)
	opts = opts or {}
	return setmetatable({ model = model, max = opts.max or 3, min_p = opts.min_p or M.MTP_MIN_P,
		flash = opts.flash_attn,
		lookup = M.lookup(opts), hidden = true, drafted = 0, used = 0 }, Mtp)
end

-- The trunk streams of rows 1 .. n of the hidden node, in a host buffer.
local function host_rows(hidden, n)
	local lgml = require("lgml")
	local sh = hidden:shape()
	local row = lgml.row_size("f32", sh[2] * (sh[3] or 1))
	local buf = lgml.hostbuf(n * row)
	hidden:to_host(buf, 0, 0, n * row)
	return buf
end

function Mtp:draft(ctx, k)
	local seq, cache = ctx.seq, ctx.cache
	local n0 = cache.n_past
	local pend = table.move(seq, n0 + 1, #seq, 1, {})
	local ok = #pend >= 1
	for _, id in ipairs(pend) do
		ok = ok and math.type(id) == "integer"
	end
	-- Tokens the cache lacks besides the last one need their trunk rows,
	-- which the last verify pass gave from the position n0 on.
	local h_rest
	if ok and #pend > 1 then
		ok = ctx.hidden ~= nil and ctx.hidden_row == #pend - 1
		h_rest = ok and host_rows(ctx.hidden, #pend - 1) or nil
	end
	local out = {}
	if ok and cache.mtp_h then
		out = require("lgml.qwen4exp").mtp_draft(self.model, cache, pend, h_rest, math.min(k, self.max),
		    self.flash, nil, self.min_p)
		self.used = self.used + 1
	end
	if h_rest then
		h_rest:close()
	end
	if #out == 0 then
		return self.lookup:draft(ctx, k)
	end
	return out
end

--- Pass times by row count, for the draft length choice.
-- Each entry is a moving average of the seconds of a pass with that
-- many rows. A count with no entry gets an estimate from its neighbors.
-- @treturn table the costs
function M.costs()
	return setmetatable({ t = {} }, { __index = M.Costs })
end

M.Costs = {}

-- Weight of a new pass time in the moving average.
local COST_W = 0.2
-- The extra time of each row over the first, as a part of the time of
-- one row, before the engine has measured it.
local ROW_PRIOR = 0.1

--- Add the time of a pass.
-- @tparam integer n the rows
-- @tparam number dt the seconds
function M.Costs:add(n, dt)
	local old = self.t[n]
	self.t[n] = old and old + COST_W * (dt - old) or dt
end

--- The time of a pass of n rows, or nil before the first decode step.
-- @tparam integer n the rows
-- @treturn ?number the seconds
function M.Costs:get(n)
	local t = self.t
	if t[n] then
		return t[n]
	end
	local t1 = t[1]
	if not t1 then
		return nil
	end
	local lo, hi = 1, nil
	for m in pairs(t) do
		if m < n and m > lo then
			lo = m
		elseif m > n and (not hi or m < hi) then
			hi = m
		end
	end
	if hi then
		return t[lo] + (t[hi] - t[lo]) * (n - lo) / (hi - lo)
	end
	-- Past the largest pass: its slope, at least the prior one.
	local slope = ROW_PRIOR * t1
	if lo > 1 then
		slope = math.max(slope, (t[lo] - t1) / (lo - 1))
	end
	return t[lo] + slope * (n - lo)
end

-- Draft lengths to choose from: few lengths keep few graphs.
local LENGTHS = { 1, 2, 4, 8, 16, 32 }
-- A draft must promise this much more output per second.
local MARGIN = 1.05
-- Weight of earlier verify passes in the acceptance rate.
local DECAY = 0.9
-- The most decode steps between two probe drafts.
local MAX_WAIT = 64

local Session = {}
Session.__index = Session

--- The speculation state of one request.
-- It chooses the draft length from the acceptance rate of this request
-- and the pass times of the engine: it drafts k tokens when the
-- expected output per pass time beats a plain decode step. When no
-- length pays, it drafts one token now and then, with longer waits
-- each time, to measure the rate again.
-- @tparam table drafter the drafter
-- @tparam table costs the pass times, from `costs`
-- @tparam table opts `max` (the most draft tokens), `recurrent` (true
-- when a rejection also runs the kept tokens again), `adaptive` (false
-- drafts `max` tokens at each step)
-- @treturn table the session
function M.session(drafter, costs, opts)
	return setmetatable({
		drafter = drafter, costs = costs, max = opts.max or 16, recurrent = opts.recurrent,
		adaptive = opts.adaptive ~= false,
		-- Decayed counts of kept and rejected guesses, with a prior of one each.
		hit = 1, miss = 1, wait = 0, backoff = 1,
		drafted = 0, accepted = 0, passes = 0,
	}, Session)
end

--- The acceptance rate: the chance that the sampler keeps a guess.
-- @treturn number the rate
function Session:rate()
	return self.hit / (self.hit + self.miss)
end

-- The expected output per second of a verify pass of k guesses, over
-- the one of a decode step.
function Session:gain(k)
	local c = self.costs
	local t1, tk = c:get(1), c:get(k + 1)
	local a = math.min(self:rate(), 0.999)
	-- Tokens per pass: the kept guesses and the sampled token.
	local out = (1 - a ^ (k + 1)) / (1 - a)
	if self.recurrent then
		-- A rejection runs the kept tokens again in the next pass.
		local rows = 1 + a * (1 - a ^ k) / (1 - a)
		local slope = (c:get(k + 1) - t1) / k
		tk = tk + (1 - a ^ k) * rows * slope
	end
	return out * t1 / tk
end

--- The draft length for the next pass.
-- @tparam integer limit the most guesses that fit
-- @treturn integer the length, 0 for a plain decode step
function Session:choose(limit)
	limit = math.min(limit, self.max)
	if limit < 1 then
		return 0
	elseif not self.adaptive then
		return limit
	elseif not self.costs:get(1) then
		return 0
	end
	local best, best_k = MARGIN, 0
	for _, k in ipairs(LENGTHS) do
		if k > limit then
			break
		end
		local g = self:gain(k)
		if g > best then
			best, best_k = g, k
		end
	end
	if best_k > 0 then
		self.wait, self.backoff = 0, 1
		return best_k
	end
	self.wait = self.wait + 1
	if self.wait >= self.backoff then
		self.wait, self.backoff = 0, math.min(self.backoff * 2, MAX_WAIT)
		return 1
	end
	return 0
end

--- Count the result of a verify pass.
-- @tparam integer k the guesses
-- @tparam integer a the guesses that stayed
function Session:update(k, a)
	if k == 0 then
		return
	end
	self.hit = self.hit * DECAY + a
	self.miss = self.miss * DECAY + (a < k and 1 or 0)
	self.drafted = self.drafted + k
	self.accepted = self.accepted + a
	self.passes = self.passes + 1
	if self.drafter.feedback then
		self.drafter:feedback(k, a)
	end
end

--- Copies of the recurrent states of a cache, on the device.
-- `save` copies the conv and delta net states of cache to spare
-- tensors, and `restore` copies them back. The copies run as graphs,
-- so the state does not leave the device.
-- @param model the model
-- @treturn table the copier, with `save(cache)`, `restore(cache)` and
-- `close()`; `cache` and `n` tell which state it holds
function M.state_copy(model)
	local sc = { spare = {}, graphs = {} }
	local function close_graphs()
		for _, g in ipairs(sc.graphs) do
			g.save:close()
			g.back:close()
		end
		sc.graphs, sc.for_cache = {}, nil
	end
	-- One graph pair per backend, for the states of cache.
	local function build(cache)
		close_graphs()
		local by = {}
		-- The PLE conv state and the MTP input state are recurrent too.
		local parts = { conv = cache.conv, ssm = cache.ssm, pconv = cache.pconv or {}, mtp = { cache.mtp_h } }
		for _, part in ipairs({ "conv", "ssm", "pconv", "mtp" }) do
			for il, t in pairs(parts[part]) do
				local key = part .. il
				local be = t:backend()
				local s = sc.spare[key]
				if not s then
					s = be:tensor(t:type(), t:shape())
					sc.spare[key] = s
				end
				local e = by[be]
				if not e then
					e = { save = be:graph(), back = be:graph(), sn = {}, bn = {} }
					by[be] = e
					sc.graphs[#sc.graphs + 1] = e
				end
				e.sn[#e.sn + 1] = e.save:cpy(t, s)
				e.bn[#e.bn + 1] = e.back:cpy(s, t)
			end
		end
		sc.for_cache = cache
	end
	function sc.save(cache)
		if sc.for_cache ~= cache then
			build(cache)
		end
		for _, e in ipairs(sc.graphs) do
			e.save:compute(table.unpack(e.sn))
		end
		sc.cache, sc.n = cache, cache.n_past
	end
	function sc.restore(cache)
		assert(sc.cache == cache, "state_copy: no saved state for this cache")
		for _, e in ipairs(sc.graphs) do
			e.back:compute(table.unpack(e.bn))
		end
		cache.n_past, cache.rs_pos = sc.n, sc.n
	end
	function sc.close()
		close_graphs()
		for k, t in pairs(sc.spare) do
			t:close()
			sc.spare[k] = nil
		end
		sc.cache = nil
	end
	return sc
end

return M
