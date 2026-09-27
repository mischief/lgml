--- Prompt cache for the chat engine, like llama-server's
-- --ctx-checkpoints and --cache-ram.
-- A checkpoint is a copy of the recurrent states and sliding rings at a
-- turn boundary. It lets a hybrid or sliding-window model go back to
-- that position; full-attention rows need no copy to go back.
-- The store keeps whole slots (tokens, full snapshot, checkpoints) in
-- host RAM under a byte budget, and gives back the one that shares the
-- most with a new prompt.
-- With a KV pool, the checkpoints also go to a state index that any
-- slot can start from: an entry is a checkpoint at position n, keyed
-- by the key of the pool block that ends at or before n and by the
-- tokens from there to n. Store entries and index entries share the
-- budget; the least recently used one goes first.
-- Each entry has the key of the engine state it came from, such as the
-- adapter set; a slot only takes entries with its current key.
-- @module lgml.prompt_cache

local M = {}
local P = {}
P.__index = P

-- A slot shorter than this is not worth a store entry.
local MIN_SAVE = 32

-- Token names that start a chat turn.
local TURN_NAMES = { "<|im_start|>", "<|turn>", "<start_of_turn>" }

--- Make a prompt cache for one engine slot.
-- @param model the model, with `snapshot`, `restore` and `snapshot_bytes`
-- @tparam table cache the slot's model cache
-- @tparam table opts options:
--
-- - `ram`: store budget in bytes (0 turns the store off)
-- - `checkpoints`: the most checkpoints per slot (0 turns them off)
-- - `special`: special token ids by name, from `tokenizer:special().tokens`
-- - `key`: the key of the current state (default ""); set the `key`
--   field when the state changes
-- @treturn PromptCache the cache
function M.new(model, cache, opts)
	local turn = {}
	for _, name in ipairs(TURN_NAMES) do
		local id = (opts.special or {})[name]
		if id then
			turn[id] = true
		end
	end
	return setmetatable({
		model = model, cache = cache,
		fixed = (model.recurrent or cache.ring and cache.ring < cache.n_ctx) and true or false,
		ram = opts.ram or 0, max_ckpt = opts.checkpoints or 0, turn = turn, key = opts.key or "",
		ckpts = {}, store = {}, states = {}, bytes = 0, tick = 0, restored = 0,
		stats = { restored = 0, checkpoints = 0, saved = 0, loaded = 0, evicted = 0, states = 0 },
	}, P)
end

--- The prompt cache of an engine slot, from `new`.
-- Fields: `ckpts` (the slot's checkpoints by position), `store` (the
-- entries), `key`, `bytes` (store bytes), `restored` (tokens the last `reuse`
-- restored from host memory) and `stats` (totals: restored tokens,
-- checkpoints, saved, loaded and evicted entries).
-- @type PromptCache

local function hold(s)
	s.refs = (s.refs or 0) + 1
	return s
end

local function release(s)
	s.refs = s.refs - 1
	if s.refs == 0 then
		s.buf:close()
	end
end

-- Length of the common prefix of a and b, at most cap.
local function common(a, b, cap)
	local n, m = 0, math.min(#a, cap)
	while n < m and a[n + 1] == b[n + 1] do
		n = n + 1
	end
	return n
end

-- The newest checkpoint at or before n, or nil.
local function newest(ckpts, n)
	for i = #ckpts, 1, -1 do
		if ckpts[i].n <= n then
			return ckpts[i]
		end
	end
end

-- Tokens a store entry gives a prompt that shares l tokens with it,
-- and the checkpoint to restore (nil for the end of the snapshot).
function P:entry_keep(e, l)
	if l == #e.tokens or not self.fixed then
		return l
	end
	local ck = newest(e.ckpts, l)
	return ck and ck.n or 0, ck
end

-- Tokens the slot keeps of its n shared ones.
function P:slot_keep(eng, n)
	if n >= #eng.tokens or not self.fixed then
		return n
	end
	if not self.model.recurrent and self.model:rewind(self.cache, n, true) == n then
		return n
	end
	local ck = newest(self.ckpts, n)
	return ck and ck.n or 0
end

local function remove(list, x)
	for i, y in ipairs(list) do
		if y == x then
			return table.remove(list, i)
		end
	end
end

-- Drop the least recently used store or state index entry.
function P:evict()
	local lru, list
	for _, l in ipairs({ self.store, self.states }) do
		for _, e in ipairs(l) do
			if not lru or e.used < lru.used then
				lru, list = e, l
			end
		end
	end
	remove(list, lru)
	self.bytes = self.bytes - lru.bytes
	if list == self.states then
		release(lru.ck)
		return
	end
	release(lru.snap)
	for _, ck in ipairs(lru.ckpts) do
		release(ck)
	end
	self.stats.evicted = self.stats.evicted + 1
end

-- Make room for bytes under the budget. False when they do not fit.
function P:room(bytes)
	if bytes > self.ram then
		return false
	end
	while self.bytes + bytes > self.ram and (#self.store > 0 or #self.states > 0) do
		self:evict()
	end
	return true
end

local function same(a, b)
	if #a ~= #b then
		return false
	end
	for i = 1, #a do
		if a[i] ~= b[i] then
			return false
		end
	end
	return true
end

--- Put a checkpoint in the state index.
-- @tparam string parent the key of the pool block that ends at or
-- before the checkpoint, or the root key
-- @tparam {integer|string,...} toks the tokens from that block to the
-- checkpoint (none when it is on a block edge)
-- @tparam table ck the checkpoint, from `checkpoint`
function P:add_state(parent, toks, ck)
	self.tick = self.tick + 1
	for _, s in ipairs(self.states) do
		if s.parent == parent and same(s.toks, toks) then
			s.used = self.tick
			return
		end
	end
	if self.ram <= 0 or not self:room(ck.bytes) then
		return
	end
	self.states[#self.states + 1] = { parent = parent, toks = toks, n = ck.n, ck = hold(ck),
		used = self.tick, bytes = ck.bytes }
	self.bytes = self.bytes + ck.bytes
	self.stats.states = self.stats.states + 1
end

--- The index entries after parent whose tokens are a prefix of
-- prompt[from..], at most max tokens in all, deepest first.
-- @tparam string parent the key before them
-- @tparam {integer|string,...} prompt the prompt
-- @tparam integer from the first prompt token after parent
-- @tparam integer max the most prompt tokens an entry may cover
-- @treturn {table,...} entries: { n, ck, toks }
function P:find_states(parent, prompt, from, max)
	local out = {}
	for _, s in ipairs(self.states) do
		if s.parent == parent and s.n <= max then
			local good = true
			for k = 1, #s.toks do
				if s.toks[k] ~= prompt[from + k - 1] then
					good = false
					break
				end
			end
			if good then
				out[#out + 1] = s
			end
		end
	end
	table.sort(out, function(a, b)
		return a.n > b.n
	end)
	return out
end

--- Make an index checkpoint the only checkpoint of the active slot,
-- after the slot starts from it.
-- @tparam table s an entry from `find_states`
function P:adopt(s)
	self.tick = self.tick + 1
	s.used = self.tick
	self:trim(-1)
	self.ckpts[1] = hold(s.ck)
	self.stats.restored = self.stats.restored + s.n
end

-- Save the slot to the store, unless it is short, too big or already there.
function P:save(eng)
	local toks = eng.tokens
	if self.ram <= 0 or #toks < MIN_SAVE then
		return
	end
	self.tick = self.tick + 1
	for _, e in ipairs(self.store) do
		if e.key == self.key and #e.tokens == #toks and common(e.tokens, toks, #toks) == #toks then
			e.used = self.tick
			return
		end
	end
	local bytes = self.model:snapshot_bytes(self.cache, true)
	for _, ck in ipairs(self.ckpts) do
		bytes = bytes + ck.bytes
	end
	if not self:room(bytes) then
		return
	end
	local ckpts = {}
	for i, ck in ipairs(self.ckpts) do
		ckpts[i] = hold(ck)
	end
	self.store[#self.store + 1] = {
		tokens = table.move(toks, 1, #toks, 1, {}), ckpts = ckpts, bytes = bytes,
		snap = hold(self.model:snapshot(self.cache, true)), used = self.tick, key = self.key,
	}
	self.bytes = self.bytes + bytes
	self.stats.saved = self.stats.saved + 1
end

-- Move store entry e into the slot at u tokens, with checkpoint ck.
function P:load(eng, e, u, ck)
	local m = self.model
	if u == #e.tokens then
		m:restore(self.cache, e.snap)
	elseif not ck then
		m:restore(self.cache, e.snap, { rows = u })
	else
		m:restore(self.cache, e.snap, { rows = u, state = false })
		m:restore(self.cache, ck)
	end
	eng.tokens = table.move(e.tokens, 1, u, 1, {})
	for _, c in ipairs(self.ckpts) do
		release(c)
	end
	self.ckpts = {}
	for _, c in ipairs(e.ckpts) do
		if c.n <= u then
			self.ckpts[#self.ckpts + 1] = c
		else
			release(c)
		end
	end
	release(e.snap)
	self.stats.loaded = self.stats.loaded + 1
end

--- Choose the prefix for a prompt before the engine keeps n tokens.
-- When a store entry gives more than the slot, the slot goes to the
-- store and the entry comes into the slot. When the slot keeps less
-- than half its tokens, a copy goes to the store.
-- @param eng the engine, with `tokens`
-- @tparam {integer,...} prompt the prompt ids
-- @tparam integer n the tokens the slot shares with the prompt
-- @treturn integer the tokens the slot now shares with the prompt
function P:reuse(eng, prompt, n)
	self.restored = 0
	local keep = self:slot_keep(eng, n)
	local best, bu, bck = nil, keep, nil
	for _, e in ipairs(self.store) do
		local u, ck = 0, nil
		if e.key == self.key then
			u, ck = self:entry_keep(e, common(e.tokens, prompt, #prompt - 1))
		end
		if u > bu then
			best, bu, bck = e, u, ck
		end
	end
	if best then
		remove(self.store, best)
		self.bytes = self.bytes - best.bytes
		self:save(eng)
		self:load(eng, best, bu, bck)
		self.restored = bu
		self.stats.restored = self.stats.restored + bu
		return bu
	end
	if keep * 2 < #eng.tokens then
		self:save(eng)
	end
	return n
end

--- Go back to the newest checkpoint at or before n.
-- The engine calls this when the model state cannot rewind to n.
-- @param eng the engine
-- @tparam integer n the wanted count of kept tokens
-- @treturn integer the kept tokens: the checkpoint position, or 0
function P:back(eng, n)
	local ck = newest(self.ckpts, n)
	if not ck then
		return 0
	end
	self.model:restore(self.cache, ck)
	self.restored = self.restored + ck.n
	self.stats.restored = self.stats.restored + ck.n
	return ck.n
end

--- Drop the checkpoints after n kept tokens.
-- @tparam integer n the kept tokens
function P:trim(n)
	while #self.ckpts > 0 and self.ckpts[#self.ckpts].n > n do
		release(table.remove(self.ckpts))
	end
end

--- Free a list of checkpoints of a slot that is not the active one.
-- @tparam {table,...} list the checkpoints; the call empties it
function P:drop(list)
	while #list > 0 do
		release(table.remove(list))
	end
end

--- Take a checkpoint at cache.n_past, unless one is there.
-- Past the limit, it drops the oldest but the first: the first one
-- often ends a system prompt that other conversations share.
function P:checkpoint()
	local n = self.cache.n_past
	local last = self.ckpts[#self.ckpts]
	if self.max_ckpt <= 0 or n == 0 or (last and last.n >= n) then
		return
	end
	self.ckpts[#self.ckpts + 1] = hold(self.model:snapshot(self.cache))
	self.stats.checkpoints = self.stats.checkpoints + 1
	if #self.ckpts > self.max_ckpt then
		release(table.remove(self.ckpts, self.max_ckpt > 1 and 2 or 1))
	end
end

--- Plan checkpoints for a prefill of prompt[from..].
-- The positions are before the second and the last turn start of the
-- prompt. A position at from - 1 gets its checkpoint now.
-- @tparam {integer,...} prompt the prompt ids
-- @tparam integer from the first id to run
-- @treturn {integer,...} positions in from .. #prompt - 1, in order
function P:plan(prompt, from)
	local stops = {}
	if not self.fixed or self.max_ckpt <= 0 or not next(self.turn) then
		return stops
	end
	local starts = {}
	for i = 1, #prompt do
		if self.turn[prompt[i]] then
			starts[#starts + 1] = i
		end
	end
	for _, s in ipairs({ starts[2] or 0, starts[#starts] or 0 }) do
		local k = s - 1
		if k > 0 and k == from - 1 then
			self:checkpoint()
		elseif k >= from and k ~= stops[#stops] then
			stops[#stops + 1] = k
		end
	end
	return stops
end

--- Free the store, the state index and the checkpoints.
function P:clear()
	while #self.store > 0 or #self.states > 0 do
		self:evict()
	end
	self:trim(-1)
end

return M
