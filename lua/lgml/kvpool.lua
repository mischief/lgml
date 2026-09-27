--- Block bookkeeping of a shared KV pool.
-- The pool has `blocks` blocks of `size` cells each. A sequence takes
-- blocks as it grows and gives them back at the end. A block has a
-- reference count. Only blocks below `cap` are usable: the owner of the
-- tensors raises `cap` when it grows them.
--
-- A full block can get a key: a hash of the key of the block before it
-- and of its own tokens. A keyed block with no reference stays cached:
-- it keeps its cells and its key until the pool needs it. A new
-- sequence with the same prefix finds the blocks by key and shares
-- them. Eviction takes the least recently used cached block, and the
-- deepest one first on a tie, so a prefix loses its last blocks first.
-- This module holds no tensors; `lgml.transformer` does.
-- @module lgml.kvpool
local lgml = require("lgml")

local M = {}
local P = {}
P.__index = P

--- The default count of sequences an engine keeps (`slots`).
M.SEQS = 4

--- The key of a block.
-- @tparam string parent the key of the block before it, or a root key
-- for the first block (such as the adapter set)
-- @tparam {integer|string,...} tokens the tokens of the block
-- @tparam[opt] integer i the first token (default 1)
-- @tparam[opt] integer n the token count (default all from i)
-- @treturn string the key
function M.key(parent, tokens, i, n)
	i = i or 1
	local j = n and i + n - 1 or #tokens
	local parts = { parent }
	for k = i, j do
		parts[#parts + 1] = tostring(tokens[k])
	end
	return lgml.sha256(table.concat(parts, ","))
end

--- Make the bookkeeping of a pool.
-- @tparam table opts `blocks` (the count), `size` (cells per block)
-- and `cap` (usable blocks now, default all)
-- @treturn Pool the pool
function M.new(opts)
	return setmetatable({
		n = opts.blocks, size = opts.size, cap = opts.cap or opts.blocks,
		refs = {}, used = 0, low = 0,
		keys = {}, bykey = {}, info = {}, kids = {}, cached = 0, tick = 0,
		stats = { evicted = 0, hits = 0, hit_blocks = 0 },
	}, P)
end

--- The bookkeeping of a pool, from `new`.
-- Fields: `n` (blocks), `size` (cells per block), `cap` (usable
-- blocks), `used` (blocks with references), `cached` (keyed blocks
-- with no reference) and `stats` (`evicted` blocks, `hits` and
-- `hit_blocks` of `lookup`s).
-- @type Pool

-- Take free block b.
local function take_at(self, b)
	self.refs[b] = 1
	self.used = self.used + 1
	if b == self.low then
		self.low = b + 1
	end
	return b
end

--- Take a free block below cap: the lowest one from min up, else the
-- lowest one. A cached block is not free.
-- Low blocks keep the cells of a sequence close together, so attention
-- reads a short window of the pool. With min past the last block of a
-- sequence, its blocks stay in cell order, and attention sums its
-- cells in position order.
-- @tparam[opt] integer min the lowest block wanted (default 0)
-- @treturn[1] integer the block id, from 0, with one reference
-- @return[2] nil when no block is free below cap
function P:take(min)
	local from = math.max(self.low, min or 0)
	for b = from, self.cap - 1 do
		if not self.refs[b] and not self.keys[b] then
			return take_at(self, b)
		end
	end
	for b = self.low, math.min(from, self.cap) - 1 do
		if not self.refs[b] and not self.keys[b] then
			return take_at(self, b)
		end
	end
	self.low = self.cap
	return nil
end

-- Remove the key of block b.
local function unkey(self, b)
	local key = self.keys[b]
	local info = self.info[b]
	self.keys[b], self.info[b] = nil, nil
	if self.bykey[key] == b then
		self.bykey[key] = nil
	end
	local kids = self.kids[info.parent]
	if kids then
		kids[b] = nil
		if next(kids) == nil then
			self.kids[info.parent] = nil
		end
	end
end

--- Take the cached block that was used least recently, and the deepest
-- one on a tie. It loses its key.
-- @treturn[1] integer the block id, with one reference
-- @return[2] nil when no block is cached
function P:evict()
	local best, bu, bd
	for b, info in pairs(self.info) do
		if not self.refs[b] and b < self.cap and
		    (not best or info.used < bu or (info.used == bu and info.depth > bd)) then
			best, bu, bd = b, info.used, info.depth
		end
	end
	if not best then
		return nil
	end
	unkey(self, best)
	self.cached = self.cached - 1
	self.stats.evicted = self.stats.evicted + 1
	self.refs[best] = 1
	self.used = self.used + 1
	return best
end

--- Add a reference to a block. A cached block is in use again.
-- @tparam integer b the block id
function P:ref(b)
	local r = self.refs[b]
	if not r then
		assert(self.keys[b], "kvpool: ref of a free block")
		self.cached = self.cached - 1
		self.used = self.used + 1
		r = 0
	end
	self.refs[b] = r + 1
end

--- Drop a reference to a block. At 0, a keyed block stays cached; any
-- other block is free.
-- @tparam integer b the block id
function P:unref(b)
	local r = assert(self.refs[b], "kvpool: unref of a free block") - 1
	if r > 0 then
		self.refs[b] = r
		return
	end
	self.refs[b] = nil
	self.used = self.used - 1
	if self.keys[b] then
		self.tick = self.tick + 1
		self.info[b].used = self.tick
		self.cached = self.cached + 1
	elseif b < self.low then
		self.low = b
	end
end

--- The reference count of a block (0 when free or cached).
-- @tparam integer b the block id
-- @treturn integer the count
function P:refs_of(b)
	return self.refs[b] or 0
end

--- Give a full block its key, so other sequences can find it.
-- A key that another block has already stays with that block.
-- @tparam integer b the block id, with a reference
-- @tparam string key the key, from `key`
-- @tparam string parent the key before it (the root key for the first
-- block)
-- @tparam integer depth the block index in its sequence, from 1
-- @tparam {integer|string,...} tokens the tokens of the block
-- @treturn boolean true when b got the key
function P:publish(b, key, parent, depth, tokens)
	if self.keys[b] or self.bykey[key] then
		return false
	end
	self.keys[b], self.bykey[key] = key, b
	self.info[b] = { parent = parent, depth = depth, tokens = tokens, used = self.tick }
	self.kids[parent] = self.kids[parent] or {}
	self.kids[parent][b] = true
	return true
end

--- Keep the partial last block of a sequence cached when the sequence
-- lets it go, so a prompt that shares its tokens can copy its cells.
-- The block gets no key: only `longest_kid` finds it.
-- @tparam integer b the block id, with a reference
-- @tparam string parent the key of the full block before it (the root
-- key for the first block)
-- @tparam integer depth the block index in its sequence, from 1
-- @tparam {integer|string,...} tokens the tokens it holds, fewer than
-- a block
-- @treturn boolean true when b is now cached on release
function P:publish_part(b, parent, depth, tokens)
	if self.keys[b] or #tokens == 0 then
		return false
	end
	-- A marker, not a hash: no lookup finds the block.
	self.keys[b] = "part"
	self.info[b] = { parent = parent, depth = depth, tokens = tokens, used = self.tick }
	self.kids[parent] = self.kids[parent] or {}
	self.kids[parent][b] = true
	return true
end

--- The key of a block, or nil.
-- @tparam integer b the block id
-- @treturn string the key
function P:key_of(b)
	return self.keys[b]
end

--- Find the block with a key.
-- @tparam string key the key
-- @treturn integer the block id, or nil
function P:lookup(key)
	return self.bykey[key]
end

--- The keyed block after parent whose tokens share the longest prefix
-- with tokens[i..], and the length of that prefix.
-- @tparam string parent the key before the block
-- @tparam {integer|string,...} tokens the tokens
-- @tparam integer i the first token to compare
-- @tparam integer max the most tokens to compare
-- @treturn integer the block id, or nil
-- @treturn integer the shared token count
function P:longest_kid(parent, tokens, i, max)
	local best, bn = nil, 0
	for b in pairs(self.kids[parent] or {}) do
		local t, k = self.info[b].tokens, 0
		while k < max and t[k + 1] == tokens[i + k] do
			k = k + 1
		end
		if k > bn then
			best, bn = b, k
		end
	end
	return best, bn
end

--- Set the usable blocks after the tensors grew.
-- @tparam integer cap the usable block count, at most `n`
function P:set_cap(cap)
	assert(cap <= self.n, "kvpool: cap past the pool")
	self.cap = cap
	self.low = math.min(self.low, cap)
end

return M
