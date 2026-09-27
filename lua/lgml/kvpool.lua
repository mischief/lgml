--- Block bookkeeping of a shared KV pool.
-- The pool has `blocks` blocks of `size` cells each. A sequence takes
-- blocks as it grows and gives them back at the end. A block has a
-- reference count; it is free at 0. Only blocks below `cap` are
-- usable: the owner of the tensors raises `cap` when it grows them.
-- This module holds no tensors; `lgml.transformer` does.
-- @module lgml.kvpool

local M = {}
local P = {}
P.__index = P

--- Make the bookkeeping of a pool.
-- @tparam table opts `blocks` (the count), `size` (cells per block)
-- and `cap` (usable blocks now, default all)
-- @treturn Pool the pool
function M.new(opts)
	return setmetatable({
		n = opts.blocks, size = opts.size, cap = opts.cap or opts.blocks,
		refs = {}, used = 0, low = 0,
	}, P)
end

--- The bookkeeping of a pool, from `new`.
-- Fields: `n` (blocks), `size` (cells per block), `cap` (usable
-- blocks), `used` (blocks with references).
-- @type Pool

--- Take the lowest free block below cap.
-- Low blocks first keep the cells of a sequence close together, so
-- attention reads a short window of the pool.
-- @treturn[1] integer the block id, from 0, with one reference
-- @return[2] nil when no block is free below cap
function P:take()
	for b = self.low, self.cap - 1 do
		if not self.refs[b] then
			self.refs[b] = 1
			self.used = self.used + 1
			self.low = b + 1
			return b
		end
	end
	self.low = self.cap
	return nil
end

--- Add a reference to a block.
-- @tparam integer b the block id
function P:ref(b)
	self.refs[b] = assert(self.refs[b], "kvpool: ref of a free block") + 1
end

--- Drop a reference to a block. The block is free at 0.
-- @tparam integer b the block id
function P:unref(b)
	local r = assert(self.refs[b], "kvpool: unref of a free block") - 1
	if r > 0 then
		self.refs[b] = r
		return
	end
	self.refs[b] = nil
	self.used = self.used - 1
	if b < self.low then
		self.low = b
	end
end

--- The reference count of a block (0 when free).
-- @tparam integer b the block id
-- @treturn integer the count
function P:refs_of(b)
	return self.refs[b] or 0
end

--- Set the usable blocks after the tensors grew.
-- @tparam integer cap the usable block count, at most `n`
function P:set_cap(cap)
	assert(cap <= self.n, "kvpool: cap past the pool")
	self.cap = cap
	self.low = math.min(self.low, cap)
end

return M
