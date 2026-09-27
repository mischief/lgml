--- KV cache element types for the chat models.
-- The same choices as llama.cpp -ctk and -ctv. A quantized cache needs
-- flash attention: the soft_max path reads V transposed, and not every
-- backend can matmul a strided quantized K.
-- As in llama.cpp, a quantized cache stores keys and values turned by
-- a Walsh-Hadamard rotation, which spreads outliers before rounding.
-- Set LGML_ATTN_ROT_DISABLE=1 to turn it off.
-- @module lgml.kv
local lgml = require("lgml")

local M = {}

--- Allowed cache types, as a set.
M.types = {
	f32 = true, f16 = true, bf16 = true,
	q8_0 = true, q4_0 = true, q4_1 = true, q5_0 = true, q5_1 = true, iq4_nl = true,
}

--- Quantized cache types, as a set.
M.quantized = { q8_0 = true, q4_0 = true, q4_1 = true, q5_0 = true, q5_1 = true, iq4_nl = true }

--- Get the K and V cache types from options.
-- opts.type sets both; opts.type_k and opts.type_v override it.
-- @tparam[opt] table opts the cache options
-- @tparam[opt] {integer,...} dims head sizes the cache must hold
-- @treturn string the K type (default "f16")
-- @treturn string the V type (default "f16")
-- @raise on an unknown type, or a type whose block does not divide a
-- head size
function M.types_of(opts, dims)
	opts = opts or {}
	local tk = opts.type_k or opts.type or "f16"
	local tv = opts.type_v or opts.type or "f16"
	for _, t in ipairs({ tk, tv }) do
		if not M.types[t] then
			error(("kv: unknown cache type %q"):format(tostring(t)), 3)
		end
		for _, d in ipairs(dims or {}) do
			if not pcall(lgml.row_size, t, d) then
				error(("kv: a %s cache does not fit head size %d"):format(t, d), 3)
			end
		end
	end
	return tk, tv
end

--- Check that attention can read a cache.
-- @tparam table cache a cache with type_v
-- @tparam boolean flash_attn true when the pass uses flash attention
-- @raise when K or V is quantized and flash_attn is off
function M.check(cache, flash_attn)
	if cache and not flash_attn then
		for _, ty in ipairs({ cache.type_k, cache.type_v }) do
			if M.quantized[ty] then
				error(("kv: a %s cache needs flash attention"):format(ty), 3)
			end
		end
	end
end

--- Views of the first n_kv rows of the K and V cache nodes.
-- @param g the graph
-- @tparam table cache the cache, with type_k and type_v
-- @param kw the K cache node after the write, (n_ctx, Hkv * D)
-- @param vw the V cache node after the write
-- @tparam integer n_kv rows to read
-- @tparam integer Hkv KV heads
-- @tparam integer D head size
-- @tparam[opt] integer lo the first row (default 0)
-- @return k and v, each (n_kv, Hkv, D)
function M.views(g, cache, kw, vw, n_kv, Hkv, D, lo)
	local rk, rv = lgml.row_size(cache.type_k, Hkv * D), lgml.row_size(cache.type_v, Hkv * D)
	local k = g:view(kw, { n_kv, Hkv, D }, { rk, lgml.row_size(cache.type_k, D) }, (lo or 0) * rk)
	local v = g:view(vw, { n_kv, Hkv, D }, { rv, lgml.row_size(cache.type_v, D) }, (lo or 0) * rv)
	return k, v
end

local rot_off = os.getenv("LGML_ATTN_ROT_DISABLE") == "1"

-- The n x n Hadamard matrix, made once per cache.
local function hadamard(cache, n)
	cache.rot = cache.rot or {}
	if not cache.rot[n] then
		local h = cache.be:tensor("f32", { n, n })
		h:fill("hadamard")
		cache.rot[n] = h
	end
	return cache.rot[n]
end

--- The K rotation for head size D, or nil.
-- It is the largest power of 2 from 64 up that divides D, as llama.cpp.
-- @tparam table cache the cache (with be and type_k)
-- @tparam integer D the head size
-- @return an f32 (n, n) leaf, or nil when K is not quantized
function M.rot_k(cache, D)
	if rot_off or not cache or not M.quantized[cache.type_k] or D % 64 ~= 0 then
		return nil
	end
	local n = 64
	while D % (n * 2) == 0 do
		n = n * 2
	end
	return hadamard(cache, n)
end

--- The V rotation for head size D, or nil. It is always 64 wide.
-- @tparam table cache the cache (with be and type_v)
-- @tparam integer D the head size
-- @return an f32 (64, 64) leaf, or nil when V is not quantized
function M.rot_v(cache, D)
	if rot_off or not cache or not M.quantized[cache.type_v] or D % 64 ~= 0 then
		return nil
	end
	return hadamard(cache, 64)
end

--- Turn every run of n values along the last dimension by rot.
-- The rotation is its own inverse. t must be contiguous.
-- @param g the graph
-- @param t the tensor
-- @param rot an (n, n) rotation, or nil to return t
-- @return the turned tensor, with the shape of t
function M.rotate(g, t, rot)
	if not rot then
		return t
	end
	local shape, total = t:shape(), 1
	for _, d in ipairs(shape) do
		total = total * d
	end
	local n = rot:shape()[1]
	return g:reshape(g:hadamard(rot, g:reshape(t, { total // n, n })), shape)
end

--- Make k or v fit flash attention: f16 and quantized types pass
-- through; others become f16.
-- @param g the graph
-- @param t the tensor
-- @return t or an f16 cast of it
function M.for_flash(g, t)
	local ty = t:type()
	if ty == "f16" or M.quantized[ty] then
		return t
	end
	return g:cast(t, "f16")
end

return M
