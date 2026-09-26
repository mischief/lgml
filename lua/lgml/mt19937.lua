--- std::mt19937 with the libstdc++ uniform_real_distribution<float> and
-- uniform_int_distribution<int32_t>, so a seed gives the same stream as
-- C++ code built with GCC.
-- @classmod lgml.mt19937
local M = {}
M.__index = M

local N, MM = 624, 397
local MASK = 0xffffffff

--- Make a generator.
-- @tparam[opt=5489] integer seed the seed; the low 32 bits are used
-- @treturn lgml.mt19937 a new generator
function M.new(seed)
	local s = { [0] = (seed or 5489) & MASK }
	for i = 1, N - 1 do
		local p = s[i - 1]
		s[i] = (1812433253 * (p ~ (p >> 30)) + i) & MASK
	end
	return setmetatable({ s = s, i = N }, M)
end

local function twist(s)
	for i = 0, N - 1 do
		local y = (s[i] & 0x80000000) | (s[(i + 1) % N] & 0x7fffffff)
		local v = s[(i + MM) % N] ~ (y >> 1)
		if y & 1 ~= 0 then
			v = v ~ 0x9908b0df
		end
		s[i] = v
	end
end

--- The next 32-bit output.
-- @treturn integer a value in [0, 2^32)
function M:next()
	if self.i >= N then
		twist(self.s)
		self.i = 0
	end
	local y = self.s[self.i]
	self.i = self.i + 1
	y = y ~ (y >> 11)
	y = y ~ ((y << 7) & 0x9d2c5680)
	y = y ~ ((y << 15) & 0xefc60000)
	return (y ~ (y >> 18)) & MASK
end

local function f32(x)
	return (string.unpack("<f", string.pack("<f", x)))
end

local ONE_BELOW = f32(1 - 2 ^ -24)

--- A float in [0, 1): generate_canonical<float, 24> on one output.
-- @treturn number a float32 value in [0, 1)
function M:uniform()
	local r = f32(self:next()) / 4294967296.0
	if r >= 1 then
		r = ONE_BELOW
	end
	return r
end

--- An integer in [0, n): Lemire's method with a 64-bit product.
-- @tparam integer n the bound, 1 to 2^32
-- @treturn integer a value in [0, n)
function M:below(n)
	local p = self:next() * n
	local low = p & MASK
	if low < n then
		local threshold = ((1 << 32) - n) % n
		while low < threshold do
			p = self:next() * n
			low = p & MASK
		end
	end
	return p >> 32
end

return M
