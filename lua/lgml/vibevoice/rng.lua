--- MT19937 and the normal draws of torch.randn on the CPU.
-- A seed gives the same diffusion noise as the reference implementation.
-- @classmod lgml.vibevoice.rng
local M = {}
M.__index = M

local MASK = 0xffffffff

local function f32(x)
	return (string.unpack("<f", string.pack("<f", x)))
end

--- Make a generator.
-- @int seed seed; the low 32 bits are used
-- @treturn lgml.vibevoice.rng the generator
-- @static
function M.new(seed)
	local mt = { [0] = seed & MASK }
	for i = 1, 623 do
		local p = mt[i - 1]
		mt[i] = (1812433253 * (p ~ (p >> 30)) + i) & MASK
	end
	return setmetatable({ mt = mt, i = 624 }, M)
end

local function twist(mt)
	for i = 0, 623 do
		local y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff)
		local v = mt[(i + 397) % 624] ~ (y >> 1)
		if y & 1 ~= 0 then
			v = v ~ 0x9908b0df
		end
		mt[i] = v
	end
end

--- Next 32-bit output.
-- @treturn int value in [0, 2^32)
function M:next()
	if self.i >= 624 then
		twist(self.mt)
		self.i = 0
	end
	local y = self.mt[self.i]
	self.i = self.i + 1
	y = y ~ (y >> 11)
	y = y ~ ((y << 7) & 0x9d2c5680)
	y = y ~ ((y << 15) & 0xefc60000)
	y = y ~ (y >> 18)
	return y & MASK
end

--- Uniform in [0, 1) from the low 24 bits, as torch does for f32.
-- @treturn number the value
function M:uniform()
	return (self:next() & 0xffffff) / 16777216.0
end

-- Box-Muller on one block of 16 uniforms, in place.
local function normal16(d, o)
	for j = 0, 7 do
		local r = f32(math.sqrt(f32(-2 * f32(math.log(f32(1 - d[o + j]))))))
		local th = f32(2 * f32(math.pi) * d[o + j + 8])
		d[o + j] = f32(r * math.cos(th))
		d[o + j + 8] = f32(r * math.sin(th))
	end
end

--- n normal values, in the order torch.randn fills them.
-- The values are rounded to f32.
-- @int n count, at least 16
-- @treturn {number,...} the values
-- @raise when n is less than 16
function M:normal(n)
	assert(n >= 16, "normal: need at least 16 values")
	local d = {}
	for i = 1, n do
		d[i] = self:uniform()
	end
	local i = 1
	while i + 15 <= n do
		normal16(d, i)
		i = i + 16
	end
	if i <= n then
		for j = n - 15, n do
			d[j] = self:uniform()
		end
		normal16(d, n - 15)
	end
	return d
end

return M
