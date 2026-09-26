--- Tensors with a torch-like feel on top of lgml.
-- Shapes are in torch order, at most 4 dimensions. Indices start at 1.
--
-- Ops do not run at once. They record into one lazy graph per device,
-- and each result is pending. A value appears when you print, `get` or
-- `eval` a tensor. Then the graph runs once, and each live pending
-- result moves into its own device buffer. Data stays on the device.
--
-- These calls run the pending work of a device: `nd.eval`, `nd.sync`,
-- `x:eval`, `x:get`, `x:item`, `x:tolist`, `x:bytes`, `x:raw`,
-- `tostring(x)` and `x:to(device)`. A device also flushes by itself
-- before its graph holds more than 1500 ggml tensors. Only live tensors
-- get a buffer, so drop references to intermediates you do not need.
--
--     local nd = require("lgml.nd")
--     local a = nd.tensor{ { 1, 2 }, { 3, 4 } }
--     print(a:mm(a:t()) + 1)
-- @module lgml.nd
local lgml = require("lgml")

local M = {}
local T = {}
local mt = { __name = "nd.tensor" }

-- Flush the graph before it holds this many ggml tensors (limit 2048).
local LIMIT = 1500

local devices = {}
local current

local function device(name)
	local d = devices[name or "best"]
	if not d then
		local be = assert(lgml.backend(name))
		d = { name = be:name(), be = be, pending = setmetatable({}, { __mode = "k" }), count = 0 }
		devices[name or "best"] = d
	end
	return d
end

--- Devices and evaluation.
-- A device is a ggml backend, for example "CPU". `lgml.devices()` lists
-- them. Each tensor lives on one device. Binary ops need both operands
-- on the same device; use `x:to(name)` to copy between devices.
-- New tensors go to the default device unless `opts.device` names one.
-- @section devices

--- Set or get the default device.
-- Without a name, the default is the best backend that ggml finds.
-- @tparam[opt] string name backend name, for example "CPU"
-- @treturn string name of the default device
-- @raise if no backend has this name
function M.device(name)
	if name then
		current = device(name)
	end
	current = current or device(nil)
	return current.name
end

local function cur()
	current = current or device(nil)
	return current
end

local function wrap(d, u, pending)
	local x = setmetatable({ u = u, d = d }, mt)
	if pending then
		d.pending[x] = true
	end
	return x
end

--- Check if a value is an nd tensor.
-- @param x any value
-- @treturn boolean true if x is an nd tensor
function M.is_tensor(x)
	return getmetatable(x) == mt
end

-- Run the graph of device d and move every live pending result into
-- its own buffer.
local function flush(d)
	local g = d.g
	if not g then
		return
	end
	local list, outs = {}, {}
	for x in pairs(d.pending) do
		list[#list + 1] = x
	end
	for i, x in ipairs(list) do
		local leaf = d.be:tensor(x.u:type(), x.u:shape())
		outs[i] = g:cpy(x.u, leaf)
		x.leaf = leaf
	end
	if #outs > 0 then
		g:compute(table.unpack(outs))
	end
	for _, x in ipairs(list) do
		x.u, x.leaf = x.leaf, nil
		d.pending[x] = nil
	end
	d.g, d.count = nil, 0
	g:close()
end

--- Compute these tensors.
-- This runs all pending work on the devices of the arguments.
-- Arguments that are not tensors are ignored.
-- @param ... tensors
function M.eval(...)
	local seen = {}
	for i = 1, select("#", ...) do
		local x = select(i, ...)
		if M.is_tensor(x) and not seen[x.d] then
			seen[x.d] = true
			flush(x.d)
		end
	end
end

--- Compute all pending work on every device.
function M.sync()
	for _, d in pairs(devices) do
		flush(d)
	end
end

-- Add a node made by f(graph) that costs n ggml tensors.
local function apply(d, n, f)
	if d.count + n > LIMIT then
		flush(d)
	end
	d.g = d.g or d.be:graph()
	d.count = d.count + n
	return wrap(d, f(d.g), true)
end

--- Add a custom op: a new pending tensor on the device of x.
-- f gets the lgml graph of the device and returns an lgml tensor. Use
-- the field `x.u` to get the lgml tensor of an nd tensor inside f.
-- @tparam tensor x tensor that picks the device
-- @tparam function f builds the node, f(graph) returns an lgml tensor
-- @tparam[opt=1] int cost number of ggml tensors that f adds
-- @treturn tensor the pending result
function M.wrap_op(x, f, cost)
	return apply(x.d, cost or 1, f)
end

local function shape_args(...)
	local a = ...
	if type(a) == "table" and not M.is_tensor(a) then
		return a
	end
	return { ... }
end

local function numel(shape)
	local n = 1
	for _, v in ipairs(shape) do
		n = n * v
	end
	return n
end

--- Constructors.
-- Constructors take an optional table `opts` with two keys.
-- opts.dtype: "f32" (default), "f16", "bf16" or "i32".
-- opts.device: device name; the default device if absent.
-- A shape is a table of sizes, for example `{ 2, 3 }`.
-- @section constructors

local seed_state = 0x5eed

--- Set the seed for `rand` and `randn`.
-- Each call to `rand` or `randn` moves the seed forward.
-- @tparam int n seed
-- @raise if n is not an integer
function M.manual_seed(n)
	seed_state = math.tointeger(n) or error("seed must be an integer", 2)
end

local function next_seed()
	seed_state = (seed_state * 6364136223846793005 + 1442695040888963407) & 0x7fffffffffffffff
	return seed_state
end

local function opts_of(o)
	o = o or {}
	local d = o.device and device(o.device) or cur()
	return d, o.dtype or "f32"
end

local function filled(shape, o, kind, a, b, seed)
	local d, dtype = opts_of(o)
	local u = d.be:tensor(dtype, shape)
	u:fill(kind, a, b, seed)
	return wrap(d, u)
end

--- A tensor with memory that is not set.
-- @function empty
-- @tparam table shape sizes
-- @tparam[opt] table opts dtype and device
-- @treturn tensor
function M.empty(shape, o)
	local d, dtype = opts_of(o)
	return wrap(d, d.be:tensor(dtype, shape))
end

--- A tensor filled with one value.
-- @function full
-- @tparam table shape sizes
-- @tparam number v value
-- @tparam[opt] table opts dtype and device
-- @treturn tensor
function M.full(shape, v, o)
	return filled(shape, o, "const", v, 0)
end

--- A tensor filled with zeros.
-- @function zeros
-- @tparam table shape sizes
-- @tparam[opt] table opts dtype and device
-- @treturn tensor
function M.zeros(shape, o)
	return filled(shape, o, "const", 0, 0)
end

--- A tensor filled with ones.
-- @function ones
-- @tparam table shape sizes
-- @tparam[opt] table opts dtype and device
-- @treturn tensor
function M.ones(shape, o)
	return filled(shape, o, "const", 1, 0)
end

--- A 1-D tensor of evenly spaced values.
-- Call it as `nd.arange(n [, opts])` for 0 to n-1, or as
-- `nd.arange(start, stop [, step [, opts]])`. stop is exclusive.
-- @function arange
-- @tparam number start first value, or n in the short form
-- @tparam[opt] number stop end value, exclusive
-- @tparam[opt=1] number step distance between values
-- @tparam[opt] table opts dtype and device
-- @treturn tensor shape `{ ceil((stop - start) / step) }`
function M.arange(a, b, step, o)
	if b == nil or type(b) == "table" then
		a, b, o = 0, a, b
	end
	step = step or 1
	local n = math.ceil((b - a) / step)
	return filled({ n }, o, "range", a, step)
end

--- A tensor of uniform random values in [0, 1).
-- @function rand
-- @tparam table shape sizes
-- @tparam[opt] table opts dtype and device
-- @treturn tensor
-- @see manual_seed
function M.rand(shape, o)
	return filled(shape, o, "uniform", 0, 1, next_seed())
end

--- A tensor of normal random values, mean 0 and standard deviation 1.
-- @function randn
-- @tparam table shape sizes
-- @tparam[opt] table opts dtype and device
-- @treturn tensor
-- @see manual_seed
function M.randn(shape, o)
	return filled(shape, o, "normal", 0, 1, next_seed())
end

local function shape_of(data)
	local shape, flat = {}, {}
	local node = data
	while type(node) == "table" do
		shape[#shape + 1] = #node
		node = node[1]
	end
	local function walk(v, depth)
		if depth > #shape then
			flat[#flat + 1] = v
			return
		end
		if type(v) ~= "table" or #v ~= shape[depth] then
			error("nd.tensor: ragged nested table", 4)
		end
		for _, e in ipairs(v) do
			walk(e, depth + 1)
		end
	end
	walk(data, 1)
	return shape, flat
end

--- A tensor from Lua data.
-- A number gives shape `{ 1 }`. Nested tables give one dimension per
-- level, for example `nd.tensor{ { 1, 2 }, { 3, 4 } }` has shape `{ 2, 2 }`.
-- @function tensor
-- @tparam number|table data a number or nested tables of numbers
-- @tparam[opt] table opts dtype and device
-- @treturn tensor
-- @raise if the nested tables are ragged
function M.tensor(data, o)
	local d, dtype = opts_of(o)
	if type(data) == "number" then
		data = { data }
	end
	local shape, flat = shape_of(data)
	return wrap(d, d.be:tensor(dtype, shape, flat))
end

-- operands

local function scalar(d, v)
	local u = d.be:tensor("f32", { 1 })
	u:fill("const", v, 0)
	return wrap(d, u)
end

local function ne4(shape)
	local ne = { 1, 1, 1, 1 }
	for i = 1, #shape do
		ne[#shape - i + 1] = shape[i]
	end
	return ne
end

-- True if x tiles to y (ggml broadcast rule).
local function tiles(xs, ys)
	local a, b = ne4(xs), ne4(ys)
	for i = 1, 4 do
		if b[i] % a[i] ~= 0 then
			return false
		end
	end
	return true
end

local FLOAT = { f32 = true, f16 = true, bf16 = true }

local function as_f32(x)
	if x.u:type() == "f32" then
		return x
	end
	return apply(x.d, 1, function(g)
		return g:cast(x.u, "f32")
	end)
end

-- Two operands on one device, same type, b tiling to a. swap is true
-- when the operands changed places.
local function pair(a, b, commutes)
	local d = M.is_tensor(a) and a.d or b.d
	if not M.is_tensor(a) then
		a = scalar(d, a)
	end
	if not M.is_tensor(b) then
		b = scalar(d, b)
	end
	if a.d ~= b.d then
		error("tensors on different devices: " .. a.d.name .. " and " .. b.d.name, 3)
	end
	if a.u:type() ~= b.u:type() then
		a, b = as_f32(a), as_f32(b)
	end
	local as, bs = a.u:shape(), b.u:shape()
	if tiles(bs, as) then
		return d, a, b
	elseif commutes and tiles(as, bs) then
		return d, b, a
	elseif tiles(as, bs) then
		local ra = apply(d, 1, function(g)
			return g:tile(a.u, b.u)
		end)
		return d, ra, b
	end
	error("shapes do not broadcast: {" .. table.concat(as, ",") .. "} and {" ..
	    table.concat(bs, ",") .. "}", 3)
end

local function binary(name, commutes)
	return function(a, b)
		local d
		d, a, b = pair(a, b, commutes)
		return apply(d, 1, function(g)
			return g[name](g, a.u, b.u)
		end)
	end
end

local add, sub, mul, div = binary("add", true), binary("sub"), binary("mul", true), binary("div")

--- Tensor object.
-- Ops on a tensor return new pending tensors; tensors do not change in
-- place. Arithmetic works with two tensors or a tensor and a number.
-- Operands broadcast by the ggml rule: each size of the smaller operand
-- must divide the matching size of the larger, aligned from the last
-- dimension. If the dtypes differ, both operands become f32.
--
-- `x[i]` is `x:select(1, i)`; negative i counts from the end. `#x` is
-- the size of the first dimension. `tostring(x)` and `print(x)` show
-- the values, dtype and device. Other keys look up methods.
-- The field `x.u` holds the lgml tensor, which can still be pending.
-- @type tensor

--- `a + b`: elementwise add with broadcast.
-- @function tensor.__add
-- @tparam tensor|number a
-- @tparam tensor|number b
-- @treturn tensor
-- @raise if the shapes do not broadcast or the devices differ
function mt.__add(a, b)
	return add(a, b)
end

--- `a - b`: elementwise subtract with broadcast.
-- @function tensor.__sub
-- @tparam tensor|number a
-- @tparam tensor|number b
-- @treturn tensor
-- @raise if the shapes do not broadcast or the devices differ
function mt.__sub(a, b)
	return sub(a, b)
end

--- `a * b`: elementwise multiply with broadcast.
-- A number operand uses `scale`.
-- @function tensor.__mul
-- @tparam tensor|number a
-- @tparam tensor|number b
-- @treturn tensor
-- @raise if the shapes do not broadcast or the devices differ
function mt.__mul(a, b)
	if type(b) == "number" then
		return a:scale(b)
	elseif type(a) == "number" then
		return b:scale(a)
	end
	return mul(a, b)
end

--- `a / b`: elementwise divide with broadcast.
-- A number divisor uses `scale` with 1 / b.
-- @function tensor.__div
-- @tparam tensor|number a
-- @tparam tensor|number b
-- @treturn tensor
-- @raise if the shapes do not broadcast or the devices differ
function mt.__div(a, b)
	if type(b) == "number" then
		return a:scale(1 / b)
	end
	return div(a, b)
end

--- `-a`: same as `a:neg()`.
-- @function tensor.__unm
-- @treturn tensor
function mt.__unm(a)
	return a:neg()
end

--- `#a`: the size of the first dimension.
-- @function tensor.__len
-- @treturn int
function mt.__len(a)
	return a.u:shape()[1]
end

-- methods

local function unary(name, cost)
	return function(x, ...)
		local args = table.pack(...)
		return apply(x.d, cost or 1, function(g)
			return g[name](g, x.u, table.unpack(args, 1, args.n))
		end)
	end
end

--- Elementwise SiLU, x * sigmoid(x).
-- @function tensor:silu
-- @treturn tensor

--- Elementwise GELU, tanh approximation.
-- @function tensor:gelu
-- @treturn tensor

--- Elementwise GELU with the exact erf form.
-- @function tensor:gelu_erf
-- @treturn tensor

--- Elementwise ReLU, max(x, 0).
-- @function tensor:relu
-- @treturn tensor

--- Elementwise ELU.
-- @function tensor:elu
-- @treturn tensor

--- Elementwise Hyperbolic tangent.
-- @function tensor:tanh
-- @treturn tensor

--- Elementwise Logistic sigmoid.
-- @function tensor:sigmoid
-- @treturn tensor

--- Elementwise Exponential.
-- @function tensor:exp
-- @treturn tensor

--- Elementwise Absolute value.
-- @function tensor:abs
-- @treturn tensor

--- Elementwise Negation.
-- @function tensor:neg
-- @treturn tensor

--- Elementwise Sine.
-- @function tensor:sin
-- @treturn tensor

--- Elementwise Cosine.
-- @function tensor:cos
-- @treturn tensor

--- Elementwise Square, x * x.
-- @function tensor:sqr
-- @treturn tensor

--- Elementwise Square root.
-- @function tensor:sqrt
-- @treturn tensor

for _, name in ipairs({ "silu", "gelu", "gelu_erf", "relu", "elu", "tanh", "sigmoid",
    "exp", "abs", "neg", "sin", "cos", "sqr", "sqrt" }) do
	T[name] = unary(name)
end

--- The shape.
-- @function tensor:shape
-- @treturn table sizes in torch order
function T.shape(x)
	return x.u:shape()
end

--- The shape, or the size of one dimension.
-- @function tensor:size
-- @tparam[opt] int dim dimension, negative counts from the end
-- @treturn int|table size of dim, or the shape without dim
function T.size(x, dim)
	local s = x.u:shape()
	if dim then
		return s[dim < 0 and #s + dim + 1 or dim]
	end
	return s
end

--- The number of dimensions.
-- @function tensor:dim
-- @treturn int
function T.dim(x)
	return #x.u:shape()
end

--- The number of elements.
-- @function tensor:numel
-- @treturn int
function T.numel(x)
	return numel(x.u:shape())
end

--- The element type, for example "f32", "f16", "bf16" or "i32".
-- @function tensor:dtype
-- @treturn string
function T.dtype(x)
	return x.u:type()
end

--- The name of the device that holds the tensor.
-- @function tensor:device
-- @treturn string
function T.device(x)
	return x.d.name
end

--- Multiply by a number.
-- A tensor that is not float becomes f32 first.
-- @function tensor:scale
-- @tparam number s factor
-- @treturn tensor contiguous result
function T.scale(x, s)
	x = FLOAT[x.u:type()] and x or as_f32(x)
	return apply(x.d, 2, function(g)
		return g:scale(g:cont(x.u), s)
	end)
end

--- A contiguous copy.
-- @function tensor:contiguous
-- @treturn tensor
function T.contiguous(x)
	return apply(x.d, 1, function(g)
		return g:cont(x.u)
	end)
end

local function reshape_to(x, shape)
	local s, n, hole = {}, 1, nil
	for i, v in ipairs(shape) do
		if v == -1 then
			hole = i
		else
			n = n * v
		end
		s[i] = v
	end
	if hole then
		s[hole] = x:numel() // n
	end
	return apply(x.d, 2, function(g)
		local ok, r = pcall(g.reshape, g, x.u, s)
		if ok then
			return r
		end
		return g:reshape(g:cont(x.u), s)
	end)
end

--- A tensor with the same data in a new shape.
-- Give the sizes as arguments or as one table. One size can be -1; the
-- other sizes and the element count set it. The data is made
-- contiguous first when the view needs it. `x:view` is the same method.
-- @function tensor:reshape
-- @tparam int|table ... sizes
-- @treturn tensor
function T.reshape(x, ...)
	return reshape_to(x, shape_args(...))
end

T.view = T.reshape

--- Same as `reshape`.
-- @function tensor:view
-- @tparam int|table ... sizes
-- @treturn tensor

--- Reorder the dimensions.
-- Result dimension i is dimension dims[i] of x, from 1. Give dims as
-- arguments or as one table. The result is a view.
-- @function tensor:permute
-- @tparam int|table ... dims
-- @treturn tensor
function T.permute(x, ...)
	local p = shape_args(...)
	return apply(x.d, 1, function(g)
		return g:permute(x.u, p)
	end)
end

--- Swap two dimensions.
-- @function tensor:transpose
-- @tparam int a dimension, negative counts from the end
-- @tparam int b dimension, negative counts from the end
-- @treturn tensor a view
function T.transpose(x, a, b)
	local n = x:dim()
	a, b = a < 0 and n + a + 1 or a, b < 0 and n + b + 1 or b
	local p = {}
	for i = 1, n do
		p[i] = i
	end
	p[a], p[b] = p[b], p[a]
	return x:permute(p)
end

--- Swap the last two dimensions.
-- @function tensor:t
-- @treturn tensor a view
function T.t(x)
	return x:transpose(-1, -2)
end

--- A slice of len elements along one dimension.
-- The result is a view when the strides allow it; otherwise it uses a
-- contiguous copy.
-- @function tensor:narrow
-- @tparam int dim dimension, negative counts from the end
-- @tparam int start first index, from 1
-- @tparam int len number of elements
-- @treturn tensor
-- @raise if the slice is out of range
function T.narrow(x, dim, start, len)
	local shape, strides = x.u:shape(), x.u:strides()
	local n = #shape
	dim = dim < 0 and n + dim + 1 or dim
	if start < 1 or len < 1 or start + len - 1 > shape[dim] then
		error("narrow out of range", 2)
	end
	local s = table.move(shape, 1, n, 1, {})
	s[dim] = len
	local off = (start - 1) * strides[dim]
	return apply(x.d, 2, function(g)
		local base = x.u
		local ok, r = pcall(g.view, g, base, s, table.move(strides, 1, n - 1, 1, {}), off)
		if ok then
			return r
		end
		base = g:cont(base)
		local st = base:strides()
		return g:view(base, s, table.move(st, 1, n - 1, 1, {}), (start - 1) * st[dim])
	end)
end

--- Drop dimension dim, keeping index i.
-- A 1-D tensor gives shape `{ 1 }`. `x[i]` is `x:select(1, i)`.
-- @function tensor:select
-- @tparam int dim dimension, negative counts from the end
-- @tparam int i index from 1, negative counts from the end
-- @treturn tensor
function T.select(x, dim, i)
	local shape = x.u:shape()
	local n = #shape
	dim = dim < 0 and n + dim + 1 or dim
	i = i < 0 and shape[dim] + i + 1 or i
	local y = x:narrow(dim, i, 1)
	if n == 1 then
		return y
	end
	table.remove(shape, dim)
	return reshape_to(y, shape)
end

--- Convert to a dtype, or copy to a device.
-- A dtype ("f32", "f16", "bf16", "i32") adds a pending cast. Any other
-- string names a device: the source device flushes, and the data copies
-- into a new buffer on the target. A tensor that already matches
-- returns itself.
-- @function tensor:to
-- @tparam string what dtype or device name
-- @treturn tensor
-- @raise if what names no backend
function T.to(x, what)
	if FLOAT[what] or what == "i32" then
		if x.u:type() == what then
			return x
		end
		return apply(x.d, 1, function(g)
			return g:cast(x.u, what)
		end)
	end
	local d = device(what)
	if d == x.d then
		return x
	end
	flush(x.d)
	local u = d.be:tensor(x.u:type(), x.u:shape())
	u:set(x.u:bytes())
	return wrap(d, u)
end

--- Same as `x:to("f32")`.
-- @function tensor:float
-- @treturn tensor
function T.float(x)
	return x:to("f32")
end

--- Same as `x:to("f16")`.
-- @function tensor:half
-- @treturn tensor
function T.half(x)
	return x:to("f16")
end

--- Matrix product, as torch matmul.
-- x is (..., n, k) and y is (k, m); the result is (..., n, m).
-- Operands that are not float become f32. `x:matmul` is the same method.
-- @function tensor:mm
-- @tparam tensor y right operand
-- @treturn tensor
function T.mm(x, y)
	x, y = FLOAT[x.u:type()] and x or as_f32(x), FLOAT[y.u:type()] and y or as_f32(y)
	return apply(x.d, 3, function(g)
		return g:matmul(x.u, y.u)
	end)
end

T.matmul = T.mm

--- Same as `mm`.
-- @function tensor:matmul
-- @tparam tensor y right operand
-- @treturn tensor

--- Linear layer: x @ w^T + b.
-- w is stored (out, in) as in torch. x is (..., in).
-- @function tensor:linear
-- @tparam tensor w weight (out, in)
-- @tparam[opt] tensor b bias (out)
-- @treturn tensor (..., out)
function T.linear(x, w, b)
	local y = apply(x.d, 1, function(g)
		return g:mul_mat(w.u, x.u)
	end)
	if b then
		y = y + b
	end
	return y
end

--- Softmax over the last dimension.
-- The result is f32.
-- @function tensor:softmax
-- @tparam[opt] int dim -1 or the last dimension
-- @treturn tensor
-- @raise if dim is not the last dimension
function T.softmax(x, dim)
	if dim and dim ~= -1 and dim ~= x:dim() then
		error("softmax works on the last dimension", 2)
	end
	x = x.u:type() == "f32" and x or as_f32(x)
	return apply(x.d, 2, function(g)
		return g:soft_max(g:cont(x.u))
	end)
end

--- RMS normalization over the last dimension, without a weight.
-- @function tensor:rms_norm
-- @tparam[opt=1e-6] number eps
-- @treturn tensor
function T.rms_norm(x, eps)
	return apply(x.d, 1, function(g)
		return g:rms_norm(x.u, eps or 1e-6)
	end)
end

--- Layer normalization over the last dimension, without weight or bias.
-- @function tensor:layer_norm
-- @tparam[opt=1e-5] number eps
-- @treturn tensor
function T.layer_norm(x, eps)
	return apply(x.d, 1, function(g)
		return g:norm(x.u, eps or 1e-5)
	end)
end

--- Limit each value to [lo, hi].
-- @function tensor:clamp
-- @tparam[opt] number lo lower limit, none if nil
-- @tparam[opt] number hi upper limit, none if nil
-- @treturn tensor
function T.clamp(x, lo, hi)
	return apply(x.d, 1, function(g)
		return g:clamp(x.u, lo or -math.huge, hi or math.huge)
	end)
end

-- x:sum([dim]) and x:mean([dim]): all elements, or the last dimension.
local function reduce(name)
	return function(x, dim)
		if dim and dim ~= -1 and dim ~= x:dim() then
			error(name .. " works on all elements or the last dimension", 2)
		end
		x = x.u:type() == "f32" and x or as_f32(x)
		if not dim then
			x = reshape_to(x, { x:numel() })
		end
		local r = apply(x.d, 2, function(g)
			return g[name](g, g:cont(x.u))
		end)
		if not dim then
			return r
		end
		local s = x.u:shape()
		table.remove(s)
		return #s > 0 and reshape_to(r, s) or r
	end
end

T.sum = reduce("sum_rows")
T.mean = reduce("mean")

--- Sum of all elements, or over the last dimension.
-- Without dim the result has shape `{ 1 }`. With dim the last dimension
-- goes away. The result is f32.
-- @function tensor:sum
-- @tparam[opt] int dim -1 or the last dimension
-- @treturn tensor
-- @raise if dim is not the last dimension

--- Mean of all elements, or over the last dimension.
-- Without dim the result has shape `{ 1 }`. With dim the last dimension
-- goes away. The result is f32.
-- @function tensor:mean
-- @tparam[opt] int dim -1 or the last dimension
-- @treturn tensor
-- @raise if dim is not the last dimension

--- Index of the largest value in each row.
-- x has 1 or 2 dimensions. The result is i32, and indices start at 0.
-- A (n, k) input gives shape `{ n }`; a 1-D input gives `{ 1 }`.
-- @function tensor:argmax
-- @treturn tensor
-- @raise if x has more than 2 dimensions
function T.argmax(x)
	x = x.u:type() == "f32" and x or as_f32(x)
	return apply(x.d, 2, function(g)
		return g:argmax(g:cont(x.u))
	end)
end

--- Join two tensors along one dimension.
-- The other sizes must match. If the dtypes differ, both become f32.
-- @function tensor:cat
-- @tparam tensor y second tensor
-- @tparam[opt=1] int dim dimension, negative counts from the end
-- @treturn tensor
function T.cat(x, y, dim)
	local _, a, b = pair(x, y, false)
	dim = dim or 1
	return apply(a.d, 1, function(g)
		return g:concat(a.u, b.u, dim < 0 and a:dim() + dim + 1 or dim)
	end)
end

--- Join a list of tensors along one dimension.
-- @function cat
-- @tparam {tensor,...} list tensors to join, in order
-- @tparam[opt=1] int dim dimension, negative counts from the end
-- @treturn tensor
-- @within Joining
function M.cat(list, dim)
	local r = list[1]
	for i = 2, #list do
		r = r:cat(list[i], dim)
	end
	return r
end

--- Compute the tensor.
-- This runs all pending work on its device.
-- @function tensor:eval
-- @treturn tensor x itself
function T.eval(x)
	flush(x.d)
	return x
end

--- A flat table of values in torch (row-major) order.
-- This runs all pending work on the device first.
-- @function tensor:get
-- @treturn table numbers
function T.get(x)
	flush(x.d)
	return x.u:get()
end

--- The value of a tensor with one element.
-- This runs all pending work on the device first.
-- @function tensor:item
-- @treturn number
-- @raise if the tensor does not have exactly one element
function T.item(x)
	local v = x:get()
	if #v ~= 1 then
		error("item needs a tensor with one element", 2)
	end
	return v[1]
end

--- The values as nested tables, one level per dimension.
-- This runs all pending work on the device first.
-- @function tensor:tolist
-- @treturn table
function T.tolist(x)
	local v, shape = x:get(), x.u:shape()
	local i = 0
	local function build(depth)
		local t = {}
		for k = 1, shape[depth] do
			if depth == #shape then
				i = i + 1
				t[k] = v[i]
			else
				t[k] = build(depth + 1)
			end
		end
		return t
	end
	return build(1)
end

--- The raw data as a string of bytes.
-- This runs all pending work on the device first.
-- @function tensor:bytes
-- @treturn string
function T.bytes(x)
	flush(x.d)
	return x.u:bytes()
end

--- The underlying lgml tensor, computed.
-- This runs all pending work on the device first.
-- @function tensor:raw
-- @return lgml tensor
function T.raw(x)
	flush(x.d)
	return x.u
end

local EDGE = 3

local function fmt(v, int)
	if int then
		return string.format("%d", v)
	end
	return string.format("%.4f", v)
end

--- `tostring(x)`: the values, dtype and device.
-- This runs all pending work on the device first. Tensors with more than
-- 1000 elements show 3 values at each end of a dimension. If the values
-- are not available, the text shows the shape only.
-- @function tensor.__tostring
-- @treturn string
function mt.__tostring(x)
	local dtype, shape = x.u:type(), x.u:shape()
	local tail = (", dtype=%s, device=%s)"):format(dtype, x.d.name)
	local ok, v = pcall(T.get, x)
	if not ok then
		return "tensor(<" .. table.concat(shape, "x") .. ">" .. tail
	end
	local int = dtype == "i32" or dtype == "i64"
	local big = #v > 1000
	local strides = {}
	local acc = 1
	for i = #shape, 1, -1 do
		strides[i] = acc
		acc = acc * shape[i]
	end
	local function build(depth, base, indent)
		local parts, n = {}, shape[depth]
		for k = 1, n do
			if big and n > 2 * EDGE and k == EDGE + 1 then
				parts[#parts + 1] = "..."
			end
			if not (big and n > 2 * EDGE and k > EDGE and k <= n - EDGE) then
				local off = base + (k - 1) * strides[depth]
				if depth == #shape then
					parts[#parts + 1] = fmt(v[off + 1], int)
				else
					parts[#parts + 1] = build(depth + 1, off, indent .. " ")
				end
			end
		end
		local sep = depth == #shape and ", " or (",\n" .. indent .. " ")
		return "[" .. table.concat(parts, sep) .. "]"
	end
	return "tensor(" .. build(1, 0, "       ") .. tail
end

--- `x[k]`: a number selects along the first dimension; other keys are methods.
-- @function tensor.__index
-- @tparam int|string k index from 1, or a method name
-- @treturn tensor|function
function mt.__index(x, k)
	if type(k) == "number" then
		return x:select(1, k)
	end
	return T[k]
end

--- State dicts.
-- A state dict maps tensor names to tensors, as in torch. Index it by
-- name: `sd["model.norm.weight"]` gives a tensor or nil. Each tensor
-- wraps once and then comes from a cache. The tensors are not pending.
-- @section state_dicts

--- Load the tensors of a gguf or safetensors file by name.
-- A path that ends in ".gguf" loads as gguf. Any other path loads as
-- safetensors: one file, or a directory of safetensors files.
-- `sd:keys()` lists the names.
-- @function load
-- @tparam string path file or directory
-- @tparam[opt] table opts only `opts.device` applies
-- @treturn table state dict
-- @raise if the file does not load
function M.load(path, o)
	local d = opts_of(o)
	local w
	if path:match("%.gguf$") then
		w = assert(d.be:load(path))
	else
		w = assert(d.be:load_safetensors(path))
	end
	local cache = {}
	local sd = {}
	function sd.keys()
		return w:names()
	end
	return setmetatable(sd, {
		__index = function(_, name)
			if cache[name] == nil then
				local u = w:get(name)
				cache[name] = u and wrap(d, u) or false
			end
			return cache[name] or nil
		end,
	})
end

--- Functions in torch.nn.functional style.
-- Each takes the tensor as the first argument, for example
-- `nd.F.softmax(x)`.
-- @table F
-- @field softmax `tensor:softmax`
-- @field silu `tensor:silu`
-- @field gelu `tensor:gelu`
-- @field relu `tensor:relu`
-- @field rms_norm `tensor:rms_norm`
-- @field layer_norm `tensor:layer_norm`
-- @field linear `tensor:linear`
-- @within Submodules
M.F = {
	softmax = T.softmax, silu = T.silu, gelu = T.gelu, relu = T.relu,
	rms_norm = T.rms_norm, layer_norm = T.layer_norm, linear = T.linear,
}

--- The `lgml.nd.nn` module table, bound to this module.
-- @field nn
-- @within Submodules
M.nn = require("lgml.nd.nn")(M)

return M
