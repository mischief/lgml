--- Small torch-style modules on nd tensors.
-- Build a module from a state dict with `Module.from(sd, prefix)`, or
-- from tensors with `Module.new(...)`. Call a module like a function:
-- `m(x)` is `m:forward(x)`. Weights are public fields of the module.
--
-- Use it as `nd.nn`. The file returns a function that takes the
-- `lgml.nd` module table and returns this module table.
--
--     local nd = require("lgml.nd")
--     local sd = nd.load("model.safetensors")
--     local proj = nd.nn.Linear.from(sd, "model.proj")
--     print(proj(nd.randn{ 2, proj.weight:size(2) }))
-- @module lgml.nd.nn
return function(nd)
	local nn = {}

	local function class()
		local c = {}
		c.__index = c
		c.__call = function(self, ...)
			return self:forward(...)
		end
		return c
	end

	local function need(sd, name)
		local t = sd[name]
		if not t then
			error("state dict has no " .. name, 3)
		end
		return t
	end

	--- Linear layer: weight (out, in), optional bias (out).
	-- @type Linear
	nn.Linear = class()

	--- Make a Linear from tensors.
	-- @function Linear.new
	-- @tparam lgml.nd.tensor weight (out, in)
	-- @tparam[opt] lgml.nd.tensor bias (out)
	-- @treturn Linear
	function nn.Linear.new(weight, bias)
		return setmetatable({ weight = weight, bias = bias }, nn.Linear)
	end

	--- Make a Linear from `prefix.weight` and `prefix.bias` in a state dict.
	-- @function Linear.from
	-- @tparam table sd state dict
	-- @tparam string prefix name prefix, for example "model.proj"
	-- @treturn Linear
	-- @raise if sd has no `prefix.weight`
	function nn.Linear.from(sd, prefix)
		return nn.Linear.new(need(sd, prefix .. ".weight"), sd[prefix .. ".bias"])
	end

	--- Compute x @ weight^T + bias.
	-- @function Linear:forward
	-- @tparam lgml.nd.tensor x (..., in)
	-- @treturn lgml.nd.tensor (..., out)
	function nn.Linear:forward(x)
		return x:linear(self.weight, self.bias)
	end

	--- Embedding table: weight (n, dim); input is an i32 tensor of ids.
	-- @type Embedding
	nn.Embedding = class()

	--- Make an Embedding from a weight tensor.
	-- @function Embedding.new
	-- @tparam lgml.nd.tensor weight (n, dim)
	-- @treturn Embedding
	function nn.Embedding.new(weight)
		return setmetatable({ weight = weight }, nn.Embedding)
	end

	--- Make an Embedding from `prefix.weight` in a state dict.
	-- @function Embedding.from
	-- @tparam table sd state dict
	-- @tparam string prefix name prefix
	-- @treturn Embedding
	-- @raise if sd has no `prefix.weight`
	function nn.Embedding.from(sd, prefix)
		return nn.Embedding.new(need(sd, prefix .. ".weight"))
	end

	--- Look up the rows of weight for each id.
	-- Ids start at 0. The result is on the device of weight.
	-- @function Embedding:forward
	-- @tparam lgml.nd.tensor ids i32 tensor of ids, shape (len)
	-- @treturn lgml.nd.tensor (len, dim)
	function nn.Embedding:forward(ids)
		local w = self.weight
		return nd.wrap_op(w, function(g)
			return g:get_rows(w.u, ids.u)
		end)
	end

	--- RMS normalization over the last dimension, times an optional weight.
	-- @type RMSNorm
	nn.RMSNorm = class()

	--- Make an RMSNorm.
	-- @function RMSNorm.new
	-- @tparam[opt] lgml.nd.tensor weight (dim), no scale if nil
	-- @tparam[opt=1e-6] number eps
	-- @treturn RMSNorm
	function nn.RMSNorm.new(weight, eps)
		return setmetatable({ weight = weight, eps = eps or 1e-6 }, nn.RMSNorm)
	end

	--- Make an RMSNorm from `prefix.weight` in a state dict.
	-- @function RMSNorm.from
	-- @tparam table sd state dict
	-- @tparam string prefix name prefix
	-- @tparam[opt=1e-6] number eps
	-- @treturn RMSNorm
	-- @raise if sd has no `prefix.weight`
	function nn.RMSNorm.from(sd, prefix, eps)
		return nn.RMSNorm.new(need(sd, prefix .. ".weight"), eps)
	end

	--- Normalize x and multiply by weight.
	-- @function RMSNorm:forward
	-- @tparam lgml.nd.tensor x (..., dim)
	-- @treturn lgml.nd.tensor same shape as x
	function nn.RMSNorm:forward(x)
		local y = x:rms_norm(self.eps)
		return self.weight and y * self.weight or y
	end

	--- Layer normalization over the last dimension, with optional weight and bias.
	-- @type LayerNorm
	nn.LayerNorm = class()

	--- Make a LayerNorm.
	-- @function LayerNorm.new
	-- @tparam[opt] lgml.nd.tensor weight (dim), no scale if nil
	-- @tparam[opt] lgml.nd.tensor bias (dim), no shift if nil
	-- @tparam[opt=1e-5] number eps
	-- @treturn LayerNorm
	function nn.LayerNorm.new(weight, bias, eps)
		return setmetatable({ weight = weight, bias = bias, eps = eps or 1e-5 }, nn.LayerNorm)
	end

	--- Make a LayerNorm from `prefix.weight` and `prefix.bias` in a state dict.
	-- Both tensors are optional.
	-- @function LayerNorm.from
	-- @tparam table sd state dict
	-- @tparam string prefix name prefix
	-- @tparam[opt=1e-5] number eps
	-- @treturn LayerNorm
	function nn.LayerNorm.from(sd, prefix, eps)
		return nn.LayerNorm.new(sd[prefix .. ".weight"], sd[prefix .. ".bias"], eps)
	end

	--- Normalize x, multiply by weight and add bias.
	-- @function LayerNorm:forward
	-- @tparam lgml.nd.tensor x (..., dim)
	-- @treturn lgml.nd.tensor same shape as x
	function nn.LayerNorm:forward(x)
		local y = x:layer_norm(self.eps)
		if self.weight then
			y = y * self.weight
		end
		if self.bias then
			y = y + self.bias
		end
		return y
	end

	--- A chain of modules that run in order.
	-- The modules are the array part of the object.
	-- @type Sequential
	nn.Sequential = class()

	--- Make a Sequential from modules.
	-- @function Sequential.new
	-- @param ... modules or other callables, in order
	-- @treturn Sequential
	function nn.Sequential.new(...)
		return setmetatable({ ... }, nn.Sequential)
	end

	--- Pass x through each module in order.
	-- @function Sequential:forward
	-- @tparam lgml.nd.tensor x input of the first module
	-- @return output of the last module
	function nn.Sequential:forward(x)
		for _, m in ipairs(self) do
			x = m(x)
		end
		return x
	end

	return nn
end
