--- Optimizers for trainable leaves, as in torch.optim.
-- An optimizer takes trainable leaves (see `tensor:requires_grad`) on
-- one backend. `step` runs one small graph on that backend that updates
-- every leaf in place from its gradient; the moments stay on the device.
-- Gradients add up over computes, so for gradient accumulation compute
-- several micro-batches (scale each loss by 1 / count) and then step.
--
--     local opt = optim.adamw(params, { lr = 1e-4 })
--     opt:zero_grad()
--     g:compute(loss)
--     opt:step()
-- @module lgml.optim
local M = {}

local Opt = {}
Opt.__index = Opt

local function new(kind, params, opts, defaults)
	assert(#params > 0, "optim: no parameters")
	for i, p in ipairs(params) do
		assert(p:is_param(), "optim: parameter " .. i .. " is not trainable; use requires_grad")
	end
	local o = setmetatable({ kind = kind, params = params, t = 0 }, Opt)
	for k, v in pairs(defaults) do
		o[k] = v
	end
	for k, v in pairs(opts or {}) do
		o[k] = v
	end
	return o
end

--- Make an AdamW optimizer.
-- The update matches torch.optim.AdamW: decoupled weight decay and bias
-- correction of both moments.
-- @tparam {lgml.tensor,...} params trainable f32 leaves on one backend
-- @tparam[opt] table opts options: lr (1e-3), betas ({ 0.9, 0.999 }),
-- eps (1e-8), weight_decay (0.01)
-- @treturn Optimizer the optimizer
-- @raise if a leaf is not trainable
function M.adamw(params, opts)
	return new("adamw", params, opts, { lr = 1e-3, betas = { 0.9, 0.999 }, eps = 1e-8,
	    weight_decay = 0.01 })
end

--- Make an SGD optimizer.
-- The update is p = p - lr * (grad + weight_decay * p).
-- @tparam {lgml.tensor,...} params trainable f32 leaves on one backend
-- @tparam[opt] table opts options: lr (1e-3), weight_decay (0)
-- @treturn Optimizer the optimizer
-- @raise if a leaf is not trainable
function M.sgd(params, opts)
	return new("sgd", params, opts, { lr = 1e-3, weight_decay = 0 })
end

--- An optimizer, from `adamw` or `sgd`.
-- Fields: params, lr, weight_decay, t (the count of steps), and for
-- AdamW betas and eps. Change lr between steps for a schedule.
-- @type Optimizer

--- Set the gradient of each parameter to zero.
function Opt:zero_grad()
	for _, p in ipairs(self.params) do
		p:grad():zero()
	end
end

--- Clip the gradients by their global norm.
-- When the L2 norm of all gradients together passes max_norm, every
-- gradient scales down by the same factor to max_norm.
-- @tparam number max_norm the largest norm
-- @treturn number the norm before clipping
function Opt:clip(max_norm)
	local be = self.params[1]:backend()
	if not self.gnorm then
		local g = be:graph(4 * #self.params + 64)
		local total
		for _, p in ipairs(self.params) do
			local s = g:sum(g:sqr(p:grad()))
			total = total and g:add(total, s) or s
		end
		self.gnorm, self.gnorm_out = g, total
		-- The scale graph multiplies every gradient by the factor in cf.
		local gs = be:graph(4 * #self.params + 64)
		self.cf = be:tensor("f32", { 1 })
		local nodes = {}
		for i, p in ipairs(self.params) do
			nodes[i] = gs:cpy(gs:mul(p:grad(), self.cf), p:grad())
		end
		self.gscale, self.gscale_nodes = gs, nodes
	end
	self.gnorm:compute(self.gnorm_out)
	local norm = math.sqrt(self.gnorm_out:get()[1])
	if norm > max_norm then
		self.cf:set({ max_norm / norm })
		self.gscale:compute(table.unpack(self.gscale_nodes))
	end
	return norm
end

-- The step graph: one update node per parameter, and the moments.
function Opt:build()
	local be = self.params[1]:backend()
	-- Each parameter has a step node, a gradient and two moments.
	local g = be:graph(4 * #self.params + 64)
	local nodes = {}
	if self.kind == "adamw" then
		self.hp = be:tensor("f32", { 7 })
		self.m, self.v = {}, {}
		for i, p in ipairs(self.params) do
			self.m[i] = be:tensor("f32", p:shape())
			self.v[i] = be:tensor("f32", p:shape())
			self.m[i]:zero()
			self.v[i]:zero()
			nodes[i] = g:opt_step_adamw(p, p:grad(), self.m[i], self.v[i], self.hp)
		end
	else
		self.hp = be:tensor("f32", { 2 })
		for i, p in ipairs(self.params) do
			nodes[i] = g:opt_step_sgd(p, p:grad(), self.hp)
		end
	end
	self.g, self.nodes = g, nodes
end

--- Update each parameter from its gradient.
-- The gradients stay as they are; call `zero_grad` before the next
-- accumulation.
function Opt:step()
	if not self.g then
		self:build()
	end
	self.t = self.t + 1
	if self.kind == "adamw" then
		local b1, b2 = self.betas[1], self.betas[2]
		self.hp:set({ self.lr, b1, b2, self.eps, self.weight_decay,
		    1 / (1 - b1 ^ self.t), 1 / (1 - b2 ^ self.t) })
	else
		self.hp:set({ self.lr, self.weight_decay })
	end
	self.g:compute(table.unpack(self.nodes))
end

--- Free the step graph and the moments.
-- The optimizer builds them again at the next step, with zero moments.
function Opt:close()
	for _, k in ipairs({ "g", "gnorm", "gscale" }) do
		if self[k] then
			self[k]:close()
			self[k] = nil
		end
	end
	if self.cf then
		self.cf:close()
		self.cf = nil
	end
	for _, list in ipairs({ self.m or {}, self.v or {} }) do
		for _, t in ipairs(list) do
			t:close()
		end
	end
	self.m, self.v = nil, nil
end

return M
