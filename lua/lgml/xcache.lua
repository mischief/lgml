--- Adaptive expert cache: the hot routed experts of a MoE model in
-- device slots.
-- All routed experts stay in host memory (see `lgml.placement`). Each
-- layer gets device slots for some of them. The router picks go to two
-- parts: the device computes the picks that a slot holds, and the CPU
-- computes the rest. The two results add up with the router weights.
--
-- A score counts the picks of each expert; old picks fall off with a
-- half-life in tokens. Before each pass, the cache moves the experts of
-- most score into slots, at most `swap` bytes per pass. A profile
-- (scores from `profile`, for example of a prompt set) gives the first
-- scores.
-- @module lgml.xcache
local lgml = require("lgml")
local log = require("lgml.log")

local L = log.new("xcache")

local M = {}
M.__index = M

-- The expert weight fields of a layer, as `lgml.transformer` names them.
local FIELDS = { "gate_up_exps", "gate_exps", "up_exps", "down_exps" }

--- Default bytes moved into slots per pass: 16 MiB.
M.SWAP = 16 << 20

--- Default half-life of a pick, in tokens: 512.
M.HALFLIFE = 512

--- Default margin: a new expert needs a score above (1 + margin) times
-- the score of the one it replaces, plus 1.
M.MARGIN = 0.25

--- The layers whose routed experts are on the CPU of the model backend.
-- @param model a model from `lgml.transformer.load`
-- @treturn table `{ [il] = bytes }`: the device bytes of one expert of
-- each such layer, layers from 1
-- @treturn integer the sum of those bytes
function M.layers(model)
	local cpu = model.be:cpu()
	local out, sum = {}, 0
	for il, L in ipairs(model.layers) do
		if cpu and L.gate_inp and L.down_exps and L.down_exps:backend() == cpu then
			local b = 0
			for _, f in ipairs(FIELDS) do
				local t = L[f]
				if t then
					b = b + t:nbytes() // t:shape()[1]
				end
			end
			out[il] = b
			sum = sum + b
		end
	end
	return out, sum
end

--- The slots per layer that fit in a byte budget.
-- Every layer gets the same count; the tables of a layer count too.
-- @param model the model
-- @tparam integer bytes the device bytes for the cache
-- @treturn integer slots per layer, 0 to the expert count
function M.slots_for(model, bytes)
	local layers, sum = M.layers(model)
	local n = 0
	for _ in pairs(layers) do
		n = n + 1
	end
	if sum == 0 then
		return 0
	end
	bytes = bytes - n * model.hp.n_expert * 8
	local s = bytes // sum
	if s >= model.hp.n_expert then
		return model.hp.n_expert
	end
	-- The zero slots of the pick positions take room too.
	return math.max(0, s - model.hp.n_used)
end

-- Read a profile file: a header line, then the scores of one layer per
-- line, "il s0 s1 ...". Return { [il] = scores }.
local function read_profile(path, ne)
	local f, err = io.open(path, "r")
	if not f then
		return nil, err
	end
	local head = f:read("l")
	if not head or not head:match("^lgml%-expert%-profile 1 ") then
		f:close()
		return nil, path .. ": not an expert profile"
	end
	local out = {}
	for line in f:lines() do
		local il, rest = line:match("^(%d+)%s+(.*)$")
		if il then
			local s = {}
			for v in rest:gmatch("%S+") do
				s[#s + 1] = tonumber(v) or 0
			end
			if #s == ne then
				out[tonumber(il)] = s
			end
		end
	end
	f:close()
	return out
end

--- Make an expert cache and attach it to a model.
-- Attach it before the first pass: the graphs of later passes take the
-- cache path. Layers with experts on the device stay as they are.
-- @param model a model from `lgml.transformer.load`, with its routed
-- experts on the CPU of its backend
-- @tparam table opts the options:
--
-- - `slots`: slots per layer, 0 to the expert count; or `bytes`: the
--   device bytes for all slots (see `slots_for`)
-- - `swap`: the most bytes moved into slots per pass (default `SWAP`);
--   0 keeps the first experts
-- - `halflife`: the half-life of a pick in tokens (default `HALFLIFE`)
-- - `margin`: see `MARGIN`
-- - `profile`: a profile file path, or `{ [il] = scores }`, for the
--   first scores
-- @treturn XCache the cache
-- @raise when no layer has experts on the CPU, or the device memory
-- does not fit
function M.new(model, opts)
	local layers = M.layers(model)
	if not next(layers) then
		error("expert cache: no layer has its routed experts on the CPU", 0)
	end
	local hp, be = model.hp, model.be
	local n = opts.slots or M.slots_for(model, opts.bytes or 0)
	n = math.max(0, math.min(hp.n_expert, math.tointeger(n) or 0))
	local self = setmetatable({
		model = model, n = n, ne = hp.n_expert,
		state = lgml.expert_cache(#model.layers, hp.n_expert),
		swap = opts.swap or M.SWAP,
		decay = 0.5 ^ (1 / (opts.halflife or M.HALFLIFE)),
		margin = opts.margin or M.MARGIN,
		layers = {}, bytes = 0, expert_bytes = 0, moved = 0,
	}, M)
	local prof = opts.profile
	if type(prof) == "string" then
		prof = assert(read_profile(prof, hp.n_expert))
	end
	local ok, err = pcall(function()
		for il, eb in pairs(layers) do
			local L = model.layers[il]
			local xc = { il = il, n = n, full = n == hp.n_expert, state = self.state }
			-- A pick that no slot holds goes to the zero slot n + k of
			-- its position k. The device ids of a token then never
			-- repeat: Vulkan mul_mat_id drops rows of a repeated id.
			xc.rows = xc.full and n or n + hp.n_used
			self.state:resize(il, n)
			if prof and prof[il] then
				self.state:seed(il, prof[il])
			end
			if n > 0 then
				for _, f in ipairs(FIELDS) do
					local t = L[f]
					if t then
						local s = t:shape()
						s[1] = xc.rows
						local c = be:tensor(t:type(), s, nil, { weights = true })
						-- Empty slots must hold no NaN: a weight 0 picks them.
						-- The miss slots hold expert 0. With zero bytes
						-- there, IQ2_XS experts give NaN on Vulkan.
						c:zero()
						for r = n, xc.rows - 1 do
							c:set_slice(r, t, 0)
						end
						xc[f] = c
						self.bytes = self.bytes + c:nbytes()
					end
				end
				-- The tables go where the router runs, so that only
				-- the ids and weights of the device picks cross over.
				local tb = L.gate_inp:backend()
				xc.map = tb:tensor("f32", { hp.n_expert, 1 }, nil, { weights = true })
				xc.hit = tb:tensor("f32", { hp.n_expert, 1 }, nil, { weights = true })
				local zero = {}
				for k = 1, hp.n_used do
					zero[k] = n + k - 1
				end
				xc.zero = tb:tensor("f32", { 1, hp.n_used }, zero, { weights = true })
				if tb == be then
					self.bytes = self.bytes + xc.map:nbytes() + xc.hit:nbytes() + xc.zero:nbytes()
				end
				xc.router_cpu = tb ~= be
			end
			self.layers[il] = xc
			self.expert_bytes = math.max(self.expert_bytes, eb)
		end
	end)
	if not ok then
		self:close()
		error(err, 0)
	end
	if n > 0 then
		-- Fill every slot now, the experts of most score first.
		self:apply(self.state:plan(n * #model.layers, self.decay, self.margin))
	end
	if opts.zerocopy and n > 0 and n < hp.n_expert and not self:zerocopy() then
		L:warn("zero-copy experts need the Vulkan host read hook of the lgml ggml patch; " ..
		    "the CPU computes the misses")
	end
	for il, xc in pairs(self.layers) do
		model.layers[il].xc = xc
	end
	model.xcache = self
	model.before_pass = M.before_pass
	return self
end

--- Give the device a pinned host copy of the experts that no slot holds,
-- which it reads over PCIe: the misses then run on the device, and a pass
-- needs no CPU part. The slots stay as they are (no moves). xc.host holds
-- the experts per field, xc.hmap (f32) the host row of each expert; held
-- experts map to a dummy row, read at weight 0. Returns false, and changes
-- nothing, when the Vulkan backend lacks the host read hook.
function M:zerocopy()
	local be = self.model.be
	if be:name():upper():match("^CPU") or not lgml.vk_host_read() then
		return false
	end
	for il, xc in pairs(self.layers) do
		local L = self.model.layers[il]
		local map, hit = self.state:tables(il)
		local rows, hmap = {}, {}
		local held = {}
		for e = 1, self.ne do
			if hit[e] == 0 then
				rows[#rows + 1] = e - 1
				hmap[e] = #rows - 1
			else
				held[#held + 1] = e
			end
		end
		-- A held expert maps to the first dummy row.
		for _, e in ipairs(held) do
			hmap[e] = #rows
		end
		-- Rows #rows .. #rows + K - 1 hold one dummy expert per pick
		-- position, for held picks: a token never names a row twice.
		local K = self.model.hp.n_used
		local host = {}
		for _, f in ipairs(FIELDS) do
			local t = L[f]
			if t then
				local s = t:shape()
				s[1] = #rows + K
				local h = be:tensor(t:type(), s, nil, { weights = true, host_read = true })
				for i, e in ipairs(rows) do
					h:set_slice(i - 1, t, e)
				end
				for k = 0, K - 1 do
					h:set_slice(#rows + k, t, rows[1])
				end
				host[f] = h
			end
		end
		local dummy = {}
		for k = 1, K do
			dummy[k] = #rows + k - 1
		end
		xc.host = host
		local tb = L.gate_inp:backend()
		xc.hmap = tb:tensor("f32", { self.ne, 1 }, hmap, { weights = true })
		xc.hzero = tb:tensor("f32", { 1, K }, dummy, { weights = true })
		xc.static = true
	end
	self.swap = 0
	return true
end

--- An expert cache from `new`.
-- Fields: n (slots per layer), bytes (device bytes of the slots and
-- tables), moved (bytes moved into slots since `new`).
-- @type XCache

-- Copy the planned experts into their slots and set the device tables.
function M:apply(plan)
	local dirty = {}
	for _, c in ipairs(plan) do
		local il, slot, e = c[1], c[2], c[3]
		local L, xc = self.model.layers[il], self.layers[il]
		for _, f in ipairs(FIELDS) do
			if xc[f] then
				xc[f]:set_slice(slot, L[f], e)
				self.moved = self.moved + xc[f]:nbytes() // xc.rows
			end
		end
		self.state:assign(il, slot, e)
		dirty[il] = true
	end
	for il in pairs(dirty) do
		local xc = self.layers[il]
		local map, hit = self.state:tables(il)
		xc.map:set(map)
		xc.hit:set(hit)
	end
	return #plan
end

--- Fold the picks of the last passes into the scores and move experts.
-- `lgml.replay` calls it before each pass, as `model:before_pass()`.
-- @treturn integer the experts moved
function M:step()
	if self.n == 0 or self.n == self.ne then
		self.state:plan(0, self.decay, self.margin)
		return 0
	end
	local k = self.expert_bytes > 0 and self.swap // self.expert_bytes or 0
	return self:apply(self.state:plan(k, self.decay, self.margin))
end

--- The `before_pass` hook of a model with an expert cache.
-- @param model the model
function M.before_pass(model)
	model.xcache:step()
end

--- The picks since the last `reset`.
-- @treturn integer picks of experts in a slot
-- @treturn integer picks of other experts
-- @treturn number the share of picks in a slot, 0 to 1
function M:stats()
	local h, m = self.state:stats()
	return h, m, h + m > 0 and h / (h + m) or 0
end

--- Set the counts of `stats` to 0.
function M:reset()
	self.state:reset_stats()
end

--- The scores of the layers with a cache.
-- @treturn table `{ [il] = { score, ... } }`, one score per expert
function M:profile()
	local out = {}
	for il in pairs(self.layers) do
		out[il] = self.state:scores(il)
	end
	return out
end

--- Write the scores to a profile file for `new`.
-- @tparam string path the file
-- @raise when the file does not write
function M:save(path)
	local f = assert(io.open(path, "w"))
	local prof = self:profile()
	local ils = {}
	for il in pairs(prof) do
		ils[#ils + 1] = il
	end
	table.sort(ils)
	f:write(("lgml-expert-profile 1 %d %d\n"):format(#self.model.layers, self.ne))
	for _, il in ipairs(ils) do
		local parts = {}
		for i, v in ipairs(prof[il]) do
			parts[i] = ("%.6g"):format(v)
		end
		f:write(il, " ", table.concat(parts, " "), "\n")
	end
	assert(f:close())
end

--- Detach the cache from the model and free its device memory.
-- Graphs built with the cache keep its tensors until they close.
function M:close()
	local model = self.model
	for il, xc in pairs(self.layers) do
		if model.layers[il].xc == xc then
			model.layers[il].xc = nil
		end
		for k, v in pairs(xc) do
			if type(v) == "userdata" and k ~= "state" then
				v:close()
			end
		end
	end
	self.layers = {}
	if model.xcache == self then
		model.xcache, model.before_pass = nil, nil
	end
end

return M
