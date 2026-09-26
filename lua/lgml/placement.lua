--- Weight placement: which tensors of a model load to the CPU.
-- A model on a GPU backend can keep part of its weights in host memory,
-- as llama.cpp does with `--n-cpu-moe`, `-ot` and `-ngl`. The ops that
-- use those weights run on the CPU (see `lgml.backend` with `cpu`).
--
-- Rules, first match wins:
--
-- 1. `override`: tensor name patterns and where they go, in order. A
--    pattern is a Lua pattern that matches anywhere in the name, such as
--    `exps` or `blk%.1[0-9]%.`. The place is `"CPU"`, or any other name
--    for the model's own backend.
-- 2. `cpu_moe` and `n_cpu_moe`: the routed expert weights
--    (`blk.N.ffn_*exps.weight`) of every layer, or of the first N layers.
-- 3. `n_gpu_layers`: the last N layers stay on the backend and the
--    others go to the CPU, with the token embeddings. The output layer
--    stays only when N is more than the layer count.
--
-- The KV cache and recurrent state of a layer go where its
-- `attn_norm` weight is.
-- @module lgml.placement
local M = {}

--- The pattern of the routed expert weights; the capture is the layer.
M.EXPERTS = "^blk%.(%d+)%.ffn_[%a_]+exps%.weight$"

--- The context that `auto` counts KV cache for when none is given.
M.AUTO_CTX = 8192

-- Parse "pattern=place" into { pattern, cpu }. Raise on a bad entry.
local function entry(s, where)
	local pat, place = tostring(s):match("^(.+)=([^=]+)$")
	if not pat then
		error(("%s: %q is not pattern=backend"):format(where, tostring(s)), 0)
	end
	if not pcall(string.find, "", pat) then
		error(("%s: bad pattern %q"):format(where, pat), 0)
	end
	return { pattern = pat, cpu = place:upper() == "CPU" }
end

-- A layer count option: an integer >= 0. Raise when it is not.
local function count(v, key)
	local n = math.tointeger(tonumber(v))
	if not n or n < 0 then
		error(("placement: %s must be a count, not %s"):format(key, tostring(v)), 0)
	end
	return n
end

--- Read the placement options into rules.
-- @tparam table opts the options:
--
-- - `override`: a list of `"pattern=backend"` strings, a string of them
--   joined by commas, or a map `{ [pattern] = backend }` (a map has no
--   order; its patterns apply in sorted order)
-- - `cpu_moe`: true for the experts of every layer on the CPU
-- - `n_cpu_moe`: the experts of the first N layers on the CPU, or
--   `"auto"` (see `auto`)
-- - `n_gpu_layers`: the count of layers that stay on the backend
-- @treturn table|nil the rules `{ override, n_cpu_moe, n_gpu_layers,
-- auto }`, or nil when nothing moves; n_cpu_moe is math.huge for all
-- layers
-- @raise on a bad pattern or count
function M.rules(opts)
	opts = opts or {}
	local list = {}
	local ov = opts.override
	if type(ov) == "string" then
		for s in ov:gmatch("[^,]+") do
			list[#list + 1] = entry(s, "override")
		end
	elseif type(ov) == "table" then
		if #ov > 0 then
			for _, s in ipairs(ov) do
				for part in tostring(s):gmatch("[^,]+") do
					list[#list + 1] = entry(part, "override")
				end
			end
		else
			local keys = {}
			for k in pairs(ov) do
				keys[#keys + 1] = tostring(k)
			end
			table.sort(keys)
			for _, k in ipairs(keys) do
				list[#list + 1] = entry(k .. "=" .. tostring(ov[k]), "override")
			end
		end
	elseif ov ~= nil then
		error("placement: override must be a string or a table", 0)
	end
	local r = { override = list }
	if opts.n_cpu_moe == "auto" then
		r.auto = true
	elseif opts.n_cpu_moe ~= nil then
		r.n_cpu_moe = count(opts.n_cpu_moe, "n_cpu_moe")
	end
	if opts.cpu_moe then
		r.n_cpu_moe = math.huge
		r.auto = nil
	end
	if opts.n_gpu_layers ~= nil then
		r.n_gpu_layers = count(opts.n_gpu_layers, "n_gpu_layers")
	end
	if #list == 0 and (r.n_cpu_moe or 0) == 0 and not r.auto and not r.n_gpu_layers then
		return nil
	end
	return r
end

-- The layer of a tensor name, from 0, or nil.
local function layer_of(name)
	local il = name:match("^blk%.(%d+)%.")
	return il and tonumber(il)
end

--- Tell if one tensor goes to the CPU.
-- @tparam table rules from `rules`
-- @tparam string name the tensor name
-- @tparam integer n_layer the layer count
-- @tparam[opt] boolean tied true when the output projection is the token
-- embedding
-- @treturn boolean true for the CPU
function M.on_cpu(rules, name, n_layer, tied)
	for _, o in ipairs(rules.override) do
		if name:find(o.pattern) then
			return o.cpu
		end
	end
	local il = name:match(M.EXPERTS)
	if il and tonumber(il) < (rules.n_cpu_moe or 0) then
		return true
	end
	local ngl = rules.n_gpu_layers
	if not ngl then
		return false
	end
	local first = n_layer - ngl
	il = layer_of(name)
	if il then
		return il < first
	end
	local out_cpu = ngl <= n_layer
	if name == "output.weight" or name == "output_norm.weight" then
		return out_cpu
	elseif name == "token_embd.weight" then
		return first > 0 and (not tied or out_cpu)
	elseif name == "per_layer_token_embd.weight" then
		return first > 0
	end
	return false
end

--- Place the tensors of a model.
-- @tparam table rules from `rules`; with `auto` and no n_cpu_moe, no
-- experts move
-- @tparam {table,...} tensors `{ name, size }` entries, as in
-- `lgml.gguf(path).tensors`
-- @tparam integer n_layer the layer count
-- @treturn table|nil the set of names on the CPU, `{ [name] = true }`, or
-- nil when none
-- @treturn integer the bytes on the CPU
-- @treturn table the set of layers, from 0, whose attn_norm is on the CPU
function M.plan(rules, tensors, n_layer)
	local tied = true
	for _, t in ipairs(tensors) do
		if t.name == "output.weight" then
			tied = false
		end
	end
	local set, bytes, layers, any = {}, 0, {}, false
	for _, t in ipairs(tensors) do
		if M.on_cpu(rules, t.name, n_layer, tied) then
			set[t.name] = true
			bytes = bytes + (t.size or 0)
			any = true
			local il = t.name:match("^blk%.(%d+)%.attn_norm%.weight$")
			if il then
				layers[tonumber(il)] = true
			end
		end
	end
	return any and set or nil, bytes, layers
end

--- Pick n_cpu_moe so the weights on the backend fit a budget.
-- It is the smallest count of layers, from the first, whose experts go
-- to the CPU so that the rest of the weights fit.
-- @tparam table rules from `rules`, without n_cpu_moe
-- @tparam {table,...} tensors as for `plan`
-- @tparam integer n_layer the layer count
-- @tparam number budget the bytes the weights may take on the backend
-- @treturn integer the count; n_layer when even all experts on the CPU
-- do not fit
-- @treturn integer the weight bytes on the backend with that count
function M.auto(rules, tensors, n_layer, budget)
	local r = { override = rules.override, n_gpu_layers = rules.n_gpu_layers }
	local total = 0
	for _, t in ipairs(tensors) do
		total = total + (t.size or 0)
	end
	local _, cpu = M.plan(r, tensors, n_layer)
	local dev = total - cpu
	local n = 0
	while dev > budget and n < n_layer do
		n = n + 1
		r.n_cpu_moe = n
		local _, c = M.plan(r, tensors, n_layer)
		dev = total - c
	end
	return n, dev
end

return M
