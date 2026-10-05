--- Graph replay for model passes.
-- A pass builds a graph once. The next pass with the same key runs
-- the same graph again: the model refills the input leaves, and the
-- graph keeps its nodes and its planned storage. Decode steps share a
-- key while the attention span stays inside one padding block.
-- The cache holds the last graph; its output nodes stay valid until
-- the next pass on that cache. With `cache.replay_keep` set, the cache
-- also keeps that many earlier graphs of small passes, so passes that
-- take turns, such as decode steps and speculative verify passes, do
-- not build their graphs again.
-- A model with weights on a device and its CPU (see `lgml.placement`)
-- runs through the ggml scheduler; a replay reuses its splits and
-- buffers as well.
-- A model with a `before_pass` method gets a call before each compute,
-- such as an expert cache that moves experts between passes.
-- At trace level each pass logs its key and whether it built a graph or
-- replayed one, and a new graph logs its scheduler splits (see
-- `lgml.log`).
-- @module lgml.replay
local log = require("lgml.log")

local L = log.new("replay")

local M = {}

-- The most tokens of a pass whose graph a cache keeps for later.
local SMALL = 64

-- Free one graph. With done true, its output nodes close too.
local function free(r, done)
	r.g:close()
	for _, t in pairs(r.inp) do
		if type(t) == "userdata" then
			t:close()
		end
	end
	if done then
		for _, t in ipairs(r.outs) do
			t:close()
		end
	end
end

--- Drop the graphs that a cache holds.
-- Their compute buffers go when their output nodes close. With done
-- true, they close now, as the caller reads them no more.
-- @tparam table cache the model cache
-- @tparam[opt] boolean done true to close the output nodes
function M.drop(cache, done)
	if not cache then
		return
	end
	local r = cache.replay
	cache.replay = nil
	if r then
		free(r, done)
	end
	M.drop_kept(cache)
end

--- Drop the earlier graphs that a cache keeps, but not the last one.
-- Their outputs are from earlier passes, so they close too.
-- @tparam table cache the model cache
function M.drop_kept(cache)
	local kept = cache.replay_kept
	cache.replay_kept = nil
	for _, r in ipairs(kept or {}) do
		free(r, true)
	end
end

--- The compute bytes of the graphs that a cache keeps, besides the last.
-- @tparam table cache the model cache
-- @treturn integer the bytes, device and host together
function M.kept_bytes(cache)
	local n = 0
	for _, r in ipairs(cache.replay_kept or {}) do
		n = n + r.bytes
	end
	return n
end

-- Put graph r aside when the cache keeps graphs and r is small, else
-- free it. Kept graphs stay within cache.replay_keep graphs and
-- cache.replay_keep_bytes bytes; the oldest goes first. The outputs of
-- r are from the pass before, so a freed r closes them: its compute
-- memory goes now, not at the next garbage collection.
local function park(cache, r)
	if not r then
		return
	end
	local keep, cap = cache.replay_keep or 0, cache.replay_keep_bytes or math.huge
	if keep < 1 or r.inp.n > SMALL or r.bytes > cap then
		free(r, true)
		return
	end
	local kept = cache.replay_kept or {}
	cache.replay_kept = kept
	table.insert(kept, 1, r)
	local total = 0
	for _, x in ipairs(kept) do
		total = total + x.bytes
	end
	while #kept > keep or total > cap do
		local x = table.remove(kept)
		total = total - x.bytes
		free(x, true)
	end
end

-- Take the kept graph with key out of the list, or nil.
local function take(cache, key)
	for i, r in ipairs(cache.replay_kept or {}) do
		if r.key == key then
			return table.remove(cache.replay_kept, i)
		end
	end
end

--- Run one pass of model on ids at cache.n_past.
-- The model gives `inputs` and `forward`. When it also gives
-- `replay_key(cache, n, opts)` and `refill(inp, ids, cache, opts)`, a
-- pass with the key of the last one, or of a kept one, reuses its
-- graph. `opts.tag` adds to the key; passes with another finish need
-- their own tag.
-- @param model the model
-- @tparam[opt] table cache the model cache; without one the pass runs
-- once and its graph lives as long as the returned nodes
-- @tparam {integer,...} ids the token ids
-- @tparam table opts the input options
-- @tparam[opt] function finish `finish(g, out, taps)` adds nodes after
-- the forward output and returns the nodes to compute; default the
-- output
-- @return the computed nodes
function M.run(model, cache, ids, opts, finish)
	finish = finish or function(_, x) return x end
	if not cache then
		local inp = model:inputs(ids, nil, opts)
		local g = model.be:graph()
		local outs = { finish(g, model:forward(g, inp)) }
		if model.before_pass then
			model:before_pass()
		end
		g:compute(table.unpack(outs))
		return table.unpack(outs)
	end
	local key = model.replay_key and model:replay_key(cache, #ids, opts)
	if key and opts.tag then
		key = key .. "|" .. opts.tag
	end
	local r = cache.replay
	local hit = key and r and r.key == key
	if key and not hit and cache.replay_kept then
		local k = take(cache, key)
		if k then
			park(cache, r)
			r, hit = k, true
			cache.replay = r
		end
	end
	if log.level >= log.TRACE then
		L:trace(hit and "graph replay" or "graph build", "n", #ids, "n_past", cache.n_past,
		    "key", key or "none")
	end
	cache.replay_hit = hit or false
	if hit then
		model:refill(r.inp, ids, cache, opts)
	else
		cache.replay = nil
		park(cache, r)
		local inp = model:inputs(ids, cache, opts)
		local g = model.be:graph()
		r = { key = key, g = g, inp = inp, outs = { finish(g, model:forward(g, inp)) }, bytes = 0 }
		cache.replay = r
	end
	if model.before_pass then
		model:before_pass()
	end
	r.g:compute(table.unpack(r.outs))
	if not hit then
		local dev, host = r.g:buffer_size()
		r.bytes = dev + (host or 0)
		if log.level >= log.TRACE then
			L:trace("graph planned", "splits", r.g:splits(), "buffer", dev, "buffer_cpu", host)
		end
	end
	cache.n_past = r.inp.n_kv
	return table.unpack(r.outs)
end

return M
