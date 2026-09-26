--- Graph replay for model passes.
-- A pass builds a graph once. The next pass with the same key runs
-- the same graph again: the model refills the input leaves, and the
-- graph keeps its nodes and its planned storage. Decode steps share a
-- key while the attention span stays inside one padding block.
-- The cache holds the last graph; its output nodes stay valid until
-- the next pass on that cache.
-- @module lgml.replay
local M = {}

--- Drop the graph that a cache holds.
-- @tparam table cache the model cache
function M.drop(cache)
	local r = cache and cache.replay
	if not r then
		return
	end
	cache.replay = nil
	r.g:close()
	for _, t in pairs(r.inp) do
		if type(t) == "userdata" then
			t:close()
		end
	end
end

--- Run one pass of model on ids at cache.n_past.
-- The model gives `inputs` and `forward`. When it also gives
-- `replay_key(cache, n, opts)` and `refill(inp, ids, cache, opts)`, a
-- pass with the key of the last one reuses its graph.
-- @param model the model
-- @tparam[opt] table cache the model cache; without one the pass runs
-- once and its graph lives as long as the returned nodes
-- @tparam {integer,...} ids the token ids
-- @tparam table opts the input options
-- @tparam[opt] function finish `finish(g, out)` adds nodes after the
-- forward output and returns the nodes to compute; default the output
-- @return the computed nodes
function M.run(model, cache, ids, opts, finish)
	if not cache then
		local inp = model:inputs(ids, nil, opts)
		local g = model.be:graph()
		local outs = { (finish or function(_, x) return x end)(g, model:forward(g, inp)) }
		g:compute(table.unpack(outs))
		return table.unpack(outs)
	end
	local key = model.replay_key and model:replay_key(cache, #ids, opts)
	local r = cache.replay
	if key and r and r.key == key then
		model:refill(r.inp, ids, cache, opts)
	else
		M.drop(cache)
		local inp = model:inputs(ids, cache, opts)
		local g = model.be:graph()
		local out = model:forward(g, inp)
		r = { key = key, g = g, inp = inp, outs = { (finish or function(_, x) return x end)(g, out) } }
		cache.replay = r
	end
	r.g:compute(table.unpack(r.outs))
	cache.n_past = r.inp.n_kv
	return table.unpack(r.outs)
end

return M
