-- Replay checks for the model tests: greedy decodes through lgml.replay
-- against the same decodes with a new graph for every pass.
local replay = require("lgml.replay")

local M = {}

-- Run ids at cache.n_past in passes of at most ub tokens. Returns the
-- last logits (as a list) and their argmax.
local function pass(m, cache, ids, ub, opts)
	local o = { logits = "last", flash_attn = opts.flash_attn }
	local logits, best
	for s = 1, #ids, ub do
		local part = table.move(ids, s, math.min(#ids, s + ub - 1), 1, {})
		local l, b = replay.run(m, cache, part, o, function(g, x)
			return x, g:argmax(x)
		end)
		logits, best = l:get(), b:get()[1]
	end
	return logits, best
end

-- Prompt, then n greedy steps. Returns the logits of each pass, the
-- ids and the count of graphs built.
local function decode(m, cache, prompt, n, opts)
	local fwd, built = m.forward, 0
	m.forward = function(...)
		built = built + 1
		return fwd(...)
	end
	local rows, ids = {}, {}
	local ub = opts.ubatch or #prompt
	local l, id = pass(m, cache, prompt, ub, opts)
	rows[1] = l
	for i = 1, n do
		ids[i] = id
		l, id = pass(m, cache, { id }, ub, opts)
		rows[i + 1] = l
	end
	m.forward = nil
	return rows, ids, built
end

local function same(a, b)
	if #a ~= #b then
		return false
	end
	for i = 1, #a do
		if #a[i] ~= #b[i] then
			return false
		end
		for j = 1, #a[i] do
			if a[i][j] ~= b[i][j] then
				return false
			end
		end
	end
	return true
end

-- A greedy decode with replay and one without give the same logits.
-- opts: n_ctx, cache (cache options), prompt, n, flash_attn, ubatch,
-- builds (the most graphs the replay decode may build).
function M.decode(ok, tag, m, opts)
	local c1 = m:cache(opts.n_ctx, opts.cache)
	local a, ids, built = decode(m, c1, opts.prompt, opts.n, opts)
	m.replay_key = false
	local c2 = m:cache(opts.n_ctx, opts.cache)
	local b = decode(m, c2, opts.prompt, opts.n, opts)
	m.replay_key = nil
	replay.drop(c1)
	replay.drop(c2)
	ok(same(a, b) and #a == opts.n + 1, tag .. "replay gives the same logits")
	ok(built <= opts.builds, ("%sreplay builds %d graphs for %d decode steps"):format(tag, built,
	    opts.n))
	return ids, c1
end

-- Restart a recurrent model at n_past 0 on a replayed graph: the state
-- clears and the logits repeat. A step back is an error.
function M.restart(ok, tag, m)
	local c = m:cache(64)
	local first = pass(m, c, { 7 }, 1, {})
	pass(m, c, { 8 }, 1, {})
	pass(m, c, { 9 }, 1, {})
	c.n_past = 0
	local again = pass(m, c, { 7 }, 1, {})
	ok(same({ first }, { again }), tag .. "a replayed pass at n_past 0 clears the state")
	pass(m, c, { 8 }, 1, {})
	c.n_past = 1
	ok(not pcall(pass, m, c, { 9 }, 1, {}), tag .. "a replayed pass cannot rewind the state")
	replay.drop(c)
end

return M
