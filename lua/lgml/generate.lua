--- Greedy generation on a model with a KV cache.
-- See `lgml.transformer` for a model and cache.
-- @module lgml.generate
local lgml = require("lgml")

local M = {}

--- Greedy decode.
-- Feed the prompt ids, then generate up to n_predict ids.
-- @param model a model with a `step(cache, ids, opts)` method
-- @param cache the model's KV cache
-- @tparam {integer,...} prompt the prompt token ids
-- @tparam integer n_predict the maximum count of new ids
-- @tparam[opt] table opts options
-- opts.flash_attn: pass flash attention to model:step
-- opts.stop: set of ids that end generation, as `{ [id] = true }`
-- @treturn {integer,...} the new ids, with the stop id when one ends it
-- @treturn table timings `{ prefill = s, decode = s, n_decode = n }`
function M.greedy(model, cache, prompt, n_predict, opts)
	opts = opts or {}
	local stop = opts.stop or {}
	local so = { flash_attn = opts.flash_attn }
	local out = {}
	local t0 = lgml.clock()
	local id = model:step(cache, prompt, so)
	local t1 = lgml.clock()
	out[1] = id
	while #out < n_predict and not stop[id] do
		id = model:step(cache, { id }, so)
		out[#out + 1] = id
	end
	local t2 = lgml.clock()
	return out, { prefill = t1 - t0, decode = t2 - t1, n_decode = #out - 1 }
end

return M
