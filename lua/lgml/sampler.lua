--- Sampler presets on top of `lgml.sampler`.
-- The C defaults are the llama.cpp common defaults.
-- @module lgml.sampler
local lgml = require("lgml")

local M = {}

--- Named sampler option sets.
-- default: the C defaults.
-- greedy: temperature 0.
-- qwen3_think: Qwen3 model card values for thinking mode.
-- qwen3: Qwen3 model card values for non-thinking mode.
-- @table presets
M.presets = {
	default = {},
	greedy = { temperature = 0 },
	-- Qwen3 model card values.
	qwen3_think = { temperature = 0.6, top_k = 20, top_p = 0.95, min_p = 0 },
	qwen3 = { temperature = 0.7, top_k = 20, top_p = 0.8, min_p = 0 },
}

--- The option values of the C sampler when an option is absent.
-- Without a seed, the sampler takes a random one.
-- @table DEFAULTS
M.DEFAULTS = {
	temperature = 0.8, top_k = 40, top_p = 0.95, min_p = 0.05, repeat_penalty = 1.0,
	frequency_penalty = 0.0, presence_penalty = 0.0, repeat_last_n = 64,
}

--- The options that `new` gives the C sampler, with the defaults filled.
-- It is for logs. It keeps only the keys of `DEFAULTS` and `seed`.
-- @tparam[opt="default"] string preset a key of `presets`
-- @tparam[opt] table opts sampler options
-- @treturn table the options
function M.settings(preset, opts)
	local o = {}
	for k, v in pairs(M.DEFAULTS) do
		o[k] = v
	end
	for _, t in ipairs({ M.presets[preset or "default"] or {}, opts or {} }) do
		for k, v in pairs(t) do
			if (M.DEFAULTS[k] ~= nil or k == "seed") and type(v) == "number" then
				o[k] = v
			end
		end
	end
	return o
end

--- Make a sampler from a preset.
-- Call as `new(preset [, opts])` or `new(opts)`.
-- Fields in opts override the preset.
-- @tparam[opt="default"] string|table preset a key of `presets`, or opts
-- @tparam[opt] table opts sampler options for `lgml.sampler`
-- @return a C sampler object from `lgml.sampler`
-- @raise when the preset name is unknown
function M.new(preset, opts)
	if type(preset) == "table" then
		preset, opts = "default", preset
	end
	local base = M.presets[preset or "default"]
	if base == nil then
		error("unknown sampler preset '" .. tostring(preset) .. "'", 2)
	end
	local o = {}
	for k, v in pairs(base) do
		o[k] = v
	end
	for k, v in pairs(opts or {}) do
		o[k] = v
	end
	return lgml.sampler(o)
end

return M
