--- Qwen3 mixture-of-experts decoder (Qwen3-30B-A3B, Qwen3-235B-A22B).
-- Each token runs its top n_used experts in place of the dense FFN.
-- The model is `lgml.transformer`; see there for the model methods.
-- @module lgml.qwen3moe
local transformer = require("lgml.transformer")

local M = {}

--- Read the config from gguf metadata. This is `lgml.transformer.config`.
-- @function hparams
-- @tparam table meta the gguf metadata
-- @treturn table the config
M.hparams = transformer.config

--- Load a model. This is `lgml.transformer.load`.
-- @function load
-- @param be the lgml backend
-- @tparam string path path to the gguf file
-- @tparam[opt] table opts options for `lgml.transformer.load`
-- @treturn lgml.transformer.Model the model
M.load = transformer.load

return M
