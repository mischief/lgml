--- Qwen3.5 dense hybrid decoder (Qwen3.5-27B and its successors).
-- The Qwen3-Next hybrid with multi-section rope and a dense SwiGLU FFN.
-- The model is `lgml.transformer`; see there for the model methods.
-- @module lgml.qwen35
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
