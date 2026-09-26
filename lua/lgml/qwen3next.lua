--- Qwen3-Next hybrid decoder.
-- Most layers are gated delta net linear attention with a recurrent state;
-- every few layers is gated full attention. Every layer has routed
-- experts and a shared expert with a sigmoid gate.
-- The model is `lgml.transformer`; see there for the model methods.
-- @module lgml.qwen3next
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
