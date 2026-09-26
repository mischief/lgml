--- Gemma 4 decoder (gguf architecture "gemma4").
-- Sliding layers see the last n_swa positions and keep their keys and
-- values in a ring. Layers have their own head sizes and rope bases,
-- a GeGLU FFN, optional routed experts and per-layer input embeddings.
-- The last layers can share the cache of earlier layers.
-- The model is `lgml.transformer`; see there for the model methods.
-- @module lgml.gemma4
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
