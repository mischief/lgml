--- Vision encoders from llama.cpp mmproj gguf files.
-- `load` picks the module for the projector type. An encoder gives
-- `prepare(img)`, `tokens(img)` and `encode(img)`; see
-- `lgml.vision.gemma4` and `lgml.vision.qwen3vl`.
-- @module lgml.vision
local lgml = require("lgml")

local M = {}

--- Encoder modules by `clip.vision.projector_type`.
M.projectors = { gemma4v = "lgml.vision.gemma4", qwen3vl_merger = "lgml.vision.qwen3vl" }

--- Load a vision encoder.
-- @param be the lgml backend
-- @tparam string path the mmproj gguf
-- @tparam[opt] table opts options for the module's `load`, such as
-- `flash_attn`
-- @treturn table the encoder, with fields `projector` (the type name)
-- and `hp.n_out` (the text embedding size)
-- @raise when the file does not load, has no vision encoder, or has a
-- projector type with no module
function M.load(be, path, opts)
	local info = assert(lgml.gguf(path))
	local meta = info.meta
	if meta["general.architecture"] ~= "clip" or meta["clip.has_vision_encoder"] == false then
		error(path .. ": not a vision mmproj file")
	end
	local ptype = meta["clip.vision.projector_type"] or meta["clip.projector_type"]
	local name = M.projectors[ptype]
	if not name then
		error(("%s: projector type %s not supported"):format(path, tostring(ptype)))
	end
	local w = assert(be:load(path))
	local enc = require(name).load(be, w, meta, opts)
	enc.projector = ptype
	return enc
end

return M
