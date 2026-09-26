--- Model kind from a path, without loading ggml.
-- The front uses it to pick the engine for a worker.
-- @module lgml.detect
local json = require("lgml.json")

local M = {}

local function read(path)
	local f = io.open(path, "rb")
	if not f then
		return nil
	end
	local s = f:read("a")
	f:close()
	return s
end

--- Test for a Hugging Face Whisper model directory.
-- The function reads `config.json` in dir.
-- @tparam string dir the model directory
-- @treturn boolean true when config.json has model_type "whisper"
function M.is_whisper(dir)
	local s = read(dir .. "/config.json")
	local cfg = s and json.decode(s)
	return type(cfg) == "table" and cfg.model_type == "whisper"
end

return M
