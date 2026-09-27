--- Model kind from a path, without starting a ggml backend.
-- The front uses it to pick the engine for a worker, and to know the
-- modes of a model before a worker starts.
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

--- The files of a Z-Image model directory.
-- The DiT is a gguf with "image" in its name, the text encoder a gguf
-- with "qwen", and the VAE a safetensors file whose name starts with
-- "ae" or has "vae". Case does not matter.
-- @tparam string dir the model directory
-- @treturn ?string the DiT path
-- @treturn ?string the text encoder path
-- @treturn ?string the VAE path
function M.image_files(dir)
	local uv = require("luv")
	dir = dir:gsub("/+$", "")
	local dit, llm, vae
	local h = uv.fs_scandir(dir)
	while h do
		local name = uv.fs_scandir_next(h)
		if not name then
			break
		end
		local low = name:lower()
		if low:match("%.gguf$") and low:find("qwen") then
			llm = dir .. "/" .. name
		elseif low:match("%.gguf$") and low:find("image") then
			dit = dir .. "/" .. name
		elseif low:match("%.safetensors$") and (low:find("^ae") or low:find("vae")) then
			vae = dir .. "/" .. name
		end
	end
	return dit, llm, vae
end

--- The mode names, in the order that lists use.
-- A mode is a kind of request that a model serves. `/v1/models` gives
-- them as `capabilities`:
--
-- - `completion`: chat completions
-- - `multimodal`: chat completions and decisions with image input
-- - `decide`: System One decisions (`/v1/systemone`)
-- - `transcription`: speech to text
-- - `speech`: text to speech
-- - `image_generation`: image generations and edits
-- - `embedding`: embeddings
M.MODES = {
	"completion", "multimodal", "decide", "transcription", "speech", "image_generation", "embedding",
}

--- The engine method that serves each mode.
-- `multimodal` is `chat` with images; decisions with images also need it.
M.METHOD = {
	completion = "chat", multimodal = "chat", decide = "decide",
	transcription = "transcribe", speech = "speak", image_generation = "generate",
	embedding = "embed",
}

-- Modes of the engine modules that lgml-worker can run.
local ENGINES = {
	["lgml.engine"] = { completion = true, decide = true },
	["lgml.diffusion_engine"] = { completion = true },
	["lgml.stt_engine"] = { transcription = true },
	["lgml.image.engine"] = { image_generation = true },
	["lgml.embed_engine"] = { embedding = true },
	["lgml.tts.orpheus"] = { speech = true },
	["lgml.tts.qwen3tts"] = { speech = true },
	["lgml.tts.vibevoice"] = { speech = true },
	["lgml.server.fake_engine"] = {
		completion = true, decide = true, transcription = true, speech = true,
		image_generation = true, embedding = true,
	},
}

-- A set of modes as a list in MODES order.
local function list(set)
	local out = {}
	for _, m in ipairs(M.MODES) do
		if set[m] then
			out[#out + 1] = m
		end
	end
	return out
end

--- Test for a gguf embedding model.
-- The function reads the gguf metadata with `lgml.gguf`, which loads
-- the C module but starts no backend.
-- @tparam string path the model path
-- @treturn boolean true when path is a gguf file whose
-- `<arch>.pooling_type` is set and not 0 (none)
function M.is_embedding(path)
	if type(path) ~= "string" or not path:match("%.gguf$") then
		return false
	end
	local ok, info = pcall(function()
		return require("lgml").gguf(path)
	end)
	local meta = ok and info and info.meta
	if not meta then
		return false
	end
	local v = meta[tostring(meta["general.architecture"]) .. ".pooling_type"]
	return v ~= nil and v ~= 0
end

--- The engine module for a model spec with no `engine`.
-- A Whisper directory or `stt` gives `lgml.stt_engine`, a gguf
-- embedding model `lgml.embed_engine`, and other models `lgml.engine`.
-- @tparam table spec the model spec: `model` and `stt`
-- @treturn string the module name
function M.engine_of(spec)
	if spec.stt or (type(spec.model) == "string" and M.is_whisper(spec.model)) then
		return "lgml.stt_engine"
	elseif M.is_embedding(spec.model) then
		return "lgml.embed_engine"
	end
	return "lgml.engine"
end

--- The modes of a model spec from a server config.
-- The spec fields are `model`, `engine`, `stt`, `mmproj` and `modes`.
-- A list in `modes` gives the modes as they are. Else the engine
-- module gives them. With no engine, `engine_of` picks the module: a
-- Whisper directory or `stt` gives transcription, a gguf embedding
-- model gives embedding, and other models are chat models
-- (`lgml.engine`): completion and decide, and multimodal with an
-- mmproj. The front calls this before any worker starts.
-- @tparam table spec the model spec
-- @treturn {string,...} the modes, in `MODES` order
-- @raise when `modes` has an unknown name, or the engine module is
-- unknown and the spec has no `modes`
function M.modes(spec)
	local set = {}
	if spec.modes ~= nil then
		if type(spec.modes) ~= "table" then
			error("modes must be a list of mode names", 0)
		end
		for _, m in ipairs(spec.modes) do
			if not M.METHOD[m] then
				error("unknown mode " .. tostring(m), 0)
			end
			set[m] = true
		end
		return list(set)
	end
	local engine = spec.engine or M.engine_of(spec)
	local known = ENGINES[engine]
	if not known then
		error(("engine %s: unknown modes; give them in the spec as modes = { ... }"):format(engine), 0)
	end
	for m in pairs(known) do
		set[m] = true
	end
	if set.completion and spec.mmproj then
		set.multimodal = true
	end
	return list(set)
end

--- The modes of a loaded engine, from its methods.
-- caps is a table of method names, as the `caps` of the worker
-- `HELLO` (see `lgml.ipc`), or the engine itself.
-- @tparam table caps `{ chat, transcribe, speak, decide, generate, embed }`,
-- each true or a function when the engine has that method
-- @tparam[opt] boolean vision true when the chat model takes images
-- @treturn {string,...} the modes, in `MODES` order
function M.modes_of(caps, vision)
	local set = {}
	for _, m in ipairs(M.MODES) do
		local v = caps[M.METHOD[m]]
		if v and v ~= false then
			set[m] = true
		end
	end
	set.multimodal = set.completion and vision and true or nil
	return list(set)
end

return M
