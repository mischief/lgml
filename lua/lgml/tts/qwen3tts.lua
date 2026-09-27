--- Qwen3-TTS engine (Qwen3-TTS-12Hz-0.6B-CustomVoice).
-- A Qwen3 talker and code predictor emit 16 codec codes per 12.5 Hz
-- frame, and the 12 Hz tokenizer decoder makes 24 kHz audio. The
-- interface is in lgml.server.engine.
-- @module lgml.tts.qwen3tts
local lgml = require("lgml")
local budget = require("lgml.budget")
local qwen3tts = require("lgml.qwen3tts")
local stream = require("lgml.server.stream")

local M = {}
local E = {}
E.__index = E

--- Input limit in characters (4096), as in the OpenAI API.
M.MAX_INPUT = 4096

--- OpenAI voice names to native speakers.
M.openai_voices = {
	alloy = "aiden", ash = "ryan", ballad = "uncle_fu", cedar = "ryan",
	coral = "serena", echo = "ryan", fable = "aiden", marin = "vivian",
	nova = "vivian", onyx = "uncle_fu", sage = "ono_anna", shimmer = "serena",
	verse = "aiden",
}

local function gguf_files(dir)
	local uv = require("luv")
	local out = {}
	local h = assert(uv.fs_scandir(dir))
	while true do
		local name = uv.fs_scandir_next(h)
		if not name then
			break
		end
		if name:match("%.gguf$") then
			out[#out + 1] = dir .. "/" .. name
		end
	end
	table.sort(out)
	return out
end

--- Tokenizer options for `lgml.tokenizer`.
-- The talker gguf has the Qwen vocab but no tokenizer model keys.
M.TOKENIZER = { model = "gpt2", pre = "qwen2" }

-- True when the gguf at path has a usable gpt2 (qwen2) tokenizer.
local function has_tokenizer(path)
	local tk = lgml.tokenizer(path, M.TOKENIZER)
	if tk then
		tk:close()
	end
	return tk ~= nil
end

--- Find the talker, codec and tokenizer gguf files.
-- The tokenizer is tokenizer.gguf, or the talker when it has one.
-- @tparam string model a directory with the files, or the talker file
-- with the rest next to it
-- @treturn table paths: `talker`, `codec`, and `tokenizer` or nil
-- @raise when a file is missing or the talker has no usable tokenizer
function M.find_models(model)
	local uv = require("luv")
	local st = uv.fs_stat(model)
	if not st then
		error(model .. ": no such file or directory")
	end
	local dir = st.type == "directory" and model or model:match("^(.*)/[^/]*$") or "."
	local f = { talker = st.type ~= "directory" and model or nil }
	for _, path in ipairs(gguf_files(dir)) do
		if path:match("/tokenizer%.gguf$") then
			f.tokenizer = path
		elseif path:match("tokenizer%-12hz") and not f.codec then
			f.codec = path
		elseif path:match("qwen3%-tts") and not f.talker then
			f.talker = path
		end
	end
	if not f.talker or not f.codec then
		error(model .. ": need a qwen3-tts talker gguf and qwen3-tts-tokenizer-12hz.gguf")
	end
	if not f.tokenizer and not has_tokenizer(f.talker) then
		error(model .. ": the talker gguf has no gpt2 tokenizer; add tokenizer.gguf" ..
		    " (a Qwen3 gguf) next to it")
	end
	return f
end

--- Make a Qwen3-TTS engine.
-- @tparam table opts options:
-- opts.model: a directory, or the talker gguf.
-- opts.backend: backend name for `lgml.backend`.
-- opts.threads: CPU thread count (default: `lgml.threads()`).
-- opts.id: model id (default "qwen3-tts").
-- @treturn Engine the engine
-- @raise when files are missing or the talker is not CustomVoice
function M.new(opts)
	local be = assert(lgml.backend(opts.backend, { threads = opts.threads }))
	local model = qwen3tts.load(be, M.find_models(opts.model))
	if model.talker.kind ~= "custom_voice" then
		error(opts.model .. ": only the CustomVoice talker is supported")
	end
	return setmetatable({
		be = be, model = model, id = opts.id or "qwen3-tts",
		busy = false, waiting = {},
	}, E)
end

--- A Qwen3-TTS engine.
-- One request runs at a time.
-- @type Engine

--- List the models of the engine.
-- @treturn table list of `{ id, context_length }`
function E:models()
	return { { id = self.id, context_length = self.model.talker.N_CTX } }
end

--- Get the engine properties.
-- @treturn table `n_ctx` and `total_slots` (1)
function E:props()
	return { n_ctx = self.model.talker.N_CTX, total_slots = 1 }
end

--- List the native speaker names.
-- @treturn {string,...} the speakers
function E:voices()
	return self.model.talker.speakers.list
end

--- Wait until the engine is free, then take it.
function E:acquire()
	while self.busy do
		stream.await(function(done)
			self.waiting[#self.waiting + 1] = done
		end)
	end
	self.busy = true
end

--- Free the engine and wake the next waiter.
function E:release()
	self.busy = false
	local w = table.remove(self.waiting, 1)
	if w then
		w()
	end
end

--- Resolve a voice to a native speaker.
-- @tparam[opt] string voice a native speaker name, an OpenAI voice
-- name, or nil for the default ("aiden"); case does not matter
-- @treturn ?string the speaker, or nil when unknown
function E:speaker(voice)
	if voice == nil or voice == "" then
		return "aiden"
	end
	local v = voice:lower()
	if self.model.talker.speakers[v] then
		return v
	end
	return M.openai_voices[v]
end

local function check(self, req)
	if type(req.input) ~= "string" or not req.input:match("%S") then
		return nil, "input must be a non-empty string", 400
	elseif #req.input > M.MAX_INPUT then
		return nil, ("input is longer than %d characters"):format(M.MAX_INPUT), 400
	elseif not utf8.len(req.input) then
		return nil, "input is not valid UTF-8", 400
	elseif req.speed ~= nil and math.abs(req.speed - 1) > 1e-3 then
		return nil, "speed other than 1 is not supported by this model", 400
	elseif req.language ~= nil and (type(req.language) ~= "string" or
	    (req.language ~= "auto" and not self.model.talker.languages[req.language:lower()])) then
		return nil, "unknown language " .. tostring(req.language), 400
	elseif req.seed ~= nil and math.type(req.seed) ~= "integer" then
		return nil, "seed must be an integer", 400
	end
	local spk = self:speaker(req.voice)
	if not spk then
		return nil, "unknown voice " .. tostring(req.voice) .. "; voices: " ..
		    table.concat(self:voices(), ", "), 400
	end
	return spk
end

-- Yield between steps so the worker loop stays live.
local function stepper(ctl)
	return function()
		if coroutine.isyieldable() then
			stream.sleep(0)
		end
		return not (ctl and ctl.cancelled and ctl.cancelled())
	end
end

--- Speak a checked request.
-- @tparam table req the request
-- @tparam string spk native speaker
-- @tparam[opt] table ctl control with `cancelled()`
-- @treturn ?table `{ audio = lgml.audio }`, or nil
-- @treturn[opt] string error message
-- @treturn[opt] int HTTP status
function E:run(req, spk, ctl)
	local t0 = lgml.clock()
	local lang = req.language ~= "auto" and req.language or nil
	local audio, st = self.model:speak(req.input, {
		speaker = spk, language = lang, seed = req.seed, step = stepper(ctl),
	})
	if ctl and ctl.cancelled and ctl.cancelled() then
		return nil, "cancelled", 499
	elseif not audio then
		return nil, "no speech produced", 500
	end
	local dt = lgml.clock() - t0
	require("lgml.log").new("tts"):info("speech", "frames", st.frames, "audio_s", audio:duration(),
	    "s", dt, "rtf", dt / audio:duration(), "frame_cap_hit", not st.eos or nil)
	return { audio = audio }
end

--- Speak a request.
-- @tparam table req request: `input`, `voice`, `speed` (1 only),
-- `language` (name or "auto") and `seed` (integer)
-- @tparam[opt] table ctl control with `cancelled()`
-- @treturn ?table `{ audio = lgml.audio }`, or nil
-- @treturn[opt] string error message
-- @treturn[opt] int HTTP status
function E:speak(req, ctl)
	local spk, msg, status = check(self, req)
	if not spk then
		return nil, msg, status
	end
	self:acquire()
	local ok, r, emsg, estatus = pcall(self.run, self, req, spk, ctl)
	self:release()
	if not ok then
		return nil, tostring(r), budget.status(r, 500)
	end
	return r, emsg, estatus
end

return M
