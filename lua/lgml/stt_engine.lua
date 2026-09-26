--- Speech-to-text engine for Whisper models.
-- It serves `lgml.server.openai`; the interface is in
-- `lgml.server.engine`. One slot: requests run one at a time. The audio
-- bytes go straight to C; Lua sees only token ids and text.
-- @module lgml.stt_engine
local lgml = require("lgml")
local whisper = require("lgml.whisper")
local stream = require("lgml.server.stream")

local M = {}
local E = {}
E.__index = E

--- Tests for a Hugging Face Whisper model directory.
-- This is `lgml.whisper.is_model`.
-- @function is_model
-- @tparam string dir the model directory
-- @treturn boolean true when config.json has model_type "whisper"
M.is_model = whisper.is_model

--- Loads a model and makes an engine.
-- @tparam table opts the options:
--
-- - `model`: the model directory
-- - `backend`: backend name (default: the best one)
-- - `threads`: CPU thread count (default: `lgml.threads()`)
-- - `id`: model id (default: the last part of the model path)
-- @treturn Engine the engine
-- @raise when the backend or the model does not load
function M.new(opts)
	local be = assert(lgml.backend(opts.backend, { threads = opts.threads }))
	local model = whisper.load(be, opts.model)
	return setmetatable({
		be = be, model = model,
		id = opts.id or opts.model:gsub("/+$", ""):match("([^/]+)$"),
		busy = false, waiting = {},
	}, E)
end

--- A speech-to-text engine. It implements `lgml.server.engine.Engine`.
-- @type Engine

--- Lists the one model of the engine.
-- @treturn {table,...} `{ { id, context_length = n_text_ctx } }`
function E:models()
	return { { id = self.id, context_length = self.model.hp.n_text_ctx } }
end

--- Returns the properties: `{ n_ctx = n_text_ctx, total_slots = 1 }`.
-- @treturn table the properties
function E:props()
	return { n_ctx = self.model.hp.n_text_ctx, total_slots = 1 }
end

function E:acquire()
	while self.busy do
		stream.await(function(done)
			self.waiting[#self.waiting + 1] = done
		end)
	end
	self.busy = true
end

function E:release()
	self.busy = false
	local w = table.remove(self.waiting, 1)
	if w then
		w()
	end
end

local function language(model, req)
	local l = req.language
	if l == nil or l == "" then
		return true, nil
	end
	if type(l) ~= "string" then
		return nil
	end
	local id = model:language_id(l)
	return id ~= nil, id
end

-- Yield between tokens so other messages run. Stop when cancelled.
local function stepper(ctl)
	return function()
		if coroutine.isyieldable() then
			stream.sleep(0)
		end
		return not (ctl and ctl.cancelled and ctl.cancelled())
	end
end

function E:run(req, audio, ctl)
	local a, err = lgml.audio(audio)
	if not a then
		return nil, "audio: " .. err, 400
	end
	if a:rate() ~= whisper.RATE then
		local b = a:resample(whisper.RATE)
		a:close()
		a = b
	end
	local ok, lang = language(self.model, req)
	if not ok then
		return nil, "unknown language " .. tostring(req.language), 400
	end
	local t0 = lgml.clock()
	local r = self.model:transcribe(a, {
		lang = lang, step = stepper(ctl), cancelled = ctl and ctl.cancelled,
	})
	a:close()
	r.seconds = lgml.clock() - t0
	return r
end

--- Transcribes speech (see `lgml.server.engine.Engine.transcribe`).
-- The audio is resampled to the Whisper rate when needed. A Lua error
-- returns status 500.
-- @tparam table req the request: `language` is a code, a name, or nil
-- to detect
-- @tparam string audio the file bytes (WAV)
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control;
-- `ctl.cancelled` is optional
-- @treturn[1] table `{ text, language, duration, segments }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function E:transcribe(req, audio, ctl)
	if type(audio) ~= "string" or #audio == 0 then
		return nil, "no audio", 400
	end
	self:acquire()
	local ok, r, msg, status = pcall(self.run, self, req, audio, ctl)
	self:release()
	if not ok then
		return nil, tostring(r), 500
	end
	return r, msg, status
end

return M
