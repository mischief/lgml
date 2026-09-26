--- Speech engine for VibeVoice realtime 0.5B.
-- The interface is in lgml.server.engine. One request runs at a time.
-- Long input is split into chunks of whole sentences. Each chunk is one
-- generation pass. Extra request fields: seed, cfg_scale (1 to 10),
-- steps (4 to 25).
-- @module lgml.tts.vibevoice
local lgml = require("lgml")
local uv = require("luv")
local model = require("lgml.vibevoice.model")
local stream = require("lgml.server.stream")

local M = {}
local E = {}
E.__index = E

--- OpenAI voice names to presets.
M.OPENAI = {
	alloy = "en-Carter_man", ash = "en-Mike_man", ballad = "en-Davis_man",
	coral = "en-Grace_woman", echo = "en-Frank_man", fable = "en-Davis_man",
	onyx = "en-Frank_man", nova = "en-Emma_woman", sage = "en-Grace_woman",
	shimmer = "en-Emma_woman", verse = "en-Mike_man",
}
--- Voice for requests without one.
M.DEFAULT_VOICE = "alloy"
--- Model types to look for in a directory, in order.
M.MODEL_ORDER = { "f16", "q8_0", "q4_k" }
--- Token limit of one text chunk.
M.CHUNK_TOKENS = 48
--- Request defaults: seed, cfg and steps.
M.DEFAULTS = { seed = 42, cfg = 2.0, steps = 20 }
--- Diffusion step limit.
-- The unrolled diffusion graph must fit the graph size.
M.MAX_STEPS = 25

local function scandir(dir)
	local out = {}
	local h = uv.fs_scandir(dir)
	while h do
		local name = uv.fs_scandir_next(h)
		if not name then
			break
		end
		out[#out + 1] = name
	end
	table.sort(out)
	return out
end

local function dirname(path)
	return path:match("^(.*)/[^/]*$") or "."
end

-- The model gguf in dir: the first of MODEL_ORDER that exists.
local function find_model(dir, files)
	for _, q in ipairs(M.MODEL_ORDER) do
		for _, f in ipairs(files) do
			if f:match("^vibevoice%-realtime.*%.gguf$") and f:lower():find(q, 1, true) then
				return dir .. "/" .. f
			end
		end
	end
	return nil
end

local function find_voices(dir, files)
	local v = {}
	for _, f in ipairs(files) do
		local name = f:match("^vibevoice%-voice%-(.+)%.gguf$")
		if name then
			v[name] = dir .. "/" .. f
		end
	end
	return v
end

--- Make a VibeVoice engine.
-- In a directory, the model is the first vibevoice-realtime gguf of
-- `MODEL_ORDER`.
-- @tparam table opts options:
-- opts.model: a gguf path or a directory.
-- opts.backend: backend name for `lgml.backend`.
-- opts.threads: CPU thread count (default: `lgml.threads()`).
-- opts.id: model id (default "vibevoice").
-- opts.voices: directory of vibevoice-voice-*.gguf presets (default:
-- the model directory).
-- @treturn Engine the engine
-- @raise when the model or the voice presets are missing
function M.new(opts)
	local path, dir = opts.model, opts.model
	if uv.fs_stat(path) and uv.fs_stat(path).type == "directory" then
		path = find_model(dir, scandir(dir))
		if not path then
			error(dir .. ": no vibevoice-realtime gguf")
		end
	else
		dir = dirname(path)
	end
	local vdir = opts.voices or dir
	local voices = find_voices(vdir, scandir(vdir))
	if next(voices) == nil then
		error(vdir .. ": no vibevoice-voice-*.gguf presets")
	end
	local be = assert(lgml.backend(opts.backend, { threads = opts.threads }))
	local e = setmetatable({
		be = be, model = model.load(be, path), voices = voices,
		id = opts.id or "vibevoice", busy = false, waiting = {},
	}, E)
	return e
end

--- A VibeVoice engine.
-- @type Engine

--- List the models of the engine.
-- @treturn table list of `{ id, context_length }`
function E:models()
	return { { id = self.id, context_length = model.MAX_FRAMES } }
end

--- Get the engine properties.
-- @treturn table `n_ctx` and `total_slots` (1)
function E:props()
	return { n_ctx = model.MAX_FRAMES, total_slots = 1 }
end

--- List the preset names.
-- @treturn {string,...} the names, sorted
function E:voice_names()
	local t = {}
	for k in pairs(self.voices) do
		t[#t + 1] = k
	end
	table.sort(t)
	return t
end

--- Find the preset path for a voice.
-- Case does not matter.
-- @tparam[opt] string name an OpenAI name, a preset name
-- ("en-Carter_man") or a short name ("carter"); nil gives
-- `DEFAULT_VOICE`
-- @treturn ?string the preset path, or nil
function E:resolve_voice(name)
	name = name or M.DEFAULT_VOICE
	local low = name:lower()
	name = M.OPENAI[low] or name
	if self.voices[name] then
		return self.voices[name]
	end
	low = name:lower()
	for k, p in pairs(self.voices) do
		local short = k:match("^%a+%-(%w+)_") or ""
		if k:lower() == low or short:lower() == low then
			return p
		end
	end
	return nil
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

-- Whitespace runs become one space; typographic apostrophes become '.
local function clean(text)
	text = require("lgml.tokenizer").clean(text)
	text = text:gsub("\u{2019}", "'"):gsub("\u{2018}", "'")
	return (text:gsub("%s+", " "):gsub("^ ", ""):gsub(" $", ""))
end

-- Sentences: text up to . ! ? ; or : and a space, or up to a CJK
-- full stop, exclamation or question mark.
local function sentences(text)
	local out = {}
	for _, c in ipairs({ "\u{3002}", "\u{ff01}", "\u{ff1f}" }) do
		text = text:gsub(c .. "%s*", c .. "\n")
	end
	for part in text:gmatch("[^\n]+") do
		local i = 1
		while i <= #part do
			local _, e = part:find("[%.!%?;:]+[\"')%]]* ", i)
			out[#out + 1] = part:sub(i, e and e - 1 or #part)
			i = (e or #part) + 1
		end
	end
	return out
end

-- Pieces of at most n characters.
local function split_chars(w, n)
	local out, cur, k = {}, {}, 0
	for _, cp in utf8.codes(w) do
		cur[#cur + 1] = utf8.char(cp)
		k = k + 1
		if k == n then
			out[#out + 1], cur, k = table.concat(cur), {}, 0
		end
	end
	if k > 0 then
		out[#out + 1] = table.concat(cur)
	end
	return out
end

-- Words of a sentence; a word longer than max tokens becomes pieces.
local function words(tk, s, max)
	local out = {}
	for w in s:gmatch("%S+") do
		if #tk:encode(w) > max then
			local p = split_chars(w, max // 2)
			table.move(p, 1, #p, #out + 1, out)
		else
			out[#out + 1] = w
		end
	end
	return out
end

-- Split a sentence that is too long at spaces.
local function split_words(tk, s, max, out)
	local cur = nil
	for _, w in ipairs(words(tk, s, max)) do
		local next_ = cur and cur .. " " .. w or w
		if cur and #tk:encode(next_) > max then
			out[#out + 1] = cur
			cur = w
		else
			cur = next_
		end
	end
	if cur then
		out[#out + 1] = cur
	end
end

--- Text helpers.
-- @section text

--- Split text into chunks of whole sentences.
-- Each chunk holds at most max tokens, when possible. A longer
-- sentence splits at spaces, and a longer word splits into pieces.
-- @tparam tokenizer tk tokenizer for the token counts
-- @tparam string text the text
-- @tparam int max token limit per chunk
-- @treturn {string,...} the chunks
function M.chunks(tk, text, max)
	local out, cur = {}, nil
	for _, s in ipairs(sentences(text)) do
		local next_ = cur and cur .. " " .. s or s
		if #tk:encode(next_) <= max then
			cur = next_
		else
			if cur then
				out[#out + 1] = cur
			end
			cur = nil
			if #tk:encode(s) > max then
				split_words(tk, s, max, out)
			else
				cur = s
			end
		end
	end
	if cur then
		out[#out + 1] = cur
	end
	return out
end

local function pause(ctl)
	if coroutine.isyieldable() then
		stream.sleep(0)
	end
	return not (ctl and ctl.cancelled and ctl.cancelled())
end

local function number(v, lo, hi, def)
	if type(v) ~= "number" then
		return def
	end
	return math.max(lo, math.min(hi, v))
end

--- A VibeVoice engine.
-- @type Engine

--- Speak a checked request, one pass per chunk.
-- @tparam table req the request
-- @tparam[opt] table ctl control with `cancelled()`
-- @tparam string voice preset path
-- @treturn ?table `{ audio, sample_rate }`, or nil
-- @treturn[opt] string error message
-- @treturn[opt] int HTTP status
function E:run(req, ctl, voice)
	local m = self.model
	local v = m:voice(voice)
	local opts = {
		seed = math.tointeger(req.seed) or M.DEFAULTS.seed,
		cfg = number(req.cfg_scale, 1, 10, M.DEFAULTS.cfg),
		steps = math.tointeger(number(req.steps, 4, M.MAX_STEPS, M.DEFAULTS.steps) // 1),
		tick = function()
			return pause(ctl)
		end,
	}
	local audio
	for i, text in ipairs(M.chunks(m.tk, clean(req.input), M.CHUNK_TOKENS)) do
		local ids = m.tk:encode(text)
		ids[#ids + 1] = model.NEWLINE
		opts.seed = opts.seed + (i > 1 and 1 or 0)
		local n = m:generate(ids, v, opts)
		if not n or not pause(ctl) then
			return nil, "cancelled", 499
		end
		local a <close>, peaks = m:decode(n)
		audio = m:finish(a, peaks, audio)
	end
	if not audio then
		return nil, "input has no text", 400
	end
	return { audio = audio, sample_rate = model.RATE }
end

--- Speak a request.
-- @tparam table req request: `input`, `voice`, `speed` (1 only),
-- `seed`, `cfg_scale` and `steps`
-- @tparam[opt] table ctl control with `cancelled()`
-- @treturn ?table `{ audio, sample_rate }`, or nil
-- @treturn[opt] string error message
-- @treturn[opt] int HTTP status
function E:speak(req, ctl)
	if type(req.input) ~= "string" or req.input:match("^%s*$") then
		return nil, "input must be a non-empty string", 400
	end
	if req.speed ~= nil and (type(req.speed) ~= "number" or math.abs(req.speed - 1) > 1e-3) then
		return nil, "vibevoice does not support speed other than 1", 400
	end
	local voice = self:resolve_voice(req.voice)
	if not voice then
		return nil, ("unknown voice '%s'; use an OpenAI voice or one of: %s"):format(
		    tostring(req.voice), table.concat(self:voice_names(), ", ")), 400
	end
	self:acquire()
	local ok, r, msg, status = pcall(self.run, self, req, ctl, voice)
	self:release()
	if not ok then
		return nil, tostring(r), 500
	end
	return r, msg, status
end

--- Close the model weights.
function E:close()
	self.model.w:close()
end

return M
