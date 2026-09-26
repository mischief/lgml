--- Orpheus TTS engine (canopylabs orpheus-3b-0.1-ft).
-- A llama-3 LM emits SNAC codes, 7 tokens per 2048-sample frame, and
-- the SNAC decoder makes 24 kHz audio. The interface is in
-- lgml.server.engine.
-- @module lgml.tts.orpheus
local lgml = require("lgml")
local transformer = require("lgml.transformer")
local replay = require("lgml.replay")
local snac = require("lgml.snac")
local stream = require("lgml.server.stream")

local M = {}
local E = {}
E.__index = E

local AUDIO_START = 128259	-- start of human
local PROMPT_END = { 128009, 128260, 128261, 128257 }
local CODE0 = 128266		-- <custom_token_10>, code 0 of slot 0
local SLOTS = 7
local CODES = 4096
local SLOT_BOOK = { 1, 2, 3, 3, 2, 3, 3 }	-- codebook of each slot

local RATE = 24000
local N_CTX = 4096
local CHUNK_BYTES = 200		-- text per LM run
local PAUSE = 0.15		-- seconds between text chunks

--- Native voice names.
M.voices = { "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe" }
--- OpenAI voice names to native voices.
M.openai_voices = {
	alloy = "tara", ash = "dan", ballad = "leo", coral = "leah",
	echo = "zac", fable = "leo", onyx = "dan", nova = "jess",
	sage = "mia", shimmer = "zoe", verse = "zac",
}

--- Sampling defaults from CrispASR and the Orpheus reference.
M.sampling = {
	temperature = 0.6, top_k = 50, top_p = 1.0, min_p = 0,
	repeat_penalty = 1.1,
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

-- Find the LM and SNAC gguf files: model is a directory with both, or
-- the LM file with the codec next to it.
local function find_models(model, codec)
	local uv = require("luv")
	local st = uv.fs_stat(model)
	if not st then
		error(model .. ": no such file or directory")
	end
	local dir = st.type == "directory" and model or model:match("^(.*)/[^/]*$") or "."
	local lm = st.type ~= "directory" and model or nil
	for _, f in ipairs(gguf_files(dir)) do
		local info = lgml.gguf(f)
		local arch = info and info.meta["general.architecture"]
		if arch == "snac" and not codec then
			codec = f
		elseif arch == "llama" and not lm then
			lm = f
		end
	end
	if not lm then
		error(model .. ": no llama gguf")
	elseif not codec then
		error(model .. ": no SNAC codec gguf (snac-24khz.gguf) next to it")
	end
	return lm, codec
end

--- Make an Orpheus engine.
-- @tparam table opts options:
-- opts.model: the LM gguf, or a directory with the LM and SNAC ggufs.
-- opts.codec: the SNAC gguf (default: found next to the LM).
-- opts.backend: backend name for `lgml.backend`.
-- opts.id: model id (default "orpheus").
-- opts.n_ctx: context size (default 4096, at most the LM context).
-- @treturn Engine the engine
-- @raise when a model file is missing or does not load
function M.new(opts)
	local be = assert(lgml.backend(opts.backend))
	local lm, codec = find_models(opts.model, opts.codec)
	local e = setmetatable({
		be = be, lm = transformer.load(be, lm), codec = snac.load(be, codec),
		tk = assert(lgml.tokenizer(lm)),
		id = opts.id or "orpheus",
		busy = false, waiting = {},
	}, E)
	e.n_ctx = math.min(opts.n_ctx or N_CTX, e.lm.hp.n_ctx)
	e.cache = e.lm:cache(e.n_ctx)
	return e
end

--- An Orpheus TTS engine.
-- One request runs at a time.
-- @type Engine

--- List the models of the engine.
-- @treturn table list of `{ id, context_length }`
function E:models()
	return { { id = self.id, context_length = self.n_ctx } }
end

--- Get the engine properties.
-- @treturn table `n_ctx` and `total_slots` (1)
function E:props()
	return { n_ctx = self.n_ctx, total_slots = 1 }
end

--- Wait until the engine is free, then take it.
-- Other callers wait in a queue on `lgml.server.stream`.
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

--- Text and voices.
-- @section text

--- Get the native name for a voice.
-- @tparam[opt] string name a native or OpenAI voice name; nil or ""
-- gives "tara"
-- @treturn ?string the native name, or nil
function M.voice(name)
	if name == nil or name == "" then
		return "tara"
	end
	name = name:lower()
	for _, v in ipairs(M.voices) do
		if v == name then
			return v
		end
	end
	return M.openai_voices[name]
end

--- Split text at sentence ends into chunks.
-- Chunks hold at most 200 bytes, when possible.
-- @tparam string text the input text
-- @treturn {string,...} the chunks
function M.split(text)
	text = text:gsub("%s+", " "):match("^%s*(.-)%s*$")
	local parts = {}
	for s in text:gmatch("[^%.!?;]*[%.!?;]+[\"')%]]*%s*") do
		parts[#parts + 1] = s
	end
	local rest = text:gsub("[^%.!?;]*[%.!?;]+[\"')%]]*%s*", "")
	if rest ~= "" then
		parts[#parts + 1] = rest
	end
	local out, cur = {}, ""
	for _, s in ipairs(parts) do
		if cur ~= "" and #cur + #s > CHUNK_BYTES then
			out[#out + 1] = cur
			cur = ""
		end
		cur = cur .. s
	end
	out[#out + 1] = cur
	for i, s in ipairs(out) do
		out[i] = s:match("^%s*(.-)%s*$")
	end
	return out
end

--- An Orpheus TTS engine.
-- @type Engine

--- Build the LM prompt for a voice and text.
-- @tparam string voice native voice name
-- @tparam string text the text
-- @treturn {int,...} prompt token ids
function E:prompt(voice, text)
	local ids = { AUDIO_START, self.tk:special().bos }
	local body = self.tk:encode(voice .. ": " .. text, { special = false, add_bos = false })
	table.move(body, 1, #body, 3, ids)
	return table.move(PROMPT_END, 1, #PROMPT_END, #ids + 1, ids)
end

--- Run ids at cache.n_past through `lgml.replay`.
-- @tparam {int,...} ids token ids
-- @treturn lgml.tensor the logits node; it stays valid until the next
-- pass on the cache
function E:eval(ids)
	return replay.run(self.lm, self.cache, ids, { logits = "last", flash_attn = true })
end

local function sampler(req, max)
	local o = { repeat_last_n = max }
	for k, v in pairs(M.sampling) do
		o[k] = v
	end
	for _, k in ipairs({ "temperature", "top_p", "top_k", "min_p", "seed" }) do
		if type(req[k]) == "number" then
			o[k] = req[k]
		end
	end
	if type(req.repetition_penalty) == "number" then
		o.repeat_penalty = req.repetition_penalty
	end
	if o.seed then
		o.seed = math.tointeger(o.seed) or math.floor(o.seed)
	end
	return lgml.sampler(o)
end

-- SNAC codebook entry of LM token id at position i (0-based), or nil.
local function code(id, i)
	local c = id - CODE0 - (i % SLOTS) * CODES
	if c < 0 or c >= CODES then
		return nil
	end
	return c
end

--- Generate audio codes for one text chunk.
-- @tparam table req the request (sampling fields)
-- @tparam string voice native voice name
-- @tparam string text the text chunk
-- @tparam[opt] table ctl control with `cancelled()`
-- @tparam table stats timing counters to update
-- @treturn ?table codes { c0, c1, c2 }, or nil when cancelled or on error
-- @return the token count, or an error message
-- @treturn[opt] int HTTP status on error
function E:codes(req, voice, text, ctl, stats)
	local prompt = self:prompt(voice, text)
	local max = math.min(self.n_ctx - #prompt - 1, 12 * #text + 700)
	if max < SLOTS then
		return nil, "input too long", 400
	end
	self.cache.n_past = 0
	local smp <close> = sampler(req, max)
	local t0 = lgml.clock()
	local logits = self:eval(prompt)
	stats.prefill = stats.prefill + (lgml.clock() - t0)
	local books, n = { {}, {}, {} }, 0
	t0 = lgml.clock()
	while n < max do
		local id = smp:sample(logits)
		local c = code(id, n)
		if not c then
			break
		end
		smp:accept(id)
		local b = books[SLOT_BOOK[n % SLOTS + 1]]
		b[#b + 1] = c
		n = n + 1
		stream.sleep(0)
		if ctl and ctl.cancelled() then
			return nil
		end
		logits = self:eval({ id })
	end
	stats.decode = stats.decode + (lgml.clock() - t0)
	stats.tokens = stats.tokens + n
	return books, n
end

-- Keep the codes of whole frames only. Returns the frame count.
local function whole_frames(books, n)
	local f = n // SLOTS
	for k, per in ipairs({ 1, 2, 4 }) do
		for i = #books[k], f * per + 1, -1 do
			books[k][i] = nil
		end
	end
	return f
end

--- Compute the frame RMS (10 ms) of pcm.
-- @tparam lgml.tensor pcm f32 samples
-- @tparam int n sample count
-- @treturn {number,...} RMS per frame
-- @treturn int samples per frame
function E:frame_rms(pcm, n)
	local fr = RATE // 100
	local nf = n // fr
	local ones = self.be:tensor("f32", { 1, fr }, string.rep(string.pack("<f", 1 / fr), fr))
	local g <close> = self.be:graph()
	local sq = g:sqr(g:view(pcm, { nf, fr }, { fr * lgml.row_size("f32", 1) }))
	local e = g:mul_mat(sq, ones)
	g:compute(e)
	local t = e:get()
	ones:close()
	for i, v in ipairs(t) do
		t[i] = math.sqrt(v)
	end
	return t, fr
end

--- Find the sample range without leading and trailing silence.
-- @tparam lgml.tensor pcm f32 samples
-- @tparam int n sample count
-- @treturn int a, the first sample
-- @treturn int b, the end sample (exclusive)
function E:trim(pcm, n)
	local rms, fr = self:frame_rms(pcm, n)
	local first, last
	for i, v in ipairs(rms) do
		if v > 0.006 then
			first = first or i
			last = i
		end
	end
	if not first then
		return 0, n
	end
	local a = math.max(0, (first - 1) * fr - RATE // 20)
	local b = math.min(n, last * fr + RATE // 10)
	return a, b
end

--- Append samples [a, b) of pcm to audio.
-- @tparam ?lgml.audio audio the audio, or nil to make a new one
-- @tparam lgml.tensor pcm f32 samples
-- @tparam int a first sample
-- @tparam int b end sample (exclusive)
-- @treturn lgml.audio the audio
function E:append(audio, pcm, a, b)
	local g <close> = self.be:graph()
	local f32 = lgml.row_size("f32", 1)
	local x = g:cont(g:view(pcm, { b - a }, nil, a * f32))
	g:compute(x)
	if not audio then
		return lgml.audio_from(x, RATE)
	end
	audio:append(x)
	return audio
end

--- Append 0.15 s of silence to audio.
-- @tparam lgml.audio audio the audio
function E:pause(audio)
	local n = math.floor(PAUSE * RATE)
	local z = self.be:tensor("f32", { n }, string.rep("\0", 4 * n))
	audio:append(z)
	z:close()
end

--- Convert one text chunk to audio.
-- @tparam table req the request
-- @tparam string voice native voice name
-- @tparam string text the text chunk
-- @tparam[opt] table ctl control with `cancelled()`
-- @tparam ?lgml.audio audio audio to append to
-- @tparam table stats timing counters to update
-- @return the audio, false when no whole frame came out, or nil when
-- cancelled or on error
-- @treturn[opt] string error message
-- @treturn[opt] int HTTP status
function E:chunk(req, voice, text, ctl, audio, stats)
	local books, n, status = self:codes(req, voice, text, ctl, stats)
	if not books then
		return nil, n, status
	end
	if whole_frames(books, n) < 1 then
		return audio or false
	end
	local t0 = lgml.clock()
	local pcm, err = self.codec:decode(books, function()
		stream.sleep(0)
		return not (ctl and ctl.cancelled())
	end)
	if not pcm then
		return nil, err ~= "cancelled" and err or nil, 500
	end
	local len = pcm:shape()[1]
	local a, b = self:trim(pcm, len)
	if audio then
		self:pause(audio)
	end
	audio = self:append(audio, pcm, a, b)
	pcm:close()
	stats.codec = stats.codec + (lgml.clock() - t0)
	return audio
end

--- Convert all chunks of req.input to audio.
-- @tparam table req the request
-- @tparam[opt] table ctl control with `cancelled()`
-- @tparam string voice native voice name
-- @tparam table stats timing counters to update
-- @treturn ?lgml.audio the audio, or nil
-- @treturn[opt] string error message
-- @treturn[opt] int HTTP status
function E:run(req, ctl, voice, stats)
	local audio
	for _, text in ipairs(M.split(req.input)) do
		if text ~= "" then
			local a, err, status = self:chunk(req, voice, text, ctl, audio, stats)
			if a == nil then
				return nil, err or "cancelled", status or 499
			end
			audio = a or audio
		end
	end
	if not audio then
		return nil, "no speech generated", 500
	end
	return audio
end

--- Speak a request.
-- The timings of the last request go to `self.last`.
-- @tparam table req request: `input`, `voice`, `speed` (1 only),
-- `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `repetition_penalty`
-- @tparam[opt] table ctl control with `cancelled()`
-- @treturn ?table `{ audio = lgml.audio }`, or nil
-- @treturn[opt] string error message
-- @treturn[opt] int HTTP status
function E:speak(req, ctl)
	if type(req.input) ~= "string" or req.input:match("^%s*$") then
		return nil, "input must be a non-empty string", 400
	end
	local voice = M.voice(req.voice)
	if not voice then
		return nil, "unknown voice '" .. tostring(req.voice) .. "'; use one of " ..
		    table.concat(M.voices, ", "), 400
	end
	if req.speed ~= nil and math.abs(req.speed - 1) > 1e-3 then
		return nil, "orpheus supports speed 1 only", 400
	end
	if not utf8.len(req.input) then
		return nil, "input is not valid UTF-8", 400
	end
	self:acquire()
	local stats = { prefill = 0, decode = 0, codec = 0, tokens = 0 }
	local t0 = lgml.clock()
	local ok, audio, err, status = pcall(self.run, self, req, ctl, voice, stats)
	self:release()
	if not ok then
		return nil, tostring(audio), 500
	elseif not audio then
		return nil, err, status
	end
	stats.total = lgml.clock() - t0
	stats.duration = audio:duration()
	self.last = stats
	return { audio = audio }
end

return M
