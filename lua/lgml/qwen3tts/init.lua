--- Qwen3-TTS CustomVoice: text to 24 kHz speech.
-- The talker and code predictor emit 16 codes per frame. The codec
-- turns frames into samples. Samples stay on the device until
-- lgml.audio_from.
-- @classmod lgml.qwen3tts
local lgml = require("lgml")
local talker = require("lgml.qwen3tts.talker")
local codec = require("lgml.qwen3tts.codec")

local M = {}
M.__index = M

--- Maximum bytes of text per talker run.
M.MAX_CHARS = 400
--- Trim resolution in samples (10 ms).
M.BLOCK = 240
--- Seconds of silence kept at each end of a run.
M.PAD = 0.1

--- Talker sampler options.
-- These are the sampling defaults of the reference (CrispASR, qwen-tts
-- generate). The code predictor uses cp_sampling.
M.talker_sampling = {
	temperature = 0.9, top_k = 50, top_p = 1, min_p = 0,
	repeat_penalty = 1.05, repeat_last_n = talker.N_CTX,
}
--- Code predictor sampler options.
M.cp_sampling = { temperature = 0.9, top_k = 50, top_p = 1, min_p = 0 }

--- Load the talker, codec and tokenizer.
-- The model keeps the loaded parts in the fields talker, codec and tk,
-- and the sample rate in rate.
-- @tparam lgml.backend be backend that holds the weights
-- @tparam table files gguf paths
-- files.talker: talker and code predictor gguf.
-- files.codec: codec (Qwen3-TTS-Tokenizer-12Hz) gguf.
-- files.tokenizer: optional tokenizer gguf. The default is files.talker.
-- @treturn lgml.qwen3tts the model
-- @raise when a file is missing or has the wrong architecture
-- @static
function M.load(be, files)
	local m = setmetatable({ be = be }, M)
	m.talker = talker.load(be, files.talker)
	m.codec = codec.load(be, files.codec)
	m.tk = assert(lgml.tokenizer(files.tokenizer or files.talker, { model = "gpt2", pre = "qwen2" }))
	m.rate = m.codec.hp.rate
	m.ones = be:tensor("f32", { 1, M.BLOCK })
	local v = {}
	for i = 1, M.BLOCK do
		v[i] = 1
	end
	m.ones:set(v)
	return m
end

--- Free the talker, codec and tokenizer.
function M:close()
	self.talker:close()
	self.codec:close()
	self.tk:close()
end

local function sampler(base, seed)
	local o = {}
	for k, v in pairs(base) do
		o[k] = v
	end
	o.seed = seed
	return lgml.sampler(o)
end

-- Frame cap for n codepoints of text: the reference bound, limited
-- by the KV cache.
local function max_frames(n, prompt)
	return math.min(math.max(240, n * 12), talker.N_CTX - prompt - 1)
end

--- Generate the codec frames of one run.
-- @string text text of the run
-- @int spk speaker token id
-- @tparam[opt] int lang codec language id, or nil for auto
-- @tparam table opts options
-- opts.seed: sampler seed. The code predictor uses seed + 1.
-- @tparam[opt] function step called after each frame; return false to stop
-- @treturn table frames, each a list of n_groups codes
-- @treturn bool true when the talker emitted codec_eos
function M:frames(text, spk, lang, opts, step)
	local t = self.talker
	local tids, cids, mask = t:prompt(self.tk:encode(text), spk, lang)
	local cap = max_frames(utf8.len(text) or #text, #tids)
	local ts <close> = sampler(M.talker_sampling, opts.seed)
	local cs <close> = sampler(M.cp_sampling, opts.seed and opts.seed + 1)
	local g, logits = t:prefill(tids, cids, mask)
	local frames, eos = {}, false
	while true do
		local id = ts:sample(logits)
		logits:close()
		g:close()
		if id == t.sp.codec_eos then
			eos = true
			break
		end
		ts:accept(id)
		local frame, prev = { id }, id
		for i = 0, t.hp.n_groups - 2 do
			local cg, cl = t:cp_step(i, prev)
			prev = cs:sample(cl)
			cl:close()
			cg:close()
			frame[i + 2] = prev
		end
		frames[#frames + 1] = frame
		if #frames >= cap or (step and not step()) then
			break
		end
		g, logits = t:step(frame, #frames >= 2)
	end
	return frames, eos
end

--- Energy of each BLOCK of the sample leaves.
-- @tparam {tensor,...} leaves f32 sample leaves in order
-- @treturn {number,...} sum of squares per block, as one flat list
function M:energy(leaves)
	local e = {}
	for _, s in ipairs(leaves) do
		local n = s:shape()[1] // M.BLOCK
		local g <close> = self.be:graph()
		local x = g:sqr(g:reshape(s, { n, M.BLOCK }))
		local r = g:mul_mat(self.ones, x)
		g:compute(r)
		table.move(r:get(), 1, n, #e + 1, e)
	end
	return e
end

--- Sample range [first, last) that holds speech, with PAD on each side.
-- A block is voiced when its energy is above 1e-3 of the peak.
-- @tparam {number,...} e block energies from energy
-- @treturn int first sample (0-based), or 0 when there is no speech
-- @treturn int last sample (exclusive), or 0 when there is no speech
function M:voiced(e)
	local peak = 0
	for _, v in ipairs(e) do
		peak = math.max(peak, v)
	end
	local thr = math.max(peak * 1e-3, M.BLOCK * 1e-5)
	local a, b = nil, nil
	for i, v in ipairs(e) do
		if v > thr then
			a = a or i
			b = i
		end
	end
	if not a then
		return 0, 0
	end
	local pad = math.floor(M.PAD * self.rate / M.BLOCK)
	a, b = math.max(1, a - pad), math.min(#e, b + pad)
	return (a - 1) * M.BLOCK, b * M.BLOCK
end

--- Append samples [first, last) of the leaves to audio.
-- @tparam[opt] lgml.audio audio lgml audio to append to, or nil to make one
-- @tparam {tensor,...} leaves f32 sample leaves in order
-- @int first first sample (0-based)
-- @int last last sample (exclusive)
-- @treturn lgml.audio the audio, or nil when the range is empty and audio is nil
function M:append(audio, leaves, first, last)
	local off = 0
	for _, s in ipairs(leaves) do
		local n = s:shape()[1]
		local a, b = math.max(first, off), math.min(last, off + n)
		if a < b then
			local g <close> = self.be:graph()
			local v = g:cont(g:view(s, { b - a }, nil, (a - off) * 4))
			g:compute(v)
			if audio then
				audio:append(v)
			else
				audio = lgml.audio_from(v, self.rate)
			end
		end
		off = off + n
	end
	return audio
end

--- Split text into runs of at most MAX_CHARS bytes.
-- The split is at sentence ends when possible, then at commas or spaces.
-- A split never falls inside a UTF-8 sequence.
-- @string text input text
-- @treturn {string,...} runs with leading and trailing space removed
-- @static
function M.split(text)
	local runs, cur = {}, ""
	local function flush()
		cur = cur:match("^%s*(.-)%s*$")
		if cur ~= "" then
			runs[#runs + 1] = cur
		end
		cur = ""
	end
	for _, p in ipairs({ "。", "！", "？" }) do
		text = text:gsub(p, p .. "\n")
	end
	for piece in text:gmatch("[^%.!?\n]*[%.!?\n]*") do
		if #cur + #piece > M.MAX_CHARS then
			flush()
		end
		while #piece > M.MAX_CHARS do
			local cut = piece:sub(1, M.MAX_CHARS):match("^.*()[,;%s]") or M.MAX_CHARS
			while cut > 1 and (piece:byte(cut + 1) or 0) & 0xc0 == 0x80 do
				cut = cut - 1
			end
			runs[#runs + 1] = piece:sub(1, cut)
			piece = piece:sub(cut + 1)
		end
		cur = cur .. piece
	end
	flush()
	return runs
end

--- Synthesize text.
-- @string text input text
-- @tparam[opt] table opts options
-- opts.speaker: speaker name, case-insensitive. The default is "aiden".
-- opts.language: language name, or nil for the speaker dialect or auto.
-- opts.seed: sampler seed.
-- opts.step: function called during generation; return false to stop.
-- @treturn lgml.audio lgml audio, or nil when there is no speech or step stopped
-- @treturn table stats: frames (count), runs (count), eos (true when
-- every run ended with codec_eos)
-- @raise when the speaker is unknown
function M:speak(text, opts)
	opts = opts or {}
	local t = self.talker
	local spk = assert(t.speakers[(opts.speaker or "aiden"):lower()], "unknown speaker")
	local lang = opts.language and t.languages[opts.language:lower()]
	if not lang and spk.dialect ~= 0 then
		lang = spk.dialect
	end
	local audio, stats = nil, { frames = 0, runs = 0, eos = true }
	for _, run in ipairs(M.split(text)) do
		local frames, eos = self:frames(run, spk.id, lang, opts, opts.step)
		stats.frames, stats.runs = stats.frames + #frames, stats.runs + 1
		stats.eos = stats.eos and eos
		if opts.step and not opts.step() then
			return nil, stats
		end
		if #frames > 0 then
			local leaves = self.codec:decode(frames, opts.step)
			local first, last = self:voiced(self:energy(leaves))
			audio = self:append(audio, leaves, first, last)
			for _, s in ipairs(leaves) do
				s:close()
			end
		end
	end
	return audio, stats
end

return M
