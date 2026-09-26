--- A fake engine for tests.
-- It streams a canned reply with reasoning, content and, when the
-- request has tools, one tool call. Each piece waits a few milliseconds
-- on a luv timer. It also transcribes, speaks, decides and makes images
-- with canned rules.
-- @module lgml.server.fake_engine

local sleep = require("lgml.server.stream").sleep
local base64 = require("lgml.base64")
local log = require("lgml.log")

local L = log.new("engine")

-- A 1x1 PNG.
local PNG = base64.decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

local M = {}
local F = {}
F.__index = F

--- Makes a fake engine.
-- @tparam[opt] table opts the options:
--
-- - `delay`: milliseconds per piece (default 5)
-- - `model`: model id (default `"fake"`)
-- - `n_ctx`: context size (default 4096)
-- - `total_slots`: slot count (default 2)
-- - `on_cancel`: a function called with the engine on each cancel
-- @treturn Fake the engine
function M.new(opts)
	opts = opts or {}
	return setmetatable({
		delay = opts.delay or 5, model = opts.model or "fake",
		n_ctx = opts.n_ctx or 4096, total_slots = opts.total_slots or 2,
		calls = 0, cancels = 0, on_cancel = opts.on_cancel,
	}, F)
end

--- A fake engine. It implements `lgml.server.engine.Engine`.
-- The field `calls` counts chat calls. The field `cancels` counts
-- cancelled chats.
-- @type Fake

--- Lists the one model.
-- @treturn {table,...} `{ { id = model, context_length = n_ctx } }`
function F:models()
	return { { id = self.model, context_length = self.n_ctx } }
end

--- Returns the properties: `{ n_ctx, total_slots }`.
-- @treturn table the properties
function F:props()
	return { n_ctx = self.n_ctx, total_slots = self.total_slots }
end

local function last_user(req)
	for i = #req.messages, 1, -1 do
		local m = req.messages[i]
		if m.role == "user" and type(m.content) == "string" then
			return m.content
		end
	end
	return ""
end

local function plan(req)
	local p = {
		{ "reasoning", "The user asks " }, { "reasoning", "about Paris." },
		{ "content", "Let me " }, { "content", "check the weather." },
	}
	local tool = req.tools and req.tools[1]
	local name = tool and tool["function"] and tool["function"].name
	if name then
		p[#p + 1] = { "tool", name }
		p[#p + 1] = { "args", '{"city":' }
		p[#p + 1] = { "args", ' "Paris"}' }
	end
	return p
end

local function prompt_tokens(req)
	local n = 0
	for _, m in ipairs(req.messages) do
		n = n + #(type(m.content) == "string" and m.content or "") // 4 + 4
	end
	return n
end

-- A plain rendering of the messages, for the prompt log.
local function render(req)
	local t = {}
	for _, m in ipairs(req.messages) do
		t[#t + 1] = ("<|%s|>\n%s\n"):format(m.role, type(m.content) == "string" and m.content or "")
	end
	t[#t + 1] = "<|assistant|>\n"
	return table.concat(t)
end

local function emit(sink, piece, index)
	local kind, text = piece[1], piece[2]
	if kind == "reasoning" then
		return sink:reasoning(text)
	elseif kind == "content" then
		return sink:content(text)
	elseif kind == "tool" then
		return sink:tool_call(text)
	end
	return sink:tool_arguments(index, text)
end

--- Counts a cancel and calls `on_cancel`.
-- @treturn nil
-- @treturn string `"cancelled"`
function F:cancel()
	self.cancels = self.cancels + 1
	if self.on_cancel then
		self.on_cancel(self)
	end
	return nil, "cancelled"
end

--- Streams the canned reply.
-- The user message `"fail"` makes the call fail before any output. The
-- reply counts one token per piece and stops at `max_tokens`. The
-- prompt log (see `lgml.log.prompt`) shows each message as
-- `<|role|>\ncontent\n`.
-- @tparam table req the chat request
-- @tparam lgml.server.engine.Sink sink the reply
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function F:chat(req, sink)
	self.calls = self.calls + 1
	if last_user(req) == "fail" then
		return nil, "fake failure", 500
	end
	local max = req.max_completion_tokens or req.max_tokens
	local usage = { prompt_tokens = prompt_tokens(req),
	    completion_tokens = 0, cached_tokens = 0 }
	if log.want_prompt() then
		log.prompt(L, "chat", render(req), "model", self.model, "tokens", usage.prompt_tokens)
	end
	local index
	for _, piece in ipairs(plan(req)) do
		sleep(self.delay)
		if sink:cancelled() then
			return self:cancel()
		end
		if max and usage.completion_tokens >= max then
			sink:finish("length", usage)
			return true
		end
		local r = emit(sink, piece, index)
		if not r then
			return self:cancel()
		end
		if piece[1] == "tool" then
			index = r
		end
		usage.completion_tokens = usage.completion_tokens + 1
	end
	sink:finish("stop", usage, {
		prompt_n = usage.prompt_tokens, predicted_n = usage.completion_tokens,
	})
	return true
end

--- Returns the text `"fake transcript of N bytes"`.
-- Audio `"fail"` gives a 400 error. It waits one delay, so ctl can
-- cancel; a cancel gives status 499.
-- @tparam table req the request
-- @tparam string audio the file bytes
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table the transcription
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function F:transcribe(req, audio, ctl)
	sleep(self.delay)
	if ctl and ctl.cancelled and ctl.cancelled() then
		return nil, "cancelled", 499
	end
	if audio == "fail" then
		return nil, "audio: not a RIFF WAVE file", 400
	end
	local text = ("fake transcript of %d bytes"):format(#audio)
	return {
		text = text, language = req.language or "en",
		language_name = "english", duration = 1.5,
		segments = { { id = 0, start = 0, ["end"] = 1.5, text = text } },
	}
end

--- Returns a WAV of silence, 10 ms per input byte, at 24 kHz s16 mono.
-- It waits one delay, so ctl can cancel; a cancel gives status 499.
-- @tparam table req the request
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table `{ wav, sample_rate = 24000 }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function F:speak(req, ctl)
	sleep(self.delay)
	if ctl and ctl.cancelled and ctl.cancelled() then
		return nil, "cancelled", 499
	end
	local n = #req.input * 240
	local data = string.rep("\0\0", n)
	local wav = "RIFF" .. string.pack("<I4", 36 + #data) .. "WAVE" .. "fmt " ..
	    string.pack("<I4I2I2I4I4I2I2", 16, 1, 1, 24000, 48000, 2, 16) ..
	    "data" .. string.pack("<I4", #data) .. data
	return { wav = wav, sample_rate = 24000 }
end

-- Lowercase words of 3 or more letters.
local function words(s)
	local t = {}
	for w in s:lower():gmatch("%a%a%a+") do
		t[w] = true
	end
	return t
end

-- Four times the number of distinct words of text that the state has.
local function overlap(state, text)
	local n = 0
	for w in pairs(words(text)) do
		if state[w] then
			n = n + 1
		end
	end
	return 4 * n
end

local function softmax(s)
	local z, out = 0, {}
	for i, v in ipairs(s) do
		out[i] = math.exp(v)
		z = z + out[i]
	end
	for i = 1, #out do
		out[i] = out[i] / z
	end
	return out
end

--- Answers typed questions by word overlap with the state.
-- The log-score of an answer is four times the number of its distinct
-- words (3 or more letters, any case) that the state has. A choice
-- option reads its name and text, a score level its text. A noul
-- scores its instructions and `yes` text against a log-score of 1 for
-- no. State `"fail"` gives status 500. It waits one delay, so ctl can
-- cancel; a cancel gives status 499. Images do not change the
-- answers; an image that is not a PNG or JPEG file gives status 400.
-- A recipe that `lgml.decide.recipe` rejects gives status 422.
-- `timings` has `images` and `image_bytes`, their count and size.
-- @tparam table req the decision request
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table `{ model, answers, usage, timings }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function F:decide(req, ctl)
	sleep(self.delay)
	if ctl and ctl.cancelled and ctl.cancelled() then
		return nil, "cancelled", 499
	end
	if req.state == "fail" then
		return nil, "fake failure", 500
	end
	if req.recipe then
		local decide = require("lgml.decide")
		local rc, err = decide.recipe(decide.default_recipe(), req.recipe)
		if not rc then
			return nil, err, 422
		end
	end
	local nbytes = 0
	for i, img in ipairs(req.images or {}) do
		if img.bytes:sub(1, 8) ~= PNG:sub(1, 8) and img.bytes:sub(1, 3) ~= "\255\216\255" then
			return nil, ("image %d: not a PNG or JPEG file"):format(i), 400
		end
		nbytes = nbytes + #img.bytes
	end
	local state = words(req.state)
	local answers, nout = {}, 0
	for _, q in ipairs(req.questions) do
		if log.want_prompt() then
			log.prompt(L, "decide", ("State:\n%s\n\nQuestion: %s\nAnswer: "):format(req.state,
			    q.instructions or ""), "question", q.id, "type", q.type)
		end
		local s = {}
		if q.type == "noul" then
			s = { overlap(state, q.instructions .. " " .. (q.yes or "")), 1 }
			answers[q.id] = { yes = softmax(s)[1] }
		else
			for i, x in ipairs(q.options or q.levels) do
				s[i] = overlap(state, type(x) == "table" and (x.key .. " " .. x.text) or x)
			end
			answers[q.id] = { probabilities = softmax(s) }
		end
		L:debug("logscores", "question", q.id, "logscores", s)
		nout = nout + #s
	end
	return {
		model = self.model, answers = answers,
		usage = { input_tokens = #req.state // 4 + 16 * #req.questions, output_tokens = nout },
		timings = { images = #(req.images or {}), image_bytes = nbytes },
	}
end

--- Returns n copies of a 1x1 PNG.
-- Prompt `"fail"` gives status 500, and an input image that is not a
-- PNG gives 400. The seed defaults to 42; steps to 8 per image. It
-- waits one delay, so ctl can cancel; a cancel gives status 499.
-- @tparam table req the image request
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table `{ images, seeds, width, height, steps, sample_ms }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function F:generate(req, ctl)
	sleep(self.delay)
	if ctl and ctl.cancelled and ctl.cancelled() then
		return nil, "cancelled", 499
	end
	if req.prompt == "fail" then
		return nil, "fake failure", 500
	elseif req.image and req.image:sub(1, 8) ~= PNG:sub(1, 8) then
		return nil, "image: not a PNG file", 400
	end
	local n, seed, steps = req.n or 1, req.seed or 42, req.steps or 8
	L:debug("image request", "size", ("%dx%d"):format(req.width or 1024, req.height or 1024), "steps", steps,
	    "seed", seed, "n", n, "edit", req.image ~= nil, "prompt", req.prompt)
	local r = { images = {}, seeds = {}, width = req.width or 1024, height = req.height or 1024,
	    steps = n * steps, sample_ms = n * steps }
	for i = 1, n do
		r.images[i], r.seeds[i] = PNG, seed + i - 1
	end
	return r
end

return M
