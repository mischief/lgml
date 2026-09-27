--- A fake engine for tests.
-- It streams a canned reply with reasoning, content and, when the
-- request has tools, one tool call. Each piece waits a few milliseconds
-- on a luv timer. It also transcribes, speaks, decides, makes images
-- and embeds with canned rules.
-- @module lgml.server.fake_engine

local sleep = require("lgml.server.stream").sleep
local base64 = require("lgml.base64")
local log = require("lgml.log")
local output = require("lgml.chat.output")

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

-- The text of message content: a string, or its text parts joined.
local function text_of(content)
	if type(content) == "string" then
		return content
	end
	local t = {}
	for _, p in ipairs(type(content) == "table" and content or {}) do
		if type(p) == "table" and p.type == "text" then
			t[#t + 1] = p.text
		end
	end
	return table.concat(t, "\n")
end

-- The number of image parts of message content.
local function images_of(content)
	local k = 0
	for _, p in ipairs(type(content) == "table" and content or {}) do
		if type(p) == "table" and p.type == "image_url" then
			k = k + 1
		end
	end
	return k
end

local function last_user(req)
	for i = #req.messages, 1, -1 do
		local m = req.messages[i]
		if m.role == "user" then
			return text_of(m.content)
		end
	end
	return ""
end

local function thinking(req)
	local k = req.chat_template_kwargs
	if type(k) == "table" and k.enable_thinking ~= nil then
		return k.enable_thinking ~= false
	end
	return req.reasoning_effort ~= "none"
end

local function plan(req)
	local p = {}
	if thinking(req) then
		local budget = math.tointeger(req.reasoning_budget) or 2
		for i, text in ipairs({ "The user asks ", "about Paris." }) do
			if i <= budget then
				p[#p + 1] = { "reasoning", text }
			end
		end
	end
	p[#p + 1] = { "content", "Let me " }
	p[#p + 1] = { "content", "check the weather." }
	local tool = req.tools and req.tools[1]
	local name = tool and tool["function"] and tool["function"].name
	local tc = req.tool_choice
	if type(tc) == "table" and type(tc["function"]) == "table" then
		name = tc["function"].name
	end
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
		n = n + #text_of(m.content) // 4 + 4 + 16 * images_of(m.content)
	end
	return n
end

-- A plain rendering of the messages, for the prompt log.
local function render(req)
	local t = {}
	for _, m in ipairs(req.messages) do
		t[#t + 1] = ("<|%s|>\n%s\n"):format(m.role, text_of(m.content))
	end
	t[#t + 1] = "<|assistant|>\n"
	return table.concat(t)
end

-- The parser event of a piece of the plan.
local function event(piece)
	local kind, text = piece[1], piece[2]
	if kind == "tool" then
		return { type = "tool_call_begin", index = 0, name = text }
	elseif kind == "args" then
		return { type = "tool_call_args", index = 0, text = text }
	end
	return { type = kind, text = text }
end

-- The logprobs entry of a piece, with k alternatives.
local function fake_logprob(text, k)
	local function entry(t, v)
		return { token = t, logprob = v, bytes = { t:byte(1, -1) } }
	end
	local e = entry(text, -0.5)
	e.top_logprobs = {}
	for i = 1, k do
		e.top_logprobs[i] = entry(i == 1 and text or string.char(118 + i), -0.5 - (i - 1))
	end
	return e
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
-- The reply has two reasoning pieces, unless thinking is off
-- (`reasoning_effort` `"none"` or `chat_template_kwargs.enable_thinking`
-- false); `reasoning_budget` keeps that many of them. With tools, the
-- reply calls the first tool, or the tool that `tool_choice` names.
-- It calls `sink:prompt` before the first piece. `count_only` finishes
-- at once with the prompt size.
-- The last user message `"fail"` makes the call fail before any output,
-- `"crash"` ends the process with code 3, as a crashed worker, and
-- `"hang"` blocks the process for 30 s, as a hung worker.
-- The prompt counts a quarter token per byte of text, 4 per message and
-- 16 per image part. The reply counts one token per piece and stops at
-- `max_tokens`. `stop`
-- strings apply to the content, as `lgml.chat.output` applies them. With
-- `logprobs`, each content piece is one token with logprob -0.5; its
-- `top_logprobs` alternatives are the piece and then "x", "y", ...
-- with logprobs -0.5, -1.5, -2.5, ... The
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
	elseif last_user(req) == "crash" then
		os.exit(3)
	elseif last_user(req) == "hang" then
		os.execute("sleep 30")
	end
	local max = req.max_completion_tokens or req.max_tokens
	local usage = { prompt_tokens = prompt_tokens(req),
	    completion_tokens = 0, cached_tokens = 0, reasoning_tokens = 0 }
	if req.count_only then
		sink:finish("stop", usage)
		return true
	end
	if log.want_prompt() then
		log.prompt(L, "chat", render(req), "model", self.model, "tokens", usage.prompt_tokens)
	end
	if sink.prompt and not sink:prompt(usage.prompt_tokens, 0) then
		return self:cancel()
	end
	local out = output.new(sink, nil, { stop = req.stop })
	for _, piece in ipairs(plan(req)) do
		sleep(self.delay)
		if sink:cancelled() then
			return self:cancel()
		end
		if max and usage.completion_tokens >= max then
			sink:finish("length", usage)
			return true
		end
		if req.logprobs == true and piece[1] == "content" then
			out:logprob(fake_logprob(piece[2], req.top_logprobs or 0))
		end
		local r = out:deliver({ event(piece) })
		if not r then
			return self:cancel()
		end
		usage.completion_tokens = usage.completion_tokens + 1
		if piece[1] == "reasoning" then
			usage.reasoning_tokens = usage.reasoning_tokens + 1
		end
		if r == "stop" then
			usage.stop_sequence = out.stop_sequence
			sink:finish("stop", usage)
			return true
		end
	end
	if not out:finish() then
		return self:cancel()
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

--- The length of a fake embedding.
M.EMBED_DIM = 8

--- Returns one vector of `EMBED_DIM` values with length 1 per input.
-- Value k of a vector comes from the bytes of the input and k, so the
-- same input gives the same vector. A token list reads as its ids
-- joined by spaces. A string counts a quarter token per byte plus
-- one; a token list counts its ids. Input `"fail"` gives status 500.
-- It waits one delay, so ctl can cancel; a cancel gives status 499.
-- @tparam table req the embedding request
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table `{ data, usage = { prompt_tokens }, timings }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function F:embed(req, ctl)
	sleep(self.delay)
	if ctl and ctl.cancelled and ctl.cancelled() then
		return nil, "cancelled", 499
	end
	local data, n = {}, 0
	for i, x in ipairs(req.input) do
		if x == "fail" then
			return nil, "fake failure", 500
		end
		local text = type(x) == "table" and table.concat(x, " ") or x
		n = n + (type(x) == "table" and #x or #text // 4 + 1)
		local v, s = {}, 0
		for k = 1, M.EMBED_DIM do
			local h = k
			for j = 1, #text do
				h = (h * 31 + text:byte(j)) % 1000003
			end
			v[k] = h % 201 - 100 + 0.5
			s = s + v[k] * v[k]
		end
		for k = 1, M.EMBED_DIM do
			v[k] = v[k] / math.sqrt(s)
		end
		data[i] = v
	end
	return { data = data, usage = { prompt_tokens = n }, timings = { prompt_n = n } }
end

return M
