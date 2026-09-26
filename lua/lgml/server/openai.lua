--- OpenAI-shaped endpoints on top of an engine.
-- The engine interface is in `lgml.server.engine`. The sink that this
-- module passes to `Engine:chat` is `lgml.server.engine.Sink`. It
-- streams chunks as server-sent events when the request has
-- `stream = true`, and sends one reply at the end otherwise.
-- @module lgml.server.openai

local uv = require("luv")
local json = require("lgml.json")
local http = require("lgml.server.http")
local sse = require("lgml.server.sse")
local engine_api = require("lgml.server.engine")
local multipart = require("lgml.server.multipart")
local metrics = require("lgml.server.metrics")
local detect = require("lgml.detect")
local media = require("lgml.chat.media")
local systemone = require("lgml.systemone")
local base64 = require("lgml.base64")
local log = require("lgml.log")

local L = log.new("openai")

local M = {}

local function rand_id(prefix, n)
	local t = {}
	for i = 1, n do
		t[i] = string.format("%x", math.random(0, 15))
	end
	return prefix .. table.concat(t)
end

-- request checks

local chat_request = require("lgml.server.chat_request")

--- The message roles that a chat request can use, as a set. Read only.
-- This is `lgml.server.chat_request.MESSAGE_ROLES`.
M.MESSAGE_ROLES = chat_request.MESSAGE_ROLES

-- Removes top-level nulls, so "max_tokens": null reads as absent.
local function strip_nulls(b)
	for k, v in pairs(b) do
		if v == json.null then
			b[k] = nil
		end
	end
end

--- The top-level fields of a chat request, with their checks and
-- compatibility status. This is `lgml.server.chat_request.FIELDS`.
M.CHAT_FIELDS = chat_request.FIELDS

--- Checks a chat request body. This is `lgml.server.chat_request.check`.
-- @tparam any b the decoded body
-- @tparam[opt] string raw the body text
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] ?string the bad parameter name
M.check_chat = chat_request.check

-- sink

-- One reply: the event stream, the id and the choices. Each choice has
-- its own Sink.
local Reply = {}
Reply.__index = Reply

local function new_reply(res, body, model)
	local so = body.stream_options
	return setmetatable({
		res = res, s = res.s, stream = body.stream == true,
		include_usage = so and so.include_usage == true,
		logprobs = body.logprobs == true,
		id = rand_id("chatcmpl-", 24), created = os.time(), model = model,
		choices = {}, writer = nil, broken = false,
	}, Reply)
end

function Reply:cancelled()
	return self.broken or self.s:gone()
end

local function chunk(r, choices, extra)
	local c = {
		id = r.id, object = "chat.completion.chunk",
		created = r.created, model = r.model, system_fingerprint = json.null,
		choices = json.array(choices),
	}
	-- With include_usage, OpenAI sends usage: null in each other chunk.
	if r.include_usage then
		c.usage = json.null
	end
	for k, v in pairs(extra or {}) do
		c[k] = v
	end
	return c
end

local function send(r, value)
	local ok = r.writer:send(value)
	if not ok then
		r.broken = true
		return nil, "cancelled"
	end
	return true
end

-- Starts the event stream.
local function start(r)
	if r.writer then
		return true
	end
	local w = sse.start(r.res)
	if not w then
		r.broken = true
		return nil, "cancelled"
	end
	r.writer = w
	return true
end

local Sink = {}
Sink.__index = Sink

local function new_sink(reply, index)
	return setmetatable({
		reply = reply, index = index,
		reasoning_parts = {}, content_parts = {}, tools = {},
		lp = reply.logprobs and json.array() or nil,
		started = false, finished = nil,
	}, Sink)
end

function Sink:cancelled()
	return self.reply:cancelled()
end

local function alive(self)
	if self:cancelled() then
		return nil, "cancelled"
	end
	return true
end

local function choice(self, delta, reason, lp)
	return {
		index = self.index, delta = json.object(delta),
		logprobs = lp or json.null, finish_reason = reason or json.null,
	}
end

-- Starts the stream and sends the role chunk of the choice.
local function open(self)
	local r = self.reply
	local ok, err = start(r)
	if not ok or self.started then
		return ok, err
	end
	self.started = true
	return send(r, chunk(r, {
		choice(self, { role = "assistant", content = "", refusal = json.null }),
	}))
end

local function emit(self, delta, lp)
	if self.finished then
		error("sink used after finish", 3)
	end
	if self:cancelled() then
		return nil, "cancelled"
	end
	if not self.reply.stream then
		return true
	end
	local ok, err = open(self)
	if not ok then
		return nil, err
	end
	return send(self.reply, chunk(self.reply, { choice(self, delta, nil, lp) }))
end

function Sink:reasoning(text)
	if text == "" then
		return alive(self)
	end
	self.reasoning_parts[#self.reasoning_parts + 1] = text
	return emit(self, { reasoning_content = text })
end

-- The OpenAI form of logprobs entries from an engine.
local function lp_entries(list)
	local out = json.array()
	for i, e in ipairs(list) do
		local top = json.array()
		for j, t in ipairs(e.top_logprobs or {}) do
			top[j] = { token = t.token, logprob = t.logprob,
			    bytes = t.bytes and json.array(table.move(t.bytes, 1, #t.bytes, 1, {})) or json.null }
		end
		out[i] = { token = e.token, logprob = e.logprob, top_logprobs = top,
		    bytes = e.bytes and json.array(table.move(e.bytes, 1, #e.bytes, 1, {})) or json.null }
	end
	return out
end

function Sink:content(text, logprobs)
	local entries = self.lp and logprobs and #logprobs > 0 and lp_entries(logprobs) or nil
	if text == "" and not entries then
		return alive(self)
	end
	self.content_parts[#self.content_parts + 1] = text
	local lp
	if entries then
		table.move(entries, 1, #entries, #self.lp + 1, self.lp)
		lp = { content = entries, refusal = json.null }
	end
	return emit(self, { content = text }, lp)
end

-- Returns the index of the new call, or nil and "cancelled".
function Sink:tool_call(name, id)
	assert(type(name) == "string", "tool_call: name must be a string")
	id = id or rand_id("call_", 24)
	local index = #self.tools
	self.tools[index + 1] = { id = id, name = name, args = {} }
	local ok, err = emit(self, { tool_calls = { {
		index = index, id = id, type = "function",
		["function"] = { name = name, arguments = "" },
	} } })
	if not ok then
		return nil, err
	end
	return index
end

function Sink:tool_arguments(index, text)
	local t = self.tools[index + 1]
	assert(t, "tool_arguments: no tool call at that index")
	if text == "" then
		return alive(self)
	end
	t.args[#t.args + 1] = text
	return emit(self, { tool_calls = { {
		index = index, ["function"] = { arguments = text },
	} } })
end

function Sink:finish(reason, usage, timings)
	assert(engine_api.reasons[reason], "finish: bad reason")
	if self.finished then
		error("sink finished twice", 2)
	end
	if reason == "stop" and #self.tools > 0 then
		reason = "tool_calls"
	end
	self.finished = { reason = reason, usage = usage or {}, timings = timings }
	if not self.reply.stream or self:cancelled() then
		return alive(self)
	end
	local ok, err = open(self)
	if ok then
		ok, err = send(self.reply, chunk(self.reply, { choice(self, {}, reason) },
		    { timings = timings }))
	end
	return ok, err
end

-- The OpenAI usage object of the reply. The choices share the prompt,
-- so the prompt counts come from the first choice.
local function usage_object(r)
	local first = r.choices[1] and r.choices[1].finished
	local u = first and first.usage or {}
	local p, c, think = u.prompt_tokens or 0, 0, 0
	for _, s in ipairs(r.choices) do
		local f = s.finished and s.finished.usage or {}
		c = c + (f.completion_tokens or 0)
		think = think + (f.reasoning_tokens or 0)
	end
	return {
		prompt_tokens = p, completion_tokens = c, total_tokens = p + c,
		prompt_tokens_details = { cached_tokens = u.cached_tokens or 0, audio_tokens = 0 },
		completion_tokens_details = {
			reasoning_tokens = think, audio_tokens = 0,
			accepted_prediction_tokens = 0, rejected_prediction_tokens = 0,
		},
	}
end

-- The timings of the reply: those of the first choice.
local function timings_of(r)
	local first = r.choices[1] and r.choices[1].finished
	return first and first.timings
end

-- responses

local function message(self)
	local content = table.concat(self.content_parts)
	local m = { role = "assistant", content = content, refusal = json.null }
	if #self.reasoning_parts > 0 then
		m.reasoning_content = table.concat(self.reasoning_parts)
	end
	if #self.tools > 0 then
		local calls = json.array()
		for i, t in ipairs(self.tools) do
			calls[i] = {
				id = t.id, type = "function",
				["function"] = { name = t.name,
				    arguments = table.concat(t.args) },
			}
		end
		m.tool_calls = calls
		if content == "" then
			m.content = json.null
		end
	end
	return m
end

local function reply_whole(r)
	local choices = json.array()
	for i, s in ipairs(r.choices) do
		choices[i] = {
			index = s.index, message = message(s), finish_reason = s.finished.reason,
			logprobs = s.lp and { content = s.lp, refusal = json.null } or json.null,
		}
	end
	local timings = timings_of(r)
	return r.res:json(200, {
		id = r.id, object = "chat.completion", created = r.created,
		model = r.model, system_fingerprint = json.null, choices = choices,
		usage = usage_object(r), timings = timings,
	}, metrics.headers(timings))
end

-- Ends the stream with an error event, or sends an error response.
local function fail(r, status, msg)
	r.res.err = msg
	if not r.writer then
		return r.res:error(status or 500, msg)
	end
	if send(r, http.error_object(status or 500, msg)) then
		r.writer:done()
	end
end

-- Notes for the access line and debug lines on the reply.
local function log_chat(r)
	if r:cancelled() then
		log.note("cancelled", true)
	end
	local first = r.choices[1]
	if not (first and first.finished) then
		return
	end
	local u, t = usage_object(r), timings_of(r) or {}
	local cached = u.prompt_tokens_details.cached_tokens
	local ncalls = 0
	for _, s in ipairs(r.choices) do
		ncalls = ncalls + #s.tools
	end
	log.note("prompt_tokens", u.prompt_tokens)
	log.note("cached_tokens", cached > 0 and cached or nil)
	log.note("completion_tokens", u.completion_tokens)
	log.note("tok_s", t.predicted_per_second)
	log.note("images", (t.images or 0) > 0 and t.images or nil)
	log.note("finish", first.finished.reason)
	log.note("choices", #r.choices > 1 and #r.choices or nil)
	log.note("tool_calls", ncalls > 0 and ncalls or nil)
	if not L:on(log.DEBUG) then
		return
	end
	L:debug("finish", "reason", first.finished.reason, "prompt_tokens", u.prompt_tokens,
	    "cached_tokens", cached, "completion_tokens", u.completion_tokens, "prompt_ms", t.prompt_ms,
	    "predicted_ms", t.predicted_ms)
	for _, s in ipairs(r.choices) do
		L:debug("reply", "choice", s.index,
		    "reasoning", #s.reasoning_parts > 0 and table.concat(s.reasoning_parts) or nil,
		    "content", table.concat(s.content_parts))
		for i, tc in ipairs(s.tools) do
			L:debug("tool call", "choice", s.index, "index", i - 1, "id", tc.id, "name", tc.name,
			    "arguments", table.concat(tc.args))
		end
	end
end

-- The request for choice i: choice i > 1 takes seed + i - 1.
local function choice_request(body, i)
	if i == 1 or math.type(body.seed) ~= "integer" then
		return body
	end
	local t = {}
	for k, v in pairs(body) do
		t[k] = v
	end
	t.seed = body.seed + i - 1
	return t
end

-- Runs the engine once for each choice, one after the other.
local function run_chat(eng, res, body, model)
	local r = new_reply(res, body, model)
	log.note("model", model)
	local ok, msg, status = true, nil, nil
	for i = 1, body.n or 1 do
		local sink = new_sink(r, i - 1)
		r.choices[i] = sink
		ok, msg, status = eng:chat(choice_request(body, i), sink)
		if sink.finished and sink.finished.timings then
			metrics.chat(model, sink.finished.timings)
		end
		if not ok or r:cancelled() then
			break
		end
		if not sink.finished then
			sink:finish("stop")
		end
	end
	metrics.request(model, "chat", ok == true)
	log_chat(r)
	if r:cancelled() then
		res.close = true
		return
	end
	if not ok then
		return fail(r, status, msg or "engine failed")
	end
	if not r.stream then
		return reply_whole(r)
	end
	if r.include_usage and not send(r, chunk(r, {}, { usage = usage_object(r) })) then
		res.close = true
		return
	end
	if r:cancelled() or not r.writer:done() then
		res.close = true
	end
end

-- handlers

local function default_model(eng)
	local list = eng:models()
	return list[1] and list[1].id or "lgml"
end

-- The id of the model that serves a request for id with method what,
-- for the reply and the metrics label. An unknown id gives the serving
-- model, so it makes no new metrics series.
local function served_model(eng, id, what, images)
	if eng.model_for then
		return eng:model_for(id, what, images) or default_model(eng)
	end
	for _, m in ipairs(eng:models()) do
		if m.id == id then
			return id
		end
	end
	return default_model(eng)
end

local function models(eng, res)
	local data = json.array()
	for i, m in ipairs(eng:models()) do
		local caps = json.array()
		for j, c in ipairs(m.capabilities or detect.modes_of(eng, m.vision)) do
			caps[j] = c
		end
		data[i] = {
			id = m.id, object = "model", created = m.created or 0,
			owned_by = m.owned_by or "lgml",
			context_length = m.context_length,
			meta = { n_ctx_train = m.context_length },
			capabilities = caps,
			status = { value = m.status or "loaded" },
			lgml = { status = m.status or "loaded", bytes = m.bytes },
		}
	end
	return res:json(200, { object = "list", data = data })
end

-- POST /models/load and /models/unload: { "model": id }.
local function load_route(eng, req, res, what)
	if type(eng[what]) ~= "function" then
		return res:error(400, engine_api.NO_LOAD)
	end
	local body, err = json.decode(req.body)
	if type(body) ~= "table" then
		return res:error(400, "bad JSON: " .. tostring(err))
	elseif type(body.model) ~= "string" then
		return res:error(400, "model must be a string", nil, "model")
	end
	local ok, msg, status = eng[what](eng, body.model)
	if not ok then
		return res:error(status or 500, msg or what .. " failed")
	end
	return res:json(200, { success = true })
end

-- POST /lora-adapters: a list of { id, scale } sets the global scales.
local function set_adapters(eng, req, res)
	local list, err = json.decode(req.body)
	if list == nil then
		return res:error(400, "bad JSON: " .. tostring(err))
	elseif not json.isarray(list) then
		return res:error(400, "the body must be an array of { id, scale }")
	end
	local ok, msg, status
	if eng.lora_adapters then
		ok, msg, status = eng:lora_adapters(list)
	else
		ok, msg = engine_api.lora_scales(list, #(eng:props().lora or {}))
	end
	if not ok then
		return res:error(status or 400, msg or "lora-adapters failed")
	end
	return res:json(200, { success = true })
end

local function props(eng, res)
	local p = eng:props()
	return res:json(200, {
		default_generation_settings = { n_ctx = p.n_ctx,
		    params = json.object() },
		total_slots = p.total_slots,
		model_path = p.model_path or served_model(eng, nil, "chat"),
		modalities = p.modalities,
		build_info = "lgml",
	})
end

local function chat(eng, req, res)
	if type(eng.chat) ~= "function" then
		return res:error(404, "this server does not serve chat completions")
	end
	local body, err = json.decode(req.body)
	if not body then
		return res:error(400, "bad JSON: " .. err)
	end
	local ok, msg, param = M.check_chat(body, req.body)
	if not ok then
		return res:error(400, msg, nil, param)
	end
	return run_chat(eng, res, body, served_model(eng, body.model, "chat", media.has_images(body.messages)))
end

-- transcriptions

local FORMATS = { json = true, text = true, verbose_json = true }
--- The transcription `response_format` values, as a set. Read only.
M.TRANSCRIPTION_FORMATS = FORMATS

local function verbose(r)
	local segs = json.array()
	for i, sg in ipairs(r.segments or {}) do
		segs[i] = { id = sg.id, start = sg.start, ["end"] = sg["end"], text = sg.text }
	end
	return {
		task = "transcribe", language = r.language_name or r.language,
		duration = r.duration, text = r.text, segments = segs,
	}
end

local function reply_transcription(res, r, fmt, headers)
	if fmt == "text" then
		headers["Content-Type"] = "text/plain; charset=utf-8"
		return res:send(200, r.text .. "\n", headers)
	elseif fmt == "verbose_json" then
		return res:json(200, verbose(r), headers)
	end
	return res:json(200, { text = r.text }, headers)
end

-- Reads the multipart form. Returns the request, the file bytes and
-- the format, or nil, a message and the bad parameter.
local function transcription_form(req)
	local form, err = multipart.from_request(req)
	if not form then
		return nil, err
	end
	local f = form.fields
	local file = form.files.file and form.files.file.data or f.file
	if not file or #file == 0 then
		return nil, "file is required", "file"
	end
	local fmt = f.response_format or "json"
	if not FORMATS[fmt] then
		return nil, "response_format must be json, text or verbose_json", "response_format"
	end
	return {
		model = f.model, language = f.language, prompt = f.prompt,
		response_format = fmt,
	}, file
end

local function transcriptions(eng, req, res)
	if type(eng.transcribe) ~= "function" then
		return res:error(404, "this server does not serve transcriptions")
	end
	local treq, file, param = transcription_form(req)
	if not treq then
		return res:error(400, file, nil, param)
	end
	local t0 = uv.hrtime()
	local model = served_model(eng, treq.model, "transcribe")
	log.note("model", model)
	L:debug("transcribe", "bytes", #file, "language", treq.language, "format", treq.response_format)
	local r, msg, status = eng:transcribe(treq, file, {
		cancelled = function()
			return res.s:gone()
		end,
	})
	if res.s:gone() then
		res.close = true
		return
	end
	metrics.request(model, "transcription", r ~= nil)
	if not r then
		return res:error(status or 500, msg or "transcription failed")
	end
	local ms = (uv.hrtime() - t0) / 1e6
	local fields = { elapsed_ms = ms, audio_seconds = r.duration }
	log.note("audio_s", r.duration)
	if r.duration and r.duration > 0 then
		fields.rtf = ms / 1000 / r.duration
		metrics.add("transcribed_seconds_total", "audio seconds transcribed", model, r.duration)
	end
	return reply_transcription(res, r, treq.response_format, metrics.headers(fields))
end

-- speech

local SPEECH_FORMATS = { wav = true, pcm = true }
--- The speech `response_format` values, as a set. Read only.
M.SPEECH_FORMATS = SPEECH_FORMATS

--- Checks a speech request body.
-- input must be a string of 1 to 16384 bytes. voice is a string. speed
-- is from 0.25 to 4. response_format is `"wav"` (the default, set in
-- b) or `"pcm"`.
-- @tparam any b the decoded body
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] ?string the bad parameter name
function M.check_speech(b)
	if type(b) ~= "table" or json.isarray(b) then
		return nil, "request body must be a JSON object"
	elseif type(b.input) ~= "string" or b.input == "" then
		return nil, "input must be a non-empty string", "input"
	elseif #b.input > 16384 then
		return nil, "input is longer than 16384 bytes", "input"
	elseif b.voice ~= nil and type(b.voice) ~= "string" then
		return nil, "voice must be a string", "voice"
	elseif b.speed ~= nil and (type(b.speed) ~= "number" or b.speed < 0.25 or b.speed > 4) then
		return nil, "speed must be a number from 0.25 to 4", "speed"
	end
	local f = b.response_format or "wav"
	if not SPEECH_FORMATS[f] then
		return nil, "response_format must be wav or pcm", "response_format"
	end
	b.response_format = f
	return true
end

-- The sample data of a WAV file: the payload of its "data" chunk.
local function wav_data(wav)
	local i = 13
	while i + 7 <= #wav do
		local id, size = wav:sub(i, i + 3), string.unpack("<I4", wav, i + 4)
		if id == "data" then
			return wav:sub(i + 8, i + 7 + size)
		end
		i = i + 8 + size + size % 2
	end
	return ""
end

local function speech(eng, req, res)
	if type(eng.speak) ~= "function" then
		return res:error(404, "this server does not serve speech")
	end
	local body, err = json.decode(req.body)
	if not body then
		return res:error(400, "bad JSON: " .. err)
	end
	local ok, msg, param = M.check_speech(body)
	if not ok then
		return res:error(400, msg, nil, param)
	end
	local t0 = uv.hrtime()
	local model = served_model(eng, body.model, "speak")
	log.note("model", model)
	L:debug("speak", "voice", body.voice, "speed", body.speed, "format", body.response_format,
	    "input", body.input)
	local r, emsg, status = eng:speak(body, {
		cancelled = function()
			return res.s:gone()
		end,
	})
	if res.s:gone() then
		res.close = true
		return
	end
	metrics.request(model, "speech", r ~= nil)
	if not r then
		return res:error(status or 500, emsg or "speech failed")
	end
	local wav = r.wav or r.audio:wav(1, "s16")
	local pcm = wav_data(wav)
	-- 16-bit samples; the rate and channels are in the canonical WAV header.
	local rate = #wav >= 28 and string.unpack("<I4", wav, 25) or 0
	local channels = #wav >= 24 and string.unpack("<I2", wav, 23) or 0
	local ms = (uv.hrtime() - t0) / 1e6
	local fields = { elapsed_ms = ms }
	if rate > 0 and channels > 0 then
		fields.sample_rate, fields.channels = rate, channels
		fields.audio_seconds = #pcm / 2 / channels / rate
		log.note("audio_s", fields.audio_seconds)
		fields.rtf = ms / 1000 / math.max(fields.audio_seconds, 1e-9)
		metrics.add("spoken_seconds_total", "audio seconds of speech made", model, fields.audio_seconds)
	end
	local headers = metrics.headers(fields)
	if body.response_format == "pcm" then
		headers["Content-Type"] = "audio/pcm"
		return res:send(200, pcm, headers)
	end
	headers["Content-Type"] = "audio/wav"
	return res:send(200, wav, headers)
end

-- System One decisions

local function systemone_route(eng, req, res)
	if type(eng.decide) ~= "function" then
		return res:json(404, { detail = "this server does not serve decisions" })
	end
	local dreq, body, bstatus = systemone.check(req.body)
	if not dreq then
		return res:json(bstatus or 422, body)
	end
	local t0 = uv.hrtime()
	local model = served_model(eng, dreq.model, "decide", #(dreq.images or {}) > 0)
	log.note("model", model)
	log.note("questions", #dreq.questions)
	log.note("images", #(dreq.images or {}) > 0 and #dreq.images or nil)
	local r, msg, status = eng:decide(dreq, {
		cancelled = function()
			return res.s:gone()
		end,
	})
	if res.s:gone() then
		res.close = true
		return
	end
	metrics.request(model, "decide", r ~= nil)
	if not r then
		if status == 422 then
			-- Only a bad lgml.recipe gets past systemone.check to give 422.
			return res:json(422, systemone.error_body({ "body", "lgml", "recipe" }, msg or "bad recipe"))
		end
		return res:json(status or 500, { detail = msg or "decision failed" })
	end
	if r.usage then
		log.note("prompt_tokens", r.usage.input_tokens)
		log.note("completion_tokens", r.usage.output_tokens)
	end
	log.note("cached_tokens", r.timings and r.timings.cache_n)
	local ok, text = pcall(systemone.reply, dreq, body, r)
	if not ok then
		return res:json(500, { detail = tostring(text) })
	end
	L:debug("decide reply", "body", text)
	local t = r.timings or {}
	if (t.images or 0) > 0 then
		metrics.add("decide_images_total", "images read by decisions", model, t.images)
		metrics.add("image_encode_seconds_total", "time spent encoding input images", model,
		    (t.image_encode_ms or 0) / 1000)
	end
	return res:send(200, text, metrics.headers(t, {
		["Content-Type"] = "application/json",
		["x-jevify-latency-ms"] = ("%.1f"):format((uv.hrtime() - t0) / 1e6),
		["x-lgml-elapsed-ms"] = ("%.3f"):format((uv.hrtime() - t0) / 1e6),
	}))
end

-- images

--- Image sides in pixels are multiples of this.
M.IMAGE_MULTIPLE = 16
--- The largest image side in pixels.
M.IMAGE_MAX_SIDE = 2048

-- A size "WxH" to width and height, or nil.
local function image_size(s)
	local w, h = s:match("^(%d+)x(%d+)$")
	w, h = math.tointeger(tonumber(w)), math.tointeger(tonumber(h))
	if not w or not h then
		return nil
	end
	for _, v in ipairs({ w, h }) do
		if v < M.IMAGE_MULTIPLE or v > M.IMAGE_MAX_SIDE or v % M.IMAGE_MULTIPLE ~= 0 then
			return nil
		end
	end
	return w, h
end

local function int_in(v, lo, hi)
	local i = math.tointeger(v)
	return i ~= nil and i >= lo and i <= hi
end

-- Fields that an image request can hold, with a check for each.
local IMAGE_CHECKS = {
	{ "model", "string" }, { "negative_prompt", "string" },
	{ "n", "number", function(v) return int_in(v, 1, 10) end, "an integer from 1 to 10" },
	{ "steps", "number", function(v) return int_in(v, 1, 100) end, "an integer from 1 to 100" },
	{ "seed", "number", function(v) return int_in(v, 0, (1 << 53) - 1) end,
	    "an integer from 0 to 2^53 - 1" },
	{ "cfg_scale", "number", function(v) return v >= 0 and v <= 30 end, "a number from 0 to 30" },
	{ "strength", "number", function(v) return v > 0 and v <= 1 end, "a number above 0, at most 1" },
}
--- The optional fields of an image request that `check_image` checks.
-- Each entry is `{ name, Lua type, check, text }`, as `CHAT_FIELDS`.
-- Read only.
M.IMAGE_FIELDS = IMAGE_CHECKS

--- Checks an image request and makes the request for `Engine:generate`.
-- b is the decoded JSON body, or the fields of an edit form with
-- numbers as strings. prompt is a string of 1 to 16384 bytes. size is
-- "WxH", with sides that are multiples of 16 from 16 to 2048, or
-- "auto" (default: 1024x1024, or the input size for an edit).
-- response_format is "b64_json", the default; output_format is "png".
-- The fields n (1 to 10), steps (1 to 100), seed, `cfg_scale`,
-- `negative_prompt` and strength (edits, above 0 to 1) are optional.
-- @tparam any b the body
-- @tparam[opt] boolean edit true for an edit
-- @treturn[1] table the request
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] ?string the bad parameter name
function M.check_image(b, edit)
	if type(b) ~= "table" or json.isarray(b) then
		return nil, "request body must be a JSON object"
	end
	strip_nulls(b)
	if type(b.prompt) ~= "string" or b.prompt == "" then
		return nil, "prompt must be a non-empty string", "prompt"
	elseif #b.prompt > 16384 then
		return nil, "prompt is longer than 16384 bytes", "prompt"
	end
	local r = { prompt = b.prompt }
	for _, c in ipairs(IMAGE_CHECKS) do
		local k, v = c[1], b[c[1]]
		if edit and c[2] == "number" and type(v) == "string" then
			v = tonumber(v)
			if v == nil then
				return nil, k .. " must be " .. c[4], k
			end
		end
		if v ~= nil then
			if type(v) ~= c[2] or (c[3] and not c[3](v)) then
				return nil, k .. " must be " .. (c[4] or "a " .. c[2]), k
			end
			r[k] = c[2] == "number" and math.tointeger(v) or v
		end
	end
	if not edit and r.strength then
		return nil, "strength is for edits only", "strength"
	end
	local size = b.size or "auto"
	if size ~= "auto" then
		r.width, r.height = image_size(tostring(size))
		if not r.width then
			return nil, ("size must be auto or WxH, with sides that are multiples of %d from %d to %d")
			    :format(M.IMAGE_MULTIPLE, M.IMAGE_MULTIPLE, M.IMAGE_MAX_SIDE), "size"
		end
	elseif not edit then
		r.width, r.height = 1024, 1024
	end
	if (b.response_format or "b64_json") ~= "b64_json" then
		return nil, "response_format must be b64_json", "response_format"
	elseif (b.output_format or "png") ~= "png" then
		return nil, "output_format must be png", "output_format"
	end
	return r
end

-- Runs the request on eng and sends the OpenAI image reply.
local function run_image(eng, res, ireq)
	local t0 = uv.hrtime()
	local model = served_model(eng, ireq.model, "generate")
	log.note("model", model)
	local r, msg, status = eng:generate(ireq, {
		cancelled = function()
			return res.s:gone()
		end,
	})
	if res.s:gone() then
		res.close = true
		return
	end
	metrics.request(model, "image", r ~= nil)
	if not r then
		return res:error(status or 500, msg or "image generation failed")
	end
	local secs = (uv.hrtime() - t0) / 1e9
	local data = json.array()
	for i, png in ipairs(r.images) do
		data[i] = { b64_json = base64.encode(png), seed = r.seeds and r.seeds[i] }
	end
	metrics.add("images_total", "images made", model, #r.images)
	log.note("images", #r.images)
	log.note("steps", r.steps)
	log.note("size", r.width and ("%dx%d"):format(r.width, r.height))
	log.note("seed", r.seeds and r.seeds[1])
	metrics.add("image_steps_total", "image sampling steps run", model, r.steps)
	metrics.add("image_seconds_total", "time spent making images", model, secs)
	local fields = {
		elapsed_ms = secs * 1000, images = #r.images, steps = r.steps,
		seed = r.seeds and r.seeds[1], sample_ms = r.sample_ms,
	}
	if r.sample_ms and (r.steps or 0) > 0 then
		fields.ms_per_step = r.sample_ms / r.steps
	end
	return res:json(200, {
		created = os.time(), data = data, output_format = "png",
		size = r.width and ("%dx%d"):format(r.width, r.height),
	}, metrics.headers(fields))
end

local function images(eng, req, res)
	if type(eng.generate) ~= "function" then
		return res:error(404, "this server does not serve images")
	end
	local body, err = json.decode(req.body)
	if not body then
		return res:error(400, "bad JSON: " .. err)
	end
	local ireq, msg, param = M.check_image(body)
	if not ireq then
		return res:error(400, msg, nil, param)
	end
	return run_image(eng, res, ireq)
end

local function image_edits(eng, req, res)
	if type(eng.generate) ~= "function" then
		return res:error(404, "this server does not serve images")
	end
	local form, err = multipart.from_request(req)
	if not form then
		return res:error(400, err)
	end
	local file = form.files.image or form.files["image[]"]
	if not file or #file.data == 0 then
		return res:error(400, "image is required", nil, "image")
	elseif form.files.mask or form.fields.mask then
		return res:error(400, "mask is not supported", nil, "mask")
	end
	local ireq, msg, param = M.check_image(form.fields, true)
	if not ireq then
		return res:error(400, msg, nil, param)
	end
	ireq.image = file.data
	return run_image(eng, res, ireq)
end

--- Adds the OpenAI routes for eng to srv.
-- Each route below is also under `/v1`, except `/metrics`:
--
-- - `/health`, `/models`, `/props` and `/openapi.json` (see
--   `lgml.server.openapi`)
-- - `/lora-adapters`: GET lists the adapters from the `lora` field of
--   the engine props, as llama-server lists them. POST takes a list of
--   `{ id, scale }` and calls `Engine:lora_adapters`.
-- - `/metrics`: Prometheus text (see `lgml.server.metrics`), at the path
--   that Prometheus scrapes by default
-- - `/chat/completions`, `/audio/speech`, `/audio/transcriptions`,
--   `/systemone`, `/images/generations` and `/images/edits`
--
-- A route for a method that eng does not have gives 404.
-- `/v1/systemone` is Jev's System One API (see `lgml.systemone`): a
-- bad request gives 422 with a FastAPI `detail` list (400 for an http
-- image URL in `attachments`), and other errors give `{ detail =
-- message }`. The `timings` of the result become `x-lgml-*` headers,
-- such as `x-lgml-images` and `x-lgml-image-encode-ms`.
--
-- Each `/v1/models` entry has `capabilities`, a list of the modes of
-- the model. The names are as llama-server gives them where it has
-- them:
--
-- - `completion`: chat completions
-- - `multimodal`: chat completions and decisions with image input
-- - `decide`: `/v1/systemone`
-- - `transcription`: `/v1/audio/transcriptions`
-- - `speech`: `/v1/audio/speech`
-- - `image_generation`: `/v1/images/generations` and `/v1/images/edits`
--
-- A model entry gives them in its `capabilities` field; else they come
-- from the methods of eng (see `lgml.detect.modes_of`). Each entry also
-- has the load state as llama-server gives it, `status = { value }`,
-- and `lgml = { status, bytes }`; see `lgml.server.engine`.
--
-- `POST /models/load` and `POST /models/unload` (also under `/v1`)
-- take `{ "model": id }` and call `Engine:load` or `Engine:unload`.
-- They give `{ "success": true }`, or an error; 400 with
-- `lgml.server.engine.NO_LOAD` when eng has no such method.
-- @tparam lgml.server.httpd.Server srv the server
-- @tparam lgml.server.engine.Engine eng the engine
-- @raise when eng does not implement the interface
function M.install(srv, eng)
	engine_api.check(eng)
	local function health(_, res)
		return res:json(200, { status = "ok" })
	end
	local function list(_, res)
		return models(eng, res)
	end
	local function completions(req, res)
		return chat(eng, req, res)
	end
	srv:route("GET", "/metrics", function(_, res)
		return res:send(200, metrics.render(),
		    { ["Content-Type"] = "text/plain; version=0.0.4; charset=utf-8" })
	end)
	srv:route("GET", "/health", health)
	local function spec(_, res)
		return res:json(200, require("lgml.server.openapi").document(srv.routes))
	end
	srv:route("GET", "/openapi.json", spec)
	srv:route("GET", "/v1/openapi.json", spec)
	srv:route("GET", "/v1/health", health)
	srv:route("GET", "/v1/models", list)
	srv:route("GET", "/models", list)
	local function get_props(_, res)
		return props(eng, res)
	end
	srv:route("GET", "/props", get_props)
	srv:route("GET", "/v1/props", get_props)
	local function adapters(_, res)
		local lora = eng:props().lora or {}
		return res:json(200, #lora > 0 and lora or json.array())
	end
	srv:route("GET", "/lora-adapters", adapters)
	srv:route("GET", "/v1/lora-adapters", adapters)
	local function post_adapters(req, res)
		return set_adapters(eng, req, res)
	end
	srv:route("POST", "/lora-adapters", post_adapters)
	srv:route("POST", "/v1/lora-adapters", post_adapters)
	srv:route("POST", "/v1/chat/completions", completions)
	srv:route("POST", "/chat/completions", completions)
	local function transcribe(req, res)
		return transcriptions(eng, req, res)
	end
	local function speak(req, res)
		return speech(eng, req, res)
	end
	srv:route("POST", "/v1/audio/speech", speak)
	srv:route("POST", "/audio/speech", speak)
	srv:route("POST", "/v1/audio/transcriptions", transcribe)
	srv:route("POST", "/audio/transcriptions", transcribe)
	local function decide(req, res)
		return systemone_route(eng, req, res)
	end
	srv:route("POST", "/v1/systemone", decide)
	srv:route("POST", "/systemone", decide)
	local function gen(req, res)
		return images(eng, req, res)
	end
	local function edit(req, res)
		return image_edits(eng, req, res)
	end
	srv:route("POST", "/v1/images/generations", gen)
	srv:route("POST", "/images/generations", gen)
	srv:route("POST", "/v1/images/edits", edit)
	srv:route("POST", "/images/edits", edit)
	for _, what in ipairs({ "load", "unload" }) do
		local function h(req, res)
			return load_route(eng, req, res, what)
		end
		srv:route("POST", "/models/" .. what, h)
		srv:route("POST", "/v1/models/" .. what, h)
	end
end

return M
