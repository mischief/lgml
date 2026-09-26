--- OpenAI-shaped endpoints on top of an engine.
-- The engine interface is in `lgml.server.engine`. The sink that this
-- module passes to `Engine:chat` is `lgml.server.engine.Sink`. It
-- streams chunks as server-sent events when the request has
-- `stream = true`, and sends one reply at the end otherwise.
-- @module lgml.server.openai

local json = require("lgml.json")
local http = require("lgml.server.http")
local sse = require("lgml.server.sse")
local engine_api = require("lgml.server.engine")
local multipart = require("lgml.server.multipart")

local M = {}

local function rand_id(prefix, n)
	local t = {}
	for i = 1, n do
		t[i] = string.format("%x", math.random(0, 15))
	end
	return prefix .. table.concat(t)
end

-- request checks

local ROLES = {
	system = true, developer = true, user = true, assistant = true,
	tool = true,
}

-- Removes top-level nulls, so "max_tokens": null reads as absent.
local function strip_nulls(b)
	for k, v in pairs(b) do
		if v == json.null then
			b[k] = nil
		end
	end
end

local function is_object(v)
	return type(v) == "table" and not json.isarray(v)
end

-- Content is a string, null or an array of parts. A text part has a
-- string text.
local function check_content(c)
	if c == nil or c == json.null or type(c) == "string" then
		return true
	elseif not json.isarray(c) then
		return false
	end
	for _, part in ipairs(c) do
		if not is_object(part) or type(part.type) ~= "string" or
		    (part.type == "text" and type(part.text) ~= "string") then
			return false
		end
	end
	return true
end

-- Each call names a function; arguments is a string or an object.
local function check_tool_calls(calls)
	if not json.isarray(calls) then
		return false
	end
	for _, tc in ipairs(calls) do
		local f = is_object(tc) and (tc["function"] or tc)
		if not is_object(f) or type(f.name) ~= "string" or
		    (type(f.arguments) ~= "string" and not is_object(f.arguments)) then
			return false
		end
	end
	return true
end

-- Checks the fields the chat template reads. A null tool_calls or
-- reasoning_content reads as absent.
local function check_messages(msgs)
	if type(msgs) ~= "table" or not json.isarray(msgs) or #msgs == 0 then
		return nil, "messages must be a non-empty array"
	end
	for i, m in ipairs(msgs) do
		local at = ("messages[%d]"):format(i - 1)
		if not is_object(m) or not ROLES[m.role] then
			return nil, at .. ".role is not valid"
		end
		strip_nulls(m)
		if not check_content(m.content) then
			return nil, at .. ".content is not valid"
		elseif m.reasoning_content ~= nil and type(m.reasoning_content) ~= "string" then
			return nil, at .. ".reasoning_content is not valid"
		elseif m.tool_calls ~= nil and not check_tool_calls(m.tool_calls) then
			return nil, at .. ".tool_calls is not valid"
		end
	end
	return true
end

local function check_type(b, key, want, pred)
	local v = b[key]
	if v ~= nil and (type(v) ~= want or (pred and not pred(v))) then
		return nil, key .. " is not valid"
	end
	return true
end

-- cjson decodes every number as a float; 5.0 counts as an integer.
local function positive_int(v)
	local i = math.tointeger(v)
	return i ~= nil and i > 0
end

local function is_list(v)
	return json.isarray(v)
end

local CHECKS = {
	{ "model", "string" }, { "stream", "boolean" },
	{ "stream_options", "table" }, { "temperature", "number" },
	{ "top_p", "number" }, { "max_tokens", "number", positive_int },
	{ "max_completion_tokens", "number", positive_int },
	{ "tools", "table", is_list },
	{ "n", "number", function(v) return v == 1 end },
}

--- Checks a chat request body.
-- It removes top-level nulls and nulls in each message. It checks the
-- messages and the types of `model`, `stream`, `stream_options`,
-- `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`,
-- `tools` and `n` (only 1).
-- @tparam any b the decoded body
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] ?string the bad parameter name
function M.check_chat(b)
	if type(b) ~= "table" or json.isarray(b) then
		return nil, "request body must be a JSON object"
	end
	strip_nulls(b)
	local ok, err = check_messages(b.messages)
	if not ok then
		return nil, err, "messages"
	end
	for _, c in ipairs(CHECKS) do
		ok, err = check_type(b, c[1], c[2], c[3])
		if not ok then
			return nil, err, c[1]
		end
	end
	return true
end

-- sink

local Sink = {}
Sink.__index = Sink

local function new_sink(res, body, model)
	local so = body.stream_options
	return setmetatable({
		res = res, s = res.s, stream = body.stream == true,
		include_usage = so and so.include_usage == true,
		id = rand_id("chatcmpl-", 24), created = os.time(), model = model,
		reasoning_parts = {}, content_parts = {}, tools = {},
		writer = nil, finished = nil, broken = false,
	}, Sink)
end

function Sink:cancelled()
	return self.broken or self.s:gone()
end

local function alive(self)
	if self:cancelled() then
		return nil, "cancelled"
	end
	return true
end

local function chunk(self, choices, extra)
	local c = {
		id = self.id, object = "chat.completion.chunk",
		created = self.created, model = self.model,
		choices = json.array(choices),
	}
	for k, v in pairs(extra or {}) do
		c[k] = v
	end
	return c
end

local function choice(delta, reason)
	return {
		index = 0, delta = json.object(delta),
		logprobs = json.null, finish_reason = reason or json.null,
	}
end

local function send(self, value)
	local ok = self.writer:send(value)
	if not ok then
		self.broken = true
		return nil, "cancelled"
	end
	return true
end

-- Starts the event stream and sends the role chunk.
local function start(self)
	if self.writer then
		return true
	end
	local w = sse.start(self.res)
	if not w then
		self.broken = true
		return nil, "cancelled"
	end
	self.writer = w
	return send(self, chunk(self, {
		choice({ role = "assistant", content = "" }),
	}))
end

local function emit(self, delta)
	if self.finished then
		error("sink used after finish", 3)
	end
	if self:cancelled() then
		return nil, "cancelled"
	end
	if not self.stream then
		return true
	end
	local ok, err = start(self)
	if not ok then
		return nil, err
	end
	return send(self, chunk(self, { choice(delta) }))
end

function Sink:reasoning(text)
	if text == "" then
		return alive(self)
	end
	self.reasoning_parts[#self.reasoning_parts + 1] = text
	return emit(self, { reasoning_content = text })
end

function Sink:content(text)
	if text == "" then
		return alive(self)
	end
	self.content_parts[#self.content_parts + 1] = text
	return emit(self, { content = text })
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

local function usage_object(u)
	u = u or {}
	local p, c = u.prompt_tokens or 0, u.completion_tokens or 0
	return {
		prompt_tokens = p, completion_tokens = c, total_tokens = p + c,
		prompt_tokens_details = { cached_tokens = u.cached_tokens or 0 },
	}
end

function Sink:finish(reason, usage, timings)
	assert(engine_api.reasons[reason], "finish: bad reason")
	if self.finished then
		error("sink finished twice", 2)
	end
	if reason == "stop" and #self.tools > 0 then
		reason = "tool_calls"
	end
	self.finished = { reason = reason, usage = usage_object(usage),
	    timings = timings }
	if not self.stream or self:cancelled() then
		return alive(self)
	end
	local ok, err = start(self)
	if ok then
		ok, err = send(self, chunk(self, { choice({}, reason) },
		    { timings = timings }))
	end
	if ok and self.include_usage then
		ok, err = send(self, chunk(self, {},
		    { usage = self.finished.usage }))
	end
	return ok, err
end

-- responses

local function message(self)
	local content = table.concat(self.content_parts)
	local m = { role = "assistant", content = content }
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

local function reply_whole(self)
	local f = self.finished
	return self.res:json(200, {
		id = self.id, object = "chat.completion", created = self.created,
		model = self.model,
		choices = json.array({ {
			index = 0, message = message(self), logprobs = json.null,
			finish_reason = f.reason,
		} }),
		usage = f.usage, timings = f.timings,
	})
end

-- Ends the stream with an error event, or sends an error response.
local function fail(self, status, msg)
	if not self.writer then
		return self.res:error(status or 500, msg)
	end
	if send(self, http.error_object(status or 500,
	    msg)) then
		self.writer:done()
	end
end

local function run_chat(eng, res, body, model)
	local sink = new_sink(res, body, model)
	local ok, msg, status = eng:chat(body, sink)
	if sink:cancelled() then
		res.close = true
		return
	end
	if not ok then
		return fail(sink, status, msg or "engine failed")
	end
	if not sink.finished then
		sink:finish("stop")
	end
	if not sink.stream then
		return reply_whole(sink)
	end
	if sink:cancelled() or not sink.writer:done() then
		res.close = true
	end
end

-- handlers

local function default_model(eng)
	local list = eng:models()
	return list[1] and list[1].id or "lgml"
end

local function models(eng, res)
	local data = json.array()
	for i, m in ipairs(eng:models()) do
		data[i] = {
			id = m.id, object = "model", created = m.created or 0,
			owned_by = m.owned_by or "lgml",
			context_length = m.context_length,
			meta = { n_ctx_train = m.context_length },
		}
	end
	return res:json(200, { object = "list", data = data })
end

local function props(eng, res)
	local p = eng:props()
	return res:json(200, {
		default_generation_settings = { n_ctx = p.n_ctx,
		    params = json.object() },
		total_slots = p.total_slots,
		model_path = p.model_path or default_model(eng),
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
	local ok, msg, param = M.check_chat(body)
	if not ok then
		return res:error(400, msg, nil, param)
	end
	return run_chat(eng, res, body, body.model or default_model(eng))
end

-- transcriptions

local FORMATS = { json = true, text = true, verbose_json = true }

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

local function reply_transcription(res, r, fmt)
	if fmt == "text" then
		return res:send(200, r.text .. "\n",
		    { ["Content-Type"] = "text/plain; charset=utf-8" })
	elseif fmt == "verbose_json" then
		return res:json(200, verbose(r))
	end
	return res:json(200, { text = r.text })
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
	local r, msg, status = eng:transcribe(treq, file, {
		cancelled = function()
			return res.s:gone()
		end,
	})
	if res.s:gone() then
		res.close = true
		return
	end
	if not r then
		return res:error(status or 500, msg or "transcription failed")
	end
	return reply_transcription(res, r, treq.response_format)
end

-- speech

local SPEECH_FORMATS = { wav = true, pcm = true }

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
	local r, emsg, status = eng:speak(body, {
		cancelled = function()
			return res.s:gone()
		end,
	})
	if res.s:gone() then
		res.close = true
		return
	end
	if not r then
		return res:error(status or 500, emsg or "speech failed")
	end
	local wav = r.wav or r.audio:wav(1, "s16")
	if body.response_format == "pcm" then
		return res:send(200, wav_data(wav), { ["Content-Type"] = "audio/pcm" })
	end
	return res:send(200, wav, { ["Content-Type"] = "audio/wav" })
end

--- Adds the OpenAI routes for eng to srv.
-- The routes are `/health`, `/v1/health`, `/models`, `/v1/models`,
-- `/props`, `/chat/completions`, `/audio/speech` and
-- `/audio/transcriptions`, the last three also under `/v1`. A route for
-- a method that eng does not have gives 404.
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
	srv:route("GET", "/health", health)
	srv:route("GET", "/v1/health", health)
	srv:route("GET", "/v1/models", list)
	srv:route("GET", "/models", list)
	srv:route("GET", "/props", function(_, res)
		return props(eng, res)
	end)
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
end

return M
