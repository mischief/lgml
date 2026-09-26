--- Parts that the chat, Anthropic Messages and OpenAI Responses
-- surfaces share.
-- Each surface translates its request into the chat request that
-- `Engine:chat` takes (see `lgml.server.engine`), and translates the
-- engine output into its own reply. This module has the shared parts:
-- ids, the served model, image sources, function tools, how a reply
-- ended, usage counts, reasoning signatures, request field checks with
-- the unknown field rule, a sink that groups the output into blocks,
-- and an event stream with keepalive events.
-- @module lgml.server.translate
local uv = require("luv")
local lgml = require("lgml")
local json = require("lgml.json")
local base64 = require("lgml.base64")
local sse = require("lgml.server.sse")
local stream = require("lgml.server.stream")
local metrics = require("lgml.server.metrics")
local log = require("lgml.log")

local L = log.new("translate")

local M = {}

-- ids

local B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

--- A random id: prefix and n base62 characters.
-- @tparam string prefix the prefix, such as `"msg_"`
-- @tparam[opt] int n the length (default 24)
-- @treturn string the id
function M.id(prefix, n)
	local t = {}
	for i = 1, n or 24 do
		local k = math.random(1, 62)
		t[i] = B62:sub(k, k)
	end
	return prefix .. table.concat(t)
end

--- A random id: prefix and n hex digits.
-- @tparam string prefix the prefix, such as `"chatcmpl-"`
-- @tparam int n the length
-- @treturn string the id
function M.hex_id(prefix, n)
	local t = {}
	for i = 1, n do
		t[i] = ("%x"):format(math.random(0, 15))
	end
	return prefix .. table.concat(t)
end

-- models

local function default_model(eng)
	local list = eng:models()
	return list[1] and list[1].id or "lgml"
end

--- The id of the model that serves a request.
-- It is the id for the reply and the metrics label. An unknown id gives
-- the serving model, so it makes no new metrics series.
-- @tparam lgml.server.engine.Engine eng the engine
-- @tparam ?string id the request model
-- @tparam string what the kind: `"chat"`, `"transcribe"` and so on
-- @tparam[opt] boolean images true when the request has images
-- @treturn string the model id
function M.served_model(eng, id, what, images)
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

-- images

--- The image media types that the vision encoders decode, as a set.
M.IMAGE_TYPES = { ["image/png"] = true, ["image/jpeg"] = true }

local MAGIC = {
	{ "image/png", "\137PNG\r\n\26\n" }, { "image/jpeg", "\255\216\255" },
	{ "image/gif", "GIF8" }, { "image/webp", "RIFF", "WEBP" },
}

--- The media type of image bytes from their first bytes.
-- @tparam string bytes the first 12 or more bytes
-- @treturn ?string `"image/png"`, `"image/jpeg"`, `"image/gif"`,
-- `"image/webp"` or nil
function M.sniff(bytes)
	for _, m in ipairs(MAGIC) do
		if bytes:sub(1, #m[2]) == m[2] and (not m[3] or bytes:sub(9, 12) == m[3]) then
			return m[1]
		end
	end
	return nil
end

--- A chat image part from base64 data.
-- The data is plain base64 with no `data:` prefix. The bytes must be a
-- PNG or JPEG image, and match mime when it is given.
-- @tparam ?string mime the media type that the client gives
-- @tparam any data the base64 text
-- @treturn[1] table `{ type = "image_url", image_url = { url } }` with a data URL
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.image_part(mime, data)
	if type(data) ~= "string" or data == "" then
		return nil, "data must be a non-empty base64 string"
	elseif data:find("^data:") then
		return nil, "data must be plain base64, not a data URL"
	end
	local head = base64.decode(data:sub(1, 16))
	local kind = head and M.sniff(head)
	if not kind then
		return nil, "the data is not a PNG, JPEG, GIF or WebP image"
	elseif mime and mime ~= kind then
		return nil, ("media_type is %s, but the data is %s"):format(mime, kind)
	elseif not M.IMAGE_TYPES[kind] then
		return nil, kind .. " is not supported by this server; send image/png or image/jpeg"
	end
	return { type = "image_url", image_url = { url = "data:" .. kind .. ";base64," .. data } }
end

--- A chat image part from a URL.
-- Only base64 data URLs work. http and https URLs give an error: the
-- server does not fetch images.
-- @tparam any url the URL
-- @treturn[1] table the part, as `image_part` gives it
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.url_image(url)
	if type(url) ~= "string" then
		return nil, "the image URL must be a string"
	end
	local scheme = (url:match("^(%a[%w+.-]*):") or ""):lower()
	if scheme == "http" or scheme == "https" then
		return nil, "http and https image URLs are not supported by this server; send base64 data"
	elseif scheme ~= "data" then
		return nil, "the image URL must be a base64 data URL"
	end
	local mime, data = url:match("^[^:]+:([^;,]*);base64,(.*)$")
	if not mime then
		return nil, "the data URL must be base64: data:image/png;base64,..."
	end
	return M.image_part(mime:lower(), data)
end

-- tools and thinking

--- A chat function tool.
-- @tparam string name the name
-- @tparam ?string description the description
-- @tparam ?table parameters the JSON schema of the arguments
-- @tparam ?boolean strict the strict flag
-- @treturn table `{ type = "function", function = { ... } }`
function M.function_tool(name, description, parameters, strict)
	return { type = "function", ["function"] = {
		name = name, description = description, parameters = parameters, strict = strict,
	} }
end

--- Turns thinking on or off in a chat request.
-- It sets `chat_template_kwargs.enable_thinking`.
-- @tparam table chat the chat request
-- @tparam boolean on true for on
function M.set_thinking(chat, on)
	chat.chat_template_kwargs = chat.chat_template_kwargs or {}
	chat.chat_template_kwargs.enable_thinking = on
end

-- endings and usage

--- How a reply ended.
-- `"tool_calls"`: the reply has tool calls. `"length"`: the token limit.
-- `"context"`: the context filled before the token limit.
-- `"stop_sequence"`: a stop string. `"stop"`: the model ended the reply.
-- @tparam ?table f the finish: `{ reason, usage }`
-- @tparam int ntools the number of tool calls
-- @tparam[opt] int max the token limit of the request
-- @treturn string the ending
function M.ending(f, ntools, max)
	if not f then
		return "stop"
	end
	local u = f.usage or {}
	if f.reason == "length" then
		if max and u.completion_tokens and u.completion_tokens < max then
			return "context"
		end
		return "length"
	elseif f.reason == "tool_calls" or ntools > 0 then
		return "tool_calls"
	elseif u.stop_sequence then
		return "stop_sequence"
	end
	return "stop"
end

--- The token counts of a finish usage.
-- @tparam ?table u the usage of `Sink:finish`
-- @treturn table `{ prompt, cached, output, reasoning }`; output counts
-- the reasoning tokens too
function M.counts(u)
	u = u or {}
	return {
		prompt = u.prompt_tokens or 0, cached = u.cached_tokens or 0,
		output = u.completion_tokens or 0, reasoning = u.reasoning_tokens or 0,
	}
end

-- reasoning signatures

local function unhex(h)
	return (h:gsub("..", function(x)
		return string.char(tonumber(x, 16))
	end))
end

local function xor(s, b)
	return (s:gsub(".", function(c)
		return string.char(c:byte() ~ b)
	end))
end

--- HMAC-SHA256 (RFC 2104).
-- @tparam string key the key
-- @tparam string msg the message
-- @treturn string the 32-byte MAC
function M.hmac(key, msg)
	if #key > 64 then
		key = unhex(lgml.sha256(key))
	end
	key = key .. ("\0"):rep(64 - #key)
	return unhex(lgml.sha256(xor(key, 0x5c) .. unhex(lgml.sha256(xor(key, 0x36) .. msg))))
end

--- Base64url with no padding.
-- @tparam string s the bytes
-- @treturn string the text
function M.b64url(s)
	return (base64.encode(s):gsub("[+/=]", { ["+"] = "-", ["/"] = "_", ["="] = "" }))
end

--- Decodes base64url with or without padding.
-- @tparam string s the text
-- @treturn ?string the bytes, or nil for bad text
function M.unb64url(s)
	s = s:gsub("[-_]", { ["-"] = "+", ["_"] = "/" })
	return base64.decode(s .. ("="):rep((4 - #s % 4) % 4))
end

--- The prefix of the reasoning signatures that lgml makes.
M.SIGNATURE_PREFIX = "lgml1."

--- Signs reasoning text.
-- The signature is `lgml1.<payload>.<mac>`. payload is the base64url
-- text when with_text is true, else empty. mac is the base64url
-- HMAC-SHA256 of the model id, a zero byte and the text.
-- @tparam string key the server key
-- @tparam string model the model id
-- @tparam string text the reasoning text
-- @tparam[opt] boolean with_text put the text in the signature
-- @treturn string the signature
function M.sign(key, model, text, with_text)
	return M.SIGNATURE_PREFIX .. (with_text and M.b64url(text) or "") .. "." ..
	    M.b64url(M.hmac(key, model .. "\0" .. text))
end

--- Tells if a signature is from lgml.
-- @tparam any sig the signature
-- @treturn boolean true when it has the lgml prefix
function M.ours(sig)
	return type(sig) == "string" and sig:sub(1, #M.SIGNATURE_PREFIX) == M.SIGNATURE_PREFIX
end

--- Checks an lgml signature and returns the reasoning text.
-- The text comes from the signature payload, or else from text.
-- @tparam string key the server key
-- @tparam string model the model id
-- @tparam string sig the signature
-- @tparam[opt] string text the text of the block
-- @treturn ?string the reasoning text, or nil when the signature fails
function M.verify(key, model, sig, text)
	local payload, mac = sig:match("^lgml1%.([%w_-]*)%.([%w_-]+)$")
	if not payload then
		return nil
	end
	if payload ~= "" then
		text = M.unb64url(payload)
	end
	if type(text) ~= "string" or M.b64url(M.hmac(key, model .. "\0" .. text)) ~= mac then
		return nil
	end
	return text
end

local function random_bytes(n)
	local f = io.open("/dev/urandom", "rb")
	local s = f and f:read(n)
	if f then
		f:close()
	end
	if s and #s == n then
		return s
	end
	local t = {}
	for i = 1, n do
		t[i] = string.char(math.random(0, 255))
	end
	return table.concat(t)
end

local function mkdirs(dir)
	local at = dir:sub(1, 1) == "/" and "" or "."
	for part in dir:gmatch("[^/]+") do
		at = at .. "/" .. part
		uv.fs_mkdir(at, 448)
	end
end

--- The size of the server key in bytes.
M.KEY_BYTES = 32

--- Loads the server key for reasoning signatures.
-- The key is the file `signing.key` in dir. When the file is missing, it
-- makes a random key and writes it with mode 0600, so signatures still
-- verify after a restart. With no dir, or when the file cannot be
-- written, the key lives in memory only.
-- @tparam ?string dir the state directory
-- @treturn string the key
function M.load_key(dir)
	if not dir then
		return random_bytes(M.KEY_BYTES)
	end
	local path = dir .. "/signing.key"
	local f = io.open(path, "rb")
	if f then
		local k = f:read("a")
		f:close()
		if k and #k == M.KEY_BYTES then
			return k
		end
		L:warn("bad key file; making a new key", "path", path)
	end
	local key = random_bytes(M.KEY_BYTES)
	mkdirs(dir)
	local fd = uv.fs_open(path, "w", 384)
	if not fd or uv.fs_write(fd, key) ~= #key then
		L:warn("cannot write the key file; signatures end at restart", "path", path)
	end
	if fd then
		uv.fs_close(fd)
	end
	return key
end

-- request fields

--- The rule for unknown request fields of the Messages and Responses
-- APIs: `"reject"` gives 400, `"ignore"` accepts them and names them in
-- the `x-lgml-ignored` header. The chat API does not use it.
M.unknown_fields = "reject"

--- Sets the rule for unknown request fields.
-- @tparam string rule `"reject"` or `"ignore"`
-- @raise when the rule is not valid
function M.set_unknown_fields(rule)
	assert(rule == "reject" or rule == "ignore", "unknown fields: reject or ignore")
	M.unknown_fields = rule
end

--- Makes the context of one request.
-- It collects the fields that the server accepts with no effect.
-- @treturn Context the context
function M.context()
	return setmetatable({ ignored = {}, seen = {} }, { __index = M.Context })
end

--- The context of one request.
-- @type Context
M.Context = {}

--- Notes a field that the server accepts with no effect.
-- The `x-lgml-ignored` header of the reply names it.
-- @tparam string path the field path
function M.Context:ignore(path)
	if not self.seen[path] then
		self.seen[path] = true
		self.ignored[#self.ignored + 1] = path
	end
end

--- Handles a field that the server does not know.
-- With the rule `"reject"` it returns an error message; else it notes
-- the field and logs it at info.
-- @tparam string path the field path
-- @tparam string msg the error message for the rule `"reject"`
-- @treturn ?string the error message, or nil when the field is ignored
function M.Context:unknown(path, msg)
	if M.unknown_fields == "reject" then
		return msg
	end
	L:info("unknown field ignored", "field", path)
	self:ignore(path)
	return nil
end

--- Adds the `x-lgml-ignored` header when fields were ignored.
-- @tparam table headers the headers to add to
-- @treturn table headers
function M.Context:headers(headers)
	if #self.ignored > 0 then
		headers["x-lgml-ignored"] = table.concat(self.ignored, ",")
	end
	return headers
end

local function type_ok(v, types)
	for t in types:gmatch("[^|]+") do
		if type(v) == t then
			return true
		end
	end
	return false
end

--- Checks the fields of an object against a field table.
-- Each entry is `{ name, types, check, text, status = s, label = l }`,
-- as `lgml.server.chat_request.FIELDS`. A null value reads as absent
-- and is removed. A key that is not in the table goes to
-- `Context:unknown` with the message unknown(path).
-- @tparam table obj the object
-- @tparam table fields the field table
-- @tparam string at the path of obj, with a trailing separator or ""
-- @tparam Context ctx the request context
-- @tparam function unknown makes the message for an unknown field path
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] string the path of the bad field
function M.check_fields(obj, fields, at, ctx, unknown)
	local by = {}
	for _, c in ipairs(fields) do
		by[c[1]] = c
	end
	for k, v in pairs(obj) do
		if v == json.null then
			obj[k] = nil
		elseif type(k) ~= "string" or not by[k] then
			local path = at .. tostring(k)
			local msg = ctx:unknown(path, unknown(path))
			if msg then
				return nil, msg, path
			end
			obj[k] = nil
		end
	end
	for _, c in ipairs(fields) do
		local v = obj[c[1]]
		if v ~= nil then
			local good, msg = type_ok(v, c[2]), nil
			if good and c[3] then
				good, msg = c[3](v)
			end
			if not good then
				local path = at .. c[1]
				return nil, msg or (c[4] and ("%s: must be %s"):format(path, c[4]) or path .. ": not valid"), path
			end
		end
	end
	return true
end

-- the block sink

--- Makes a sink that groups engine output into blocks.
-- A block is reasoning, text or one tool call, in output order. Each
-- block is `{ kind, index, parts }`, kind `"reasoning"`, `"text"` or
-- `"tool"`, index the position from 0; a tool block also has `name`,
-- `id` and `call`, the tool call index from 0. The handler, when given,
-- gets each change as it happens: `prompt(n, cached)`, `open(block)`,
-- `delta(block, text)` and `close(block)`. Each returns true, or nil to
-- cancel the reply.
-- @tparam lgml.server.http.Response res the response
-- @tparam[opt] table handler the handler
-- @tparam[opt] table opts the options: `tool_id`, the prefix of the
-- ids that the sink makes for tool calls without one (default `"call_"`)
-- @treturn Blocks the sink; it implements `lgml.server.engine.Sink`
function M.blocks(res, handler, opts)
	return setmetatable({
		res = res, handler = handler, list = {}, calls = {}, cur = nil,
		finished = nil, broken = false, tool_id = opts and opts.tool_id or "call_",
	}, { __index = M.Blocks })
end

--- A sink that groups engine output into blocks.
-- The field `list` holds the blocks, `calls` the tool blocks by call
-- index from 1, and `finished` the finish: `{ reason, usage, timings }`.
-- `final` holds the finish already while the last block closes.
-- After `prompt`, `prompt_n` and `cached_n` hold the prompt size.
-- @type Blocks
M.Blocks = {}

function M.Blocks:cancelled()
	return self.broken or self.res.s:gone()
end

local function call(self, name, ...)
	if self.finished then
		error("sink used after finish", 3)
	elseif self:cancelled() then
		return nil, "cancelled"
	end
	local h = self.handler
	if not (h and h[name]) then
		return true
	end
	local ok, err = h[name](h, ...)
	if not ok then
		self.broken = true
		return nil, err or "cancelled"
	end
	return true
end

local function close(self)
	local b = self.cur
	self.cur = nil
	if b then
		return call(self, "close", b)
	end
	return true
end

local function open(self, b)
	local ok, err = close(self)
	if not ok then
		return nil, err
	end
	b.index = #self.list
	self.list[#self.list + 1] = b
	self.cur = b
	return call(self, "open", b)
end

--- Tells the prompt size (see `lgml.server.engine.Sink.prompt`).
-- @tparam int n the prompt tokens
-- @tparam int cached the cached prompt tokens
function M.Blocks:prompt(n, cached)
	self.prompt_n, self.cached_n = n, cached
	return call(self, "prompt", n, cached)
end

local function text_block(self, kind, text)
	if text == "" then
		return call(self, "none")
	end
	local b = self.cur
	if not b or b.kind ~= kind then
		b = { kind = kind, parts = {} }
		local ok, err = open(self, b)
		if not ok then
			return nil, err
		end
	end
	b.parts[#b.parts + 1] = text
	return call(self, "delta", b, text)
end

function M.Blocks:reasoning(text)
	return text_block(self, "reasoning", text)
end

function M.Blocks:content(text)
	return text_block(self, "text", text)
end

function M.Blocks:tool_call(name, id)
	assert(type(name) == "string", "tool_call: name must be a string")
	local b = { kind = "tool", parts = {}, name = name, id = id or M.id(self.tool_id), call = #self.calls }
	self.calls[#self.calls + 1] = b
	local ok, err = open(self, b)
	if not ok then
		return nil, err
	end
	return b.call
end

function M.Blocks:tool_arguments(index, text)
	local b = self.calls[index + 1]
	assert(b, "tool_arguments: no tool call at that index")
	if text == "" then
		return call(self, "none")
	end
	b.parts[#b.parts + 1] = text
	return call(self, "delta", b, text)
end

function M.Blocks:finish(reason, usage, timings)
	if self.finished then
		error("sink finished twice", 2)
	end
	-- The handler can read the finish while the last block closes.
	self.final = { reason = reason, usage = usage or {}, timings = timings }
	close(self)
	self.finished = self.final
	if self:cancelled() then
		return nil, "cancelled"
	end
	return true
end

--- The text of a block.
-- @tparam table b the block
-- @treturn string the joined parts
function M.text(b)
	return table.concat(b.parts)
end

-- the event stream

--- Milliseconds with no bytes out before a keepalive event.
M.KEEPALIVE_MS = 10000

--- Makes an event stream writer on res. It writes nothing yet.
-- Writes from several coroutines (the request and the keepalive timer)
-- go out in order, each whole.
-- @tparam lgml.server.http.Response res the response
-- @treturn Writer the writer
function M.writer(res)
	return setmetatable({ res = res, started = false, busy = false, waiters = {}, last = uv.now(),
		broken = false }, { __index = M.Writer })
end

--- An event stream writer.
-- The field `started` is true after the status line went out.
-- @type Writer
M.Writer = {}

local function lock(w)
	while w.busy do
		stream.await(function(done)
			w.waiters[#w.waiters + 1] = done
		end)
	end
	w.busy = true
end

local function unlock(w)
	w.busy = false
	local nxt = table.remove(w.waiters, 1)
	if nxt then
		nxt()
	end
end

--- Starts the stream when needed and writes text.
-- @tparam string|function text the events, or a function that returns
-- them, called in the lock after first
-- @tparam[opt] function first called once before the first write, in
-- the lock; it returns text to write first, such as the start events
-- @tparam[opt] table headers headers of the start
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string `"cancelled"`
function M.Writer:send(text, first, headers)
	lock(self)
	local ok = not self.broken and not self.res.s:gone()
	if ok and not self.started then
		self.started = true
		ok = sse.start(self.res, headers) ~= nil
		if ok and first then
			local head = first()
			if type(text) == "function" then
				text = text()
			end
			text = head .. text
		end
	end
	if type(text) == "function" then
		text = ok and text() or ""
	end
	if ok and not self.res.done and text ~= "" then
		ok = self.res:write(text) == true
	end
	self.last = uv.now()
	self.broken = self.broken or not ok
	unlock(self)
	if not ok then
		return nil, "cancelled"
	end
	return true
end

--- Ends the body and stops the keepalive timer.
-- @treturn boolean true when the body ended
function M.Writer:finish()
	self:stop()
	if not self.started then
		return false
	end
	lock(self)
	local ok = not self.broken and self.res:finish() == true
	unlock(self)
	return ok
end

--- Calls fn in its own coroutine after ms (default `KEEPALIVE_MS`)
-- with no writes, and again after each such time, until `stop`. fn can
-- call `send`. It does not run while a write is under way.
-- @tparam function fn the function
-- @tparam[opt] int ms the time
function M.Writer:keepalive(fn, ms)
	ms = ms or M.KEEPALIVE_MS
	local t = uv.new_timer()
	self.timer = t
	local step = math.max(1, ms // 4)
	t:start(step, step, function()
		-- A handler that raised leaves the timer; the closed connection stops it.
		if self.res.s:gone() then
			self:stop()
			return
		elseif self.busy or self.res.done or self.broken or uv.now() - self.last < ms then
			return
		end
		self.last = uv.now()
		coroutine.wrap(function()
			local ok, err = pcall(fn)
			if not ok then
				L:error("keepalive error", "err", err)
			end
		end)()
	end)
end

--- Stops the keepalive timer.
function M.Writer:stop()
	if self.timer then
		self.timer:stop()
		self.timer:close()
		self.timer = nil
	end
end

-- running

--- Runs a chat request on eng and counts it.
-- It notes the model and the usage for the access line, counts the
-- metrics and finishes the sink with reason `stop` when the engine did
-- not.
-- @tparam lgml.server.engine.Engine eng the engine
-- @tparam table chat the chat request
-- @tparam Blocks sink the sink
-- @tparam string model the served model id
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] ?int an HTTP status
function M.run(eng, chat, sink, model)
	log.note("model", model)
	local ok, msg, status = eng:chat(chat, sink)
	local f = sink.finished
	if ok and not f and not sink:cancelled() then
		sink:finish("stop")
		f = sink.finished
	end
	if f and f.timings then
		metrics.chat(model, f.timings)
	end
	metrics.request(model, "chat", ok == true)
	if sink:cancelled() then
		log.note("cancelled", true)
	end
	if f then
		local c, t = M.counts(f.usage), f.timings or {}
		log.note("prompt_tokens", c.prompt)
		log.note("cached_tokens", c.cached > 0 and c.cached or nil)
		log.note("completion_tokens", c.output)
		log.note("tok_s", t.predicted_per_second)
		log.note("images", (t.images or 0) > 0 and t.images or nil)
		log.note("finish", f.reason)
		log.note("tool_calls", #sink.calls > 0 and #sink.calls or nil)
		if L:on(log.DEBUG) then
			for _, b in ipairs(sink.list) do
				L:debug("reply block", "kind", b.kind, "name", b.name, "text", M.text(b))
			end
		end
	end
	return ok, msg, status
end

return M
