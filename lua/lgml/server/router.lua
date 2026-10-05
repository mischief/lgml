--- Routes requests to one of several engines by the request's model id.
-- A request with no model (or an empty one) goes to the first engine
-- whose model has the mode of the request. An unknown id gives 404, and
-- a model without that mode gives 400. A router is an engine (see
-- `lgml.server.engine`). The
-- modes of an engine come from `Engine:modes` when it has that method,
-- else from `ready().caps` (as `lgml.server.remote_engine` gives
-- them), else from its methods. The mode names are in `lgml.detect`.
-- Each request logs its choice and why at debug level (see `lgml.log`).
-- @module lgml.server.router
local media = require("lgml.chat.media")
local detect = require("lgml.detect")
local log = require("lgml.log")
local engine_api = require("lgml.server.engine")
local systemone = require("lgml.systemone")

local L = log.new("router")

local M = {}
local R = {}
R.__index = R

--- Makes a router.
-- @tparam {lgml.server.engine.Engine,...} engines the engines. The first
-- chat engine gives `/props`.
-- @treturn Router the router
function M.new(engines)
	return setmetatable({ engines = engines }, R)
end

-- The modes of engine e for model m, as a list.
local function modes(e, m)
	if e.modes then
		return e:modes()
	elseif m and m.capabilities then
		return m.capabilities
	end
	local h = e.ready and e:ready()
	local caps = h and h.caps
	return detect.modes_of(caps or e, m and m.vision)
end

-- True when e has a mode that the method what serves.
local function can(e, what)
	for _, mode in ipairs(modes(e)) do
		if detect.METHOD[mode] == what then
			return true
		end
	end
	return false
end

local function has(list, v)
	for _, x in ipairs(list) do
		if x == v then
			return true
		end
	end
	return false
end

-- The models of e, each with capabilities.
local function entries(e)
	local out = {}
	for i, m in ipairs(e:models()) do
		local c = {}
		for k, v in pairs(m) do
			c[k] = v
		end
		c.capabilities = modes(e, m)
		out[i] = c
	end
	return out
end

--- A router. It implements `lgml.server.engine.Engine`.
-- @type Router

--- Lists the models of all engines, in engine order.
-- Each entry has `capabilities`, the list of its mode names.
-- @treturn {table,...} the model list
function R:models()
	local out = {}
	for _, e in ipairs(self.engines) do
		for _, m in ipairs(entries(e)) do
			out[#out + 1] = m
		end
	end
	return out
end

-- The engine with a model of this id, and that model.
local function find(self, id)
	for _, e in ipairs(self.engines) do
		for _, m in ipairs(e:models()) do
			if m.id == id then
				return e, m
			end
		end
	end
end

--- Picks the engine for a model id and a method.
-- It uses the modes of each engine, so it does not start a model.
-- @tparam ?string id the model id
-- @tparam string what `"chat"`, `"transcribe"`, `"speak"`, `"decide"`,
-- `"generate"` or `"embed"`
-- @treturn ?lgml.server.engine.Engine the engine that serves id with
-- what; with no id, the first engine with what; else nil
-- @treturn ?string the id of the picked model
-- @treturn ?string why: `"named"` (id matched), `"default"` (no id),
-- `"unknown"` (no model has id) or `"capability"` (the model of id
-- does not serve what); the last two come with a nil engine
function R:pick(id, what)
	-- Jevify clients name their default model.
	if id == "" or (what == "decide" and id == systemone.DEFAULT_MODEL) then
		id = nil
	end
	local first, first_id
	for _, e in ipairs(self.engines) do
		if can(e, what) then
			local list = e:models()
			if not first then
				first, first_id = e, list[1] and list[1].id
			end
			for _, m in ipairs(list) do
				if m.id == id then
					return e, id, "named"
				end
			end
		end
	end
	if id ~= nil then
		return nil, nil, find(self, id) and "capability" or "unknown"
	end
	if not first then
		return nil
	end
	return first, first_id, "default"
end

-- Words for a method in error messages.
local SERVES = {
	chat = "chat", transcribe = "transcriptions", speak = "speech",
	decide = "decisions", generate = "images", embed = "embeddings",
}

-- The ids of the models that serve what.
local function ids_for(self, what)
	local ids = {}
	for _, e in ipairs(self.engines) do
		if can(e, what) then
			for _, m in ipairs(e:models()) do
				ids[#ids + 1] = m.id
			end
		end
	end
	return table.concat(ids, ", ")
end

-- The engines of method what with a vision model, and the ids of
-- those models.
local function vision_engines(self, what)
	local es, ids = {}, {}
	for _, e in ipairs(self.engines) do
		if can(e, what) then
			for _, m in ipairs(entries(e)) do
				if m.vision or has(m.capabilities, "multimodal") then
					es[#es + 1], ids[#ids + 1] = e, m.id
				end
			end
		end
	end
	return es, ids
end

-- The model of engine e with the id, with capabilities, or nil.
local function model_of(e, id)
	for _, m in ipairs(entries(e)) do
		if m.id == id then
			return m
		end
	end
end

-- The engine for a request with images: e when its model has vision,
-- else the first vision engine when the request names no model. A named
-- model without vision is an error that names the models with vision.
local function to_vision(self, e, id, what)
	local m = model_of(e, id)
	if m and (m.vision or has(m.capabilities, "multimodal")) then
		return e, m.id
	end
	local es, ids = vision_engines(self, what)
	if #es == 0 then
		return nil, "image input is not supported: no model has an mmproj", 400
	elseif m then
		return nil, ("image input is not supported by %s; models with vision: %s")
		    :format(m.id, table.concat(ids, ", ")), 400
	end
	return es[1], ids[1]
end

-- The engine for a request: by id and method, then by vision when the
-- request has images. Returns the engine, the model id and why, or nil,
-- a message, a status and why.
local function resolve(self, id, what, images)
	local e, mid, why = self:pick(id, what)
	if not e then
		local word = SERVES[what] or what
		if why == "unknown" then
			return nil, ("model %s not found; models for %s: %s"):format(id, word,
			    ids_for(self, what)), 404, why
		elseif why == "capability" then
			return nil, ("model %s does not serve %s; models for %s: %s"):format(id, word,
			    word, ids_for(self, what)), 400, why
		end
		return nil, nil, 400, "none"
	end
	if images then
		local v, vid, status = to_vision(self, e, id, what)
		if not v then
			return nil, vid, status, "vision"
		end
		if v ~= e or vid ~= mid then
			e, mid, why = v, vid, "vision"
		end
	end
	return e, mid, why
end

--- The id of the model that serves a request.
-- It picks as the request methods do, so it does not start a model.
-- @tparam ?string id the model id of the request
-- @tparam string what the method, as for `pick`
-- @tparam[opt] boolean images true when the request has images
-- @treturn ?string the model id, or nil when no model serves the request
function R:model_for(id, what, images)
	local e, mid = resolve(self, id, what, images)
	return e and mid or nil
end

-- The engine for a request, as resolve gives it. It logs the choice and
-- notes the model for the access line. Returns the engine, or nil, a
-- message and a status.
local function route(self, id, what, images, none)
	local e, mid, why, how = resolve(self, id, what, images)
	if not e then
		-- On failure mid is the message, why the status.
		L:debug("route", "what", what, "requested", id, "why", how, "err", mid)
		return nil, mid or none, why
	end
	L:debug("route", "what", what, "requested", id, "model", mid, "why", why,
	    "images", images or nil)
	log.note("model", mid)
	return e
end

--- Returns the properties of the first chat engine.
-- With no chat engine, it uses the first engine.
-- @treturn table the properties
function R:props()
	local e = self:pick(nil, "chat") or self.engines[1]
	return e:props()
end

--- Sets the global LoRA scales of the first chat engine, whose adapters
-- `props` lists (see `lgml.server.engine.Engine.lora_adapters`).
-- @tparam {table,...} list `{ id, scale }` entries
-- @return the results of `Engine:lora_adapters`; an engine without it
-- has no adapters
function R:lora_adapters(list)
	local e = self:pick(nil, "chat") or self.engines[1]
	if e.lora_adapters then
		return e:lora_adapters(list)
	end
	local ok, err = engine_api.lora_scales(list, 0)
	if not ok then
		return nil, err, 400
	end
	return true
end

--- Runs a chat completion on the engine for `req.model`.
-- A request with images and no model goes to the
-- first model with vision. A named model without vision is an error
-- that names the models with vision.
-- @tparam table req the chat request
-- @tparam lgml.server.engine.Sink sink the reply
-- @return the results of `Engine:chat`, or nil, a message and 400 when
-- no engine serves chat
function R:chat(req, sink)
	local images = type(req.messages) == "table" and media.has_images(req.messages)
	local e, err, status = route(self, req.model, "chat", images, "no model serves chat")
	if not e then
		return nil, err, status
	end
	return e:chat(req, sink)
end

--- Transcribes on the engine for `req.model`.
-- @tparam table req the request
-- @tparam string audio the file bytes
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:transcribe`, or nil, a message and 400
-- when no engine serves transcription
function R:transcribe(req, audio, ctl)
	local e, err, status = route(self, req.model, "transcribe", false, "no model serves transcription")
	if not e then
		return nil, err, status
	end
	return e:transcribe(req, audio, ctl)
end

--- Makes speech on the engine for `req.model`.
-- @tparam table req the request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:speak`, or nil, a message and 400 when
-- no engine serves speech
function R:speak(req, ctl)
	local e, err, status = route(self, req.model, "speak", false, "no model serves speech")
	if not e then
		return nil, err, status
	end
	return e:speak(req, ctl)
end

--- Answers typed questions on the engine for `req.model`.
-- A request for `"jevify-latest"` or with no model goes to
-- the first engine that can decide. With images (`req.images`), such a
-- request goes to the first model with vision that can decide, and a
-- named model without vision is an error, as for chat.
-- @tparam table req the decision request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:decide`, or nil, a message and 400
-- when no engine can decide
function R:decide(req, ctl)
	local images = type(req.images) == "table" and #req.images > 0
	local e, err, status = route(self, req.model, "decide", images, "no model serves decisions")
	if not e then
		return nil, err, status
	end
	return e:decide(req, ctl)
end

--- Makes images on the engine for `req.model`.
-- @tparam table req the image request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:generate`, or nil, a message and 400
-- when no engine makes images
function R:generate(req, ctl)
	local e, err, status = route(self, req.model, "generate", false, "no model serves images")
	if not e then
		return nil, err, status
	end
	return e:generate(req, ctl)
end

--- Embeds text on the engine for `req.model`.
-- @tparam table req the embedding request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:embed`, or nil, a message and 400
-- when no engine serves embeddings
function R:embed(req, ctl)
	local e, err, status = route(self, req.model, "embed", false, "no model serves embeddings")
	if not e then
		return nil, err, status
	end
	return e:embed(req, ctl)
end

-- The engine of model id for a load or unload, or nil, a message and a
-- status. A router with no engine that loads on demand gives 400.
local function loader(self, id, what)
	local any = false
	for _, e in ipairs(self.engines) do
		any = any or e[what] ~= nil
	end
	if not any then
		return nil, engine_api.NO_LOAD, 400
	end
	local e = find(self, id)
	if not e then
		return nil, "unknown model " .. tostring(id), 404
	elseif not e[what] then
		return nil, ("model %s does not load on demand"):format(id), 400
	end
	return e
end

--- Loads a model by id, or waits for its load under way.
-- @tparam string id the model id
-- @treturn[1] boolean true when the model is loaded
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int 400 when the router or the model does not load on
-- demand, 404 for an unknown id, 503 when the load fails
function R:load(id)
	local e, err, code = loader(self, id, "load")
	if not e then
		return nil, err, code
	end
	local ok, msg, status = e:load()
	if not ok then
		return nil, msg, status
	end
	return true
end

--- Unloads a model by id and waits until its worker exits.
-- @tparam string id the model id
-- @treturn[1] boolean true, also when the model was not loaded
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int 400 when the router or the model does not load on
-- demand, 404 for an unknown id, 409 while it has requests in flight or loads
function R:unload(id)
	local e, err, code = loader(self, id, "unload")
	if not e then
		return nil, err, code
	end
	return e:unload()
end

--- Tells what the slots of each model do, in engine order (see
-- `lgml.server.engine.Engine.status`).
-- @treturn {table,...} one entry per model
function R:status()
	local out = {}
	for _, e in ipairs(self.engines) do
		local list = e.status and e:status() or engine_api.plain_status(e)
		table.move(list, 1, #list, #out + 1, out)
	end
	return out
end

--- Stops each engine that has a `stop` method.
function R:stop()
	for _, e in ipairs(self.engines) do
		if e.stop then
			e:stop()
		end
	end
end

return M
