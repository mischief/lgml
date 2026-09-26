--- Routes requests to one of several engines by the request's model id.
-- A request with no model or an unknown one goes to the first engine
-- whose model has the mode of the request, as llama.cpp ignores the
-- model field. A router is an engine (see `lgml.server.engine`). The
-- modes of an engine come from `Engine:modes` when it has that method,
-- else from `ready().caps` (as `lgml.server.remote_engine` gives
-- them), else from its methods. The mode names are in `lgml.detect`.
-- @module lgml.server.router
local media = require("lgml.chat.media")
local detect = require("lgml.detect")

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
	local caps = e.ready and e:ready().caps
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

--- Picks the engine for a model id and a method.
-- It uses the modes of each engine, so it does not start a model.
-- @tparam ?string id the model id
-- @tparam string what `"chat"`, `"transcribe"`, `"speak"`, `"decide"` or
-- `"generate"`
-- @treturn ?lgml.server.engine.Engine the engine that serves id with
-- what, else the first engine with what, else nil
function R:pick(id, what)
	local first
	for _, e in ipairs(self.engines) do
		if can(e, what) then
			first = first or e
			for _, m in ipairs(e:models()) do
				if m.id == id then
					return e
				end
			end
		end
	end
	return first
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
-- else the first vision engine for an unknown model. A named model
-- without vision is an error that names the models with vision.
local function to_vision(self, e, id, what)
	local m = model_of(e, id)
	if m and (m.vision or has(m.capabilities, "multimodal")) then
		return e
	end
	local es, ids = vision_engines(self, what)
	if #es == 0 then
		return nil, "image input is not supported: no model has an mmproj", 400
	elseif m then
		return nil, ("image input is not supported by %s; models with vision: %s")
		    :format(m.id, table.concat(ids, ", ")), 400
	end
	return es[1]
end

--- Returns the properties of the first chat engine.
-- With no chat engine, it uses the first engine.
-- @treturn table the properties
function R:props()
	local e = self:pick(nil, "chat") or self.engines[1]
	return e:props()
end

--- Runs a chat completion on the engine for `req.model`.
-- A request with images and no model, or an unknown one, goes to the
-- first model with vision. A named model without vision is an error
-- that names the models with vision.
-- @tparam table req the chat request
-- @tparam lgml.server.engine.Sink sink the reply
-- @return the results of `Engine:chat`, or nil, a message and 400 when
-- no engine serves chat
function R:chat(req, sink)
	local e = self:pick(req.model, "chat")
	if not e then
		return nil, "no model serves chat", 400
	end
	if type(req.messages) == "table" and media.has_images(req.messages) then
		local err, status
		e, err, status = to_vision(self, e, req.model, "chat")
		if not e then
			return nil, err, status
		end
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
	local e = self:pick(req.model, "transcribe")
	if not e then
		return nil, "no model serves transcription", 400
	end
	return e:transcribe(req, audio, ctl)
end

--- Makes speech on the engine for `req.model`.
-- @tparam table req the request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:speak`, or nil, a message and 400 when
-- no engine serves speech
function R:speak(req, ctl)
	local e = self:pick(req.model, "speak")
	if not e then
		return nil, "no model serves speech", 400
	end
	return e:speak(req, ctl)
end

--- Answers typed questions on the engine for `req.model`.
-- A request for `"jevify-latest"`, no model or an unknown one goes to
-- the first engine that can decide. With images (`req.images`), such a
-- request goes to the first model with vision that can decide, and a
-- named model without vision is an error, as for chat.
-- @tparam table req the decision request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:decide`, or nil, a message and 400
-- when no engine can decide
function R:decide(req, ctl)
	local e = self:pick(req.model, "decide")
	if not e then
		return nil, "no model serves decisions", 400
	end
	if type(req.images) == "table" and #req.images > 0 then
		local err, status
		e, err, status = to_vision(self, e, req.model, "decide")
		if not e then
			return nil, err, status
		end
	end
	return e:decide(req, ctl)
end

--- Makes images on the engine for `req.model`.
-- @tparam table req the image request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:generate`, or nil, a message and 400
-- when no engine makes images
function R:generate(req, ctl)
	local e = self:pick(req.model, "generate")
	if not e then
		return nil, "no model serves images", 400
	end
	return e:generate(req, ctl)
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

--- Loads a model by id, or waits for its load under way.
-- @tparam string id the model id
-- @treturn[1] boolean true when the model is loaded
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int 404 for an unknown id, 400 for a model that does not
-- load on demand, 503 when the load fails
function R:load(id)
	local e = find(self, id)
	if not e then
		return nil, "unknown model " .. tostring(id), 404
	elseif not e.load then
		return nil, ("model %s does not load on demand"):format(id), 400
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
-- @treturn[2] int 404 for an unknown id, 400 for a model that does not
-- load on demand, 409 while it has requests in flight or loads
function R:unload(id)
	local e = find(self, id)
	if not e then
		return nil, "unknown model " .. tostring(id), 404
	elseif not e.unload then
		return nil, ("model %s does not load on demand"):format(id), 400
	end
	return e:unload()
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
