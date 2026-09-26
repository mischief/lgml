--- Routes requests to one of several engines by the request's model id.
-- A request with no model or an unknown one goes to the first engine
-- that can serve it, as llama.cpp ignores the model field. A router is
-- an engine (see `lgml.server.engine`). An engine with a `ready` method
-- gives its capabilities in `ready().caps`, as
-- `lgml.server.remote_engine` does.
-- @module lgml.server.router
local media = require("lgml.chat.media")

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

local function can(e, what)
	local caps = e.ready and e:ready().caps
	if caps then
		return caps[what] == true
	end
	return e[what] ~= nil
end

--- A router. It implements `lgml.server.engine.Engine`.
-- @type Router

--- Lists the models of all engines, in engine order.
-- @treturn {table,...} the model list
function R:models()
	local out = {}
	for _, e in ipairs(self.engines) do
		for _, m in ipairs(e:models()) do
			out[#out + 1] = m
		end
	end
	return out
end

--- Picks the engine for a model id and a method.
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

-- The chat engines with a vision model, and the ids of those models.
local function vision_engines(self)
	local es, ids = {}, {}
	for _, e in ipairs(self.engines) do
		if can(e, "chat") then
			for _, m in ipairs(e:models()) do
				if m.vision then
					es[#es + 1], ids[#ids + 1] = e, m.id
				end
			end
		end
	end
	return es, ids
end

-- The model of engine e with the id, or nil.
local function model_of(e, id)
	for _, m in ipairs(e:models()) do
		if m.id == id then
			return m
		end
	end
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
		local m = model_of(e, req.model)
		if not (m and m.vision) then
			local es, ids = vision_engines(self)
			if #es == 0 then
				return nil, "image input is not supported: no model has an mmproj", 400
			elseif m then
				return nil, ("image input is not supported by %s; models with vision: %s")
				    :format(m.id, table.concat(ids, ", ")), 400
			end
			e = es[1]
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
-- the first engine that can decide.
-- @tparam table req the decision request
-- @tparam lgml.server.engine.Control ctl the cancel control
-- @return the results of `Engine:decide`, or nil, a message and 400
-- when no engine can decide
function R:decide(req, ctl)
	local e = self:pick(req.model, "decide")
	if not e then
		return nil, "no model serves decisions", 400
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

--- Stops each engine that has a `stop` method.
function R:stop()
	for _, e in ipairs(self.engines) do
		if e.stop then
			e:stop()
		end
	end
end

return M
