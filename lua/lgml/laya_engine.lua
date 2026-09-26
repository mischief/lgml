--- System One engine for Laya decision models (`lgml.laya`).
-- It serves `/v1/systemone` only; the interface is in
-- `lgml.server.engine`. One request runs at a time, and all its
-- questions go through one graph. Laya calibrates with its own
-- temperatures, so the engine takes no `lgml.recipe`.
--
-- Answers map onto the Jev wire as for other engines: a noul gives
-- P(yes), the probability of the `true:` option; a choice or a score
-- gives the calibrated probabilities in option or level order, and
-- `lgml.systemone` derives `choice`, `score` and `confidence` from them.
-- With `lgml.raw`, each answer also has `lgml`: the raw marker
-- `logits`, the `temperature`, Laya's own `confidence` (1 minus the
-- normalized entropy) and `act_probability`.
-- @module lgml.laya_engine
local lgml = require("lgml")
local laya = require("lgml.laya")
local stream = require("lgml.server.stream")

local M = {}
local E = {}
E.__index = E

--- Loads a Laya model directory and makes an engine.
-- @tparam table opts the options:
--
-- - `model`: the model directory (see `lgml.laya.load`)
-- - `backend`: backend name (default: the best one)
-- - `threads`: CPU thread count (default: `lgml.threads()`)
-- - `id`: model id (default: the last part of the model path)
-- @treturn Engine the engine
-- @raise when the backend or the model does not load
function M.new(opts)
	local be = assert(lgml.backend(opts.backend, { threads = opts.threads }))
	local model = laya.load(be, opts.model)
	return setmetatable({
		be = be, model = model,
		id = opts.id or opts.model:gsub("/+$", ""):match("([^/]+)$"),
		busy = false, waiting = {},
	}, E)
end

--- A Laya engine. It implements `lgml.server.engine.Engine`.
-- @type Engine

--- Lists the one model of the engine, with capability `decide`.
-- @treturn {table,...} `{ { id, context_length = max_len, capabilities } }`
function E:models()
	return { { id = self.id, context_length = self.model.max_len, capabilities = { "decide" } } }
end

--- Returns the properties: `{ n_ctx = max_len, total_slots = 1, model_path }`.
-- @treturn table the properties
function E:props()
	return { n_ctx = self.model.max_len, total_slots = 1, model_path = self.model.dir }
end

--- Closes the model and the backend.
function E:close()
	self.model:close()
	self.be:close()
end

function E:acquire()
	while self.busy do
		stream.await(function(done)
			self.waiting[#self.waiting + 1] = done
		end)
	end
	self.busy = true
end

function E:release()
	self.busy = false
	local w = table.remove(self.waiting, 1)
	if w then
		w()
	end
end

local function round4(x)
	return math.floor(x * 1e4 + 0.5) / 1e4
end

function E:run(req, ctl)
	if ctl and ctl.cancelled and ctl.cancelled() then
		return nil, "cancelled", 499
	end
	local t0 = lgml.clock()
	local res, err = self.model:decide(req.state_json or req.state, req.questions)
	if not res then
		return nil, err, 400
	end
	local answers, tokens = {}, 0
	for i, q in ipairs(req.questions) do
		local r = res[i]
		tokens = tokens + r.n_tokens
		local a = q.type == "noul" and { yes = r.probabilities[2] } or { probabilities = r.probabilities }
		if req.raw then
			a.lgml = {
				logits = r.logits, temperature = round4(r.temperature),
				confidence = round4(r.confidence),
				act_probability = r.act_probability and round4(r.act_probability),
			}
		end
		answers[q.id] = a
	end
	return {
		model = self.id, answers = answers,
		usage = { input_tokens = tokens, output_tokens = 0 },
		timings = { decide_ms = (lgml.clock() - t0) * 1000 },
	}
end

--- Answers typed questions (see `lgml.server.engine.Engine.decide`).
-- A request with images or with `lgml.recipe` gives status 400, and so
-- does a question whose options do not fit in `head_max_len` tokens. A
-- Lua error gives status 500.
-- @tparam table req the request from `lgml.systemone.check`
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table `{ model, answers, usage, timings }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function E:decide(req, ctl)
	if req.images then
		return nil, "this model does not read images", 400
	elseif req.recipe then
		return nil, "lgml.recipe: Laya uses the temperatures of its rl_agent_config.json", 400
	end
	self:acquire()
	local ok, r, msg, status = pcall(self.run, self, req, ctl)
	self:release()
	if not ok then
		return nil, tostring(r), 500
	end
	return r, msg, status
end

return M
