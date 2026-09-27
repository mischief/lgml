--- Image engine for Z-Image: text-to-image and image-to-image.
-- It serves `lgml.server.openai`; the interface is in
-- `lgml.server.engine`. One slot: requests run one at a time. The
-- engine yields after each sampling step, so a worker reads cancel
-- messages while it runs.
-- At debug level each request logs its resolved size, steps and seed,
-- and each image its timings (see `lgml.log`).
-- @module lgml.image.engine
local lgml = require("lgml")
local budget = require("lgml.budget")
local detect = require("lgml.detect")
local pipeline = require("lgml.image.pipeline")
local stream = require("lgml.server.stream")
local log = require("lgml.log")

local L = log.new("image")

local M = {}
local E = {}
E.__index = E

--- Load the models and make an engine.
-- @tparam table opts the options:
--
-- - `model`: a directory with the DiT gguf (a name with "image"), the
--   Qwen3 gguf (a name with "qwen") and the VAE (ae.safetensors)
-- - `dit`, `llm`, `vae`: file paths in place of the directory scan
-- - `backend`: backend name (default: the best one)
-- - `text_backend`: backend name for the text encoder (default:
--   `backend`), such as "CPU" to keep its weights out of GPU memory
-- - `threads`: CPU thread count (default: `lgml.threads()`)
-- - `id`: model id (default: the DiT file name without .gguf)
-- @treturn Engine the engine
-- @raise when a model does not load
function M.new(opts)
	local dit, llm, vae = opts.dit, opts.llm, opts.vae
	if not (dit and llm and vae) then
		dit, llm, vae = detect.image_files(opts.model)
		assert(dit and llm and vae,
		    opts.model .. ": need a z_image gguf, a Qwen3 gguf and ae.safetensors")
	end
	local p = pipeline.load({
		dit = dit, llm = llm, vae = vae, backend = opts.backend, text_backend = opts.text_backend,
		threads = opts.threads,
	})
	return setmetatable({
		p = p, id = opts.id or dit:match("([^/]+)%.gguf$"), busy = false, waiting = {},
	}, E)
end

--- An image engine. It implements `lgml.server.engine.Engine`.
-- @type Engine

--- Lists the one model of the engine.
-- @treturn {table,...} `{ { id, context_length = 0 } }`
function E:models()
	return { { id = self.id, context_length = 0 } }
end

--- Returns the properties: `{ n_ctx = 0, total_slots = 1 }`.
-- @treturn table the properties
function E:props()
	return { n_ctx = 0, total_slots = 1 }
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

-- The largest multiple of m at most v, within the pipeline limits.
local function fit(v, m)
	return math.max(m, math.min(pipeline.MAX_SIDE, v // m * m))
end

function E:run(req, ctl)
	local image
	if req.image then
		local err
		image, err = lgml.image(req.image)
		if not image then
			return nil, "image: " .. tostring(err), 400
		end
	end
	local w, h = req.width, req.height
	if not w then
		w = image and fit(image.width, pipeline.MULTIPLE) or 1024
		h = image and fit(image.height, pipeline.MULTIPLE) or 1024
	end
	local ok, err = pipeline.check_size(w, h)
	if not ok then
		return nil, err, 400
	end
	local steps = req.steps or pipeline.STEPS
	local seed = req.seed or math.random(0, (1 << 31) - 1)
	L:debug("image request", "size", ("%dx%d"):format(w, h), "steps", steps, "seed", seed, "n", req.n or 1,
	    "cfg", req.cfg_scale, "strength", req.strength, "edit", image ~= nil, "prompt", req.prompt,
	    "negative_prompt", req.negative_prompt)
	local out = { images = {}, seeds = {}, width = w, height = h, steps = 0, sample_ms = 0,
	    decode_ms = 0, encode_ms = 0 }
	local stopped = false
	for i = 1, req.n or 1 do
		local img, tm = self.p:generate({
			prompt = req.prompt, negative_prompt = req.negative_prompt, width = w, height = h,
			steps = steps, seed = seed + i - 1, cfg = req.cfg_scale, image = image,
			strength = req.strength,
			step = function()
				if coroutine.isyieldable() then
					stream.sleep(0)
				end
				stopped = ctl and ctl.cancelled and ctl.cancelled() or false
				return not stopped
			end,
		})
		if not img then
			if image then
				image:close()
			end
			-- 499 only for a cancel; any other failure is a server error.
			return nil, tostring(tm or "image generation failed"), stopped and 499 or 500
		end
		L:debug("image", "index", i, "seed", seed + i - 1, "steps", tm.steps, "encode_ms", tm.encode_ms,
		    "sample_ms", tm.sample_ms, "decode_ms", tm.decode_ms)
		out.images[i] = img:png()
		out.seeds[i] = seed + i - 1
		img:close()
		out.steps = out.steps + tm.steps
		for _, k in ipairs({ "encode_ms", "sample_ms", "decode_ms" }) do
			out[k] = out[k] + tm[k]
		end
	end
	if image then
		image:close()
	end
	return out
end

--- Makes images (see `lgml.server.engine.Engine.generate`).
-- A Lua error returns status 500.
-- @tparam table req the request, checked by
-- `lgml.server.openai.check_image`
-- @tparam[opt] lgml.server.engine.Control ctl the cancel control
-- @treturn[1] table `{ images, seeds, width, height, steps, encode_ms,
-- sample_ms, decode_ms }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
function E:generate(req, ctl)
	self:acquire()
	local ok, r, msg, status = pcall(self.run, self, req, ctl)
	self:release()
	if not ok then
		return nil, tostring(r), budget.status(r, 500)
	end
	return r, msg, status
end

return M
