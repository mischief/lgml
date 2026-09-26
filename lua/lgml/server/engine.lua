--- The engine interface that `lgml.server.openai` calls.
-- An engine is a table with methods. It has `models` and `props`. It
-- also has `chat`, `transcribe`, `speak`, `decide`, `generate`, or
-- several of them. `lgml.engine`, `lgml.stt_engine`,
-- `lgml.diffusion_engine`, `lgml.image.engine`,
-- `lgml.server.router`, `lgml.server.remote_engine` and
-- `lgml.server.fake_engine` are engines.
--
-- The `Engine`, `Sink` and `Control` sections describe the protocol.
-- No Lua code in this module implements them. The server calls each
-- engine method in the request coroutine, so a method can yield.
-- @module lgml.server.engine

local M = {}

--- The finish reasons that `Sink:finish` accepts.
-- The keys are `stop`, `length` and `tool_calls`. Each value is true.
M.reasons = { stop = true, length = true, tool_calls = true }

--- The message of a load or unload request to a server whose models do
-- not load on demand. Such a request gives status 400.
M.NO_LOAD = "this server does not load models on demand; start it with -C config"

--- Reads a LoRA list as llama-server does.
-- Each entry is `{ id, scale }`, with an id from 0 to n - 1. Adapters
-- that the list does not name get scale 0.
-- @tparam any list the list from a request
-- @tparam int n the number of adapters
-- @treturn[1] {number,...} the scales, by adapter index from 1
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.lora_scales(list, n)
	if type(list) ~= "table" then
		return nil, "lora must be an array of { id, scale }"
	end
	local scales = {}
	for i = 1, n do
		scales[i] = 0
	end
	for _, x in ipairs(list) do
		local id = type(x) == "table" and math.tointeger(x.id)
		if not id or id < 0 or id >= n or type(x.scale) ~= "number" then
			if n == 0 then
				return nil, "lora: the model has no adapters"
			end
			return nil, ("lora: need { id, scale } with id from 0 to %d"):format(n - 1)
		end
		scales[id + 1] = x.scale
	end
	return scales
end

--- The adapter list of the props with other scales.
-- @tparam {table,...} list `{ id, path, scale }` entries
-- @tparam[opt] {number,...} scales the scales by index from 1; nil keeps list
-- @treturn {table,...} a new list
function M.lora_view(list, scales)
	local out = {}
	for i, a in ipairs(list or {}) do
		out[i] = { id = a.id, path = a.path, scale = scales and scales[i] or a.scale }
	end
	return out
end

--- A request with the global scales when it has no `lora` field.
-- Front processes keep the scales that `POST /lora-adapters` set, and
-- send them with each chat and decide request.
-- @tparam table req the request
-- @tparam[opt] {number,...} scales the global scales; nil leaves req as is
-- @treturn table req, or a shallow copy with `lora`
function M.with_lora(req, scales)
	if not scales or req.lora ~= nil then
		return req
	end
	local t = {}
	for k, v in pairs(req) do
		t[k] = v
	end
	t.lora = {}
	for i, s in ipairs(scales) do
		t.lora[i] = { id = i - 1, scale = s }
	end
	return t
end

--- Checks that e has the methods of the interface.
-- @tparam Engine e the engine to check
-- @treturn Engine e
-- @raise when e has no `models` or `props` method, or when e has none
-- of `chat`, `transcribe`, `speak`, `decide` and `generate`.
function M.check(e)
	for _, name in ipairs({ "models", "props" }) do
		if type(e[name]) ~= "function" then
			error("engine has no method " .. name, 2)
		end
	end
	if type(e.chat) ~= "function" and type(e.transcribe) ~= "function" and
	    type(e.speak) ~= "function" and type(e.decide) ~= "function" and
	    type(e.generate) ~= "function" then
		error("engine has no method chat, transcribe, speak, decide or generate", 2)
	end
	return e
end

--- The cancel control for `Engine:transcribe`, `Engine:speak`,
-- `Engine:decide` and `Engine:generate`.
-- The server makes a new control for each request.
-- @table Control
-- @field cancelled a function with no arguments. It returns true when
-- the client is gone. Stop work then.

--- The methods of an engine.
-- @type Engine

--- Lists the models that the engine serves.
-- Each entry is `{ id = string, context_length = n }`. An entry can
-- also have `created` and `owned_by`; `/v1/models` shows them. It can
-- have `capabilities`, a list of mode names from `lgml.detect.MODES`;
-- without it, `/v1/models` finds them from the engine methods, and
-- `vision = true` adds "multimodal". An entry can have `status`, the
-- load state (`"unloaded"`, `"loading"`, `"loaded"` or `"unloading"`;
-- default `"loaded"`), and `bytes`, the memory it takes.
-- @treturn {table,...} the model list
-- @function models

--- Lists the modes of the engine's model without loading it.
-- The names are from `lgml.detect.MODES`. `lgml.server.router` uses
-- them to pick an engine. This method is optional.
-- @treturn {string,...} the mode names
-- @function modes

--- Returns the server properties for `/props`.
-- The table is `{ n_ctx = n, total_slots = n }`. It can also have
-- `model_path` and `modalities` (such as `{ vision = true }`).
-- @treturn table the properties
-- @function props

--- Runs one chat completion.
-- req is the decoded request body, checked by
-- `lgml.server.openai.check_chat`. The engine sends output through
-- sink. An engine may call `sink:finish` and return true when it stops
-- early. When it returns true and did not call `sink:finish`, the
-- server finishes the reply with reason `stop`. This method is
-- optional.
-- @tparam table req the request: `messages`, `model`, `tools`,
-- `max_tokens`, `max_completion_tokens`, `stream`, sampler settings,
-- `seed`, `stop`, `logprobs`, `top_logprobs`, `logit_bias`,
-- `reasoning_effort` and `chat_template_kwargs`. An engine that
-- cannot do a field of the request returns status 400. The server can
-- also set `reasoning_budget`, the most reasoning tokens before the
-- engine closes the reasoning block, and `count_only`: the engine then
-- renders the prompt and finishes with reason `stop` and the prompt
-- size in `usage.prompt_tokens`, with no prefill and no output.
-- @tparam Sink sink the reply
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] ?int an HTTP status
-- @function chat

--- Transcribes speech to text. This method is optional.
-- The result can also have `language_name`. Each segment is
-- `{ id, start, end, text }`, with times in seconds.
-- @tparam table req the request: `model`, `language` (code, name or
-- nil to detect), `prompt` and `response_format`
-- @tparam string audio the uploaded file bytes
-- @tparam Control ctl the cancel control
-- @treturn[1] table `{ text, language, duration, segments }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
-- @function transcribe

--- Makes speech from text. This method is optional.
-- The result can also have `sample_rate`.
-- @tparam table req the request, checked by
-- `lgml.server.openai.check_speech`: `model`, `input`, `voice`,
-- `speed` and `response_format`
-- @tparam Control ctl the cancel control
-- @treturn[1] table `{ audio = lgml audio }` or `{ wav = bytes }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
-- @function speak

--- Answers typed questions about a state, as Jev's System One API.
-- This method is optional. `lgml.systemone.check` makes req from the
-- request body; its fields are `model`, `state` (text), `questions`,
-- `raw`, `recipe` and `images` (`{ bytes, mime }` each, or nil). Each answer is `{ yes = P(yes) }` for a noul and
-- `{ probabilities }` in option or level order for a choice or a
-- score, unrounded. An answer can also have `lgml`, the raw log-scores,
-- when `req.raw` is true.
-- @tparam table req the request
-- @tparam Control ctl the cancel control
-- @treturn[1] table `{ model, answers = { [id] = answer }, usage =
-- { input_tokens, output_tokens }, timings }`; timings is optional and
-- its fields become `x-lgml-*` headers
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
-- @function decide

--- Makes images from a prompt, and from an input image for edits.
-- This method is optional. `lgml.server.openai.check_image` makes req.
-- Its fields are `model`, `prompt`, `negative_prompt`, `n`, `width` and
-- `height` (nil for an edit: the input size), `steps`, `seed` (nil:
-- random), `cfg_scale`, and for an edit `image` (the file bytes) and
-- `strength`. Image i uses seed + i - 1.
-- @tparam table req the request
-- @tparam Control ctl the cancel control
-- @treturn[1] table `{ images = { png, ... }, seeds, width, height,
-- steps }`; it can also have `encode_ms`, `sample_ms` and `decode_ms`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
-- @function generate

--- Loads a model now. This method is optional.
-- `lgml.server.router` takes a model id; `lgml.server.pool` models
-- take none.
-- @treturn[1] any a true value when the model is loaded
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
-- @function load

--- Unloads a model and frees its memory. This method is optional.
-- It takes the same arguments as `load`.
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
-- @function unload

--- Sets the global LoRA scales, as `POST /lora-adapters` of llama-server.
-- This method is optional. The list is as `lora_scales` reads it:
-- adapters that it does not name get scale 0. A chat or decide request
-- with no `lora` field uses the global scales; one with `lora` uses its
-- own. The `lora` field of `props` shows the global scales.
-- @tparam {table,...} list `{ id, scale }` entries
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
-- @function lora_adapters

--- The id of the model that serves a request. This method is optional.
-- `lgml.server.router` has it. Without it, the server uses the request
-- id when a model has it, else the first model id.
-- @tparam ?string id the request `model`
-- @tparam string what `"chat"`, `"transcribe"`, `"speak"`, `"decide"` or
-- `"generate"`
-- @tparam[opt] boolean images true when the request has images
-- @treturn ?string the model id
-- @function model_for

--- Stops the engine and its worker processes. This method is optional.
-- `lgml-server` calls it at exit.
-- @function stop

--- The reply object that `Engine:chat` writes to.
-- The text methods and the tool call methods return true, or nil and
-- `"cancelled"`. After `"cancelled"`, stop work.
-- @type Sink

--- Adds reasoning text.
-- @tparam string text the text; an empty string adds nothing
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string `"cancelled"`
-- @raise when the reply is finished
-- @function reasoning

--- Adds content text.
-- With logprobs, the text can be empty.
-- @tparam string text the text; an empty string with no logprobs adds
-- nothing
-- @tparam[opt] {table,...} logprobs one entry for each token of the
-- text, when the request has `logprobs`: `{ token, logprob, bytes,
-- top_logprobs }`, with bytes a list of byte values and top_logprobs a
-- list of `{ token, logprob, bytes }`. The sink drops them when the
-- request does not ask for them.
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string `"cancelled"`
-- @raise when the reply is finished
-- @function content

--- Opens a tool call and returns its index.
-- Indexes start at 0 and count up in call order.
-- @tparam string name the function name
-- @tparam[opt] string id the call id; when nil, the server makes one
-- @treturn[1] int the index of the call
-- @treturn[2] nil
-- @treturn[2] string `"cancelled"`
-- @raise when name is not a string, or when the reply is finished
-- @function tool_call

--- Adds argument text to a tool call.
-- @tparam int index the index that `tool_call` returned
-- @tparam string text JSON text; an empty string adds nothing
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string `"cancelled"`
-- @raise when no call has that index, or when the reply is finished
-- @function tool_arguments

--- Tells the prompt size before the prefill. This method is optional
-- for a sink; an engine calls it when the sink has it.
-- @tparam int n the prompt tokens
-- @tparam int cached the prompt tokens reused from the cache
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string `"cancelled"`
-- @function prompt

--- Ends the reply. Call it once.
-- The server changes reason `stop` to `tool_calls` when the reply has
-- tool calls.
-- @tparam string reason `"stop"`, `"length"` or `"tool_calls"`
-- @tparam[opt] table usage `{ prompt_tokens, completion_tokens, cached_tokens,
-- reasoning_tokens, stop_sequence }`; completion_tokens counts the
-- reasoning tokens too; stop_sequence is the stop string that ended
-- the reply, if one did
-- @tparam[opt] table timings llama.cpp timings: `cache_n`, `prompt_n`,
-- `prompt_ms`, `predicted_n`, `predicted_ms` and `predicted_per_second`
-- @raise when reason is not valid, or on a second call
-- @function finish

--- Tells if the client is gone. Stop work when it returns true.
-- @treturn boolean true when the client is gone
-- @function cancelled

return M
