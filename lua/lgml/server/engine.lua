--- The engine interface that `lgml.server.openai` calls.
-- An engine is a table with methods. It has `models` and `props`. It
-- also has `chat`, `transcribe`, `speak`, `decide`, or several of them.
-- `lgml.engine`, `lgml.stt_engine`, `lgml.diffusion_engine`,
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

--- Checks that e has the methods of the interface.
-- @tparam Engine e the engine to check
-- @treturn Engine e
-- @raise when e has no `models` or `props` method, or when e has none
-- of `chat`, `transcribe`, `speak` and `decide`.
function M.check(e)
	for _, name in ipairs({ "models", "props" }) do
		if type(e[name]) ~= "function" then
			error("engine has no method " .. name, 2)
		end
	end
	if type(e.chat) ~= "function" and type(e.transcribe) ~= "function" and
	    type(e.speak) ~= "function" and type(e.decide) ~= "function" then
		error("engine has no method chat, transcribe, speak or decide", 2)
	end
	return e
end

--- The cancel control for `Engine:transcribe`, `Engine:speak` and
-- `Engine:decide`.
-- The server makes a new control for each request.
-- @table Control
-- @field cancelled a function with no arguments. It returns true when
-- the client is gone. Stop work then.

--- The methods of an engine.
-- @type Engine

--- Lists the models that the engine serves.
-- Each entry is `{ id = string, context_length = n }`. An entry can
-- also have `created` and `owned_by`; `/v1/models` shows them.
-- @treturn {table,...} the model list
-- @function models

--- Returns the server properties for `/props`.
-- The table is `{ n_ctx = n, total_slots = n }`. It can also have
-- `model_path`.
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
-- `seed`, `reasoning_effort` and `chat_template_kwargs`
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
-- `raw` and `recipe`. Each answer is `{ yes = P(yes) }` for a noul and
-- `{ probabilities }` in option or level order for a choice or a
-- score, unrounded. An answer can also have `lgml`, the raw log-scores,
-- when `req.raw` is true.
-- @tparam table req the request
-- @tparam Control ctl the cancel control
-- @treturn[1] table `{ model, answers = { [id] = answer }, usage =
-- { input_tokens, output_tokens } }`
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] int an HTTP status
-- @function decide

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
-- @tparam string text the text; an empty string adds nothing
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

--- Ends the reply. Call it once.
-- The server changes reason `stop` to `tool_calls` when the reply has
-- tool calls.
-- @tparam string reason `"stop"`, `"length"` or `"tool_calls"`
-- @tparam[opt] table usage `{ prompt_tokens, completion_tokens, cached_tokens }`
-- @tparam[opt] table timings llama.cpp timings: `cache_n`, `prompt_n`,
-- `prompt_ms`, `predicted_n`, `predicted_ms` and `predicted_per_second`
-- @raise when reason is not valid, or on a second call
-- @function finish

--- Tells if the client is gone. Stop work when it returns true.
-- @treturn boolean true when the client is gone
-- @function cancelled

return M
