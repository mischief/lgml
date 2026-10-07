--- The OpenAPI 3.1 document of the HTTP API.
-- `document` builds it at run time. The paths come from the routes of
-- a server. The request fields come from the tables that the request
-- checks use: `lgml.server.openai.CHAT_FIELDS`, `IMAGE_FIELDS`,
-- `EMBEDDING_FIELDS`, `SPEECH_FORMATS`, `TRANSCRIPTION_FORMATS`,
-- `MESSAGE_ROLES` and `lgml.systemone.fields`. Where a check is a function, its text goes
-- into `description`. The descriptions are short; `docs/api.md` has
-- the full reference.
-- @module lgml.server.openapi
local json = require("lgml.json")
local openai = require("lgml.server.openai")
local systemone = require("lgml.systemone")
local detect = require("lgml.detect")
local anthropic = require("lgml.server.anthropic")
local responses = require("lgml.server.responses")

local M = {}

--- The OpenAPI version of the document.
M.OPENAPI = "3.1.0"

-- schema helpers

local function ref(name)
	return { ["$ref"] = "#/components/schemas/" .. name }
end

local function arr(items, desc)
	return { type = "array", items = items, description = desc }
end

local function str(desc)
	return { type = "string", description = desc }
end

local function num(desc)
	return { type = "number", description = desc }
end

local function int(desc)
	return { type = "integer", description = desc }
end

-- An object schema. required is a list of names, or nil.
local function obj(props, required, desc)
	local o = { type = "object", properties = json.object(props or {}), description = desc }
	if required and #required > 0 then
		o.required = json.array(required)
	end
	return o
end

-- The keys of a set, sorted, as a JSON array.
local function keys(set)
	local t = json.array()
	for k in pairs(set) do
		t[#t + 1] = k
	end
	table.sort(t)
	return t
end

-- Marks s as an lgml or llama.cpp extension of an OpenAI shape.
local function ext(s)
	s["x-lgml-extension"] = true
	return s
end

local JSON_TYPES = { boolean = "boolean", number = "number", string = "string" }

--- Makes the schema of one field table entry.
-- The entry is `{ name, Lua type, check, text }`; the Lua type can be
-- several names joined by `|`. A "table" whose check accepts an array
-- and not an object is an array; other tables are objects. text
-- becomes the description.
-- @tparam table c the entry
-- @treturn table the schema
function M.field_schema(c)
	local types = {}
	for t in c[2]:gmatch("[^|]+") do
		if t == "table" then
			local pred = c[3]
			types[#types + 1] = (pred and pred(json.array()) and not pred({})) and "array" or "object"
		else
			types[#types + 1] = JSON_TYPES[t] or t
		end
	end
	local s = { type = #types == 1 and types[1] or json.array(types) }
	if c[4] then
		s.description = "Must be " .. c[4] .. "."
	end
	return s
end

-- Adds the entries of a field table to props. A field that props has
-- keeps its schema; the check text goes after its description. note
-- gives a description for fields whose entry has no text. An entry
-- with a status or a label other than OpenAI marks the schema.
local function add_fields(props, tbl, note, skip)
	for _, c in ipairs(tbl) do
		if not (skip and skip[c[1]]) then
			local s = M.field_schema(c)
			local have = props[c[1]]
			if have then
				if s.description then
					have.description = have.description and have.description .. " " .. s.description
					    or s.description
				end
				s = have
			end
			if not s.description and note[c[1]] then
				s.description = note[c[1]]
			end
			s["x-lgml-status"] = c.status
			if c.label and c.label ~= "OpenAI" and c.label ~= "Anthropic" then
				ext(s)
			end
			props[c[1]] = s
		end
	end
end

-- chat

local CHAT_NOTES = {
	model = "Model id. A missing or unknown id goes to the first model that serves chat.",
	stream = "Stream the reply as server-sent events.",
	stream_options = "`include_usage`: send a last chunk with usage.",
	temperature = "Sampler temperature.",
	top_p = "Nucleus sampling threshold.",
}

-- Schemas of chat fields with more shape than their CHAT_FIELDS entry
-- gives.
local function chat_extras()
	return {
		messages = arr(ref("ChatMessage"), "A non-empty array of messages."),
		grammar = str("GBNF grammar for the answer (llama.cpp). Not with a JSON schema."),
		response_format = obj({
			type = { type = "string", enum = json.array({ "text", "json_object", "json_schema" }) },
			json_schema = obj({ schema = { type = "object" } }),
			schema = { type = "object" },
		}, { "type" }, "Constrains the answer to JSON; json_object may carry `schema` (llama.cpp)."),
		json_schema = { type = "object", description = "JSON schema for the answer (llama.cpp)." },
		seed = int("Sampler seed."),
		top_k = int("Top-k sampling."),
		reasoning_effort = str('"none" turns thinking off. minimal and low give a reasoning budget of 1024 tokens, medium 2048, high 4096, xhigh 8192 and max 16384, at most the server budget.'),
		tools = arr(ref("Tool")),
		chat_template_kwargs = ext(obj({
			enable_thinking = { type = "boolean" },
			reasoning_effort = { type = "string" },
			preserve_thinking = { type = "boolean" },
		}, nil, "Values for the chat template (llama.cpp).")),
		lora = ext(arr(obj({ id = int(), scale = num() }, { "id", "scale" }),
		    "Adapter scales for this request, ids from 0 (llama.cpp).")),
	}
end

local function chat_schemas(c)
	local props = chat_extras()
	add_fields(props, openai.CHAT_FIELDS, CHAT_NOTES)
	c.ChatRequest = obj(props, { "messages" })
	local part = {
		oneOf = json.array({
			obj({ type = { const = "text" }, text = str() }, { "type", "text" }),
			obj({
				type = { const = "image_url" },
				image_url = { oneOf = json.array({ str(), obj({
					url = str(), detail = { type = "string", enum = json.array({ "auto", "low", "high" }),
					    description = "No effect." },
				}, { "url" }) }) },
			}, { "type", "image_url" }, "A data URL of a PNG or JPEG image. Needs a model with vision."),
		}),
	}
	c.ChatMessage = obj({
		role = { type = "string", enum = keys(openai.MESSAGE_ROLES) },
		content = { oneOf = json.array({ str(), { type = "null" }, arr(part) }) },
		reasoning_content = ext(str("Reasoning text of an earlier assistant turn.")),
		tool_calls = arr(ref("ToolCall")),
		tool_call_id = str(),
		name = str("No effect."),
		annotations = arr({ type = "object" }, "No effect."),
		refusal = { type = { "string", "null" }, description = "Only null or empty." },
	}, { "role" })
	c.ChatMessage.additionalProperties = false
	c.Tool = obj({
		type = { const = "function" },
		["function"] = obj({
			name = str(), description = str(), parameters = { type = "object" },
			strict = { type = "boolean" },
		}, { "name" }),
	}, { "type", "function" })
	c.ToolCall = obj({
		id = str(), type = { const = "function" },
		["function"] = obj({
			name = str(),
			arguments = { oneOf = json.array({ str(), { type = "object" } }) },
		}, { "name", "arguments" }),
	}, { "function" })
	c.Usage = obj({
		prompt_tokens = int(), completion_tokens = int(), total_tokens = int(),
		prompt_tokens_details = obj({ cached_tokens = int(), audio_tokens = int() }),
		completion_tokens_details = obj({
			reasoning_tokens = int(), audio_tokens = int(),
			accepted_prediction_tokens = int(), rejected_prediction_tokens = int(),
		}),
	})
	c.Timings = ext(obj({
		cache_n = int(), restored_n = int(), prompt_n = int(), prompt_ms = num(),
		prompt_per_token_ms = num(), prompt_per_second = num(), predicted_n = int(),
		predicted_ms = num(), predicted_per_token_ms = num(), predicted_per_second = num(),
	}, nil, "llama.cpp timings."))
	local finish = { type = "string", enum = json.array({ "stop", "length", "tool_calls" }) }
	local nullable = { type = { "string", "null" } }
	local logprob = obj({
		token = str(), logprob = num(), bytes = { type = { "array", "null" }, items = int() },
	})
	logprob.properties.top_logprobs = arr(obj({
		token = str(), logprob = num(), bytes = { type = { "array", "null" }, items = int() },
	}))
	c.ChatLogprobs = obj({ content = arr(logprob), refusal = { type = "null" } }, nil,
	    "Log probabilities of the content tokens.")
	local logprobs = { anyOf = json.array({ ref("ChatLogprobs"), { type = "null" } }) }
	c.ChatCompletion = obj({
		id = str(), object = { const = "chat.completion" }, created = int(), model = str(),
		system_fingerprint = { type = "null" },
		choices = arr(obj({
			index = int(), finish_reason = finish, logprobs = logprobs,
			message = obj({
				role = { const = "assistant" }, content = nullable, refusal = { type = "null" },
				reasoning_content = ext(str()), tool_calls = arr(ref("ToolCall")),
			}),
		})),
		usage = ref("Usage"), timings = ref("Timings"),
	})
	c.ChatCompletionChunk = obj({
		id = str(), object = { const = "chat.completion.chunk" }, created = int(), model = str(),
		system_fingerprint = { type = "null" },
		choices = arr(obj({
			index = int(), finish_reason = { type = { "string", "null" } }, logprobs = logprobs,
			delta = obj({
				role = str(), content = str(), refusal = { type = "null" },
				reasoning_content = ext(str()), tool_calls = arr({ type = "object" }),
			}),
		})),
		usage = { anyOf = json.array({ ref("Usage"), { type = "null" } }) }, timings = ref("Timings"),
	}, nil, "One `data:` event. The stream ends with `data: [DONE]`.")
end

-- Anthropic Messages

local MESSAGES_NOTES = {
	model = "Model id. A missing or unknown id goes to the first model that serves chat.",
	system = "A string or an array of text blocks.",
	stream = "Stream the reply as named server-sent events.",
}

local function messages_schemas(c)
	local props = {
		messages = arr(ref("AnthropicMessage"), "Messages; consecutive messages of one role merge."),
		tools = arr(ref("AnthropicTool")),
		tool_choice = obj({
			type = { type = "string", enum = json.array({ "auto", "any", "tool", "none" }) },
			name = str(), disable_parallel_tool_use = { type = "boolean" },
		}, { "type" }),
		thinking = obj({
			type = { type = "string", enum = json.array({ "enabled", "adaptive", "disabled" }) },
			budget_tokens = int("At least 1024; the engine ends the reasoning after this many tokens."),
			display = { type = "string", enum = json.array({ "summarized", "omitted" }) },
		}, { "type" }),
		output_config = obj({
			effort = str("`none` turns thinking off; other values have no effect."),
			format = obj({ type = { const = "json_schema" }, schema = { type = "object" } }, { "type", "schema" }),
		}),
		max_tokens = int("The most output tokens, reasoning too. 0 runs the prompt only."),
		top_k = int("Top-k sampling."),
	}
	add_fields(props, anthropic.FIELDS, MESSAGES_NOTES)
	c.MessagesRequest = obj(props, { "model", "messages", "max_tokens" })
	c.CountTokensRequest = obj(props, { "model", "messages" },
	    "The body of /v1/messages; the generation fields have no effect.")
	local block = { type = "object", description = "A content block: text, image (base64 source), " ..
	    "document (text source), search_result, tool_use, tool_result, thinking or redacted_thinking.",
	    properties = { type = str() }, required = json.array({ "type" }) }
	c.AnthropicMessage = obj({
		role = { type = "string", enum = json.array({ "user", "assistant", "system" }) },
		content = { oneOf = json.array({ str(), arr(block) }) },
	}, { "role", "content" })
	c.AnthropicTool = obj({
		name = str(), description = str(), input_schema = { type = "object" },
		type = { const = "custom" }, strict = { type = "boolean" }, cache_control = { type = "object" },
	}, { "name", "input_schema" })
	c.AnthropicUsage = obj({
		input_tokens = int("Prompt tokens not read from the cache."),
		output_tokens = int(), cache_creation_input_tokens = int(),
		cache_read_input_tokens = int("Prompt tokens reused from the cache."),
		cache_creation = obj({ ephemeral_5m_input_tokens = int(), ephemeral_1h_input_tokens = int() }),
		output_tokens_details = obj({ thinking_tokens = int() }),
		service_tier = { const = "standard" },
	})
	c.Message = obj({
		id = str("`msg_` and 24 base62 characters."), type = { const = "message" },
		role = { const = "assistant" }, model = str(),
		content = arr({ type = "object", description = "text, thinking or tool_use." }),
		stop_reason = { type = "string", enum = json.array({ "end_turn", "max_tokens", "stop_sequence",
			"tool_use", "model_context_window_exceeded" }) },
		stop_sequence = { type = { "string", "null" } }, stop_details = { type = "null" },
		usage = ref("AnthropicUsage"),
	})
	c.MessageStreamEvent = obj({
		type = { type = "string", enum = json.array({ "message_start", "content_block_start",
			"content_block_delta", "content_block_stop", "message_delta", "message_stop", "ping", "error" }) },
		message = ref("Message"), index = int(), content_block = { type = "object" },
		delta = { type = "object" }, usage = ref("AnthropicUsage"),
	}, { "type" }, "The data of one event; the `event:` line has the same name. No `[DONE]`.")
	c.AnthropicError = obj({
		type = { const = "error" },
		error = obj({ type = str(), message = str() }, { "type", "message" }),
		request_id = str(),
	}, { "type", "error" })
end

-- OpenAI Responses

local RESPONSES_NOTES = {
	model = "Model id. A missing or unknown id goes to the first model that serves chat.",
	stream = "Stream the reply as named server-sent events with sequence numbers.",
}

local function responses_schemas(c)
	local item = { type = "object", description = "A message (role and content), function_call, " ..
	    "function_call_output, reasoning or item_reference item." }
	local props = {
		input = { oneOf = json.array({ str(), arr(item) }) },
		tools = arr(ref("ResponsesTool")),
		tool_choice = { oneOf = json.array({ { type = "string", enum = json.array({ "none", "auto", "required" }) },
			obj({ type = { const = "function" }, name = str() }, { "type", "name" }),
			obj({ type = { const = "allowed_tools" }, mode = str(), tools = arr({ type = "object" }) },
			    { "type", "mode", "tools" }) }) },
		reasoning = obj({ effort = str("`none` turns thinking off."), summary = str("No effect.") }),
		text = obj({ format = obj({ type = { type = "string", enum = json.array({ "text", "json_object",
			"json_schema" }) }, name = str(), schema = { type = "object" }, strict = { type = "boolean" } },
			{ "type" }), verbosity = { const = "medium" } }),
	}
	add_fields(props, responses.FIELDS, RESPONSES_NOTES)
	c.ResponsesRequest = obj(props, { "input" })
	c.ResponsesTool = obj({
		type = { const = "function" }, name = str(), description = str(),
		parameters = { type = "object" }, strict = { type = "boolean" },
	}, { "type", "name" }, "A flat function tool.")
	c.ResponsesUsage = obj({
		input_tokens = int("The whole prompt."),
		input_tokens_details = obj({ cached_tokens = int(), cache_write_tokens = int() }),
		output_tokens = int(), output_tokens_details = obj({ reasoning_tokens = int() }),
		total_tokens = int(),
	})
	c.Response = obj({
		id = str("`resp_` and 24 base62 characters."), object = { const = "response" },
		created_at = int(), completed_at = { type = { "integer", "null" } },
		status = { type = "string", enum = json.array({ "completed", "incomplete", "in_progress", "failed" }) },
		error = { type = { "object", "null" } }, incomplete_details = { type = { "object", "null" } },
		model = str(), instructions = { type = { "string", "null" } },
		output = arr({ type = "object", description = "reasoning, message or function_call items." }),
		usage = { anyOf = json.array({ ref("ResponsesUsage"), { type = "null" } }) },
		previous_response_id = { type = { "string", "null" } }, store = { type = "boolean" },
	}, nil, "It also echoes the request settings.")
	c.ResponseStreamEvent = obj({
		type = str("The event type; the `event:` line has the same name."),
		sequence_number = int("From 0, one more for each event."),
		response = ref("Response"), output_index = int(), item = { type = "object" }, item_id = str(),
		content_index = int(), summary_index = int(), delta = str(), part = { type = "object" },
	}, { "type", "sequence_number" }, "The data of one event. No `[DONE]`.")
end

-- images

local IMAGE_NOTES = {
	model = "Model id.",
	negative_prompt = "Text for the unconditional pass.",
}

local function image_props(edit)
	local props = {
		prompt = str("1 to 16384 bytes."),
		size = str(("`auto` or WxH; sides are multiples of %d from %d to %d."):format(
		    openai.IMAGE_MULTIPLE, openai.IMAGE_MULTIPLE, openai.IMAGE_MAX_SIDE)),
		response_format = str("Only `b64_json`."),
		output_format = str("Only `png`."),
	}
	-- check_image takes strength for edits only.
	add_fields(props, openai.IMAGE_FIELDS, IMAGE_NOTES, not edit and { strength = true } or nil)
	if edit then
		props.image = { type = "string", contentMediaType = "application/octet-stream",
		    description = "The input image, PNG or JPEG. `image[]` also works." }
	end
	return props
end

local function image_schemas(c)
	c.ImageRequest = obj(image_props(false), { "prompt" })
	c.ImageEditRequest = obj(image_props(true), { "image", "prompt" })
	c.ImagesResponse = obj({
		created = int(), output_format = { const = "png" }, size = str(),
		data = arr(obj({ b64_json = str(), seed = ext(int()) })),
	})
end

-- audio

local function audio_schemas(c)
	c.SpeechRequest = obj({
		input = str("1 to 16384 bytes."),
		model = str("Model id."),
		voice = str("Voice name of the model."),
		speed = num("0.25 to 4. Some models take only 1."),
		response_format = { type = "string", enum = keys(openai.SPEECH_FORMATS) },
	}, { "input" }, "Models can read more fields, such as seed or language; see docs/api.md.")
	c.TranscriptionRequest = obj({
		file = { type = "string", contentMediaType = "audio/wav", description = "WAV audio." },
		model = str("Model id."),
		language = str("Code or name; missing: detect."),
		prompt = str("Text that the first 30 s window sees as its previous text."),
		response_format = { type = "string", enum = keys(openai.TRANSCRIPTION_FORMATS) },
	}, { "file" })
	c.Transcription = obj({ text = str() })
	c.TranscriptionVerbose = obj({
		task = { const = "transcribe" }, language = str(), duration = num(), text = str(),
		segments = arr(obj({ id = int(), start = num(), ["end"] = num(), text = str() })),
	})
end

-- embeddings

local EMBEDDING_NOTES = {
	model = "Model id. A missing or unknown id goes to the first model that serves embeddings.",
	user = "Accepted and ignored.",
}

local function embedding_schemas(c)
	local ids = arr({ type = "integer", minimum = 0 })
	local props = {
		input = { oneOf = json.array({
			str("One text."), arr(str(), "Texts."), ids, arr(ids, "Token id lists."),
		}), description = "Texts or token id lists. A text gets the end token of the model; " ..
		    "token ids run as they are." },
		dimensions = int("Keep the first values of each vector and scale it to length 1 again."),
		encoding_format = { type = "string", enum = json.array({ "base64", "float" }),
		    description = "base64: little-endian float32 values." },
	}
	add_fields(props, openai.EMBEDDING_FIELDS, EMBEDDING_NOTES)
	c.EmbeddingRequest = obj(props, { "input" })
	-- A field not in EMBEDDING_FIELDS gives 400.
	c.EmbeddingRequest.additionalProperties = false
	c.EmbeddingsResponse = obj({
		object = { const = "list" }, model = str(),
		data = arr(obj({
			object = { const = "embedding" }, index = int(),
			embedding = { oneOf = json.array({ arr(num()), str("base64 float32 values.") }) },
		})),
		usage = obj({ prompt_tokens = int(), total_tokens = int() }),
	})
end

-- System One

local content = { type = { "string", "object", "array" } }

local SYSTEMONE_NOTES = {
	state = { type = { "string", "object", "array" }, description = "The state. JSON becomes indented text." },
	model = str("Model id. Missing or unknown: the first model that decides."),
	questions = { type = "object", additionalProperties = ref("SystemOneQuestion"),
	    description = "Questions by id, answered in this order." },
	lgml = ext(ref("SystemOneExtension")),
	attachments = ext(arr({ oneOf = json.array({
		str("A data URL of a PNG or JPEG image."),
		obj({ type = { const = "image_url" }, image_url = obj({ url = str() }, { "url" }) },
		    { "type", "image_url" }),
	}) }, "Images for the state. Needs a model with capability multimodal.")),
}

local QUESTION_NOTES = {
	type = { type = "string", enum = keys(systemone.fields.types) },
	instructions = { type = { "string", "object", "array", "null" } },
	criteria = { description = ("noul: `{true, false}` texts or null. choice: object of %d to %d options. " ..
	    "score: array of %d to %d levels."):format(2, systemone.MAX_CHOICE_OPTIONS,
	    systemone.MIN_SCORE_LEVELS, systemone.MAX_SCORE_LEVELS) },
}

local LGML_NOTES = {
	raw = { type = "boolean", description = "Add the raw log-scores to each answer as `lgml`." },
	recipe = { type = "object", description = "Recipe fields over the server recipe." },
}

-- An object schema with exactly the fields of set; notes gives the
-- schema of each, and a field with no note gets a pointer to the docs.
local function closed(set, notes, required, desc)
	local props = {}
	for k in pairs(set) do
		props[k] = notes[k] or { description = "See docs/api.md." }
	end
	local o = obj(props, required, desc)
	o.additionalProperties = false
	return o
end

local function systemone_schemas(c)
	c.SystemOneRequest = closed(systemone.fields.request, SYSTEMONE_NOTES, { "state", "questions" })
	c.SystemOneQuestion = closed(systemone.fields.question, QUESTION_NOTES, { "type" })
	c.SystemOneExtension = closed(systemone.fields.lgml, LGML_NOTES, nil, "lgml extension.")
	local probs = { type = "object", additionalProperties = { type = "number" } }
	c.SystemOneResponse = obj({
		model = str(),
		answers = { type = "object", additionalProperties = obj({
			type = { type = "string", enum = keys(systemone.fields.types) },
			noul = num("P(yes), to 0.01."), choice = str(), score = num(), confidence = num(),
			legend = { type = "object", additionalProperties = content },
			probabilities = probs, lgml = ext({ type = "object" }),
		}, { "type" }) },
		usage = obj({ input_tokens = int(), output_tokens = int() }),
	})
	c.SystemOneValidationError = obj({
		detail = arr(obj({
			loc = arr({ type = { "string", "integer" } }), msg = str(), type = str(),
		})),
	}, nil, "FastAPI validation error.")
	c.SystemOneError = obj({ detail = str() })
end

-- other schemas

local function misc_schemas(c)
	c.Error = obj({
		error = obj({
			message = str(), type = str(),
			param = { type = { "string", "null" } }, code = { type = { "string", "null" } },
		}, { "message", "type" }),
	}, { "error" })
	local modes = { type = "string", enum = json.array(table.move(detect.MODES, 1, #detect.MODES, 1, {})) }
	c.Model = obj({
		id = str(), object = { const = "model" }, created = int(), owned_by = str(),
		context_length = ext(int()), meta = ext(obj({ n_ctx_train = int() })),
		capabilities = ext(arr(modes)),
		status = ext(obj({ value = str() }, nil, "Load state, as llama-server.")),
		lgml = ext(obj({ status = str(), bytes = int() })),
	})
	c.ModelList = obj({ object = { const = "list" }, data = arr(ref("Model")) })
	c.AnthropicModelList = obj({
		data = arr(obj({ type = { const = "model" }, id = str(), display_name = str(), created_at = str() })),
		has_more = { const = false }, first_id = { type = { "string", "null" } },
		last_id = { type = { "string", "null" } },
	})
	c.Props = ext(obj({
		default_generation_settings = obj({ n_ctx = int(), params = { type = "object" } }),
		total_slots = int(), model_path = str(), build_info = str(),
		modalities = obj({ vision = { type = "boolean" } }),
	}, nil, "llama-server /props subset."))
	c.LoraAdapter = ext(obj({ id = int(), path = str(), scale = num() }))
	c.DecisionRequest = obj({
		model = str("Any name; one this server does not serve goes to the default decision model."),
		input = { anyOf = json.array({ str(), arr({ type = "object" }) }),
			description = "Text, or user messages of input_text and input_image (data URL) parts." },
		questions = arr(obj({
			type = { type = "string", enum = json.array({ "predicate", "choice", "score" }) },
			name = str(), instructions = str(),
			choices = arr(obj({ value = { type = { "string", "number" } }, description = str() }, { "value" })),
			levels = arr(obj({ label = str(), description = str() }, { "label" })),
		}, { "type", "instructions" })),
	}, { "input", "questions" })
	c.Decision = obj({
		id = str(), object = { const = "decision" }, created_at = int(), model = str(),
		answers = arr(obj({
			type = { type = "string", enum = json.array({ "predicate", "choice", "score" }) },
			name = str(), probability = num(), choice = { type = { "string", "number" } },
			score = num(), confidence = num(),
			probabilities = arr(obj({ value = { type = { "string", "number" } }, label = str(),
				probability = num() })),
		}, { "type" })),
		usage = obj({ input_tokens = int(), total_tokens = int() }),
	})
	c.LoadRequest = obj({ model = str() }, { "model" })
	c.Success = obj({ success = { const = true } })
	c.Health = obj({ status = { const = "ok" } })
	local slot = obj({
		id = int("From 0."),
		state = { type = "string", enum = json.array({ "idle", "prompt", "answer" }) },
		request_id = str("The x-lgml-request-id of the request."),
		prompt_tokens = int(), prompt_done = int("Prompt tokens read so far, the cached ones too."),
		cached_tokens = int(), tokens = int("Tokens made so far."),
	}, { "id", "state" })
	c.Status = obj({
		models = arr(obj({
			id = str(), status = str("Load state, as in /v1/models."), slots = arr(slot),
			queued = int("Requests that wait for a slot or for the load."),
		}, { "id", "status", "slots", "queued" })),
	}, { "models" })
end

-- operations

local function json_body(schema)
	return { required = true, content = { ["application/json"] = { schema = schema } } }
end

local function form_body(schema)
	return { required = true, content = { ["multipart/form-data"] = { schema = schema } } }
end

-- The x-lgml-* headers of a reply, from their names without the prefix.
local function header_set(names)
	local t = {}
	for _, h in ipairs(names) do
		t["x-lgml-" .. h] = { schema = { type = "string" } }
	end
	return t
end

local function reply(desc, mt, schema, headers)
	return {
		description = desc, content = { [mt] = { schema = schema or {} } },
		headers = headers and header_set(headers),
	}
end

local function errors(codes)
	local t = {}
	for _, c in ipairs(codes) do
		t[tostring(c)] = reply("Error.", "application/json", ref("Error"))
	end
	return t
end

-- Merges error replies into responses.
local function with(responses, codes)
	for k, v in pairs(errors(codes)) do
		responses[k] = v
	end
	return responses
end

local AUDIO_HEADERS = { "elapsed-ms", "audio-seconds", "rtf" }
local SPEECH_HEADERS = { "elapsed-ms", "audio-seconds", "rtf", "sample-rate", "channels" }
local CHAT_HEADERS = { "cache-n", "restored-n", "prompt-n", "prompt-ms", "predicted-n",
	"predicted-ms", "predicted-per-second" }
local IMAGE_HEADERS = { "elapsed-ms", "images", "steps", "seed", "sample-ms", "ms-per-step" }
local EMBEDDING_HEADERS = { "elapsed-ms", "prompt-n", "prompt-ms", "prompt-per-second" }

-- Error replies in the Anthropic shape.
local function anthropic_errors(responses)
	for _, code in ipairs({ 400, 404, 500, 503 }) do
		responses[tostring(code)] = reply("Error.", "application/json", ref("AnthropicError"))
	end
	return responses
end

-- The headers of a Messages reply.
local function anthropic_headers()
	local str_schema = { schema = { type = "string" } }
	return {
		["request-id"] = { schema = { type = "string" }, description = "`req_` and the lgml request id." },
		["x-lgml-ignored"] = { schema = { type = "string" }, description = "Accepted fields with no effect." },
		["x-lgml-ignored-betas"] = str_schema,
		["x-lgml-thinking-dropped"] = { schema = { type = "integer" },
		    description = "Thinking blocks with a foreign signature that the server dropped." },
	}
end

local RESPONSE_ID = { { name = "response_id", ["in"] = "path", required = true, schema = { type = "string" } } }

-- Operations by path without the /v1 prefix, then by method. tag is
-- openai, llama.cpp or lgml.
local function operations()
	local E = { 400, 404, 500, 503 }
	return {
		["/chat/completions"] = { POST = {
			tag = "openai", summary = "Chat completion",
			requestBody = json_body(ref("ChatRequest")),
			responses = with({ ["200"] = {
				description = "The reply, or with stream an event stream of chunks.",
				content = {
					["application/json"] = { schema = ref("ChatCompletion") },
					["text/event-stream"] = { schema = ref("ChatCompletionChunk") },
				},
				headers = header_set(CHAT_HEADERS),
			} }, E),
		} },
		["/messages"] = { POST = {
			tag = "anthropic", summary = "Anthropic Messages",
			requestBody = json_body(ref("MessagesRequest")),
			responses = anthropic_errors({ ["200"] = {
				description = "The message, or with stream an event stream of named events.",
				content = {
					["application/json"] = { schema = ref("Message") },
					["text/event-stream"] = { schema = ref("MessageStreamEvent") },
				},
				headers = anthropic_headers(),
			} }),
		} },
		["/responses"] = { POST = {
			tag = "openai", summary = "Create a response",
			requestBody = json_body(ref("ResponsesRequest")),
			responses = with({ ["200"] = {
				description = "The response, or with stream an event stream of named events.",
				content = {
					["application/json"] = { schema = ref("Response") },
					["text/event-stream"] = { schema = ref("ResponseStreamEvent") },
				},
				headers = { ["x-lgml-ignored"] = { schema = { type = "string" },
				    description = "Accepted fields with no effect." } },
			} }, E),
		} },
		["/responses/input_tokens"] = { POST = {
			tag = "openai", summary = "Count the prompt tokens of a Responses request",
			requestBody = json_body(ref("ResponsesRequest")),
			responses = with({ ["200"] = reply("The prompt size.", "application/json", obj({
				object = { const = "response.input_tokens" }, input_tokens = int() })) }, E),
		} },
		["/responses/{response_id}"] = {
			GET = {
				tag = "openai", summary = "Get a stored response", parameters = RESPONSE_ID,
				responses = with({ ["200"] = reply("The response.", "application/json", ref("Response")) },
				    { 400, 404 }),
			},
			DELETE = {
				tag = "openai", summary = "Delete a stored response", parameters = RESPONSE_ID,
				responses = with({ ["200"] = reply("Deleted.", "application/json", obj({
					id = str(), object = { const = "response.deleted" }, deleted = { const = true } })) }, { 404 }),
			},
		},
		["/responses/{response_id}/input_items"] = { GET = {
			tag = "openai", summary = "List the input items of a stored response",
			parameters = { RESPONSE_ID[1],
				{ name = "limit", ["in"] = "query", schema = { type = "integer", minimum = 1, maximum = 100 } },
				{ name = "order", ["in"] = "query", schema = { type = "string", enum = json.array({ "asc", "desc" }) } },
				{ name = "after", ["in"] = "query", schema = { type = "string" } } },
			responses = with({ ["200"] = reply("A page of items.", "application/json", obj({
				object = { const = "list" }, data = arr({ type = "object" }), first_id = { type = { "string", "null" } },
				last_id = { type = { "string", "null" } }, has_more = { type = "boolean" } })) }, { 400, 404 }),
		} },
		["/responses/{response_id}/cancel"] = { POST = {
			tag = "openai", summary = "Cancel a background response (there are none)", parameters = RESPONSE_ID,
			requestBody = { required = false, content = { ["application/json"] = { schema = { type = "object" } } } },
			responses = errors({ 400, 404 }),
		} },
		["/messages/count_tokens"] = { POST = {
			tag = "anthropic", summary = "Count the prompt tokens of a Messages request",
			requestBody = json_body(ref("CountTokensRequest")),
			responses = anthropic_errors({ ["200"] = {
				description = "The prompt size.",
				content = { ["application/json"] = { schema = obj({ input_tokens = int() }, { "input_tokens" }) } },
				headers = anthropic_headers(),
			} }),
		} },
		["/images/generations"] = { POST = {
			tag = "openai", summary = "Make images",
			requestBody = json_body(ref("ImageRequest")),
			responses = with({ ["200"] = reply("Images.", "application/json", ref("ImagesResponse"),
			    IMAGE_HEADERS) }, E),
		} },
		["/images/edits"] = { POST = {
			tag = "openai", summary = "Edit an image",
			requestBody = form_body(ref("ImageEditRequest")),
			responses = with({ ["200"] = reply("Images.", "application/json", ref("ImagesResponse"),
			    IMAGE_HEADERS) }, E),
		} },
		["/embeddings"] = { POST = {
			tag = "openai", summary = "Embeddings",
			requestBody = json_body(ref("EmbeddingRequest")),
			responses = with({ ["200"] = reply("A vector per input.", "application/json",
			    ref("EmbeddingsResponse"), EMBEDDING_HEADERS) }, E),
		} },
		["/audio/speech"] = { POST = {
			tag = "openai", summary = "Text to speech",
			requestBody = json_body(ref("SpeechRequest")),
			responses = with({ ["200"] = {
				description = "16-bit mono audio.",
				content = {
					["audio/wav"] = { schema = { type = "string", contentMediaType = "audio/wav" } },
					["audio/pcm"] = { schema = { type = "string", contentMediaType = "audio/pcm" } },
				},
				headers = header_set(SPEECH_HEADERS),
			} }, E),
		} },
		["/audio/transcriptions"] = { POST = {
			tag = "openai", summary = "Speech to text",
			requestBody = form_body(ref("TranscriptionRequest")),
			responses = with({ ["200"] = {
				description = "The text, as response_format asks.",
				content = {
					["application/json"] = { schema = { anyOf = json.array({
						ref("Transcription"), ref("TranscriptionVerbose") }) } },
					["text/plain"] = { schema = { type = "string" } },
				},
				headers = header_set(AUDIO_HEADERS),
			} }, E),
		} },
		["/systemone"] = { POST = {
			tag = "lgml", summary = "System One decisions (Jev wire format)",
			requestBody = json_body(ref("SystemOneRequest")),
			responses = {
				["200"] = reply("Answers.", "application/json", ref("SystemOneResponse"), { "elapsed-ms" }),
				["404"] = reply("No model decides.", "application/json", ref("SystemOneError")),
				["422"] = reply("Bad request, or a bad recipe.", "application/json",
				    ref("SystemOneValidationError")),
				["500"] = reply("Engine error.", "application/json", ref("SystemOneError")),
				["503"] = reply("Worker or load failure.", "application/json", ref("SystemOneError")),
			},
		} },
		["/decisions"] = { POST = {
			tag = "openai", summary = "Decisions (OpenAI): predicate, choice and score answers",
			requestBody = json_body(ref("DecisionRequest")),
			responses = with({ ["200"] = reply("Answers.", "application/json", ref("Decision"),
			    { "elapsed-ms" }) }, { 400, 404, 500, 503 }),
		} },
		["/models"] = { GET = {
			tag = "openai", summary = "List models",
			responses = { ["200"] = reply("Models; with an anthropic-version header, the Anthropic list.",
			    "application/json", { anyOf = json.array({ ref("ModelList"), ref("AnthropicModelList") }) }) },
		} },
		["/models/load"] = { POST = {
			tag = "lgml", summary = "Load a model now",
			requestBody = json_body(ref("LoadRequest")),
			responses = with({ ["200"] = reply("Loaded.", "application/json", ref("Success")) },
			    { 400, 404, 503 }),
		} },
		["/models/unload"] = { POST = {
			tag = "lgml", summary = "Unload a model",
			requestBody = json_body(ref("LoadRequest")),
			responses = with({ ["200"] = reply("Unloaded.", "application/json", ref("Success")) },
			    { 400, 404, 409 }),
		} },
		["/props"] = { GET = {
			tag = "llama.cpp", summary = "Server properties",
			responses = { ["200"] = reply("Properties.", "application/json", ref("Props")) },
		} },
		["/lora-adapters"] = {
			GET = {
				tag = "llama.cpp", summary = "List LoRA adapters with the global scales",
				responses = { ["200"] = reply("Adapters.", "application/json", arr(ref("LoraAdapter"))) },
			},
			POST = {
				tag = "llama.cpp", summary = "Set the global LoRA scales",
				requestBody = json_body(arr(obj({ id = int(), scale = num() }, { "id", "scale" }),
				    "Adapters that the list does not name get scale 0.")),
				responses = with({ ["200"] = reply("Set.", "application/json", ref("Success")) }, { 400 }),
			},
		},
		["/status"] = { GET = {
			tag = "lgml", summary = "What the slots of each model do",
			responses = { ["200"] = reply("Slots.", "application/json", ref("Status")) },
		} },
		["/health"] = { GET = {
			tag = "llama.cpp", summary = "Health",
			responses = { ["200"] = reply("Up.", "application/json", ref("Health")) },
		} },
		["/metrics"] = { GET = {
			tag = "lgml", summary = "Prometheus metrics",
			responses = { ["200"] = reply("Metrics text.", "text/plain", { type = "string" }) },
		} },
		["/openapi.json"] = { GET = {
			tag = "lgml", summary = "This document",
			responses = { ["200"] = reply("OpenAPI document.", "application/json", { type = "object" }) },
		} },
	}
end

--- The route key of a path: the path without a `/v1` prefix.
-- @tparam string path the path
-- @treturn string the key
function M.canonical(path)
	return (path:gsub("^/v1/", "/"))
end

--- Builds the OpenAPI document.
-- Each route becomes an operation. A route with no operation in this
-- module gets a stub marked `x-lgml-undocumented`.
-- @tparam table routes the routes of a server: `{ [path] = { [method] =
-- handler } }`, as `lgml.server.httpd.Server` keeps them
-- @treturn table the document, ready for `lgml.json.encode`
function M.document(routes)
	local ops = operations()
	local schemas = {}
	chat_schemas(schemas)
	messages_schemas(schemas)
	responses_schemas(schemas)
	image_schemas(schemas)
	audio_schemas(schemas)
	embedding_schemas(schemas)
	systemone_schemas(schemas)
	misc_schemas(schemas)
	local paths = {}
	for path, methods in pairs(routes) do
		local item = {}
		for method in pairs(methods) do
			local src = ops[M.canonical(path)] and ops[M.canonical(path)][method]
			local op
			if src then
				op = {}
				for k, v in pairs(src) do
					op[k] = v
				end
				op.tag = nil
				op.tags = json.array({ src.tag })
			else
				op = {
					summary = method .. " " .. path, ["x-lgml-undocumented"] = true,
					responses = { default = { description = "See docs/api.md." } },
				}
			end
			op.operationId = method:lower() .. path:gsub("[^%w]+", "_")
			item[method:lower()] = op
		end
		paths[path] = item
	end
	return {
		openapi = M.OPENAPI,
		info = {
			title = "lgml", version = "scm", license = { name = "ISC", identifier = "ISC" },
			description = "OpenAI-shaped API of lgml-server, with the Anthropic Messages and OpenAI " ..
			    "Responses APIs. docs/api.md has the full reference. " ..
			    "Fields marked x-lgml-extension are lgml or llama.cpp extensions. " ..
			    "x-lgml-status gives the compatibility status of a chat request field.",
		},
		tags = json.array({
			{ name = "openai", description = "OpenAI-compatible, with lgml extensions." },
			{ name = "anthropic", description = "Anthropic-compatible." },
			{ name = "llama.cpp", description = "llama-server compatible." },
			{ name = "lgml", description = "lgml only." },
		}),
		-- The server has no authentication.
		security = json.array(),
		servers = json.array({ { url = "/" } }),
		paths = paths,
		components = { schemas = schemas },
	}
end

return M
