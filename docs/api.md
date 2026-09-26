# lgml HTTP API

`lgml-server` serves an HTTP/1.1 API in the shape of the OpenAI API.
It also gives some llama-server (llama.cpp) endpoints and fields, and
lgml extensions. This file is the reference for each route. The Lua
reference (`ldoc .`) documents the modules behind it.

Labels in this file:

- **OpenAI**: the route or field has the shape of the OpenAI API.
- **llama.cpp**: the route or field has the shape of llama-server.
- **lgml**: an lgml extension. OpenAI clients ignore it.

## Contents

- [Routes](#routes)
- [Common behavior](#common-behavior)
- [Errors](#errors)
- [Response headers](#response-headers)
- [Chat completions](#chat-completions)
- [Images](#images)
- [Speech](#speech)
- [Transcriptions](#transcriptions)
- [System One decisions](#system-one-decisions)
- [Models](#models)
- [Load and unload](#load-and-unload)
- [Server properties and adapters](#server-properties-and-adapters)
- [Health](#health)
- [Metrics](#metrics)
- [CORS](#cors)
- [OpenAPI document](#openapi-document)

## Routes

Each route below is exact. There is no prefix matching. The paths in
one row are aliases for one handler.

| Method | Paths | Label | Section |
| --- | --- | --- | --- |
| POST | `/v1/chat/completions`, `/chat/completions` | OpenAI | [Chat completions](#chat-completions) |
| POST | `/v1/images/generations`, `/images/generations` | OpenAI | [Images](#images) |
| POST | `/v1/images/edits`, `/images/edits` | OpenAI | [Images](#images) |
| POST | `/v1/audio/speech`, `/audio/speech` | OpenAI | [Speech](#speech) |
| POST | `/v1/audio/transcriptions`, `/audio/transcriptions` | OpenAI | [Transcriptions](#transcriptions) |
| POST | `/v1/systemone`, `/systemone` | lgml (Jev) | [System One decisions](#system-one-decisions) |
| GET | `/v1/models`, `/models` | OpenAI | [Models](#models) |
| POST | `/models/load`, `/v1/models/load` | lgml | [Load and unload](#load-and-unload) |
| POST | `/models/unload`, `/v1/models/unload` | lgml | [Load and unload](#load-and-unload) |
| GET | `/props` | llama.cpp | [Server properties and adapters](#server-properties-and-adapters) |
| GET | `/lora-adapters` | llama.cpp | [Server properties and adapters](#server-properties-and-adapters) |
| GET | `/health`, `/v1/health` | llama.cpp | [Health](#health) |
| GET | `/metrics` | lgml | [Metrics](#metrics) |
| GET | `/openapi.json`, `/v1/openapi.json` | lgml | [OpenAPI document](#openapi-document) |

## Common behavior

- The server speaks HTTP/1.0 and HTTP/1.1. It keeps connections alive
  and answers pipelined requests in order. HTTP/2 gives 505.
- A GET route also answers HEAD, with no body.
- The request body is JSON, except for the multipart routes
  (`/images/edits`, `/audio/transcriptions`). The server does not read
  `Content-Type` on JSON routes.
- Chunked request bodies give 411. Send `Content-Length`.
- Limits: a header block of at most 64 KiB (else 431), a body of at
  most 64 MiB (else 413). The server waits 30 s for the rest of a
  header block and 60 s for each piece of a body (else 408). An idle
  connection closes after 60 s.
- `Expect: 100-continue` gets `100 Continue`.
- There is no authentication. An `Authorization` header has no effect.
- Unknown JSON fields are ignored, except on `/v1/systemone`, which
  rejects them.
- A JSON `null` at the top level of a chat or image request reads as
  an absent field.
- When the client closes the connection, the server cancels the work.
- Each reply has the header `x-lgml-request-id`, an id of 8 hex
  digits. The log lines of the request, in the server and in the
  worker, carry it as `rid=<id>`. See the Logs section of README.md.

### Model selection

With `-m`, the server has one model. With `-C config`, it has several.
The `model` field picks one by id. A request with no `model`, or with
an unknown id, goes to the first model that serves that kind of
request (as llama-server ignores `model`). Models of a config load on
the first request for them; see [Load and unload](#load-and-unload).

A chat request with images and no `model` (or an unknown one) goes to
the first model with vision. A request with images for a named model
without vision gives 400, and the message names the models with vision.

The `model` field of a chat reply is the id of the model that served
the request. With no `model` or an unknown id, it is the id of the
model that the server picked, not the id in the request.

## Errors

The OpenAI routes give the OpenAI error object:

```json
{"error": {"message": "max_tokens is not valid", "type": "invalid_request_error",
  "param": "max_tokens", "code": null}}
```

- `type` is `not_found_error` for 404, `authentication_error` for 401,
  `permission_error` for 403, `rate_limit_error` for 429, and
  `server_error` for 500 and higher. Other codes give
  `invalid_request_error`.
- `param` names the bad field when the server knows it, else `null`.
- `code` is always `null`.

`/v1/systemone` uses the FastAPI error shape instead; see
[System One errors](#system-one-errors).

Status codes:

| Code | When |
| --- | --- |
| 400 | Bad JSON, a field that fails a check, a bad grammar or schema, a prompt longer than the context, image input to a model without vision, an engine that rejects the request, a load or unload request to a server started without `-C config`, or with `--inproc`. |
| 404 | No route for the path. The engine does not serve this kind of request (for example `/v1/audio/speech` with a chat model only). An unknown model id on load or unload. |
| 405 | The path exists, the method does not. The `Allow` header lists the methods. |
| 408, 411, 413, 431, 505 | HTTP errors; see [Common behavior](#common-behavior). |
| 409 | `/models/unload` while the model loads or has requests in flight. |
| 422 | `/v1/systemone`: the body fails a check, or `lgml.recipe` is not valid. |
| 500 | The engine failed, or a handler raised an error. |
| 503 | A model worker died or is not running, a load failed, a load waited `load_wait` seconds for memory, or the server is stopping. |

A streamed chat reply that fails after the first event ends with an
error event and `[DONE]`; see [Streaming](#streaming).

## Response headers

Every reply, errors too, has `x-lgml-request-id`: the id of the request
in the server log (`rid=<id>`).

Replies carry `x-lgml-*` headers with numbers for the request. The
header name is `x-lgml-` plus the field name, with `_` as `-`.
Integers are plain; other numbers have 3 decimals. These headers are
lgml extensions.

| Route | Headers |
| --- | --- |
| chat, not streamed | the llama.cpp timings of the reply: `x-lgml-cache-n` (prompt tokens reused from the KV cache), `x-lgml-restored-n` (prompt tokens restored from the host prompt store), `x-lgml-prompt-n` (prompt tokens computed), `x-lgml-prompt-ms`, `x-lgml-prompt-per-token-ms`, `x-lgml-prompt-per-second`, `x-lgml-predicted-n` (tokens made), `x-lgml-predicted-ms`, `x-lgml-predicted-per-token-ms`, `x-lgml-predicted-per-second`. An engine can give fewer. |
| images | `x-lgml-elapsed-ms` (wall time), `x-lgml-images` (count), `x-lgml-steps` (sampling steps of all images), `x-lgml-seed` (seed of the first image), `x-lgml-sample-ms` (sampling time), `x-lgml-ms-per-step` |
| speech | `x-lgml-elapsed-ms`, `x-lgml-audio-seconds` (length of the audio), `x-lgml-rtf` (wall time over audio time) |
| transcriptions | `x-lgml-elapsed-ms`, `x-lgml-audio-seconds` (length of the input), `x-lgml-rtf` |
| systemone | `x-lgml-elapsed-ms`, and `x-jevify-latency-ms` (the same time with 1 decimal, as Jevify sends) |

A streamed chat reply has no `x-lgml-*` headers. Its last chunk holds
the `timings` object instead. With `--cors`, browsers can read these
headers (`Access-Control-Expose-Headers: *`).

## Chat completions

`POST /v1/chat/completions` (alias `/chat/completions`). OpenAI, with
llama.cpp and lgml extensions.

### Request

| Field | Type | Label | Notes |
| --- | --- | --- | --- |
| `messages` | array | OpenAI | Required, not empty. See [Messages](#messages). |
| `model` | string | OpenAI | See [Model selection](#model-selection). |
| `stream` | boolean | OpenAI | Send server-sent events. Default false. |
| `stream_options` | object | OpenAI | `include_usage: true` adds a usage chunk at the end of a stream. |
| `max_tokens`, `max_completion_tokens` | integer | OpenAI | A positive integer. `max_completion_tokens` wins when both are set. Default: the rest of the context. |
| `temperature` | number | OpenAI | Sampler temperature. |
| `top_p` | number | OpenAI | Nucleus sampling. |
| `presence_penalty`, `frequency_penalty` | number | OpenAI | Sampler penalties. |
| `seed` | integer | OpenAI | Sampler seed. |
| `n` | integer | OpenAI | Only 1. |
| `tools` | array | OpenAI | Function tools. The chat template renders them. |
| `response_format` | object | OpenAI | See [Constrained output](#constrained-output). |
| `top_k`, `min_p`, `repeat_penalty` | number | llama.cpp | Sampler settings. |
| `grammar` | string | llama.cpp | GBNF grammar for the answer. |
| `json_schema` | object | llama.cpp | JSON schema for the answer. |
| `lora` | array | llama.cpp | Adapter scales for this request: `[{"id": 0, "scale": 0.5}]`. |
| `chat_template_kwargs` | object | llama.cpp | `enable_thinking`, `reasoning_effort` (a string for the template), `preserve_thinking` (false drops old reasoning). |
| `reasoning_effort` | string | OpenAI | `"none"` turns thinking off. Other values leave it on. |

The server checks the types of `model`, `stream`, `stream_options`,
`temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `tools`
and `n`, and each message. A failed check gives 400 with `param`. The
engine reads the other fields. It ignores fields that it does not
know, such as `stop`, `tool_choice`, `logprobs` and `user`.

Thinking is on by default for templates that have it.
`chat_template_kwargs.enable_thinking` decides when it is set. Else
`reasoning_effort: "none"` turns it off.

Sampler defaults of the gguf chat engine (the Qwen3 model card
values): with thinking, temperature 0.6, top_k 20, top_p 0.95, min_p 0;
without thinking, temperature 0.7, top_k 20, top_p 0.8, min_p 0.
Request fields override them. The other settings have the llama.cpp
defaults.

### Messages

Each message is an object:

- `role`: `system`, `developer`, `user`, `assistant` or `tool`.
- `content`: a string, `null`, or an array of parts.
  - `{"type": "text", "text": "..."}`
  - `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`.
    `image_url` can also be the URL string itself.
- `reasoning_content` (lgml, llama.cpp): a string, the reasoning of an
  earlier assistant turn.
- `tool_calls`: an array. Each call has `function.name` and
  `function.arguments`, a JSON string or an object.
- `tool_call_id`, `name`: passed to the chat template.

A `null` field in a message reads as absent.

Image input:

- The model needs a vision projector (`--mmproj`, or `mmproj` in a
  config model). Its `capabilities` then has `multimodal`.
- The URL must be a `data:image/...;base64,` URL. http and https URLs
  give 400.
- The image is PNG or JPEG, at most 32 MiB after base64 decode.
- `--image-min-tokens` and `--image-max-tokens` set the token range of
  one image.

### Constrained output

- `response_format: {"type": "text"}`: no constraint.
- `response_format: {"type": "json_object"}`: any JSON object. With
  `schema` in the object (llama.cpp), that schema.
- `response_format: {"type": "json_schema", "json_schema": {"schema": {...}}}`:
  the schema. The server keeps the order of the schema properties.
- `json_schema: {...}` (llama.cpp): the schema.
- `grammar: "root ::= ..."` (llama.cpp): GBNF text.

A request with both `grammar` and a schema gives 400. A schema or
grammar that does not compile gives 400. With thinking on, the
reasoning block stays free; the constraint applies to the answer after
it.

### Response

```json
{
  "id": "chatcmpl-...", "object": "chat.completion", "created": 1790414504,
  "model": "qwen3-14b",
  "choices": [{"index": 0, "finish_reason": "stop", "logprobs": null,
    "message": {"role": "assistant", "content": "Paris.",
      "reasoning_content": "The user asks..."}}],
  "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29,
    "prompt_tokens_details": {"cached_tokens": 12}},
  "timings": {"cache_n": 12, "prompt_n": 8, "prompt_ms": 31.2, "predicted_n": 9,
    "predicted_ms": 140.5, "predicted_per_second": 64.1}
}
```

- `finish_reason` is `stop`, `length` or `tool_calls`. A reply with
  tool calls gives `tool_calls`.
- `message.reasoning_content` (llama.cpp) holds the reasoning. It is
  absent when the reply has none.
- `message.tool_calls` holds the calls, each with `id` (`call_...`),
  `type: "function"` and `function: {name, arguments}`. `arguments` is
  a JSON string. With tool calls and no text, `content` is `null`.
- `usage.prompt_tokens_details.cached_tokens` counts the prompt tokens
  reused from the cache.
- `timings` (llama.cpp): `cache_n`, `restored_n`, `prompt_n`,
  `prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`,
  `predicted_n`, `predicted_ms`, `predicted_per_token_ms`,
  `predicted_per_second`. See [Response headers](#response-headers).

### Streaming

With `stream: true`, the reply is `text/event-stream`. Each event is
one `data:` line with a `chat.completion.chunk` object:

1. A first chunk with `delta: {"role": "assistant", "content": ""}`.
2. Chunks with `delta.reasoning_content` (llama.cpp) or
   `delta.content` text.
3. For a tool call: a chunk with `delta.tool_calls[0]` that has
   `index`, `id`, `type` and `function: {name, arguments: ""}`, then
   chunks with `index` and `function.arguments` text.
4. A last choice chunk with `delta: {}`, `finish_reason` and `timings`
   (llama.cpp).
5. With `stream_options.include_usage`, a chunk with `choices: []` and
   `usage`.
6. `data: [DONE]`.

When the engine fails before the first event, the reply is a plain
JSON error with its status. When it fails later, the stream sends
`data: {"error": {...}}` and then `data: [DONE]`.

### Examples

```sh
curl -s localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Name the capital of France."}],
  "max_tokens": 64}'

curl -sN localhost:8080/v1/chat/completions -d '{
  "stream": true, "stream_options": {"include_usage": true},
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "Count to 3."}]}'

curl -s localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Give a city as JSON."}],
  "response_format": {"type": "json_schema", "json_schema": {"schema": {
    "type": "object", "properties": {"city": {"type": "string"}},
    "required": ["city"]}}}}'

curl -s localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}}]}]}'
```

## Images

`POST /v1/images/generations` (alias `/images/generations`) takes JSON.
`POST /v1/images/edits` (alias `/images/edits`) takes
`multipart/form-data`. OpenAI, with lgml extensions. A model with
capability `image_generation` serves them, such as Z-Image with
`--engine lgml.image.engine`.

### Request

| Field | Label | Notes |
| --- | --- | --- |
| `prompt` | OpenAI | Required. 1 to 16384 bytes. |
| `model` | OpenAI | See [Model selection](#model-selection). |
| `n` | OpenAI | An integer from 1 to 10. Default 1. |
| `size` | OpenAI | `auto` or `WxH`. Each side is a multiple of 16 from 16 to 2048. Default for generations: `1024x1024`. For edits, `auto` is the input size, rounded down to a multiple of 16. |
| `response_format` | OpenAI | Only `b64_json` (the default). |
| `output_format` | OpenAI | Only `png` (the default). |
| `steps` | lgml | Sampling steps, an integer from 1 to 100. Z-Image default: 8. |
| `seed` | lgml | An integer from 0 to 2^53 - 1. Image i uses seed + i - 1. Default: random. |
| `cfg_scale` | lgml | Guidance scale, a number from 0 to 30. 1 (the Z-Image default) runs one pass per step. |
| `negative_prompt` | lgml | Text for the unconditional pass. |
| `strength` | lgml | Edits only: above 0, at most 1. The part of the schedule that runs on the input image. Z-Image default: 0.75. |

Edits take the same fields as form fields, with numbers as text. The
input image is the file field `image` (or `image[]`), PNG or JPEG. A
`mask` gives 400.

### Response

```json
{"created": 1790414504, "output_format": "png", "size": "1024x1024",
 "data": [{"b64_json": "iVBORw0...", "seed": 7}]}
```

`data[].seed` (lgml) is the seed of each image. See also
[Response headers](#response-headers).

### Examples

```sh
curl -s localhost:8080/v1/images/generations -d '{
  "prompt": "a red fox in snow", "size": "768x512", "steps": 8, "seed": 7}' |
  jq -r '.data[0].b64_json' | base64 -d > fox.png

curl -s localhost:8080/v1/images/edits -F image=@fox.png \
  -F prompt='a grey wolf in snow' -F strength=0.6 |
  jq -r '.data[0].b64_json' | base64 -d > wolf.png
```

## Speech

`POST /v1/audio/speech` (alias `/audio/speech`). OpenAI. A model with
capability `speech` serves it, such as `lgml.tts.orpheus`,
`lgml.tts.qwen3tts` or `lgml.tts.vibevoice`.

| Field | Label | Notes |
| --- | --- | --- |
| `input` | OpenAI | Required. 1 to 16384 bytes. A model can set a lower limit. |
| `model` | OpenAI | See [Model selection](#model-selection). |
| `voice` | OpenAI | A voice of the model. Orpheus and Qwen3-TTS also map the OpenAI voice names. |
| `speed` | OpenAI | A number from 0.25 to 4. The current models take only 1. |
| `response_format` | OpenAI | `wav` (default) or `pcm`. |
| `seed` | lgml | Qwen3-TTS and VibeVoice. |
| `language` | lgml | Qwen3-TTS: a language name or `auto`. |
| `steps`, `cfg_scale` | lgml | VibeVoice. |
| `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` | lgml | Orpheus sampler settings. |

The reply is 16-bit mono audio. `wav` is a WAV file (`audio/wav`).
`pcm` is the raw little-endian samples (`audio/pcm`) at the rate of
the model; use `wav` to learn the rate.

```sh
curl -s localhost:8080/v1/audio/speech -d '{"input": "Hello there.", "voice": "tara"}' -o hello.wav
```

## Transcriptions

`POST /v1/audio/transcriptions` (alias `/audio/transcriptions`).
OpenAI. The body is `multipart/form-data`. A Whisper model (capability
`transcription`) serves it.

| Field | Notes |
| --- | --- |
| `file` | Required. A WAV file: PCM with 8, 16, 24 or 32 bits, or float with 32 or 64 bits. Channels mix to mono. The server resamples to 16 kHz. |
| `model` | See [Model selection](#model-selection). |
| `language` | A language code or name. Default: detect. |
| `prompt` | Accepted. Not used. |
| `response_format` | `json` (default), `text` or `verbose_json`. Others, such as `srt`, give 400. |

Replies:

- `json`: `{"text": "..."}`
- `text`: the text and a newline, `text/plain`.
- `verbose_json`: `{"task": "transcribe", "language": "english",
  "duration": 11.0, "text": "...", "segments": [{"id": 0, "start": 0.0,
  "end": 4.2, "text": "..."}]}`. Times are in seconds.

```sh
curl -s localhost:8080/v1/audio/transcriptions -F file=@jfk.wav -F response_format=verbose_json
```

## System One decisions

`POST /v1/systemone` (alias `/systemone`). This is the System One API
of Jev 1.13.0, as Jevify implements it. A model does not generate an
answer. It reads a state and scores the allowed answers of typed
questions in one pass each. A chat model with capability `decide`
serves it. The fields `lgml` and `attachments` are lgml extensions.

### Request

```json
{
  "state": {"hp": 3, "enemy": "troll"},
  "model": "qwen3-14b",
  "questions": {
    "flee": {"type": "noul", "instructions": "Should the player flee?"},
    "act": {"type": "choice", "instructions": "Pick an action.",
      "criteria": {"attack": "Hit the enemy.", "run": "Leave the room."}},
    "mood": {"type": "score", "criteria": ["calm", "tense", "panic"]}
  }
}
```

Top-level fields. Any other field gives 422.

- `state` (required): a string, an object or an array. A JSON state
  becomes its indented JSON text (as Python `json.dumps(indent=2)`).
  The prompt keeps the first 12000 characters.
- `model`: a string. See [Model selection](#model-selection).
- `questions` (required): an object of questions by id, with at least
  one. The answers keep this order. A repeated id gives 422.
- `lgml` (lgml): an object with `raw` and `recipe`; see below.
- `attachments` (lgml): an array of images for the state. Each item is
  a data URL string (`data:image/png;base64,...`) or an OpenAI part
  `{"type": "image_url", "image_url": {"url": "data:..."}}`. The images
  are PNG or JPEG. A request with attachments goes to a model with
  capability `multimodal`.

Question fields. Any other field gives 422.

- `type` (required): `noul`, `choice` or `score`.
- `instructions`: a string, an object, an array or `null`.
- `criteria`:
  - noul: optional. `null`, or an object with only the keys `"true"`
    and `"false"`, which describe each answer.
  - choice: required. An object of 2 to 255 options, from name to
    description. Names are not empty and not repeated.
  - score: required. An array of 2 to 10 levels, lowest first. Each
    level is a string, an object or an array.

A description that is not a string goes into the prompt as one-line
JSON.

### lgml extension object

- `lgml.raw`: a boolean. When true, each answer has an `lgml` object
  with the raw log-scores: `runs` (one `{keys, logscores}` per option
  order) and `prior` (the content-free run, or absent).
- `lgml.recipe`: an object with recipe fields. They change the server
  recipe (`--recipe`, a Jevify `recipe.json`) for this request. A
  `{"recipe": {...}}` object (the `recipe.json` form) also works. Each
  field is a number for all question types, or an object by type
  (`choice`, `score`, `noul`):
  - `permutations`: option orders to average, an integer from 1 to 8.
    Default: 2 for choice, 1 for the others.
  - `temperature`: above 0. Default 1.
  - `prior_weight`: at least 0. Default 0 (no prior correction).
  - `bias`: log-odds added to the noul answer. Default 0.
  - `temp_k_slope`: for K > 2 options, the temperature grows by
    `temp_k_slope * log2(K / 2)`. Default 0.
  - `mode` must be `index`, `state_last` must be false, and `prompt`
    must be `jevify`, when given.

  A recipe that fails these rules gives 422 with a `detail` list, and
  `loc` is `["body", "lgml", "recipe"]`.

### Response

```json
{"model": "qwen3-14b", "answers": {
  "flee": {"type": "noul", "noul": 0.27},
  "act": {"type": "choice", "choice": "attack", "confidence": 0.4,
    "probabilities": {"attack": 0.7, "run": 0.3}},
  "mood": {"type": "score", "score": 0.99, "confidence": 0.1815,
    "legend": {"0": "calm", "1": "tense", "2": "panic"},
    "probabilities": {"0": 0.34, "1": 0.33, "2": 0.33}}},
 "usage": {"input_tokens": 51, "output_tokens": 7}}
```

- noul: `noul` is P(yes), rounded to 0.01. There is no confidence.
- choice: `choice` is the option with the highest probability.
  `confidence` is `(p_max - 1/K) / (1 - 1/K)` for K options.
- score: `score` is the expected level index, from 0. `confidence` is
  `1 - std / ((K - 1) / 2)`. `legend` maps each index to its level.
- `probabilities` sum to exactly 1, with 0.01 steps (largest remainder
  rounding). `score` and `confidence` have 4 decimal places.
- `model` is the id of the model that answered.
- `usage` counts the prompt tokens run and the candidate tokens read.

The body is formatted as Python `json.dumps` formats it, with the key
order of the request. The reply has the headers `x-lgml-elapsed-ms`
and `x-jevify-latency-ms`.

### System One errors

- 422 for a body that fails a check, as FastAPI gives it:
  `{"detail": [{"loc": ["body", "questions", "q", "criteria"], "msg":
  "...", "type": "value_error"}]}`. Bad JSON has type `json_invalid`.
- 422 for a bad `lgml.recipe`, in the same shape, with `loc`
  `["body", "lgml", "recipe"]`.
- 404 `{"detail": "this server does not serve decisions"}` when no
  model decides.
- 400, 500 and 503 give `{"detail": "message"}`. For example, a prompt
  longer than the context gives 400.

### Example

```sh
curl -s localhost:8080/v1/systemone -d '{
  "state": "The player has 3 hp. A troll blocks the door.",
  "questions": {"flee": {"type": "noul", "instructions": "Should the player flee?"}},
  "lgml": {"raw": true, "recipe": {"temperature": {"noul": 1.5}}}}'
```

## Models

`GET /v1/models` (alias `/models`). OpenAI, with llama.cpp and lgml
fields. It does not load a model.

```json
{"object": "list", "data": [{
  "id": "qwen3-14b", "object": "model", "created": 0, "owned_by": "lgml",
  "context_length": 16384, "meta": {"n_ctx_train": 16384},
  "capabilities": ["completion", "decide"],
  "status": {"value": "loaded"},
  "lgml": {"status": "loaded", "bytes": 10485760000}}]}
```

- `context_length` (lgml) and `meta.n_ctx_train` (llama.cpp): the
  context size of the model, in tokens.
- `capabilities` (llama.cpp names where llama-server has them): the
  kinds of request that the model serves.
  - `completion`: chat completions
  - `multimodal`: chat completions with image input
  - `decide`: `/v1/systemone`
  - `transcription`: `/v1/audio/transcriptions`
  - `speech`: `/v1/audio/speech`
  - `image_generation`: `/v1/images/generations` and `/v1/images/edits`
- `status.value` (llama.cpp) and `lgml.status`: `unloaded`, `loading`,
  `loaded` or `unloading`. A server without a config gives `loaded`.
- `lgml.bytes`: with a config, the memory that the model counts
  against its budget.

```sh
curl -s localhost:8080/v1/models | jq '.data[] | {id, capabilities, status}'
```

## Load and unload

`POST /models/load` and `POST /models/unload` (also under `/v1`).
lgml. The body is `{"model": "id"}`. The reply is `{"success": true}`.

These routes work with `-C config`, where each model runs in its own
worker. A worker also starts on the first request for its model.

- Load starts the worker and waits for it. When the model is loaded,
  load returns at once. When the memory budget has no room, the pool
  unloads idle models, least recently used first. When that is not
  enough, the load waits up to `load_wait` seconds (default 60) for
  requests of other models to end, then fails with 503.
- Unload stops the worker. It gives 409 while the model loads or has
  requests in flight. An unloaded model gives success.
- A missing `model` gives 400. An unknown id gives 404.
- A server started with `-m`, `--fake` or `--inproc` does not load
  models on demand. Load and unload give 400 with the message "this
  server does not load models on demand; start it with -C config".

```sh
curl -s localhost:8080/models/load -d '{"model": "qwen3-14b"}'
curl -s localhost:8080/models/unload -d '{"model": "qwen3-14b"}'
```

## Server properties and adapters

`GET /props`. llama.cpp. It gives a subset of the llama-server fields:

```json
{"default_generation_settings": {"n_ctx": 16384, "params": {}},
 "total_slots": 1, "model_path": "qwen3-14b",
 "modalities": {"vision": false}, "build_info": "lgml"}
```

With several models, the fields come from the first chat model.
`model_path` is the model id. `modalities` is absent when the engine
does not give it.

`GET /lora-adapters`. llama.cpp. It lists the adapters of the first
chat model, as llama-server does:

```json
[{"id": 0, "path": "/models/tuned.gguf", "scale": 1}]
```

`scale` is the scale from `--lora file:scale`. A chat request can
change the scales for itself with `"lora": [{"id": 0, "scale": 0.5}]`.
Adapters that the request does not list keep their start scale.

```sh
curl -s localhost:8080/props
curl -s localhost:8080/lora-adapters
```

## Health

`GET /health` (alias `/v1/health`). llama.cpp. It gives
`{"status": "ok"}` when the server listens. It does not check the
models.

## Metrics

`GET /metrics`. lgml. It gives Prometheus text
(`text/plain; version=0.0.4`). Each metric has the prefix `lgml_` and
the label `model`, the id of the model that served the request. An
unknown id in a request does not make a new label value. A request
that no model serves uses the first model id. A metric appears after
its first event.

| Metric | Type | Meaning |
| --- | --- | --- |
| `lgml_<kind>_requests_total` | counter | Requests, by kind: `chat`, `transcription`, `speech`, `decide`, `image`. |
| `lgml_<kind>_errors_total` | counter | Requests of that kind that failed. |
| `lgml_prompt_tokens_total` | counter | Prompt tokens computed. |
| `lgml_prompt_seconds_total` | counter | Time spent on prompts. |
| `lgml_predicted_tokens_total` | counter | Tokens made. |
| `lgml_predicted_seconds_total` | counter | Time spent making tokens. |
| `lgml_cache_tokens_total` | counter | Prompt tokens reused from the KV cache. |
| `lgml_restored_tokens_total` | counter | Prompt tokens restored from host memory. |
| `lgml_transcribed_seconds_total` | counter | Audio seconds transcribed. |
| `lgml_spoken_seconds_total` | counter | Audio seconds of speech made. |
| `lgml_images_total` | counter | Images made. |
| `lgml_image_steps_total` | counter | Image sampling steps run. |
| `lgml_image_seconds_total` | counter | Time spent making images. |
| `lgml_model_loads_total` | counter | Model loads (config only). |
| `lgml_model_load_seconds_total` | counter | Time spent loading models. |
| `lgml_model_load_failures_total` | counter | Model loads that failed. |
| `lgml_model_unloads_total` | counter | Model unloads. |
| `lgml_model_loaded_bytes` | gauge | Memory of the loaded model, 0 when unloaded. |

## CORS

By default the server sends no CORS headers, and `OPTIONS` gives 405.
`--cors origin` (or `cors = "origin"` in a config) turns CORS on. The
origin is a value such as `http://localhost:8000` or `*`.

- Every response then has `Access-Control-Allow-Origin: origin` and
  `Access-Control-Expose-Headers: *`. An origin other than `*` also
  adds `Vary: Origin`.
- `OPTIONS` on a known path gives 204 with
  `Access-Control-Allow-Methods` (the methods of the path, plus `HEAD`
  for a GET path, plus `OPTIONS`), `Access-Control-Allow-Headers` (the
  request `Access-Control-Request-Headers`, else `*`) and
  `Access-Control-Max-Age: 600`.
- `OPTIONS` on an unknown path gives 404.

## OpenAPI document

`GET /openapi.json` (alias `/v1/openapi.json`). lgml. It gives an
OpenAPI 3.1 document of the routes of this server. The server builds
it from the tables that its request checks use, so the request fields
match the checks. Fields and schemas with `x-lgml-extension: true` are
lgml or llama.cpp extensions. The descriptions are short; this file
is the full reference.

```sh
curl -s localhost:8080/openapi.json | jq '.paths | keys'
```
