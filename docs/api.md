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
- [Compatibility](#compatibility)
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
| GET | `/props`, `/v1/props` | llama.cpp | [Server properties and adapters](#server-properties-and-adapters) |
| GET | `/lora-adapters`, `/v1/lora-adapters` | llama.cpp | [Server properties and adapters](#server-properties-and-adapters) |
| POST | `/lora-adapters`, `/v1/lora-adapters` | llama.cpp | [Server properties and adapters](#server-properties-and-adapters) |
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
- A chat request with a field that the server does not know gives
  400 (see [Compatibility](#compatibility)), and so does
  `/v1/systemone`. The other routes ignore unknown fields.
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
- `code` is `null`, except for a chat request that fails a check:
  `unknown_parameter` for a field that the server does not know,
  `unsupported_value` for a value that lgml does not support, and
  `invalid_value` for another bad value.

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
| speech | `x-lgml-elapsed-ms`, `x-lgml-audio-seconds` (length of the audio), `x-lgml-rtf` (wall time over audio time), `x-lgml-sample-rate` (samples per second), `x-lgml-channels` (1) |
| transcriptions | `x-lgml-elapsed-ms`, `x-lgml-audio-seconds` (length of the input), `x-lgml-rtf` |
| systemone | `x-lgml-elapsed-ms`, and `x-jevify-latency-ms` (the same time with 1 decimal, as Jevify sends) |

A streamed chat reply has no `x-lgml-*` headers. Its last chunk holds
the `timings` object instead. With `--cors`, browsers can read these
headers (`Access-Control-Expose-Headers: *`).

## Compatibility

`POST /v1/chat/completions` is a superset of the OpenAI Chat
Completions API:

- The server accepts each request that the OpenAI API accepts. A
  standard OpenAI client works with no change.
- A field that lgml cannot do as the OpenAI reference says gives 400
  with an OpenAI error object. Its `param` names the field. The server
  does not ignore such a field.
- A field that the OpenAI API does not know, and that is not an
  extension in the table below, gives 400, as the OpenAI API does.
- The extensions (llama.cpp and lgml fields) are optional. The server
  never requires them.

The request fields list is the one of `CompletionCreateParams` in the
openai-python SDK. Status values:

- **supported**: the server does what the OpenAI reference says.
- **limited**: supported, with the limits in the notes.
- **no-op**: accepted. It has no effect. For OpenAI it only labels,
  stores or routes the request, or it changes speed or cost only.
- **rejected**: a value other than the default gives 400.
- **planned**: a value other than the default gives 400 until lgml
  supports it.

`test/test_compat_openai.lua` reads this table and checks it against
the request checks (`lgml.server.chat_request.FIELDS`). It also sends
the request examples of the OpenAI reference (in
`test/fixtures/openai`) to the server with the fake engine, and checks
each reply against the OpenAI reply examples: the fields and their
types, the id prefixes, the object names, the finish reasons, the
usage, the event stream framing and the tool call fragments.

### Request fields

| Field | Label | Status | Notes |
| --- | --- | --- | --- |
| `messages` | OpenAI | supported | See [Messages](#messages). |
| `model` | OpenAI | supported | See [Model selection](#model-selection). |
| `stream` | OpenAI | supported | See [Streaming](#streaming). |
| `stream_options` | OpenAI | supported | `include_usage` adds a usage chunk. `include_obfuscation` has no effect: the server sends no `obfuscation` field. It needs `stream: true`, as for OpenAI. |
| `max_tokens` | OpenAI | supported | A positive integer. Default: the rest of the context. |
| `max_completion_tokens` | OpenAI | supported | As `max_tokens`; it counts reasoning tokens too. It wins when both are set. |
| `temperature` | OpenAI | supported | A number from 0. 0 picks the most likely token. |
| `top_p` | OpenAI | supported | From 0 to 1. |
| `presence_penalty` | OpenAI | supported | From -2 to 2. |
| `frequency_penalty` | OpenAI | supported | From -2 to 2. |
| `seed` | OpenAI | supported | Sampler seed. The same seed and request give the same reply on the same build. |
| `stop` | OpenAI | supported | A string or up to 4 strings. They apply to the content, not to the reasoning or the tool calls. The reply ends before the first stop string, with `finish_reason` `stop`; the stop string is not in it. An empty string has no effect. |
| `n` | OpenAI | supported | From 1 to 128. The engine makes the choices one after the other; each reuses the cached prompt. With `seed`, choice i takes seed + i. `usage` counts the prompt once and the tokens of all choices. A stream sends the chunks of choice 0, then those of choice 1, and so on. |
| `logprobs` | OpenAI | supported | `choices[].logprobs.content` lists the content tokens, in the stream too; see [Log probabilities](#log-probabilities). |
| `top_logprobs` | OpenAI | supported | From 0 to 20. It needs `logprobs: true`. |
| `logit_bias` | OpenAI | supported | Token ids of the model's tokenizer to a bias from -100 to 100. An id that the tokenizer does not have gives 400. |
| `tools` | OpenAI | limited | Function tools only. A `custom` tool gives 400. `strict` is accepted, but the arguments follow the parameter schema only when `tool_choice` forces a call on a Qwen3 template; see [Tool choice](#tool-choice). |
| `tool_choice` | OpenAI | supported | `none`, `auto`, `required`, `{"type": "function", "function": {"name": ...}}` or `allowed_tools`. A `custom` choice gives 400. See [Tool choice](#tool-choice). |
| `parallel_tool_calls` | OpenAI | supported | `false` ends the reply after the first tool call. |
| `response_format` | OpenAI | supported | See [Constrained output](#constrained-output). |
| `reasoning_effort` | OpenAI | limited | `none` turns thinking off. The other values turn it on and do not change its length. |
| `modalities` | OpenAI | limited | Only `["text"]`. |
| `user` | OpenAI | no-op | A string. |
| `safety_identifier` | OpenAI | no-op | A string. |
| `metadata` | OpenAI | no-op | At most 16 string values. |
| `store` | OpenAI | no-op | The server stores nothing. |
| `service_tier` | OpenAI | no-op | `auto`, `default`, `flex`, `scale`, `priority` or `fast`. |
| `prompt_cache_key` | OpenAI | no-op | The server reuses each shared prompt prefix; see `timings.cache_n`. |
| `prompt_cache_retention` | OpenAI | no-op | `in_memory` or `24h`. |
| `prompt_cache_options` | OpenAI | no-op | `mode` and `ttl`. |
| `prediction` | OpenAI | no-op | The reply is the same without it. |
| `verbosity` | OpenAI | rejected | Only `medium`. |
| `audio` | OpenAI | rejected | No audio output. |
| `moderation` | OpenAI | rejected | No moderation. |
| `web_search_options` | OpenAI | rejected | No web search. |
| `functions` | OpenAI | rejected | Deprecated by OpenAI. Use `tools`. |
| `function_call` | OpenAI | rejected | Deprecated by OpenAI. Use `tool_choice`. |
| `top_k` | llama.cpp | supported | Top-k sampling; 0 turns it off. |
| `min_p` | llama.cpp | supported | Min-p sampling, from 0 to 1. |
| `repeat_penalty` | llama.cpp | supported | Repeat penalty, above 0. |
| `grammar` | llama.cpp | supported | GBNF text for the answer. |
| `json_schema` | llama.cpp | supported | A JSON schema for the answer. |
| `chat_template_kwargs` | llama.cpp | supported | `enable_thinking`, `reasoning_effort` (a string for the template), `preserve_thinking` (false drops old reasoning). |
| `lora` | llama.cpp | supported | Adapter scales for this request: `[{"id": 0, "scale": 0.5}]`. Adapters that the list does not name get scale 0. Without `lora`, the global scales apply; see [Server properties and adapters](#server-properties-and-adapters). |

### Reply fields

Each field of the OpenAI reply is in the lgml reply, with the same
type. The table lists the paths; `[]` is each entry of an array.
`test/test_compat_openai.lua` checks each path against a reply of the
fake engine.

| Field | Status | Notes |
| --- | --- | --- |
| `id` | supported | `chatcmpl-` and 24 hex digits. The chunks of a stream share it. |
| `object` | supported | `chat.completion`, or `chat.completion.chunk` for each event of a stream. |
| `created` | supported | Unix time in seconds. |
| `model` | supported | The id of the model that served the request. |
| `system_fingerprint` | limited | Always `null`. |
| `choices[].index` | supported | From 0. |
| `choices[].message.role` | supported | `assistant`. |
| `choices[].message.content` | supported | `null` when the reply has tool calls and no text. |
| `choices[].message.refusal` | limited | Always `null`. |
| `choices[].message.tool_calls` | supported | Present when the reply has tool calls. |
| `choices[].finish_reason` | supported | `stop`, `length` or `tool_calls`. lgml has no content filter, so it never gives `content_filter`. |
| `choices[].logprobs` | supported | `null` unless the request sets `logprobs`. |
| `usage.prompt_tokens` | supported | |
| `usage.completion_tokens` | supported | It counts the reasoning tokens too. |
| `usage.total_tokens` | supported | |
| `usage.prompt_tokens_details.cached_tokens` | supported | Prompt tokens reused from the cache. |
| `usage.prompt_tokens_details.audio_tokens` | limited | Always 0. |
| `usage.completion_tokens_details.reasoning_tokens` | supported | Tokens of the reasoning block and its tags. |
| `usage.completion_tokens_details.audio_tokens` | limited | Always 0. |
| `usage.completion_tokens_details.accepted_prediction_tokens` | limited | Always 0. |
| `usage.completion_tokens_details.rejected_prediction_tokens` | limited | Always 0. |

The reply has no `service_tier`; the field is optional. The extensions
are `choices[].message.reasoning_content` and `timings` (llama.cpp).

### Message fields

| Field | Status | Notes |
| --- | --- | --- |
| `role` | supported | `system`, `developer`, `user`, `assistant` or `tool`. The deprecated `function` role gives 400. |
| `content` | limited | A string, `null`, or text and `image_url` parts. `input_audio` and `file` parts give 400. |
| `name` | no-op | The chat templates have no field for it. |
| `tool_calls` | supported | Function calls only. |
| `tool_call_id` | supported | Passed to the chat template. |
| `refusal` | rejected | Only `null` or `""`. |
| `annotations` | no-op | An array. |
| `audio`, `function_call` | rejected | 400. |
| `reasoning_content` | supported | llama.cpp: the reasoning of an earlier assistant turn. `reasoning` (vLLM) is a synonym. |

An `image_url.detail` value (`auto`, `low`, `high` or `original`) has
no effect. Image parts work in each role (lgml); OpenAI takes them in
`user` messages only. In a `tool` message, OpenAI takes text parts
only, so a standard client sends the images of a tool result in a
`user` message after the `tool` messages of the batch. lgml takes both
forms.
A text part can have `prompt_cache_breakpoint`; it has no effect.

## Chat completions

`POST /v1/chat/completions` (alias `/chat/completions`). OpenAI, with
llama.cpp and lgml extensions.

### Request

The [Compatibility](#compatibility) section lists each request field
with its status. The server checks each field and each message; a
failed check gives 400 with `param`. A field that is not in the table
gives 400.

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
- `tool_call_id`: passed to the chat template.
- `name`, `annotations`: accepted; they have no effect.

A `null` field in a message reads as absent. Another field gives 400;
see [Message fields](#message-fields).

Image input:

- The model needs a vision projector (`--mmproj`, or `mmproj` in a
  config model). Its `capabilities` then has `multimodal`.
- The URL must be a `data:image/...;base64,` URL. http and https URLs
  give 400.
- The image is PNG or JPEG, at most 32 MiB after base64 decode.
- `--image-min-tokens` and `--image-max-tokens` set the token range of
  one image.

### Tool choice

- `none`: the server does not give the tools to the chat template, and
  the reply has no tool calls.
- `auto` (the default): the model answers or calls tools. Without
  `tools`, the reply has no tool calls.
- `required`: the model must call one or more of the tools; with
  `parallel_tool_calls: false`, exactly one.
- `{"type": "function", "function": {"name": "f"}}`: the model must
  call `f`, once.
- `{"type": "allowed_tools", "allowed_tools": {"mode": ..., "tools":
  [...]}}`: the server keeps only the named tools and applies the mode,
  `auto` or `required`.

A name that is not in `tools`, or `required` without tools, gives 400.

`required` and a named function run with a grammar that allows only
tool calls in the syntax of the chat template, after the reasoning
block. The grammar replaces a `response_format` of the same request: a
reply with only tool calls has no content to format.

| Template | Forced call |
| --- | --- |
| Qwen3 (hermes) | The name and the arguments; the arguments follow the parameter schema. |
| Qwen3.5 | The name and the parameter names; the values are free text. |
| Gemma 4 | The name; the argument text is free. |

The diffusion engine cannot force a call; there `required` and a named
function give 400.

### Diffusion models

A diffusion model (such as diffusion-gemma) makes blocks of tokens at
once, with no sampler chain. It takes `seed`, `max_tokens`,
`max_completion_tokens`, `stop`, `n`, `tools`, `tool_choice` `none` and
`auto`, `parallel_tool_calls`, `reasoning_effort` and
`chat_template_kwargs.enable_thinking`. These fields give 400 there:
`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty` and
`frequency_penalty` (except 0), `repeat_penalty` (except 1),
`logit_bias`, `logprobs: true`, `response_format` with a schema,
`json_schema`, `grammar`, `lora`, other `chat_template_kwargs`, and
`tool_choice` `required` or a named function.

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
  "model": "qwen3-14b", "system_fingerprint": null,
  "choices": [{"index": 0, "finish_reason": "stop", "logprobs": null,
    "message": {"role": "assistant", "content": "Paris.", "refusal": null,
      "reasoning_content": "The user asks..."}}],
  "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29,
    "prompt_tokens_details": {"cached_tokens": 12, "audio_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 5, "audio_tokens": 0,
      "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}},
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
  reused from the cache. `usage.completion_tokens_details.reasoning_tokens`
  counts the tokens of the reasoning block.
- [Reply fields](#reply-fields) lists each field.
- `timings` (llama.cpp): `cache_n`, `restored_n`, `prompt_n`,
  `prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`,
  `predicted_n`, `predicted_ms`, `predicted_per_token_ms`,
  `predicted_per_second`. See [Response headers](#response-headers).

### Log probabilities

With `logprobs: true`, `choices[].logprobs` is `{"content": [...],
"refusal": null}`. Each entry is a content token: `token` (its text),
`logprob`, `bytes` (the UTF-8 bytes of the token) and `top_logprobs`,
the `top_logprobs` most likely tokens at that position with the same
fields. In a stream, each chunk with content carries the entries of
its tokens.

- The values are the log-softmax of the logits after `logit_bias` and
  the penalties, before temperature, `top_k`, `top_p` and `min_p`. With
  a grammar, the tokens that the grammar blocks can have -9999.
- As for OpenAI, the list covers the content only: reasoning and tool
  call tokens are not in it.
- A token of held-back text (such as the start of a stop string) goes
  out with the next content. The token that holds a stop string is in
  the list.

### Streaming

With `stream: true`, the reply is `text/event-stream`. Each event is
one `data:` line with a `chat.completion.chunk` object:

1. A first chunk with `delta: {"role": "assistant", "content": "", "refusal": null}`.
2. Chunks with `delta.reasoning_content` (llama.cpp) or
   `delta.content` text.
3. For a tool call: a chunk with `delta.tool_calls[0]` that has
   `index`, `id`, `type` and `function: {name, arguments: ""}`, then
   chunks with `index` and `function.arguments` text.
4. A last choice chunk with `delta: {}`, `finish_reason` and `timings`
   (llama.cpp). With `n` above 1, steps 1 to 4 repeat for each choice,
   with its `index`.
5. With `stream_options.include_usage`, a chunk with `choices: []` and
   `usage`. The other chunks then have `usage: null`, as OpenAI sends.
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
the model. Both replies have the headers `x-lgml-sample-rate` (for
example 24000) and `x-lgml-channels` (1).

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
| `prompt` | Text that guides the style and spelling, as the OpenAI Whisper initial prompt. The first 30 s window sees it as its previous text: the token `startofprev`, the prompt tokens, then the task tokens. It keeps its last 223 tokens (half the text context, less one). It needs `merges.txt` in the model directory. |
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

`GET /props` (alias `/v1/props`). llama.cpp. It gives a subset of the llama-server fields:

```json
{"default_generation_settings": {"n_ctx": 16384, "params": {}},
 "total_slots": 1, "model_path": "qwen3-14b",
 "modalities": {"vision": false}, "build_info": "lgml"}
```

With several models, the fields come from the first chat model.
`model_path` is the model id. `modalities` is absent when the engine
does not give it.

`GET /lora-adapters` (alias `/v1/lora-adapters`). llama.cpp. It lists
the adapters of the first chat model with their global scales, as
llama-server does:

```json
[{"id": 0, "path": "/models/tuned.gguf", "scale": 1}]
```

`POST /lora-adapters` (alias `/v1/lora-adapters`). llama.cpp. It sets
the global scales of the first chat model. The body is a list of
`{"id", "scale"}`. Adapters that the list does not name get scale 0.
The reply is `{"success": true}`. A bad id gives 400.

The scale rules are those of llama-server:

- The global scales start at the scales of `--lora file:scale`.
  `POST /lora-adapters` changes them. `GET /lora-adapters` shows them.
- A chat request with no `lora` field, and every `/v1/systemone`
  request, uses the global scales.
- A chat request with `"lora": [{"id": 0, "scale": 0.5}]` uses those
  scales for itself only. Adapters that it does not name get scale 0.
  The global scales do not change.
- A change of scales between two requests empties the slot cache. The
  prompt cache keeps entries of each scale set apart.
- With `-C config`, the server keeps the global scales of a model
  across unloads.

```sh
curl -s localhost:8080/props
curl -s localhost:8080/lora-adapters
curl -s localhost:8080/lora-adapters -d '[{"id": 0, "scale": 0.5}]'
```

## Health

`GET /health` (alias `/v1/health`). llama.cpp. It gives
`{"status": "ok"}` when the server listens. It does not check the
models.

## Metrics

`GET /metrics`. lgml. It has no `/v1` alias: `/metrics` is the path
that Prometheus scrapes by default. It gives Prometheus text
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
