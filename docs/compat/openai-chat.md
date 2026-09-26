# OpenAI Chat Completions compatibility reference

Endpoint: `POST /v1/chat/completions`. Snapshot date: 2026-09-26.

This document describes the wire format that clients use today, with focus on
image input. lgml uses it to implement the endpoint. The Responses API and the
Anthropic Messages API have separate documents.

## Sources

Each point cites a source key from this table. The SDK paths are in the
`openai/openai-python` repository, branch `main`, under `src/openai/`.

| Key | Source |
| --- | --- |
| `[params]` | `types/chat/completion_create_params.py` |
| `[msg-user]` | `types/chat/chat_completion_user_message_param.py` |
| `[msg-sys]` | `types/chat/chat_completion_system_message_param.py` |
| `[msg-dev]` | `types/chat/chat_completion_developer_message_param.py` |
| `[msg-asst]` | `types/chat/chat_completion_assistant_message_param.py` |
| `[msg-tool]` | `types/chat/chat_completion_tool_message_param.py` |
| `[msg-fn]` | `types/chat/chat_completion_function_message_param.py` |
| `[part]` | `types/chat/chat_completion_content_part_param.py` (union, `file` part) |
| `[part-text]` | `types/chat/chat_completion_content_part_text_param.py` |
| `[part-image]` | `types/chat/chat_completion_content_part_image_param.py` |
| `[part-audio]` | `types/chat/chat_completion_content_part_input_audio_param.py` |
| `[completion]` | `types/chat/chat_completion.py` |
| `[message]` | `types/chat/chat_completion_message.py` |
| `[chunk]` | `types/chat/chat_completion_chunk.py` |
| `[usage]` | `types/completion_usage.py` |
| `[logprob]` | `types/chat/chat_completion_token_logprob.py` |
| `[effort]` | `types/shared/reasoning_effort.py` |
| `[fndef]` | `types/shared_params/function_definition.py` |
| `[toolcall]` | `types/chat/chat_completion_message_function_tool_call.py` |
| `[toolchoice]` | `types/chat/chat_completion_tool_choice_option_param.py`, `chat_completion_named_tool_choice_param.py`, `chat_completion_allowed_tool_choice_param.py`, `chat_completion_allowed_tools_param.py` |
| `[custom]` | `types/chat/chat_completion_custom_tool_param.py`, `chat_completion_message_custom_tool_call.py` |
| `[rf]` | `types/shared_params/response_format_{text,json_object,json_schema}.py` |
| `[stream-opts]` | `types/chat/chat_completion_stream_options_param.py` |
| `[audio]` | `types/chat/chat_completion_audio_param.py`, `chat_completion_audio.py` |
| `[client]` | `_client.py` (auth header, status to exception map) |
| `[base]` | `_base_client.py` (retry rules) |
| `[sse]` | `_streaming.py` (SSE loop) |
| `[node-stream]` | `openai/openai-node`, `src/lib/ChatCompletionStream.ts` |
| `[vision]` | https://developers.openai.com/api/docs/guides/images-vision |
| `[files]` | https://developers.openai.com/api/docs/guides/file-inputs |
| `[fc]` | https://developers.openai.com/api/docs/guides/function-calling |
| `[reasoning]` | https://developers.openai.com/api/docs/guides/reasoning |
| `[errors]` | https://developers.openai.com/api/docs/guides/error-codes |
| `[ratelimits]` | https://developers.openai.com/api/docs/guides/rate-limits |
| `[stream-ref]` | https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events |
| `[spec]` | `openai/openai-openapi`, version 2.3.0 (stale; used for defaults and bounds that the SDK docstrings omit) |
| `[deepseek]` | https://api-docs.deepseek.com/guides/thinking_mode |
| `[vllm]` | https://github.com/vllm-project/vllm/issues/27755 and https://docs.vllm.ai/en/v0.14.0/features/reasoning_outputs/ |
| `[llamacpp]` | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md |
| `[live]` | Live test against api.openai.com on 2026-09-26 by another agent. A data point, not a source. |
| `[obs]` | Widely observed OpenAI behavior that no current primary document states. Treat as unverified. |

Every developers.openai.com guide is also available as markdown. Append `.md`
to the URL.

### Where openai-openapi 2.3.0 differs from the current SDK

| Item | openai-openapi 2.3.0 | Current SDK |
| --- | --- | --- |
| `reasoning_effort` values | `low`, `medium`, `high` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` `[effort]` |
| `stream_options.include_obfuscation`, chunk `obfuscation` | absent | present `[stream-opts]` `[chunk]` |
| `verbosity`, `safety_identifier`, `prompt_cache_key`, `prompt_cache_retention`, `prompt_cache_options`, `moderation` | absent | present `[params]` |
| `prompt_cache_breakpoint` on content parts | absent | present on text, image, audio, file parts `[part-text]` `[part-image]` |
| `usage.prompt_tokens_details` | `audio_tokens`, `cached_tokens` | adds `cache_write_tokens`, `image_tokens`, `text_tokens` `[usage]` |
| `usage.completion_tokens_details` | four fields | adds `text_tokens` `[usage]` |
| `tools` item types | `function` only, "max of 128 functions" | `function` or `custom`; the docstring no longer states a count limit `[params]` `[custom]` |
| `tool_choice` | string or named function | adds `allowed_tools` and named custom tool `[toolchoice]` |
| `service_tier` values | fewer | `auto`, `default`, `flex`, `scale`, `priority`, `fast` `[params]` |
| `image_url.detail` | `auto`, `low`, `high`, default `auto` | same literal in the SDK; the vision guide also lists `original` (see below) |

## Authentication and headers

- The client sends `Authorization: Bearer <api key>` `[client]`.
- The client can also send `OpenAI-Organization` and `OpenAI-Project` `[client]`.
  lgml can ignore these two headers.
- The request body is JSON with `Content-Type: application/json`.
- The SDK retries on status 408, 409, 429, and all 5xx. The header
  `x-should-retry: true|false` overrides that choice `[base]`.
- The SDK reads `retry-after-ms`, then `retry-after` (seconds or an HTTP date)
  `[base]`.

## Request body

### Top-level fields

"Req" means required. Bounds and defaults come from `[params]` unless marked.

| Field | Type | Req | Default | Notes |
| --- | --- | --- | --- | --- |
| `model` | string | yes | | Model ID. |
| `messages` | array of message | yes | | At least 1 item `[spec]`. |
| `stream` | bool or null | no | false | SSE output when true. |
| `stream_options` | object or null | no | null | Only valid with `stream: true`. See Streaming. |
| `max_completion_tokens` | int or null | no | model limit | Bound on visible plus reasoning tokens. |
| `max_tokens` | int or null | no | | Deprecated. Not compatible with o-series models `[params]`. |
| `temperature` | number or null | no | 1 | 0 to 2 `[spec]`. |
| `top_p` | number or null | no | 1 | 0 to 1 `[spec]`. |
| `n` | int or null | no | 1 | 1 to 128 `[spec]`. |
| `stop` | string, array of string, or null | no | null | Up to 4 sequences. The output does not contain the stop sequence. Not supported with latest reasoning models. |
| `presence_penalty` | number or null | no | 0 | -2.0 to 2.0. |
| `frequency_penalty` | number or null | no | 0 | -2.0 to 2.0. |
| `seed` | int or null | no | | Best-effort determinism. Beta. |
| `logit_bias` | map of token-id string to int | no | null | -100 to 100. |
| `logprobs` | bool or null | no | false | Return logprobs of output tokens. |
| `top_logprobs` | int or null | no | | 0 to 20. Needs `logprobs: true`. |
| `response_format` | object | no | `{"type":"text"}` | See Response format. |
| `tools` | array | no | | `function` or `custom` tools. |
| `tool_choice` | string or object | no | `none` without tools, `auto` with tools | See Tools. |
| `parallel_tool_calls` | bool | no | true `[spec]` | false means zero or one tool call per turn `[fc]`. |
| `reasoning_effort` | string or null | no | model-dependent | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` `[effort]`. |
| `verbosity` | string or null | no | `medium` | `low`, `medium`, `high`. |
| `modalities` | array or null | no | `["text"]` | `text`, `audio`. |
| `audio` | object or null | no | | Required when `modalities` has `audio`. `{format, voice}` `[audio]`. |
| `prediction` | object or null | no | | `{"type":"content","content": string or text parts}`. Predicted Outputs. |
| `web_search_options` | object | no | | `{search_context_size, user_location}`. Hosted web search. |
| `metadata` | map or null | no | | Up to 16 pairs. Key up to 64 chars, value up to 512 chars. |
| `store` | bool or null | no | | Store output for distillation or evals. |
| `service_tier` | string or null | no | `auto` | Processing tier. |
| `user` | string | no | | Deprecated. Replaced by `safety_identifier` and `prompt_cache_key`. |
| `safety_identifier` | string or null | no | | Up to 64 chars. Abuse detection. |
| `prompt_cache_key` | string or null | no | | Cache bucketing hint. |
| `prompt_cache_retention` | string or null | no | | Deprecated. `in_memory` or `24h`. |
| `prompt_cache_options` | object | no | | `{mode: implicit or explicit, ttl: "30m"}`. New for gpt-5.6 and later. |
| `moderation` | object or null | no | | `{model, policy:{input:{mode}, output:{mode}}}`. Moderated completions. |
| `functions` | array | no | | Deprecated. Replaced by `tools`. |
| `function_call` | string or object | no | | Deprecated. Replaced by `tool_choice`. |

Unknown fields: OpenAI answers 400 with a message such as
`Unrecognized request argument supplied: <name>` `[obs]`. A field that a model
does not support gets 400 with `code: "unsupported_parameter"` and the field
name in `param` `[obs]`. The reasoning guide confirms one case: `reasoning_effort:
"none"` on gpt-6-astra returns HTTP 400 `[reasoning]`.

### Message roles

`messages[]` is a union of six shapes `[params]`.

| Role | Fields | `content` type | Source |
| --- | --- | --- | --- |
| `developer` | `role`, `content` (req), `name` | string or array of text parts | `[msg-dev]` |
| `system` | `role`, `content` (req), `name` | string or array of text parts | `[msg-sys]` |
| `user` | `role`, `content` (req), `name` | string or array of text, image, audio, file parts | `[msg-user]` |
| `assistant` | `role`, `content`, `refusal`, `tool_calls`, `audio`, `name`, `function_call` | string, array of text and refusal parts, or null | `[msg-asst]` |
| `tool` | `role`, `content` (req), `tool_call_id` (req) | string or array of text parts | `[msg-tool]` |
| `function` | `role`, `content` (req, nullable), `name` (req) | string or null | `[msg-fn]` (deprecated) |

Rules:

- Assistant `content` is required unless `tool_calls` or `function_call` is
  present `[msg-asst]`. Clients send `"content": null` or omit it on a
  tool-call turn.
- Assistant `audio` is `{"id": "..."}`, a reference to an earlier audio reply
  `[msg-asst]`.
- Assistant `tool_calls[]` items echo the reply shape:
  `{"id","type":"function","function":{"name","arguments"}}` or the custom
  shape `[toolcall]` `[custom]`.
- `name` is an optional participant label `[msg-user]`.
- An assistant message with `tool_calls` must be followed by `tool` messages
  that answer each `tool_call_id`. OpenAI rejects the request with 400
  otherwise `[obs]`.

#### developer versus system

- `developer` messages are "Developer-provided instructions that the model
  should follow, regardless of messages sent by the user. With o1 models and
  newer, `developer` messages replace the previous `system` messages"
  `[msg-dev]`.
- `system` has the same shape and the same meaning. The SDK tells users to
  prefer `developer` on o1 and newer models `[msg-sys]`.
- OpenAI accepts both roles on current models `[obs]`.
- lgml should treat both roles the same: map each to the chat template's system
  slot. Keep the message order. If a template allows only one leading system
  block, join all leading `system` and `developer` messages with a blank line.
  If a `system` or `developer` message appears after the first non-system
  message and the template cannot render it, return 400. Do not drop it.

### Content parts

A string `content` equals one text part. An array holds typed parts.

#### text

```json
{"type": "text", "text": "Describe this image."}
```

`text` is required `[part-text]`. An optional `prompt_cache_breakpoint:
{"mode":"explicit"}` marks a cache boundary `[part-text]`. lgml can accept and
ignore this hint because it does not change the output.

#### image_url

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,iVBORw0KGgo...",
    "detail": "auto"
  }
}
```

- `image_url.url` is required. It is "Either a URL of the image or the base64
  encoded image data" `[part-image]`. In practice the value is an `https://`
  URL or a data URL `data:<media type>;base64,<data>` `[vision]`.
- `image_url.detail` is optional: `auto`, `low`, `high` `[part-image]`. The
  default is `auto` in both APIs `[vision]`.
- The vision guide also lists `original` for some models (gpt-5.4 and later)
  and does not restrict it to the Responses API `[vision]`. The SDK literal for
  Chat does not include `original` `[part-image]`. Status: unresolved. lgml
  should accept `original` and treat it as "no downscale beyond the model's
  pixel limit".
- The `image_url` part is valid only in `user` messages per the SDK types
  `[msg-user]`.
- Image parts must not be sent to `system`, `developer`, `assistant`, or `tool`
  messages per the SDK types. See "Images in tool results" for the live
  behavior.

Supported image files `[vision]`:

- PNG (`image/png`), JPEG (`image/jpeg`, `.jpg` or `.jpeg`), WEBP
  (`image/webp`), and non-animated GIF (`image/gif`).

Limits `[vision]`:

- Up to 512 MB total payload per request.
- Up to 1,500 images per request.
- For patch-based models, up to 30,000 patches (32x32 px) per image after
  resize. An image over that limit is rejected, not resized.
- Image tokens and the rest of the prompt must fit the model's context.
- `store: true` drops image inputs over 8 MB from stored data `[params]`.

Detail levels and resize rules, per the model sizing table `[vision]`:

| Model family | Detail levels | Resize behavior |
| --- | --- | --- |
| gpt-6-astra | low, high, original, auto | low: fit 512x512. high: 2,500 patches, 65,535 px max side. original: keep size up to 65,535 px. auto = original. |
| gpt-5.6-sol, -terra, -luna | low, high, original, auto | low: fit 512x512. high: fit 2048x2048 and 2,500 patches. original: keep size up to 65,535 px. auto = original. |
| gpt-5.5 | low, high, original, auto | low: fit 512x512. high: 2,500 patches, 2048 px max. original: 10,000 patches, 6000 px max. auto = original. |
| gpt-5.4, -mini, -nano | low, high, original, auto | low: 2048 px max, 6,144 patches. high: 2,500 patches, 2048 px max. original: 10,000 patches, 6000 px max. auto = high. |
| gpt-5.2, gpt-4.1-mini | low, high, auto | All: 2048 px max, 6,144 patches. No original. |
| gpt-5.1, gpt-4.1, gpt-4o, gpt-4o-mini | low, high, auto | low: fixed tokens. high, auto: tile rules. |

Image token accounting `[vision]`:

- Patch-based models: fit the image in the pixel limit (no upscale). Compute
  `patches = ceil(w/32) * ceil(h/32)`. If over the patch budget, scale by
  `sqrt(32^2 * budget / (w*h))`, adjusted so the patch grid fits, then floor
  the dimensions. Tokens are `ceil(patches * multiplier)`. The multiplier is
  1.2 for gpt-6-astra, gpt-5.6, gpt-5.5, gpt-5.4, gpt-5.2; 1.62 for
  gpt-4.1-mini.
- Example: gpt-6-astra, `detail: high`, 1024x1024 gives 1024 patches and
  `ceil(1024 * 1.2) = 1229` tokens.
- Tile-based models: fit in 2048x2048; if the short side exceeds 768, scale it
  to 768; count 512 px tiles. Tokens are `base + tiles * tile_tokens`. With
  `low`, tokens are `base` only. Base/tile: gpt-5.1 70/140, gpt-4o and gpt-4.1
  85/170, gpt-4o-mini 2833/5667.
- Image tokens count in `usage.prompt_tokens`. The current SDK also has
  `usage.prompt_tokens_details.image_tokens` `[usage]`.

For lgml the numbers above are OpenAI's billing rules, not a requirement.
lgml reports the real number of positions its vision encoder puts into the
context.

Remote URLs: OpenAI fetches `https://` URLs itself. A local server that
fetches remote URLs opens a request-forgery path. lgml can support data URLs
only and answer a remote URL with 400 (see "What lgml needs"). llama.cpp also
accepts raw base64 without the `data:` prefix and local file paths
`[llamacpp]`. Those forms are non-standard.

#### input_audio

```json
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
```

`format` is `wav` or `mp3` `[part-audio]`. User messages only.

#### file

```json
{"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,..."}}
```

- `file.file_id` (an uploaded file ID), `file.file_data` (base64 data), and
  `file.filename` are all optional in the type `[part]`.
- Chat Completions accepts only PDF (`application/pdf`) for both `file_data`
  and `file_id`. `file_url` is not supported in Chat `[files]`.
- Chat file parts do not support `detail` `[files]`.
- Each file must be under 50 MB; all files in one request total 50 MB `[files]`.
- In the guide examples, `file_data` is a full data URL
  (`data:application/pdf;base64,...`) `[files]`. The SDK docstring says
  "base64 encoded file data" `[part]`.

#### refusal (assistant history only)

```json
{"type": "refusal", "refusal": "I can't help with that."}
```

Valid in assistant `content` arrays `[msg-asst]`.

### Images in tool results

- Per the SDK, `tool` message `content` is a string or an array of
  `ChatCompletionContentPartTextParam` only `[msg-tool]`. openai-openapi 2.3.0
  agrees: `ChatCompletionRequestToolMessageContentPart` is text only `[spec]`.
- Chat has no image output part in a tool result. The Responses API accepts
  image and file arrays in `function_call_output` `[fc]`; Chat does not.
- Data point `[live]`: OpenAI accepted an `image_url` part inside a
  `role: "tool"` message with no error, but the model did not see the image.
- Data point `[live]`: a `user` message that carries the image, placed after
  the complete batch of tool messages, works. This document calls it the
  "carrier" message.

The carrier pattern:

1. The assistant message has `tool_calls`.
2. One `tool` message follows for each `tool_call_id`, with text content (for
   example "image attached in next message").
3. After the last `tool` message, one `user` message holds a text part that
   names the tool call and the `image_url` part.

The carrier must not come between tool messages. It must come after all
answers for that assistant turn, because each `tool_call_id` needs an answer
right after the assistant message `[obs]`.

## Tools

### Function tool definition

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"],
      "additionalProperties": false
    },
    "strict": true
  }
}
```

- `name` is required: `a-z`, `A-Z`, `0-9`, `_`, `-`, up to 64 chars `[fndef]`.
- `description` is optional `[fndef]`.
- `parameters` is a JSON Schema object. If absent, the function takes no
  parameters `[fndef]`.
- `strict: true` forces exact schema adherence with a subset of JSON Schema
  `[fndef]`. Strict requires `additionalProperties: false` on every object and
  every property in `required` `[fc]`. OpenAI rejects a strict schema that
  breaks these rules `[fc]`.
- Chat Completions is non-strict by default when `strict` is absent `[fc]`.
- Custom tools: `{"type":"custom","custom":{"name","description","format"}}`.
  `format` is `{"type":"text"}` or `{"type":"grammar","grammar":{"syntax":
  "lark"|"regex","definition"}}` `[custom]`.
- gpt-6-astra does not support tool calls on Chat Completions `[fc]`
  `[reasoning]`.

### tool_choice

Values `[toolchoice]`:

- `"none"`: no tool call; the model writes a message.
- `"auto"`: the model picks a message or one or more tool calls.
- `"required"`: the model must call one or more tools.
- `{"type":"function","function":{"name":"get_weather"}}`: force that tool.
- `{"type":"custom","custom":{"name":"..."}}`: force a custom tool.
- `{"type":"allowed_tools","allowed_tools":{"mode":"auto"|"required","tools":[{"type":"function","function":{"name":"..."}}]}}`:
  restrict the choice to a subset.

Default: `none` if no tools are present, `auto` if tools are present `[params]`.

### parallel_tool_calls

- Default true `[spec]`.
- false means "exactly zero or one tool is called" `[fc]`.

### Tool calls in the reply

```json
{
  "role": "assistant",
  "content": null,
  "refusal": null,
  "tool_calls": [
    {
      "id": "call_Jd8x2Qm4",
      "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"city\":\"Boston\"}"}
    }
  ]
}
```

- `id` is a string. OpenAI IDs start with `call_` (for example `call_abc123`
  in `[spec]`, `call_12345xyz` in `[fc]`). openai-node makes a fallback ID
  `call_<uuid4>` when the stream has none `[node-stream]`.
- `function.arguments` is a JSON-encoded string, not an object. The model can
  produce invalid JSON; the client must validate `[toolcall]`.
- `finish_reason` is `tool_calls` when the model called a tool `[completion]`.
- A custom call is `{"id","type":"custom","custom":{"name","input"}}` `[custom]`.

### Tool result message

```json
{"role": "tool", "tool_call_id": "call_Jd8x2Qm4", "content": "{\"temp_f\":61}"}
```

`content` is a string or an array of text parts `[msg-tool]`.

## Response object (non-streaming)

```json
{
  "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
  "object": "chat.completion",
  "created": 1741569952,
  "model": "gpt-4.1-2025-04-14",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
        "refusal": null,
        "annotations": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 10,
    "total_tokens": 29,
    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "service_tier": "default"
}
```

Example from `[spec]`. Fields `[completion]`:

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string | Prefix `chatcmpl-` in all OpenAI examples `[spec]`. The type does not require the prefix. |
| `object` | `"chat.completion"` | Always this value. |
| `created` | int | Unix seconds. |
| `model` | string | The model that served the request. |
| `choices` | array | One per `n`. |
| `usage` | object or null | See Usage. |
| `system_fingerprint` | string or null | Backend config fingerprint. |
| `service_tier` | string or null | Tier used. |
| `metadata` | map or null | Echo of request metadata. |
| `moderation` | object or null | Only with moderated completions. |

`choices[]`: `index`, `message`, `finish_reason`, `logprobs` `[completion]`.

`message` fields `[message]`:

- `role`: always `"assistant"`.
- `content`: string or null. Plain string only; Chat never returns content
  parts or images in the reply.
- `refusal`: string or null.
- `tool_calls`: array or absent.
- `annotations`: array of `{"type":"url_citation","url_citation":{"start_index","end_index","title","url"}}`
  (web search only).
- `audio`: `{"id","data","expires_at","transcript"}` when audio output was
  requested `[audio]`.
- `function_call`: deprecated.

`finish_reason` values `[completion]`: `stop` (natural end or stop sequence),
`length` (token limit), `tool_calls`, `content_filter`, `function_call`
(deprecated). No other value is valid in the SDK type.

`logprobs` `[completion]` `[logprob]`:

```json
{
  "content": [
    {
      "token": "Hello",
      "logprob": -0.0012,
      "bytes": [72, 101, 108, 108, 111],
      "top_logprobs": [
        {"token": "Hello", "logprob": -0.0012, "bytes": [72, 101, 108, 108, 111]},
        {"token": "Hi", "logprob": -6.9, "bytes": [72, 105]}
      ]
    }
  ],
  "refusal": null
}
```

- `bytes` is the UTF-8 byte list, or null.
- `logprob` is `-9999.0` for a token outside the top 20.
- `top_logprobs` can have fewer entries than requested.

## Usage

`[usage]`:

| Field | Req | Meaning |
| --- | --- | --- |
| `prompt_tokens` | yes | All prompt tokens, including cached, image, and audio tokens. |
| `completion_tokens` | yes | All generated tokens, including reasoning tokens. |
| `total_tokens` | yes | `prompt_tokens + completion_tokens`. |
| `prompt_tokens_details.cached_tokens` | no | Prompt tokens served from cache (a subset of `prompt_tokens`). |
| `prompt_tokens_details.audio_tokens` | no | Audio input tokens. |
| `prompt_tokens_details.image_tokens` | no | Image input tokens. New in the current SDK. |
| `prompt_tokens_details.text_tokens` | no | Text input tokens. New. |
| `prompt_tokens_details.cache_write_tokens` | no | Prompt tokens written to cache. New. |
| `completion_tokens_details.reasoning_tokens` | no | Hidden reasoning tokens. |
| `completion_tokens_details.audio_tokens` | no | Audio output tokens. |
| `completion_tokens_details.accepted_prediction_tokens` | no | Predicted Outputs hits. |
| `completion_tokens_details.rejected_prediction_tokens` | no | Predicted Outputs misses; still billed. |
| `completion_tokens_details.text_tokens` | no | Text output tokens. New. |

## Response format

`[rf]`:

- `{"type":"text"}`: default.
- `{"type":"json_object"}`: JSON mode. The output is valid JSON. The prompt
  must ask for JSON.
- `{"type":"json_schema","json_schema":{"name","description","schema","strict"}}`:
  Structured Outputs. `name` is required (same charset and 64-char limit as
  function names). `strict: true` forces exact schema adherence.

## Streaming

### Request

- `stream: true`.
- `stream_options.include_usage: true` adds one chunk before `[DONE]`. That
  chunk has `usage` set and `choices: []`. The other chunks have `usage: null`
  `[stream-opts]`.
- `stream_options.include_obfuscation`: OpenAI adds an `obfuscation` string to
  each chunk by default to hide payload sizes. false turns it off
  `[stream-opts]` `[chunk]`. lgml does not need to emit `obfuscation`; clients
  ignore it.
- `stream_options` is valid only with `stream: true` `[params]`.

### Framing

- `Content-Type: text/event-stream`.
- Each event is one line `data: <json>` followed by a blank line, so the bytes
  are `data: {...}\n\n` `[stream-ref]` `[spec]`.
- The stream ends with `data: [DONE]\n\n` `[stream-opts]`. openai-python stops
  when the data starts with `[DONE]` `[sse]`.
- OpenAI does not use `event:` lines for Chat chunks. A line that starts with
  `:` is an SSE comment. SSE parsers skip it, so lgml can send `: keepalive\n\n`
  during long prompt processing.
- An error after the stream starts: send `data: {"error":{...}}\n\n`.
  openai-python raises `APIError` for any event whose JSON has a truthy
  `error` key `[sse]`.

### Chunk object

`[chunk]` `[stream-ref]`:

| Field | Notes |
| --- | --- |
| `id` | Same for all chunks of one completion. |
| `object` | `"chat.completion.chunk"`. |
| `created` | Same for all chunks. |
| `model` | Model ID. |
| `system_fingerprint` | Optional. |
| `service_tier` | Optional. |
| `choices[]` | `{index, delta, finish_reason, logprobs}`. Empty in the usage chunk. |
| `usage` | null, except in the final usage chunk. |
| `obfuscation` | Optional padding string. |

`delta` fields: `role`, `content`, `refusal`, `tool_calls`, `function_call`
`[chunk]`.

Chunk order for one choice:

1. The first chunk has `delta.role: "assistant"`. OpenAI examples also set
   `content: ""` `[spec]`. On a tool-call turn OpenAI sends `content: null`
   and `refusal: null` `[obs]`.
2. Content chunks have `delta.content` with a text fragment.
3. Tool-call chunks have `delta.tool_calls[]` items. Each item has `index`
   (required). The first fragment for an index has `id`, `type: "function"`,
   `function.name`, and `function.arguments: ""`. Later fragments for that
   index have only `index` and `function.arguments` `[chunk]` `[obs]`.
4. A chunk with `delta: {}` and `finish_reason` set ends the choice `[spec]`.
5. With `include_usage`, a chunk with `choices: []` and `usage`.
6. `data: [DONE]`.

Client requirements from the openai-node stream helper `[node-stream]`:

- Each choice must get a `finish_reason`, or the helper throws `missing
  finish_reason for choice N`.
- Each choice must get `role`, or it throws `missing role`.
- Each tool call must get `type` and `function.name` at least once, or it
  throws.
- `function.name` is assigned, not appended. Send the whole name in one
  fragment. `arguments` is appended.
- A changed `type` or `name` for the same index is an error.
- `index` must be a non-negative safe integer.

## Errors

Shape `[spec]` (`Error` schema; all four keys required, `code` and `param`
nullable):

```json
{
  "error": {
    "message": "Invalid value for 'temperature': must be between 0 and 2.",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_value"
  }
}
```

The SDK reads `body["error"]` when present, else the whole body `[client]`.

Status codes. The SDK exception per status comes from `[client]`; the cases
come from `[errors]`.

| Status | SDK exception | Use |
| --- | --- | --- |
| 400 | `BadRequestError` | Malformed request, bad or unsupported parameter, context too long. |
| 401 | `AuthenticationError` | Missing or bad API key. |
| 403 | `PermissionDeniedError` | Not allowed (for example region). |
| 404 | `NotFoundError` | Unknown model or path. |
| 409 | `ConflictError` | Resource conflict. Retried by the SDK. |
| 422 | `UnprocessableEntityError` | Well-formed but not processable. |
| 429 | `RateLimitError` | Rate limit or quota. Retried by the SDK. |
| 500 | `InternalServerError` | Server error. Retried. |
| 503 | `InternalServerError` | `type: "service_unavailable_error"`, `code: "server_is_overloaded"`. Retried. |

Known `type` and `code` values:

- `rate_limit_error` with code `slow_down` (429) `[errors]`.
- 429 codes: `credit_balance_exhausted`, `organization_spend_limit_exceeded`,
  `project_spend_limit_exceeded`, `organization_usage_limit_exceeded`; the type
  can be `insufficient_quota` `[errors]`.
- `service_unavailable_error` with code `server_is_overloaded` (503)
  `[errors]`.
- `invalid_request_error` with codes such as `invalid_value`,
  `unsupported_parameter`, `context_length_exceeded`, `model_not_found`,
  `invalid_api_key` `[obs]`.
- Clients should follow `Retry-After` on 429 and 503 `[errors]`. Rate-limit
  headers include `x-ratelimit-limit-requests`, `x-ratelimit-remaining-tokens`,
  and more `[ratelimits]`.

## Reasoning

- `reasoning_effort` sets reasoning depth. Values: `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, `max`. Not every model supports every value
  `[effort]` `[reasoning]`.
- Defaults are model-dependent (for example `medium` for gpt-5.5 and gpt-5.6)
  `[reasoning]`.
- `max_completion_tokens` bounds visible plus reasoning tokens `[params]`. The
  reply can reach `length` before any visible text.
- OpenAI Chat Completions does not return reasoning text. Only the count
  appears, in `usage.completion_tokens_details.reasoning_tokens` `[usage]`
  `[reasoning]`. Reasoning summaries exist only in the Responses API
  `[reasoning]`.
- Non-standard `reasoning_content`: DeepSeek returns the reasoning text in
  `message.reasoning_content` and `delta.reasoning_content` `[deepseek]`.
  llama.cpp does the same with `--reasoning-format deepseek` `[llamacpp]`.
  vLLM renamed the field to `reasoning` and keeps `reasoning_content` for
  compatibility `[vllm]`. None of these fields exist in the OpenAI SDK types.
- DeepSeek with `tools`: clients must send `reasoning_content` of earlier
  turns back in assistant messages, or DeepSeek returns 400. Without `tools`,
  DeepSeek ignores it `[deepseek]`.

## Other fields: what they are

| Field | Meaning | Changes output? |
| --- | --- | --- |
| `metadata` | Tags for stored completions. | No |
| `store` | Keep the completion for evals or distillation. | No |
| `user`, `safety_identifier` | End-user ID for abuse detection. | No |
| `prompt_cache_key`, `prompt_cache_retention`, `prompt_cache_options`, part `prompt_cache_breakpoint` | Cache hints. | No |
| `service_tier` | Processing tier and price. | No |
| `modalities`, `audio` | Audio output. | Yes |
| `prediction` | Predicted Outputs for faster regeneration. | Speed only; rejected tokens are billed. |
| `verbosity` | Output length hint. | Yes |
| `web_search_options` | Hosted web search. Adds `annotations`. | Yes |
| `moderation` | Moderated completions. Adds `moderation` to the reply. | Yes |
| `logit_bias` | Token bias by token ID. IDs are tokenizer-specific. | Yes |
| `seed` | Best-effort determinism. | Yes |
| `n` | Number of choices. | Yes |

## The clm client

The clm client uses Chat Completions as its internal format. It sends and reads
the fields below.

Sends:

- Header `Authorization: Bearer <key>`, only when a key is set.
- Header `Content-Type: application/json`, always.
- `model`, `messages`, `stream` (true or false).
- `stream_options: {"include_usage": true}` when streaming.
- `reasoning_effort` (string) when the user sets an effort. The client
  assumes that servers without reasoning accept and ignore it.
- `tools`: `[{"type":"function","function":{"name","description","parameters"}}]`.
  `description` can be `""`. `parameters` defaults to `{}` when the tool has no
  schema, which has no `type` key. The client never sends `strict`.
- `parallel_tool_calls: false`, only when serial dispatch is configured, and
  only with `tools`.
- `tool_choice: "none"` when it must forbid tool calls.
- Messages: roles `system`, `user`, `assistant`, `tool`. `content` is always a
  plain string. Assistant turns with tool calls send `"content": null` and
  `tool_calls[]` with `id`, `type: "function"`, `function.name`,
  `function.arguments`. Tool results send `tool_call_id` and a string
  `content`. The client does not send image parts or `developer` messages.

Reads (non-streaming):

- `choices[0].message.content`, `.tool_calls[]`, `.reasoning_content`, then
  `.reasoning` as a fallback.
- `choices[0].finish_reason`: acts on `content_filter` (fails the turn) and on
  the non-standard `error`.
- Top-level `error.message` in a 200 body or in any chunk.
- `usage.prompt_tokens`, `completion_tokens`, `total_tokens`,
  `prompt_tokens_details.cached_tokens`. Also non-standard
  `usage.cache_read_tokens`, `usage.cache_write_tokens`, and llama.cpp
  `timings.predicted_per_second`.

Reads (streaming):

- Lines that start with `data:`; other lines are skipped. A trailing `\r` is
  removed. `[DONE]` is skipped; the end of the HTTP body ends the stream.
- `choices[0]` only (no `n > 1`).
- `delta.content`, `delta.reasoning_content` or `delta.reasoning`,
  `delta.tool_calls[]`.
- Tool-call merge: keyed by `index` (array position if `index` is absent).
  The first `id` and first `name` win. `arguments` fragments are appended.
- `finish_reason`: the last non-null value wins.
- `usage`: the first chunk with a `usage` object wins. A null `usage` is
  skipped.

Error handling:

- Non-2xx: prints `error.message` from the body, else the raw body.
- 429: waits and retries. The delay comes from the text `try again in <N>s`
  or `<N>ms` in the body. The client does not read `Retry-After`.
- 400 whose body contains `does not support tools`: resends without `tools`
  and stops sending tools for the session.

Other endpoints: `GET /v1/models` (reads `data[].id`, `context_length`,
`top_provider.context_length`) and llama.cpp `GET /props`.

## Examples

### 1. Plain text

Request:

```json
{
  "model": "qwen3-vl-8b",
  "messages": [
    {"role": "developer", "content": "Answer in one sentence."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_completion_tokens": 64
}
```

Response:

```json
{
  "id": "chatcmpl-7a1f0c9e2b",
  "object": "chat.completion",
  "created": 1790400000,
  "model": "qwen3-vl-8b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "The capital of France is Paris.", "refusal": null},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 24, "completion_tokens": 8, "total_tokens": 32}
}
```

### 2. Image by data URL

Request:

```json
{
  "model": "qwen3-vl-8b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What color is the square?"},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP4z8CAFWEVHbQSACj/P8Fu7N9hAAAAAElFTkSuQmCC", "detail": "low"}
        }
      ]
    }
  ]
}
```

Response (the image tokens are inside `prompt_tokens`):

```json
{
  "id": "chatcmpl-0b77d1",
  "object": "chat.completion",
  "created": 1790400010,
  "model": "qwen3-vl-8b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "The square is red.", "refusal": null},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 96,
    "completion_tokens": 5,
    "total_tokens": 101,
    "prompt_tokens_details": {"cached_tokens": 0, "image_tokens": 80}
  }
}
```

### 3. Tool call round trip

Request 1:

```json
{
  "model": "qwen3-vl-8b",
  "messages": [{"role": "user", "content": "Weather in Boston?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ]
}
```

Response 1:

```json
{
  "id": "chatcmpl-5c2e",
  "object": "chat.completion",
  "created": 1790400020,
  "model": "qwen3-vl-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "refusal": null,
        "tool_calls": [
          {
            "id": "call_q8Zb3LwN",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Boston\"}"}
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {"prompt_tokens": 70, "completion_tokens": 18, "total_tokens": 88}
}
```

Request 2 (history plus the tool result; `tools` sent again):

```json
{
  "model": "qwen3-vl-8b",
  "messages": [
    {"role": "user", "content": "Weather in Boston?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {"id": "call_q8Zb3LwN", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\":\"Boston\"}"}}
      ]
    },
    {"role": "tool", "tool_call_id": "call_q8Zb3LwN", "content": "{\"temp_f\":61,\"sky\":\"clear\"}"}
  ],
  "tools": ["...same as request 1..."]
}
```

Response 2: a normal text reply with `finish_reason: "stop"`.

### 4. Tool result with an image (carrier user message)

The tool `screenshot` returns an image. The `tool` message carries text only.
A `user` message after the tool batch carries the image.

```json
{
  "model": "qwen3-vl-8b",
  "messages": [
    {"role": "user", "content": "Take a screenshot and read the error dialog."},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {"id": "call_A1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}},
        {"id": "call_B2", "type": "function", "function": {"name": "list_windows", "arguments": "{}"}}
      ]
    },
    {"role": "tool", "tool_call_id": "call_A1", "content": "Screenshot captured (1280x800). The image follows in the next user message."},
    {"role": "tool", "tool_call_id": "call_B2", "content": "[\"Terminal\",\"Error\"]"},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Image output of tool call call_A1 (screenshot):"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
      ]
    }
  ],
  "tools": ["...screenshot, list_windows..."]
}
```

Do not use this form. The SDK types forbid it, and OpenAI ignores the image
`[msg-tool]` `[live]`:

```json
{"role": "tool", "tool_call_id": "call_A1", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}
```

### 5. Streaming with a tool call

Request:

```json
{
  "model": "qwen3-vl-8b",
  "stream": true,
  "stream_options": {"include_usage": true},
  "messages": [{"role": "user", "content": "Weather in Boston?"}],
  "tools": ["...get_weather..."]
}
```

Response (`Content-Type: text/event-stream`):

```text
data: {"id":"chatcmpl-9f3","object":"chat.completion.chunk","created":1790400030,"model":"qwen3-vl-8b","choices":[{"index":0,"delta":{"role":"assistant","content":null,"refusal":null},"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-9f3","object":"chat.completion.chunk","created":1790400030,"model":"qwen3-vl-8b","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_q8Zb3LwN","type":"function","function":{"name":"get_weather","arguments":""}}]},"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-9f3","object":"chat.completion.chunk","created":1790400030,"model":"qwen3-vl-8b","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-9f3","object":"chat.completion.chunk","created":1790400030,"model":"qwen3-vl-8b","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Boston\"}"}}]},"logprobs":null,"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-9f3","object":"chat.completion.chunk","created":1790400030,"model":"qwen3-vl-8b","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}],"usage":null}

data: {"id":"chatcmpl-9f3","object":"chat.completion.chunk","created":1790400030,"model":"qwen3-vl-8b","choices":[],"usage":{"prompt_tokens":70,"completion_tokens":18,"total_tokens":88}}

data: [DONE]

```

A second parallel call uses `index: 1` with its own `id`.

### 6. Reasoning reply

Request:

```json
{
  "model": "qwen3-vl-8b-thinking",
  "reasoning_effort": "high",
  "max_completion_tokens": 4096,
  "messages": [{"role": "user", "content": "Is 221 prime?"}]
}
```

Response from OpenAI (no reasoning text; only the count):

```json
{
  "id": "chatcmpl-r1",
  "object": "chat.completion",
  "created": 1790400040,
  "model": "gpt-5.6-terra",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "No. 221 = 13 x 17.", "refusal": null},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 212,
    "total_tokens": 226,
    "completion_tokens_details": {"reasoning_tokens": 198}
  }
}
```

The same reply from a DeepSeek-style server (non-standard field):

```json
"message": {
  "role": "assistant",
  "reasoning_content": "221 = 13 * 17 because 13 * 17 = 221 ...",
  "content": "No. 221 = 13 x 17."
}
```

Streaming form of the non-standard field:
`data: {..."choices":[{"index":0,"delta":{"reasoning_content":"221 = 13"},...}]}`.

### 7. Error

```text
HTTP/1.1 400 Bad Request
Content-Type: application/json

{"error":{"message":"Unsupported parameter: 'web_search_options' is not supported by this server.","type":"invalid_request_error","param":"web_search_options","code":"unsupported_parameter"}}
```

## What lgml needs

### Required behavior

- [ ] Accept `POST /v1/chat/completions` with `Content-Type: application/json`.
- [ ] Accept `Authorization: Bearer <key>`. When lgml has no key configured,
      accept requests with or without the header. When a key is configured,
      answer a bad or missing key with 401.
- [ ] Parse `content` as a string or as an array of parts, for every role.
- [ ] Accept `system` and `developer` with the same meaning. Merge leading
      system and developer messages when the template needs one block.
- [ ] Accept `image_url` parts in `user` messages. Support data URLs with
      `image/png`, `image/jpeg`, `image/webp`, and `image/gif` (first frame, or
      400 for animated GIF). Accept `detail` values `auto`, `low`, `high`,
      `original`, and map each to a resize policy.
- [ ] Accept the carrier pattern: a `user` message with images directly after
      a batch of `tool` messages.
- [ ] Decide one policy for image parts in `tool` messages and document it.
      Recommended: accept them as an lgml extension and render them into the
      prompt (after the tool text) when the template supports it; return 400
      when it does not. Do not accept and then drop them, as OpenAI does.
- [ ] Validate that each assistant `tool_calls[].id` has one `tool` message
      after it, and that each `tool_call_id` matches an earlier call. Answer a
      mismatch with 400.
- [ ] Accept assistant history with `content: null`, with `tool_calls`, with
      content as text parts, and with `refusal`.
- [ ] Accept non-standard `reasoning_content` (and `reasoning`) on assistant
      history messages. Feed it to templates that render earlier reasoning.
- [ ] Parse `tools` (function type), `tool_choice` (`none`, `auto`,
      `required`, named function, `allowed_tools`), and `parallel_tool_calls`.
      Enforce them in the grammar or sampler. `none` must produce no tool
      call.
- [ ] Emit tool calls with `id` prefix `call_`, `type: "function"`, and
      `arguments` as a JSON string. Set `finish_reason: "tool_calls"`.
- [ ] Honor `strict: true` with grammar-constrained decoding, or return 400.
- [ ] Honor `max_completion_tokens`, and `max_tokens` as an alias. If both are
      present, use `max_completion_tokens`.
- [ ] Honor `temperature`, `top_p`, `stop` (string or up to 4), `seed`,
      `presence_penalty`, `frequency_penalty`. Validate ranges and answer 400
      out of range.
- [ ] Honor `response_format` `json_object` and `json_schema` with a grammar.
- [ ] Map `reasoning_effort` to the model's thinking control. For a model with
      no thinking control, return 400 with `param: "reasoning_effort"`
      (OpenAI does this for non-reasoning models) or document one fixed
      mapping. Report `completion_tokens_details.reasoning_tokens`.
- [ ] Return reasoning text in `message.reasoning_content` and
      `delta.reasoning_content` (llama.cpp and DeepSeek name; clm reads it
      first). Keep it out of `content`.
- [ ] Non-streaming reply: `id` with `chatcmpl-` prefix, `object:
      "chat.completion"`, `created`, `model`, `choices[].index`,
      `message.role: "assistant"`, `message.content` (string or null),
      `message.refusal: null`, `logprobs: null` when not requested,
      `finish_reason`, and `usage`.
- [ ] `finish_reason` only from `stop`, `length`, `tool_calls`,
      `content_filter`. Use `length` when the token limit or the context
      limit ends the output.
- [ ] Streaming: `text/event-stream`, `data: <json>\n\n` events, final
      `data: [DONE]\n\n`. First chunk has `delta.role: "assistant"`. Every
      choice gets one chunk with `finish_reason`. Tool-call deltas carry
      `index`; the first fragment carries `id`, `type`, and the full `name`.
- [ ] `stream_options.include_usage`: one extra chunk with `choices: []` and
      `usage`, before `[DONE]`. Omit `usage` or set it to null in all other
      chunks. clm keeps the first `usage` object it sees.
- [ ] `usage.prompt_tokens` includes image tokens. Add
      `prompt_tokens_details.cached_tokens` for KV-cache prefix reuse and
      `prompt_tokens_details.image_tokens` when known.
- [ ] Errors as `{"error":{"message","type","param","code"}}` with all four
      keys (`param` and `code` can be null). Status: 400 bad request, 401 auth,
      404 unknown model, 413 or 400 for a too-large body, 429 busy queue,
      500 internal, 503 overloaded or model loading.
- [ ] On 429 and 503, send `Retry-After` and put `try again in <N>s` in
      `error.message` (clm reads only the message text).
- [ ] Mid-stream errors: send `data: {"error":{...}}\n\n`, then close.
- [ ] For a model without tool support, a request with `tools` gets 400 with a
      message that contains `does not support tools` (clm retries without
      tools on that text).
- [ ] `GET /v1/models` with `data[].id` and `context_length`.

### Accept as no-ops (they do not change the generated tokens)

`user`, `safety_identifier`, `metadata`, `store` (false or true; lgml stores
nothing), `prompt_cache_key`, `prompt_cache_retention`,
`prompt_cache_options`, part-level `prompt_cache_breakpoint`, `service_tier`
(echo `"default"` or omit), `stream_options.include_obfuscation`, `name` on
messages (or render it when the template supports names). The headers
`OpenAI-Organization` and `OpenAI-Project`.

### Reject with 400 when unsupported (never ignore)

Use `type: "invalid_request_error"`, `code: "unsupported_parameter"` or
`"unsupported_value"`, and set `param`.

- `n` greater than 1 (if lgml does not implement it).
- `logprobs: true` and `top_logprobs` (if not implemented).
- `logit_bias` (if not implemented; the IDs are tokenizer-specific anyway).
- `modalities` containing `audio`, and `audio`.
- `input_audio` content parts (for a model without audio input).
- `file` content parts (PDF), unless lgml renders PDF pages.
- `image_url` with an `http://` or `https://` URL, unless remote fetch is
  enabled. Also any image part sent to a model without a vision encoder.
- Image media types other than PNG, JPEG, WEBP, GIF. Malformed base64.
- `prediction`.
- `web_search_options`.
- `moderation`.
- `verbosity` (unless mapped to a prompt or template control).
- `custom` tools and a named custom `tool_choice`.
- `functions` and `function_call` (deprecated), unless lgml maps them to
  `tools` and `tool_choice`.
- `response_format` `json_schema` or `strict: true` tools, if the grammar
  engine cannot compile the schema. Say which keyword failed.
- `reasoning_effort` values the model cannot map.
- `stream_options` without `stream: true`.
- Any unknown top-level field.

## Pitfalls

- `function.arguments` is a string with JSON in it. An object breaks
  openai-python, openai-node, and clm.
- In a stream, send the tool name in one fragment. openai-node assigns
  `name`; it does not append it.
- Every streamed choice needs a `finish_reason` chunk and a `role`, or the
  openai-node helper throws.
- The usage chunk has `choices: []`. Clients that read `choices[0]` without a
  check fail on it; lgml sends it only when `include_usage` is true.
- clm keeps the first `usage` object in a stream. Do not send running totals
  in earlier chunks.
- The carrier `user` message must come after all `tool` messages of the batch.
  A `user` message between tool messages breaks the pairing rule.
- Tool message content arrays hold text parts per the SDK. A server that
  silently drops image parts there repeats OpenAI's behavior and hides the
  image from the model.
- `system` versus `developer`: some chat templates accept only one system
  message at index 0. Merge, do not drop.
- `max_tokens` and `max_completion_tokens` both appear in the wild. Reasoning
  tokens count against the limit, so a reasoning model can hit `length` with
  empty `content`.
- `reasoning_content` is not OpenAI. vLLM now uses `reasoning`. Pick one name
  for output; accept both on input.
- `detail: "original"` is in the vision guide but not in the SDK literal.
  Accept it.
- `[DONE]` is not JSON. Send it exactly as `data: [DONE]`.
- `Retry-After` alone does not reach clm; clm reads `try again in` from the
  message text.
- A remote image URL makes the server fetch arbitrary URLs. Keep it off by
  default.

## Open questions

- Does OpenAI Chat accept `detail: "original"` today? The vision guide
  suggests yes; the SDK literal says no.
- What exact error does OpenAI return today for an image part in a
  `system`, `developer`, or `assistant` message?
- Does OpenAI still reject an unknown top-level field with "Unrecognized
  request argument supplied"? This is observed behavior, not documented.
- What does OpenAI return for `tool_choice: "required"` with
  `parallel_tool_calls: false` and several tools? Not documented.
- Should lgml emit `reasoning_content`, `reasoning`, or both? clm reads both;
  other clients read one.
