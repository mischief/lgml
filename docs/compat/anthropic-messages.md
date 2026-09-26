# Anthropic Messages API: compatibility reference

This file describes the Anthropic Messages API as clients use it on
2026-09-26. It covers `POST /v1/messages` and
`POST /v1/messages/count_tokens`. It gives most detail to multimodal
input (images). The goal is a faithful implementation in lgml.

The last section, [What lgml needs](#what-lgml-needs), is the
checklist. Everything before it is reference.

## Contents

- [Sources](#sources)
- [Endpoints](#endpoints)
- [Request headers](#request-headers)
- [Response headers](#response-headers)
- [Request body](#request-body)
- [Messages and roles](#messages-and-roles)
- [Content blocks (input)](#content-blocks-input)
- [Images](#images)
- [Tools](#tools)
- [Thinking](#thinking)
- [Effort and output_config](#effort-and-output_config)
- [Prompt caching](#prompt-caching)
- [Response object](#response-object)
- [Streaming](#streaming)
- [Errors](#errors)
- [count_tokens](#count_tokens)
- [The clm client](#the-clm-client)
- [Claude Code as a client](#claude-code-as-a-client)
- [Examples](#examples)
- [What lgml needs](#what-lgml-needs)
- [Pitfalls](#pitfalls)

## Sources

The Anthropic docs now live on `platform.claude.com`. The older
`docs.anthropic.com` and `docs.claude.com` URLs redirect there. Each
page is also served as Markdown when you append `.md` to the URL.

Short names used in this file:

| Name | Source |
| --- | --- |
| [overview] | https://platform.claude.com/docs/en/api/overview |
| [versioning] | https://platform.claude.com/docs/en/api/versioning |
| [beta-headers] | https://platform.claude.com/docs/en/api/beta-headers |
| [errors] | https://platform.claude.com/docs/en/api/errors |
| [create] | https://platform.claude.com/docs/en/api/messages/create (API reference) |
| [count-ref] | https://platform.claude.com/docs/en/api/messages/count_tokens (API reference) |
| [messages-guide] | https://platform.claude.com/docs/en/build-with-claude/working-with-messages |
| [vision] | https://platform.claude.com/docs/en/build-with-claude/vision |
| [files] | https://platform.claude.com/docs/en/build-with-claude/files |
| [pdf] | https://platform.claude.com/docs/en/build-with-claude/pdf-support |
| [search-results] | https://platform.claude.com/docs/en/build-with-claude/search-results |
| [caching] | https://platform.claude.com/docs/en/build-with-claude/prompt-caching |
| [define-tools] | https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools |
| [handle-tools] | https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls |
| [streaming] | https://platform.claude.com/docs/en/build-with-claude/streaming |
| [thinking] | https://platform.claude.com/docs/en/build-with-claude/thinking |
| [ext-thinking] | https://platform.claude.com/docs/en/build-with-claude/extended-thinking |
| [thinking-ts] | https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting |
| [preserved] | https://platform.claude.com/docs/en/build-with-claude/preserved-thinking |
| [effort] | https://platform.claude.com/docs/en/build-with-claude/effort |
| [stop-reasons] | https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons |
| [token-counting] | https://platform.claude.com/docs/en/build-with-claude/token-counting |
| [midsys] | https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages |
| [cc-gateway] | https://code.claude.com/docs/en/llm-gateway-protocol (Claude Code gateway compatibility guide) |
| [sdk-migration] | https://github.com/anthropics/anthropic-sdk-python/blob/main/MIGRATION.md |

SDK type files are in `anthropics/anthropic-sdk-python`, branch
`main`, release 1.8.0 (2026-09-22). This file cites them as
`types/<name>.py`. The full path is
`src/anthropic/types/<name>.py`. Raw copies are at
`https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/src/anthropic/types/<name>.py`.
Other SDK files cited:

- `_client.py`: `src/anthropic/_client.py` (auth headers).
- `_streaming.py`: `src/anthropic/_streaming.py` (SSE dispatch).
- `lib/streaming/_messages.py`: `src/anthropic/lib/streaming/_messages.py`
  (stream accumulator).
- `resources/beta/messages/messages.py` (beta paths).

Items marked "observed" come from API behavior, not from the docs.
Treat them as likely but not guaranteed.

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/v1/messages` | Create a message. Streams when `stream: true`. [create] |
| POST | `/v1/messages/count_tokens` | Count input tokens. Free on Anthropic. [count-ref], [token-counting] |

- The SDK beta namespace posts to `/v1/messages?beta=true` and
  `/v1/messages/count_tokens?beta=true`
  (`resources/beta/messages/messages.py`). Claude Code also posts to
  `/v1/messages?beta=true` [cc-gateway]. A server must match the path
  and ignore the query string.
- Claude Code sends a best-effort `HEAD /api/hello` probe at startup.
  A 404 is fine [cc-gateway].
- Claude Code can call `GET /v1/models?limit=1000` for model discovery
  when `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`. It keeps only IDs
  that contain `claude` or `anthropic` [cc-gateway].

## Request headers

| Header | Rule | Source |
| --- | --- | --- |
| `x-api-key` | API key. The docs now call it the legacy fallback, still supported. | [overview] |
| `Authorization` | `Bearer <token>`. An API key or a short-lived access token (OAuth or Workload Identity Federation). Required unless `x-api-key` is set. | [overview] |
| `anthropic-version` | Required. The only current value is `2023-06-01`. | [overview], [versioning] |
| `anthropic-beta` | Optional. Comma-separated list of beta names. The header can also repeat; the API reads every copy. | [beta-headers] |
| `content-type` | `application/json`. Required. | [overview] |
| `anthropic-workspace-id` | Optional. Selects a workspace for multi-workspace keys. | [overview], [create] |

Notes:

- The Python SDK sends `X-Api-Key` when `api_key` is set and
  `Authorization: Bearer` when `auth_token` is set. It can send both
  (`_client.py`, `auth_headers`). Claude Code also sends one or both,
  depending on `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN`
  [cc-gateway].
- OAuth tokens (claude.ai login) go in `Authorization: Bearer` with the
  beta value `oauth-2025-04-20` (claude-api skill reference; also
  [cc-gateway], which says the OAuth capability rides in
  `anthropic-beta`).
- The SDKs join many beta values with a comma and no space
  (anthropic-sdk-python CHANGELOG, 1.7.0). Parse with optional spaces.
- An unknown or unauthorized beta name gets a 400
  `invalid_request_error` on Anthropic, with the message
  ``Unexpected value(s) `invalid-beta-name` for the `anthropic-beta` header. ...``
  [beta-headers]. See [What lgml needs](#what-lgml-needs) for why lgml
  must not copy this.
- SDKs add `x-stainless-*` headers. Claude Code adds
  `x-claude-code-session-id` and other `x-claude-code-*` headers
  [cc-gateway]. A server must ignore unknown headers.

## Response headers

| Header | Meaning | Source |
| --- | --- | --- |
| `request-id` | Unique request ID, for example `req_018EeWyXxfu5pfWkrYcMdjWG`. On every response. Same value as `request_id` in error bodies. | [overview], [errors] |
| `anthropic-organization-id` | Organization of the credential. | [overview] |
| `content-type` | `application/json`, or `text/event-stream` for streams. | [cc-gateway] |
| `retry-after` | Seconds to wait (on 429 and some 5xx). Claude Code wants integer seconds; above 60 it stops retrying. | [errors], [cc-gateway] |
| `x-should-retry` | `true` or `false`. Claude Code reads it when it decides to retry. | [cc-gateway] |

## Request body

Types come from `types/message_create_params.py` and [create].

| Field | Type | Required | Default and limits | Source |
| --- | --- | --- | --- | --- |
| `model` | string | yes | Model ID. | [create] |
| `max_tokens` | integer | yes | Minimum 0. `0` pre-warms the prompt cache and generates nothing. Per-model maximum (128K on current Opus and Sonnet, 64K on Haiku 4.5). | [create], [thinking] |
| `messages` | array of MessageParam | yes | At most 100,000 messages. | [create] |
| `system` | string, or array of `text` blocks | no | Blocks can carry `cache_control`. | [create] |
| `stream` | boolean | no | `false`. | [create] |
| `stop_sequences` | array of string | no | Stop on match. Sets `stop_reason: "stop_sequence"` and `stop_sequence`. | [create] |
| `temperature` | number | no | Default 1.0, range 0.0 to 1.0. Deprecated (see below). | [create] |
| `top_p` | number | no | Deprecated (see below). | [create] |
| `top_k` | integer | no | Deprecated (see below). | [create] |
| `metadata` | object | no | `{"user_id": string or null}`. An opaque user ID. | `types/metadata_param.py` |
| `service_tier` | string | no | `"auto"` or `"standard_only"`. | [create] |
| `tools` | array | no | See [Tools](#tools). | [create] |
| `tool_choice` | object | no | See [Tools](#tools). | [create] |
| `thinking` | object | no | See [Thinking](#thinking). | [create] |
| `output_config` | object | no | `effort`, `format` (structured outputs). See [Effort and output_config](#effort-and-output_config). | `types/output_config_param.py` |
| `cache_control` | object or null | no | Top-level automatic caching: marks the last cacheable block. | `types/message_create_params.py` |
| `container` | object or string | no | Code execution container reuse. Server tools only. | [create] |
| `inference_geo` | string | no | Inference region. | `types/message_create_params.py` |
| `context_management` | object | no | Beta (`context-management-2025-06-27`). Clears old tool results or thinking. Claude Code sends it with the beta. | [cc-gateway] |
| `mcp_servers` | array | no | Beta MCP connector. | claude-api skill reference |
| `speed` | string | no | Beta fast mode (`"fast"`). | claude-api skill reference |

Sampling parameters (changed for new models):

- On models released after Claude Opus 4.6, `temperature` other than
  1.0 is a 400. `top_p` below 0.99 is a 400. Any `top_k` is a 400
  [create], [thinking] (section "Sampling parameters").
- On older models, `temperature` and `top_k` are not allowed with
  thinking on, and `top_p` must be 0.95 to 1 with thinking on
  [thinking].
- The Python SDK 1.x removed `temperature`, `top_p`, and `top_k` from
  the typed message methods. Callers pass them with `extra_body`
  [sdk-migration]. So a Python client can still send them on the wire.

Unknown top-level fields: the API rejects them with a 400
`invalid_request_error`. Its message ends with
`Extra inputs are not permitted` [cc-gateway] (for example,
`block_binding: Extra inputs are not permitted` in [preserved]).

## Messages and roles

Source: [create], `types/message_param.py`, [messages-guide], [midsys].

- Each message is `{"role": ..., "content": ...}`.
- `role` is `"user"`, `"assistant"`, or `"system"`.
  `"system"` inside `messages` is the newer mid-conversation system
  message (see below). The top-level `system` field is the usual place
  for a system prompt.
- `content` is a string, or an array of content blocks. A string is
  shorthand for one `text` block.
- The models expect alternating `user` and `assistant` turns. The API
  merges consecutive `user` turns, and consecutive `assistant` turns,
  into one turn [create].
- Final assistant message (prefill): if the last message is
  `assistant`, the model continues from its content [create].
  - Prefill is a 400 on Claude 4.6 and later models and on Claude
    Mythos Preview [messages-guide].
  - Prefill is not allowed while thinking is on, on any model
    [thinking] ("Response prefill and forced tool use").
  - Observed: the API rejects a final assistant message that ends in
    whitespace.
- Text blocks need at least one character: `text` has `minLength: 1`
  [create]. Observed: the API also rejects a message whose content is
  an empty string, except the optional final assistant message.
- Mid-conversation system messages (`role: "system"` inside
  `messages`) [midsys]:
  - Supported on Opus 5.5, Opus 5, Opus 4.8, and the Fable and Mythos
    models. Not on Sonnet 5.
  - The message must immediately follow a `user` turn (a user turn with
    only `tool_result` blocks counts), or an `assistant` turn that ends
    in a server tool result.
  - The message must be last in `messages`, or be followed directly by
    an `assistant` turn.
  - It can carry `clear_at` (beta
    `mid-conversation-system-clear-at-2026-08-21`) and `tool_addition`
    or `tool_removal` blocks (beta `inline-tools-2026-09-15`).
  - Claude Code sends these messages, with `cache_control` on them
    [cc-gateway].

## Content blocks (input)

The union is `ContentBlockParam` in `types/content_block_param.py`.
The common client-side types:

| `type` | Fields | Where | Source |
| --- | --- | --- | --- |
| `text` | `text` (min length 1), `cache_control`, `citations` | user, assistant, system | `types/text_block_param.py` |
| `image` | `source`, `cache_control`, `transformations` | user, inside `tool_result` | `types/image_block_param.py` |
| `document` | `source`, `title`, `context`, `citations`, `cache_control` | user, inside `tool_result` | `types/document_block_param.py` |
| `search_result` | `source` (string), `title`, `content` (array of `text`), `citations`, `cache_control` | user, inside `tool_result` | `types/search_result_block_param.py` |
| `tool_use` | `id`, `name`, `input` (object), `cache_control`, `caller`, `toolset_name` | assistant | `types/tool_use_block_param.py` |
| `tool_result` | `tool_use_id`, `content`, `is_error`, `cache_control`, `toolset_name` | user | `types/tool_result_block_param.py` |
| `thinking` | `thinking`, `signature` (both required) | assistant | `types/thinking_block_param.py` |
| `redacted_thinking` | `data` (required) | assistant | `types/redacted_thinking_block_param.py` |

Other block types exist for server tools (`server_tool_use`,
`web_search_tool_result`, `web_fetch_tool_result`,
`code_execution_tool_result`, `bash_code_execution_tool_result`,
`text_editor_code_execution_tool_result`, `tool_search_tool_result`,
`container_upload`) and for newer betas (`tool_reference`,
`browser_state`, compaction blocks). See `types/message_param.py`.

### document (brief)

- Sources: `base64` PDF (`media_type: "application/pdf"`), `text`
  (`media_type: "text/plain"`), `content` (an array of blocks), `url`
  PDF, and `file` (`file_id`) (`types/document_block_param.py`).
- Limits: 32 MB request, 600 pages per request (100 when the context
  window is under 1M) [pdf].
- `citations: {"enabled": true}` turns on citations. The response then
  splits text into many `text` blocks with `citations` arrays
  (claude-api skill reference; [create]).

### search_result (brief)

`{"type": "search_result", "source": "https://...", "title": "...",
"content": [{"type": "text", "text": "..."}]}`. It gives citations
for RAG content [search-results].

## Images

Sources: [vision], [create], `types/image_block_param.py`,
`types/base64_image_source_param.py`,
`types/url_image_source_param.py`, `types/file_image_source_param.py`.

### Image block

```json
{"type": "image",
 "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KG..."}}
```

`source` is one of three types:

| `source.type` | Fields | Notes |
| --- | --- | --- |
| `base64` | `media_type`, `data` | `data` is plain base64 (`format: byte`) with no prefix. |
| `url` | `url` | The API fetches the image. |
| `file` | `file_id` | A Files API upload. |

- Anthropic has no data URLs. A client that holds
  `data:image/png;base64,XXXX` must split it: `media_type` gets
  `image/png`, `data` gets `XXXX`. The `url` source takes an HTTP(S)
  URL. Observed: `data:` URLs in `url` are not accepted.
- `media_type` is one of `image/jpeg`, `image/png`, `image/gif`,
  `image/webp` [vision], `types/base64_image_source_param.py`.
  Animated images use the first frame only [vision].
- Observed: the API checks that the bytes match `media_type`, and
  returns 400 when they do not.
- On Amazon Bedrock and Google Cloud, only `base64` sources work
  [vision].
- The Files API is GA and needs no beta header. The old beta
  `files-api-2025-04-14` still works [files]. A `file` source must
  match the block type: images go in `image` blocks, PDFs and text in
  `document` blocks (claude-api skill reference; [files]).
- `transformations` (new field): `{"oversized_image": "downsize"}` is
  the default. `"error"` makes an oversized image a 400 instead of a
  silent downscale (`types/image_transformations_param.py`, [vision]).
- `cache_control` is allowed on an image block.
- The docs advise images before text in a message [vision].

### Where images can appear

| Place | Allowed | Source |
| --- | --- | --- |
| `user` message content | yes | [vision] |
| `tool_result.content` array | yes (`text`, `image`, `document`, `search_result`) | [handle-tools], `types/tool_result_block_param.py` |
| `assistant` message content | no image input block type in the assistant turn is documented | [create] |
| `system` | no; `system` takes `text` blocks only | [create] |
| Model output | never; Claude outputs no image blocks | `types/content_block.py` |

### Limits (current numbers)

From [vision], [overview], [errors]:

| Limit | Value |
| --- | --- |
| Images per request | 100 for models with a 200K context window; 600 for other models. |
| Max dimensions per image | 8000 x 8000 px. |
| More than 20 images in one request | Every image must be at most 2000 x 2000 px. All `image` blocks count, including earlier turns and images inside `tool_result`. The error is `invalid_request_error` and mentions "many-image requests". |
| Max size per image | 10 MB base64-encoded on the Claude API; 5 MB on Bedrock and Google Cloud. |
| Request size | 32 MB for Messages and count_tokens. Over the limit gives 413 `request_too_large`. |

### Image tokens

- Current rule: an image costs `ceil(width/28) * ceil(height/28)`
  visual tokens. Each 28 x 28 px patch is one token [vision].
- Images larger than the model tier limit are downscaled first,
  keeping the aspect ratio [vision]:

  | Tier | Models | Max long edge | Max visual tokens |
  | --- | --- | --- | --- |
  | High-resolution | Claude 4.7 and later | 2576 px | 4784 |
  | Standard | other models | 1568 px | 1568 |

- Example: 1000 x 1000 px costs 1296 tokens on both tiers [vision].
- Older versions of the vision guide gave
  `tokens = width * height / 750`. Clients that estimate with that
  formula get numbers near the new rule for small images. Do not
  depend on either formula for a local model; count the real tokens.

## Tools

### Tool definitions

Source: `types/tool_param.py`, [define-tools], [create].

| Field | Type | Notes |
| --- | --- | --- |
| `name` | string | Required. `^[a-zA-Z0-9_-]{1,128}$`. |
| `input_schema` | JSON Schema object | Required. `type` must be `"object"`. |
| `description` | string | Optional but strongly recommended. |
| `type` | `"custom"` or absent | Absent means a custom (client) tool. |
| `cache_control` | object or null | Cache breakpoint. |
| `strict` | boolean | Guarantees schema-valid `input`. Needs `additionalProperties: false` and `required` (claude-api skill reference). |
| `eager_input_streaming` | boolean or null | Streams tool input as it is generated, without buffering. |
| `defer_loading` | boolean | Tool search: withhold until a `tool_reference` loads it. |
| `input_examples` | array of objects | Examples shown to the model. |
| `allowed_callers` | array | `"direct"` or code execution versions (programmatic tool calling). |

Server tools and Anthropic-defined tools have a dated `type`, for
example `web_search_20260209`, `web_fetch_20260209`,
`code_execution_20260521`, `bash_20250124`,
`text_editor_20250728`, `memory_20250818`,
`tool_search_tool_regex_20251119`, `advisor_20260301`. They run on
Anthropic servers or have fixed schemas. Their results come back as
their own block types (claude-api skill reference, [cc-gateway]).

### tool_choice

Source: `types/tool_choice_*_param.py`.

| Value | Meaning |
| --- | --- |
| `{"type": "auto"}` | The model decides. Default when `tools` is set. |
| `{"type": "any"}` | The model must call some tool. |
| `{"type": "tool", "name": "x"}` | The model must call tool `x`. |
| `{"type": "none"}` | The model must not call tools. |

- `auto`, `any`, and `tool` take `disable_parallel_tool_use`
  (default `false`). With `true`, `auto` gives at most one tool call,
  and `any` or `tool` give exactly one.
- `any` and `tool` are a 400 with manual thinking
  (`type: "enabled"`). Adaptive thinking allows them, except on Claude
  Opus 5.5, Fable 5.1, and Mythos 5.1, which reject forced tool use on
  every request [thinking], [define-tools].

### tool_use block (model output)

Source: `types/tool_use_block.py`, [handle-tools].

```json
{"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9",
 "name": "get_weather", "input": {"location": "San Francisco, CA"}}
```

- `id` starts with `toolu_` on Anthropic. Input validation only
  requires `^[a-zA-Z0-9_-]+$` for the `id` of a `tool_use` block and
  for `tool_use_id` [create]. Server tools use `srvtoolu_`.
- `input` is always a JSON object, never a string.
- The response has `stop_reason: "tool_use"`.
- One assistant message can hold many `tool_use` blocks (parallel
  tool use).

### tool_result block (client input)

Source: `types/tool_result_block_param.py`, [handle-tools].

- `tool_use_id` (required), `content` (optional: a string or an array
  of `text`, `image`, `document`, `search_result` blocks), `is_error`
  (optional boolean), `cache_control`.
- Images inside tool results are allowed. They count toward the image
  limits [vision].
- Ordering rules [handle-tools]:
  - Tool results must come in the user message directly after the
    assistant message with the matching `tool_use` blocks. No messages
    can come between.
  - In that user message, all `tool_result` blocks come first. Text
    blocks come after them. Text before a `tool_result` is a 400.
  - Every `tool_use` needs its `tool_result`. The error for a missing
    one says "tool_use ids were found without tool_result blocks
    immediately after".
- Return all results of one parallel batch in one user message
  (claude-api skill reference).

## Thinking

Sources: [thinking], [ext-thinking], [thinking-ts], [preserved],
`types/thinking_config_*_param.py`, `types/thinking_block.py`,
`types/redacted_thinking_block.py`.

### Request: `thinking`

| Value | Meaning |
| --- | --- |
| `{"type": "enabled", "budget_tokens": N}` | Manual extended thinking. `N >= 1024` and `N < max_tokens`. Interleaved thinking lets `N` exceed `max_tokens`. |
| `{"type": "adaptive"}` | The model decides when and how much to think. Interleaved thinking is automatic. |
| `{"type": "disabled"}` | No thinking. |

- `enabled` and `adaptive` take `display`: `"summarized"`,
  `"omitted"`, or `"updates"` (beta
  `thinking-display-updates-2026-08-18`). `display` with `disabled` is
  invalid [thinking].
- `"omitted"` returns thinking blocks with `thinking: ""` and a full
  `signature` [thinking].
- The SDK docstring says `display` defaults to `summarized`. The
  current docs say the default is `"omitted"` on Opus 4.7 and later,
  Sonnet 5, and the Fable and Mythos models, and `"summarized"` on
  Opus 4.6, Sonnet 4.6, and earlier [thinking]. The per-model default
  applies on the wire.
- Per-model rules (changed through 2026) [thinking], [ext-thinking]:
  - `budget_tokens` (`type: "enabled"`) is a 400 on Opus 4.7 and later,
    Sonnet 5, and the Fable models. It is deprecated on Opus 4.6 and
    Sonnet 4.6. Haiku 4.5 still uses it.
  - Thinking is on by default on Opus 5.5, Opus 5, Sonnet 5, and the
    Fable and Mythos models. `disabled` is a 400 on Opus 5.5 and the
    Fable and Mythos models. On Opus 5, `disabled` is a 400 at effort
    `xhigh` or `max`.
- The beta `interleaved-thinking-2025-05-14` enables thinking between
  tool calls in manual mode on older models. The API accepts it on any
  model and ignores it where it has no effect [ext-thinking].
- Newer beta: `thinking.block_binding.prefix_mismatch_behavior`
  (`"error"` or `"drop_block"`) with `thinking-binding-controls-2026-08-01`
  [preserved].

### Response: thinking blocks

```json
{"type": "thinking", "thinking": "Let me break this down...", "signature": "WaUjzkypQ2mUEVM36O2Txu..."}
{"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4a..."}
```

- `signature` is opaque. It is an encrypted copy of the full reasoning.
  Clients must not parse it [thinking], `types/thinking_block.py`.
- `thinking` text is a summary, never the raw chain of thought
  [thinking].
- `redacted_thinking.data` is encrypted, with no readable text
  [thinking].
- Thinking blocks come before the `text` blocks. With tools, thinking
  can also come between tool calls (interleaved thinking) [thinking].

### Passing thinking back

From [thinking] ("Preserving thinking blocks", "Thinking encryption"):

- Required: within a tool-use turn, pass the thinking blocks back
  with the `tool_use` blocks, complete and unmodified.
- Recommended: pass everything back across turns. The API filters old
  blocks per model and bills only what it keeps.
- Allowed: outside tool use, omit prior turns' thinking.
- In the latest assistant message, the thinking blocks must keep
  their original order and content. Changing them gives a 400 whose
  message contains
  ``thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified``
  [thinking-ts].
- A signature that does not verify gives a 400:
  ``messages.{i}.content.{j}: Invalid `signature` in `thinking` block``
  [thinking-ts].
- A block from a model that the current model cannot read is dropped
  without an error and without billing [thinking], [preserved].
- Text placed in the `thinking` field of a round-tripped omitted block
  is ignored; the server rebuilds the reasoning from `signature`
  [thinking].
- A tool-use loop is one assistant turn. Toggling thinking mid-turn
  does not error; the API disables thinking for that request
  [thinking].
- Manual mode needs the final assistant turn of a thinking request to
  start with a thinking block. Adaptive mode drops that rule
  [thinking].

### Thinking restrictions

- No prefill with thinking on [thinking].
- Forced `tool_choice` is a 400 with manual thinking [thinking].
- `temperature`, `top_k` not allowed with thinking; `top_p` only 0.95
  to 1 (older models) [thinking].
- Thinking tokens count toward `max_tokens` and bill as output
  [thinking]. `usage.output_tokens_details.thinking_tokens` reports
  them [ext-thinking].

## Effort and output_config

Source: `types/output_config_param.py`, [effort].

- `output_config.effort`: `"low"`, `"medium"`, `"high"`, `"xhigh"`,
  `"max"`. GA, no beta header. Default `"high"`, except `"medium"` on
  Claude Opus 5.5. Sending the default equals omitting the field.
- `output_config.format`: structured outputs,
  `{"type": "json_schema", "schema": {...}}`. It replaces the old
  top-level `output_format` [sdk-migration].
- `output_config.task_budget` (beta `task-budgets-2026-03-13`)
  (claude-api skill reference).
- Claude Code sends `output_config.effort` and adaptive thinking to any
  model ID it does not recognize [cc-gateway].

## Prompt caching

Source: [caching], `types/cache_control_ephemeral_param.py`.

- `cache_control: {"type": "ephemeral", "ttl": "5m" | "1h"}`. `ttl`
  defaults to `"5m"`.
- Allowed on `system` text blocks, message content blocks (text,
  image, document, tool_use, tool_result, search_result), and tool
  definitions. Not on thinking blocks.
- At most 4 breakpoints per request. Observed: a fifth is a 400.
- A `1h` entry must come before any `5m` entry.
- Top-level `cache_control` (automatic caching) puts the breakpoint on
  the last cacheable block.
- Prompts shorter than the model minimum are not cached. That is not
  an error [caching].
- Claude Code asks for 1h TTL with the `extended-cache-ttl-*` beta
  value [cc-gateway].

## Response object

Source: `types/message.py`, `types/usage.py`, `types/stop_reason.py`,
[create].

```json
{
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-5",
  "content": [{"type": "text", "text": "Hi! My name is Claude."}],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "stop_details": null,
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
    "service_tier": "standard"
  }
}
```

- `id`: `msg_` prefix. Format and length may change.
- `type`: always `"message"`. `role`: always `"assistant"`.
- `content`: array of output blocks: `text`, `thinking`,
  `redacted_thinking`, `tool_use`, and server tool blocks
  (`types/content_block.py`). No image output.
- `model`: the model that ran.
- `stop_reason` (`types/stop_reason.py`):

  | Value | Meaning |
  | --- | --- |
  | `end_turn` | Natural stop. |
  | `max_tokens` | Hit `max_tokens` or the model maximum. |
  | `stop_sequence` | Matched a `stop_sequences` entry. |
  | `tool_use` | The model called tools. |
  | `pause_turn` | A server-tool loop paused; send the response back to continue. |
  | `refusal` | Safety classifiers declined. `stop_details` explains. |
  | `model_context_window_exceeded` | Output filled the context window. Returned without a beta on Sonnet 4.5 and newer [stop-reasons]. |

  Non-streaming: always non-null. Streaming: `null` in
  `message_start`, set in `message_delta` [stop-reasons].
- `stop_sequence`: the matched string, or `null`.
- `stop_details`: `null` unless `stop_reason` is `"refusal"`. Then
  `{"type": "refusal", "category": ..., "explanation": ...}`
  (`types/refusal_stop_details.py`).
- `usage` (`types/usage.py`):

  | Field | Type | Notes |
  | --- | --- | --- |
  | `input_tokens` | int | Uncached input tokens after the last breakpoint. |
  | `output_tokens` | int | Includes thinking tokens. Non-zero even for empty text. |
  | `cache_creation_input_tokens` | int or null | Tokens written to cache. |
  | `cache_read_input_tokens` | int or null | Tokens read from cache. |
  | `cache_creation` | object or null | `ephemeral_5m_input_tokens`, `ephemeral_1h_input_tokens`. |
  | `server_tool_use` | object or null | `web_search_requests`, `web_fetch_requests`. |
  | `service_tier` | `"standard"`, `"priority"`, `"batch"`, or null | |
  | `output_tokens_details` | object or null | `thinking_tokens`. |
  | `inference_geo` | string or null | |

  Total input = `input_tokens + cache_creation_input_tokens +
  cache_read_input_tokens` (`types/message.py`).
- Newer optional fields: `container`, `diagnostics`,
  `input_transformations` (betas) [create], [preserved].

## Streaming

Sources: [streaming], [versioning], `types/raw_*.py`, `_streaming.py`,
`lib/streaming/_messages.py`, [cc-gateway].

### Framing

- Response `content-type: text/event-stream`.
- Every event is a named SSE event. Each event is one `event:` line,
  one `data:` line with one-line JSON, and a blank line:

  ```
  event: content_block_delta
  data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}

  ```

- The JSON `type` equals the event name.
- There is no `data: [DONE]`. Version `2023-06-01` removed it
  [versioning].
- The Python SDK dispatches on the SSE `event:` name, not on the JSON.
  An event with no `event:` line is dropped (`_streaming.py`).

### Event order

1. `message_start`: `{"type": "message_start", "message": {...}}`.
   The message has `content: []`, `stop_reason: null`,
   `stop_sequence: null`, and `usage` with `input_tokens` and a small
   `output_tokens`.
2. For each content block, in index order:
   - `content_block_start`: `{"index": i, "content_block": {...}}`.
     The block is empty: `{"type":"text","text":""}`,
     `{"type":"tool_use","id":...,"name":...,"input":{}}`, or
     `{"type":"thinking","thinking":"","signature":""}`.
   - Zero or more `content_block_delta`: `{"index": i, "delta": {...}}`.
   - `content_block_stop`: `{"index": i}`.
3. One or more `message_delta`:
   `{"delta": {"stop_reason": ..., "stop_sequence": ...}, "usage": {"output_tokens": N}}`.
   Usage counts are cumulative, not increments [streaming]. The final
   `message_delta` can also carry `input_tokens`, cache counts,
   `server_tool_use`, and `output_tokens_details`.
4. `message_stop`: `{"type": "message_stop"}`.

`ping` events (`{"type": "ping"}`) can appear anywhere [streaming].

The `index` matches the block position in the final `content`. The
Python SDK accumulator appends blocks in arrival order and then looks
up deltas by `index`, so indexes must start at 0 and grow by one, and
blocks must not interleave (`lib/streaming/_messages.py`).

### Delta types

Source: `types/raw_content_block_delta.py`.

| `delta.type` | Field | For block |
| --- | --- | --- |
| `text_delta` | `text` | `text` |
| `input_json_delta` | `partial_json` (string fragment) | `tool_use` |
| `thinking_delta` | `thinking` | `thinking` |
| `signature_delta` | `signature` | `thinking` (sent once, just before `content_block_stop`) |
| `citations_delta` | `citation` | `text` |

- `input_json_delta` fragments concatenate to the JSON text of
  `input`. The first fragment can be `""`. Clients parse at
  `content_block_stop` [streaming].
- With `display: "omitted"`, each thinking block gets one
  `thinking_delta` with `""` and then one `signature_delta`
  [streaming].
- `redacted_thinking` arrives complete in `content_block_start`, with
  no deltas (observed).

### Errors in a stream

- After the 200 status, the API sends errors as an SSE event
  [streaming]:

  ```
  event: error
  data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}

  ```

- The Python SDK raises an API error on `event: error`
  (`_streaming.py`).
- Errors found before the stream starts use a normal HTTP status and
  JSON body.
- New event types can appear. Clients must ignore unknown events
  [streaming], [versioning].

### What Claude Code requires of a stream

From [cc-gateway]:

- Stream as tokens arrive. Buffering stalls Claude Code.
- Send each event once, in order. A delta or stop for a block without a
  `content_block_start`, or after its `content_block_stop`, stops the
  read.
- End with `message_delta` (with `stop_reason`) and `message_stop`. A
  body that ends early after a block started counts as a dropped
  connection.
- Claude Code aborts a stream after 300 seconds with no bytes. `ping`
  events count as bytes. Send pings during long silent phases.

## Errors

Source: [errors].

Body shape:

```json
{"type": "error",
 "error": {"type": "not_found_error", "message": "The requested resource could not be found."},
 "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy"}
```

| HTTP | `error.type` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request_error` | Bad format or content. Also used for other 4xx not listed. |
| 401 | `authentication_error` | Bad API key. |
| 402 | `billing_error` | Billing problem. |
| 403 | `permission_error` | Key lacks permission. |
| 404 | `not_found_error` | Not found. |
| 409 | `conflict_error` | Resource conflict. |
| 413 | `request_too_large` | Request over the size limit. |
| 429 | `rate_limit_error` | Rate limit. |
| 500 | `api_error` | Internal error. |
| 504 | `timeout_error` | Timed out while processing. |
| 529 | `overloaded_error` | API overloaded. |

- The `error.type` set can grow [errors].
- SDKs retry connection errors, 408, 409, 429, and 5xx, twice by
  default, and honor `retry-after` [errors].
- Claude Code matches error wording to recover [cc-gateway]:
  - A rejected `thinking` field, mid-conversation system message, or
    `cache_control` on such a message: Claude Code retries and
    disables that feature for the conversation.
  - A rejected thinking signature (for example
    `bound to a different conversation`): Claude Code removes earlier
    thinking blocks and retries.
  - A rejected tool type: a 400 or 422 whose message contains
    `Input tag '<type>'`, for example `Input tag 'advisor_20260301'`.
    Claude Code retries once without that tool.
  - Rejected `context_management` or tool schema fields: no retry. The
    400 reaches the user.

## count_tokens

Sources: [count-ref], [token-counting],
`types/message_count_tokens_params.py`,
`types/message_tokens_count.py`.

- `POST /v1/messages/count_tokens`. Same headers as `/v1/messages`.
- Body: `model`, `messages` (required); `system`, `tools`,
  `tool_choice`, `thinking`, `output_config`, `cache_control`
  (optional). No `max_tokens`, `stream`, `stop_sequences`, or
  `metadata` in the SDK type.
- Response: `{"input_tokens": 2095}`. The count is an estimate.
- It returns `invalid_request_error` for server tools (except the
  advisor tool), the MCP connector, and `image` or `document` blocks
  with a `url` or `file` source. Send images as base64 to count them
  [token-counting].
- Thinking: prior-turn thinking counts on models that keep all turns;
  current-turn thinking counts [token-counting].
- Claude Code treats the endpoint as optional. Without it, Claude Code
  uses a character-based estimate [cc-gateway].

## The clm client

This section describes the Anthropic provider of the clm client, read
from its source. It is one real client that lgml must serve.

Transport:

- Path: `<base>/messages`, so the base URL must end in `/v1`.
- Headers: `Content-Type: application/json`, `x-api-key: <key>` only
  when a key is set, `anthropic-version: 2023-06-01`. No
  `Authorization`. No `anthropic-beta`.

Request fields it sends:

| Field | Value |
| --- | --- |
| `model` | Configured model name. |
| `system` | An array of one `text` block with `cache_control: {"type":"ephemeral"}`. Falls back to a string when the block cannot be built. Omitted when there is no system text. Many system messages are joined with a blank line. |
| `messages` | See below. |
| `max_tokens` | Always `64000`. |
| `stream` | Always present, `true` or `false`. |
| `output_config` | `{"effort": "<value>"}` only when the user sets an effort. |
| `tools` | `{"type":"custom","name","description","input_schema"}`. `description` is `""` when absent. |
| `tool_choice` | `{"type":"auto","disable_parallel_tool_use":true}` when parallel calls are off. `{"type":"none"}` when tool calls are forbidden. |

Messages:

- Plain user and assistant messages: `content` is a string. It can be
  `""`.
- An assistant turn with tool calls: an array with an optional `text`
  block, then `tool_use` blocks. `input` is the parsed arguments
  object, or `{}` when they do not parse.
- Tool results: consecutive tool messages become one `user` message
  with `tool_result` blocks. `content` is always a string. clm never
  sets `is_error`.
- A user message right after tool results joins that same user message
  as a `text` block after the results.
- A cache breakpoint goes on the last non-empty block of the last
  message. String content turns into a one-block array for this.
- clm sends no `image` blocks, no `document` blocks, and no `thinking`
  or `redacted_thinking` blocks. It drops reasoning from history.

Response handling:

- Non-streaming: reads `text`, `thinking`, and `tool_use` blocks.
  Ignores `signature` and `redacted_thinking`. Usage: prompt tokens =
  `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`.
- Streaming: handles `message_start` (usage), `content_block_start`
  (only `tool_use`: id and name), `content_block_delta` (`text_delta`,
  `thinking_delta`, `input_json_delta`), and `message_delta`
  (`stop_reason`, `usage.output_tokens`). It ignores
  `content_block_stop`, `message_stop`, `ping`, `signature_delta`, and
  `error`. An `error` event mid-stream is lost.
- It uses the content block `index` as the tool call index.
- Stop reason map: `end_turn` and `stop_sequence` to `stop`,
  `max_tokens` to `length`, `tool_use` to `tool_calls`. Other values
  pass through.

## Claude Code as a client

Claude Code with `ANTHROPIC_BASE_URL` treats the server as the Claude
API [cc-gateway]. For lgml this means:

- It sends the full set of beta values and body fields, unless
  `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` is set. That variable
  stops context management and beta tool fields. It does not stop
  adaptive thinking.
- For a model ID it does not know, it sends
  `thinking: {"type": "adaptive"}`, effort, and context management.
  It assumes a 200K context window (1M when the ID has `[1m]`).
- It sends `cache_control` on `system` blocks and on messages,
  including mid-conversation system messages.
- It puts an attribution block first in the `system` array. Only
  `api.anthropic.com` strips it. `CLAUDE_CODE_ATTRIBUTION_HEADER=0`
  turns it off.
- Fine-grained tool streaming is off by default for a custom base URL.
- It reads `retry-after`, `x-should-retry`, and
  `anthropic-ratelimit-unified-*` headers.
- It needs pings during long pauses (300-second stall limit).

## Examples

All examples use `anthropic-version: 2023-06-01` and
`content-type: application/json`. Base64 data is shortened.

### Plain text

Request:

```json
{"model": "claude-opus-5-5", "max_tokens": 1024,
 "messages": [{"role": "user", "content": "Hello, Claude"}]}
```

Response:

```json
{"id": "msg_01XFDUDYJgAACzvnptvVoYEL", "type": "message", "role": "assistant",
 "model": "claude-opus-5-5",
 "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
 "stop_reason": "end_turn", "stop_sequence": null, "stop_details": null,
 "usage": {"input_tokens": 10, "output_tokens": 12,
           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
```

### Image by base64

There is no `data:` URL. The media type and the base64 payload are
separate fields [vision].

```json
{"model": "claude-opus-5-5", "max_tokens": 1024,
 "messages": [{"role": "user", "content": [
   {"type": "image",
    "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRgABAQ..."}},
   {"type": "text", "text": "Describe this image."}]}]}
```

The same image by URL:

```json
{"type": "image", "source": {"type": "url", "url": "https://example.com/cat.jpg"}}
```

### Tool call round trip

Request 1:

```json
{"model": "claude-opus-5-5", "max_tokens": 1024,
 "tools": [{"name": "get_weather", "description": "Get the current weather",
            "input_schema": {"type": "object",
                             "properties": {"location": {"type": "string"}},
                             "required": ["location"]}}],
 "messages": [{"role": "user", "content": "What is the weather in Paris?"}]}
```

Response 1:

```json
{"id": "msg_01Aq9w938a90dw8q", "type": "message", "role": "assistant",
 "model": "claude-opus-5-5",
 "content": [
   {"type": "text", "text": "I'll check the weather in Paris."},
   {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9",
    "name": "get_weather", "input": {"location": "Paris"}}],
 "stop_reason": "tool_use", "stop_sequence": null,
 "usage": {"input_tokens": 380, "output_tokens": 60}}
```

Request 2 (the client echoes the assistant turn and adds the result;
`tools` stays the same):

```json
{"model": "claude-opus-5-5", "max_tokens": 1024,
 "tools": [{"name": "get_weather", "description": "Get the current weather",
            "input_schema": {"type": "object",
                             "properties": {"location": {"type": "string"}},
                             "required": ["location"]}}],
 "messages": [
   {"role": "user", "content": "What is the weather in Paris?"},
   {"role": "assistant", "content": [
     {"type": "text", "text": "I'll check the weather in Paris."},
     {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9",
      "name": "get_weather", "input": {"location": "Paris"}}]},
   {"role": "user", "content": [
     {"type": "tool_result", "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
      "content": "20°C, sunny"}]}]}
```

Response 2:

```json
{"id": "msg_01Bq...", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
 "content": [{"type": "text", "text": "It is 20°C and sunny in Paris."}],
 "stop_reason": "end_turn", "stop_sequence": null,
 "usage": {"input_tokens": 460, "output_tokens": 14}}
```

### Tool result with an image

The image goes in the `content` array of the `tool_result`
[handle-tools]:

```json
{"role": "user", "content": [
  {"type": "tool_result", "tool_use_id": "toolu_01Xq...",
   "content": [
     {"type": "text", "text": "Screenshot of the page:"},
     {"type": "image",
      "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo..."}}]},
  {"type": "text", "text": "Is the login button visible?"}]}
```

An error result: `{"type": "tool_result", "tool_use_id": "toolu_...",
"content": "ConnectionError: timeout", "is_error": true}`.

### Streaming with a tool call

Request: the first tool request above, plus `"stream": true`.
Response (from [streaming], shortened):

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-opus-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Okay, let's check the weather:"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"San Francisco, CA\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}

event: message_stop
data: {"type":"message_stop"}

```

### Thinking reply (non-streaming)

Request:

```json
{"model": "claude-opus-4-8", "max_tokens": 16000,
 "thinking": {"type": "adaptive", "display": "summarized"},
 "messages": [{"role": "user", "content": "What is the GCD of 1071 and 462?"}]}
```

Response:

```json
{"id": "msg_01...", "type": "message", "role": "assistant", "model": "claude-opus-4-8",
 "content": [
   {"type": "thinking",
    "thinking": "Euclid: 1071 = 2*462 + 147; 462 = 3*147 + 21; 147 = 7*21. GCD is 21.",
    "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."},
   {"type": "text", "text": "The greatest common divisor of 1071 and 462 is **21**."}],
 "stop_reason": "end_turn", "stop_sequence": null,
 "usage": {"input_tokens": 22, "output_tokens": 180,
           "output_tokens_details": {"thinking_tokens": 150}}}
```

Manual mode on an older model uses
`"thinking": {"type": "enabled", "budget_tokens": 10000}` with
`max_tokens` above 10000 [ext-thinking].

### Thinking reply (streaming)

From [streaming], shortened:

```
event: message_start
data: {"type": "message_start", "message": {"id": "msg_01...", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-5-5", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 22, "output_tokens": 1}}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I need to find the GCD of 1071 and 462.\n1071 = 2 × 462 + 147"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "\n462 = 3 × 147 + 21\n147 = 7 × 21 + 0"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The greatest common divisor of 1071 and 462 is **21**."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 180}}

event: message_stop
data: {"type": "message_stop"}

```

The docs example omits `usage` in `message_start` and
`message_delta`. The Python SDK accumulator reads
`event.usage.output_tokens` on every `message_delta`, so a server must
always send it (`lib/streaming/_messages.py`).

### Error

```
HTTP/1.1 400 Bad Request
content-type: application/json
request-id: req_011CSHoEeqs5C35K2UUqR7Fy

{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: Field required"},"request_id":"req_011CSHoEeqs5C35K2UUqR7Fy"}
```

## What lgml needs

A checklist for a server that serves the Anthropic SDKs (Python,
TypeScript), Claude Code with `ANTHROPIC_BASE_URL`, and the clm
client.

### Routes and headers

- [ ] Serve `POST /v1/messages` and `POST /v1/messages/count_tokens`.
      Ignore the query string (`?beta=true`).
- [ ] Accept the credential in `x-api-key` or `Authorization: Bearer`.
      Check either one against the lgml key, when one is set.
- [ ] Require `anthropic-version`. Accept `2023-06-01`. Reply 400
      `invalid_request_error` when it is absent or unknown.
- [ ] Parse `anthropic-beta` as a comma list, spaces optional, and
      across repeated headers.
- [ ] Do not reject unknown beta names. Claude Code sends a changing
      set and does not recover from a 400 on the header. The header
      alone changes nothing in lgml. Report the beta values lgml
      ignored in a response header, for example
      `x-lgml-ignored-betas: context-1m-2025-08-07,oauth-2025-04-20`.
      The body fields that pair with a beta get their own checks
      (below).
- [ ] Ignore unknown request headers (`x-stainless-*`,
      `x-claude-code-*`).
- [ ] Send `request-id: req_<random>` on every response, and the same
      value as `request_id` in error bodies.
- [ ] Answer `HEAD /api/hello` with 404 or 200; either is fine.
- [ ] Optional: in `GET /v1/models`, add `display_name`. Claude Code
      discovery keeps only IDs that contain `claude` or `anthropic`.

### Request validation

- [ ] Require `model`, `max_tokens` (integer, at least 0), and
      `messages`. `max_tokens: 0` should process the prompt (warm the
      KV cache) and return no content.
- [ ] Accept string or block-array `content` and `system`.
- [ ] Merge consecutive same-role messages into one turn, as the API
      does.
- [ ] Accept `role: "system"` inside `messages`. Render it as a system
      turn at that position when the chat template allows it. When the
      template cannot, fold it into the next turn with a clear marker,
      or reply 400 that names `role: "system"`. Claude Code then turns
      the feature off.
- [ ] A final `assistant` message is a prefill. Continue from it. With
      thinking on, reply 400, as Anthropic does.
- [ ] Reject empty text blocks (`text` min length 1) with 400.
- [ ] Reject unknown top-level fields and unknown block fields with
      400: `"<path>: Extra inputs are not permitted"`. Match the
      Anthropic wording so clients and logs read the same.
- [ ] Enforce the tool ordering rules: every `tool_use` in the last
      assistant turn needs a `tool_result` in the next user message,
      and results come before text. Use the Anthropic wording:
      `tool_use ids were found without tool_result blocks immediately after`.
- [ ] Accept any `tool_use.id` that matches `^[a-zA-Z0-9_-]+$`. Clients
      can replay IDs from other servers.
- [ ] Accept assistant turns with `tool_use` blocks and no thinking
      blocks. The clm client never sends thinking back.

### Images

- [ ] Accept `image` blocks in user content and inside
      `tool_result.content`.
- [ ] `base64` source: decode `data` as plain base64. A `data:` prefix
      in `data` is a 400. Check `media_type` against the bytes; a
      mismatch is a 400.
- [ ] Media types: `image/jpeg`, `image/png`, `image/gif`,
      `image/webp`. lgml decodes PNG and JPEG today. Add GIF (first
      frame) and WebP, or reply 400 that names the supported types.
      Never skip an image in silence.
- [ ] `url` source: lgml does not fetch http or https today. Reply 400:
      `image source type "url" is not supported by this server; send base64`.
      If fetching is added later, make it opt-in in the config.
- [ ] `file` source: reply 400 (no Files API).
- [ ] `transformations.oversized_image`: `"downsize"` (default) scales
      down to the vision encoder limit. `"error"` replies 400 that
      names the image size and the largest size that fits.
- [ ] Limits: at most 8000 x 8000 px; at most 2000 x 2000 px when the
      request has more than 20 images; at most 10 MB base64 per image;
      32 MB request (413 `request_too_large`). A model without a vision
      projector replies 400, as lgml does for chat completions.
- [ ] Count image tokens with the real encoder output, in both
      `usage.input_tokens` and count_tokens.
- [ ] `document` blocks: `text` source can render as text. PDF sources
      need a PDF path; until then, reply 400.

### Tools

- [ ] Accept custom tools (`type` absent or `"custom"`): `name`,
      `description`, `input_schema`, `cache_control`, `strict`,
      `eager_input_streaming`, `input_examples`.
- [ ] Reply 400 for any tool with a dated `type` (server tools,
      Anthropic-defined client tools). Put `Input tag '<type>'` in the
      message, for example
      `tools.2: Input tag 'web_search_20260209' found using 'type' does not match any of the expected tags: 'custom'`.
      Claude Code matches `Input tag '...'` for its advisor-tool retry.
- [ ] `tool_choice`: implement `auto`, `any`, `tool`, `none`, and
      `disable_parallel_tool_use` with grammar constraints. `strict:
      true` should force schema-valid `input` with the JSON-schema
      grammar lgml has.
- [ ] Emit `tool_use` blocks with `id` = `toolu_` + random base62,
      `input` as an object, and `stop_reason: "tool_use"`.
- [ ] Render `tool_result` content with text and images. Map
      `is_error: true` into the tool message for the template.

### Thinking

- [ ] Map `thinking`:
  - `adaptive` and absent: thinking on when the template has it (the
    lgml default).
  - `enabled` with `budget_tokens`: check `>= 1024` and
    `< max_tokens` (400 otherwise, as Anthropic). Enforce the budget
    as a cap on reasoning tokens. If lgml cannot cap it, reply 400
    rather than run without the cap.
  - `disabled`: thinking off.
  - `display`: `summarized` returns the reasoning text. `omitted`
    returns `thinking: ""` and a signature. `updates` is a 400 (beta
    not supported).
- [ ] Return reasoning as a `thinking` block before the `text` blocks.
      The text is the model's raw reasoning, not a summary. State that
      in the lgml docs.
- [ ] Signatures. A local model cannot make Anthropic-valid
      signatures. Use an opaque lgml signature:
  - Format: `lgml1.<payload>.<mac>`. `payload` is base64url of the
    reasoning text when `display` is `omitted`, or empty when the text
    is in the block. `mac` is base64url of
    HMAC-SHA256(server key, model id, 0x00, reasoning text).
  - Keep the server key stable across restarts (a file under the
    state directory), so replays verify after a restart.
  - On input, a block whose signature starts with `lgml1.` and
    verifies: use its reasoning (from `payload` when present, else from
    `thinking`). Pass it to the template as `reasoning_content`, so
    `preserve_thinking` applies.
  - A block whose signature starts with `lgml1.` and fails
    verification: reply 400
    ``messages.{i}.content.{j}: Invalid `signature` in `thinking` block``.
    Claude Code then removes thinking blocks and retries.
  - A foreign signature (any other prefix, for example a real
    Anthropic signature after the user switches the base URL): drop
    the block and continue. This matches the Anthropic rule for a block
    the model cannot read. Report the count in a response header, for
    example `x-lgml-thinking-dropped: 2`.
  - `redacted_thinking` blocks are always foreign: drop and count.
- [ ] Streaming thinking: `content_block_start` with
      `{"type":"thinking","thinking":"","signature":""}`, then
      `thinking_delta` events, then one `signature_delta`, then
      `content_block_stop`. With `omitted`, send one empty
      `thinking_delta`.
- [ ] Report `usage.output_tokens_details.thinking_tokens`.

### Sampling and output_config

- [ ] Honor `temperature`, `top_p`, `top_k`, and `stop_sequences`.
      lgml runs local models, so there is no reason to copy the
      Anthropic 400s for new model IDs.
- [ ] `output_config.effort`: Claude Code sends it for every unknown
      model ID and does not retry a 400 on it. Accept it. Map it to the
      template `reasoning_effort` when the template uses one. When the
      template has no such control, accept it and report
      `x-lgml-ignored: output_config.effort` in a response header.
- [ ] `output_config.format` (`json_schema`): implement with the lgml
      JSON-schema grammar.
- [ ] `metadata.user_id`: accept; log it at most.
- [ ] `service_tier`: accept `"auto"` and `"standard_only"`; return
      `usage.service_tier: "standard"`.

### Caching

- [ ] Accept `cache_control` on all allowed places, and top-level
      `cache_control`. Check at most 4 breakpoints and `ttl` values
      `5m` or `1h`.
- [ ] Map caching to the lgml KV prefix cache. Report prompt tokens
      reused from the KV cache as `cache_read_input_tokens`, and the
      rest as `input_tokens`. Keep
      `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`
      equal to the full prompt; clm and Claude Code sum them.

### Response and stream

- [ ] `id` = `msg_` + random; `type: "message"`; `role: "assistant"`;
      `model` = the lgml model name that ran.
- [ ] `stop_reason`: `end_turn`, `max_tokens`, `stop_sequence`,
      `tool_use`, and `model_context_window_exceeded` when the context
      fills before `max_tokens`. Never `pause_turn` or `refusal`.
      `stop_details: null`.
- [ ] Always send `usage` with `input_tokens` and `output_tokens`, in
      the full response, in `message_start`, and in every
      `message_delta` (cumulative).
- [ ] SSE: `content-type: text/event-stream`; `event:` line, one-line
      `data:`, blank line; no `[DONE]`. Flush after each event.
- [ ] Block indexes from 0, one block at a time, each with start, deltas,
      and stop.
- [ ] `tool_use` streaming: start with `"input": {}`, then
      `input_json_delta` fragments that join to the full JSON. Do not
      also put the input in `content_block_start`.
- [ ] Send `ping` events at least every 15 seconds while the model
      reads a long prompt or encodes images, and during long reasoning
      with `display: "omitted"`.
- [ ] An error after the 200 status: `event: error` with the error
      JSON, then close. Before the stream starts, use the HTTP status.

### Errors

- [ ] Use the body shape and status map from [Errors](#errors). Use
      `overloaded_error` (529) when all slots are busy and the request
      cannot queue, with `retry-after` in integer seconds.

### count_tokens

- [ ] Take the same body as `/v1/messages` without the generation
      fields. Render the prompt with the same template path as
      generation, including images, tools, and thinking blocks, and
      return `{"input_tokens": N}`.
- [ ] Accept and ignore `max_tokens`, `stream`, `stop_sequences`,
      `temperature`, and `metadata` there; they do not change the
      count. (See the open question on Anthropic's own behavior.)

### Fields safe to reject with 400 `invalid_request_error`

Reject these with a clear message that names the field. Never ignore
them in silence.

| Field or value | Suggested message |
| --- | --- |
| Tool `type` other than `custom` (server tools, `bash_*`, `text_editor_*`, `memory_*`, `computer_*`, `advisor_*`, `mcp_toolset`) | `Input tag '<type>' ... expected tags: 'custom'` |
| `mcp_servers` | `mcp_servers: not supported by this server` |
| `container` | `container: not supported by this server` |
| `context_management` | `context_management: not supported by this server`; tell Claude Code users to set `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` |
| `speed` | `speed: not supported by this server` |
| `inference_geo` | `inference_geo: not supported by this server` |
| `output_config.task_budget` | `output_config.task_budget: not supported by this server` |
| `thinking.display: "updates"` | `thinking.display: "updates" is not supported by this server` |
| `thinking.block_binding` | `thinking.block_binding: not supported by this server` |
| `image` or `document` with `source.type: "url"` or `"file"` | `messages.N.content.M.source: type "<t>" is not supported by this server; send base64` |
| `document` PDF sources, until PDF input exists | `messages.N.content.M: PDF documents are not supported by this server` |
| `citations` on documents | `citations: not supported by this server` |
| `defer_loading`, `allowed_callers`, `tool_reference` blocks | `<path>: tool search is not supported by this server` |
| `tool_addition`, `tool_removal`, `clear_at` in system messages | `<path>: not supported by this server` |
| Server tool result blocks in history (`server_tool_use`, `web_search_tool_result`, and the rest) | `<path>: block type "<t>" is not supported by this server` |
| Unknown top-level field | `<field>: Extra inputs are not permitted` |

## Pitfalls

- Anthropic has no data URLs. `source.data` is bare base64 and
  `media_type` is a separate field.
- Images can hide in `tool_result.content`. Count them toward the
  image limits and the vision check.
- `tool_result` blocks must come first in their user message. A text
  block before them is a 400 on Anthropic. Clients such as clm depend
  on text after results working.
- The Python SDK ignores SSE events without an `event:` line. A stream
  of bare `data:` lines, as in the OpenAI format, shows as an empty
  reply.
- `message_delta.usage.output_tokens` is cumulative. Sending
  increments makes clients undercount. Omitting `usage` breaks the
  Python SDK accumulator.
- Block indexes must be dense and in order; the SDK appends blocks in
  arrival order and looks them up by index.
- There is no `[DONE]`. Sending one breaks strict SSE parsers.
- The clm client ignores `event: error`. A mid-stream failure there
  shows as a truncated reply. Prefer to fail before the 200 when
  possible.
- Rejecting unknown `anthropic-beta` values, as Anthropic does, breaks
  Claude Code on every release that adds a beta.
- Rejecting `output_config.effort` or `thinking.type: "adaptive"` breaks
  Claude Code for any non-Claude model ID.
- Thinking signatures from Anthropic cannot verify on lgml. Dropping
  them keeps a conversation alive when a user moves a session from
  Anthropic to lgml. A 400 would end it for clients that do not strip
  thinking.
- A per-process random signature key invalidates every thinking block
  on restart. Keep the key on disk.
- The attribution block that Claude Code puts first in `system` reaches
  a local model and uses tokens. Set
  `CLAUDE_CODE_ATTRIBUTION_HEADER=0` when this matters.
- Claude Code aborts a stream after 300 seconds of silence. Long
  image encoding or prompt processing on a slow machine needs pings.
- The image token formula `width * height / 750` is out of date. Use
  the real encoder count.
- `temperature` and friends are gone from the typed Python SDK 1.x
  methods; Python users pass them with `extra_body`. Do not assume
  their absence means defaults were wanted.

## Open questions

These points have no primary source. Treat the doc text on them as a
proposal.

1. Whether Anthropic rejects `max_tokens`, `stream` or `metadata` on
   count_tokens. The SDK type omits them. This doc recommends that lgml
   accept and ignore them on that route only.
2. The exact error text that makes Claude Code disable mid-conversation
   system messages or `thinking` after a rejection. The gateway guide
   describes the behavior but not the text.
3. Whether the `url` image source accepts a `data:` URL. Also the exact
   errors for a base64 and `media_type` mismatch, for empty content, and
   for a fifth `cache_control` breakpoint. The doc marks these
   "observed".
4. Whether Claude Code sends tool-search fields (`defer_loading`,
   `tool_reference`) to a custom base URL by default.
5. Whether a streamed `redacted_thinking` block arrives complete in
   `content_block_start` with no deltas.
6. The TypeScript SDK stream code was not checked line by line. This doc
   assumes it matches the Python SDK.
