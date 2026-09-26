# OpenAI Responses API: compatibility reference

Scope: `POST /v1/responses` and its companion endpoints, as clients use them
on 2026-09-26. The focus is multimodal input (images) and the parts that a
local server (lgml) must copy exactly so that openai-python, openai-node,
Codex CLI, and the clm client work.

Status markers:

- **(verified)**: read directly from a primary source listed below.
- **(data point)**: seen live or reported, but no primary source says it.
- **(uncertain)**: from memory or from a secondary source. Test it before you
  depend on it.
- **(changed)**: the current SDK differs from openai-openapi 2.3.0 or from
  older docs. The note gives the difference.

## Sources

Short tags used in this document:

| Tag | Source |
| --- | --- |
| `[py:X]` | openai-python type file, `https://github.com/openai/openai-python/blob/main/src/openai/types/responses/X` |
| `[py-shared:X]` | openai-python `src/openai/types/shared/X` |
| `[py-res]` | openai-python `src/openai/resources/responses/responses.py`, `input_items.py`, `input_tokens.py` |
| `[py-sse]` | openai-python `src/openai/_streaming.py` |
| `[py-acc]` | openai-python `src/openai/lib/streaming/responses/_responses.py` (stream accumulator) |
| `[node-sse]` | openai-node `src/core/streaming.ts` |
| `[codex-sse]` | openai/codex `codex-rs/codex-api/src/sse/responses.rs` |
| `[codex-req]` | openai/codex `codex-rs/codex-api/src/common.rs` and `codex-rs/core/src/client.rs` |
| `[codex-models]` | openai/codex `codex-rs/protocol/src/models.rs` |
| `[docs-vision]` | https://developers.openai.com/api/docs/guides/images-vision |
| `[docs-files]` | https://developers.openai.com/api/docs/guides/file-inputs |
| `[docs-fc]` | https://developers.openai.com/api/docs/guides/function-calling |
| `[docs-reasoning]` | https://developers.openai.com/api/docs/guides/reasoning |
| `[docs-state]` | https://developers.openai.com/api/docs/guides/conversation-state |
| `[docs-bg]` | https://developers.openai.com/api/docs/guides/background |
| `[docs-stream]` | https://developers.openai.com/api/docs/guides/streaming-responses and https://developers.openai.com/api/reference/resources/responses/streaming-events |
| `[community-stream]` | https://community.openai.com/t/responses-api-streaming-the-simple-guide-to-events/1363122 (secondary) |
| `[dynamo-206]` | https://github.com/ai-dynamo/frontend-crates/issues/206 (secondary, third-party server bug report) |
| `[OR]` | Open Responses (openresponses.org / github): spec `2026-04-24` and `schema/openapi.json` |
| `[2.3.0]` | openai-openapi 2.3.0 (stale OpenAPI spec) |
| `[clm]` | the clm client, `libclm/provider_responses.c` and `libclm/agent.c` |

The OpenAI HTML API reference pages did not render useful detail through a
fetch. The SDK type files are generated from the current OpenAPI spec
(header: "File generated from our OpenAPI spec"), so this document treats
them as the primary wire reference.

---

## 1. Endpoints

| Method and path | Purpose | Returns | Source |
| --- | --- | --- | --- |
| `POST /v1/responses` | Create a response. | `Response` object, or an SSE stream when `stream: true`. | [py-res] |
| `GET /v1/responses/{response_id}` | Get a stored response. Query: `include[]`, `include_obfuscation`, `starting_after` (int), `stream` (bool). | `Response`, or SSE stream. | [py:response_retrieve_params.py] |
| `DELETE /v1/responses/{response_id}` | Delete a stored response. | `{"id": "...", "object": "response.deleted", "deleted": true}` | [OR] `DeletedResponseResource.json`; the SDK casts the body to `None` [py-res]. [2.3.0] shows `"object": "response"` in its example (changed). |
| `POST /v1/responses/{response_id}/cancel` | Cancel a background response. Only responses created with `background: true` can be cancelled. Idempotent: a second call returns the final `Response`. | `Response` | [py-res], [docs-bg] |
| `GET /v1/responses/{response_id}/input_items` | List the input items of a stored response. Query: `after`, `include[]`, `limit` (1 to 100, default 20), `order` (`asc` or `desc`). | `{"object": "list", "data": [...], "first_id", "last_id", "has_more"}` | [py:input_item_list_params.py], [py:response_item_list.py] |
| `POST /v1/responses/input_tokens` | Count input tokens for a would-be request. Body: `model`, `input`, `instructions`, `tools`, `tool_choice`, `parallel_tool_calls`, `previous_response_id`, `conversation`, `reasoning`, `text`, `truncation`, `personality`. | `{"object": "response.input_tokens", "input_tokens": 123}` | [py:input_token_count_params.py], [py:input_token_count_response.py] |
| `POST /v1/responses/compact` | Compact a context window. Out of scope for lgml. | compacted items | [py-res] |

`/cancel`, `/input_tokens`, and `/compact` are absent from [2.3.0] (changed).

A WebSocket transport also exists (`response.create` messages over one
socket, same event objects). Codex uses it against OpenAI [codex-req]. [OR]
specifies it. lgml can skip it; Codex falls back to HTTP SSE for providers
without WebSocket support **(uncertain: check the Codex provider flag
before you depend on this)**.

### Authentication

- Header: `Authorization: Bearer <key>` [py-res] (`security={"bearer_auth": True}`).
- Optional headers from the SDKs: `OpenAI-Organization`, `OpenAI-Project`.
  A local server can ignore them.
- Request body: `Content-Type: application/json` [OR].

---

## 2. Request body: `POST /v1/responses`

All fields are optional in the SDK type (`TypedDict, total=False`)
[py:response_create_params.py]. In practice `model` and `input` are
required. OpenAI can omit `input` only when `prompt` or `conversation`
supplies context.

### 2.1 Core fields

| Field | Type | Default | Notes | Source |
| --- | --- | --- | --- | --- |
| `model` | string | none | Model id. | [py:response_create_params.py] |
| `input` | string, or array of input items | none | A string is the same as one user message with that text. | same |
| `instructions` | string or null | null | A system (or developer) message inserted in front of the context. With `previous_response_id`, the instructions of the previous response are **not** carried over. The `Response` echoes it as a string or as a list of input items. | same, [py:response.py] |
| `stream` | bool | false | SSE when true. | same |
| `stream_options` | `{include_obfuscation: bool}` | obfuscation on | Set only with `stream: true`. Codex also sends `reasoning_summary_delivery` (private) to OpenAI only. | same, [codex-req] |
| `max_output_tokens` | int or null | model limit | Includes reasoning tokens. | same |
| `temperature` | number 0 to 2 | 1 | OpenAI rejects it for some reasoning models **(uncertain)**. | same |
| `top_p` | number | 1 | | same |
| `top_logprobs` | int 0 to 20 | none | Logprobs appear only with `include: ["message.output_text.logprobs"]` **(uncertain: the include is documented, the coupling is inferred)**. | same, [py:response_includable.py] |
| `truncation` | `"auto"` or `"disabled"` | `"disabled"` | `disabled`: context overflow fails with 400. `auto`: drop items from the start. | same |
| `metadata` | map, up to 16 pairs | none | Keys up to 64 chars, values up to 512 chars. | same |
| `store` | bool | true | Store the response for later retrieval (30 days at OpenAI). | same, [docs-state] |
| `previous_response_id` | string | none | Continue from a stored response. Cannot be used with `conversation`. | same |
| `conversation` | string id, or `{id}` | none | Durable conversation object (`conv_...`). Items are prepended to input, and results are appended. | same |
| `background` | bool | false | Run asynchronously. Status starts at `queued`. | same, [docs-bg] |
| `include` | array of strings | none | See 2.8. | [py:response_includable.py] |
| `parallel_tool_calls` | bool | true | Response echoes it. | [py:response_create_params.py] |
| `max_tool_calls` | int | none | Limit on calls to **built-in** tools across the response. | same |
| `tools` | array | `[]` | See 2.5. | same |
| `tool_choice` | see 2.5 | `"auto"` | | same |
| `reasoning` | object | model default | See 2.7. | [py-shared:reasoning.py] |
| `text` | object | `{format: {type: "text"}}` | See 2.9. | [py:response_text_config_param.py] |
| `prompt_cache_key` | string | none | Cache routing hint. Replaces `user`. | [py:response_create_params.py] |
| `prompt_cache_options` | `{mode, ttl, prewarm, comparison_response_id}` | implicit | New (gpt-5.6 and later). **(changed)** | same |
| `prompt_cache_retention` | `"in_memory"` or `"24h"` | varies | Deprecated. | same |
| `safety_identifier` | string, up to 64 chars | none | Abuse tracking id. Replaces `user`. | same |
| `user` | string | none | Deprecated. | same |
| `service_tier` | `auto`, `default`, `flex`, `scale`, `priority`, `fast`, `ultrafast` | `auto` | The response echoes the tier actually used. | [py:service_tier.py] |
| `prompt` | `{id, version, variables}` | none | Stored prompt template. | [py:response_create_params.py] |
| `moderation`, `access_programs`, `context_management` | objects | none | OpenAI-specific. New. **(changed)** | same |

Codex also sends `client_metadata` (a string map). This field is not in the
public SDK types [codex-req]. A server that rejects unknown top-level fields
will break Codex.

### 2.2 Input items

`input` as an array is a list of items. Each item has a `type`. The message
item is the exception: `type: "message"` is optional on the "easy" message
form [py:easy_input_message_param.py]. The clm client omits `type` on
messages [clm].

Input item types in the current SDK union [py:response_input_param.py]:
`message` (three shapes, below), `function_call`, `function_call_output`,
`reasoning`, `item_reference`, `image_generation_call`, `computer_call`,
`computer_call_output`, `file_search_call`, `web_search_call`,
`code_interpreter_call`, `local_shell_call`, `local_shell_call_output`,
`shell_call`, `shell_call_output`, `apply_patch_call`,
`apply_patch_call_output`, `mcp_list_tools`, `mcp_approval_request`,
`mcp_approval_response`, `mcp_call`, `custom_tool_call`,
`custom_tool_call_output`, `tool_search_call`, `tool_search_output`,
`additional_tools`, `configuration_update`, `compaction`,
`compaction_trigger`, `program`, `program_output`.

A server needs these for function-tool clients: `message`, `function_call`,
`function_call_output`, `reasoning`, and (for stateful servers)
`item_reference`.

#### Message item shapes

1. Easy input message [py:easy_input_message_param.py]:

   ```json
   {"type": "message", "role": "user", "content": "text or parts"}
   ```

   - `role`: `user`, `assistant`, `system`, or `developer`.
   - `content`: string, or an array of input content parts.
   - `phase` (assistant only): `commentary` or `final_answer`. For
     gpt-5.3-codex and later, clients must preserve and resend it. New.
     **(changed)**

2. Input message with a status [py:response_input_param.py] `Message`:
   `role` is `user`, `system`, or `developer`. `content` must be a list.
   Carries an optional `status`.

3. Output message replay [py:response_output_message_param.py]:

   ```json
   {"type": "message", "id": "msg_...", "role": "assistant", "status": "completed",
    "content": [{"type": "output_text", "text": "...", "annotations": []}]}
   ```

   Content parts are `output_text` (`text`, `annotations`, optional
   `logprobs`) or `refusal` (`refusal`).

Clients replay assistant turns in different ways:

- openai-python users often append `response.output` items as they are,
  with `id`, `status`, and `annotations`.
- Codex sends `{"type": "message", "role": "assistant", "content":
  [{"type": "output_text", "text": "..."}]}` with no `annotations` and no
  `id` [codex-models] `ContentItem::OutputText`.
- The clm client sends `{"role": "assistant", "content": "..."}` (a string)
  [clm].

A server must accept all three forms. It must not require `id`, `status`, or
`annotations` on replayed assistant content.

OpenAI rejects `input_text` or `input_image` inside an assistant message
**(uncertain: from memory; the easy-message type does not forbid it)**.

#### Input content parts (user, system, developer messages)

Union: `input_text`, `input_image`, `input_file`
[py:response_input_message_content_list_param.py]. Codex also defines
`input_audio` [codex-models].

`input_text` [py:response_input_text_param.py]:

```json
{"type": "input_text", "text": "..."}
```

Optional: `prompt_cache_breakpoint: {"mode": "explicit"}` (new, on all three
part types). **(changed)**

`input_image` [py:response_input_image_param.py]:

| Field | Type | Notes |
| --- | --- | --- |
| `type` | `"input_image"` | required |
| `image_url` | string or null | A fully qualified URL (`https://...`) or a base64 data URL (`data:image/png;base64,...`). |
| `file_id` | string or null | A Files API id (`purpose: "vision"`). |
| `detail` | `low`, `high`, `auto`, `original` | Marked required in the message-content param type. Optional (nullable) in the tool-output variant [py:response_input_image_content_param.py]. Default `auto`. |

Notes:

- `detail: "original"` is current [py:image_detail.py], [docs-vision].
  [2.3.0] and [OR] list only `low`, `high`, `auto`. **(changed)**
- Exactly one of `image_url` or `file_id` must be set **(uncertain: implied
  by the docs, not stated as a rule)**.
- Codex always sets `detail`. Its default is `high` [codex-models]
  `DEFAULT_IMAGE_DETAIL`. Other clients can omit it. A server must default a
  missing `detail` to `auto`.

`input_file` [py:response_input_file_param.py], [docs-files]:

| Field | Notes |
| --- | --- |
| `file_id` | Files API id. |
| `file_url` | URL to fetch. |
| `file_data` | A data URL, for example `data:application/pdf;base64,...` [docs-files]. The tool-output variant says "base64-encoded data" [py:response_input_file_content_param.py]. Accept both the data-URL form and bare base64. |
| `filename` | Required with `file_data` [docs-files]. |
| `detail` | `auto`, `low`, `high`. New. **(changed)** |

For PDFs, OpenAI sends both the extracted text and page images to vision
models. For other file types, OpenAI sends only the text [docs-files].

### 2.3 Image media types and limits

From [docs-vision]:

- Types: PNG, JPEG (`.jpeg`, `.jpg`), WEBP, and non-animated GIF.
- Up to 512 MB total payload per request.
- Up to 1,500 images per request.
- Up to 65,535 pixels on either side (varies by model).
- Up to 30,000 patches per image after processing. OpenAI rejects larger
  images.
- `detail` behavior:
  - `low`: resize to fit 512x512.
  - `high`: up to 2,500 patches, 65,535 px maximum side.
  - `original`: keep the dimensions. Scale down only past the pixel-side
    limit. This is not resized to meet the patch limit.
  - `auto`: the model default.

Older guides gave 50 MB per request and 500 images per request **(changed;
the values are from memory of older docs)**.

File inputs: under 50 MB per file, 50 MB total per request [docs-files].

Image errors come as `error.code` values in `ResponseError`
[py:response_error.py]: `invalid_image`, `invalid_image_format`,
`invalid_base64_image`, `invalid_image_url`, `image_too_large`,
`image_too_small`, `image_parse_error`, `image_content_policy_violation`,
`invalid_image_mode`, `image_file_too_large`,
`unsupported_image_media_type`, `empty_image_file`,
`failed_to_download_image`, `image_file_not_found`.

### 2.4 Function call items in input

`function_call` (replay of a model call) [py:response_function_tool_call_param.py]:

```json
{"type": "function_call", "id": "fc_...", "call_id": "call_...",
 "name": "get_weather", "arguments": "{\"city\":\"Paris\"}", "status": "completed"}
```

- Required: `type`, `call_id`, `name`, `arguments` (a JSON string).
- Optional: `id`, `status`, `namespace`, `caller`, `async`.
- The clm client and Codex omit `id` and `status` [clm], [codex-models].

`function_call_output` [py:response_input_param.py] `FunctionCallOutput`:

```json
{"type": "function_call_output", "call_id": "call_...", "output": "..."}
```

- `output` (required): a string, **or** an array of `input_text`,
  `input_image`, and `input_file` parts
  [py:response_function_call_output_item_list_param.py].
- [2.3.0] allows a string only (`maxLength: 10485760`). **(changed)**
- [OR] allows both forms.
- Optional: `id`, `status`, `name`, `namespace`, `caller`.
- `call_id` is typed optional in the SDK. [2.3.0] and [OR] require it
  (1 to 64 chars). Treat it as required.
- The function-calling guide says: "you can pass an array of image or file
  objects instead of a string" [docs-fc].
- Data point: another agent verified live on 2026-09-26 that OpenAI
  accepts `output: [{"type": "input_text", ...}, {"type": "input_image",
  ...}]` with model `gpt-6-luna`. **(data point)**
- Codex sends list-form outputs with `input_text` and `input_image` parts
  (data URLs, `detail` set) [codex-models] `FunctionCallOutputContentItem`.
  Codex also defines `input_audio` and `encrypted_content` parts in that
  list. lgml must reject those with a 400 unless it supports them.

### 2.5 Tools and tool choice

Function tool (flat shape) [py:function_tool_param.py]:

```json
{"type": "function", "name": "get_weather", "description": "...",
 "parameters": {"type": "object", "properties": {...}, "required": [...]},
 "strict": true}
```

- This is flat. Chat Completions nests the same data:
  `{"type": "function", "function": {"name", "description", "parameters", "strict"}}`.
- `name`, `parameters`, `strict`, and `type` are marked required (nullable)
  in the SDK. `description` is optional.
- `strict`: Responses tries to make the schema strict when it can, and falls
  back to best effort when it cannot [docs-fc]. Strict needs
  `additionalProperties: false` and every property listed in `required`.
- New optional fields: `defer_loading`, `allowed_callers`, `output_schema`,
  `async`. **(changed)**
- The clm client sends `type`, `name`, `description` (an empty string when
  absent), and `parameters`. It never sends `strict` [clm].

Other tool types, listed only for reference [py:tool_param.py]:
`file_search`, `web_search`, `web_search_preview`, `computer`,
`computer_use_preview`, `code_interpreter`, `image_generation`, `mcp`,
`local_shell`, `shell`, `apply_patch`, `custom` (free-form text input),
`namespace`, `tool_search`.

`tool_choice` [py:tool_choice_options.py], [py:tool_choice_function_param.py], [py:tool_choice_allowed_param.py]:

- `"none"`, `"auto"` (default), or `"required"`.
- `{"type": "function", "name": "x"}`. This is flat. Chat uses
  `{"type": "function", "function": {"name": "x"}}`.
- `{"type": "allowed_tools", "mode": "auto" | "required", "tools": [{"type": "function", "name": "x"}, ...]}`.
- Built-in selectors, for example `{"type": "web_search"}`, and `mcp`,
  `custom`, `apply_patch`, `shell`, `programmatic_tool_calling`.

[OR] says a server must enforce `allowed_tools` as a hard constraint.

### 2.6 Reasoning items in input

[py:response_reasoning_item_param.py]:

```json
{"type": "reasoning", "id": "rs_...", "summary": [{"type": "summary_text", "text": "..."}],
 "content": [{"type": "reasoning_text", "text": "..."}], "encrypted_content": "gAAAA..."}
```

- Required: `type`, `id`, `summary` (may be `[]`).
- Optional: `content`, `encrypted_content`, `status`.
- Reasoning items returned with tool calls must be passed back with the tool
  outputs [docs-fc].

### 2.7 `reasoning` parameter

[py-shared:reasoning.py], [py-shared:reasoning_effort.py]:

| Field | Values |
| --- | --- |
| `effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Not every model supports every value. [2.3.0] lists only `low`, `medium`, `high` (changed). |
| `summary` | `auto`, `concise`, `detailed`. |
| `generate_summary` | Deprecated alias of `summary`. |
| `context` | `auto`, `current_turn`, `all_turns`. Controls which earlier reasoning items reach the model. gpt-5.6 defaults to `all_turns`. Earlier models default to `current_turn`. New. **(changed)** |
| `mode` | `standard`, `pro`, or a string. New. **(changed)** |

Codex always sends `reasoning`. `effort` and `summary` are omitted when not
configured [codex-req] `build_reasoning`. The clm client sends
`{"reasoning": {"effort": "<value>"}}` only when effort is configured [clm].

### 2.8 `include` values

[py:response_includable.py]:

- `reasoning.encrypted_content`: return `encrypted_content` on reasoning
  items for stateless use. The current SDK docstring says `encrypted_content`
  is now "populated by default for reasoning items returned by `POST
  /v1/responses`" [py:response_reasoning_item.py], and [docs-reasoning] says
  the same for `store: false` and ZDR. Codex still sends this include every
  time [codex-req]. **(changed; date unknown)**
- `message.output_text.logprobs`: add `logprobs` to `output_text` parts.
- `message.input_image.image_url`: include input image URLs in stored or
  listed items. This implies that listed input images omit `image_url` by
  default **(uncertain)**.
- `computer_call_output.output.image_url`, `file_search_call.results`,
  `web_search_call.results`, `web_search_call.action.sources`,
  `code_interpreter_call.outputs`: built-in tools only.

### 2.9 `text` parameter

[py:response_text_config_param.py], [py:response_format_text_json_schema_config_param.py]:

```json
{"text": {"format": {"type": "json_schema", "name": "person", "schema": {...}, "strict": true},
          "verbosity": "medium"}}
```

- `format.type`: `text` (default), `json_object`, or `json_schema`.
- `json_schema` is flat: `name` (a-z, A-Z, 0-9, `_`, `-`, up to 64 chars),
  `schema`, `strict`, and `description` sit next to `type`. Chat nests them
  under `response_format.json_schema`.
- `verbosity`: `low`, `medium` (default), or `high`.

---

## 3. Response object

[py:response.py]. The wire object:

| Field | Notes |
| --- | --- |
| `id` | `resp_...` |
| `object` | `"response"` |
| `created_at` | Unix seconds (a number). |
| `completed_at` | Unix seconds when status is `completed`, else null. New. **(changed)** |
| `status` | `completed`, `failed`, `in_progress`, `cancelled`, `queued`, `incomplete` [py:response_status.py]. [2.3.0] lists only four (no `queued` or `cancelled`). **(changed)** |
| `error` | null, or `{code, message}` [py:response_error.py]. |
| `incomplete_details` | null, or `{reason}` with `reason` one of `max_output_tokens`, `max_messages`, `content_filter`, `steered`. [2.3.0] has only the first and third. **(changed)** |
| `model` | string |
| `instructions` | string, list of items, or null |
| `output` | array of output items (3.1) |
| `usage` | 3.2. Null until the response is complete. |
| `parallel_tool_calls`, `tool_choice`, `tools`, `temperature`, `top_p`, `max_output_tokens`, `max_tool_calls`, `previous_response_id`, `reasoning`, `text`, `truncation`, `metadata`, `store`, `background`, `service_tier`, `prompt_cache_key`, `safety_identifier`, `user`, `top_logprobs`, `conversation` | Echo of the effective request settings. |

[OR] `ResponseResource.json` marks these required (use `null` when not
set): `id`, `object`, `created_at`, `completed_at`, `status`,
`incomplete_details`, `model`, `previous_response_id`, `instructions`,
`output`, `error`, `tools`, `tool_choice`, `truncation`,
`parallel_tool_calls`, `text`, `top_p`, `presence_penalty`,
`frequency_penalty`, `top_logprobs`, `temperature`, `reasoning`, `user`,
`usage`, `max_output_tokens`, `max_tool_calls`, `store`, `background`,
`service_tier`, `metadata`, `safety_identifier`, `prompt_cache_key`.
Emit all of them.

`output_text` is not a wire field. openai-python defines it as a computed
`@property` that joins every `output_text` part of every `message` item
[py:response.py]. openai-node adds the same convenience after parsing. A
server should not send it **(verified for Python; for Node, from memory)**.

### 3.1 Output items

`message` [py:response_output_message.py], [py:response_output_text.py]:

```json
{"type": "message", "id": "msg_...", "status": "completed", "role": "assistant",
 "content": [{"type": "output_text", "text": "Hi.", "annotations": [], "logprobs": []}]}
```

- `status`: `in_progress`, `completed`, or `incomplete`.
- A content part is `output_text` or `{"type": "refusal", "refusal": "..."}`.
- `annotations`: `file_citation`, `url_citation`,
  `container_file_citation`, or `file_path`. A local server sends `[]`.
- `logprobs`: a list of `{token, bytes, logprob, top_logprobs: [{token,
  bytes, logprob}]}`.
- `phase` (optional): `commentary` or `final_answer`.

`function_call` [py:response_function_tool_call.py]:

```json
{"type": "function_call", "id": "fc_...", "call_id": "call_...", "name": "get_weather",
 "arguments": "{\"city\":\"Paris\"}", "status": "completed"}
```

- `id` has the `fc_` prefix. `call_id` has the `call_` prefix [docs-fc].
- `call_id` links the call to its `function_call_output`.
- `arguments` is always a JSON string.

`reasoning` [py:response_reasoning_item.py]:

```json
{"type": "reasoning", "id": "rs_...", "summary": [{"type": "summary_text", "text": "..."}],
 "content": [{"type": "reasoning_text", "text": "..."}], "encrypted_content": "...", "status": "completed"}
```

`summary` is always present (may be `[]`). `content` holds raw reasoning
text when a provider exposes it. OpenAI models do not expose it; open-weight
servers do.

Model image output: `image_generation_call` [py:response_output_item.py]
`ImageGenerationCall`:

```json
{"type": "image_generation_call", "id": "ig_...", "status": "completed",
 "result": "<base64 image>", "output_format": "png", "size": "1024x1024",
 "quality": "high", "background": "opaque", "revised_prompt": "...", "action": "generate"}
```

Streaming adds `response.image_generation_call.in_progress`, `.generating`,
`.partial_image` (`partial_image_b64`, `partial_image_index`), and
`.completed` [py:response_image_gen_call_partial_image_event.py]. The `ig_`
prefix is **(uncertain)**. This item appears only with the built-in
`image_generation` tool. A client can also send it back as an input item.
Assistant `message` items never carry images.

Other output items (built-in tools, MCP, shell, apply_patch, custom tool,
compaction, program) exist [py:response_output_item.py]. A local server with
function tools only never emits them.

### 3.2 Usage

[py:response_usage.py]:

```json
{"input_tokens": 120, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
 "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 12}, "total_tokens": 160}
```

- `input_tokens` includes cached tokens. `cached_tokens` is a subset.
- `cache_write_tokens` is new and typed required (`int`) in openai-python.
  [dynamo-206] reports pydantic validation failures when a server omits it.
  Codex defaults it to 0 [codex-sse]. Always send it. **(changed)**
- `output_tokens` includes `reasoning_tokens`.
- `total_tokens` = `input_tokens` + `output_tokens`.
- In a stream, only the terminal event carries real usage [community-stream].

---

## 4. Streaming

### 4.1 Framing

- The response has `Content-Type: text/event-stream` [OR].
- Each event is sent as:

  ```
  event: <type>\n
  data: <one-line JSON>\n
  \n
  ```

  The `event:` value equals the JSON `type` [OR]. The SSE `id:` field is
  not used [OR].
- Every data object carries `type` and `sequence_number`
  [py:response_created_event.py] and all event types. `sequence_number` is
  an integer that increases by one per event. It starts at 0
  **(uncertain: observed in transcripts; no document states the start
  value)**. `GET /v1/responses/{id}?stream=true&starting_after=N` resumes a
  background stream after event N [docs-bg].
- Delta events (`*.delta`) carry an `obfuscation` string of random padding
  unless `stream_options.include_obfuscation` is false [py:response_create_params.py],
  [OR] schema. The openai-python event types do not declare the field.
  Clients ignore it. lgml can omit it.
- Terminal sentinel:
  - OpenAI does not send `data: [DONE]` on Responses streams. The stream
    closes after the terminal event (`response.completed`,
    `response.incomplete`, or `response.failed`) [dynamo-206],
    [community-stream]. This is **(uncertain)** because only secondary
    sources state it. No primary OpenAI source defines a sentinel for
    Responses.
  - [OR] spec 2026-04-24 says the terminal event "MUST be the literal
    string `[DONE]`". This conflicts with OpenAI.
  - Client tolerance: openai-python stops at `[DONE]` [py-sse].
    openai-node stops at `[DONE]` [node-sse]. Codex skips data that is not
    JSON [codex-sse]. The clm client ignores it.
  - Recommendation for lgml: do not send `[DONE]`. Close the stream after
    the terminal event.
- Clients that stream require a terminal event. Codex reports "stream
  closed before response.completed" if the connection ends first
  [codex-sse].

### 4.2 Event order: text reply

Order from [OR] item lifecycle rules, [py-acc], and [community-stream]:

1. `response.created` (`response.status: "in_progress"`, `output: []`)
2. `response.in_progress`
3. `response.output_item.added` (`item`: message, `status: "in_progress"`, `content: []`)
4. `response.content_part.added` (`part`: `{"type": "output_text", "text": "", "annotations": []}`)
5. `response.output_text.delta` (repeat)
6. `response.output_text.done` (full `text`)
7. `response.content_part.done` (full `part`)
8. `response.output_item.done` (full `item`, `status: "completed"`)
9. `response.completed` (full `response` with `output` and `usage`)

Event field sets (from the SDK event types):

| Event | Fields besides `type` and `sequence_number` |
| --- | --- |
| `response.created`, `response.in_progress`, `response.queued`, `response.completed`, `response.incomplete`, `response.failed` | `response` |
| `response.output_item.added`, `response.output_item.done` | `output_index`, `item` |
| `response.content_part.added`, `response.content_part.done` | `item_id`, `output_index`, `content_index`, `part` (`output_text`, `refusal`, or `reasoning_text`) |
| `response.output_text.delta` | `item_id`, `output_index`, `content_index`, `delta`, `logprobs` (list, send `[]`) |
| `response.output_text.done` | `item_id`, `output_index`, `content_index`, `text`, `logprobs` |
| `response.refusal.delta` / `.done` | `item_id`, `output_index`, `content_index`, `delta` / `refusal` |
| `response.function_call_arguments.delta` | `item_id`, `output_index`, `delta` |
| `response.function_call_arguments.done` | `item_id`, `output_index`, `arguments`. The SDK type and [OR] have no `name`. |
| `response.reasoning_summary_part.added` / `.done` | `item_id`, `output_index`, `summary_index`, `part` (`{"type": "summary_text", "text"}`) |
| `response.reasoning_summary_text.delta` / `.done` | `item_id`, `output_index`, `summary_index`, `delta` / `text` |
| `response.reasoning_text.delta` / `.done` | `item_id`, `output_index`, `content_index`, `delta` / `text` |
| `response.output_text.annotation.added` | `item_id`, `output_index`, `content_index`, `annotation_index`, `annotation` |
| `error` | See 4.6. |

### 4.3 Event order: function call

1. `response.created`, `response.in_progress`
2. (optional reasoning item, see 4.4)
3. `response.output_item.added` with `item`: `{"type": "function_call", "id": "fc_...", "call_id": "call_...", "name": "...", "arguments": "", "status": "in_progress"}`
4. `response.function_call_arguments.delta` (repeat)
5. `response.function_call_arguments.done` (full `arguments`)
6. `response.output_item.done` (full item, `status: "completed"`)
7. More `function_call` items for parallel calls, each with its own `output_index`.
8. `response.completed`

The `name` and `call_id` must be present in `output_item.added`:

- [OR] says the added event must carry at least `name`.
- The clm client reads `call_id` and `name` only from `output_item.added`
  [clm].
- The argument delta events do not repeat them.

### 4.4 Event order: reasoning

Summary reasoning (OpenAI models):

1. `response.output_item.added` (`{"type": "reasoning", "id": "rs_...", "summary": []}`)
2. `response.reasoning_summary_part.added` (`summary_index: 0`)
3. `response.reasoning_summary_text.delta` (repeat)
4. `response.reasoning_summary_text.done`
5. `response.reasoning_summary_part.done`
6. Steps 2 to 5 repeat for each summary part.
7. `response.output_item.done` (full item, with the final `encrypted_content`)

Then the message or function-call items follow.

The `encrypted_content` in `output_item.added` may be incomplete. Clients
must use the value from `output_item.done` [py:response_reasoning_item.py].

Raw reasoning (open-weight models): `response.content_part.added` with
`part: {"type": "reasoning_text", "text": ""}`, then
`response.reasoning_text.delta`, `response.reasoning_text.done`, and
`response.content_part.done`. These events are in the SDK union
[py:response_stream_event.py], [py:response_content_part_added_event.py].
Codex reads `response.reasoning_text.delta` [codex-sse]. The clm client
reads only `response.reasoning_summary_text.delta` [clm].

### 4.5 Terminal events

- `response.completed`: `status: "completed"`.
- `response.incomplete`: `status: "incomplete"`, with
  `incomplete_details.reason`. For example, `max_output_tokens` can end the
  response with no visible output [docs-reasoning].
- `response.failed`: `status: "failed"`, with `response.error: {code,
  message}`.

Clients treat each of these as the end of the stream.

### 4.6 The `error` event

The shapes disagree:

- openai-python type [py:response_error_event.py]: flat,
  `{"type": "error", "sequence_number": N, "code": "...", "message": "...", "param": null}`.
- [OR] `ErrorStreamingEvent`: nested,
  `{"type": "error", "sequence_number": N, "error": {"type", "code", "message", "param"}}`.
- Codex reads the nested `error` object [codex-sse].
- openai-python raises `APIError` for any stream event whose data has a
  truthy top-level `error` key [py-sse].
- openai-node throws `APIError` when the SSE `event:` is `error`, or when the
  data has an `error` key [node-sse].
- [OR] says a `response.failed` event follows any error during streaming.
- The clm client ignores the `error` event. It learns about the failure
  only from `response.failed` [clm].

Pitfall: every `response.*` event embeds the response, and the response
object has an `error` field. The SDK checks only the top level of `data`,
so a nested `response.error` does not trigger the SDK check.

Recommendation for lgml: emit
`{"type": "error", "sequence_number": N, "code": c, "message": m, "param": p, "error": {"type": t, "code": c, "message": m, "param": p}}`.
Then emit `response.failed` with the same `code` and `message` in
`response.error`. Then close the stream. **(uncertain: this combined shape
is a proposal)**

### 4.7 Errors before streaming starts

Request validation happens before the server sends the SSE headers. A
validation failure returns a normal HTTP error with a JSON body, not an SSE
stream.

---

## 5. Errors (HTTP)

Body shape [2.3.0] `Error`, [OR]:

```json
{"error": {"message": "Unsupported parameter: 'background'.", "type": "invalid_request_error",
           "param": "background", "code": "unsupported_parameter"}}
```

- Required keys: `message`, `type`, `param` (nullable), `code` (nullable).

Status codes:

| Status | Use |
| --- | --- |
| 400 | `invalid_request_error`: bad field, bad value, unsupported feature, or context overflow with `truncation: "disabled"`. |
| 401 | Bad or missing key. |
| 403 | No permission, for example a missing access program [py:response_create_params.py]. |
| 404 | Unknown model, or unknown response id on GET, DELETE, cancel, or input_items. |
| 429 | Rate limit. |
| 500 | Server error. |
| 503 | Overloaded. |

[OR] maps its types this way: `invalid_request` 400, `not_found` 404,
`too_many_requests` 429, `server_error` 500, `model_error` 500.

Unknown `previous_response_id` returns code `previous_response_not_found`
with `param: "previous_response_id"` and message "Previous response with id
'resp_abc' not found." [OR] (WebSocket section). The HTTP status for this
case is 400 at OpenAI **(uncertain)**.

---

## 6. Statefulness

- `store` defaults to true. OpenAI keeps stored responses for 30 days
  [py:response_create_params.py], [docs-state].
- With `previous_response_id`, the server loads the input and output of the
  prior response (recursively, the whole chain) and puts them before the new
  `input` [OR], [docs-state].
- `instructions` do not carry over [py:response_create_params.py].
- All earlier tokens count again as input tokens [docs-state].
- With `store: false`, the server keeps nothing. A later request cannot use
  `previous_response_id` to reach that response. It fails with
  `previous_response_not_found` [OR]. The exception is the WebSocket mode,
  which keeps connection-local state.
- Stateless clients (`store: false`, the Codex pattern) resend the full
  history every turn:
  - They append every item of `response.output` (reasoning, message, and
    function_call) to the next `input`.
  - They add `function_call_output` items after the calls.
  - Reasoning items must keep `id`, `summary`, and `encrypted_content`. The
    server uses `encrypted_content` to restore the hidden reasoning
    [docs-reasoning], [py:response_reasoning_item.py].
- `item_reference` (`{"type": "item_reference", "id": "msg_..."}`) points to
  a stored item by id.

---

## 7. The clm client

Source: `libclm/provider_responses.c` and `libclm/agent.c` [clm].

Fields it sends:

- `model`, `input`, `stream`.
- `max_output_tokens`: always 64000.
- `previous_response_id`: set from the id of the prior response once the
  earlier messages are on the server.
- `reasoning: {effort}`: only when effort is configured.
- `tools`: flat function tools with `type`, `name`, `description`, and
  `parameters`. No `strict`.
- `tool_choice: "auto"`: sent only when tools are present. The "forbid
  tools" path replaces it with `"none"`.
- `parallel_tool_calls: false`: sent only when parallel calls are disabled.

It never sends `store` (so the server default applies), `instructions`,
`include`, `text`, `temperature`, or `stream_options`.

How it builds `input`:

- Every message is `{"role": r, "content": "<string>"}`, with no `type` and
  no content parts. System prompts go as `role: "system"` messages, not as
  `instructions`.
- An assistant turn with tool calls becomes one `function_call` item per
  call (`call_id`, `name`, `arguments`), with no `id` or `status`. Any text
  that went with those calls is dropped.
- A tool result becomes
  `{"type": "function_call_output", "call_id": ..., "output": "<string>"}`.
  It always uses the string form.
- Images: the clm client sends none. It has no `input_image` or
  `input_file` path, and tool results are always strings.

Reasoning:

- Non-stream: it joins `summary[].text` of `reasoning` items into
  `reasoning_content`.
- Stream: it maps `response.reasoning_summary_text.delta` to
  `reasoning_content`.
- It never echoes reasoning items back. It depends on
  `previous_response_id` (server state) to keep reasoning context.
- It reads neither `encrypted_content` nor `content` (`reasoning_text`).

Non-stream parsing:

- Reads `output[]`: `message` (`output_text` parts), `reasoning`
  (summary), and `function_call` (`call_id`, `name`, `arguments`).
- Reads `id` (kept for chaining).
- Reads `status`: `completed` maps to stop; `incomplete` maps to length,
  or to content_filter from `incomplete_details.reason`; `failed` and
  `cancelled` map to error, with the message from `error.message`. Any
  function_call item maps to tool_calls.
- Reads `usage.input_tokens`, `usage.output_tokens`, and
  `usage.input_tokens_details.cached_tokens`.

Stream events it handles:

- `response.output_text.delta`
- `response.reasoning_summary_text.delta`
- `response.output_item.added` (function_call only: it reads `call_id`,
  `name`, and `output_index`)
- `response.function_call_arguments.delta` (keyed by `output_index`)
- `response.completed`, `response.incomplete`, `response.failed` (`id`,
  `status`, `usage`, `error.message`)

It ignores all other events, including `error`, `output_item.done`,
`content_part.*`, and `function_call_arguments.done`.

Chain fallback: when a request fails and the error body contains the text
`previous_response`, the clm client drops the chain and resends the full
history [clm] `agent.c`. A server that rejects `previous_response_id` with a
400 whose body names `previous_response_id` still works. The cost is two
requests on every turn after the first.

---

## 8. Examples

IDs are shortened. Headers are omitted except where they matter.

### 8.1 Plain text, non-streaming

Request:

```http
POST /v1/responses
Authorization: Bearer sk-local
Content-Type: application/json

{"model": "qwen3-vl-8b", "instructions": "Be brief.", "input": "Say hi."}
```

Response (200):

```json
{
  "id": "resp_01",
  "object": "response",
  "created_at": 1790000000,
  "completed_at": 1790000001,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "model": "qwen3-vl-8b",
  "instructions": "Be brief.",
  "previous_response_id": null,
  "output": [
    {"type": "message", "id": "msg_01", "status": "completed", "role": "assistant",
     "content": [{"type": "output_text", "text": "Hi.", "annotations": [], "logprobs": []}]}
  ],
  "tools": [], "tool_choice": "auto", "parallel_tool_calls": true,
  "temperature": 1.0, "top_p": 1.0, "top_logprobs": 0, "presence_penalty": 0, "frequency_penalty": 0,
  "max_output_tokens": null, "max_tool_calls": null, "truncation": "disabled",
  "text": {"format": {"type": "text"}, "verbosity": "medium"},
  "reasoning": {"effort": null, "summary": null},
  "store": true, "background": false, "service_tier": "default", "metadata": {},
  "user": null, "safety_identifier": null, "prompt_cache_key": null,
  "usage": {"input_tokens": 14, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 17}
}
```

### 8.2 Image by data URL

```json
{
  "model": "qwen3-vl-8b",
  "input": [
    {"role": "user", "content": [
      {"type": "input_text", "text": "What is in this image?"},
      {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...", "detail": "auto"}
    ]}
  ]
}
```

The response has the same shape as in 8.1. An https URL works the same way
(`"image_url": "https://example.com/cat.jpg"`). A Files API reference uses
`"file_id": "file-abc"` in place of `image_url`.

### 8.3 Tool call round trip (stateless)

Turn 1 request:

```json
{
  "model": "m", "store": false,
  "tools": [{"type": "function", "name": "get_weather", "description": "Current weather.",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                            "required": ["city"], "additionalProperties": false},
             "strict": true}],
  "input": [{"role": "user", "content": "Weather in Paris?"}]
}
```

Turn 1 response (extract):

```json
{"id": "resp_02", "object": "response", "status": "completed",
 "output": [{"type": "function_call", "id": "fc_02", "call_id": "call_02", "name": "get_weather",
             "arguments": "{\"city\":\"Paris\"}", "status": "completed"}], "...": "..."}
```

`status` is `completed` even though the model stopped for a tool call. The
client detects a tool call from the `function_call` item.

Turn 2 request:

```json
{
  "model": "m", "store": false, "tools": ["...same..."],
  "input": [
    {"role": "user", "content": "Weather in Paris?"},
    {"type": "function_call", "id": "fc_02", "call_id": "call_02", "name": "get_weather",
     "arguments": "{\"city\":\"Paris\"}", "status": "completed"},
    {"type": "function_call_output", "call_id": "call_02", "output": "{\"temp_c\":18,\"sky\":\"clear\"}"}
  ]
}
```

Stateful variant: turn 2 sends
`{"model": "m", "previous_response_id": "resp_02", "tools": [...], "input":
[{"type": "function_call_output", "call_id": "call_02", "output": "..."}]}`.
Turn 1 must have been stored.

### 8.4 Tool result with an image (list form)

```json
{
  "model": "m",
  "input": [
    {"role": "user", "content": "Take a screenshot and describe it."},
    {"type": "function_call", "call_id": "call_03", "name": "screenshot", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "call_03", "output": [
      {"type": "input_text", "text": "Screenshot of display :0"},
      {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo...", "detail": "high"}
    ]}
  ]
}
```

This form is accepted by OpenAI (the data point from 2026-09-26, model
`gpt-6-luna`) and by the current SDK types. Codex sends it. [2.3.0] would
reject it.

### 8.5 Streaming with a tool call

Request: as in 8.3 turn 1, plus `"stream": true`. Response headers:
`Content-Type: text/event-stream`.

```
event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_04","object":"response","created_at":1790000000,"status":"in_progress","output":[],"usage":null,"error":null,"incomplete_details":null,"model":"m","...":"..."}}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_04","object":"response","status":"in_progress","output":[],"usage":null,"...":"..."}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"function_call","id":"fc_04","call_id":"call_04","name":"get_weather","arguments":"","status":"in_progress"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":3,"item_id":"fc_04","output_index":0,"delta":"{\"city\":"}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","sequence_number":4,"item_id":"fc_04","output_index":0,"delta":"\"Paris\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","sequence_number":5,"item_id":"fc_04","output_index":0,"arguments":"{\"city\":\"Paris\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":6,"output_index":0,"item":{"type":"function_call","id":"fc_04","call_id":"call_04","name":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"}}

event: response.completed
data: {"type":"response.completed","sequence_number":7,"response":{"id":"resp_04","object":"response","status":"completed","completed_at":1790000001,"output":[{"type":"function_call","id":"fc_04","call_id":"call_04","name":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"}],"usage":{"input_tokens":60,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":9,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":69},"...":"..."}}

```

The server closes the connection after `response.completed`. It sends no
`[DONE]`.

A text reply streams the same way. In place of the function-call events,
it has `output_item.added` (message), `content_part.added`,
`output_text.delta` repeated, `output_text.done`, `content_part.done`, and
`output_item.done`:

```
event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":3,"item_id":"msg_05","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_05","output_index":0,"content_index":0,"delta":"Hi","logprobs":[]}
```

### 8.6 Reasoning reply with summary and `encrypted_content`, and echo back

Request:

```json
{"model": "m", "store": false, "include": ["reasoning.encrypted_content"],
 "reasoning": {"effort": "medium", "summary": "auto"},
 "input": [{"role": "user", "content": "Is 1001 prime?"}]}
```

Stream (extract):

```
event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"reasoning","id":"rs_06","summary":[]}}

event: response.reasoning_summary_part.added
data: {"type":"response.reasoning_summary_part.added","sequence_number":3,"item_id":"rs_06","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","sequence_number":4,"item_id":"rs_06","output_index":0,"summary_index":0,"delta":"Checking small factors; 7 x 143 = 1001."}

event: response.reasoning_summary_text.done
data: {"type":"response.reasoning_summary_text.done","sequence_number":5,"item_id":"rs_06","output_index":0,"summary_index":0,"text":"Checking small factors; 7 x 143 = 1001."}

event: response.reasoning_summary_part.done
data: {"type":"response.reasoning_summary_part.done","sequence_number":6,"item_id":"rs_06","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":"Checking small factors; 7 x 143 = 1001."}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":{"type":"reasoning","id":"rs_06","summary":[{"type":"summary_text","text":"Checking small factors; 7 x 143 = 1001."}],"encrypted_content":"gAAAAABo...","status":"completed"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":8,"output_index":1,"item":{"type":"message","id":"msg_06","status":"in_progress","role":"assistant","content":[]}}

... content_part.added, output_text.delta..., output_text.done, content_part.done, output_item.done ...

event: response.completed
data: {"type":"response.completed","sequence_number":15,"response":{"id":"resp_06","status":"completed","output":[{"type":"reasoning","id":"rs_06","summary":[...],"encrypted_content":"gAAAAABo..."},{"type":"message","id":"msg_06","...":"..."}],"usage":{"input_tokens":12,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":210,"output_tokens_details":{"reasoning_tokens":180},"total_tokens":222},"...":"..."}}

```

Next turn (stateless echo back). The client copies the reasoning item
exactly, including `encrypted_content`:

```json
{"model": "m", "store": false, "include": ["reasoning.encrypted_content"],
 "reasoning": {"effort": "medium", "summary": "auto"},
 "input": [
   {"role": "user", "content": "Is 1001 prime?"},
   {"type": "reasoning", "id": "rs_06", "summary": [{"type": "summary_text", "text": "Checking small factors; 7 x 143 = 1001."}],
    "encrypted_content": "gAAAAABo..."},
   {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "No. 1001 = 7 x 11 x 13."}]},
   {"role": "user", "content": "And 1009?"}
 ]}
```

Open-weight variant: the reasoning item carries
`"content": [{"type": "reasoning_text", "text": "<raw thinking>"}]` and
streams through `response.reasoning_text.delta`. `summary` stays `[]`
unless the server makes a summary.

---

## 9. What lgml needs

### 9.1 Checklist

Transport and framing:

- [ ] `POST /v1/responses` accepts `Authorization: Bearer`. Local mode may
  accept any key.
- [ ] Non-stream: `application/json` with the full `Response` object
  (section 3, every [OR]-required key, null when not set).
- [ ] Stream: `text/event-stream`. Each event is `event: <type>\ndata:
  <json>\n\n`, and the `event:` value equals `type`.
- [ ] Every event has `type` and `sequence_number`. The number starts at 0
  and increases by one.
- [ ] The first event is always `response.created`. The openai-python
  accumulator raises an error on any other first event [py-acc].
- [ ] Send `response.in_progress` right after `response.created`.
- [ ] For each output item, send `output_item.added` before any event for
  that item. Send `content_part.added` before any text delta. openai-python
  raises an error when a content event arrives before its item [py-acc].
- [ ] `output_index` is the position of the item in `output`. It stays the
  same for all events of that item. Reasoning, message, and function_call
  items share one index space. The clm client keys tool calls by it [clm].
- [ ] `function_call` `output_item.added` carries `name`, `call_id`, `id`,
  and `arguments: ""`.
- [ ] `output_item.done` carries the complete item. Codex builds its history
  only from `output_item.done` [codex-sse].
- [ ] End with exactly one of `response.completed`, `response.incomplete`,
  or `response.failed`, with the full response and `usage`. Then close the
  connection. Do not send `[DONE]`.
- [ ] On a failure during generation, send `error` (shape in 4.6), then
  `response.failed`.
- [ ] Validation errors before the stream starts are HTTP 4xx with a JSON
  error body, not SSE.

Input handling:

- [ ] `input` is a string or an array. A message item without `type` but
  with `role` is a message.
- [ ] `content` is a string or a list of parts.
- [ ] Roles: `user`, `assistant`, `system`, `developer`. Map `developer` to
  the system slot of the chat template, unless the template has a
  developer role.
- [ ] `instructions` goes before all input as a system message. Do not carry
  it through `previous_response_id`.
- [ ] Assistant replay accepts `output_text` parts with or without
  `annotations`, `id`, `status`, `logprobs`, and `phase`. It also accepts
  plain string content. It accepts `refusal` parts.
- [ ] `input_image`:
  - Accept `image_url` as a data URL (`data:image/{png,jpeg,webp,gif};base64,...`).
  - Accept `image_url` as `https://`, or reject it with a clear 400 when
    URL fetch is off.
  - Reject `file_id` with a 400 unless lgml implements a Files store.
  - Accept `detail` values `low`, `high`, `auto`, and `original`, and
    default to `auto`.
  - Enforce the media types. Use OpenAI-style codes in messages
    (`unsupported_image_media_type`, `invalid_base64_image`).
- [ ] `input_file`: implement it, or reject it with a 400 that names
  `input[i].content[j]`.
- [ ] `function_call` items: `id` and `status` are optional. `arguments` is
  a string.
- [ ] `function_call_output.output`: accept a string **and** a list of
  `input_text` and `input_image` parts. Put the images in the tool-result
  turn of the chat template, in order. This is the main multimodal path for
  Codex.
- [ ] Reject list parts that lgml cannot handle (`input_file`,
  `input_audio`, `encrypted_content`) with a 400. Do not drop them.
- [ ] Match each `function_call_output.call_id` to an earlier
  `function_call.call_id`. A mismatch is a 400.
- [ ] `reasoning` input items: accept them. Use `content[].text` if
  present. Use `encrypted_content` if lgml issued it (see the next item).
  Otherwise, the chat template decides whether earlier reasoning is shown.
- [ ] `encrypted_content`: lgml cannot encrypt in the OpenAI format. Issue
  an lgml-owned opaque string, for example a version prefix, the base64 of
  the raw reasoning text, and an HMAC. Accept it back. Reject foreign
  (OpenAI) blobs with a 400 and `param` set to the item path.

Tools:

- [ ] Flat function tools. Reject the Chat nested shape
  (`{"type": "function", "function": {...}}`) with a 400 that explains the
  flat shape. This is a common porting mistake.
- [ ] `tool_choice`: support `none`, `auto`, `required`, and
  `{"type": "function", "name"}`. Support `allowed_tools`, or reject it.
- [ ] `parallel_tool_calls: false`: at most one call per response.
- [ ] Generate `fc_` item ids and `call_` call ids. Make them unique per
  response at least.

Output:

- [ ] Ids: `resp_`, `msg_`, `fc_`, `call_`, `rs_`.
- [ ] `status` is `completed` for tool-call turns. There is no separate
  "tool_calls" status.
- [ ] On the token limit: `status: "incomplete"`,
  `incomplete_details: {"reason": "max_output_tokens"}`, and the message
  item status is `incomplete`.
- [ ] `usage` always has `input_tokens_details.cached_tokens`,
  `input_tokens_details.cache_write_tokens`, and
  `output_tokens_details.reasoning_tokens` (use 0 when unknown).
- [ ] Do not add `output_text` on the wire.
- [ ] `text.format`: support `text`. Support `json_object` and flat
  `json_schema` through grammar-constrained decoding, or reject them.
  `text.verbosity`: accept it and map it to a prompt hint, or reject it.
- [ ] `max_output_tokens`, `temperature`, `top_p`: apply them.

Statefulness (choose one and document it):

- Option A (recommended): a bounded in-memory store.
  - With `store` true or omitted, keep the resolved input and the output of
    each response, keyed by `resp_` id, with an LRU and TTL limit.
  - Support `previous_response_id` by prepending the stored chain.
  - Support `GET /v1/responses/{id}` and `DELETE`.
  - Support `GET .../input_items` (`data`, `first_id`, `last_id`,
    `has_more`; `limit` 1 to 100, default 20; `order` default `desc`).
  - An unknown id returns 404 on GET and DELETE. On create, it returns a 400
    with `code: "previous_response_not_found"` and
    `param: "previous_response_id"`.
  - This keeps the clm client on its short-request path.
- Option B: stateless only.
  - Accept `store: true` but store nothing.
  - Reject any `previous_response_id` with a 400: code
    `previous_response_not_found`, param `previous_response_id`, and a
    message that says lgml does not store responses.
  - The clm client falls back to full history, but each turn costs two
    requests. Codex is unaffected (`store: false` and no
    `previous_response_id` on HTTP).
  - GET, DELETE, and input_items return 404 with a clear message.
- Either way: `conversation` returns a 400 unless implemented.
  `background: true` returns a 400 unless implemented. `/cancel` returns
  400 or 404 for non-background responses.
- `POST /v1/responses/input_tokens`: easy to support with the tokenizer.
  Returns `{"object": "response.input_tokens", "input_tokens": N}`.

### 9.2 Fields to reject with a 400 when unsupported

Never ignore these without an error. Each changes behavior that a client
depends on. Use `type: "invalid_request_error"`, `code:
"unsupported_parameter"` or `"unsupported_value"`, and `param` set to the
field path.

- `background: true`
- `conversation`
- `previous_response_id` (in Option B)
- `prompt` (stored prompt templates)
- `tools[]` entries with `type` other than `function` (`web_search`,
  `file_search`, `code_interpreter`, `image_generation`, `mcp`, `computer*`,
  `shell`, `local_shell`, `apply_patch`, `custom`, `namespace`,
  `tool_search`)
- `tool_choice` forms other than `none`, `auto`, `required`, and function
  (unless `allowed_tools` is implemented)
- Chat-style nested function tools
- `text.format` of `json_object` or `json_schema` without constrained
  decoding
- `include` values other than `reasoning.encrypted_content` and (if
  implemented) `message.output_text.logprobs`
- `top_logprobs` > 0 when logprobs are not implemented
- `input_image.file_id` and `input_file` (without a Files store); an
  `https` `image_url` when fetch is off
- Input item types other than `message`, `function_call`,
  `function_call_output`, `reasoning`, and `item_reference` (the last one
  only with a store)
- `reasoning.effort` values the model cannot honor. The alternative is to
  map them and document the mapping. `reasoning.context` and
  `reasoning.mode` values other than `auto` and `standard` count here too.
- `truncation: "auto"` (if lgml cannot drop items safely)
- `context_management`, `moderation`, `prompt_cache_options.prewarm: true`
- `stream_options` without `stream: true` (OpenAI documents this as "only
  set this when you set `stream: true`")

Fields that are safe to accept without an effect, because they do not
change the output:

- `metadata` (echo it)
- `user`, `safety_identifier`, `prompt_cache_key`
- `client_metadata` (Codex)
- `service_tier` values `auto` and `default`. Echo `default`.
- `stream_options.include_obfuscation`
- `store` (Option B, as documented above)
- `max_tool_calls` (it limits built-in tools only; lgml has none)
- `parallel_tool_calls: true`
- `include: ["reasoning.encrypted_content"]`

Document this list.

### 9.3 Pitfalls

1. Nested versus flat: Responses tools, `tool_choice`, and
   `text.format.json_schema` are flat. The Chat Completions shapes are
   nested. Porting code from the Chat handler breaks all three.
2. Top-level `error` key in a stream event: openai-python raises `APIError`
   for any stream data with a truthy top-level `error` [py-sse]. Never put
   a non-null top-level `error` on normal events. Keep `error` inside
   `response`.
3. `[DONE]`: OpenAI does not send it on Responses streams. [OR] requires
   it. The SDKs tolerate it. Strict typed clients do not [dynamo-206]. Omit
   it.
4. Event order: a missing `response.created`, `output_item.added`, or
   `content_part.added` crashes the openai-python stream helper [py-acc].
5. `cache_write_tokens` missing from usage fails openai-python validation
   [dynamo-206].
6. `function_call_output.output` can be a list with images. A server built
   from openai-openapi 2.3.0 accepts only strings and rejects valid Codex
   traffic.
7. `detail: "original"` is valid. Older schemas and [OR] do not list it.
8. `input_image.detail` is marked required in the SDK message param but
   optional elsewhere. Default it. Do not reject a missing value.
9. A message item without `type` is valid (the clm client sends these).
10. Tool-call turns end with `status: "completed"`. Clients detect tool calls
    from `function_call` items.
11. `encrypted_content` in `output_item.added` may be partial. Put the final
    value in `output_item.done` and in `response.completed`.
12. `instructions` do not carry through `previous_response_id`. A stored
    chain must not replay the old instructions.
13. The clm client needs `name` and `call_id` in `output_item.added`. It
    ignores `function_call_arguments.done` and `output_item.done`.
14. The clm client ignores the `error` event. Always follow it with
    `response.failed`, or the clm client sees a stream with no end.
15. Codex sends `client_metadata` and, to OpenAI only, private
    `stream_options` fields. Strict unknown-field rejection breaks Codex.
    Accept `client_metadata`.
16. The openai-python and Open Responses `error` event shapes differ (flat
    versus nested). Send both forms in one object (4.6).

---

## 10. Differences: current SDK versus openai-openapi 2.3.0

| Area | openai-openapi 2.3.0 | Current SDK (2026-09) |
| --- | --- | --- |
| `function_call_output.output` | string only (max 10 MiB) | string, or list of `input_text` / `input_image` / `input_file` |
| `input_image.detail` | `low`, `high`, `auto` | adds `original` |
| `reasoning.effort` | `low`, `medium`, `high` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `reasoning` fields | `effort`, `summary`, `generate_summary` | adds `context`, `mode` |
| `Response.status` | `completed`, `failed`, `in_progress`, `incomplete` | adds `cancelled`, `queued` |
| `incomplete_details.reason` | `max_output_tokens`, `content_filter` | adds `max_messages`, `steered` |
| Stream events | no `sequence_number` | `sequence_number` on every event; adds `reasoning_text.*`, `response.queued`, image generation, MCP, and shell events |
| `include` | no `reasoning.encrypted_content` or logprobs | both present; `encrypted_content` now returned by default |
| Endpoints | no `/cancel` or `/input_tokens` | both present, plus `/compact` |
| Usage | `cached_tokens` only | adds `cache_write_tokens` |
| Request fields | none of: `text.verbosity`, `max_tool_calls`, `prompt_cache_key`, `safety_identifier`, `conversation`, `background` | all present |
| Delete response | example `object: "response"` | [OR]: `object: "response.deleted"` |

---

## 11. Open questions

These points have no primary source. Treat the doc text on them as a
proposal.

1. `data: [DONE]` at the end of a stream. Secondary sources say OpenAI
   does not send it. The Open Responses spec requires it. openai-python,
   openai-node and Codex accept both. This doc recommends that lgml omit
   it.
2. The shape of the `error` stream event. openai-python types it flat
   (`code`, `message`, `param`). Open Responses and Codex expect a nested
   `error` object. The combined shape in 4.6 is a proposal.
3. Not confirmed: `sequence_number` starts at 0; the HTTP status for
   `previous_response_not_found`; whether OpenAI rejects `input_text` or
   `input_image` in assistant messages; the `ig_` id prefix; whether
   openai-node adds `output_text` as openai-python does; whether logprobs
   appear only with `include: ["message.output_text.logprobs"]`.
4. The date on which `encrypted_content` became a default output.
5. lgml design choice: send open-weight raw reasoning as `content:
   [reasoning_text]` with `response.reasoning_text.*` events (Codex reads
   these), copy it into `summary` (the clm client reads only this), or
   both.
