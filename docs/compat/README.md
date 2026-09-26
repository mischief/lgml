# API compatibility references

These documents describe three hosted LLM HTTP APIs as clients use them
in September 2026. They focus on images. Each one ends with a "What lgml
needs" checklist. Each point cites a source. Points without a primary
source are marked "observed" or "unverified".

| Document | API | Route |
| --- | --- | --- |
| [openai-chat.md](openai-chat.md) | OpenAI Chat Completions | `POST /v1/chat/completions` |
| [openai-responses.md](openai-responses.md) | OpenAI Responses | `POST /v1/responses` and `/v1/responses/{id}` routes |
| [anthropic-messages.md](anthropic-messages.md) | Anthropic Messages | `POST /v1/messages`, `POST /v1/messages/count_tokens` |

The primary sources are the vendor docs (developers.openai.com,
platform.claude.com), the openai-python and anthropic-sdk-python type
files, and openai-openapi. openai-openapi 2.3.0 is stale. Each OpenAI
document lists where 2.3.0 differs from the current SDK types.

## How each API carries images

| | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| Image part in user input | `{"type":"image_url","image_url":{"url":…,"detail":…}}` | `{"type":"input_image","image_url":…,"detail":…}` (`image_url` is a string, not an object) | `{"type":"image","source":{…}}` |
| By URL | `url`: https URL | `image_url`: https URL | `source: {"type":"url","url":…}` |
| By base64 | `url`: `data:image/…;base64,…` | `image_url`: `data:image/…;base64,…` | `source: {"type":"base64","media_type":…,"data":…}`, bare base64, no data URL |
| By file id | No image by file id (the `file` part is for documents) | `file_id` in `input_image` | `source: {"type":"file","file_id":…}` |
| `detail` | `auto` (default), `low`, `high`; the vision guide also lists `original`, but the SDK literal does not | `auto` (default), `low`, `high`, `original` | None. The server downscales long edges over its cap. |
| Media types | png, jpeg, webp, non-animated gif | same as Chat | jpeg, png, gif, webp |
| Roles that can hold images | `user` only | `user` messages, and function call outputs | `user` messages, and `tool_result` blocks |
| Image in a tool result | No. The tool message allows text parts only. OpenAI accepts an image part there, but the model ignores it (observed). A `user` message after the tool batch that carries the image works. | Yes. `function_call_output.output` is a string or a list of `input_text`, `input_image` and `input_file`. openai-openapi 2.3.0 allows a string only. Verified live with gpt-6-luna. | Yes. `tool_result.content` is a string or a list of `text`, `image`, `document` and `search_result` blocks. `tool_result` blocks come first in the user message. |
| Image in model output | Never. | Only as an `image_generation_call` item from the built-in image generation tool (`result` is base64). Message items never contain images. | Never. |
| Per-request limits | 512 MB payload, 1,500 images (vision guide) | same as Chat | 32 MB request, 10 MB base64 per image; 100 images (200K-context models) or 600; max 8000x8000 px, or 2000x2000 when there are more than 20 images |

## Other differences a server must get right

| | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| Auth | `Authorization: Bearer` | `Authorization: Bearer` | `x-api-key` or `Authorization: Bearer`; `anthropic-version` is required |
| System prompt | `system` or `developer` message | `instructions`, or a `system` or `developer` message item | top-level `system`; `role: "system"` in `messages` on current models |
| Tool definition | nested: `{"type":"function","function":{name,…}}` | flat: `{"type":"function","name":…,"parameters":…}` | `{name, description, input_schema}` |
| Tool call arguments | JSON string | JSON string | JSON object (`input`) |
| Tool call id prefix | `call_` | `call_` (`call_id`); item id `fc_` | `toolu_` |
| Tool call finish | `finish_reason: "tool_calls"` | `status: "completed"` with `function_call` items | `stop_reason: "tool_use"` |
| Structured output | `response_format.json_schema.{name,schema,strict}` (nested) | `text.format` `{type:"json_schema",name,schema,strict}` (flat) | `output_config.format` |
| Reasoning request | `reasoning_effort` | `reasoning.{effort,summary}` | `thinking` (`enabled` with `budget_tokens`, or `adaptive`), `output_config.effort` |
| Reasoning output | none from OpenAI; `reasoning_content` is a de facto extension of other servers | `reasoning` items (`rs_`) with `summary` and `encrypted_content` | `thinking` blocks with `signature`; `redacted_thinking` |
| SSE framing | `data: {json}` only; ends with `data: [DONE]` | `event: <type>` and `data:`; every event has `sequence_number`; no `[DONE]` from OpenAI | `event: <type>` and `data:`; `ping` events; no `[DONE]` |
| Usage in stream | final chunk with `choices: []` when `stream_options.include_usage` | in `response.completed` | `message_start` (input), cumulative `message_delta.usage.output_tokens` |
| Error body | `{"error":{"message","type","param","code"}}` | same as Chat; in-stream `error` event then `response.failed` | `{"type":"error","error":{"type","message"},"request_id"}`; 529 for overloaded |
| State | stateless | `store`, `previous_response_id`, `conversation` | stateless |
