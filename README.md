# lgml

lgml runs models on ggml from Lua 5.4. A C module (`lgml.so`) gives
tensors, graphs, backends, samplers, tokenizers and codecs. Lua code
on top of it loads gguf and safetensors models: chat models (llama,
Qwen3, Qwen3.5 and Gemma 4 with vision), Whisper, text-to-speech models
and Z-Image. `lgml-server` serves them over an OpenAI-shaped HTTP API.
Each model runs in its own worker process, which loads on demand under
a memory budget.

## Build

You need Lua 5.4, ggml (from the system, or `-Dggml=subproject`),
libpng, libjpeg and zlib. The Lua rocks install into `lua_modules`:

```sh
luarocks --lua-version 5.4 --tree lua_modules install --only-deps lgml-scm-1.rockspec
meson setup build
ninja -C build
```

## Run

```sh
bin/lgml-server -m model.gguf                 # one model, port 8080
bin/lgml-server -m model.gguf --mmproj mmproj.gguf -c 16384
bin/lgml-server -C etc/lgml.lua               # several models from a config
```

`bin/lgml-server -h` lists the options. `etc/lgml.lua.example` is a
sample config with chat, vision, Whisper, image and speech models.

## Logs

The server and its workers write one line per event to stderr, in
`key=value` form:

```
ts=2026-09-26T10:02:18.647Z level=info comp=http rid=605156c1 msg=access method=POST path=/v1/chat/completions status=200 ms=121 in=148 out=384 model=qwen3-14b prompt_tokens=14 completion_tokens=3 tok_s=41.2 finish=length
```

`comp` is the component, such as `http`, `router`, `pool`, `openai`,
or `worker:<model id>/engine` for a line from a worker. `rid` is the
request id. Each reply has it in the `x-lgml-request-id` header, so
`grep rid=<id>` finds all lines of a request.

The levels are error, warn, info, debug and trace:

- info (the default): one `access` line per request, with the status,
  time, bytes, model, token counts, tokens per second and the error
  message of a failed request; model loads and unloads.
- debug: the request body, the router choice and why, memory budget
  checks, the sampling settings, the reply, tool calls, and the full
  prompt after the chat template (`msg="chat prompt"`). An image in
  the prompt shows as `<image 1 img:0a1b2c3d 48213B 256 tokens>`. A
  decision logs the prompt of each question and option order, the
  candidate tokens, the raw log-scores, the prior run and the recipe.
- trace: each token of a chat, and each graph build or replay.

`-v` sets debug and `-vv` trace. `--log-level`, the config key
`log_level` and the environment variable `LGML_LOG` set a level by
name. Data URLs and base64 never print in full: they show as kind,
size and a sha256 prefix. A value longer than 2 KiB is cut, except a
prompt. `--log-file path` appends to a file; SIGHUP opens it again.
`--dump-prompts dir` writes each rendered chat and decision prompt to
`dir/<request id>.chat.txt` or `dir/<request id>.decide.txt`.

## Tools

- `bin/lgml-finetune`: train a LoRA adapter on JSONL chat or text data.
  The adapter loads in lgml-server and llama.cpp with `--lora`.
- `bin/lgml-repl`: a Lua REPL with lgml on the path.

## Tests

```sh
meson test -C build --setup cpu
```

The `cpu` setup hides the Vulkan driver, so the tests use the CPU. The
`model` suite needs model files. Give them as meson options (`-Dmodel=`,
`-Dwhisper=`, `-Dwhisper_wav=`, `-Dorpheus=`, `-Dqwen3tts=`,
`-Dvibevoice=`) or environment variables (`LGML_DIFFUSION_GEMMA`,
`LGML_ZIMAGE`, `LGML_ZIMAGE_VAE`, `LGML_GEMMA4_VISION`,
`LGML_VISION_IMAGE`), then run:

```sh
meson test -C build --suite model
```

A test without its model skips.

## Docs

- `docs/api.md`: the HTTP API reference. A running server also gives
  its OpenAPI document at `GET /openapi.json`.
- `ldoc .`: the Lua reference, in `build/doc`.
