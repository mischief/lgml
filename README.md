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

- `docs/api.md`: the HTTP API reference.
- `ldoc .`: the Lua reference, in `build/doc`.
