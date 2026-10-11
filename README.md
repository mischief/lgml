# lgml

lgml runs models on ggml from Lua 5.4. A C module (`lgml.so`) gives
tensors, graphs, backends, samplers, tokenizers and codecs. Lua code
on top of it loads gguf and safetensors models: chat models (llama,
Qwen3, Qwen3.5 and Gemma 4 with vision), embedding models
(Qwen3-Embedding), Whisper, text-to-speech models and Z-Image. `lgml-server` serves them over an OpenAI-shaped HTTP API;
chat models also serve the Anthropic Messages and OpenAI Responses
APIs, so Claude Code, Codex and the Anthropic and OpenAI SDKs work
with no change.
Each model runs in its own worker process, which loads on demand under
a memory budget.

## Build

You need Lua 5.4, a C compiler, meson, cmake, the Vulkan headers,
glslc, libpng, libjpeg, zlib and utf8proc. Meson builds ggml from
`subprojects/ggml.wrap` into `lgml.so`, static, with the lgml patch:
fused ops and zero-copy experts need it. `-Dggml=system` links the
system ggml in place; lgml then runs without those two. It builds the
other four from the wraps in `subprojects` when the system lacks them.
liburing is optional: with it, the n-gram table reader reads through an
io_uring. The default build type is debugoptimized.

### Packages

Debian 13 (all of this builds ggml from the wrap):

```sh
apt install build-essential git meson ninja-build cmake pkg-config \
    lua5.4 liblua5.4-dev luarocks libpng-dev libjpeg62-turbo-dev \
    zlib1g-dev libutf8proc-dev libvulkan-dev glslc glslang-tools \
    spirv-headers mesa-vulkan-drivers liburing-dev
```

Gentoo:

```sh
emerge dev-lang/lua:5.4 dev-lua/luarocks dev-build/meson dev-build/ninja \
    dev-build/cmake media-libs/libpng media-libs/libjpeg-turbo sys-libs/zlib \
    dev-libs/libutf8proc media-libs/vulkan-loader dev-util/vulkan-headers \
    media-libs/shaderc dev-util/glslang dev-util/spirv-headers sys-libs/liburing
```

The GPU needs a Vulkan driver (Mesa RADV for AMD). Without one, lgml
runs on the CPU.

Older luarocks (3.8 on Debian 13) runs on Lua 5.1, so the hooks
build type of picohttpparser must be installed for that Lua too. Prefer
luarocks 3.12 or later: older versions on Lua 5.1 run bytecode that a
server sends in place of a rockspec.

```sh
luarocks --local install luarocks-build-hooks
eval "$(luarocks path)"; eval "$(luarocks --lua-version 5.4 path --append)"
luarocks --lua-version 5.4 --local make lgml-scm-1.rockspec
```

Install with luarocks. The rock builds with meson and installs
`lgml-server`, `lgml-worker`, `lgml-repl`, `lgml-finetune` and `lgml-convert`:

```sh
luarocks --lua-version 5.4 install lgml-scm-1.rockspec
```

For work in the source tree, the Lua rocks install into `lua_modules`:

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
bin/lgml-server --engine lgml.laya_engine -m laya/   # Laya decisions, /v1/systemone
```

`bin/lgml-server -h` lists the options. `etc/lgml.lua.example` is a
sample config with chat, vision, embedding, Whisper, image and speech
models.

### Weights in host memory

A chat model that does not fit in GPU memory can keep part of its
weights in host memory, as llama.cpp does. The CPU runs the ops on
those weights; the GPU runs the rest, and the ggml scheduler copies
activations between them. For prompts, the GPU copies the CPU weights
and runs the large batches itself.

```sh
bin/lgml-server -m moe.gguf --cpu-moe          # routed experts on the CPU
bin/lgml-server -m moe.gguf --n-cpu-moe 20     # experts of the first 20 layers
bin/lgml-server -m moe.gguf --n-cpu-moe auto   # as few layers as fit
bin/lgml-server -m model.gguf -ngl 30          # the last 30 layers on the GPU
bin/lgml-server -m moe.gguf -ot 'blk%.[0-9]%.ffn_.*exps=CPU'
```

`-ot` takes a Lua pattern (as `string.find` uses), not the regular
expression that llama.cpp's `-ot` takes. A plain name such as `exps`
works in both. For a llama.cpp recipe, change these:

- The escape character is `%`, not `\`: `blk\.1` becomes `blk%.1`.
- `%d` replaces `\d`; there are no `{n,m}` counts.
- There is no alternation. Write one rule for each branch:
  `-ot 'ffn_(up|down)_exps=CPU'` becomes
  `-ot 'ffn_up_exps=CPU,ffn_down_exps=CPU'`.

Model specs in a
config take the keys `cpu_moe`, `n_cpu_moe`, `n_gpu_layers` and
`override`. The host part counts against `ram`, the rest against
`vram`; with `n_cpu_moe = "auto"` the server picks the count that fits
`vram`. A mixture-of-experts model with its experts on the CPU keeps
a large part of its GPU speed: on a Ryzen 9 7950X with a 32 GB GPU,
gemma-4-26B-A4B Q4_K_M runs 24 tokens/s from 3 GB of GPU memory, and
34 tokens/s from 10 GB with `--n-cpu-moe 15`.

### KV cache and slots

A chat model has one KV pool of `-c` cells for all its requests. `-c`
is also the most tokens of one request. The pool starts with
`--kv-initial` cells (default 4096) and doubles as sequences need, up
to `-c`; `--kv-initial 0` gives all cells at load. The model pool
counts the whole pool, `-c` positions, against the memory budget.

`-np N` (`--parallel`, config key `slots`) sets the most
sequences the model keeps. A slot is one sequence: a token list and
the pool cells that hold its keys and values, in blocks of 64 cells. A
sequence takes blocks as it grows. A request takes the slot that
shares the longest prefix with its prompt; a prompt that shares less
than 32 tokens with every slot takes an empty slot or the least
recently used one. Clients that take turns keep their caches. So a
short one-off request takes another slot and a few blocks, and a long
chat keeps its prefix. Slots cost no cells of their own; only the
recurrent states (qwen35, qwen3next) and the sliding-window rings
(gemma4) are per slot, and the model pool counts them once per slot.
The default is 4 slots, or fewer when the states and rings of 4 slots
would take more than a quarter of the pool: gemma-4-31B at 64k keeps
1 slot (its rings take about 640 MiB each), Qwen3.8-27B at 128k keeps
4 (about 150 MiB each).

A full block gets a key: a hash of the key of the block before it and
of its own tokens. When a slot lets a keyed block go, the block stays
cached with its cells until the pool needs it. A request whose prompt
starts with known blocks shares them with no copy, even from another
slot or after its own slot went to another client; a block that it
shares only in part gives it a copy of those cells. Two clients with
one system prompt hold it once.

A block holds no recurrent state (qwen35, qwen3next) and no ring
(gemma4). For these models, the state checkpoints that a slot takes at
turn starts (`--ctx-checkpoints`) also go to a state index in host RAM,
keyed by the block that ends at or before the checkpoint and the
tokens after it. A request can then start from cached blocks and the
deepest checkpoint that its prompt matches; the cells between the last
block edge and the checkpoint come from a copy of a cached block. The
last partial block of such a slot stays cached when the slot lets it
go, for that copy. The index shares the `--cache-ram` budget with the
prompt store.

When the pool has no free block, it takes cached blocks, the least
recently used first and the last blocks of a prefix before the first
ones. A keyed block that the pool takes goes to the RAM prompt store
(`--cache-ram`) first, and a later prompt that needs it copies it back.
Then idle slots give their blocks up, the least recently used first. An
evicted slot goes to the RAM prompt store too, so its next request
restores it from host memory.

Each request has a priority: low for decisions (`/v1/systemone`),
normal for chats, and high for a chat with `prompt_cache_retention`
`24h`. A slot has the priority of its last request, and a block the
highest priority of the requests that used it. The pool takes cached
blocks and idle slots of low priority first, then normal, then high;
within one priority, the least recently used first. A new prompt that
shares little with every slot takes an empty slot, else the idle slot
of the lowest priority. So a burst of one-off checks evicts its own
blocks before the prefix of a long chat.

A prompt that does not fit next to the running requests waits for one
of them to end. When running requests fill the pool, the request that
gets no block lets its cells go and waits for another request to end;
a running request of lower priority does so for it first. Its cells
stay cached, and a checkpoint of its state goes to the state index, so
it takes them up again and runs on, and its reply is whole. Only a
request that runs alone and fills the pool ends with finish reason
`length`; `-c` bounds that.

With two or more slots, requests run at the same time: prompt chunks
(up to 512 tokens) and decode steps of all requests go into one batched
pass. Matmuls and experts run on all rows. Attention of each sequence
reads the pool window of its blocks under a mask that shows only its
own cells, and recurrent layers run per sequence on their states. A
request that runs alone takes its slot's own graph, at the same speed
as one slot. Decode is bound by memory bandwidth, so a pass
of several slots costs little more than one. On a 32 GB Radeon, with
200 tokens per request:

| model | 1 request | 2 at once | 3 at once |
|---|---|---|---|
| gemma-4-26B-A4B | 89 tokens/s | ~120 total | ~155 total |
| Ornith-1.5-35B (qwen35moe) | 114 | 149 | 168 |

Prefill is bound by compute, so batched prompts finish about as fast as
queued ones; they only progress together. On a GPU, a batched pass can
round differently from a single one, so a greedy reply can change at a
near tie, as in llama.cpp. A prompt with images prefills alone.

### Memory

With a config, each model counts against the budget of its backend.
Before its first load, the count is an estimate: the files, the KV
cache at `ctx` and 1 GiB. After a load, the worker reports the exact
parts, and the count is their sum:

- weights: the buffers of the weights and other leaves
- cache: the KV pool at `ctx` cells, and the state of each slot
- compute: the largest compute buffers, which the worker plans at
  load (a chat model: prompt chunks of 512 tokens over all `ctx`
  cells, alone and one per batched slot, plus an image of the most
  tokens; an image model: its largest side)
- overhead: the driver memory of the worker beyond its buffers, from
  the DRM fdinfo on Linux, at least 256 MiB on a GPU

A worker refuses a load that does not fit the free device memory less
`mem_margin` (default 256 MiB), and a KV pool grow or graph that does
not fit either. Such a load or request gets status 507. `GET
/v1/models` and `/metrics` give the parts. Speech, transcription and
embedding engines do not plan their compute buffers; they count the
estimate margin in place of them.

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

- `bin/lgml-finetune`: train a LoRA adapter on JSONL chat or text data,
  or with `--dpo` on preference pairs (chosen and rejected answers). The
  adapter loads in lgml-server and llama.cpp with `--lora`.
- `bin/lgml-repl`: a Lua REPL with lgml on the path.
- `bin/lgml-convert`: convert a Hugging Face safetensors model to gguf
  as bf16, q8_0 or a llama.cpp K-quant mix (`-t q4_K_M`). The
  architecture table in `lgml.convert` maps names and metadata; it
  knows `llada_moe_v2`. `scripts/convert_check.lua` compares the
  output with its source.

## Tests

```sh
meson test -C build --setup cpu
```

The `cpu` setup hides the Vulkan driver, so the tests use the CPU. The
`model` suite needs model files. Give them as meson options (`-Dmodel=`,
`-Dwhisper=`, `-Dwhisper_wav=`, `-Dorpheus=`, `-Dqwen3tts=`,
`-Dvibevoice=`) or environment variables (`LGML_DIFFUSION_GEMMA`,
`LGML_ZIMAGE`, `LGML_ZIMAGE_VAE`, `LGML_GEMMA4_VISION`,
`LGML_CLEF_VISION`, `LGML_VISION_IMAGE`), then run:

```sh
meson test -C build --suite model
```

A test without its model skips.

## Docs

- `docs/api.md`: the HTTP API reference. A running server also gives
  its OpenAPI document at `GET /openapi.json`.
- `ldoc .`: the Lua reference, in `build/doc`.
