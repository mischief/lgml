project('lgml',
  'c',
  version : '0.1.0',
  meson_version : '>=1.4.0',
  default_options : [
    'c_std=c17',
    'warning_level=3',
    'werror=true',
    'buildtype=debugoptimized',
  ],
)

cc = meson.get_compiler('c')
m = cc.find_library('m', required : false)
threads = dependency('threads')

add_project_arguments(cc.get_supported_arguments([
  '-Wvla',
  '-Wshadow',
  '-Wstrict-prototypes',
  '-Wmissing-prototypes',
  '-Wformat=2',
]), language : 'c')

lua = dependency('lua5.4', required : false)
if not lua.found()
  lua = dependency('lua54')
endif

# The cmake package names ggml-base, which holds the op API. The
# pkg-config file names only libggml.
ggml = dependency('', required : false)
if get_option('ggml') != 'subproject'
  ggml = dependency('ggml', method : 'cmake', modules : ['ggml::ggml', 'ggml::ggml-base'],
    required : get_option('ggml') == 'system')
endif
if not ggml.found()
  # The wrap pins a ggml revision; it builds static, with Vulkan and the CPU backend.
  cmake = import('cmake')
  ggml_opts = cmake.subproject_options()
  ggml_opts.add_cmake_defines({
    'BUILD_SHARED_LIBS' : false,
    'CMAKE_POSITION_INDEPENDENT_CODE' : true,
    'CMAKE_BUILD_TYPE' : 'Release',
    'GGML_VULKAN' : true,
    'GGML_NATIVE' : true,
    'GGML_BUILD_TESTS' : false,
    'GGML_BUILD_EXAMPLES' : false,
  })
  # Vendored code: new compiler warnings must not stop the build.
  ggml_opts.set_override_option('werror', 'false')
  ggml_proj = cmake.subproject('ggml', options : ggml_opts)
  ggml = declare_dependency(dependencies : [
    ggml_proj.dependency('ggml'), ggml_proj.dependency('ggml-base'),
    ggml_proj.dependency('ggml-cpu'), ggml_proj.dependency('ggml-vulkan'),
  ])
endif

# System libraries when found, else the wraps in subprojects/, built
# with their own C dialect and warnings.
wrap_opts = ['c_std=gnu17', 'werror=false', 'warning_level=0', 'default_library=static']
# zlib first, so a zlib wrap for libpng gets these options too.
zlib = dependency('zlib', default_options : wrap_opts)
png = dependency('libpng', default_options : wrap_opts)
jpeg = dependency('libjpeg', default_options : wrap_opts)
# utf8proc: Unicode NFC for tokenizer.json normalizers. Its tests fetch data at build time.
utf8proc = dependency('libutf8proc', default_options : wrap_opts + ['tests=disabled'])
# liburing: the n-gram reader reads through an io_uring; without it, on threads.
uring = dependency('liburing', required : false)
if uring.found()
  uring = declare_dependency(dependencies : uring, compile_args : '-DLGML_HAVE_URING')
endif

lua_dir = get_option('lua_dir')
if lua_dir == ''
  lua_dir = get_option('prefix') / 'share' / 'lua' / '5.4'
endif
lua_cdir = get_option('lua_cdir')
if lua_cdir == ''
  lua_cdir = get_option('prefix') / 'lib' / 'lua' / '5.4'
endif

# Fused ops carry SPIR-V shaders for the Vulkan backend; without glslc
# they run on the CPU only.
glslc = find_program('glslc', required : false)
spv_srcs = []
spv_args = []
if glslc.found()
  foreach s : ['hc_add', 'hc_mean']
    spv_srcs += custom_target(s + '.spv.h',
      input : 'src/shaders' / s + '.comp',
      output : s + '.spv.h',
      command : [glslc, '--target-env=vulkan1.2', '-O', '-mfmt=c', '@INPUT@', '-o', '@OUTPUT@'])
  endforeach
  spv_args += '-DLGML_HAVE_SPV'
endif

lgml = shared_module('lgml',
  'src/lgml.c',
  'src/ops_fused.c',
  spv_srcs,
  'src/ops_orpheus.c',
  'src/ops_qwen3tts.c',
  'src/ops_vibevoice.c',
  'src/ops_nn.c',
  'src/proctitle.c',
  'src/xcache.c',
  'src/train.c',
  'src/sample.c',
  'src/grammar.c',
  'src/tokenize.c',
  'src/json.c',
  'src/safetensors.c',
  'src/quant.c',
  'src/audio.c',
  'src/laudio.c',
  'src/image.c',
  'src/limage.c',
  'src/sha256.c',
  'src/vocab.c',
  'src/mem.c',
  'src/ngram.c',
  'src/lngram.c',
  'src/gload.c',
  dependencies : [ggml, lua, m, threads, png, jpeg, utf8proc, uring],
  c_args : spv_args,
  name_prefix : '',
  install : true,
  install_dir : lua_cdir,
)

install_subdir('lua/lgml', install_dir : lua_dir)
install_data('bin/lgml-server', 'bin/lgml-worker', 'bin/lgml-repl', 'bin/lgml-finetune',
  'bin/lgml-convert',
  install_dir : get_option('bindir'), install_mode : 'rwxr-xr-x')

# Rock dependencies install into lua_modules (see lgml-scm-1.rockspec).
rocks = meson.current_source_dir() / 'lua_modules'
lua_env = environment()
lua_env.prepend('LUA_CPATH', rocks / 'lib' / 'lua' / '5.4' / '?.so', separator : ';')
lua_env.prepend('LUA_PATH', rocks / 'share' / 'lua' / '5.4' / '?.lua', separator : ';')
lua_env.prepend('LUA_CPATH', meson.current_build_dir() / '?.so', separator : ';')
lua_env.append('LUA_CPATH', ';', separator : ';')
lua_env.prepend('LUA_PATH', meson.current_source_dir() / 'lua' / '?' / 'init.lua', separator : ';')
lua_env.prepend('LUA_PATH', meson.current_source_dir() / 'lua' / '?.lua', separator : ';')
lua_env.append('LUA_PATH', ';', separator : ';')
meson.add_devenv(lua_env)
# Tests run with strict globals: see lgml.strict.
lua_env.set('LUA_INIT_5_4', 'require("lgml.strict").on()')
# Tests run in parallel, so each one gets few CPU threads.
lua_env.set('LGML_THREADS', '2')
add_test_setup('default', is_default : true, env : lua_env)
# meson test --setup cpu: the Vulkan loader finds no driver, so tests use only the CPU.
cpu_env = lua_env
cpu_env.set('VK_DRIVER_FILES', '/nonexistent')
cpu_env.set('VK_ICD_FILENAMES', '/nonexistent')
add_test_setup('cpu', env : cpu_env)

test_audio = executable('test_audio',
  'test/test_audio.c',
  'src/audio.c',
  include_directories : include_directories('src'),
  dependencies : [m],
)
test('audio', test_audio, timeout : 60)

# The real table check runs when the ngram option names the gguf file
# that holds it, and ngram_meta the file with the ple keys.
test_ngram = executable('test_ngram',
  'test/test_ngram.c',
  'src/ngram.c',
  include_directories : include_directories('src'),
  dependencies : [ggml, m, threads, uring],
)
test('ngram', test_ngram,
  env : { 'LGML_NGRAM' : get_option('ngram'), 'LGML_NGRAM_META' : get_option('ngram_meta') },
  timeout : 120)

lua_bin = find_program('lua5.4', 'lua', required : false)
if lua_bin.found()
  test('basic', lua_bin, args : [files('test/test_basic.lua')], timeout : 60)
  test('json', lua_bin, args : [files('test/test_json.lua')], timeout : 60)
  test('strict', lua_bin, args : [files('test/test_strict.lua')], timeout : 30)
  test('log', lua_bin, args : [files('test/test_log.lua')], timeout : 30)
  test('http', lua_bin, args : [files('test/test_http.lua')], timeout : 60)
  test('openapi', lua_bin, args : [files('test/test_openapi.lua')], timeout : 30)
  test('translate', lua_bin, args : [files('test/test_translate.lua')], timeout : 30)
  test('compat_openai', lua_bin, args : [files('test/test_compat_openai.lua')], timeout : 60)
  test('compat_anthropic', lua_bin, args : [files('test/test_compat_anthropic.lua')], timeout : 60)
  test('compat_responses', lua_bin, args : [files('test/test_compat_responses.lua')], timeout : 60)
  test('engine_compat', lua_bin, args : [files('test/test_engine_compat.lua')], timeout : 120)
  test('kv_grow', lua_bin, args : [files('test/test_kv_grow.lua')], timeout : 120)
  test('slots', lua_bin, args : [files('test/test_slots.lua')], timeout : 120)
  test('spec', lua_bin, args : [files('test/test_spec.lua')], timeout : 120)
  test('graph_memory', lua_bin, args : [files('test/test_graph_memory.lua')], timeout : 120)
  test('batch', lua_bin, args : [files('test/test_batch.lua')], timeout : 120)
  test('kvpool', lua_bin, args : [files('test/test_kvpool.lua')], timeout : 120)
  test('engine_apis', lua_bin, args : [files('test/test_engine_apis.lua')], timeout : 120)
  test('lazy_tools', lua_bin, args : [files('test/test_lazy_tools.lua')], timeout : 120)
  test('tokenize', lua_bin, args : [files('test/test_tokenize.lua')],
    env : { 'LGML_VOCAB_DIR' : get_option('vocab_dir') }, timeout : 60)
  test('nd', lua_bin, args : [files('test/test_nd.lua')], timeout : 120)
  test('router', lua_bin, args : [files('test/test_router.lua')], timeout : 30)
  test('embeddings', lua_bin, args : [files('test/test_embeddings.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  # LGML_SYSTEMONE_MODEL (a chat gguf) from the environment adds real-model checks.
  test('systemone', lua_bin, args : [files('test/test_systemone.lua')], timeout : 600)
  test('decisions', lua_bin, args : [files('test/test_decisions.lua')])
  test('decide_head', lua_bin, args : [files('test/test_decide_head.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 300)
  # LGML_LAYA (a Laya model directory) from the environment adds real-model checks
  # (backend LGML_LAYA_BACKEND, default CPU).
  test('laya', lua_bin, args : [files('test/test_laya.lua')], timeout : 300)
  test('worker', lua_bin, args : [files('test/test_worker.lua')], timeout : 60)
  test('stream', lua_bin, args : [files('test/test_stream.lua')], timeout : 30)
  test('status', lua_bin, args : [files('test/test_status.lua')], timeout : 60)
  test('ubatch', lua_bin, args : [files('test/test_ubatch.lua')], timeout : 120)
  test('pool', lua_bin, args : [files('test/test_pool.lua')], timeout : 120)
  test('budget', lua_bin, args : [files('test/test_budget.lua')], timeout : 120)
  test('prompt_cache', lua_bin, args : [files('test/test_prompt_cache.lua')], timeout : 30)
  test('placement', lua_bin, args : [files('test/test_placement.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 60)
  test('load', lua_bin, args : [files('test/test_load.lua')], timeout : 60)
  test('xcache', lua_bin, args : [files('test/test_xcache.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('chat', lua_bin, args : [files('test/test_chat.lua')], timeout : 60)
  test('sample', lua_bin, args : [files('test/test_sample.lua')], timeout : 120)
  test('grammar', lua_bin, args : [files('test/test_grammar.lua')], timeout : 60)
  test('json_schema', lua_bin, args : [files('test/test_json_schema.lua')], timeout : 60)
  test('ops', lua_bin, args : [files('test/test_ops.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 60)
  test('train', lua_bin, args : [files('test/test_train.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('lora', lua_bin, args : [files('test/test_lora.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('convert', lua_bin, args : [files('test/test_convert.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('finetune', lua_bin, args : [files('test/test_finetune.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 300)
  test('diffusion_gemma', lua_bin, args : [files('test/test_diffusion_gemma.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('llada', lua_bin, args : [files('test/test_llada.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('qwen_moe', lua_bin, args : [files('test/test_qwen_moe.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('gemma4', lua_bin, args : [files('test/test_gemma4.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('qwen4exp', lua_bin, args : [files('test/test_qwen4exp.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 300)
  test('mtp', lua_bin, args : [files('test/test_mtp.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 300)
  test('zimage', lua_bin, args : [files('test/test_zimage.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('image', lua_bin, args : [files('test/test_image.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('ngram_lua', lua_bin, args : [files('test/test_ngram.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 60)
  test('vision', lua_bin, args : [files('test/test_vision.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('qwen3vl', lua_bin, args : [files('test/test_qwen3vl.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)

  # Tests that load large models run one at a time, in the 'model'
  # suite. Each skips when its option is empty. An empty LGML_THREADS
  # gives them the default CPU thread count.
  model_tests = {
    'qwen3' : { 'LGML_MODEL' : get_option('model') },
    'tts_orpheus' : { 'LGML_ORPHEUS' : get_option('orpheus') },
    'tts_qwen3tts' : { 'LGML_QWEN3TTS' : get_option('qwen3tts') },
    'tts_vibevoice' : { 'LGML_VIBEVOICE' : get_option('vibevoice') },
    'whisper' : { 'LGML_WHISPER' : get_option('whisper'),
      'LGML_WHISPER_WAV' : get_option('whisper_wav') },
  }
  foreach name, env : model_tests
    env += { 'LGML_THREADS' : '' }
    test(name, lua_bin, args : [files('test/test_' + name + '.lua')], env : env,
      suite : 'model', is_parallel : false, timeout : 600)
  endforeach
  # LGML_DIFFUSION_GEMMA (a gguf path) comes from the environment.
  test('diffusion_gemma_model', lua_bin, args : [files('test/test_diffusion_gemma_model.lua')],
    suite : 'model', env : { 'LGML_THREADS' : '' },
    is_parallel : false, timeout : 3600)
  # LGML_EMBED_MODEL (a Qwen3-Embedding gguf) and LGML_BACKEND come from the environment.
  test('embeddings_model', lua_bin, args : [files('test/test_embeddings_model.lua')],
    suite : 'model', env : { 'LGML_THREADS' : '' },
    is_parallel : false, timeout : 600)
  # LGML_ZIMAGE (a directory with the Z-Image files) comes from the environment.
  test('zimage_model', lua_bin, args : [files('test/test_zimage_model.lua')],
    suite : 'model', env : { 'LGML_THREADS' : '' },
    is_parallel : false, timeout : 600)
  # LGML_GEMMA4_VISION, LGML_VISION_IMAGE, LGML_MTMD_EMBD and LGML_MTMD_CLI come
  # from the environment; see the test file.
  test('vision_model', lua_bin, args : [files('test/test_vision_model.lua')],
    suite : 'model', env : { 'LGML_THREADS' : '' },
    is_parallel : false, timeout : 3600)
  # LGML_CLEF_VISION, LGML_VISION_IMAGE and the llama.cpp tools come from the environment.
  test('qwen3vl_model', lua_bin, args : [files('test/test_qwen3vl_model.lua')],
    suite : 'model', env : { 'LGML_THREADS' : '' },
    is_parallel : false, timeout : 3600)
  # LGML_ZIMAGE_VAE (ae.safetensors) and optional LGML_VAE_REF come from the environment.
  test('image_model', lua_bin, args : [files('test/test_image_model.lua')],
    suite : 'model', env : { 'LGML_THREADS' : '' },
    is_parallel : false, timeout : 600)
endif

# Fuzz harness. Built with afl-clang-fast it runs under afl-fuzz. The
# 'fuzz' suite replays the saved corpus, with no Vulkan driver. See
# fuzz/README.
lfuzz = executable('lfuzz', 'fuzz/lfuzz.c',
  objects : lgml.extract_all_objects(recursive : true),
  dependencies : [ggml, lua, m, png, jpeg, utf8proc, uring, cc.find_library('dl', required : false)],
  export_dynamic : true,
)
foreach t : ['json', 'chat', 'http', 'safetensors', 'vocab', 'wav', 'tokenize', 'image', 'grammar']
  test('fuzz_' + t, lfuzz,
    args : [files('fuzz' / t + '.lua'), meson.current_source_dir() / 'fuzz' / 'corpus' / t],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    suite : 'fuzz', timeout : 120)
endforeach
