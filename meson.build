project('lgml',
  'c',
  version : '0.1.0',
  meson_version : '>=1.4.0',
  default_options : [
    'c_std=c17',
    'warning_level=3',
    'werror=true',
  ],
)

cc = meson.get_compiler('c')
m = cc.find_library('m', required : false)

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
if get_option('ggml') == 'subproject'
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
  ggml_proj = cmake.subproject('ggml', options : ggml_opts)
  ggml = declare_dependency(dependencies : [
    ggml_proj.dependency('ggml'), ggml_proj.dependency('ggml-base'),
    ggml_proj.dependency('ggml-cpu'), ggml_proj.dependency('ggml-vulkan'),
  ])
else
  ggml = dependency('ggml', method : 'cmake', modules : ['ggml::ggml', 'ggml::ggml-base'])
endif

# System libraries when found, else the wraps in subprojects/, built
# with their own C dialect and warnings.
wrap_opts = ['c_std=gnu17', 'werror=false', 'warning_level=0', 'default_library=static']
# zlib first, so a zlib wrap for libpng gets these options too.
zlib = dependency('zlib', default_options : wrap_opts)
png = dependency('libpng', default_options : wrap_opts)
jpeg = dependency('libjpeg', default_options : wrap_opts)

lgml = shared_module('lgml',
  'src/lgml.c',
  'src/ops_orpheus.c',
  'src/ops_qwen3tts.c',
  'src/ops_vibevoice.c',
  'src/ops_nn.c',
  'src/train.c',
  'src/sample.c',
  'src/grammar.c',
  'src/tokenize.c',
  'src/json.c',
  'src/safetensors.c',
  'src/audio.c',
  'src/laudio.c',
  'src/image.c',
  'src/limage.c',
  'src/sha256.c',
  'src/vocab.c',
  dependencies : [ggml, lua, m, png, jpeg],
  name_prefix : '',
  install : true,
  install_dir : get_option('prefix') / 'lib' / 'lua' / '5.4',
)

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

lua_bin = find_program('lua5.4', 'lua', required : false)
if lua_bin.found()
  test('basic', lua_bin, args : [files('test/test_basic.lua')], timeout : 60)
  test('json', lua_bin, args : [files('test/test_json.lua')], timeout : 60)
  test('strict', lua_bin, args : [files('test/test_strict.lua')], timeout : 30)
  test('http', lua_bin, args : [files('test/test_http.lua')], timeout : 60)
  test('tokenize', lua_bin, args : [files('test/test_tokenize.lua')],
    env : { 'LGML_VOCAB_DIR' : get_option('vocab_dir'),
      'LGML_TOK_MODEL' : get_option('model'),
      'LGML_TOK_GEMMA4' : get_option('tok_gemma4') }, timeout : 60)
  test('nd', lua_bin, args : [files('test/test_nd.lua')], timeout : 120)
  test('router', lua_bin, args : [files('test/test_router.lua')], timeout : 30)
  # LGML_SYSTEMONE_MODEL (a chat gguf) from the environment adds real-model checks.
  test('systemone', lua_bin, args : [files('test/test_systemone.lua')], timeout : 600)
  test('worker', lua_bin, args : [files('test/test_worker.lua')], timeout : 60)
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
  test('finetune', lua_bin, args : [files('test/test_finetune.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 300)
  test('diffusion_gemma', lua_bin, args : [files('test/test_diffusion_gemma.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('qwen_moe', lua_bin, args : [files('test/test_qwen_moe.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('gemma4', lua_bin, args : [files('test/test_gemma4.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('zimage', lua_bin, args : [files('test/test_zimage.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('image', lua_bin, args : [files('test/test_image.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)
  test('vision', lua_bin, args : [files('test/test_vision.lua')],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    timeout : 120)

  # Tests that load large models run one at a time, in the 'model'
  # suite. Each skips when its option is empty.
  model_tests = {
    'qwen3' : { 'LGML_MODEL' : get_option('model') },
    'tts_orpheus' : { 'LGML_ORPHEUS' : get_option('orpheus') },
    'tts_qwen3tts' : { 'LGML_QWEN3TTS' : get_option('qwen3tts') },
    'tts_vibevoice' : { 'LGML_VIBEVOICE' : get_option('vibevoice') },
    'whisper' : { 'LGML_WHISPER' : get_option('whisper'),
      'LGML_WHISPER_WAV' : get_option('whisper_wav') },
  }
  foreach name, env : model_tests
    test(name, lua_bin, args : [files('test/test_' + name + '.lua')], env : env,
      suite : 'model', is_parallel : false, timeout : 600)
  endforeach
  # LGML_DIFFUSION_GEMMA (a gguf path) comes from the environment.
  test('diffusion_gemma_model', lua_bin, args : [files('test/test_diffusion_gemma_model.lua')],
    suite : 'model', is_parallel : false, timeout : 3600)
  # LGML_ZIMAGE (a directory with the Z-Image files) comes from the environment.
  test('zimage_model', lua_bin, args : [files('test/test_zimage_model.lua')],
    suite : 'model', is_parallel : false, timeout : 600)
  # LGML_GEMMA4_VISION, LGML_VISION_IMAGE, LGML_MTMD_EMBD and LGML_MTMD_CLI come
  # from the environment; see the test file.
  test('vision_model', lua_bin, args : [files('test/test_vision_model.lua')],
    suite : 'model', is_parallel : false, timeout : 3600)
  # LGML_ZIMAGE_VAE (ae.safetensors) and optional LGML_VAE_REF come from the environment.
  test('image_model', lua_bin, args : [files('test/test_image_model.lua')],
    suite : 'model', is_parallel : false, timeout : 600)
endif

# Fuzz harness. Built with afl-clang-fast it runs under afl-fuzz. The
# 'fuzz' suite replays the saved corpus, with no Vulkan driver. See
# fuzz/README.
lfuzz = executable('lfuzz', 'fuzz/lfuzz.c',
  objects : lgml.extract_all_objects(recursive : true),
  dependencies : [ggml, lua, m, png, jpeg, cc.find_library('dl', required : false)],
  export_dynamic : true,
)
foreach t : ['json', 'chat', 'http', 'safetensors', 'vocab', 'wav', 'tokenize', 'image', 'grammar']
  test('fuzz_' + t, lfuzz,
    args : [files('fuzz' / t + '.lua'), meson.current_source_dir() / 'fuzz' / 'corpus' / t],
    env : { 'VK_DRIVER_FILES' : '/nonexistent', 'VK_ICD_FILENAMES' : '/nonexistent' },
    suite : 'fuzz', timeout : 120)
endforeach
