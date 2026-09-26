// Reference VAE runs from stable-diffusion.cpp for the lgml image tests.
// Files hold raw f32 in torch order (N, C, H, W).
//   vae_ref decode AE IN OUT N H W: latents (N, 16, H, W) to pixels
//   vae_ref encode AE IN OUT N H W: pixels (N, 3, H, W) to (N, 32, H/8, W/8)
// VAE_DEV=cpu runs on the CPU, else Vulkan0. VAE_DIRECT=1 uses direct
// conv. VAE_FA=0 turns off flash attention. VAE_THREADS sets CPU threads.

// Build with SD a built stable-diffusion.cpp tree and B=$SD/build/ggml/src:
//   g++ -O2 -std=c++17 -DGGML_MAX_NAME=128 -I$SD -I$SD/ggml/include -I$SD/thirdparty
//     scripts/vae_ref.cpp $SD/build/libstable-diffusion.a $B/libggml.a
//     $B/libggml-cpu.a $B/ggml-vulkan/libggml-vulkan.a $B/libggml-base.a
//     -lvulkan -lgomp -lpthread -ldl -lm -o build/vae_ref

#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include "model.h"
#include "vae.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 8 || (strcmp(argv[1], "decode") != 0 && strcmp(argv[1], "encode") != 0)) {
        fprintf(stderr, "usage: vae_ref decode|encode AE IN OUT N H W\n");
        return 2;
    }
    bool decode = strcmp(argv[1], "decode") == 0;
    int64_t n = atoll(argv[5]), h = atoll(argv[6]), w = atoll(argv[7]);
    int64_t c = decode ? 16 : 3;
    const char * dev = getenv("VAE_DEV");
    ggml_backend_t be = dev && strcmp(dev, "cpu") == 0 ? ggml_backend_cpu_init() : ggml_backend_vk_init(0);
    int threads = getenv("VAE_THREADS") ? atoi(getenv("VAE_THREADS")) : 8;

    ModelLoader ml;
    if (!ml.init_from_file_and_convert_name(argv[2], "vae.", VERSION_FLUX)) {
        fprintf(stderr, "cannot load %s\n", argv[2]);
        return 1;
    }
    auto vae = std::make_shared<AutoEncoderKL>(be, false, ml.get_tensor_storage_map(),
                                               "first_stage_model", false, false, VERSION_FLUX);
    if (getenv("VAE_DIRECT") && atoi(getenv("VAE_DIRECT")) != 0) {
        vae->set_conv2d_direct_enabled(true);
    }
    // Without flash attention, this revision mixes up v in the VAE attention.
    vae->set_flash_attention_enabled(!getenv("VAE_FA") || atoi(getenv("VAE_FA")) != 0);
    vae->alloc_params_buffer();
    std::map<std::string, ggml_tensor *> tensors;
    vae->get_param_tensors(tensors, "first_stage_model");
    if (!ml.load_tensors(tensors)) {
        fprintf(stderr, "cannot load tensors\n");
        return 1;
    }

    size_t nin = (size_t) (n * c * h * w);
    ggml_init_params ip = { nin * 4 * 64 + (64 << 20), nullptr, false };
    ggml_context * work = ggml_init(ip);
    ggml_tensor * in = ggml_new_tensor_4d(work, GGML_TYPE_F32, w, h, c, n);
    FILE * f = fopen(argv[3], "rb");
    if (!f || fread(in->data, 4, nin, f) != nin) {
        fprintf(stderr, "cannot read %s\n", argv[3]);
        return 1;
    }
    fclose(f);
    ggml_tensor * out = nullptr;
    int64_t t0 = ggml_time_ms();
    if (!vae->compute(threads, in, decode, &out, work)) {
        fprintf(stderr, "compute failed\n");
        return 1;
    }
    fprintf(stderr, "%s: %lld ms, out %lldx%lldx%lldx%lld\n", argv[1], (long long) (ggml_time_ms() - t0),
            (long long) out->ne[3], (long long) out->ne[2], (long long) out->ne[1], (long long) out->ne[0]);
    f = fopen(argv[4], "wb");
    if (!f || fwrite(out->data, 1, ggml_nbytes(out), f) != ggml_nbytes(out)) {
        fprintf(stderr, "cannot write %s\n", argv[4]);
        return 1;
    }
    fclose(f);
    return 0;
}
