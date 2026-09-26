// Reference dumps from stable-diffusion.cpp (Vulkan build, its defines)
// for the Z-Image checks. Files: "ZREF", 4 int64 sizes, f32 data.
//   noise SEED N OUT; te LLM PROMPT OUT (hidden states);
//   dit DIT X CTX SIGMA OUT (one pass, flash attention);
//   sample DIT CTX SEED STEPS H W OUT (final latent).
// ZREF_CPU=1 runs on the CPU.

#include "conditioner.hpp"
#include "denoiser.hpp"
#include "model.h"
#include "rng_philox.hpp"
#include "z_image.hpp"
#include "ggml-vulkan.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void save(const char* path, ggml_tensor* t) {
    FILE* f = fopen(path, "wb");
    int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    fwrite("ZREF", 1, 4, f);
    fwrite(ne, 8, 4, f);
    fwrite(t->data, 4, ggml_nelements(t), f);
    fclose(f);
}

static ggml_tensor* load(ggml_context* ctx, const char* path) {
    FILE* f = fopen(path, "rb");
    char magic[4];
    int64_t ne[4];
    if (!f || fread(magic, 1, 4, f) != 4 || fread(ne, 8, 4, f) != 4) {
        fprintf(stderr, "bad file %s\n", path);
        exit(1);
    }
    ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    if (fread(t->data, 4, ggml_nelements(t), f) != (size_t)ggml_nelements(t)) {
        fprintf(stderr, "short file %s\n", path);
        exit(1);
    }
    fclose(f);
    return t;
}

static ggml_context* work() {
    ggml_init_params p = {(size_t)2 << 30, nullptr, false};
    return ggml_init(p);
}

static void log_cb(enum sd_log_level_t level, const char* text, void*) {
    if (level >= SD_LOG_WARN)
        fputs(text, stderr);
}

// Load the DiT with its weights; flash attention as --diffusion-fa.
static ZImage::ZImageRunner* dit(ggml_backend_t be, ModelLoader& ml, const char* path) {
    ml.init_from_file(path, "model.diffusion_model.");
    ml.convert_tensors_name();
    auto* r = new ZImage::ZImageRunner(be, false, ml.get_tensor_storage_map(),
                                       "model.diffusion_model", VERSION_Z_IMAGE);
    r->alloc_params_buffer();
    std::map<std::string, ggml_tensor*> tensors;
    r->get_param_tensors(tensors, "model.diffusion_model");
    if (!ml.load_tensors(tensors))
        exit(1);
    r->set_flash_attention_enabled(true);
    return r;
}

static ggml_tensor* timestep(ggml_context* ctx, float sigma) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float*)t->data)[0] = 1000.f - sigma * 1000.f;
    return t;
}

int main(int argc, char** argv) {
    sd_set_log_callback(log_cb, nullptr);
    std::string mode = argc > 1 ? argv[1] : "";
    ggml_context* ctx = work();
    if (mode == "noise" && argc == 5) {
        PhiloxRNG rng;
        rng.manual_seed(atoll(argv[2]));
        int n = atoi(argv[3]);
        auto v = rng.randn(n);
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        memcpy(t->data, v.data(), 4 * (size_t)n);
        save(argv[4], t);
        return 0;
    }
    ggml_backend_t be = getenv("ZREF_CPU") ? ggml_backend_cpu_init() : ggml_backend_vk_init(0);
    ModelLoader ml;
    if (mode == "te" && argc == 5) {
        ml.init_from_file(argv[2], "text_encoders.llm.");
        ml.convert_tensors_name();
        LLMEmbedder emb(be, false, ml.get_tensor_storage_map(), VERSION_Z_IMAGE);
        emb.alloc_params_buffer();
        std::map<std::string, ggml_tensor*> tensors;
        emb.get_param_tensors(tensors);
        if (!ml.load_tensors(tensors))
            return 1;
        ConditionerParams cp;
        cp.text = argv[3];
        save(argv[4], emb.get_learned_condition(ctx, 8, cp).c_crossattn);
        return 0;
    }
    if (mode == "dit" && argc == 7) {
        ZImage::ZImageRunner* r = dit(be, ml, argv[2]);
        ggml_tensor* out = nullptr;
        r->compute(8, load(ctx, argv[3]), timestep(ctx, atof(argv[5])), load(ctx, argv[4]),
                   {}, true, &out, ctx);
        save(argv[6], out);
        return 0;
    }
    if (mode == "sample" && argc == 9) {
        // The Euler loop of the reference, with its f32 math.
        ZImage::ZImageRunner* r = dit(be, ml, argv[2]);
        ggml_tensor* c = load(ctx, argv[3]);
        int steps = atoi(argv[5]), h = atoi(argv[6]), w = atoi(argv[7]);
        DiscreteFlowDenoiser den(3.0f);
        auto sig = den.get_sigmas(steps, 0, DISCRETE_SCHEDULER, VERSION_Z_IMAGE);
        PhiloxRNG rng;
        rng.manual_seed(atoll(argv[4]));
        ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, 16, 1);
        auto noise = rng.randn(ggml_nelements(x));
        float* xv = (float*)x->data;
        for (int64_t i = 0; i < ggml_nelements(x); i++)
            xv[i] = noise[i] * sig[0] + 0.1159f * (1.0f - sig[0]);
        for (int s = 0; s < steps; s++) {
            ggml_tensor* out = nullptr;
            r->compute(8, x, timestep(ctx, sig[s]), c, {}, true, &out, ctx);
            float* ov = (float*)out->data;
            for (int64_t i = 0; i < ggml_nelements(x); i++) {
                float den_i = ov[i] * -sig[s] + xv[i];
                xv[i] = xv[i] + (xv[i] - den_i) / sig[s] * (sig[s + 1] - sig[s]);
            }
        }
        save(argv[8], x);
        return 0;
    }
    fprintf(stderr, "usage: see the comment at the top of zimage_ref.cpp\n");
    return 2;
}
