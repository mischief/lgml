// Reference dumps from the llama.cpp DiffusionGemma fork, for the lgml
// tests. PROMPT and CANVAS are comma-separated token ids. DG_NGL sets
// the GPU layers (default 0) and DG_THREADS the CPU threads (default 1).
//   dg_ref logits MODEL OUT PROMPT CANVAS [SC TEMP_INV]: canvas logits (f32)
//   dg_ref denoise MODEL PROMPT SEED KV: argmax canvas of each step
//   dg_ref rng SEED N_VOCAB N: the denoiser RNG stream

#include "diffusion.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<llama_token> ids(const char * s) {
    std::vector<llama_token> v;
    while (*s) {
        char * end;
        v.push_back((llama_token) strtol(s, &end, 10));
        s = *end == ',' ? end + 1 : end;
    }
    return v;
}

static llama_context * open_ctx(llama_model * model, int n) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n;
    cp.n_batch         = n;
    cp.n_ubatch        = n;
    cp.n_threads       = getenv("DG_THREADS") ? atoi(getenv("DG_THREADS")) : 1;
    cp.n_threads_batch = cp.n_threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return llama_init_from_model(model, cp);
}

static bool step_cb(int32_t step, int32_t, const llama_token * tokens, int32_t n, void * ud) {
    const int32_t n_input = *(const int32_t *) ud;
    printf("step %d:", step);
    for (int32_t i = n_input; i < n; i++) {
        printf(" %d", tokens[i]);
    }
    printf("\n");
    return true;
}

int main(int argc, char ** argv) {
    if (argc >= 5 && strcmp(argv[1], "rng") == 0) {
        std::mt19937 rng(atoi(argv[2]));
        std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
        std::uniform_int_distribution<int32_t> vocab_dist(0, atoi(argv[3]) - 1);
        for (int i = 0; i < atoi(argv[4]); i++) {
            const float u = uni01(rng);
            printf("%.9g %d\n", u, vocab_dist(rng));
        }
        return 0;
    }
    if (argc < 5) {
        fprintf(stderr, "usage: dg_ref logits|denoise|rng ...\n");
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("DG_NGL") ? atoi(getenv("DG_NGL")) : 0;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) {
        return 1;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    if (strcmp(argv[1], "logits") == 0 && argc >= 6) {
        const auto prompt = ids(argv[4]);
        const auto canvas = ids(argv[5]);
        const int32_t P = (int32_t) prompt.size(), C = (int32_t) canvas.size();
        std::vector<float> sc((size_t) C * n_vocab, 0.0f);
        float use = 0.0f, temp_inv = 1.0f;
        if (argc >= 8) {
            FILE * f = fopen(argv[6], "rb");
            if (!f || fread(sc.data(), sizeof(float), sc.size(), f) != sc.size()) {
                return 1;
            }
            fclose(f);
            use = 1.0f;
            temp_inv = (float) atof(argv[7]);
        }
        llama_context * ctx = open_ctx(model, P + C);
        llama_set_causal_attn(ctx, false);
        llama_diffusion_set_sc(model, sc.data(), use, temp_inv, true);
        llama_batch b = llama_batch_init(P + C, 0, 1);
        b.n_tokens = P + C;
        for (int32_t i = 0; i < P + C; i++) {
            b.token[i]     = i < P ? prompt[i] : canvas[i - P];
            b.pos[i]       = i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = 1;
        }
        if (llama_decode(ctx, b) != 0) {
            return 1;
        }
        const float * logits = llama_get_logits(ctx);
        FILE * f = fopen(argv[3], "wb");
        fwrite(logits + (size_t) P * n_vocab, sizeof(float), (size_t) C * n_vocab, f);
        fclose(f);
        llama_batch_free(b);
        llama_free(ctx);
    } else if (strcmp(argv[1], "denoise") == 0 && argc >= 6) {
        const auto prompt = ids(argv[3]);
        char buf[32];
        llama_model_meta_val_str(model, "diffusion.canvas_length", buf, sizeof(buf));
        const int32_t P = (int32_t) prompt.size(), C = atoi(buf);
        llama_context * ctx = open_ctx(model, P + C);
        diffusion_eb_params eb;
        eb.seed       = atoi(argv[4]);
        eb.kv_cache   = atoi(argv[5]) != 0;
        eb.max_length = P + C;
        int32_t n_input = P;
        eb.step_callback = step_cb;
        eb.step_callback_user_data = &n_input;
        std::vector<llama_token> out(P + C);
        int32_t n_gen = 0;
        diffusion_generate_entropy_bound(ctx, prompt.data(), out.data(), P, eb, n_gen);
        llama_free(ctx);
    }
    llama_model_free(model);
    return 0;
}
