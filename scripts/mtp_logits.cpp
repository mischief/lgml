// MTP head logits from llama.cpp: the target runs the prompt, the MTP
// context runs it with the target states shifted by one, then the head
// runs steps more steps on its own guesses and states. Prints the
// logits of the last prompt row ("p") and of each step ("s").
// Build against a llama.cpp tree L with its build in L/build:
// c++ -std=c++17 -IL/include -IL/ggml/include -IL/common -IL/src mtp_logits.cpp
//   -LL/build/bin -lllama-common -lllama -lggml -lggml-base
// Usage: mtp_logits -m model.gguf --spec-type draft-mtp ... -- steps id ...
#include "arg.h"
#include "common.h"
#include "speculative.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// The draft context samples on the backend, which changes the logits
// it gives back, so the callback keeps the result_output node. With
// MTP_TRACE set, it prints the sum and first values of each node.
static std::vector<float> g_out;
static bool g_trace = false;

static bool trace(struct ggml_tensor * t, bool ask, void *) {
    if (ask) return true;
    if (strcmp(t->name, "result_output") == 0) {
        g_out.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, g_out.data(), 0, ggml_nbytes(t));
    }
    if (!g_trace) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return true;
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    double sum = 0;
    for (float x : v) sum += x;
    fprintf(stderr, "%-28s %5lld %5lld %5lld sum %.6f first %.6f %.6f\n", t->name, (long long) t->ne[0],
            (long long) t->ne[1], (long long) t->ne[2], sum, v.size() > 0 ? v[0] : 0, v.size() > 1 ? v[1] : 0);
    return true;
}

static int print_row(const char * tag, int id, const float * p, int nv) {
    int best = 0;
    printf("%s %d", tag, id);
    for (int j = 0; j < nv; j++) {
        printf(" %.9g", p[j]);
        if (p[j] > p[best]) best = j;
    }
    printf("\n");
    return best;
}

int main(int argc, char ** argv) {
    int sep = 1;
    while (sep < argc && strcmp(argv[sep], "--") != 0) sep++;
    if (sep + 2 > argc) return 2;
    const int steps = atoi(argv[sep + 1]);
    std::vector<llama_token> ids;
    for (int i = sep + 2; i < argc; i++) ids.push_back(atoi(argv[i]));
    const int n = ids.size();

    common_params params;
    common_init();
    if (!common_params_parse(sep, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) return 1;
    const auto lim = common_speculative_get_output_limits(params.n_batch, params.n_parallel,
            common_speculative_n_max(&params.speculative));
    params.n_outputs_max = lim.total;
    params.n_outputs_max_per_seq = lim.per_seq;
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * tgt = init->context();
    common_params pd = common_base_params_to_speculative(params);
    g_trace = getenv("MTP_TRACE") != nullptr;
    pd.cb_eval = trace;
    pd.cb_eval_user_data = nullptr;
    auto spec_init = common_speculative_init_from_params(pd, model, tgt);
    llama_context * dft = spec_init->context();
    if (!tgt || !dft) return 1;
    params.speculative.draft.ctx_tgt = tgt;
    params.speculative.draft.ctx_dft = dft;
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (!spec) return 1;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int ne = llama_model_n_embd_out(model);

    common_batch b(tgt);
    for (int i = 0; i < n; i++) b.add(ids[i], i, 0, i == n - 1);
    if (llama_process(tgt, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) return 1;
    std::vector<float> h((size_t) n * ne);
    for (int i = 0; i < n; i++) memcpy(h.data() + (size_t) i * ne, llama_get_embeddings_nextn_ith(tgt, i), ne * sizeof(float));

    std::vector<float> zero(ne, 0.0f);
    // the prompt rows but the last without outputs, as the speculative
    // process hook runs them, then the last row as a draft step
    common_batch d(dft);
    {
        common_batch bp(tgt);
        for (int i = 0; i < n - 1; i++) bp.add(ids[i], i, 0, true);
        if (n > 1 && !common_speculative_process(spec, bp)) return 1;
    }
    d.clear();
    {
        int idx = d.add(ids[n - 1], n - 1, 0, true);
        d.set_embd(idx, { n == 1 ? zero.data() : h.data() + (size_t) (n - 2) * ne, 1, (size_t) ne });
    }
    if (llama_process(dft, LLAMA_PROCESS_TYPE_DECODE, d.get()) != 0) return 1;
    int last = print_row("p", ids[n - 1], g_out.data(), nv);
    std::vector<float> hr(ne);
    memcpy(hr.data(), llama_get_embeddings_nextn_ith(dft, 0), ne * sizeof(float));
    for (int s = 0; s < steps; s++) {
        d.clear();
        int idx = d.add(last, n + s, 0, true);
        d.set_embd(idx, { hr.data(), 1, (size_t) ne });
        if (llama_process(dft, LLAMA_PROCESS_TYPE_DECODE, d.get()) != 0) return 1;
        int id = last;
        last = print_row("s", id, g_out.data(), nv);
        memcpy(hr.data(), llama_get_embeddings_nextn_ith(dft, 0), ne * sizeof(float));
    }
    return 0;
}
