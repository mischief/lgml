/*
 * Print llama.cpp logits for fixed token ids. Build against a llama.cpp
 * tree: cc -I$L/include -I$L/ggml/include llama_logits.c -L$L/build/bin -lllama
 * Usage: llama_logits model.gguf n_predict id ... [/ id ...]
 * "/" ends a batch. Each token of a batch, then each greedy step, prints
 * one row: the input id, then n_vocab logits.
 * Env: LL_NGL sets GPU layers (default 0); LL_FA turns on flash attention;
 * LL_CTK and LL_CTV set the cache types (f16 or q8_0 or q4_0).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <llama.h>

static void
quiet(enum ggml_log_level level, const char *text, void *arg)
{
	(void)arg;
	if (level == GGML_LOG_LEVEL_ERROR)
		fputs(text, stderr);
}

static int
row(struct llama_context *ctx, int i, int id, int nv)
{
	const float *p = llama_get_logits_ith(ctx, i);
	int j, best = 0;

	printf("%d", id);
	for (j = 0; j < nv; j++) {
		printf(" %.9g", p[j]);
		if (p[j] > p[best])
			best = j;
	}
	printf("\n");
	return best;
}

static int
run(struct llama_context *ctx, const int *ids, int n, int nv, int *pos)
{
	struct llama_batch b = llama_batch_init(n, 0, 1);
	int i, best = 0;

	for (i = 0; i < n; i++) {
		b.token[i] = ids[i];
		b.pos[i] = *pos + i;
		b.n_seq_id[i] = 1;
		b.seq_id[i][0] = 0;
		b.logits[i] = 1;
	}
	b.n_tokens = n;
	if (llama_decode(ctx, b) != 0) {
		fprintf(stderr, "llama_decode failed\n");
		exit(1);
	}
	for (i = 0; i < n; i++)
		best = row(ctx, i, ids[i], nv);
	*pos += n;
	llama_batch_free(b);
	return best;
}

/* Cache type from a name; f16 when name is NULL. */
static enum ggml_type
cache_type(const char *name)
{
	if (name == NULL || strcmp(name, "f16") == 0)
		return GGML_TYPE_F16;
	if (strcmp(name, "q8_0") == 0)
		return GGML_TYPE_Q8_0;
	if (strcmp(name, "q4_0") == 0)
		return GGML_TYPE_Q4_0;
	fprintf(stderr, "unknown cache type %s\n", name);
	exit(2);
}

int
main(int argc, char *argv[])
{
	struct llama_model_params mp;
	struct llama_context_params cp;
	struct llama_model *model;
	struct llama_context *ctx;
	int *ids, i, n = 0, nv, npred, pos = 0, next = 0;

	if (argc < 4) {
		fprintf(stderr, "usage: llama_logits model n_predict id ... [/ id ...]\n");
		return 2;
	}
	llama_log_set(quiet, NULL);
	llama_backend_init();
	mp = llama_model_default_params();
	mp.n_gpu_layers = getenv("LL_NGL") != NULL ? atoi(getenv("LL_NGL")) : 0;
	model = llama_model_load_from_file(argv[1], mp);
	if (model == NULL)
		return 1;
	cp = llama_context_default_params();
	cp.n_ctx = 256;
	cp.n_batch = 256;
	cp.n_ubatch = 256;
	cp.n_threads = cp.n_threads_batch = 4;
	cp.flash_attn_type = getenv("LL_FA") != NULL ? LLAMA_FLASH_ATTN_TYPE_ENABLED :
	    LLAMA_FLASH_ATTN_TYPE_DISABLED;
	cp.type_k = cache_type(getenv("LL_CTK"));
	cp.type_v = cache_type(getenv("LL_CTV"));
	ctx = llama_init_from_model(model, cp);
	if (ctx == NULL)
		return 1;
	nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
	npred = atoi(argv[2]);
	if ((ids = calloc(argc, sizeof(*ids))) == NULL)
		return 1;
	for (i = 3; i <= argc; i++) {
		if (i == argc || strcmp(argv[i], "/") == 0) {
			if (n > 0)
				next = run(ctx, ids, n, nv, &pos);
			n = 0;
		} else
			ids[n++] = atoi(argv[i]);
	}
	for (i = 0; i < npred; i++)
		next = run(ctx, &next, 1, nv, &pos);
	llama_free(ctx);
	llama_model_free(model);
	free(ids);
	return 0;
}
