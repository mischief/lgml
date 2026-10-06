/* llama.cpp normed hidden states of a prompt with images, on the CPU.
 * cc -I$L/include -I$L/ggml/include -I$L/tools/mtmd mtmd_hidden.c -L$B -lmtmd -lllama -lggml-base
 * Usage: mtmd_hidden model mmproj entries.txt out.f32 [embd.f32]
 * entries.txt: token ids and "img:<path>" words, one image per word.
 * Writes a row per token and the image embeddings. MTMD_NO_FA turns off
 * flash attention. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>

struct image {
	mtmd_input_chunks *chunks;
	const mtmd_input_chunk *chunk;
	float *embd;
	size_t n;
};

static void
quiet(enum ggml_log_level level, const char *text, void *arg)
{
	(void)arg;
	if (level == GGML_LOG_LEVEL_ERROR || getenv("MTMD_VERBOSE") != NULL)
		fputs(text, stderr);
}

/* MTMD_DUMP: comma-separated encoder node names to write to <name>.f32. */
static bool
dump(struct ggml_tensor *t, bool ask, void *arg)
{
	const char *want = getenv("MTMD_DUMP");
	char path[GGML_MAX_NAME + 8];
	size_t n = strlen(t->name);
	const char *p;
	FILE *f;
	void *buf;

	(void)arg;
	for (p = want; p != NULL && *p != '\0'; p += strcspn(p, ",") + (p[strcspn(p, ",")] != '\0'))
		if (strncmp(p, t->name, n) == 0 && (p[n] == ',' || p[n] == '\0'))
			break;
	if (p == NULL || *p == '\0')
		return ask ? false : true;
	if (ask)
		return true;
	if (t->type != GGML_TYPE_F32 || (buf = malloc(ggml_nbytes(t))) == NULL)
		return true;
	ggml_backend_tensor_get(t, buf, 0, ggml_nbytes(t));
	snprintf(path, sizeof(path), "%s.f32", t->name);
	if ((f = fopen(path, "wb")) != NULL) {
		fwrite(buf, 1, ggml_nbytes(t), f);
		fclose(f);
	}
	free(buf);
	return true;
}

static int
encode(mtmd_context *ctx, const char *path, int n_embd, struct image *im)
{
	struct mtmd_helper_bitmap_wrapper bw;
	const mtmd_bitmap *bm[1];
	mtmd_input_text text;
	size_t i;

	bw = mtmd_helper_bitmap_init_from_file(ctx, path, false, mtmd_helper_init_opt_default());
	if (bw.bitmap == NULL) {
		fprintf(stderr, "%s: decode failed\n", path);
		return -1;
	}
	memset(&text, 0, sizeof(text));
	text.text = mtmd_default_marker();
	text.text_len = strlen(text.text);
	text.add_special = false;
	text.parse_special = true;
	bm[0] = bw.bitmap;
	im->chunks = mtmd_input_chunks_init();
	if (mtmd_tokenize(ctx, im->chunks, &text, bm, 1) != 0) {
		fprintf(stderr, "%s: tokenize failed\n", path);
		return -1;
	}
	for (i = 0; i < mtmd_input_chunks_size(im->chunks); i++) {
		const mtmd_input_chunk *c = mtmd_input_chunks_get(im->chunks, i);

		if (mtmd_input_chunk_get_type(c) != MTMD_INPUT_CHUNK_TYPE_IMAGE)
			continue;
		if (mtmd_encode_chunk(ctx, c) != 0) {
			fprintf(stderr, "%s: encode failed\n", path);
			return -1;
		}
		im->chunk = c;
		im->n = mtmd_input_chunk_get_n_tokens(c);
		im->embd = malloc(im->n * (size_t)n_embd * sizeof(float));
		if (im->embd == NULL)
			return -1;
		memcpy(im->embd, mtmd_get_output_embd(ctx), im->n * (size_t)n_embd * sizeof(float));
		return 0;
	}
	fprintf(stderr, "%s: no image chunk\n", path);
	return -1;
}

int
main(int argc, char **argv)
{
	struct llama_model_params mp;
	struct llama_context_params cp;
	struct mtmd_context_params mcp;
	struct llama_model *model;
	struct llama_context *lctx;
	struct llama_batch_ext *batch;
	struct image *ims = NULL;
	mtmd_context *ctx;
	char word[4096];
	int *ids = NULL, *kind = NULL, n_embd, n_img = 0, cap = 0, n = 0, i, total = 0;
	llama_pos p = 0, pp[4];
	FILE *f, *out, *eout = NULL;
	size_t j;

	if (argc != 5 && argc != 6) {
		fprintf(stderr, "usage: mtmd_hidden model mmproj entries.txt out.f32 [embd.f32]\n");
		return 2;
	}
	llama_log_set(quiet, NULL);
	mtmd_helper_log_set(quiet, NULL);
	llama_backend_init();
	mp = llama_model_default_params();
	mp.n_gpu_layers = 0;
	model = llama_model_load_from_file(argv[1], mp);
	if (model == NULL) {
		fprintf(stderr, "%s: load failed\n", argv[1]);
		return 1;
	}
	mcp = mtmd_context_params_default();
	mcp.use_gpu = false;
	mcp.print_timings = false;
	mcp.warmup = false;
	if (getenv("MTMD_NO_FA") != NULL)
		mcp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
	if (getenv("MTMD_DUMP") != NULL)
		mcp.cb_eval = dump;
	ctx = mtmd_init_from_file(argv[2], model, mcp);
	if (ctx == NULL) {
		fprintf(stderr, "%s: load failed\n", argv[2]);
		return 1;
	}
	n_embd = llama_model_n_embd_inp(model);
	if ((f = fopen(argv[3], "r")) == NULL) {
		perror(argv[3]);
		return 1;
	}
	/* kind: -1 for a token, else the image index */
	while (fscanf(f, "%4095s", word) == 1) {
		if (n == cap) {
			cap = cap ? 2 * cap : 1024;
			ids = realloc(ids, cap * sizeof(int));
			kind = realloc(kind, cap * sizeof(int));
		}
		if (strncmp(word, "img:", 4) == 0) {
			ims = realloc(ims, (n_img + 1) * sizeof(*ims));
			memset(&ims[n_img], 0, sizeof(*ims));
			if (encode(ctx, word + 4, n_embd, &ims[n_img]) != 0)
				return 1;
			total += (int)ims[n_img].n;
			ids[n] = 0;
			kind[n++] = n_img++;
		} else {
			ids[n] = atoi(word);
			kind[n++] = -1;
			total++;
		}
	}
	fclose(f);
	cp = llama_context_default_params();
	cp.n_ctx = total + 64;
	cp.n_batch = cp.n_ubatch = total;
	cp.embeddings = true;
	cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
	if (getenv("MTMD_NO_FA") != NULL)
		cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
	lctx = llama_init_from_model(model, cp);
	if (lctx == NULL) {
		fprintf(stderr, "context failed\n");
		return 1;
	}
	batch = llama_batch_ext_init(lctx);
	for (i = 0; i < n; i++) {
		int idx;

		if (kind[i] < 0) {
			idx = llama_batch_ext_add_token(batch, 0, ids[i]);
			pp[0] = p++;
			if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, pp))
				return 1;
			llama_batch_ext_set_output_embd(batch, idx, true);
			continue;
		}
		struct image *im = &ims[kind[i]];
		const mtmd_image_tokens *it = mtmd_input_chunk_get_tokens_image(im->chunk);
		for (j = 0; j < im->n; j++) {
			struct llama_embd e = { im->embd + j * (size_t)n_embd, 1, (size_t)n_embd };
			struct mtmd_decoder_pos d = mtmd_image_tokens_get_decoder_pos(it, p, j);

			idx = llama_batch_ext_add_embd(batch, 0, e);
			pp[0] = (llama_pos)d.t;
			pp[1] = (llama_pos)d.y;
			pp[2] = (llama_pos)d.x;
			pp[3] = (llama_pos)d.z;
			if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, pp))
				return 1;
			llama_batch_ext_set_output_embd(batch, idx, true);
		}
		printf("image %zu %d\n", im->n, (int)mtmd_input_chunk_get_n_pos(im->chunk));
		p += mtmd_input_chunk_get_n_pos(im->chunk);
	}
	if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
		fprintf(stderr, "decode failed\n");
		return 1;
	}
	if ((out = fopen(argv[4], "wb")) == NULL) {
		perror(argv[4]);
		return 1;
	}
	for (i = 0; i < total; i++) {
		const float *h = llama_get_embeddings_ith(lctx, i);

		if (h == NULL || fwrite(h, sizeof(float), llama_model_n_embd(model), out) !=
		    (size_t)llama_model_n_embd(model)) {
			fprintf(stderr, "no hidden state %d\n", i);
			return 1;
		}
	}
	fclose(out);
	if (argc == 6 && (eout = fopen(argv[5], "wb")) != NULL) {
		for (i = 0; i < n_img; i++)
			fwrite(ims[i].embd, sizeof(float), ims[i].n * (size_t)n_embd, eout);
		fclose(eout);
	}
	printf("%d %d\n", total, llama_model_n_embd(model));
	llama_batch_ext_free(batch);
	llama_free(lctx);
	mtmd_free(ctx);
	llama_model_free(model);
	return 0;
}
