/* Write llama.cpp (mtmd) image embeddings, from the CPU. Build with
 * cc -I$L/include -I$L/ggml/include -I$L/tools/mtmd mtmd_embd.c -L$L/build/bin -lmtmd -lllama
 * Usage: mtmd_embd model.gguf mmproj.gguf image out.f32 [out.rgb]
 * out.f32 gets n_tokens rows of n_embd f32; out.rgb the decoded RGB
 * bytes; stdout "n_tokens n_embd nx ny". MTMD_NO_FA turns off flash attention. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>

static void
quiet(enum ggml_log_level level, const char *text, void *arg)
{
	(void)arg;
	if (level == GGML_LOG_LEVEL_ERROR)
		fputs(text, stderr);
}

static int
save(const char *path, const void *p, size_t n)
{
	FILE *f = fopen(path, "wb");

	if (f == NULL || fwrite(p, 1, n, f) != n || fclose(f) != 0) {
		perror(path);
		return -1;
	}
	return 0;
}

int
main(int argc, char **argv)
{
	struct llama_model_params mp;
	struct mtmd_context_params cp;
	struct mtmd_helper_bitmap_wrapper bw;
	struct llama_model *model;
	mtmd_context *ctx;
	mtmd_input_chunks *chunks;
	mtmd_input_text text;
	const mtmd_bitmap *bm[1];
	size_t i, n;
	int n_embd;

	if (argc != 5 && argc != 6) {
		fprintf(stderr, "usage: mtmd_embd model mmproj image out.f32 [out.rgb]\n");
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
	cp = mtmd_context_params_default();
	cp.use_gpu = false;
	cp.print_timings = false;
	cp.warmup = false;
	if (getenv("MTMD_NO_FA") != NULL)
		cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
	ctx = mtmd_init_from_file(argv[2], model, cp);
	if (ctx == NULL) {
		fprintf(stderr, "%s: load failed\n", argv[2]);
		return 1;
	}
	bw = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default());
	if (bw.bitmap == NULL) {
		fprintf(stderr, "%s: decode failed\n", argv[3]);
		return 1;
	}
	if (argc == 6 && save(argv[5], mtmd_bitmap_get_data(bw.bitmap),
	    mtmd_bitmap_get_n_bytes(bw.bitmap)) != 0)
		return 1;
	memset(&text, 0, sizeof(text));
	text.text = mtmd_default_marker();
	text.text_len = strlen(text.text);
	text.add_special = false;
	text.parse_special = true;
	bm[0] = bw.bitmap;
	chunks = mtmd_input_chunks_init();
	if (mtmd_tokenize(ctx, chunks, &text, bm, 1) != 0) {
		fprintf(stderr, "tokenize failed\n");
		return 1;
	}
	n_embd = llama_model_n_embd_inp(model);
	for (i = 0; i < mtmd_input_chunks_size(chunks); i++) {
		const mtmd_input_chunk *c = mtmd_input_chunks_get(chunks, i);

		if (mtmd_input_chunk_get_type(c) != MTMD_INPUT_CHUNK_TYPE_IMAGE)
			continue;
		if (mtmd_encode_chunk(ctx, c) != 0) {
			fprintf(stderr, "encode failed\n");
			return 1;
		}
		n = mtmd_input_chunk_get_n_tokens(c);
		if (save(argv[4], mtmd_get_output_embd(ctx), n * (size_t)n_embd * sizeof(float)) != 0)
			return 1;
		printf("%zu %d %u %u\n", n, n_embd, mtmd_bitmap_get_nx(bw.bitmap),
		    mtmd_bitmap_get_ny(bw.bitmap));
	}
	mtmd_input_chunks_free(chunks);
	mtmd_free(ctx);
	llama_model_free(model);
	return 0;
}
