/*
 * Tests of the n-gram table reader: the hash against rows from the
 * llama.cpp qwen4exp hash loop, IQ4_NL dequantization, the three read
 * modes on a small table, reads with and without the io_uring, and the
 * real table against ggml_get_rows on a map of the file when LGML_NGRAM
 * names it. Output is TAP.
 */

#define _GNU_SOURCE

#include <sys/mman.h>

#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ggml.h>
#include <ggml-cpu.h>
#include <gguf.h>

#include "ngram.h"

#define EOS	248044

#if defined(LGML_HAVE_URING)
#define HAVE_RING 1
#else
#define HAVE_RING 0
#endif

static int ntest;

static void
ok(int cond, const char *name)
{
	printf("%s %d - %s\n", cond ? "ok" : "not ok", ++ntest, name);
	if (!cond)
		exit(1);
}

/* The hparams of Qwen3.8-Flash-Next. */
static void
real_params(struct ngram_params *p)
{
	static const uint64_t mult[] = { 23703573157769ULL, 20109073645365ULL, 8052911324071ULL };
	static const uint64_t vocab[] = {
		20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
		20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
	};
	int h;

	memset(p, 0, sizeof(*p));
	p->ngram_size = 3;
	p->heads_per_ngram = 8;
	p->n_heads = 16;
	p->eos = EOS;
	p->image_token = 248056;
	memcpy(p->mult, mult, sizeof(mult));
	for (h = 0; h < 16; h++) {
		p->vocab[h] = vocab[h];
		p->offset[h] = h == 0 ? 0 : p->offset[h - 1] + vocab[h - 1];
	}
}

/* Rows from the llama.cpp qwen4exp set_input loop, with the same hparams. */
static const int64_t want_start[] = {
	16410909, 39682429, 55103279, 60931720, 87006904, 116506179, 131512017, 152932897,
	169641436, 182022480, 209277433, 237891023, 256841529, 277007269, 290665954, 300984276,
	18158303, 36390029, 45652312, 70783524, 98191154, 114024957, 127804916, 159566246,
	175132467, 192583600, 217919476, 237199054, 259336807, 261799367, 289359313, 307699687,
	6380558, 26411572, 56460672, 78566983, 94693008, 106742196, 124822692, 148942556,
	164226950, 190352573, 210933682, 238908951, 242182004, 265475238, 299910982, 312121804,
};
static const int64_t want_eos_inside[] = {
	4377671, 22168600, 41067116, 71528461, 86212755, 117774917, 125118646, 148244800,
	164469549, 193283512, 210236278, 232977369, 257391123, 261805360, 289943793, 310636062,
	2760535, 21194529, 54821219, 66837341, 96008662, 113339850, 120218497, 157703057,
	168750120, 196316277, 203828420, 222763388, 257661759, 273618569, 292506187, 308707943,
	15389869, 39778609, 55713969, 62213332, 88817728, 118483999, 133731511, 155458159,
	179763390, 197956758, 205378969, 220499474, 242466248, 265658744, 293662119, 315720898,
	12441580, 26378836, 53347667, 75104214, 99467174, 114254887, 126436461, 156012011,
	169119442, 187827161, 214803956, 239809754, 242938905, 266427765, 294337448, 314484167,
};
static const int64_t want_own_eos[] = {
	11415239, 27851578, 50479349, 71096940, 99069428, 116000989, 132280215, 148965400,
	177284249, 185027128, 216077513, 222255373, 247238985, 273280895, 295530149, 308539434,
};
static const int64_t want_eos_back[] = {
	3507028, 28606606, 41157075, 74728528, 97790480, 118811318, 130342650, 152384705,
	165718412, 181070525, 210799816, 237959572, 259412465, 261224893, 295242566, 305929133,
};
static const int64_t want_one_prev[] = {
	14403033, 33639717, 43258169, 64724141, 80266516, 112114081, 129885364, 153580456,
	163045457, 183233866, 205271491, 221880011, 245749780, 269979123, 291468916, 319516527,
	11503690, 35232165, 59792284, 73195777, 96059626, 104256025, 127089827, 145207973,
	172047824, 192317046, 212568720, 235554561, 252482672, 270057781, 289393368, 307148989,
	16713332, 30002223, 53336330, 61496421, 86876599, 110097155, 126265854, 156988273,
	163419784, 186995665, 212617537, 222558678, 241204334, 261455488, 281896090, 306774286,
};

static void
check_rows(const struct ngram_params *p, const int32_t *prev, size_t nprev,
    const int32_t *ids, size_t n, const int64_t *want, const char *name)
{
	int64_t rows[4 * 16];

	ngram_rows(p, prev, nprev, ids, n, rows);
	ok(memcmp(rows, want, n * 16 * sizeof(*rows)) == 0, name);
}

static void
test_hash(void)
{
	struct ngram_params p;
	int32_t ids0[] = { 9707, 11, 1879 };
	int32_t prev1[] = { 100, 200 }, ids1[] = { 1, EOS, 5, 7 };
	int32_t prev2[] = { 3, 4 }, ids2[] = { EOS };
	int32_t prev3[] = { EOS, 9 }, ids3[] = { 42 };
	int32_t prev4[] = { 7 }, ids4[] = { 0, 151643, 248055 };
	int32_t prev5[] = { 1, 2, 3, 9707, 11 }, ids5[] = { 1879 };
	int32_t prev6[] = { 9707 }, ids6[] = { 11, 1879 };

	real_params(&p);
	check_rows(&p, NULL, 0, ids0, 3, want_start, "hash: sequence start");
	check_rows(&p, prev1, 2, ids1, 4, want_eos_inside, "hash: EOS inside the run");
	check_rows(&p, prev2, 2, ids2, 1, want_own_eos, "hash: own EOS keeps context");
	check_rows(&p, prev3, 2, ids3, 1, want_eos_back, "hash: EOS two back");
	check_rows(&p, prev4, 1, ids4, 3, want_one_prev, "hash: one token before");
	/* Older tokens of prev do not count. */
	check_rows(&p, prev5, 5, ids5, 1, want_start + 32, "hash: long prev");
	check_rows(&p, prev6, 1, ids6, 2, want_start + 16, "hash: split run");
}

/* An IQ4_NL block: f16 scale, then 16 bytes of two 4-bit codes each. */
struct iq4 {
	ggml_fp16_t	d;
	uint8_t		qs[16];
};

static const int8_t kvalues[16] = {
	-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

/* Code j of a block goes to element j, code j + 16 to the high nibble. */
static void
iq4_expect(const struct iq4 *b, float *y)
{
	float d = ggml_fp16_to_fp32(b->d);
	int j;

	for (j = 0; j < 16; j++) {
		y[j] = d * kvalues[b->qs[j] & 15];
		y[j + 16] = d * kvalues[b->qs[j] >> 4];
	}
}

static void
fill_block(struct iq4 *b, int64_t row, int blk)
{
	int j;

	b->d = ggml_fp32_to_fp16(1.0f / (float)(1 + (row + blk) % 7));
	for (j = 0; j < 16; j++)
		b->qs[j] = (uint8_t)(row * 31 + blk * 7 + j * 13);
}

static void
test_iq4(void)
{
	struct iq4 b;
	float y[32], want[32];
	int j;

	ok(sizeof(b) == 18 && ggml_type_size(GGML_TYPE_IQ4_NL) == 18, "iq4_nl block is 18 bytes");
	b.d = ggml_fp32_to_fp16(0.5f);
	for (j = 0; j < 16; j++)
		b.qs[j] = (uint8_t)(j | (15 - j) << 4);
	ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(&b, y, 32);
	for (j = 0; j < 16; j++) {
		want[j] = 0.5f * kvalues[j];
		want[j + 16] = 0.5f * kvalues[15 - j];
	}
	ok(memcmp(y, want, sizeof(y)) == 0, "iq4_nl: known block");
	fill_block(&b, 12345, 1);
	ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(&b, y, 32);
	iq4_expect(&b, want);
	ok(memcmp(y, want, sizeof(y)) == 0, "iq4_nl: generated block");
}

#define T_DIM	64		/* two blocks per row */
#define T_ROWS	20011
#define T_HEADS	4

/* Write a gguf with a T_ROWS x T_DIM IQ4_NL table and ple keys. */
static void
write_table(const char *path)
{
	static const uint64_t mult[] = { 1000003, 999983, 7 };
	uint64_t off[T_HEADS], voc[T_HEADS];
	struct ggml_init_params ip = { 1 << 20, NULL, 1 };
	struct ggml_context *ctx = ggml_init(ip);
	struct gguf_context *g = gguf_init_empty();
	struct ggml_tensor *t;
	struct iq4 *data;
	int64_t r;
	int h;

	for (h = 0; h < T_HEADS; h++) {
		voc[h] = 5000 + (uint64_t)h;
		off[h] = h == 0 ? 0 : off[h - 1] + voc[h - 1];
	}
	gguf_set_val_str(g, "general.architecture", "test");
	gguf_set_val_u32(g, "test.ple.ngram_size", 3);
	gguf_set_val_u32(g, "test.ple.heads_per_ngram", 2);
	gguf_set_val_u32(g, "test.ple.eos_token_id", 2);
	gguf_set_arr_data(g, "test.ple.layer_multipliers", GGUF_TYPE_UINT64, mult, 3);
	gguf_set_arr_data(g, "test.ple.head_offsets", GGUF_TYPE_UINT64, off, T_HEADS);
	gguf_set_arr_data(g, "test.ple.head_vocab_sizes", GGUF_TYPE_UINT64, voc, T_HEADS);
	t = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ4_NL, T_DIM, T_ROWS);
	ggml_set_name(t, "per_layer_token_embd.weight");
	data = malloc(ggml_nbytes(t));
	for (r = 0; r < T_ROWS; r++) {
		fill_block(&data[r * 2], r, 0);
		fill_block(&data[r * 2 + 1], r, 1);
	}
	t->data = data;
	gguf_add_tensor(g, t);
	ok(gguf_write_to_file(g, path, 0), "write a small table");
	free(data);
	gguf_free(g);
	ggml_free(ctx);
}

static void
test_modes(const char *path)
{
	static const char *names[] = { "direct", "pread", "mmap" };
	int32_t ids[] = { 5, 9, 5, 9, 5, 1, 2, 7, 7, 7, 7, 3, 2, 2 };
	size_t n = sizeof(ids) / sizeof(ids[0]), nr = n * T_HEADS, i;
	int64_t rows[sizeof(ids) / sizeof(ids[0]) * T_HEADS], all[] = { 0, T_ROWS - 1, 100, 99,
		101, 100 };
	float *out[3], want[T_DIM], got[6 * T_DIM];
	static float got2[sizeof(ids) / sizeof(ids[0]) * T_HEADS * T_DIM];
	struct ngram *r;
	char err[256], name[300];
	int m, good;

	for (m = 0; m < 3; m++) {
		r = ngram_open(path, NULL, "per_layer_token_embd.weight", (enum ngram_mode)m, 4096,
		    m == 0 ? 4 : 1, err, sizeof(err));
		snprintf(name, sizeof(name), "open %s: %s", names[m], r != NULL ? "ok" : err);
		ok(r != NULL, name);
		if (m == 0)
			printf("# direct mode %s\n", r->fell_back ? "fell back to pread" : "in use");
		ok(r->p.n_heads == T_HEADS && r->head_dim == T_DIM && r->nrows == T_ROWS &&
		    r->p.image_token == 2, "params and shape");
		ngram_rows(&r->p, NULL, 0, ids, n, rows);
		out[m] = malloc(nr * T_DIM * sizeof(float));
		ok(ngram_read(r, rows, nr, out[m]) == 0, "read rows");
		ok(r->st.rows == nr && r->st.unique < nr, "dedupe");
		good = 1;
		for (i = 0; i < nr; i++) {
			struct iq4 b[2];

			fill_block(&b[0], rows[i], 0);
			fill_block(&b[1], rows[i], 1);
			iq4_expect(&b[0], want);
			iq4_expect(&b[1], want + 32);
			good = good && memcmp(out[m] + i * T_DIM, want, sizeof(want)) == 0;
		}
		ok(good, "rows match their blocks");
		/* The first and last rows, and neighbors that share a block. */
		ok(ngram_read(r, all, 6, got) == 0, "read edge rows");
		good = 1;
		for (i = 0; i < 6; i++) {
			struct iq4 b[2];

			fill_block(&b[0], all[i], 0);
			fill_block(&b[1], all[i], 1);
			iq4_expect(&b[0], want);
			iq4_expect(&b[1], want + 32);
			good = good && memcmp(got + i * T_DIM, want, sizeof(want)) == 0;
		}
		ok(good, "edge rows match");
		all[0] = T_ROWS;
		ok(ngram_read(r, all, 1, got) == -1, "row past the end fails");
		all[0] = 0;
		ngram_close(r);
	}
	ok(memcmp(out[0], out[1], nr * T_DIM * sizeof(float)) == 0 &&
	    memcmp(out[1], out[2], nr * T_DIM * sizeof(float)) == 0, "modes give the same rows");
	/* A unit of 1 byte makes direct reads fail on most file systems. */
	r = ngram_open(path, NULL, "per_layer_token_embd.weight", NGRAM_DIRECT, 1, 1, err,
	    sizeof(err));
	ok(r != NULL, "open direct with a bad unit");
	printf("# direct mode with a 1-byte unit %s\n", r->fell_back ? "fell back" : "in use");
	ok(ngram_read(r, rows, 6, got) == 0, "read after fallback");
	ok(memcmp(got, out[0], 6 * T_DIM * sizeof(float)) == 0, "fallback gives the same rows");
	ngram_close(r);
	r = ngram_open(path, NULL, "per_layer_token_embd.weight", NGRAM_DIRECT, 0, 2, err,
	    sizeof(err));
	ok(r != NULL && (r->align == 512 || r->align == 4096), "open direct with a found unit");
	printf("# found unit %d, mode %d\n", (int)r->align, (int)r->mode);
	ok(ngram_read(r, rows, nr, got2) == 0 &&
	    memcmp(got2, out[0], nr * T_DIM * sizeof(float)) == 0, "found unit gives the same rows");
	ngram_close(r);
	ok(ngram_open(path, NULL, "nope", NGRAM_PREAD, 4096, 1, err, sizeof(err)) == NULL &&
	    strstr(err, "no tensor") != NULL, "missing tensor");
	ok(ngram_open(path, NULL, "per_layer_token_embd.weight", NGRAM_PREAD, 3000, 1, err,
	    sizeof(err)) == NULL, "align must be a power of two");
	for (m = 0; m < 3; m++)
		free(out[m]);
}

/* Check rows[0..n) of out against their blocks. */
static int
rows_match(const int64_t *rows, size_t n, const float *out)
{
	struct iq4 b[2];
	float want[T_DIM];
	size_t i;

	for (i = 0; i < n; i++) {
		fill_block(&b[0], rows[i], 0);
		fill_block(&b[1], rows[i], 1);
		iq4_expect(&b[0], want);
		iq4_expect(&b[1], want + 32);
		if (memcmp(out + i * T_DIM, want, sizeof(want)) != 0)
			return 0;
	}
	return 1;
}

struct far_read {
	struct ngram	*r;
	const int64_t	*rows;
	size_t		 n;
	float		*out;
	int		 ret;
};

static void *
far_read(void *arg)
{
	struct far_read *f = arg;

	f->ret = ngram_read(f->r, f->rows, f->n, f->out);
	return NULL;
}

/*
 * Reads with and without the io_uring. Every other row in pread mode
 * makes a span per row, many more than the ring holds at once.
 */
static void
test_ring(const char *path)
{
	static int64_t rows[T_ROWS / 2 + 1];
	static float out[2][(T_ROWS / 2 + 1) * T_DIM];
	struct far_read f;
	struct ngram *r;
	pthread_t tid;
	char err[256], name[64];
	size_t n = 0;
	int64_t row;
	int m, on, use[2] = { 0, 0 };

	for (row = T_ROWS - 1; row >= 0; row -= 2)
		rows[n++] = row;
	for (m = 0; m < 2; m++) {
		for (on = 0; on < 2; on++) {
			r = ngram_open(path, NULL, "per_layer_token_embd.weight", (enum ngram_mode)m,
			    0, 4, err, sizeof(err));
			ok(r != NULL, "ring: open");
			use[on] = ngram_uring(r, on);
			ok(ngram_read(r, rows, n, out[on]) == 0, "ring: read every other row");
			printf("# mode %d, ring %d: %d reads\n", (int)r->mode, use[on],
			    (int)r->st.reads);
			ngram_close(r);
		}
		snprintf(name, sizeof(name), "ring: %s rows match", m == 0 ? "direct" : "pread");
		ok(!use[0] && rows_match(rows, n, out[0]) &&
		    memcmp(out[0], out[1], n * T_DIM * sizeof(float)) == 0, name);
	}
	ok(use[1] == HAVE_RING, "ring: in use when built with liburing");
	/* Another thread reads: the reader makes a new ring there. */
	r = ngram_open(path, NULL, "per_layer_token_embd.weight", NGRAM_DIRECT, 0, 4, err,
	    sizeof(err));
	ok(r != NULL, "ring: open for threads");
	f.r = r;
	f.rows = rows;
	f.n = 50;
	f.out = out[0];
	f.ret = -1;
	ok(pthread_create(&tid, NULL, far_read, &f) == 0 && pthread_join(tid, NULL) == 0 &&
	    f.ret == 0 && rows_match(rows, 50, out[0]), "ring: read on another thread");
	ok(ngram_read(r, rows + 50, 50, out[1]) == 0 && rows_match(rows + 50, 50, out[1]),
	    "ring: read on this thread again");
	printf("# ring %s after the thread reads\n", r->ring != NULL ? "in use" : "not in use");
	ngram_close(r);
}

/* Many random rows of the real table, with and without the ring. */
static void
test_real_ring(const char *path, const char *meta)
{
	static const struct {
		enum ngram_mode	 mode;
		int		 ring;
		const char	*name;
	} cfg[] = {
		{ NGRAM_MMAP, 0, "mmap" },
		{ NGRAM_DIRECT, 1, "direct, ring" },
		{ NGRAM_DIRECT, 0, "direct, threads" },
		{ NGRAM_PREAD, 1, "pread, ring" },
	};
	size_t nr = 8192, i, c, len = 0;
	uint64_t x = 88172645463325252ULL;
	struct ngram *r;
	int64_t *rows;
	float *out[4];
	char err[512], name[128];

	rows = malloc(nr * sizeof(*rows));
	for (c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++) {
		r = ngram_open(path, meta, "per_layer_token_embd.weight", cfg[c].mode, 0, 8, err,
		    sizeof(err));
		ok(r != NULL, r != NULL ? "real ring: open" : err);
		if (c == 0)
			for (i = 0; i < nr; i++) {
				x ^= x << 13;
				x ^= x >> 7;
				x ^= x << 17;
				rows[i] = (int64_t)(x % (uint64_t)r->nrows);
			}
		len = nr * (size_t)r->head_dim;
		out[c] = malloc(len * sizeof(float));
		snprintf(name, sizeof(name), "real ring: %s", cfg[c].name);
		ok(ngram_uring(r, cfg[c].ring) == (cfg[c].ring && HAVE_RING) &&
		    ngram_read(r, rows, nr, out[c]) == 0, name);
		printf("# %s: %d reads, %.3f ms\n", cfg[c].name, (int)r->st.reads,
		    r->st.total_sec * 1e3);
		if (c > 0) {
			snprintf(name, sizeof(name), "real ring: %s matches mmap", cfg[c].name);
			ok(memcmp(out[0], out[c], len * sizeof(float)) == 0, name);
		}
		ngram_close(r);
	}
	for (c = 0; c < sizeof(cfg) / sizeof(cfg[0]); c++)
		free(out[c]);
	free(rows);
}

/* The real table against ggml_get_rows on a map of the file. */
static void
test_real(const char *path, const char *meta)
{
	int32_t ids[] = { 248045, 846, 198, 9707, 11, 1879, 0, 248044, 151643, 9707, 11 };
	int32_t prev[] = { 13, 248044 };
	size_t n = sizeof(ids) / sizeof(ids[0]), nr, i;
	struct ggml_init_params ip = { 64 << 20, NULL, 0 }, wp = { 1 << 16, NULL, 1 };
	struct ggml_context *ctx, *wctx;
	struct ggml_tensor *tab, *idx, *res;
	struct ggml_cgraph *gf;
	struct ngram *r;
	char err[512];
	int64_t *rows;
	float *out[3];
	void *map;
	size_t maplen;
	int fd, m;

	for (m = 0; m < 3; m++) {
		r = ngram_open(path, meta, "per_layer_token_embd.weight", (enum ngram_mode)m, 0,
		    8, err, sizeof(err));
		ok(r != NULL, r != NULL ? "real: open" : err);
		nr = n * (size_t)r->p.n_heads;
		rows = malloc(nr * sizeof(*rows));
		out[m] = malloc(nr * (size_t)r->head_dim * sizeof(float));
		ngram_rows(&r->p, prev, 2, ids, n, rows);
		ok(ngram_read(r, rows, nr, out[m]) == 0, "real: read");
		printf("# mode %d: %d rows, %d unique, %d reads, %d bytes, %.3f ms\n", (int)r->mode,
		    (int)r->st.rows, (int)r->st.unique, (int)r->st.reads, (int)r->st.bytes,
		    r->st.total_sec * 1e3);
		if (m < 2) {
			ngram_close(r);
			free(rows);
		}
	}
	ok(memcmp(out[0], out[1], nr * (size_t)r->head_dim * sizeof(float)) == 0 &&
	    memcmp(out[0], out[2], nr * (size_t)r->head_dim * sizeof(float)) == 0,
	    "real: modes agree");

	fd = open(path, O_RDONLY);
	maplen = (size_t)lseek(fd, 0, SEEK_END);
	map = mmap(NULL, maplen, PROT_READ, MAP_SHARED, fd, 0);
	ok(map != MAP_FAILED, "real: map the file");
	ctx = ggml_init(ip);
	wctx = ggml_init(wp);
	tab = ggml_new_tensor_2d(wctx, r->type, r->head_dim, r->nrows);
	idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)nr);
	/* get_rows reads the table in place; nothing writes to it. */
	tab->data = (char *)map + r->base;
	for (i = 0; i < nr; i++)
		((int32_t *)idx->data)[i] = (int32_t)rows[i];
	res = ggml_get_rows(ctx, tab, idx);
	gf = ggml_new_graph(ctx);
	ggml_build_forward_expand(gf, res);
	ok(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS, "real: get_rows");
	ok(memcmp(res->data, out[0], nr * (size_t)r->head_dim * sizeof(float)) == 0,
	    "real: reader matches get_rows");
	ggml_free(ctx);
	ggml_free(wctx);
	munmap(map, maplen);
	close(fd);
	ngram_close(r);
	free(rows);
	for (m = 0; m < 3; m++)
		free(out[m]);
}

int
main(void)
{
	const char *real = getenv("LGML_NGRAM"), *meta = getenv("LGML_NGRAM_META");
	char path[64];

	test_hash();
	test_iq4();
	snprintf(path, sizeof(path), "ngram-test-%d.gguf", (int)getpid());
	write_table(path);
	test_modes(path);
	test_ring(path);
	unlink(path);
	if (meta != NULL && *meta == '\0')
		meta = NULL;
	if (real != NULL && *real != '\0' && access(real, R_OK) == 0) {
		test_real(real, meta);
		test_real_ring(real, meta);
	} else
		printf("ok %d - real table # SKIP LGML_NGRAM not set\n", ++ntest);
	printf("1..%d\n", ntest);
	return 0;
}
