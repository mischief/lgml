/*
 * PLE n-gram hash embedding reader. The table is too large for memory,
 * so each pass reads only the rows it needs. A pass dedupes its rows,
 * sorts them by file offset, merges them into aligned spans, reads the
 * spans on a few threads and dequantizes each row to f32.
 */

/* O_DIRECT, posix_fadvise and posix_memalign. */
#define _GNU_SOURCE

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <ggml.h>
#include <gguf.h>

#include "ngram.h"

#define SPAN_MAX	((size_t)1 << 20)	/* longest merged read */
#define BUF_MAX		((size_t)32 << 20)	/* read buffer of one chunk */

struct pair {
	int64_t		 row;
	size_t		 i;		/* index in the caller's row list */
};

struct span {
	off_t		 off;		/* file offset */
	size_t		 len;		/* bytes to read */
	size_t		 need;		/* bytes that must arrive */
	size_t		 boff;		/* offset in the read buffer */
};

struct job {
	struct ngram	*r;
	int		 fd;
	struct span	*sp;
	size_t		 from, to;
	uint64_t	 reads;
	int		 err;		/* errno of the first failure */
	pthread_t	 tid;
};

static double
now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void seterr(char *, size_t, const char *, ...)
    __attribute__((__format__(__printf__, 3, 4)));

static void
seterr(char *err, size_t len, const char *fmt, ...)
{
	va_list ap;

	if (err == NULL || len == 0)
		return;
	va_start(ap, fmt);
	vsnprintf(err, len, fmt, ap);
	va_end(ap);
}

/* An integer of a gguf scalar type at p, or -1 for other types. */
static int
gguf_int(enum gguf_type type, const void *p, uint64_t *v)
{
	switch (type) {
	case GGUF_TYPE_UINT8:	*v = *(const uint8_t *)p; return 0;
	case GGUF_TYPE_INT8:	*v = (uint64_t)*(const int8_t *)p; return 0;
	case GGUF_TYPE_UINT16:	*v = *(const uint16_t *)p; return 0;
	case GGUF_TYPE_INT16:	*v = (uint64_t)*(const int16_t *)p; return 0;
	case GGUF_TYPE_UINT32:	*v = *(const uint32_t *)p; return 0;
	case GGUF_TYPE_INT32:	*v = (uint64_t)*(const int32_t *)p; return 0;
	case GGUF_TYPE_UINT64:	*v = *(const uint64_t *)p; return 0;
	case GGUF_TYPE_INT64:	*v = (uint64_t)*(const int64_t *)p; return 0;
	default:		return -1;
	}
}

static size_t
gguf_int_size(enum gguf_type type)
{
	switch (type) {
	case GGUF_TYPE_UINT8:
	case GGUF_TYPE_INT8:	return 1;
	case GGUF_TYPE_UINT16:
	case GGUF_TYPE_INT16:	return 2;
	case GGUF_TYPE_UINT32:
	case GGUF_TYPE_INT32:	return 4;
	default:		return 8;
	}
}

/* Key prefix + name: a scalar integer into *v, or n array integers into v. */
static int
get_ints(const struct gguf_context *c, const char *prefix, const char *name,
    uint64_t *v, int n, int optional, char *err, size_t errlen)
{
	char key[256];
	const char *p;
	enum gguf_type type;
	int64_t id;
	size_t sz;
	int i;

	snprintf(key, sizeof(key), "%s%s", prefix, name);
	id = gguf_find_key(c, key);
	if (id < 0) {
		if (optional)
			return 1;
		seterr(err, errlen, "%s%s: key missing", prefix, name);
		return -1;
	}
	type = gguf_get_kv_type(c, id);
	if (n == 0) {
		/* gguf_get_val_data aborts on a string. */
		if (type != GGUF_TYPE_ARRAY && type != GGUF_TYPE_STRING &&
		    gguf_int(type, gguf_get_val_data(c, id), v) == 0)
			return 0;
		seterr(err, errlen, "%s%s: not an integer", prefix, name);
		return -1;
	}
	if (type != GGUF_TYPE_ARRAY || gguf_get_arr_n(c, id) < (size_t)n) {
		seterr(err, errlen, "%s%s: need a longer integer array", prefix, name);
		return -1;
	}
	type = gguf_get_arr_type(c, id);
	if (type == GGUF_TYPE_STRING || type == GGUF_TYPE_ARRAY) {
		seterr(err, errlen, "%s%s: not an integer array", prefix, name);
		return -1;
	}
	sz = gguf_int_size(type);
	p = gguf_get_arr_data(c, id);
	for (i = 0; i < n; i++)
		if (gguf_int(type, p + (size_t)i * sz, &v[i]) != 0) {
			seterr(err, errlen, "%s%s: not an integer array", prefix, name);
			return -1;
		}
	return 0;
}

/*
 * Fill p from the <arch>.ple.* keys of c. The arch prefix is that of the
 * key that ends in ".ple.ngram_size". Returns 0, or -1 with a message in
 * err.
 */
int
ngram_params_gguf(struct ngram_params *p, const struct gguf_context *c, char *err,
    size_t errlen)
{
	static const char suffix[] = "ngram_size";
	char prefix[200];
	const char *key;
	uint64_t v;
	size_t len;
	int64_t i, n;
	int h, r;

	memset(p, 0, sizeof(*p));
	prefix[0] = '\0';
	n = gguf_get_n_kv(c);
	for (i = 0; i < n; i++) {
		key = gguf_get_key(c, i);
		len = strlen(key);
		if (len > 15 && len < sizeof(prefix) &&
		    strcmp(key + len - 15, ".ple.ngram_size") == 0) {
			memcpy(prefix, key, len - (sizeof(suffix) - 1));
			prefix[len - (sizeof(suffix) - 1)] = '\0';
			break;
		}
	}
	if (prefix[0] == '\0') {
		seterr(err, errlen, "no <arch>.ple.%s key", suffix);
		return -1;
	}
	if (get_ints(c, prefix, "ngram_size", &v, 0, 0, err, errlen) != 0)
		return -1;
	if (v < 2 || v > NGRAM_MAX_N) {
		seterr(err, errlen, "%s%s out of range", prefix, "ngram_size");
		return -1;
	}
	p->ngram_size = (int)v;
	if (get_ints(c, prefix, "heads_per_ngram", &v, 0, 0, err, errlen) != 0)
		return -1;
	if (v == 0 || v * (uint64_t)(p->ngram_size - 1) > NGRAM_MAX_HEADS) {
		seterr(err, errlen, "%s%s out of range", prefix, "heads_per_ngram");
		return -1;
	}
	p->heads_per_ngram = (int)v;
	p->n_heads = (p->ngram_size - 1) * p->heads_per_ngram;
	if (get_ints(c, prefix, "eos_token_id", &v, 0, 0, err, errlen) != 0)
		return -1;
	p->eos = (int32_t)v;
	r = get_ints(c, prefix, "image_token_id", &v, 0, 1, err, errlen);
	if (r < 0)
		return -1;
	p->image_token = r == 0 && v != 0 ? (int32_t)v : p->eos;
	if (get_ints(c, prefix, "layer_multipliers", p->mult, p->ngram_size, 0, err,
	    errlen) != 0 ||
	    get_ints(c, prefix, "head_offsets", p->offset, p->n_heads, 0, err, errlen) != 0 ||
	    get_ints(c, prefix, "head_vocab_sizes", p->vocab, p->n_heads, 0, err,
	    errlen) != 0)
		return -1;
	for (h = 0; h < p->n_heads; h++)
		if (p->vocab[h] == 0 || p->offset[h] > INT32_MAX || p->vocab[h] > INT32_MAX ||
		    p->offset[h] + p->vocab[h] > INT32_MAX) {
			seterr(err, errlen, "%s%s: a head range is out of range", prefix,
			    "head_vocab_sizes");
			return -1;
		}
	return 0;
}

/* The token s places before ids[i], or -1 before the history starts. */
static int32_t
before(const int32_t *prev, size_t nprev, const int32_t *ids, size_t i, size_t s)
{
	if (i >= s)
		return ids[i - s];
	if (nprev + i >= s)
		return prev[nprev + i - s];
	return -1;
}

/*
 * Write the n_heads table rows of each token of ids to rows, token-major.
 * prev holds the nprev tokens before ids[0], oldest first. A missing
 * token, or an EOS before the current token, ends the context: it and
 * all older tokens read as EOS. This matches llama.cpp qwen4exp.
 */
void
ngram_rows(const struct ngram_params *p, const int32_t *prev, size_t nprev,
    const int32_t *ids, size_t n, int64_t *rows)
{
	int64_t ctx[NGRAM_MAX_N];
	uint64_t mixed;
	size_t i;
	int32_t t;
	int stop, g, h, j, k, s;

	for (i = 0; i < n; i++) {
		ctx[0] = ids[i];
		stop = 0;
		for (s = 1; s < p->ngram_size; s++) {
			t = stop ? -1 : before(prev, nprev, ids, i, (size_t)s);
			stop = stop || t < 0 || t == p->eos;
			ctx[s] = stop ? p->eos : t;
		}
		for (k = 2; k <= p->ngram_size; k++) {
			mixed = (uint64_t)ctx[0] * p->mult[0];
			for (j = 1; j < k; j++)
				mixed ^= (uint64_t)ctx[j] * p->mult[j];
			for (g = 0; g < p->heads_per_ngram; g++) {
				h = (k - 2) * p->heads_per_ngram + g;
				rows[i * (size_t)p->n_heads + (size_t)h] =
				    (int64_t)(mixed % p->vocab[h] + p->offset[h]);
			}
		}
	}
}

/* Open path for direct reads, or return -1. */
static int
open_direct(const char *path)
{
#if defined(O_DIRECT)
	return open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
#elif defined(F_NOCACHE)
	int fd = open(path, O_RDONLY | O_CLOEXEC);

	if (fd >= 0 && fcntl(fd, F_NOCACHE, 1) == -1) {
		close(fd);
		return -1;
	}
	return fd;
#else
	(void)path;
	errno = EINVAL;
	return -1;
#endif
}

static int
grow_buf(struct ngram *r, size_t n)
{
	void *p;
	size_t a = r->align < 4096 ? 4096 : r->align;

	if (n <= r->bufsize)
		return 0;
	if (posix_memalign(&p, a, n) != 0)
		return -1;
	free(r->buf);
	r->buf = p;
	r->bufsize = n;
	return 0;
}

/* Read one aligned block of the table, to see that direct reads work. */
static int
probe_direct(struct ngram *r, size_t align)
{
	off_t off = r->base - r->base % (off_t)align;
	ssize_t k;

	if (grow_buf(r, align) != 0)
		return -1;
	k = pread(r->dfd, r->buf, align, off);
	return k > 0 ? 0 : -1;
}

/* Find a unit that direct reads accept: align, or 512 then 4096 for 0. */
static int
pick_align(struct ngram *r, size_t align)
{
	static const size_t units[] = { 512, 4096 };
	size_t i;

	if (align != 0) {
		r->align = align != 0 ? align : 4096;
		return probe_direct(r, align);
	}
	for (i = 0; i < sizeof(units) / sizeof(units[0]); i++)
		if (probe_direct(r, units[i]) == 0) {
			r->align = units[i];
			return 0;
		}
	r->align = 4096;
	return -1;
}

/*
 * Open the table tensor of the gguf file path. meta is the gguf file
 * with the <arch>.ple.* keys, or NULL for path. align is the direct I/O
 * unit, a power of two, or 0 to find one. Direct mode falls back to
 * pread when the file system refuses it. On failure, returns NULL with a message in err.
 */
struct ngram *
ngram_open(const char *path, const char *meta, const char *tensor, enum ngram_mode mode,
    size_t align, int threads, char *err, size_t errlen)
{
	struct gguf_init_params gp;
	struct gguf_context *c = NULL, *mc = NULL;
	struct ggml_context *gctx = NULL;
	struct ggml_tensor *t;
	struct ngram *r;
	struct stat sb;
	int64_t id;
	int h;

	if ((align & (align - 1)) != 0) {
		seterr(err, errlen, "align must be 0 or a power of two");
		return NULL;
	}
	if ((r = calloc(1, sizeof(*r))) == NULL) {
		seterr(err, errlen, "out of memory");
		return NULL;
	}
	r->fd = r->dfd = -1;
	r->align = align;
	r->threads = threads < 1 ? 1 : threads;
	r->mode = mode;

	gp.no_alloc = 1;
	gp.ctx = &gctx;
	if ((c = gguf_init_from_file(path, gp)) == NULL) {
		seterr(err, errlen, "%s: not a gguf file", path);
		goto fail;
	}
	if ((id = gguf_find_tensor(c, tensor)) < 0 || (t = ggml_get_tensor(gctx, tensor)) == NULL) {
		seterr(err, errlen, "%s: no tensor %s", path, tensor);
		goto fail;
	}
	if (t->ne[2] != 1 || t->ne[3] != 1 || ggml_get_type_traits(t->type)->to_float == NULL) {
		seterr(err, errlen, "%s: tensor %s is not a 2-d table of a known type", path,
		    tensor);
		goto fail;
	}
	r->type = t->type;
	r->head_dim = t->ne[0];
	r->nrows = t->ne[1];
	r->row_size = ggml_row_size(t->type, t->ne[0]);
	r->base = (off_t)(gguf_get_data_offset(c) + gguf_get_tensor_offset(c, id));

	if (meta != NULL && strcmp(meta, path) != 0) {
		gp.ctx = NULL;
		if ((mc = gguf_init_from_file(meta, gp)) == NULL) {
			seterr(err, errlen, "%s: not a gguf file", meta);
			goto fail;
		}
	}
	if (ngram_params_gguf(&r->p, mc != NULL ? mc : c, err, errlen) != 0)
		goto fail;
	for (h = 0; h < r->p.n_heads; h++)
		if ((int64_t)(r->p.offset[h] + r->p.vocab[h]) > r->nrows) {
			seterr(err, errlen, "%s: tensor %s has too few rows for the heads", path,
			    tensor);
			goto fail;
		}

	if ((r->fd = open(path, O_RDONLY | O_CLOEXEC)) == -1 || fstat(r->fd, &sb) == -1) {
		seterr(err, errlen, "%s: %s", path, strerror(errno));
		goto fail;
	}
	if ((off_t)(r->base + r->nrows * (off_t)r->row_size) > sb.st_size) {
		seterr(err, errlen, "%s: tensor %s runs past the end of the file", path, tensor);
		goto fail;
	}
#if defined(POSIX_FADV_RANDOM)
	posix_fadvise(r->fd, 0, 0, POSIX_FADV_RANDOM);
#endif
	if (mode == NGRAM_DIRECT) {
		r->dfd = open_direct(path);
		if (r->dfd == -1 || pick_align(r, align) != 0) {
			if (r->dfd != -1)
				close(r->dfd);
			r->dfd = -1;
			r->mode = NGRAM_PREAD;
			r->fell_back = 1;
		}
	} else if (mode == NGRAM_MMAP) {
		r->maplen = (size_t)sb.st_size;
		r->map = mmap(NULL, r->maplen, PROT_READ, MAP_SHARED, r->fd, 0);
		if (r->map == MAP_FAILED) {
			r->map = NULL;
			seterr(err, errlen, "%s: mmap: %s", path, strerror(errno));
			goto fail;
		}
#if defined(MADV_RANDOM)
		madvise(r->map, r->maplen, MADV_RANDOM);
#endif
	}
	gguf_free(c);
	if (mc != NULL)
		gguf_free(mc);
	ggml_free(gctx);
	return r;
fail:
	if (c != NULL)
		gguf_free(c);
	if (mc != NULL)
		gguf_free(mc);
	if (gctx != NULL)
		ggml_free(gctx);
	ngram_close(r);
	return NULL;
}

void
ngram_close(struct ngram *r)
{
	if (r == NULL)
		return;
	if (r->map != NULL)
		munmap(r->map, r->maplen);
	if (r->dfd != -1)
		close(r->dfd);
	if (r->fd != -1)
		close(r->fd);
	free(r->buf);
	free(r);
}

static int
pair_cmp(const void *a, const void *b)
{
	const struct pair *x = a, *y = b;

	if (x->row != y->row)
		return x->row < y->row ? -1 : 1;
	return x->i < y->i ? -1 : x->i > y->i;
}

static void *
read_spans(void *arg)
{
	struct job *j = arg;
	struct span *sp;
	size_t i, got;
	ssize_t k;

	for (i = j->from; i < j->to && j->err == 0; i++) {
		sp = &j->sp[i];
		/* Stop at need: a direct read past EOF leaves an unaligned rest. */
		for (got = 0; got < sp->need; got += (size_t)k) {
			k = pread(j->fd, j->r->buf + sp->boff + got, sp->len - got,
			    sp->off + (off_t)got);
			j->reads++;
			if (k == -1 && errno == EINTR) {
				k = 0;
				continue;
			}
			if (k <= 0) {
				j->err = k == 0 ? EIO : errno;
				break;
			}
		}
	}
	return NULL;
}

/* Read spans [0, n) with up to r->threads threads. Returns 0 or an errno. */
static int
read_all(struct ngram *r, int fd, struct span *sp, size_t n)
{
	struct job jobs[64];
	size_t nt, i, per;
	int err = 0;

	nt = (size_t)r->threads;
	if (nt > sizeof(jobs) / sizeof(jobs[0]))
		nt = sizeof(jobs) / sizeof(jobs[0]);
	/* Two spans or more for each thread. */
	if (nt > n / 2)
		nt = n / 2;
	if (nt < 1)
		nt = 1;
	per = (n + nt - 1) / nt;
	for (i = 0; i < nt; i++) {
		jobs[i].r = r;
		jobs[i].fd = fd;
		jobs[i].sp = sp;
		jobs[i].from = i * per < n ? i * per : n;
		jobs[i].to = (i + 1) * per < n ? (i + 1) * per : n;
		jobs[i].reads = 0;
		jobs[i].err = 0;
	}
	for (i = 1; i < nt; i++)
		if (pthread_create(&jobs[i].tid, NULL, read_spans, &jobs[i]) != 0) {
			/* Run it on this thread instead. */
			jobs[i].tid = pthread_self();
			read_spans(&jobs[i]);
		}
	read_spans(&jobs[0]);
	for (i = 0; i < nt; i++) {
		if (i > 0 && !pthread_equal(jobs[i].tid, pthread_self()))
			pthread_join(jobs[i].tid, NULL);
		r->st.reads += jobs[i].reads;
		if (err == 0)
			err = jobs[i].err;
	}
	return err;
}

/*
 * Dequantize the unique rows u[0..n) into out at the first caller index
 * of each, through the read buffer. pos[k] is the buffer offset of row k.
 */
static void
dequant(struct ngram *r, const char *src0, const struct pair *u, const size_t *pos,
    size_t n, float *out)
{
	ggml_to_float_t to_float = ggml_get_type_traits(r->type)->to_float;
	size_t k;

	for (k = 0; k < n; k++)
		to_float(src0 + pos[k], out + u[k].i * (size_t)r->head_dim, r->head_dim);
}

/* Read and dequantize the unique rows u[0..n). */
static int
read_chunk(struct ngram *r, const struct pair *u, size_t n, float *out, struct span *sp,
    size_t *pos, size_t *spi)
{
	size_t a, k, ns, total;
	off_t s, e, as, ae;
	double t0;
	int fd, err;

	if (r->mode == NGRAM_MMAP) {
		for (k = 0; k < n; k++)
			pos[k] = (size_t)(r->base + u[k].row * (off_t)r->row_size);
		dequant(r, r->map, u, pos, n, out);
		r->st.bytes += n * r->row_size;
		return 0;
	}
retry:
	a = r->mode == NGRAM_DIRECT ? r->align : 1;
	fd = r->mode == NGRAM_DIRECT ? r->dfd : r->fd;
	ns = 0;
	for (k = 0; k < n; k++) {
		s = r->base + u[k].row * (off_t)r->row_size;
		e = s + (off_t)r->row_size;
		as = s - s % (off_t)a;
		ae = (e + (off_t)a - 1) / (off_t)a * (off_t)a;
		if (ns > 0 && as <= sp[ns - 1].off + (off_t)sp[ns - 1].len &&
		    (size_t)(ae - sp[ns - 1].off) <= SPAN_MAX) {
			if ((size_t)(ae - sp[ns - 1].off) > sp[ns - 1].len)
				sp[ns - 1].len = (size_t)(ae - sp[ns - 1].off);
		} else {
			sp[ns].off = as;
			sp[ns].len = (size_t)(ae - as);
			ns++;
		}
		sp[ns - 1].need = (size_t)(e - sp[ns - 1].off);
		spi[k] = ns - 1;
		pos[k] = (size_t)(s - sp[ns - 1].off);
	}
	total = 0;
	for (k = 0; k < ns; k++) {
		sp[k].boff = total;
		total += sp[k].len;
	}
	if (grow_buf(r, total) != 0)
		return ENOMEM;
	t0 = now();
	err = read_all(r, fd, sp, ns);
	r->st.read_sec += now() - t0;
	if (err == EINVAL && r->mode == NGRAM_DIRECT) {
		r->mode = NGRAM_PREAD;
		r->fell_back = 1;
		goto retry;
	}
	if (err != 0)
		return err;
	r->st.bytes += total;
	for (k = 0; k < n; k++)
		pos[k] += sp[spi[k]].boff;
	dequant(r, r->buf, u, pos, n, out);
	return 0;
}

/*
 * Dequantize table rows rows[0..n) to f32, row i at out + i * head_dim.
 * Rows must be in [0, nrows). Returns 0, or -1 with errno set.
 */
int
ngram_read(struct ngram *r, const int64_t *rows, size_t n, float *out)
{
	struct pair *pr, *u;
	struct span *sp;
	size_t *pos, *spi, i, k, nu, from, cap, per;
	double t0 = now();
	int err = 0;

	if (n == 0)
		return 0;
	for (i = 0; i < n; i++)
		if (rows[i] < 0 || rows[i] >= r->nrows) {
			errno = ERANGE;
			return -1;
		}
	pr = malloc(n * sizeof(*pr));
	u = malloc(n * sizeof(*u));
	sp = malloc(n * sizeof(*sp));
	pos = malloc(n * sizeof(*pos));
	spi = malloc(n * sizeof(*spi));
	if (pr == NULL || u == NULL || sp == NULL || pos == NULL || spi == NULL) {
		err = ENOMEM;
		goto done;
	}
	for (i = 0; i < n; i++) {
		pr[i].row = rows[i];
		pr[i].i = i;
	}
	qsort(pr, n, sizeof(*pr), pair_cmp);
	/* u gets each row once, with its first caller index. */
	for (i = nu = 0; i < n; i++)
		if (i == 0 || pr[i].row != pr[i - 1].row)
			u[nu++] = pr[i];
	/* A row takes at most two aligned blocks of the read buffer. */
	per = r->row_size + 2 * r->align;
	cap = BUF_MAX / per > 0 ? BUF_MAX / per : 1;
	for (from = 0; from < nu && err == 0; from += k) {
		k = nu - from < cap ? nu - from : cap;
		err = read_chunk(r, u + from, k, out, sp, pos, spi);
	}
	/* Copy each row to the later indices that ask for it. */
	for (i = 1; i < n && err == 0; i++)
		if (pr[i].row == pr[i - 1].row)
			memcpy(out + pr[i].i * (size_t)r->head_dim,
			    out + pr[i - 1].i * (size_t)r->head_dim,
			    (size_t)r->head_dim * sizeof(float));
	r->st.calls++;
	r->st.rows += n;
	r->st.unique += nu;
done:
	r->st.total_sec += now() - t0;
	free(pr);
	free(u);
	free(sp);
	free(pos);
	free(spi);
	if (err != 0) {
		errno = err;
		return -1;
	}
	return 0;
}
