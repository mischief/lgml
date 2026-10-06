/*** Tensor graphs on ggml backends.
 * @module lgml
 */

/*
 * Read the tensor data of a gguf file into allocated tensors. The ring
 * path reads aligned spans with O_DIRECT through an io_uring into a few
 * buffers, and copies tensors out of one buffer while the reads into
 * the others run. A Vulkan device imports the buffers and copies its
 * tensors from them itself. Without a ring, it reads with stdio.
 */

/* O_DIRECT and posix_memalign. */
#define _GNU_SOURCE

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>

#if defined(LGML_HAVE_URING)
#include <liburing.h>
#endif

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

#define SLOT_SIZE	((size_t)16 << 20)	/* bytes of one read buffer */
#define NSLOTS		4		/* read buffers */
#define MERGE_GAP	((uint64_t)1 << 20)	/* read through skipped bytes up to this */
#define DIO_ALIGN	4096		/* direct read unit */

struct piece {
	struct ggml_tensor	*t;
	uint64_t		 off;		/* file offset */
	uint64_t		 size;
	int			 dma;		/* the device copies it from hb */
};

/* The read buffers imported into a device, for copies without the CPU. */
struct dma {
	ggml_backend_buffer_t	 hb;
	const char		*buf;		/* the read buffers */
	struct ggml_context	*ctx;		/* room for two tensors */
};

struct chunk {
	uint64_t		 off;		/* file offset, aligned */
	size_t			 len;		/* bytes to read, aligned */
	size_t			 need;		/* bytes that must arrive */
	size_t			 got;
};

struct load_info {
	const char		*mode;		/* "direct", "buffered" or "stdio" */
	uint64_t		 bytes;		/* bytes read */
	uint64_t		 reads;
	uint64_t		 dma;		/* bytes the device copied from the buffers */
	double			 secs;
	double			 wait;		/* seconds spent waiting for reads */
};

static _Thread_local struct load_info last;

static double
now(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int
piece_cmp(const void *a, const void *b)
{
	const struct piece *x = a, *y = b;

	return x->off < y->off ? -1 : x->off > y->off;
}

/* The tensors of ctxs with data, sorted by file offset, or NULL. */
static struct piece *
collect(const struct gguf_context *gctx, struct ggml_context *const *ctxs, int nctx,
    size_t *np)
{
	struct ggml_tensor *gt;
	struct piece *pc;
	uint64_t base;
	size_t n = 0;
	int i;

	for (i = 0; i < nctx; i++)
		for (gt = ggml_get_first_tensor(ctxs[i]); gt != NULL;
		    gt = ggml_get_next_tensor(ctxs[i], gt))
			n++;
	if ((pc = calloc(n + 1, sizeof(*pc))) == NULL)
		return NULL;
	base = gguf_get_data_offset(gctx);
	n = 0;
	for (i = 0; i < nctx; i++)
		for (gt = ggml_get_first_tensor(ctxs[i]); gt != NULL;
		    gt = ggml_get_next_tensor(ctxs[i], gt)) {
			if (ggml_nbytes(gt) == 0)
				continue;
			pc[n].t = gt;
			pc[n].off = base + gguf_get_tensor_offset(gctx,
			    gguf_find_tensor(gctx, ggml_get_name(gt)));
			pc[n].size = ggml_nbytes(gt);
			n++;
		}
	qsort(pc, n, sizeof(*pc), piece_cmp);
	*np = n;
	return pc;
}

static int
read_stdio(const char *path, const struct piece *pc, size_t np)
{
	size_t i, n, done;
	char *chunk;
	FILE *f;
	int e = 0;

	if ((f = fopen(path, "rb")) == NULL)
		return errno;
	if ((chunk = malloc(LOAD_CHUNK)) == NULL) {
		fclose(f);
		return ENOMEM;
	}
	for (i = 0; i < np && e == 0; i++) {
		if (fseeko(f, (off_t)pc[i].off, SEEK_SET) != 0) {
			e = EIO;
			break;
		}
		for (done = 0; done < pc[i].size; done += n) {
			n = pc[i].size - done < LOAD_CHUNK ? pc[i].size - done : LOAD_CHUNK;
			if (fread(chunk, 1, n, f) != n) {
				e = EIO;
				break;
			}
			ggml_backend_tensor_set(pc[i].t, chunk, done, n);
			last.bytes += n;
			last.reads++;
		}
	}
	free(chunk);
	fclose(f);
	return e;
}

#if defined(LGML_HAVE_URING)
/*
 * Cover the pieces with aligned chunks of up to SLOT_SIZE bytes. Close
 * pieces share a span; a skipped gap of over MERGE_GAP starts a new one.
 */
static struct chunk *
make_chunks(const struct piece *pc, size_t np, uint64_t fsize, size_t *ncp)
{
	struct chunk *c = NULL, *t;
	size_t n = 0, cap = 0, i = 0;
	uint64_t a, b, o, prev = 0;

	while (i < np) {
		a = pc[i].off;
		b = pc[i].off + pc[i].size;
		for (i++; i < np && pc[i].off <= b + MERGE_GAP; i++)
			if (pc[i].off + pc[i].size > b)
				b = pc[i].off + pc[i].size;
		a -= a % DIO_ALIGN;
		if (a < prev)
			a = prev;
		b = (b + DIO_ALIGN - 1) / DIO_ALIGN * DIO_ALIGN;
		for (o = a; o < b; o += SLOT_SIZE) {
			if (n == cap) {
				cap = cap == 0 ? 64 : cap * 2;
				if ((t = reallocarray(c, cap, sizeof(*c))) == NULL) {
					free(c);
					return NULL;
				}
				c = t;
			}
			c[n].off = o;
			c[n].len = b - o < SLOT_SIZE ? b - o : SLOT_SIZE;
			c[n].need = fsize - o < c[n].len ? fsize - o : c[n].len;
			c[n].got = 0;
			n++;
		}
		prev = b;
	}
	*ncp = n;
	return c;
}

/* Copy n bytes at src in d->hb to offset off of t. */
static void
dma_copy(struct dma *d, struct ggml_tensor *t, const char *src, size_t off, size_t n)
{
	struct ggml_tensor *a, *b;

	ggml_reset(d->ctx);
	a = ggml_new_tensor_1d(d->ctx, GGML_TYPE_I8, (int64_t)n);
	b = ggml_new_tensor_1d(d->ctx, GGML_TYPE_I8, (int64_t)n);
	/* Tensor addresses in a Vulkan buffer are offsets from its base. */
	ggml_backend_tensor_alloc(d->hb, a, (char *)ggml_backend_buffer_get_base(d->hb) +
	    (src - d->buf));
	ggml_backend_tensor_alloc(t->buffer, b, (char *)t->data + off);
	ggml_backend_tensor_copy(a, b);
	last.dma += n;
}

/*
 * Import buf into the Vulkan device of the first device tensor,
 * and mark the pieces on that device. Without one, d->hb stays NULL.
 */
static void
dma_init(struct dma *d, char *buf, size_t size, struct piece *pc, size_t np)
{
	struct ggml_init_params ip = { 0, NULL, true };
	ggml_backend_dev_t dev = NULL, td;
	ggml_backend_buffer_t b;
	size_t i, n = 0;

	memset(d, 0, sizeof(*d));
	d->buf = buf;
	for (i = 0; i < np; i++) {
		b = pc[i].t->buffer;
		if (b == NULL || ggml_backend_buffer_is_host(b))
			continue;
		td = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(b));
		if (dev == NULL)
			dev = td;
		pc[i].dma = td != NULL && td == dev;
		n += pc[i].dma;
	}
	if (n == 0 || strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "Vulkan") != 0)
		goto none;
	ip.mem_size = 2 * ggml_tensor_overhead();
	if ((d->ctx = ggml_init(ip)) == NULL)
		goto none;
	if ((d->hb = ggml_backend_dev_buffer_from_host_ptr(dev, buf, size, size)) == NULL)
		goto none;
	return;
none:
	if (d->ctx != NULL)
		ggml_free(d->ctx);
	d->ctx = NULL;
	for (i = 0; i < np; i++)
		pc[i].dma = 0;
}

static void
dma_free(struct dma *d)
{
	if (d->hb != NULL)
		ggml_backend_buffer_free(d->hb);
	if (d->ctx != NULL)
		ggml_free(d->ctx);
}

/* Copy the parts of pieces from *cur on that chunk c holds. */
static void
consume(struct dma *d, const struct chunk *c, const char *buf, const struct piece *pc,
    size_t np, size_t *cur)
{
	uint64_t end = c->off + c->len, lo, hi;
	size_t p;

	for (p = *cur; p < np && pc[p].off < end; p++) {
		lo = pc[p].off > c->off ? pc[p].off : c->off;
		hi = pc[p].off + pc[p].size < end ? pc[p].off + pc[p].size : end;
		if (hi <= lo)
			continue;
		if (pc[p].dma)
			dma_copy(d, pc[p].t, buf + (lo - c->off), lo - pc[p].off, hi - lo);
		else
			ggml_backend_tensor_set(pc[p].t, buf + (lo - c->off), lo - pc[p].off,
			    hi - lo);
	}
	while (*cur < np && pc[*cur].off + pc[*cur].size <= end)
		(*cur)++;
}

static void
prep(struct io_uring *ring, int fd, int fixed, struct chunk *c, size_t i, char *buf)
{
	struct io_uring_sqe *sqe = io_uring_get_sqe(ring);

	/* Never NULL: the ring holds as many entries as reads in flight. */
	io_uring_prep_read(sqe, fd, buf + c[i].got, (unsigned)(c[i].len - c[i].got),
	    c[i].off + c[i].got);
	if (fixed)
		sqe->flags |= IOSQE_FIXED_FILE;
	io_uring_sqe_set_data64(sqe, i);
	last.reads++;
}

/*
 * Read the pieces from fd through a ring, NSLOTS chunks in flight, and
 * copy each chunk out as it arrives. Returns 0, an errno, or -1 when no
 * ring works.
 */
static int
read_ring(int fd, struct piece *pc, size_t np, uint64_t fsize)
{
	static const unsigned flags[] = {
		IORING_SETUP_CLAMP | IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN,
		IORING_SETUP_CLAMP
	};
	struct io_uring_params p;
	struct io_uring ring;
	struct io_uring_cqe *cqe;
	struct chunk *c;
	struct dma d;
	size_t nc = 0, next = 0, done = 0, fly = 0, cur = 0, i;
	unsigned head, seen;
	int err = 0, ret, res, fixed, rfd;
	char *buf;
	double t;

	if ((c = make_chunks(pc, np, fsize, &nc)) == NULL && np > 0)
		return ENOMEM;
	if (posix_memalign((void **)&buf, DIO_ALIGN, NSLOTS * SLOT_SIZE) != 0) {
		free(c);
		return ENOMEM;
	}
	for (i = 0; i < sizeof(flags) / sizeof(flags[0]); i++) {
		memset(&p, 0, sizeof(p));
		p.flags = flags[i];
		if (io_uring_queue_init_params(NSLOTS, &ring, &p) == 0)
			break;
	}
	if (i == sizeof(flags) / sizeof(flags[0])) {
		free(buf);
		free(c);
		return -1;
	}
	dma_init(&d, buf, NSLOTS * SLOT_SIZE, pc, np);
	fixed = io_uring_register_files(&ring, &fd, 1) == 0;
	rfd = fixed ? 0 : fd;
	while (done < nc && err == 0) {
		for (; next < nc && next - done < NSLOTS; next++, fly++)
			prep(&ring, rfd, fixed, c, next, buf + next % NSLOTS * SLOT_SIZE);
		t = now();
		while (err == 0 && c[done].got < c[done].need) {
			ret = io_uring_submit_and_wait(&ring, 1);
			if (ret == -EINTR || ret == -EAGAIN || ret == -EBUSY)
				continue;
			if (ret < 0) {
				err = -1;
				break;
			}
			seen = 0;
			io_uring_for_each_cqe(&ring, head, cqe) {
				seen++;
				fly--;
				i = (size_t)io_uring_cqe_get_data64(cqe);
				res = cqe->res;
				if (res > 0) {
					c[i].got += (size_t)res;
					last.bytes += (uint64_t)res;
				}
				if (err == 0 && (res == -EINTR || res == -EAGAIN ||
				    (res > 0 && c[i].got < c[i].need))) {
					/* Read the rest: the completed read freed an entry. */
					prep(&ring, rfd, fixed, c, i, buf + i % NSLOTS * SLOT_SIZE);
					fly++;
					continue;
				}
				if (res <= 0 && err == 0)
					err = res == 0 ? EIO : -res;
			}
			io_uring_cq_advance(&ring, seen);
		}
		last.wait += now() - t;
		if (err == 0)
			consume(&d, &c[done], buf + done % NSLOTS * SLOT_SIZE, pc, np, &cur);
		done++;
	}
	/* Wait for the reads in flight: they write to buf. */
	if (fly > 0)
		io_uring_submit(&ring);
	fly -= io_uring_sq_ready(&ring);
	while (fly > 0 && io_uring_wait_cqe(&ring, &cqe) == 0) {
		io_uring_cqe_seen(&ring, cqe);
		fly--;
	}
	io_uring_queue_exit(&ring);
	dma_free(&d);
	free(buf);
	free(c);
	return err;
}

/* Open path for direct reads and check that one works, or return -1. */
static int
open_direct(const char *path)
{
	void *buf;
	int fd;

	if ((fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC)) == -1)
		return -1;
	if (posix_memalign(&buf, DIO_ALIGN, DIO_ALIGN) != 0) {
		close(fd);
		return -1;
	}
	if (pread(fd, buf, DIO_ALIGN, 0) <= 0) {
		close(fd);
		fd = -1;
	}
	free(buf);
	return fd;
}
#endif

/*
 * Read the data of each tensor of ctxs[0..nctx) from the gguf file
 * path. uring false, or LGML_LOAD_URING=0, reads with stdio. Returns 0
 * or an errno.
 */
int
lgml_gguf_read(const char *path, const struct gguf_context *gctx,
    struct ggml_context *const *ctxs, int nctx, int uring)
{
	const char *env = getenv("LGML_LOAD_URING");
	struct piece *pc;
	struct stat sb;
	size_t np, i;
	double t0 = now();
	int e, fd;

	memset(&last, 0, sizeof(last));
	if ((pc = collect(gctx, ctxs, nctx, &np)) == NULL)
		return ENOMEM;
	if (env != NULL && strcmp(env, "0") == 0)
		uring = 0;
#if defined(LGML_HAVE_URING)
	e = -1;
	if (uring) {
		last.mode = "direct";
		if ((fd = open_direct(path)) == -1) {
			last.mode = "buffered";
			fd = open(path, O_RDONLY | O_CLOEXEC);
		}
		if (fd != -1 && fstat(fd, &sb) == 0) {
			for (i = 0; i < np; i++)
				if (pc[i].off + pc[i].size > (uint64_t)sb.st_size)
					break;
			e = i < np ? EIO : read_ring(fd, pc, np, (uint64_t)sb.st_size);
		}
		if (fd != -1)
			close(fd);
	}
	if (e != 0) {
		memset(&last, 0, sizeof(last));
		last.mode = "stdio";
		e = read_stdio(path, pc, np);
	}
#else
	(void)uring;
	(void)sb;
	(void)i;
	(void)fd;
	last.mode = "stdio";
	e = read_stdio(path, pc, np);
#endif
	free(pc);
	last.secs = now() - t0;
	return e;
}

/*** How the last `backend:load` on this thread read its file.
 * @treturn table `mode` ("direct", "buffered" or "stdio"), `bytes`,
 * `reads`, `dma` (bytes that the device copied from the read buffers),
 * `seconds`, and `wait`, the seconds spent waiting for reads
 * @function load_info
 */
static int
l_load_info(lua_State *L)
{
	lua_createtable(L, 0, 6);
	lua_pushstring(L, last.mode != NULL ? last.mode : "none");
	lua_setfield(L, -2, "mode");
	lua_pushinteger(L, (lua_Integer)last.bytes);
	lua_setfield(L, -2, "bytes");
	lua_pushinteger(L, (lua_Integer)last.reads);
	lua_setfield(L, -2, "reads");
	lua_pushinteger(L, (lua_Integer)last.dma);
	lua_setfield(L, -2, "dma");
	lua_pushnumber(L, last.secs);
	lua_setfield(L, -2, "seconds");
	lua_pushnumber(L, last.wait);
	lua_setfield(L, -2, "wait");
	return 1;
}

void
lgml_open_gload(lua_State *L)
{
	lua_pushcfunction(L, l_load_info);
	lua_setfield(L, -2, "load_info");
}
