/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * lgml: lua 5.4 binding for ggml. A store owns device storage for leaf
 * tensors. Graph nodes get storage at compute time. Children count
 * references to parents, so close order does not matter. Shapes and
 * dimension numbers use torch order, outermost first, from 1.
 */

/* fseeko and off_t. */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <gguf.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

static int log_level = GGML_LOG_LEVEL_WARN;

static void
log_cb(enum ggml_log_level level, const char *text, void *arg)
{
	static enum ggml_log_level last = GGML_LOG_LEVEL_NONE;

	(void)arg;
	if (level != GGML_LOG_LEVEL_CONT)
		last = level;
	if ((int)last >= log_level)
		fputs(text, stderr);
}

static void
backend_release(struct backend *b)
{
	struct backend *cpu;

	if (b->closed && b->refs == 0 && b->be != NULL) {
		ggml_backend_free(b->be);
		b->be = NULL;
		if ((cpu = b->cpu) != NULL) {
			b->cpu = NULL;
			cpu->refs--;
			backend_release(cpu);
		}
	}
}

static void
store_release(struct store *s)
{
	if (!s->closed || s->refs != 0 || s->ctx == NULL)
		return;
	lgml_store_unaccount(s);
	if (s->buf != NULL)
		ggml_backend_buffer_free(s->buf);
	ggml_free(s->ctx);
	s->buf = NULL;
	s->ctx = NULL;
	s->b->refs--;
	backend_release(s->b);
}

static void
graph_release(struct graph *g)
{
	int i;

	if (!g->closed || g->refs != 0 || g->b == NULL)
		return;
	for (i = 0; i < 2; i++)
		if (g->mem_bytes[i] > 0)
			lgml_mem_add(g->mem_key[i], MEM_COMPUTE, 0, g->mem_bytes[i]);
	if (g->galloc != NULL)
		ggml_gallocr_free(g->galloc);
	if (g->sched != NULL)
		ggml_backend_sched_free(g->sched);
	free(g->saved);
	g->sched = NULL;
	g->saved = NULL;
	g->nsaved = 0;
	for (i = 0; i < g->nctx; i++)
		ggml_free(g->ctxs[i]);
	free(g->ctxs);
	g->galloc = NULL;
	g->ctxs = NULL;
	g->nctx = 0;
	g->ctx = NULL;
	g->gf = NULL;
	g->b->refs--;
	backend_release(g->b);
	g->b = NULL;
}

/* Extra tensor slots in each context, beyond what reserve counts. */
#define CTX_SLACK	16

/* Add a context of mem bytes, without data, to the graph. */
static struct ggml_context *
graph_ctx(lua_State *L, struct graph *g, size_t mem)
{
	struct ggml_init_params ip = { mem, NULL, true };
	struct ggml_context **v, *c;
	int cap;

	if (g->nctx == g->ctxcap) {
		cap = g->ctxcap > 0 ? 2 * g->ctxcap : 8;
		v = realloc(g->ctxs, (size_t)cap * sizeof(*v));
		if (v == NULL)
			luaL_error(L, "out of memory");
		g->ctxs = v;
		g->ctxcap = cap;
	}
	c = ggml_init(ip);
	if (c == NULL)
		luaL_error(L, "ggml_init failed");
	g->ctxs[g->nctx++] = c;
	return c;
}

/* Make an empty cgraph of size slots in its own context. */
struct ggml_cgraph *
lgml_new_cgraph(lua_State *L, struct graph *g, int size, int grads)
{
	struct ggml_context *c;

	c = graph_ctx(L, g, ggml_graph_overhead_custom((size_t)size, grads != 0));
	return ggml_new_graph_custom(c, (size_t)size, grads != 0);
}

/*
 * Make room for n more tensors: in the tensor context, and as nodes or
 * leaves of the cgraph. The cgraph grows to at least twice its size.
 */
void
lgml_graph_room(lua_State *L, struct graph *g, int n)
{
	struct ggml_cgraph *gf;
	int need, size, cap;

	if (n < 0 || n > GRAPH_MAX || g->ntensors + g->nleaves > GRAPH_MAX - n)
		luaL_error(L, "graph full (%d tensors)", GRAPH_MAX);
	need = g->ntensors + g->nleaves + n;
	if (n > g->ctx_left) {
		cap = n > g->ntensors ? n : g->ntensors;
		cap = cap > 256 ? cap : 256;
		g->ctx = graph_ctx(L, g, (size_t)(cap + CTX_SLACK) * ggml_tensor_overhead());
		g->ctx_left = cap;
	}
	if (need > g->size) {
		size = g->size < GRAPH_MAX / 2 ? 2 * g->size : GRAPH_MAX;
		size = size > need ? size : need;
		gf = lgml_new_cgraph(L, g, size, g->grads);
		ggml_graph_cpy(g->gf, gf);
		g->gf = gf;
		g->size = size;
	}
}

static struct backend *
check_backend(lua_State *L, int idx)
{
	struct backend *b = luaL_checkudata(L, idx, BACKEND_MT);

	if (b->closed)
		luaL_error(L, "backend closed");
	return b;
}

static struct store *
check_store(lua_State *L, int idx)
{
	struct store *s = luaL_checkudata(L, idx, STORE_MT);

	if (s->closed)
		luaL_error(L, "weights closed");
	return s;
}

static struct graph *
check_graph(lua_State *L, int idx)
{
	struct graph *g = luaL_checkudata(L, idx, GRAPH_MT);

	if (g->closed)
		luaL_error(L, "graph closed");
	return g;
}

static struct tensor *
check_tensor(lua_State *L, int idx)
{
	struct tensor *t = luaL_checkudata(L, idx, TENSOR_MT);

	if (t->t == NULL)
		luaL_error(L, "tensor closed");
	return t;
}

static struct hostbuf *
check_hostbuf(lua_State *L, int idx)
{
	struct hostbuf *hb = luaL_checkudata(L, idx, HOSTBUF_MT);

	if (hb->p == NULL)
		luaL_error(L, "hostbuf closed");
	return hb;
}

static enum ggml_type
check_type(lua_State *L, int idx)
{
	const char *name = luaL_checkstring(L, idx);
	const char *tn;
	int i;

	for (i = 0; i < GGML_TYPE_COUNT; i++) {
		tn = ggml_type_name((enum ggml_type)i);
		if (tn != NULL && strcmp(tn, name) == 0)
			return (enum ggml_type)i;
	}
	luaL_error(L, "unknown type '%s'", name);
	return GGML_TYPE_COUNT;
}

/* Read a torch-order shape table into ggml ne[]. Return the rank. */
static int
check_shape(lua_State *L, int idx, int64_t ne[GGML_MAX_DIMS])
{
	lua_Integer d;
	int i, n;

	luaL_checktype(L, idx, LUA_TTABLE);
	n = (int)luaL_len(L, idx);
	luaL_argcheck(L, n >= 1 && n <= GGML_MAX_DIMS, idx, "rank must be 1 to 4");
	for (i = 0; i < GGML_MAX_DIMS; i++)
		ne[i] = 1;
	for (i = 0; i < n; i++) {
		lua_geti(L, idx, i + 1);
		d = luaL_checkinteger(L, -1);
		lua_pop(L, 1);
		luaL_argcheck(L, d >= 1, idx, "dimension must be positive");
		ne[n - 1 - i] = d;
	}
	return n;
}

/* Lowest rank that holds all non-1 dimensions of t. */
static int
min_rank(const struct ggml_tensor *t)
{
	int n;

	for (n = GGML_MAX_DIMS; n > 1; n--)
		if (t->ne[n - 1] != 1)
			break;
	return n;
}

static int
max_int(int a, int b)
{
	return a > b ? a : b;
}

struct backend *
lgml_check_backend(lua_State *L, int idx)
{
	return check_backend(L, idx);
}

struct tensor *
lgml_check_tensor(lua_State *L, int idx)
{
	return check_tensor(L, idx);
}

/* backend */

/* The most threads of the default count. */
#define THREADS_DEFAULT_MAX	64
/* The most threads that a caller or LGML_THREADS can ask for. */
#define THREADS_MAX		1024

/*
 * Count the physical cores in Linux sysfs: a CPU that is first in its
 * sibling list starts a core. Return 0 when the files are not there.
 */
static int
physical_cores(long ncpu)
{
	char path[96];
	FILE *f;
	long cpu;
	int first, n = 0;

	for (cpu = 0; cpu < ncpu; cpu++) {
		snprintf(path, sizeof(path),
		    "/sys/devices/system/cpu/cpu%ld/topology/thread_siblings_list", cpu);
		if ((f = fopen(path, "r")) == NULL)
			return 0;
		if (fscanf(f, "%d", &first) != 1)
			first = -1;
		fclose(f);
		if (first == cpu)
			n++;
	}
	return n;
}

/* Parse a thread count from 1 to THREADS_MAX. Return 0 when s is bad. */
static int
parse_threads(const char *s)
{
	char *end;
	long v;

	if (s == NULL || *s == '\0')
		return 0;
	errno = 0;
	v = strtol(s, &end, 10);
	if (errno != 0 || *end != '\0' || v < 1 || v > THREADS_MAX)
		return 0;
	return (int)v;
}

/*
 * The default thread count: LGML_THREADS when set, else the physical
 * cores, else the online CPUs, at most THREADS_DEFAULT_MAX.
 */
static int
default_threads(void)
{
	static int n;
	long ncpu;

	if (n > 0)
		return n;
	if ((n = parse_threads(getenv("LGML_THREADS"))) > 0)
		return n;
	ncpu = sysconf(_SC_NPROCESSORS_ONLN);
	if (ncpu < 1)
		ncpu = 1;
	n = physical_cores(ncpu);
	if (n < 1 || n > ncpu)
		n = (int)(ncpu < THREADS_DEFAULT_MAX ? ncpu : THREADS_DEFAULT_MAX);
	if (n > THREADS_DEFAULT_MAX)
		n = THREADS_DEFAULT_MAX;
	return n;
}

/*
 * Set the thread count of a backend that runs ops on CPU threads. Return
 * n, or 0 when the backend has no thread count.
 */
static int
set_threads(ggml_backend_t be, int n)
{
	ggml_backend_dev_t dev = ggml_backend_get_device(be);
	ggml_backend_reg_t reg = dev != NULL ? ggml_backend_dev_backend_reg(dev) : NULL;
	ggml_backend_set_n_threads_t fn;
	void *p = NULL;

	if (reg != NULL)
		p = ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
	if (p == NULL)
		return 0;
	/* ISO C has no cast from an object pointer to a function pointer. */
	memcpy(&fn, &p, sizeof(fn));
	fn(be, n);
	return n;
}

/* Check a thread count argument. */
static int
check_threads(lua_State *L, int idx, const char *what)
{
	lua_Integer n = luaL_checkinteger(L, idx);

	if (n < 1 || n > THREADS_MAX)
		return luaL_error(L, "%s must be 1 to %d", what, THREADS_MAX);
	return (int)n;
}

/*** Get the default CPU thread count of new backends.
 * It is the LGML_THREADS environment variable when set, else the count of
 * physical cores, else the count of online CPUs, at most 64.
 * @treturn integer the thread count
 * @function threads
 */
static int
l_threads(lua_State *L)
{
	lua_pushinteger(L, default_threads());
	return 1;
}

/* Push a backend userdata that owns be. */
static struct backend *
push_backend(lua_State *L, ggml_backend_t be, int threads)
{
	struct backend *b;

	b = lua_newuserdatauv(L, sizeof(*b), 1);
	b->be = be;
	b->cpu = NULL;
	b->threads = set_threads(be, threads);
	b->refs = 0;
	b->closed = 0;
	luaL_setmetatable(L, BACKEND_MT);
	return b;
}

/*** Open a backend.
 * Without a name, open the best backend on this machine. A backend that
 * runs ops on CPU threads, such as "CPU", uses `threads` of them.
 * With `cpu`, the backend also opens a CPU backend (see `backend:cpu`).
 * A graph that uses tensors on both runs through a ggml scheduler: each
 * op runs where its weights are, and the scheduler copies activations
 * between the two.
 * @tparam[opt] string name ggml device name, such as "CPU"
 * @tparam[opt] table opts `threads`: CPU thread count (default:
 * `lgml.threads()`); `cpu`: true to open a CPU backend too
 * @treturn[1] backend the backend
 * @return[2] nil
 * @treturn[2] string error message
 * @raise if threads is not 1 to 1024
 * @function backend
 */
static int
l_backend(lua_State *L)
{
	const char *name = luaL_optstring(L, 1, NULL);
	struct backend *b;
	ggml_backend_t be, cbe = NULL;
	int threads = 0, cpu = 0;

	if (!lua_isnoneornil(L, 2)) {
		luaL_checktype(L, 2, LUA_TTABLE);
		if (lua_getfield(L, 2, "threads") != LUA_TNIL)
			threads = check_threads(L, -1, "threads");
		lua_pop(L, 1);
		lua_getfield(L, 2, "cpu");
		cpu = lua_toboolean(L, -1);
		lua_pop(L, 1);
	}
	if (threads == 0)
		threads = default_threads();
	be = name != NULL ? ggml_backend_init_by_name(name, NULL) : ggml_backend_init_best();
	if (be == NULL) {
		luaL_pushfail(L);
		lua_pushfstring(L, "no backend '%s'", name != NULL ? name : "best");
		return 2;
	}
	if (cpu && (cbe = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL)) == NULL) {
		ggml_backend_free(be);
		luaL_pushfail(L);
		lua_pushliteral(L, "no CPU backend");
		return 2;
	}
	b = push_backend(L, be, threads);
	if (cbe != NULL) {
		b->cpu = push_backend(L, cbe, threads);
		b->cpu->refs++;
		lua_setiuservalue(L, -2, UV_CPU);
	}
	return 1;
}

/*** A compute device.
 * Weights, leaf tensors and graphs keep their backend alive, so close
 * order does not matter.
 * @type backend
 */

/*** Close the backend.
 * The device stays open until all weights, tensors and graphs on it close.
 * The `__gc` and `__close` metamethods call this function.
 * @function backend:close
 */
static int
backend_close(lua_State *L)
{
	struct backend *b = luaL_checkudata(L, 1, BACKEND_MT);

	b->closed = 1;
	backend_release(b);
	return 0;
}

/*** Get the name of the backend.
 * @treturn string ggml backend name
 * @raise if the backend is closed
 * @function backend:name
 */
static int
backend_name(lua_State *L)
{
	struct backend *b = check_backend(L, 1);

	lua_pushstring(L, ggml_backend_name(b->be));
	return 1;
}

/*** Get or set the CPU thread count of the backend.
 * Graphs that run after a set use the new count.
 * A set also applies to the CPU backend of `backend:cpu`, and a get
 * without CPU threads here gives the count of that CPU backend.
 * @tparam[opt] integer n new thread count, 1 to 1024
 * @treturn integer|nil the thread count, or nil when the backend does
 * not run ops on CPU threads
 * @raise if the backend is closed, or n is out of range
 * @function backend:threads
 */
static int
backend_threads(lua_State *L)
{
	struct backend *b = check_backend(L, 1);
	int n;

	if (!lua_isnoneornil(L, 2)) {
		n = check_threads(L, 2, "n");
		b->threads = set_threads(b->be, n);
		if (b->cpu != NULL)
			b->cpu->threads = set_threads(b->cpu->be, n);
	}
	if (b->threads == 0 && b->cpu != NULL)
		b = b->cpu;
	if (b->threads == 0)
		lua_pushnil(L);
	else
		lua_pushinteger(L, b->threads);
	return 1;
}

/*** Get the CPU backend of the backend.
 * It exists when `lgml.backend` had `cpu = true`. Weights that
 * `backend:load` puts on the CPU and leaves from its `backend:tensor`
 * can take part in graphs of this backend.
 * @treturn backend|nil the CPU backend, or nil
 * @function backend:cpu
 */
static int
backend_cpu(lua_State *L)
{
	struct backend *b = check_backend(L, 1);

	if (b->cpu == NULL)
		lua_pushnil(L);
	else
		lua_getiuservalue(L, 1, UV_CPU);
	return 1;
}

/* Push a store that owns ctx and buf, on the backend at bidx. */
static struct store *
push_store(lua_State *L, int bidx, struct ggml_context *ctx, ggml_backend_buffer_t buf)
{
	struct backend *b = lua_touserdata(L, bidx);
	struct store *s;

	s = lua_newuserdatauv(L, sizeof(*s), 3);
	s->b = b;
	s->ctx = ctx;
	s->buf = buf;
	s->refs = 0;
	s->closed = 0;
	s->mem_bytes = 0;
	lgml_store_account(s, MEM_TENSORS);
	b->refs++;
	luaL_setmetatable(L, STORE_MT);
	lua_pushvalue(L, bidx);
	lua_setiuservalue(L, -2, UV_PARENT);
	lua_newtable(L);
	lua_setiuservalue(L, -2, UV_EXTRA);
	return s;
}

struct store *
lgml_push_store(lua_State *L, int bidx, struct ggml_context *ctx, ggml_backend_buffer_t buf)
{
	return push_store(L, lua_absindex(L, bidx), ctx, buf);
}

/* Push a leaf tensor for gt in the store at sidx. */
static struct tensor *
push_leaf(lua_State *L, int sidx, struct ggml_tensor *gt, int rank)
{
	struct store *s = lua_touserdata(L, sidx);
	struct tensor *t;

	t = lua_newuserdatauv(L, sizeof(*t), 2);
	t->t = gt;
	t->s = s;
	t->g = NULL;
	t->rank = rank;
	t->out = 0;
	s->refs++;
	luaL_setmetatable(L, TENSOR_MT);
	lua_pushvalue(L, sidx);
	lua_setiuservalue(L, -2, UV_PARENT);
	return t;
}

static int tensor_set(lua_State *L);

/*** Make a leaf tensor with its own device storage.
 * The tensor frees its storage when it closes.
 * @tparam string type ggml type name, such as "f32" or `"q8_0"`
 * @tparam {int,...} shape 1 to 4 positive sizes
 * @tparam[opt] string|table data initial contents, as for `tensor:set`
 * @treturn tensor the leaf
 * @raise if the type is unknown, the innermost size is not a multiple of
 * the type block size, or the device is out of memory
 * @function backend:tensor
 */
static int
backend_tensor(lua_State *L)
{
	struct backend *b = check_backend(L, 1);
	enum ggml_type type = check_type(L, 2);
	int64_t ne[GGML_MAX_DIMS];
	struct ggml_init_params ip;
	struct ggml_context *ctx;
	struct ggml_tensor *gt;
	ggml_backend_buffer_type_t bt;
	ggml_backend_buffer_t buf;
	struct store *s;
	int rank, hasdata = !lua_isnoneornil(L, 4);

	rank = check_shape(L, 3, ne);
	if (ne[0] % ggml_blck_size(type) != 0)
		return luaL_error(L, "inner dimension not a multiple of %s block size",
		    ggml_type_name(type));

	ip.mem_size = ggml_tensor_overhead();
	ip.mem_buffer = NULL;
	ip.no_alloc = true;
	ctx = ggml_init(ip);
	if (ctx == NULL)
		return luaL_error(L, "ggml_init failed");
	gt = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, ne);
	bt = ggml_backend_get_default_buffer_type(b->be);
	if (lgml_mem_check(L, bt, ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, bt),
	    "a tensor") != NULL) {
		ggml_free(ctx);
		return lua_error(L);
	}
	buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, bt);
	if (buf == NULL) {
		ggml_free(ctx);
		return luaL_error(L, "out of device memory");
	}
	s = push_store(L, 1, ctx, buf);
	s->closed = 1;
	push_leaf(L, lua_gettop(L), gt, rank);
	lua_remove(L, -2);

	if (hasdata) {
		lua_pushcfunction(L, tensor_set);
		lua_pushvalue(L, -2);
		lua_pushvalue(L, 4);
		lua_call(L, 2, 0);
	}
	return 1;
}

/* Read the data of each tensor of ctx from the gguf file f. */
static int
read_tensors(FILE *f, const struct gguf_context *gctx, struct ggml_context *ctx, char *chunk)
{
	struct ggml_tensor *gt;
	size_t base, off, n, done, size;

	base = gguf_get_data_offset(gctx);
	for (gt = ggml_get_first_tensor(ctx); gt != NULL; gt = ggml_get_next_tensor(ctx, gt)) {
		off = base + gguf_get_tensor_offset(gctx, gguf_find_tensor(gctx, ggml_get_name(gt)));
		size = ggml_nbytes(gt);
		if (fseeko(f, (off_t)off, SEEK_SET) != 0)
			return -1;
		for (done = 0; done < size; done += n) {
			n = size - done < LOAD_CHUNK ? size - done : LOAD_CHUNK;
			if (fread(chunk, 1, n, f) != n)
				return -1;
			ggml_backend_tensor_set(gt, chunk, done, n);
		}
	}
	return 0;
}

/*
 * Move the tensors of *ctx that the set at hidx names to a new context
 * *hctx, and the others to a new *ctx. Return the count moved; with none,
 * the contexts stay as they are.
 */
static int
split_ctx(lua_State *L, int hidx, struct ggml_context **ctx, struct ggml_context **hctx)
{
	struct ggml_init_params ip = { 0, NULL, true };
	struct ggml_context *d, *h, *c = *ctx;
	struct ggml_tensor *gt, *nt;
	int n = 0, nh = 0, on;

	for (gt = ggml_get_first_tensor(c); gt != NULL; gt = ggml_get_next_tensor(c, gt)) {
		lua_getfield(L, hidx, ggml_get_name(gt));
		nh += lua_toboolean(L, -1);
		lua_pop(L, 1);
		n++;
	}
	if (nh == 0)
		return 0;
	ip.mem_size = (size_t)(n - nh + 1) * ggml_tensor_overhead();
	d = ggml_init(ip);
	ip.mem_size = (size_t)(nh + 1) * ggml_tensor_overhead();
	h = ggml_init(ip);
	if (d == NULL || h == NULL) {
		if (d != NULL)
			ggml_free(d);
		if (h != NULL)
			ggml_free(h);
		return -1;
	}
	for (gt = ggml_get_first_tensor(c); gt != NULL; gt = ggml_get_next_tensor(c, gt)) {
		lua_getfield(L, hidx, ggml_get_name(gt));
		on = lua_toboolean(L, -1);
		lua_pop(L, 1);
		nt = ggml_dup_tensor(on ? h : d, gt);
		ggml_set_name(nt, ggml_get_name(gt));
	}
	ggml_free(c);
	*ctx = d;
	*hctx = h;
	return nh;
}

/*
 * Allocate the tensors of ctx in bt as weights. NULL for no tensors.
 * A failed allocation leaves the tensors without data, for a retry.
 */
static int
alloc_weights(struct ggml_context *ctx, ggml_backend_buffer_type_t bt, ggml_backend_buffer_t *buf)
{
	struct ggml_tensor *gt;

	*buf = NULL;
	if (ggml_get_first_tensor(ctx) == NULL)
		return 0;
	if ((*buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, bt)) == NULL) {
		for (gt = ggml_get_first_tensor(ctx); gt != NULL; gt = ggml_get_next_tensor(ctx, gt)) {
			gt->data = NULL;
			gt->buffer = NULL;
		}
		return -1;
	}
	ggml_backend_buffer_set_usage(*buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
	return 0;
}

/*** Load every tensor of a gguf file to the device.
 * With `opts.cpu`, the tensors it names go to the CPU backend of this
 * backend (see `backend:cpu`); `weights:get` finds tensors of both parts.
 * They go to pinned host memory of the device when it has some, so that
 * the device can read them fast for large batches, else to plain memory.
 * @tparam string path gguf file
 * @tparam[opt] table opts `cpu`: a set of tensor names, `{ [name] =
 * true }`, to put on the CPU backend; `pinned`: false to use plain
 * memory for them (default true)
 * @treturn[1] weights the tensors
 * @return[2] nil if the file is not a gguf file
 * @treturn[2] string error message
 * @raise if the device is out of memory, the read fails, or `opts.cpu`
 * names a tensor and the backend has no CPU backend
 * @function backend:load
 */
static int
backend_load(lua_State *L)
{
	struct backend *b = check_backend(L, 1);
	const char *path = luaL_checkstring(L, 2);
	struct gguf_init_params gp;
	struct gguf_context *gctx;
	struct ggml_context *ctx = NULL, *hctx = NULL;
	ggml_backend_buffer_t buf, hbuf = NULL;
	ggml_backend_buffer_type_t hbt, cbt;
	char *chunk;
	FILE *f;
	int e, hidx = 0, sidx, nh = 0, pinned = 1;

	if (!lua_isnoneornil(L, 3)) {
		luaL_checktype(L, 3, LUA_TTABLE);
		if (lua_getfield(L, 3, "pinned") != LUA_TNIL)
			pinned = lua_toboolean(L, -1);
		lua_pop(L, 1);
		if (lua_getfield(L, 3, "cpu") != LUA_TNIL) {
			luaL_checktype(L, -1, LUA_TTABLE);
			hidx = lua_gettop(L);
		}
	}
	gp.no_alloc = true;
	gp.ctx = &ctx;
	gctx = gguf_init_from_file(path, gp);
	if (gctx == NULL) {
		luaL_pushfail(L);
		lua_pushfstring(L, "%s: not a gguf file", path);
		return 2;
	}
	if (hidx != 0 && (nh = split_ctx(L, hidx, &ctx, &hctx)) != 0) {
		if (nh > 0 && b->cpu != NULL) {
			cbt = ggml_backend_get_default_buffer_type(b->cpu->be);
			if (lgml_mem_check(L, cbt, ggml_backend_alloc_ctx_tensors_from_buft_size(hctx,
			    cbt), path) != NULL)
				goto refuse;
			cbt = ggml_backend_get_default_buffer_type(b->be);
			if (lgml_mem_check(L, cbt, ggml_backend_alloc_ctx_tensors_from_buft_size(ctx,
			    cbt), path) != NULL)
				goto refuse;
		}
		if (nh < 0 || b->cpu == NULL) {
			gguf_free(gctx);
			ggml_free(ctx);
			if (hctx != NULL)
				ggml_free(hctx);
			return luaL_error(L, nh < 0 ? "ggml_init failed" :
			    "%s: tensors for the CPU, but the backend has no CPU backend", path);
		}
		/*
		 * Pinned host memory of the device, when it has some, makes
		 * the copies of op offload faster. Plain memory if that fails.
		 */
		cbt = ggml_backend_get_default_buffer_type(b->cpu->be);
		hbt = pinned ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(b->be)) : NULL;
		if (hbt == NULL || alloc_weights(hctx, hbt, &hbuf) != 0)
			e = alloc_weights(hctx, cbt, &hbuf);
		else
			e = 0;
		if (e != 0 || alloc_weights(ctx, ggml_backend_get_default_buffer_type(b->be), &buf) != 0) {
			if (hbuf != NULL)
				ggml_backend_buffer_free(hbuf);
			gguf_free(gctx);
			ggml_free(ctx);
			ggml_free(hctx);
			return luaL_error(L, "out of device memory");
		}
	} else if (lgml_mem_check(L, ggml_backend_get_default_buffer_type(b->be),
	    ggml_backend_alloc_ctx_tensors_from_buft_size(ctx,
	    ggml_backend_get_default_buffer_type(b->be)), path) != NULL) {
		goto refuse;
	} else if ((buf = ggml_backend_alloc_ctx_tensors(ctx, b->be)) == NULL) {
		gguf_free(gctx);
		ggml_free(ctx);
		return luaL_error(L, "out of device memory");
	}
	lgml_store_account(push_store(L, 1, ctx, buf), MEM_WEIGHTS);
	sidx = lua_gettop(L);
	if (hctx != NULL) {
		lua_getiuservalue(L, 1, UV_CPU);
		lgml_store_account(push_store(L, lua_gettop(L), hctx, hbuf), MEM_WEIGHTS);
		lua_setiuservalue(L, sidx, UV_HOST);
		lua_pop(L, 1);
	}

	f = fopen(path, "rb");
	e = errno;
	chunk = f != NULL ? malloc(LOAD_CHUNK) : NULL;
	if (chunk == NULL) {
		if (f != NULL) {
			e = ENOMEM;
			fclose(f);
		}
		gguf_free(gctx);
		return luaL_error(L, "%s: %s", path, strerror(e));
	}
	e = read_tensors(f, gctx, ctx, chunk);
	if (e == 0 && hctx != NULL)
		e = read_tensors(f, gctx, hctx, chunk);
	free(chunk);
	fclose(f);
	gguf_free(gctx);
	if (e != 0)
		return luaL_error(L, "%s: short read", path);
	return 1;
refuse:
	gguf_free(gctx);
	ggml_free(ctx);
	if (hctx != NULL)
		ggml_free(hctx);
	return lua_error(L);
}

/*** Make an empty graph on this backend.
 * The graph grows as ops add ggml tensors, up to 1048576 of them. Some
 * ops add more than one. The size is a hint for the first room; a close
 * hint saves a few copies of the graph as it grows.
 * @tparam[opt=16384] int size the first room in ggml tensors, 1 to 1048576
 * @treturn graph the graph
 * @function backend:graph
 */
static int
backend_graph(lua_State *L)
{
	struct backend *b = check_backend(L, 1);
	lua_Integer size = luaL_optinteger(L, 2, GRAPH_SIZE);
	struct graph *g;

	luaL_argcheck(L, size >= 1 && size <= GRAPH_MAX, 2, "size out of range");
	g = lua_newuserdatauv(L, sizeof(*g), 2);
	memset(g, 0, sizeof(*g));
	g->closed = 1;
	luaL_setmetatable(L, GRAPH_MT);
	lua_pushvalue(L, 1);
	lua_setiuservalue(L, -2, UV_PARENT);
	lua_newtable(L);
	lua_setiuservalue(L, -2, UV_EXTRA);

	/* From here graph_release frees what exists, on error too. */
	g->b = b;
	b->refs++;
	g->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(b->be));
	if (g->galloc == NULL)
		return luaL_error(L, "ggml_gallocr_new failed");
	g->size = size < 64 ? 64 : (int)size;
	g->gf = lgml_new_cgraph(L, g, g->size, 0);
	g->ctx = graph_ctx(L, g, (size_t)(size + CTX_SLACK) * ggml_tensor_overhead());
	g->ctx_left = (int)size;
	g->plan_nodes = -1;
	g->closed = 0;
	return 1;
}

/* weights (store) */

static int tensor_close(lua_State *L);

/*** Device storage for leaf tensors.
 * Get one from `backend:load` or `backend:load_safetensors`.
 * @type weights
 */

/*** Close the weights and every tensor taken from them.
 * The `__gc` and `__close` metamethods call this function.
 * @function weights:close
 */
static int
store_close(lua_State *L)
{
	struct store *s = luaL_checkudata(L, 1, STORE_MT);

	if (s->closed)
		return 0;
	s->closed = 1;
	lua_getiuservalue(L, 1, UV_EXTRA);
	lua_pushnil(L);
	while (lua_next(L, -2) != 0) {
		lua_pushcfunction(L, tensor_close);
		lua_insert(L, -2);
		lua_call(L, 1, 0);
	}
	lua_pop(L, 1);
	if (lua_getiuservalue(L, 1, UV_HOST) == LUA_TUSERDATA) {
		lua_pushcfunction(L, store_close);
		lua_insert(L, -2);
		lua_call(L, 1, 0);
	} else {
		lua_pop(L, 1);
	}
	store_release(s);
	return 0;
}

/*** Get a tensor by name.
 * Each call with the same name returns the same tensor object. Its shape
 * drops outer dimensions of size 1.
 * @tparam string name tensor name
 * @treturn tensor|nil the leaf, or nil if the weights have no such tensor
 * @function weights:get
 */
static int
store_get(lua_State *L)
{
	struct store *s = check_store(L, 1), *h;
	const char *name = luaL_checkstring(L, 2);
	struct ggml_tensor *gt;
	int sidx = 1;

	lua_getiuservalue(L, 1, UV_EXTRA);
	if (lua_getfield(L, -1, name) != LUA_TNIL)
		return 1;
	lua_pop(L, 1);
	gt = ggml_get_tensor(s->ctx, name);
	if (gt == NULL && lua_getiuservalue(L, 1, UV_HOST) == LUA_TUSERDATA) {
		h = lua_touserdata(L, -1);
		sidx = lua_gettop(L);
		gt = ggml_get_tensor(h->ctx, name);
	}
	if (gt == NULL) {
		lua_pushnil(L);
		return 1;
	}
	push_leaf(L, sidx, gt, min_rank(gt));
	lua_pushvalue(L, -1);
	lua_setfield(L, 3, name);
	return 1;
}

/*** List the tensor names.
 * @treturn {string,...} names in storage order, the CPU part last
 * @function weights:names
 */
static int
store_names(lua_State *L)
{
	struct store *s = check_store(L, 1), *h = NULL;
	struct ggml_context *c;
	struct ggml_tensor *gt;
	int i = 0;

	if (lua_getiuservalue(L, 1, UV_HOST) == LUA_TUSERDATA)
		h = lua_touserdata(L, -1);
	lua_newtable(L);
	for (c = s->ctx; c != NULL; c = c == s->ctx && h != NULL ? h->ctx : NULL) {
		for (gt = ggml_get_first_tensor(c); gt != NULL; gt = ggml_get_next_tensor(c, gt)) {
			lua_pushstring(L, ggml_get_name(gt));
			lua_seti(L, -2, ++i);
		}
	}
	return 1;
}

/*** Get the size of the device buffer.
 * @treturn int bytes on the backend
 * @treturn int bytes on its CPU backend (see `backend:load`)
 * @function weights:size
 */
static int
store_size(lua_State *L)
{
	struct store *s = check_store(L, 1), *h;

	lua_pushinteger(L, s->buf != NULL ? (lua_Integer)ggml_backend_buffer_get_size(s->buf) : 0);
	if (lua_getiuservalue(L, 1, UV_HOST) == LUA_TUSERDATA) {
		h = lua_touserdata(L, -1);
		lua_pushinteger(L, h->buf != NULL ? (lua_Integer)ggml_backend_buffer_get_size(h->buf) : 0);
	} else {
		lua_pushinteger(L, 0);
	}
	lua_remove(L, -2);
	return 2;
}

/* graph */

/*** A graph of ops on one backend.
 * Each op returns a node tensor. Nodes get storage when `graph:compute`
 * runs. An operand is a leaf on the same backend or its CPU backend (see
 * `backend:cpu`), or a node of this graph. A graph with leaves on both
 * runs through a ggml scheduler. The graph keeps its leaves alive until
 * it closes.
 * @type graph
 */

static void restore_srcs(struct graph *);

/*
 * Free the compute buffers of a graph. Its nodes lose their storage, so
 * a read of a node then raises an error.
 */
static void
free_compute(struct graph *g)
{
	struct ggml_tensor *node;
	int i;

	for (i = 0; i < 2; i++) {
		if (g->mem_bytes[i] > 0)
			lgml_mem_add(g->mem_key[i], MEM_COMPUTE, 0, g->mem_bytes[i]);
		g->mem_bytes[i] = 0;
	}
	restore_srcs(g);
	if (g->sched != NULL)
		ggml_backend_sched_free(g->sched);
	if (g->galloc != NULL)
		ggml_gallocr_free(g->galloc);
	g->sched = NULL;
	g->galloc = NULL;
	for (i = 0; i < g->nctx; i++) {
		for (node = ggml_get_first_tensor(g->ctxs[i]); node != NULL;
		    node = ggml_get_next_tensor(g->ctxs[i], node)) {
			node->data = NULL;
			node->buffer = NULL;
		}
	}
	g->plan_nodes = -1;
}

/*** Close the graph.
 * The compute buffers go when no output node stays open: at once, or
 * when the last output closes. The rest stays until every node tensor
 * of the graph closes. The `__gc` and `__close` metamethods call this
 * function.
 * @function graph:close
 */
static int
graph_close(lua_State *L)
{
	struct graph *g = luaL_checkudata(L, 1, GRAPH_MT);

	if (!g->closed && g->b != NULL && g->nouts == 0)
		free_compute(g);
	g->closed = 1;
	graph_release(g);
	return 0;
}

/*
 * Get an op operand: a leaf on the graph backend, or a node of this
 * graph. The graph keeps leaves alive until it closes.
 */
static struct tensor *
operand(lua_State *L, struct graph *g, int idx)
{
	struct tensor *t = check_tensor(L, idx);

	if (t->g == g)
		return t;
	if (t->g != NULL)
		luaL_argerror(L, idx, "node of another graph");
	if (t->s->b != g->b) {
		if (g->b->cpu == NULL || t->s->b != g->b->cpu)
			luaL_argerror(L, idx, "tensor on another backend");
		g->mixed = 1;
	}
	lua_getiuservalue(L, 1, UV_EXTRA);
	lua_pushvalue(L, idx);
	if (lua_rawget(L, -2) == LUA_TNIL) {
		/* A leaf takes a graph slot as a leaf, or as a node if trainable. */
		lgml_graph_room(L, g, 1);
		g->nleaves++;
		lua_pushvalue(L, idx);
		lua_pushboolean(L, 1);
		lua_rawset(L, -4);
	}
	lua_pop(L, 2);
	return t;
}

static struct tensor *
opt_operand(lua_State *L, struct graph *g, int idx)
{
	return lua_isnoneornil(L, idx) ? NULL : operand(L, g, idx);
}

/* Reserve room for n new ggml tensors in the graph context. */
static void
reserve(lua_State *L, struct graph *g, int n)
{
	lgml_graph_room(L, g, n);
	g->ntensors += n;
	g->ctx_left -= n;
}

static int
push_node(lua_State *L, struct graph *g, struct ggml_tensor *r, int rank)
{
	struct tensor *t;

	t = lua_newuserdatauv(L, sizeof(*t), 1);
	t->t = r;
	t->s = NULL;
	t->g = g;
	t->out = 0;
	t->rank = max_int(rank, min_rank(r));
	g->refs++;
	luaL_setmetatable(L, TENSOR_MT);
	lua_pushvalue(L, 1);
	lua_setiuservalue(L, -2, UV_PARENT);
	return 1;
}

typedef struct ggml_tensor *(*binfn)(struct ggml_context *, struct ggml_tensor *,
    struct ggml_tensor *);
typedef struct ggml_tensor *(*unfn)(struct ggml_context *, struct ggml_tensor *);
typedef struct ggml_tensor *(*normfn)(struct ggml_context *, struct ggml_tensor *, float);

/* Elementwise binary op. b broadcasts to the shape of a. */
static int
binop(lua_State *L, binfn fn)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	struct tensor *b = operand(L, g, 3);

	enum ggml_type ta = a->t->type, tb = b->t->type;

	if (!ggml_can_repeat(b->t, a->t))
		return luaL_error(L, "shape of arg 2 does not broadcast to arg 1");
	/* The CPU backend aborts on other type pairs. */
	if ((ta != GGML_TYPE_F32 && ta != GGML_TYPE_F16 && ta != GGML_TYPE_BF16) ||
	    (tb != GGML_TYPE_F32 && tb != ta))
		return luaL_error(L, "types %s and %s: arg 2 must be f32 or match arg 1; use cast",
		    ggml_type_name(ta), ggml_type_name(tb));
	reserve(L, g, 1);
	return push_node(L, g, fn(g->ctx, a->t, b->t), a->rank);
}

/*** Elementwise a + b.
 * b broadcasts to the shape of a: each size of a is a multiple of the
 * same size of b. a is f32, f16 or bf16. b is f32 or the type of a.
 * @tparam tensor a left operand
 * @tparam tensor b right operand
 * @treturn tensor node with the shape of a
 * @raise if b does not broadcast to a, or on other types; use `graph:cast`
 * @function graph:add
 */
static int graph_add(lua_State *L) { return binop(L, ggml_add); }

/*** Elementwise a - b.
 * b broadcasts to the shape of a: each size of a is a multiple of the
 * same size of b. a is f32, f16 or bf16. b is f32 or the type of a.
 * @tparam tensor a left operand
 * @tparam tensor b right operand
 * @treturn tensor node with the shape of a
 * @raise if b does not broadcast to a, or on other types; use `graph:cast`
 * @function graph:sub
 */
static int graph_sub(lua_State *L) { return binop(L, ggml_sub); }

/*** Elementwise a * b.
 * b broadcasts to the shape of a: each size of a is a multiple of the
 * same size of b. a is f32, f16 or bf16. b is f32 or the type of a.
 * @tparam tensor a left operand
 * @tparam tensor b right operand
 * @treturn tensor node with the shape of a
 * @raise if b does not broadcast to a, or on other types; use `graph:cast`
 * @function graph:mul
 */
static int graph_mul(lua_State *L) { return binop(L, ggml_mul); }

/*** Elementwise a / b.
 * b broadcasts to the shape of a: each size of a is a multiple of the
 * same size of b. a is f32, f16 or bf16. b is f32 or the type of a.
 * @tparam tensor a left operand
 * @tparam tensor b right operand
 * @treturn tensor node with the shape of a
 * @raise if b does not broadcast to a, or on other types; use `graph:cast`
 * @function graph:div
 */
static int graph_div(lua_State *L) { return binop(L, ggml_div); }

static int
unop(lua_State *L, unfn fn)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);

	if (!ggml_is_contiguous_rows(a->t))
		return luaL_error(L, "rows not contiguous; use cont");
	reserve(L, g, 1);
	return push_node(L, g, fn(g->ctx, a->t), a->rank);
}

/*** Elementwise SiLU: a * sigmoid(a).
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:silu
 */
static int graph_silu(lua_State *L) { return unop(L, ggml_silu); }

/*** Elementwise tanh.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:tanh
 */
static int graph_tanh(lua_State *L) { return unop(L, ggml_tanh); }

/*** Elementwise logistic sigmoid.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:sigmoid
 */
static int graph_sigmoid(lua_State *L) { return unop(L, ggml_sigmoid); }

/*** Elementwise ReLU: max(a, 0).
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:relu
 */
static int graph_relu(lua_State *L) { return unop(L, ggml_relu); }

/*** Elementwise ELU with alpha 1.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:elu
 */
static int graph_elu(lua_State *L) { return unop(L, ggml_elu); }

/*** Elementwise exp.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:exp
 */
static int graph_exp(lua_State *L) { return unop(L, ggml_exp); }

/*** Elementwise absolute value.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:abs
 */
static int graph_abs(lua_State *L) { return unop(L, ggml_abs); }

/*** Elementwise negation.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:neg
 */
static int graph_neg(lua_State *L) { return unop(L, ggml_neg); }

/*** Elementwise sine.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:sin
 */
static int graph_sin(lua_State *L) { return unop(L, ggml_sin); }

/*** Elementwise cosine.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:cos
 */
static int graph_cos(lua_State *L) { return unop(L, ggml_cos); }

/*** Elementwise square.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:sqr
 */
static int graph_sqr(lua_State *L) { return unop(L, ggml_sqr); }

/*** Elementwise square root.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:sqrt
 */
static int graph_sqrt(lua_State *L) { return unop(L, ggml_sqrt); }

/*** Elementwise GELU, tanh approximation.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:gelu
 */
static int graph_gelu(lua_State *L) { return unop(L, ggml_gelu); }

/*** Elementwise GELU with the exact erf form.
 * @tparam tensor a input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if rows of a are not contiguous; use `graph:cont`
 * @function graph:gelu_erf
 */
static int graph_gelu_erf(lua_State *L) { return unop(L, ggml_gelu_erf); }

/*** Multiply every element by a number, then add a number.
 * Rows may have a stride. The other dimensions must be contiguous.
 * @tparam tensor a input
 * @tparam number s factor
 * @tparam[opt=0] number b added after the product
 * @treturn tensor node with the shape of a
 * @raise if a is not contiguous; use `graph:cont`
 * @function graph:scale
 */
static int
graph_scale(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	float s = (float)luaL_checknumber(L, 3);
	float bias = (float)luaL_optnumber(L, 4, 0);
	const struct ggml_tensor *t = a->t;

	if (t->nb[0] != ggml_type_size(t->type) || t->nb[2] != t->nb[1] * (size_t)t->ne[1] ||
	    t->nb[3] != t->nb[2] * (size_t)t->ne[2])
		return luaL_error(L, "tensor not contiguous; use cont");
	reserve(L, g, 1);
	if (bias != 0)
		return push_node(L, g, ggml_scale_bias(g->ctx, a->t, s, bias), a->rank);
	return push_node(L, g, ggml_scale(g->ctx, a->t, s), a->rank);
}

static int
normop(lua_State *L, normfn fn)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	float eps = (float)luaL_optnumber(L, 3, 1e-6);

	reserve(L, g, 1);
	return push_node(L, g, fn(g->ctx, a->t, eps), a->rank);
}

/*** RMS-normalize each row.
 * Divide each row by the root mean square of the row. No weight applies.
 * @tparam tensor a input
 * @tparam[opt=1e-6] number eps added to the mean square
 * @treturn tensor node with the shape of a
 * @function graph:rms_norm
 */
static int graph_rms_norm(lua_State *L) { return normop(L, ggml_rms_norm); }

/*** Normalize each row to zero mean and unit variance.
 * No weight or bias applies.
 * @tparam tensor a input
 * @tparam[opt=1e-6] number eps added to the variance
 * @treturn tensor node with the shape of a
 * @function graph:norm
 */
static int graph_norm(lua_State *L) { return normop(L, ggml_norm); }

/*** Swap the two innermost dimensions.
 * The result is a view with at least rank 2. Use `graph:cont` for a
 * contiguous copy.
 * @tparam tensor a input
 * @treturn tensor view node
 * @function graph:transpose
 */
static int
graph_transpose(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);

	reserve(L, g, 1);
	return push_node(L, g, ggml_transpose(g->ctx, a->t), max_int(a->rank, 2));
}

/*** Make a contiguous copy.
 * @tparam tensor a input, such as a view or a permuted tensor
 * @treturn tensor node with the shape of a
 * @function graph:cont
 */
static int
graph_cont(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);

	reserve(L, g, 1);
	return push_node(L, g, ggml_cont(g->ctx, a->t), a->rank);
}

/*** Convert to another type.
 * @tparam tensor a input
 * @tparam string type ggml type name, such as "f16"
 * @treturn tensor node with the shape of a
 * @raise if the type is unknown
 * @function graph:cast
 */
static int
graph_cast(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	enum ggml_type type = check_type(L, 3);

	reserve(L, g, 1);
	return push_node(L, g, ggml_cast(g->ctx, a->t, type), a->rank);
}

/*** Give a contiguous tensor a new shape.
 * @tparam tensor a contiguous input
 * @tparam {int,...} shape 1 to 4 sizes with the element count of a
 * @treturn tensor view node
 * @raise if a is not contiguous or the element count changes
 * @function graph:reshape
 */
static int
graph_reshape(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	int64_t ne[GGML_MAX_DIMS];
	int rank = check_shape(L, 3, ne);

	if (!ggml_is_contiguous(a->t))
		return luaL_error(L, "tensor not contiguous; use cont");
	if (ne[0] * ne[1] * ne[2] * ne[3] != ggml_nelements(a->t))
		return luaL_error(L, "reshape changes element count");
	reserve(L, g, 1);
	return push_node(L, g, ggml_reshape_4d(g->ctx, a->t, ne[0], ne[1], ne[2], ne[3]), rank);
}

/*** Make a view into the storage of a.
 * Strides are in bytes, in torch order, one for each dimension except the
 * innermost. The default is the strides of a.
 * @tparam tensor a input
 * @tparam {int,...} shape 1 to 4 sizes
 * @tparam[opt] {int,...} strides rank - 1 positive byte strides
 * @tparam[opt=0] int offset byte offset from the start of a
 * @treturn tensor view node
 * @raise if the view is out of bounds, or the innermost size is not a
 * multiple of the type block size
 * @function graph:view
 */
static int
graph_view(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	int64_t ne[GGML_MAX_DIMS];
	size_t nb[GGML_MAX_DIMS], off, need;
	const struct ggml_tensor *root;
	lua_Integer v;
	int i, rank = check_shape(L, 3, ne);

	for (i = 1; i < GGML_MAX_DIMS; i++)
		nb[i] = a->t->nb[i];
	if (!lua_isnoneornil(L, 4)) {
		luaL_checktype(L, 4, LUA_TTABLE);
		luaL_argcheck(L, luaL_len(L, 4) == rank - 1, 4, "need rank - 1 strides");
		for (i = 0; i < rank - 1; i++) {
			lua_geti(L, 4, i + 1);
			v = luaL_checkinteger(L, -1);
			lua_pop(L, 1);
			luaL_argcheck(L, v > 0, 4, "stride must be positive");
			nb[rank - 1 - i] = (size_t)v;
		}
		for (i = rank; i < GGML_MAX_DIMS; i++)
			nb[i] = nb[i - 1] * (size_t)ne[i - 1];
	}
	v = luaL_optinteger(L, 5, 0);
	luaL_argcheck(L, v >= 0, 5, "offset must not be negative");
	if (ne[0] % ggml_blck_size(a->t->type) != 0)
		return luaL_error(L, "inner dimension not a multiple of block size");
	root = a->t->view_src != NULL ? a->t->view_src : a->t;
	off = (size_t)v + (a->t->view_src != NULL ? a->t->view_offs : 0);
	need = ggml_row_size(a->t->type, ne[0]) * (size_t)(ne[1] * ne[2] * ne[3]);
	if (off + need > ggml_nbytes(root))
		return luaL_error(L, "view out of bounds");
	reserve(L, g, 1);
	return push_node(L, g, ggml_view_4d(g->ctx, a->t, ne[0], ne[1], ne[2], ne[3],
	    nb[1], nb[2], nb[3], (size_t)v), rank);
}

/*** Reorder dimensions.
 * Dimension i of the result is dimension perm[i] of a. The result is a view.
 * @tparam tensor a input
 * @tparam {int,...} perm one entry for each dimension of a, from 1
 * @treturn tensor view node with the rank of a
 * @raise if perm is not a permutation of 1 to the rank of a
 * @function graph:permute
 */
static int
graph_permute(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	int ax[GGML_MAX_DIMS], seen[GGML_MAX_DIMS] = { 0 };
	int i, p, n = a->rank;
	lua_Integer v;

	luaL_checktype(L, 3, LUA_TTABLE);
	luaL_argcheck(L, luaL_len(L, 3) == n, 3, "need one entry per dimension");
	for (i = 0; i < GGML_MAX_DIMS; i++)
		ax[i] = i;
	for (i = 0; i < n; i++) {
		lua_geti(L, 3, i + 1);
		v = luaL_checkinteger(L, -1);
		lua_pop(L, 1);
		luaL_argcheck(L, v >= 1 && v <= n, 3, "dimension out of range");
		p = (int)v - 1;
		luaL_argcheck(L, !seen[p], 3, "dimension repeated");
		seen[p] = 1;
		ax[n - 1 - p] = n - 1 - i;
	}
	reserve(L, g, 1);
	return push_node(L, g, ggml_permute(g->ctx, a->t, ax[0], ax[1], ax[2], ax[3]), n);
}

static int
can_mul_mat(const struct ggml_tensor *a, const struct ggml_tensor *b)
{
	return a->ne[0] == b->ne[0] && b->ne[2] % a->ne[2] == 0 &&
	    b->ne[3] % a->ne[3] == 0;
}

/*** Matrix product in ggml convention.
 * a is (M, K), b is (N, K), and the result is (N, M). Each row of the
 * result is a times one row of b. Each outer size of b is a multiple of
 * the same size of a, and a broadcasts.
 * @tparam tensor a (M, K), often a weight
 * @tparam tensor b (N, K)
 * @tparam[opt=false] boolean f32 ask the backend to add up in f32; some
 * GPU backends add up f16 products in f16 by default
 * @treturn tensor (N, M) node
 * @raise on a shape mismatch
 * @function graph:mul_mat
 */
static int
graph_mul_mat(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	struct tensor *b = operand(L, g, 3);
	int f32 = lua_toboolean(L, 4);
	struct ggml_tensor *r;

	if (!can_mul_mat(a->t, b->t))
		return luaL_error(L, "mul_mat shape mismatch");
	reserve(L, g, 1);
	r = ggml_mul_mat(g->ctx, a->t, b->t);
	if (f32)
		ggml_prec_set_acc(r, GGML_PREC_F32);
	return push_node(L, g, r, max_int(a->rank, b->rank));
}

/*** Matrix product in torch convention.
 * x is (N, K), y is (K, M), and the result is (N, M). Each outer size of x
 * is a multiple of the same size of y, and y broadcasts. The op copies a
 * transpose of y.
 * @tparam tensor x (N, K)
 * @tparam tensor y (K, M)
 * @treturn tensor (N, M) node
 * @raise on a shape mismatch
 * @function graph:matmul
 */
static int
graph_matmul(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *x = operand(L, g, 2);
	struct tensor *y = operand(L, g, 3);
	struct ggml_tensor *yt;

	if (x->t->ne[0] != y->t->ne[1] || x->t->ne[2] % y->t->ne[2] != 0 ||
	    x->t->ne[3] % y->t->ne[3] != 0)
		return luaL_error(L, "matmul shape mismatch");
	reserve(L, g, 3);
	yt = ggml_cont(g->ctx, ggml_transpose(g->ctx, y->t));
	return push_node(L, g, ggml_mul_mat(g->ctx, yt, x->t), max_int(x->rank, y->rank));
}

/*** 1-D convolution, as torch conv1d without bias.
 * The result length is (L + 2 * pad - dilation * (K - 1) - 1) / stride + 1.
 * The op runs as im2col and `mul_mat`, with f32 columns.
 * @tparam tensor w contiguous (OC, IC, K)
 * @tparam tensor x contiguous f32 (IC, L) or (N, IC, L)
 * @tparam[opt=1] int stride 1 to 65536
 * @tparam[opt=0] int pad zeros on each side, 0 to 65536
 * @tparam[opt=1] int dilation 1 to 65536
 * @treturn tensor (OC, OL) or (N, OC, OL) node
 * @raise on bad shapes, types or ranges, or if the kernel is longer than
 * the padded input
 * @function graph:conv1d
 */
static int
graph_conv1d(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *w = operand(L, g, 2);
	struct tensor *x = operand(L, g, 3);
	lua_Integer s = luaL_optinteger(L, 4, 1);
	lua_Integer p = luaL_optinteger(L, 5, 0);
	lua_Integer d = luaL_optinteger(L, 6, 1);
	const struct ggml_tensor *tw = w->t, *tx = x->t;
	struct ggml_tensor *col, *w2, *r;
	int64_t ol;

	luaL_argcheck(L, s >= 1 && s <= 65536, 4, "stride out of range");
	luaL_argcheck(L, p >= 0 && p <= 65536, 5, "pad out of range");
	luaL_argcheck(L, d >= 1 && d <= 65536, 6, "dilation out of range");
	if (tw->ne[3] != 1 || tx->ne[3] != 1 || tw->ne[1] != tx->ne[1])
		return luaL_error(L, "conv1d: need w (OC, IC, K) and x (N, IC, L)");
	if (tx->type != GGML_TYPE_F32 || !ggml_is_contiguous(tx) || !ggml_is_contiguous(tw))
		return luaL_error(L, "conv1d: need contiguous w and f32 x");
	ol = (tx->ne[0] + 2 * p - d * (tw->ne[0] - 1) - 1) / s + 1;
	if (tx->ne[0] + 2 * p - d * (tw->ne[0] - 1) - 1 < 0 || ol < 1)
		return luaL_error(L, "conv1d: kernel larger than padded input");
	reserve(L, g, 7);
	col = ggml_im2col(g->ctx, w->t, x->t, (int)s, 0, (int)p, 0, (int)d, 0, false,
	    GGML_TYPE_F32);
	w2 = ggml_reshape_2d(g->ctx, w->t, tw->ne[0] * tw->ne[1], tw->ne[2]);
	r = ggml_mul_mat(g->ctx, w2, ggml_reshape_2d(g->ctx, col, col->ne[0],
	    col->ne[1] * col->ne[2]));
	r = ggml_reshape_3d(g->ctx, r, tw->ne[2], col->ne[1], col->ne[2]);
	return push_node(L, g, ggml_cont(g->ctx, ggml_transpose(g->ctx, r)), x->rank);
}

/*** Pick rows of a by index.
 * For a of (R, C) and ids of (n), the result is (n, C). For a of (B, R, C),
 * ids is (B, n) and the result is (B, n, C). Ids count from 0.
 * @tparam tensor a source
 * @tparam tensor ids i32 row indices
 * @treturn tensor node with the rank of ids plus 1
 * @raise if ids is not i32 or the shapes do not match
 * @function graph:get_rows
 */
static int
graph_get_rows(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	struct tensor *b = operand(L, g, 3);

	if (b->t->type != GGML_TYPE_I32)
		return luaL_error(L, "ids must be i32");
	if (a->t->ne[2] != b->t->ne[1] || a->t->ne[3] != b->t->ne[2] || b->t->ne[3] != 1)
		return luaL_error(L, "get_rows shape mismatch");
	reserve(L, g, 1);
	return push_node(L, g, ggml_get_rows(g->ctx, a->t, b->t), b->rank + 1);
}

/*** Write rows of src into dst at ids.
 * dst is (..., R, C), src is (..., n, C) and ids is (n) or has outer sizes
 * that divide those of src. Ids count from 0. The result is dst after the
 * write. Compute it, or pass it to `graph:expand`, to do the write.
 * @tparam tensor dst destination with contiguous rows
 * @tparam tensor src f32 or f16 rows with contiguous rows
 * @tparam tensor ids i32 or i64 row indices
 * @treturn tensor node with the shape of dst
 * @raise on bad types or shapes, or rows that are not contiguous
 * @function graph:set_rows
 */
static int
graph_set_rows(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	struct tensor *b = operand(L, g, 3);
	struct tensor *c = operand(L, g, 4);
	const struct ggml_tensor *ta = a->t, *tb = b->t, *tc = c->t;

	if (tb->type != GGML_TYPE_F32 && tb->type != GGML_TYPE_F16)
		return luaL_error(L, "src must be f32 or f16");
	if (tc->type != GGML_TYPE_I32 && tc->type != GGML_TYPE_I64)
		return luaL_error(L, "ids must be i32 or i64");
	if (ta->ne[0] != tb->ne[0] || ta->ne[2] != tb->ne[2] || ta->ne[3] != tb->ne[3] ||
	    tb->ne[1] != tc->ne[0] || tb->ne[2] % tc->ne[1] != 0 ||
	    tb->ne[3] % tc->ne[2] != 0 || tc->ne[3] != 1)
		return luaL_error(L, "set_rows shape mismatch");
	if (!ggml_is_contiguous_rows(ta) || !ggml_is_contiguous_rows(tb))
		return luaL_error(L, "rows not contiguous");
	reserve(L, g, 1);
	return push_node(L, g, ggml_set_rows(g->ctx, a->t, b->t, c->t), a->rank);
}

/*** Copy a into b and convert to the type of b.
 * The result is a view of b. Compute it, or pass it to `graph:expand`, to
 * do the write.
 * @tparam tensor a source
 * @tparam tensor b destination with the element count of a
 * @treturn tensor view node with the shape of b
 * @raise if the element counts differ
 * @function graph:cpy
 */
static int
graph_cpy(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	struct tensor *b = operand(L, g, 3);

	if (ggml_nelements(a->t) != ggml_nelements(b->t))
		return luaL_error(L, "cpy element count mismatch");
	reserve(L, g, 1);
	return push_node(L, g, ggml_cpy(g->ctx, a->t, b->t), b->rank);
}

static float
optfield(lua_State *L, int idx, const char *k, float def)
{
	float v;

	lua_getfield(L, idx, k);
	v = (float)luaL_optnumber(L, -1, def);
	lua_pop(L, 1);
	return v;
}

/*** Apply rotary position embedding.
 * opts fields: `n_dims` (even, at most `head_dim`; default `head_dim`), mode
 * ("normal" or "neox"), base (10000), scale (1), `n_ctx_orig` (0),
 * `ext_factor` (0), `attn_factor` (1), `beta_fast` (32), `beta_slow` (1).
 * @tparam tensor a (tokens, heads, `head_dim`)
 * @tparam tensor pos i32 (tokens) positions
 * @tparam table opts options
 * @tparam[opt] tensor freq f32 frequency factors, at least `n_dims` / 2
 * @treturn tensor node with the shape of a
 * @raise on a bad `n_dims`, pos or freq
 * @function graph:rope
 */
static int
graph_rope(lua_State *L)
{
	static const char *const modes[] = { "normal", "neox", NULL };
	static const int modev[] = { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX };
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	struct tensor *b = operand(L, g, 3);
	struct tensor *c = opt_operand(L, g, 5);
	struct ggml_tensor *r;
	int n_dims, mode, n_ctx_orig;

	luaL_checktype(L, 4, LUA_TTABLE);
	lua_getfield(L, 4, "n_dims");
	n_dims = lua_isnil(L, -1) ? (int)a->t->ne[0] : (int)luaL_checkinteger(L, -1);
	lua_pop(L, 1);
	lua_getfield(L, 4, "mode");
	mode = modev[luaL_checkoption(L, -1, "normal", modes)];
	lua_pop(L, 1);
	lua_getfield(L, 4, "n_ctx_orig");
	n_ctx_orig = (int)luaL_optinteger(L, -1, 0);
	lua_pop(L, 1);

	if (n_dims <= 0 || n_dims > a->t->ne[0] || n_dims % 2 != 0)
		return luaL_error(L, "n_dims must be even and at most head_dim");
	if (b->t->type != GGML_TYPE_I32 || !ggml_is_vector(b->t) || a->t->ne[2] != b->t->ne[0])
		return luaL_error(L, "pos must be i32 with one entry per token");
	if (c != NULL && (c->t->type != GGML_TYPE_F32 || c->t->ne[0] < n_dims / 2))
		return luaL_error(L, "freq must be f32 with n_dims / 2 entries");
	reserve(L, g, 1);
	r = ggml_rope_ext(g->ctx, a->t, b->t, c != NULL ? c->t : NULL, n_dims, mode, n_ctx_orig,
	    optfield(L, 4, "base", 10000.0f), optfield(L, 4, "scale", 1.0f),
	    optfield(L, 4, "ext_factor", 0.0f), optfield(L, 4, "attn_factor", 1.0f),
	    optfield(L, 4, "beta_fast", 32.0f), optfield(L, 4, "beta_slow", 1.0f));
	return push_node(L, g, r, a->rank);
}

/*** Softmax of a * scale + mask along each row.
 * The mask is f16 or f32 and contiguous. Its innermost size is that of a.
 * It has at least as many rows as a, and it broadcasts over outer
 * dimensions. `max_bias` > 0 adds ALiBi and needs a mask.
 * @tparam tensor a contiguous input
 * @tparam[opt] tensor mask additive mask
 * @tparam[opt=1] number scale factor on a
 * @tparam[opt=0] number max_bias ALiBi maximum bias
 * @treturn tensor node with the shape of a
 * @raise on a bad mask, or `max_bias` without a mask
 * @function graph:soft_max
 */
static int
graph_soft_max(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	struct tensor *m = opt_operand(L, g, 3);
	float scale = (float)luaL_optnumber(L, 4, 1.0);
	float max_bias = (float)luaL_optnumber(L, 5, 0.0);
	const struct ggml_tensor *ta = a->t;

	if (!ggml_is_contiguous(ta))
		return luaL_error(L, "tensor not contiguous; use cont");
	if (m != NULL) {
		if (m->t->type != GGML_TYPE_F16 && m->t->type != GGML_TYPE_F32)
			return luaL_error(L, "mask must be f16 or f32");
		if (!ggml_is_contiguous(m->t) || m->t->ne[0] != ta->ne[0] ||
		    m->t->ne[1] < ta->ne[1] || ta->ne[2] % m->t->ne[2] != 0 ||
		    ta->ne[3] % m->t->ne[3] != 0)
			return luaL_error(L, "mask shape mismatch");
	} else if (max_bias > 0.0f)
		return luaL_error(L, "max_bias needs a mask");
	reserve(L, g, 1);
	return push_node(L, g, ggml_soft_max_ext(g->ctx, a->t, m != NULL ? m->t : NULL,
	    scale, max_bias), a->rank);
}

/*** Fused attention: softmax(q k^T * scale + mask) v.
 * heads is a multiple of `kv_heads`. The mask is contiguous f16 of
 * (rows, kv) with rows >= tokens. `max_bias` needs a mask. softcap > 0
 * applies tanh soft capping to the scores.
 * @tparam tensor q (heads, tokens, d)
 * @tparam tensor k (`kv_heads`, kv, d)
 * @tparam tensor v (`kv_heads`, kv, d)
 * @tparam[opt] tensor mask additive f16 mask
 * @tparam[opt=1] number scale factor on the scores
 * @tparam[opt=0] number max_bias ALiBi maximum bias
 * @tparam[opt=0] number softcap score cap
 * @tparam[opt=false] boolean f32 ask the backend for f32 precision, for
 * inputs whose scores overflow f16
 * @treturn tensor (tokens, heads, d) node; note the swap of heads and tokens
 * @raise on a shape mismatch or a bad mask
 * @function graph:flash_attn
 */
static int
graph_flash_attn(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *q = operand(L, g, 2);
	struct tensor *k = operand(L, g, 3);
	struct tensor *v = operand(L, g, 4);
	struct tensor *m = opt_operand(L, g, 5);
	float scale = (float)luaL_optnumber(L, 6, 1.0);
	float max_bias = (float)luaL_optnumber(L, 7, 0.0);
	float softcap = (float)luaL_optnumber(L, 8, 0.0);
	int f32 = lua_toboolean(L, 9);
	const struct ggml_tensor *tq = q->t;
	struct ggml_tensor *r;

	if (!can_mul_mat(k->t, tq) || tq->ne[3] != k->t->ne[3] || tq->ne[3] != v->t->ne[3] ||
	    v->t->ne[1] != k->t->ne[1] || v->t->ne[2] != k->t->ne[2])
		return luaL_error(L, "flash_attn shape mismatch");
	if (m != NULL) {
		if (m->t->type != GGML_TYPE_F16 || !ggml_is_contiguous(m->t))
			return luaL_error(L, "mask must be contiguous f16");
		if (m->t->ne[0] != k->t->ne[1] || m->t->ne[1] < tq->ne[1] ||
		    tq->ne[2] % m->t->ne[2] != 0 || tq->ne[3] % m->t->ne[3] != 0)
			return luaL_error(L, "mask shape mismatch");
	} else if (max_bias > 0.0f)
		return luaL_error(L, "max_bias needs a mask");
	reserve(L, g, 1);
	r = ggml_flash_attn_ext(g->ctx, q->t, k->t, v->t, m != NULL ? m->t : NULL, scale,
	    max_bias, softcap);
	if (f32)
		ggml_prec_set_acc(r, GGML_PREC_F32);
	return push_node(L, g, r, max_int(q->rank, 3));
}

/*** Repeat a to the shape of b.
 * Each size of b is a multiple of the same size of a.
 * @tparam tensor a input
 * @tparam tensor b tensor with the target shape
 * @treturn tensor node with the shape of b
 * @raise if a does not tile to b
 * @function graph:tile
 */
static int
graph_repeat(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	struct tensor *b = operand(L, g, 3);

	if (!ggml_can_repeat(a->t, b->t))
		return luaL_error(L, "shape of arg 1 does not tile to arg 2");
	reserve(L, g, 1);
	return push_node(L, g, ggml_repeat(g->ctx, a->t, b->t), b->rank);
}

/*** Join a and b along one dimension.
 * The types match. All other sizes match.
 * @tparam tensor a first part
 * @tparam tensor b second part
 * @tparam int dim torch dimension, from 1
 * @treturn tensor node
 * @raise if dim is out of range, or the types or other sizes differ
 * @function graph:concat
 */
static int
graph_concat(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	struct tensor *b = operand(L, g, 3);
	int rank = max_int(a->rank, b->rank);
	lua_Integer dim = luaL_checkinteger(L, 4);
	int d, i;

	luaL_argcheck(L, dim >= 1 && dim <= rank, 4, "dimension out of range");
	d = rank - (int)dim;
	if (a->t->type != b->t->type)
		return luaL_error(L, "concat type mismatch");
	for (i = 0; i < GGML_MAX_DIMS; i++)
		if (i != d && a->t->ne[i] != b->t->ne[i])
			return luaL_error(L, "concat shape mismatch");
	reserve(L, g, 1);
	return push_node(L, g, ggml_concat(g->ctx, a->t, b->t, d), rank);
}

/*** Sum each row.
 * The innermost size of the result is 1.
 * @tparam tensor a input
 * @treturn tensor node with the rank of a
 * @function graph:sum_rows
 */
static int
graph_sum_rows(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);

	reserve(L, g, 1);
	return push_node(L, g, ggml_sum_rows(g->ctx, a->t), a->rank);
}

/*** Mean of each row.
 * The innermost size of the result is 1.
 * @tparam tensor a input
 * @treturn tensor node with the rank of a
 * @function graph:mean
 */
static int
graph_mean(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);

	reserve(L, g, 1);
	return push_node(L, g, ggml_mean(g->ctx, a->t), a->rank);
}

/*** Limit each element to [lo, hi].
 * @tparam tensor a input
 * @tparam number lo lower bound
 * @tparam number hi upper bound
 * @treturn tensor node with the shape of a
 * @function graph:clamp
 */
static int
graph_clamp(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	float lo = (float)luaL_checknumber(L, 3);
	float hi = (float)luaL_checknumber(L, 4);

	reserve(L, g, 1);
	return push_node(L, g, ggml_clamp(g->ctx, a->t, lo, hi), a->rank);
}

/*** Index of the largest value in each row.
 * Indices count from 0.
 * @tparam tensor a rank 1 or 2 input
 * @treturn tensor i32 node with one entry for each row
 * @raise if the rank of a is above 2
 * @function graph:argmax
 */
static int
graph_argmax(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);

	if (a->t->ne[2] != 1 || a->t->ne[3] != 1)
		return luaL_error(L, "argmax needs rank 1 or 2");
	reserve(L, g, 1);
	return push_node(L, g, ggml_argmax(g->ctx, a->t), max_int(a->rank - 1, 1));
}

/*** Indices of the k largest values in each row, in no order.
 * Indices count from 0. Use `graph:argsort_top_k` for sorted indices.
 * @tparam tensor a input
 * @tparam int k 1 to the row length
 * @treturn tensor i32 node with k entries in each row
 * @raise if k is out of range
 * @function graph:top_k
 */
static int
graph_top_k(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	struct tensor *a = operand(L, g, 2);
	lua_Integer k = luaL_checkinteger(L, 3);

	luaL_argcheck(L, k >= 1 && k <= a->t->ne[0], 3, "k out of range");
	reserve(L, g, 1);
	return push_node(L, g, ggml_top_k(g->ctx, a->t, (int)k), a->rank);
}

/* Put back the srcs that the scheduler replaced with its copies. */
static void
restore_srcs(struct graph *g)
{
	struct ggml_tensor **p;
	int i, j;

	for (i = 0; i < g->nsaved; i++) {
		p = g->saved + (size_t)i * (GGML_MAX_SRC + 1);
		for (j = 0; j < GGML_MAX_SRC; j++)
			p[0]->src[j] = p[j + 1];
	}
	g->nsaved = 0;
}

/* Keep the node srcs, which the scheduler points at its copies. */
static int
save_srcs(struct graph *g)
{
	struct ggml_tensor **p, *node;
	int i, j, n = ggml_graph_n_nodes(g->gf);

	free(g->saved);
	g->saved = malloc((size_t)n * (GGML_MAX_SRC + 1) * sizeof(*g->saved));
	if (g->saved == NULL)
		return -1;
	for (i = 0; i < n; i++) {
		node = ggml_graph_node(g->gf, i);
		p = g->saved + (size_t)i * (GGML_MAX_SRC + 1);
		p[0] = node;
		for (j = 0; j < GGML_MAX_SRC; j++)
			p[j + 1] = node->src[j];
	}
	g->nsaved = n;
	return 0;
}

/*
 * Plan a mixed graph through the scheduler. The scheduler points node
 * srcs at its copies, so keep the srcs for the next plan.
 */
static int
plan_sched(struct graph *g)
{
	ggml_backend_t bes[2] = { g->b->be, g->b->cpu->be };

	if (g->sched != NULL && g->sched_size < g->size) {
		ggml_backend_sched_free(g->sched);
		g->sched = NULL;
	}
	if (g->sched == NULL) {
		g->sched = ggml_backend_sched_new(bes, NULL, 2, (size_t)g->size, false, true);
		g->sched_size = g->size;
	}
	if (save_srcs(g) != 0)
		return -1;
	ggml_backend_sched_reset(g->sched);
	return ggml_backend_sched_alloc_graph(g->sched, g->gf) ? 0 : -1;
}

/* Set the outputs from the arguments at 2 and up. */
static void
mark_outputs(lua_State *L, struct graph *g)
{
	struct tensor *t;
	int i, n = lua_gettop(L);

	luaL_argcheck(L, n >= 2, 2, "no outputs");
	for (i = 2; i <= n; i++) {
		t = check_tensor(L, i);
		luaL_argcheck(L, t->g == g, i, "not a node of this graph");
		if (!t->out) {
			t->out = 1;
			g->nouts++;
		}
		ggml_set_output(t->t);
		ggml_build_forward_expand(g->gf, t->t);
	}
}

/* Check the ops and clear the node storage, for a new plan. */
static void
clear_plan(lua_State *L, struct graph *g)
{
	struct ggml_tensor *node;
	int i;

	restore_srcs(g);
	for (i = 0; i < ggml_graph_n_nodes(g->gf); i++) {
		node = ggml_graph_node(g->gf, i);
		if (!ggml_backend_supports_op(g->b->be, node) &&
		    !(g->mixed && ggml_backend_supports_op(g->b->cpu->be, node)))
			luaL_error(L, "%s: op %s not supported",
			    ggml_backend_name(g->b->be), ggml_op_desc(node));
	}
	/*
	 * gallocr skips tensors that have data. After an earlier compute,
	 * node data points into a buffer gallocr may free, so plan again.
	 */
	for (i = 0; i < g->nctx; i++) {
		for (node = ggml_get_first_tensor(g->ctxs[i]); node != NULL;
		    node = ggml_get_next_tensor(g->ctxs[i], node)) {
			node->data = NULL;
			node->buffer = NULL;
		}
	}
	g->plan_nodes = -1;
}

/*
 * The compute buffer sizes that a plan of the graph needs, on the
 * backend and on its CPU backend. It uses a new allocator or scheduler,
 * so the plan of the graph does not change.
 */
static void
measure_plan(struct graph *g, size_t sz[2])
{
	ggml_backend_t bes[2];
	ggml_backend_sched_t sched;
	ggml_gallocr_t ga;

	sz[0] = sz[1] = 0;
	if (!g->mixed) {
		ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g->b->be));
		if (ga != NULL) {
			ggml_gallocr_reserve_n_size(ga, g->gf, NULL, NULL, sz);
			ggml_gallocr_free(ga);
		}
		return;
	}
	bes[0] = g->b->be;
	bes[1] = g->b->cpu->be;
	if (save_srcs(g) != 0)
		return;
	sched = ggml_backend_sched_new(bes, NULL, 2, (size_t)g->size, false, true);
	if (sched != NULL) {
		ggml_backend_sched_reserve_size(sched, g->gf, sz);
		ggml_backend_sched_free(sched);
	}
	restore_srcs(g);
}

/* Count the compute buffers of the graph in the memory accounting. */
static void
account_graph(struct graph *g)
{
	ggml_backend_buffer_type_t bt[2];
	size_t sz[2] = { 0, 0 };
	int i;

	bt[0] = ggml_backend_get_default_buffer_type(g->b->be);
	bt[1] = g->b->cpu != NULL ? ggml_backend_get_default_buffer_type(g->b->cpu->be) : bt[0];
	if (g->use_sched && g->sched != NULL) {
		sz[0] = ggml_backend_sched_get_buffer_size(g->sched, g->b->be);
		sz[1] = ggml_backend_sched_get_buffer_size(g->sched, g->b->cpu->be);
	} else if (g->galloc != NULL) {
		sz[0] = ggml_gallocr_get_buffer_size(g->galloc, 0);
	}
	for (i = 0; i < 2; i++) {
		if (g->mem_bytes[i] > 0)
			lgml_mem_add(g->mem_key[i], MEM_COMPUTE, 0, g->mem_bytes[i]);
		g->mem_key[i] = lgml_mem_key(bt[i]);
		g->mem_bytes[i] = sz[i];
		if (sz[i] > 0)
			lgml_mem_add(g->mem_key[i], MEM_COMPUTE, sz[i], 0);
	}
}

/*** Run the graph.
 * The arguments become outputs. Each call runs every output of this and
 * earlier calls, and every node passed to `graph:expand`. Storage is
 * planned again only when the graph has new nodes, so a graph can run
 * many times with new leaf values at the cost of the compute alone.
 * @tparam tensor t output node of this graph
 * @tparam tensor ... more output nodes
 * @raise if a tensor is not a node of this graph, the backend does not
 * support an op, or allocation or compute fails
 * @function graph:compute
 */
static int
graph_compute(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	enum ggml_status st;
	size_t sz[2], cur;

	mark_outputs(L, g);
	if (g->plan_nodes == ggml_graph_n_nodes(g->gf) && g->plan_tensors == g->ntensors)
		goto run;
	clear_plan(L, g);
	/* The memory guard sees the growth of the buffer before it happens. */
	if (!g->mixed) {
		measure_plan(g, sz);
		cur = ggml_gallocr_get_buffer_size(g->galloc, 0);
		if (sz[0] > cur)
			lgml_mem_need(L, ggml_backend_get_default_buffer_type(g->b->be), sz[0] - cur,
			    "a graph");
	}
	g->use_sched = g->mixed;
	if (g->mixed ? plan_sched(g) != 0 : !ggml_gallocr_alloc_graph(g->galloc, g->gf))
		return luaL_error(L, "graph allocation failed");
	account_graph(g);
	g->plan_nodes = ggml_graph_n_nodes(g->gf);
	g->plan_tensors = g->ntensors;
run:
	if (g->use_sched)
		st = ggml_backend_sched_graph_compute(g->sched, g->gf);
	else
		st = ggml_backend_graph_compute(g->b->be, g->gf);
	if (st != GGML_STATUS_SUCCESS)
		return luaL_error(L, "compute: %s", ggml_status_to_string(st));
	return 0;
}

/*** Get the size of the compute buffer.
 * The buffer holds the nodes of the graph. Its size is 0 before the
 * first `graph:compute`.
 * @treturn int bytes on the backend
 * @treturn int bytes on its CPU backend, for a graph with leaves on both
 * @function graph:buffer_size
 */
static int
graph_buffer_size(lua_State *L)
{
	struct graph *g = check_graph(L, 1);

	if (g->use_sched && g->sched != NULL) {
		lua_pushinteger(L, (lua_Integer)ggml_backend_sched_get_buffer_size(g->sched, g->b->be));
		lua_pushinteger(L, (lua_Integer)ggml_backend_sched_get_buffer_size(g->sched,
		    g->b->cpu->be));
		return 2;
	}
	lua_pushinteger(L, (lua_Integer)ggml_gallocr_get_buffer_size(g->galloc, 0));
	lua_pushinteger(L, 0);
	return 2;
}

/*** Find the compute buffer sizes that the graph needs.
 * The arguments become outputs, as for `graph:compute`. It plans the
 * storage with a new allocator and allocates nothing, so a worst-case
 * graph gives the peak of its passes before they run. The next
 * `graph:compute` plans again.
 * @tparam tensor t output node of this graph
 * @tparam tensor ... more output nodes
 * @treturn int bytes on the backend
 * @treturn int bytes on its CPU backend, for a graph with leaves on both
 * @raise if a tensor is not a node of this graph, or the backend does
 * not support an op
 * @function graph:measure
 */
static int
graph_measure(lua_State *L)
{
	struct graph *g = check_graph(L, 1);
	size_t sz[2];

	mark_outputs(L, g);
	clear_plan(L, g);
	measure_plan(g, sz);
	lua_pushinteger(L, (lua_Integer)sz[0]);
	lua_pushinteger(L, (lua_Integer)sz[1]);
	return 2;
}

/*** Get the count of scheduler splits.
 * A graph with leaves on a backend and its CPU backend runs in splits,
 * each on one of the two; between splits the scheduler copies inputs.
 * @treturn int splits of the last plan; 1 for a graph on one backend,
 * 0 before the first `graph:compute`
 * @function graph:splits
 */
static int
graph_splits(lua_State *L)
{
	struct graph *g = check_graph(L, 1);

	if (g->plan_nodes < 0)
		lua_pushinteger(L, 0);
	else if (g->use_sched)
		lua_pushinteger(L, ggml_backend_sched_get_n_splits(g->sched));
	else
		lua_pushinteger(L, 1);
	return 1;
}

/* tensor */

/*** A leaf with storage, or a node of a graph.
 * A leaf comes from `backend:tensor` or `weights:get`. A node comes from a
 * graph op and gets storage when `graph:compute` runs. The rank is 1 to 4.
 * @type tensor
 */

/*** Close the tensor.
 * This releases the reference to its weights or graph.
 * The `__gc` and `__close` metamethods call this function.
 * @function tensor:close
 */
static int
tensor_close(lua_State *L)
{
	struct tensor *t = luaL_checkudata(L, 1, TENSOR_MT);

	if (t->t == NULL)
		return 0;
	if (t->g != NULL) {
		if (t->out && --t->g->nouts == 0 && t->g->closed && t->g->b != NULL)
			free_compute(t->g);
		t->g->refs--;
		graph_release(t->g);
	} else {
		t->s->refs--;
		store_release(t->s);
	}
	memset(t, 0, sizeof(*t));
	return 0;
}

static struct tensor *
check_storage(lua_State *L, int idx)
{
	struct tensor *t = check_tensor(L, idx);

	if (t->t->buffer == NULL)
		luaL_error(L, "tensor has no storage; compute it first");
	if (!ggml_is_contiguous(t->t))
		luaL_error(L, "tensor not contiguous; use cont");
	return t;
}

static int
table_type(enum ggml_type type)
{
	switch (type) {
	case GGML_TYPE_F32:
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16:
	case GGML_TYPE_I32:
	case GGML_TYPE_I64:
		return 1;
	default:
		return 0;
	}
}

/*** Write the contents.
 * data is a string of raw bytes, or a flat table of numbers in row-major
 * order. Tables work for f32, f16, bf16, i32 and i64.
 * @tparam string|table data nbytes bytes, or one number for each element
 * @raise if the tensor has no storage or is not contiguous, or the data
 * size is wrong
 * @function tensor:set
 */
static int
tensor_set(lua_State *L)
{
	struct tensor *t = check_storage(L, 1);
	enum ggml_type type = t->t->type;
	size_t len, nbytes = ggml_nbytes(t->t);
	int64_t i, n = ggml_nelements(t->t);
	const char *s;
	void *p;

	if (lua_type(L, 2) == LUA_TSTRING) {
		s = lua_tolstring(L, 2, &len);
		if (len != nbytes)
			return luaL_error(L, "need %d bytes, got %d", (int)nbytes, (int)len);
		ggml_backend_tensor_set(t->t, s, 0, nbytes);
		return 0;
	}
	luaL_checktype(L, 2, LUA_TTABLE);
	if (!table_type(type))
		return luaL_error(L, "table data not supported for %s, use a string",
		    ggml_type_name(type));
	if (luaL_len(L, 2) != n)
		return luaL_error(L, "need %d elements, got %d", (int)n, (int)luaL_len(L, 2));
	p = malloc((size_t)n * 8);
	if (p == NULL)
		return luaL_error(L, "out of memory");
	for (i = 0; i < n; i++) {
		lua_geti(L, 2, i + 1);
		if (type == GGML_TYPE_I32)
			((int32_t *)p)[i] = (int32_t)lua_tointeger(L, -1);
		else if (type == GGML_TYPE_I64)
			((int64_t *)p)[i] = (int64_t)lua_tointeger(L, -1);
		else
			((float *)p)[i] = (float)lua_tonumber(L, -1);
		lua_pop(L, 1);
	}
	/* The row converters write a narrower type, so in place is safe. */
	if (type == GGML_TYPE_F16)
		ggml_fp32_to_fp16_row(p, p, n);
	else if (type == GGML_TYPE_BF16)
		ggml_fp32_to_bf16_row(p, p, n);
	ggml_backend_tensor_set(t->t, p, 0, nbytes);
	free(p);
	return 0;
}

/*** Read the contents as a flat table in row-major order.
 * Works for f32, f16, bf16, i32 and i64. Use `tensor:bytes` for others.
 * @treturn {number,...} one value for each element
 * @raise if the tensor has no storage, is not contiguous, or has another type
 * @function tensor:get
 */
static int
tensor_get(lua_State *L)
{
	struct tensor *t = check_storage(L, 1);
	enum ggml_type type = t->t->type;
	int64_t i, n = ggml_nelements(t->t);
	float *f;
	void *p;

	if (!table_type(type))
		return luaL_error(L, "get not supported for %s, use bytes", ggml_type_name(type));
	p = malloc((size_t)n * 8);
	f = malloc((size_t)n * sizeof(*f));
	if (p == NULL || f == NULL) {
		free(p);
		free(f);
		return luaL_error(L, "out of memory");
	}
	ggml_backend_tensor_get(t->t, p, 0, ggml_nbytes(t->t));
	if (type == GGML_TYPE_F16)
		ggml_fp16_to_fp32_row(p, f, n);
	else if (type == GGML_TYPE_BF16)
		ggml_bf16_to_fp32_row(p, f, n);
	else if (type == GGML_TYPE_F32)
		memcpy(f, p, (size_t)n * sizeof(*f));
	lua_createtable(L, (int)n, 0);
	for (i = 0; i < n; i++) {
		if (type == GGML_TYPE_I32)
			lua_pushinteger(L, ((int32_t *)p)[i]);
		else if (type == GGML_TYPE_I64)
			lua_pushinteger(L, ((int64_t *)p)[i]);
		else
			lua_pushnumber(L, f[i]);
		lua_seti(L, -2, i + 1);
	}
	free(p);
	free(f);
	return 1;
}

/* Leaf with storage and at most 2 dimensions. */
static struct tensor *
check_leaf2d(lua_State *L, int idx)
{
	struct tensor *t = check_storage(L, idx);

	if (t->s == NULL)
		luaL_argerror(L, idx, "need a leaf tensor");
	if (t->t->ne[2] != 1 || t->t->ne[3] != 1)
		luaL_argerror(L, idx, "need rank 1 or 2");
	return t;
}

/*** Fill a causal attention mask.
 * The tensor is an f32 or f16 leaf of (rows, `n_kv`). Row i < n is query
 * `n_past` + i. It sees columns 0 to `n_past` + i. All other entries are -inf.
 * @tparam int n_past number of earlier positions
 * @tparam[opt=rows] int n number of query rows, 1 to rows
 * @raise on a bad tensor, or if `n_past` + n is over `n_kv`
 * @function tensor:fill_causal
 */
static int
tensor_fill_causal(lua_State *L)
{
	struct tensor *t = check_leaf2d(L, 1);
	lua_Integer n_past = luaL_checkinteger(L, 2);
	int64_t nkv = t->t->ne[0], rows = t->t->ne[1];
	lua_Integer n = luaL_optinteger(L, 3, rows);
	int64_t i, j;
	float *p;

	if (t->t->type != GGML_TYPE_F32 && t->t->type != GGML_TYPE_F16)
		return luaL_error(L, "mask must be f32 or f16");
	luaL_argcheck(L, n_past >= 0, 2, "n_past must not be negative");
	luaL_argcheck(L, n >= 1 && n <= rows, 3, "n out of range");
	if (n_past + n > nkv)
		return luaL_error(L, "n_past + n exceeds mask width");
	p = malloc((size_t)(nkv * rows) * sizeof(*p));
	if (p == NULL)
		return luaL_error(L, "out of memory");
	for (i = 0; i < rows; i++)
		for (j = 0; j < nkv; j++)
			p[i * nkv + j] = i < n && j <= n_past + i ? 0.0f : -INFINITY;
	if (t->t->type == GGML_TYPE_F16)
		ggml_fp32_to_fp16_row(p, (ggml_fp16_t *)p, nkv * rows);
	ggml_backend_tensor_set(t->t, p, 0, ggml_nbytes(t->t));
	free(p);
	return 0;
}

/*** Fill an attention mask with options.
 * The tensor is an f32 or f16 leaf of (rows, `n_kv`). Row i is the query at
 * `n_past` + i. Allowed entries are 0 and others are -inf. opts fields:
 * `n_past` (0), causal (true), window (0 is none), canvas, ring (false).
 * Queries at or past canvas see all keys (with a window, only keys >=
 * canvas - window + 1). Other queries never see canvas keys. No query
 * sees keys at or past `n_past` + rows. With ring, column j holds the
 * latest key p < `n_past` + rows with p % `n_kv` == j.
 * @tparam table opts options
 * @raise on a bad tensor or out-of-range options
 * @function tensor:fill_mask
 */
static int
tensor_fill_mask(lua_State *L)
{
	struct tensor *t = check_leaf2d(L, 1);
	int64_t nkv = t->t->ne[0], rows = t->t->ne[1], i, j, kp, qp;
	lua_Integer n_past, window, canvas;
	int causal, ring, allow;
	float *p;

	luaL_checktype(L, 2, LUA_TTABLE);
	lua_getfield(L, 2, "n_past");
	n_past = luaL_optinteger(L, -1, 0);
	lua_getfield(L, 2, "window");
	window = luaL_optinteger(L, -1, 0);
	lua_getfield(L, 2, "canvas");
	canvas = luaL_optinteger(L, -1, -1);
	lua_getfield(L, 2, "causal");
	causal = lua_isnil(L, -1) || lua_toboolean(L, -1);
	lua_getfield(L, 2, "ring");
	ring = lua_toboolean(L, -1);
	lua_pop(L, 5);
	if (t->t->type != GGML_TYPE_F32 && t->t->type != GGML_TYPE_F16)
		return luaL_error(L, "mask must be f32 or f16");
	if (n_past < 0 || window < 0 || canvas < -1 || n_past > INT32_MAX ||
	    window > INT32_MAX || canvas > INT32_MAX)
		return luaL_error(L, "fill_mask: n_past, window or canvas out of range");
	p = malloc((size_t)(nkv * rows) * sizeof(*p));
	if (p == NULL)
		return luaL_error(L, "out of memory");
	for (i = 0; i < rows; i++) {
		qp = n_past + i;
		for (j = 0; j < nkv; j++) {
			kp = j;
			if (ring && j < n_past + rows)
				kp = j + (n_past + rows - 1 - j) / nkv * nkv;
			if (kp >= n_past + rows)
				allow = 0;
			else if (canvas >= 0 && qp >= canvas)
				allow = window == 0 || kp >= canvas || kp >= canvas - window + 1;
			else
				allow = (!causal || kp <= qp) && (canvas < 0 || kp < canvas) &&
				    (window == 0 || (kp <= qp ? qp - kp : kp - qp) < window);
			p[i * nkv + j] = allow ? 0.0f : -INFINITY;
		}
	}
	if (t->t->type == GGML_TYPE_F16)
		ggml_fp32_to_fp16_row(p, (ggml_fp16_t *)p, nkv * rows);
	ggml_backend_tensor_set(t->t, p, 0, ggml_nbytes(t->t));
	free(p);
	return 0;
}

/*** Fill the attention mask of one sequence in a shared KV pool.
 * The tensor is an f32 or f16 leaf of (rows, `n_kv`). Column j is pool
 * cell `lo` + j. Position p of the sequence is in cell
 * `blocks[p // block + 1] * block + p % block`. Row i is the query at
 * `n_past` + i. It sees the cells of positions 0 to `n_past` + i, or of
 * all positions before `n_past` + rows when causal is false. All other
 * entries are -inf. opts fields: `n_past` (0), `block`, `blocks` (block
 * ids from 0, in position order), `lo` (0), causal (true).
 * @tparam table opts options
 * @raise on a bad tensor, too few blocks, or a cell outside the columns
 * @function tensor:fill_blocks
 */
static int
tensor_fill_blocks(lua_State *L)
{
	struct tensor *t = check_leaf2d(L, 1);
	int64_t nkv = t->t->ne[0], rows = t->t->ne[1], i, p, last, col, nb, need;
	lua_Integer n_past, block, lo;
	int64_t *cells;
	int causal;
	float *m;

	luaL_checktype(L, 2, LUA_TTABLE);
	lua_getfield(L, 2, "n_past");
	n_past = luaL_optinteger(L, -1, 0);
	lua_getfield(L, 2, "block");
	block = luaL_checkinteger(L, -1);
	lua_getfield(L, 2, "lo");
	lo = luaL_optinteger(L, -1, 0);
	lua_getfield(L, 2, "causal");
	causal = lua_isnil(L, -1) || lua_toboolean(L, -1);
	lua_pop(L, 4);
	if (t->t->type != GGML_TYPE_F32 && t->t->type != GGML_TYPE_F16)
		return luaL_error(L, "mask must be f32 or f16");
	if (n_past < 0 || block < 1 || lo < 0 || n_past > INT32_MAX)
		return luaL_error(L, "fill_blocks: n_past, block or lo out of range");
	lua_getfield(L, 2, "blocks");
	luaL_checktype(L, -1, LUA_TTABLE);
	nb = (int64_t)luaL_len(L, -1);
	need = (n_past + rows + block - 1) / block;
	if (nb < need)
		return luaL_error(L, "fill_blocks: %I blocks hold fewer than %I positions",
		    (lua_Integer)nb, (lua_Integer)(n_past + rows));
	cells = malloc((size_t)(need > 0 ? need : 1) * sizeof(*cells));
	m = malloc((size_t)(nkv * rows) * sizeof(*m));
	if (cells == NULL || m == NULL) {
		free(cells);
		free(m);
		return luaL_error(L, "out of memory");
	}
	for (i = 0; i < need; i++) {
		lua_geti(L, -1, i + 1);
		cells[i] = (int64_t)lua_tointeger(L, -1) * block - lo;
		lua_pop(L, 1);
	}
	lua_pop(L, 1);
	for (i = 0; i < nkv; i++)
		m[i] = -INFINITY;
	/* Each row sees what the row before it sees, and more. */
	for (i = 0, p = 0; i < rows; i++) {
		if (i > 0)
			memcpy(m + i * nkv, m + (i - 1) * nkv, (size_t)nkv * sizeof(*m));
		last = causal ? n_past + i : n_past + rows - 1;
		for (; p <= last; p++) {
			col = cells[p / block] + p % block;
			if (col < 0 || col >= nkv) {
				free(cells);
				free(m);
				return luaL_error(L, "fill_blocks: position %I is in no column",
				    (lua_Integer)p);
			}
			m[i * nkv + col] = 0.0f;
		}
	}
	free(cells);
	if (t->t->type == GGML_TYPE_F16)
		ggml_fp32_to_fp16_row(m, (ggml_fp16_t *)m, nkv * rows);
	ggml_backend_tensor_set(t->t, m, 0, ggml_nbytes(t->t));
	free(m);
	return 0;
}

/*** Set every byte of a leaf to zero, on the device.
 * Zero is 0.0 in float types and a zero block in quantized types.
 * @raise if the tensor has no storage
 * @function tensor:zero
 */
static int
tensor_zero(lua_State *L)
{
	struct tensor *t = check_storage(L, 1);

	ggml_backend_tensor_memset(t->t, 0, 0, ggml_nbytes(t->t));
	return 0;
}

/*** Fill an integer leaf: element i gets start + i * step.
 * @tparam int start first value
 * @tparam[opt=1] int step increment
 * @raise unless the tensor is an i32 or i64 leaf of rank 1 or 2
 * @function tensor:fill_range
 */
static int
tensor_fill_range(lua_State *L)
{
	struct tensor *t = check_leaf2d(L, 1);
	lua_Integer start = luaL_checkinteger(L, 2);
	lua_Integer step = luaL_optinteger(L, 3, 1);
	int64_t i, n = ggml_nelements(t->t);
	void *p;

	if (t->t->type != GGML_TYPE_I32 && t->t->type != GGML_TYPE_I64)
		return luaL_error(L, "fill_range needs i32 or i64");
	p = malloc((size_t)n * sizeof(int64_t));
	if (p == NULL)
		return luaL_error(L, "out of memory");
	for (i = 0; i < n; i++) {
		if (t->t->type == GGML_TYPE_I32)
			((int32_t *)p)[i] = (int32_t)(start + i * step);
		else
			((int64_t *)p)[i] = (int64_t)(start + i * step);
	}
	ggml_backend_tensor_set(t->t, p, 0, ggml_nbytes(t->t));
	free(p);
	return 0;
}

/* splitmix64: small, seedable, good enough for test data and init. */
static uint64_t
splitmix64(uint64_t *x)
{
	uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);

	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

static double
unit(uint64_t *x)
{
	return (double)(splitmix64(x) >> 11) * (1.0 / 9007199254740992.0);
}

/*** Fill with constant, range, random or Hadamard values.
 * kind "const" gives a, "range" gives a + i * b, "uniform" draws from a to
 * b and "normal" draws with mean a and standard deviation b. "hadamard"
 * makes the orthonormal Walsh-Hadamard matrix of a square f32 tensor
 * whose size is a power of 2. The values are made in C and copied to
 * the device once.
 * @tparam string kind "const", "range", "uniform", "normal" or "hadamard"
 * @tparam[opt=0] number a first parameter
 * @tparam[opt=1] number b second parameter
 * @tparam[opt=0x5eed] int seed random seed
 * @raise if the tensor has no storage, is not contiguous, or is not f32,
 * f16, bf16, i32 or i64
 * @function tensor:fill
 */
static int
tensor_fill(lua_State *L)
{
	static const char *const kinds[] = { "const", "range", "uniform", "normal", "hadamard",
	    NULL };
	struct tensor *t = check_storage(L, 1);
	int kind = luaL_checkoption(L, 2, NULL, kinds);
	double a = luaL_optnumber(L, 3, 0), b = luaL_optnumber(L, 4, 1), v, u1, u2;
	uint64_t seed = (uint64_t)luaL_optinteger(L, 5, 0x5eed);
	enum ggml_type type = t->t->type;
	int64_t i, n = ggml_nelements(t->t);
	float *f;
	void *p;

	int64_t side = t->t->ne[0];

	if (!table_type(type))
		return luaL_error(L, "fill not supported for %s", ggml_type_name(type));
	if (kind == 4 && (type != GGML_TYPE_F32 || t->t->ne[1] != side || t->t->ne[2] != 1 ||
	    t->t->ne[3] != 1 || (side & (side - 1)) != 0))
		return luaL_error(L, "hadamard needs a square f32 tensor with a power of 2 side");
	p = malloc((size_t)n * 8);
	f = malloc((size_t)n * sizeof(*f));
	if (p == NULL || f == NULL) {
		free(p);
		free(f);
		return luaL_error(L, "out of memory");
	}
	for (i = 0; i < n; i++) {
		if (kind == 0)
			v = a;
		else if (kind == 1)
			v = a + (double)i * b;
		else if (kind == 2)
			v = a + (b - a) * unit(&seed);
		else if (kind == 4)
			v = (__builtin_popcountll((uint64_t)(i / side & i % side)) & 1 ? -1.0 : 1.0) /
			    sqrt((double)side);
		else {
			u1 = unit(&seed);
			u2 = unit(&seed);
			v = a + b * sqrt(-2.0 * log(u1 > 0 ? u1 : 1e-300)) * cos(6.283185307179586 * u2);
		}
		if (type == GGML_TYPE_I32)
			((int32_t *)p)[i] = (int32_t)v;
		else if (type == GGML_TYPE_I64)
			((int64_t *)p)[i] = (int64_t)v;
		else
			f[i] = (float)v;
	}
	if (type == GGML_TYPE_F32)
		memcpy(p, f, (size_t)n * sizeof(*f));
	else if (type == GGML_TYPE_F16)
		ggml_fp32_to_fp16_row(f, p, n);
	else if (type == GGML_TYPE_BF16)
		ggml_fp32_to_bf16_row(f, p, n);
	ggml_backend_tensor_set(t->t, p, 0, ggml_nbytes(t->t));
	free(p);
	free(f);
	return 0;
}

/*** Read the raw contents.
 * @treturn string nbytes bytes
 * @raise if the tensor has no storage or is not contiguous
 * @function tensor:bytes
 */
static int
tensor_bytes(lua_State *L)
{
	struct tensor *t = check_storage(L, 1);
	size_t nbytes = ggml_nbytes(t->t);
	luaL_Buffer lb;

	ggml_backend_tensor_get(t->t, luaL_buffinitsize(L, &lb, nbytes), 0, nbytes);
	luaL_pushresultsize(&lb, nbytes);
	return 1;
}

/* Checked byte range of a tensor and a host buffer, from args 2 to 5. */
static void
host_range(lua_State *L, struct tensor *t, struct hostbuf **hb, size_t *hoff, size_t *off,
    size_t *n)
{
	size_t nbytes = ggml_nbytes(t->t);
	lua_Integer a, b, c;

	*hb = check_hostbuf(L, 2);
	a = luaL_optinteger(L, 3, 0);
	b = luaL_optinteger(L, 4, 0);
	luaL_argcheck(L, a >= 0 && (lua_Unsigned)a <= (*hb)->size, 3, "offset out of range");
	luaL_argcheck(L, b >= 0 && (lua_Unsigned)b <= nbytes, 4, "offset out of range");
	c = luaL_optinteger(L, 5, (lua_Integer)(nbytes - (size_t)b));
	luaL_argcheck(L, c >= 0 && (lua_Unsigned)c <= nbytes - (size_t)b &&
	    (lua_Unsigned)c <= (*hb)->size - (size_t)a, 5, "count out of range");
	*hoff = (size_t)a;
	*off = (size_t)b;
	*n = (size_t)c;
}

/*** Copy bytes of the contents into a host buffer.
 * @tparam hostbuf buf the host buffer
 * @tparam[opt=0] int buf_off first byte in buf
 * @tparam[opt=0] int off first byte of the tensor
 * @tparam[opt] int n bytes to copy (default: from off to the end)
 * @raise if the tensor has no storage or is not contiguous, or a range
 * is out of bounds
 * @function tensor:to_host
 */
static int
tensor_to_host(lua_State *L)
{
	struct tensor *t = check_storage(L, 1);
	struct hostbuf *hb;
	size_t hoff, off, n;

	host_range(L, t, &hb, &hoff, &off, &n);
	if (n > 0)
		ggml_backend_tensor_get(t->t, hb->p + hoff, off, n);
	return 0;
}

/*** Copy bytes from a host buffer into the contents.
 * @tparam hostbuf buf the host buffer
 * @tparam[opt=0] int buf_off first byte in buf
 * @tparam[opt=0] int off first byte of the tensor
 * @tparam[opt] int n bytes to copy (default: from off to the end)
 * @raise if the tensor has no storage or is not contiguous, or a range
 * is out of bounds
 * @function tensor:from_host
 */
static int
tensor_from_host(lua_State *L)
{
	struct tensor *t = check_storage(L, 1);
	struct hostbuf *hb;
	size_t hoff, off, n;

	host_range(L, t, &hb, &hoff, &off, &n);
	if (n > 0)
		ggml_backend_tensor_set(t->t, hb->p + hoff, off, n);
	return 0;
}

/*** Get the shape in torch order.
 * @treturn {int,...} one size for each dimension
 * @function tensor:shape
 */
static int
tensor_shape(lua_State *L)
{
	struct tensor *t = check_tensor(L, 1);
	int i, n = t->rank;

	lua_createtable(L, n, 0);
	for (i = 0; i < n; i++) {
		lua_pushinteger(L, t->t->ne[n - 1 - i]);
		lua_seti(L, -2, i + 1);
	}
	return 1;
}

/*** Get the strides in bytes, in torch order.
 * @treturn {int,...} one stride for each dimension
 * @function tensor:strides
 */
static int
tensor_strides(lua_State *L)
{
	struct tensor *t = check_tensor(L, 1);
	int i, n = t->rank;

	lua_createtable(L, n, 0);
	for (i = 0; i < n; i++) {
		lua_pushinteger(L, (lua_Integer)t->t->nb[n - 1 - i]);
		lua_seti(L, -2, i + 1);
	}
	return 1;
}

/*** Get the element type.
 * @treturn string ggml type name, such as "f32"
 * @function tensor:type
 */
static int
tensor_type(lua_State *L)
{
	struct tensor *t = check_tensor(L, 1);

	lua_pushstring(L, ggml_type_name(t->t->type));
	return 1;
}

/*** Get the size of the data.
 * @treturn int bytes
 * @function tensor:nbytes
 */
static int
tensor_nbytes(lua_State *L)
{
	struct tensor *t = check_tensor(L, 1);

	lua_pushinteger(L, (lua_Integer)ggml_nbytes(t->t));
	return 1;
}

/*** Get or set the name.
 * With an argument, set the name and return nothing.
 * @tparam[opt] string name new name
 * @treturn string the name, when called without an argument
 * @function tensor:name
 */
static int
tensor_name(lua_State *L)
{
	struct tensor *t = check_tensor(L, 1);

	if (!lua_isnoneornil(L, 2)) {
		ggml_set_name(t->t, luaL_checkstring(L, 2));
		return 0;
	}
	lua_pushstring(L, ggml_get_name(t->t));
	return 1;
}

/*** Get the backend of the tensor.
 * @treturn backend the backend of its weights or graph
 * @function tensor:backend
 */
static int
tensor_backend(lua_State *L)
{
	check_tensor(L, 1);
	lua_getiuservalue(L, 1, UV_PARENT);
	lua_getiuservalue(L, -1, UV_PARENT);
	return 1;
}

/*** Describe the tensor, as "tensor TYPE OP (SHAPE)".
 * @treturn string description
 * @function tensor:__tostring
 */
static int
tensor_tostring(lua_State *L)
{
	struct tensor *t = luaL_checkudata(L, 1, TENSOR_MT);
	luaL_Buffer lb;
	int i, n;

	if (t->t == NULL) {
		lua_pushliteral(L, "tensor (closed)");
		return 1;
	}
	n = t->rank;
	luaL_buffinit(L, &lb);
	lua_pushfstring(L, "tensor %s %s (", ggml_type_name(t->t->type), ggml_op_desc(t->t));
	luaL_addvalue(&lb);
	for (i = 0; i < n; i++) {
		lua_pushfstring(L, i == 0 ? "%d" : ", %d", (int)t->t->ne[n - 1 - i]);
		luaL_addvalue(&lb);
	}
	luaL_addchar(&lb, ')');
	luaL_pushresult(&lb);
	return 1;
}

/* gguf metadata */

static size_t
gguf_scalar_size(enum gguf_type type)
{
	switch (type) {
	case GGUF_TYPE_UINT8:
	case GGUF_TYPE_INT8:
	case GGUF_TYPE_BOOL:
		return 1;
	case GGUF_TYPE_UINT16:
	case GGUF_TYPE_INT16:
		return 2;
	case GGUF_TYPE_UINT32:
	case GGUF_TYPE_INT32:
	case GGUF_TYPE_FLOAT32:
		return 4;
	case GGUF_TYPE_UINT64:
	case GGUF_TYPE_INT64:
	case GGUF_TYPE_FLOAT64:
		return 8;
	default:
		return 0;
	}
}

static void
push_gguf_scalar(lua_State *L, enum gguf_type type, const void *p)
{
	switch (type) {
	case GGUF_TYPE_UINT8:
		lua_pushinteger(L, *(const uint8_t *)p);
		break;
	case GGUF_TYPE_INT8:
		lua_pushinteger(L, *(const int8_t *)p);
		break;
	case GGUF_TYPE_UINT16:
		lua_pushinteger(L, *(const uint16_t *)p);
		break;
	case GGUF_TYPE_INT16:
		lua_pushinteger(L, *(const int16_t *)p);
		break;
	case GGUF_TYPE_UINT32:
		lua_pushinteger(L, *(const uint32_t *)p);
		break;
	case GGUF_TYPE_INT32:
		lua_pushinteger(L, *(const int32_t *)p);
		break;
	case GGUF_TYPE_UINT64:
		lua_pushinteger(L, (lua_Integer)*(const uint64_t *)p);
		break;
	case GGUF_TYPE_INT64:
		lua_pushinteger(L, *(const int64_t *)p);
		break;
	case GGUF_TYPE_FLOAT32:
		lua_pushnumber(L, *(const float *)p);
		break;
	case GGUF_TYPE_FLOAT64:
		lua_pushnumber(L, *(const double *)p);
		break;
	case GGUF_TYPE_BOOL:
		lua_pushboolean(L, *(const int8_t *)p != 0);
		break;
	default:
		lua_pushnil(L);
		break;
	}
}

static void
push_gguf_value(lua_State *L, const struct gguf_context *c, int64_t id)
{
	enum gguf_type type = gguf_get_kv_type(c, id), at;
	const char *data;
	size_t i, n, sz;

	if (type == GGUF_TYPE_STRING) {
		lua_pushstring(L, gguf_get_val_str(c, id));
		return;
	}
	if (type != GGUF_TYPE_ARRAY) {
		push_gguf_scalar(L, type, gguf_get_val_data(c, id));
		return;
	}
	at = gguf_get_arr_type(c, id);
	n = gguf_get_arr_n(c, id);
	lua_createtable(L, n > INT32_MAX ? 0 : (int)n, 0);
	if (at == GGUF_TYPE_STRING) {
		for (i = 0; i < n; i++) {
			lua_pushstring(L, gguf_get_arr_str(c, id, i));
			lua_seti(L, -2, (lua_Integer)i + 1);
		}
		return;
	}
	sz = gguf_scalar_size(at);
	if (sz == 0)
		return;
	data = gguf_get_arr_data(c, id);
	for (i = 0; i < n; i++) {
		push_gguf_scalar(L, at, data + i * sz);
		lua_seti(L, -2, (lua_Integer)i + 1);
	}
}

/***
 * @section end
 */

/*** Read the metadata of a gguf file.
 * The result is { version, alignment, `data_offset`, meta = { key = value },
 * tensors = { { name, type, shape, offset, size }, ... } }. Shapes drop
 * outer dimensions of size 1.
 * @tparam string path gguf file
 * @treturn[1] table metadata
 * @return[2] nil if the file is not a gguf file
 * @treturn[2] string error message
 * @function gguf
 */
static int
l_gguf(lua_State *L)
{
	const char *path = luaL_checkstring(L, 1);
	struct gguf_init_params gp = { true, NULL };
	struct gguf_context *c;
	const int64_t *ne;
	int64_t i, j, n;
	int rank;

	c = gguf_init_from_file(path, gp);
	if (c == NULL) {
		luaL_pushfail(L);
		lua_pushfstring(L, "%s: not a gguf file", path);
		return 2;
	}
	lua_createtable(L, 0, 5);
	lua_pushinteger(L, gguf_get_version(c));
	lua_setfield(L, -2, "version");
	lua_pushinteger(L, (lua_Integer)gguf_get_alignment(c));
	lua_setfield(L, -2, "alignment");
	lua_pushinteger(L, (lua_Integer)gguf_get_data_offset(c));
	lua_setfield(L, -2, "data_offset");

	n = gguf_get_n_kv(c);
	lua_createtable(L, 0, (int)n);
	for (i = 0; i < n; i++) {
		push_gguf_value(L, c, i);
		lua_setfield(L, -2, gguf_get_key(c, i));
	}
	lua_setfield(L, -2, "meta");

	n = gguf_get_n_tensors(c);
	lua_createtable(L, (int)n, 0);
	for (i = 0; i < n; i++) {
		lua_createtable(L, 0, 5);
		lua_pushstring(L, gguf_get_tensor_name(c, i));
		lua_setfield(L, -2, "name");
		lua_pushstring(L, ggml_type_name(gguf_get_tensor_type(c, i)));
		lua_setfield(L, -2, "type");
		ne = gguf_get_tensor_ne(c, i);
		for (rank = GGML_MAX_DIMS; rank > 1 && ne[rank - 1] == 1; rank--)
			;
		lua_createtable(L, rank, 0);
		for (j = 0; j < rank; j++) {
			lua_pushinteger(L, ne[rank - 1 - j]);
			lua_seti(L, -2, j + 1);
		}
		lua_setfield(L, -2, "shape");
		lua_pushinteger(L, (lua_Integer)gguf_get_tensor_offset(c, i));
		lua_setfield(L, -2, "offset");
		lua_pushinteger(L, (lua_Integer)gguf_get_tensor_size(c, i));
		lua_setfield(L, -2, "size");
		lua_seti(L, -2, i + 1);
	}
	lua_setfield(L, -2, "tensors");
	gguf_free(c);
	return 1;
}

/* module */

static const char *const dev_types[] = { "cpu", "gpu", "igpu", "accel", "meta" };

/*** List the ggml devices.
 * Each entry is { name, description, type, free, total, id }. type is
 * "cpu", "gpu", "igpu", "accel", "meta" or "unknown". free and total are
 * bytes. id is the bus id of the device, such as "0000:03:00.0", when
 * the backend knows it.
 * @treturn {table,...} devices
 * @function devices
 */
static int
l_devices(lua_State *L)
{
	struct ggml_backend_dev_props props;
	ggml_backend_dev_t dev;
	size_t i, n = ggml_backend_dev_count();
	size_t mfree, mtotal;
	unsigned int ty;

	lua_createtable(L, (int)n, 0);
	for (i = 0; i < n; i++) {
		dev = ggml_backend_dev_get(i);
		ggml_backend_dev_memory(dev, &mfree, &mtotal);
		ty = (unsigned int)ggml_backend_dev_type(dev);
		lua_createtable(L, 0, 5);
		lua_pushstring(L, ggml_backend_dev_name(dev));
		lua_setfield(L, -2, "name");
		lua_pushstring(L, ggml_backend_dev_description(dev));
		lua_setfield(L, -2, "description");
		lua_pushstring(L, ty < sizeof(dev_types) / sizeof(dev_types[0]) ?
		    dev_types[ty] : "unknown");
		lua_setfield(L, -2, "type");
		lua_pushinteger(L, (lua_Integer)mfree);
		lua_setfield(L, -2, "free");
		lua_pushinteger(L, (lua_Integer)mtotal);
		lua_setfield(L, -2, "total");
		memset(&props, 0, sizeof(props));
		ggml_backend_dev_get_props(dev, &props);
		if (props.device_id != NULL && props.device_id[0] != '\0') {
			lua_pushstring(L, props.device_id);
			lua_setfield(L, -2, "id");
		}
		lua_seti(L, -2, (lua_Integer)i + 1);
	}
	return 1;
}

/*** Set the lowest ggml log level that prints to stderr.
 * The default is "warn".
 * @tparam string level "debug", "info", "warn", "error" or "none"
 * @function log
 */
static int
l_log(lua_State *L)
{
	static const char *const names[] = { "none", "debug", "info", "warn", "error", NULL };
	int lv = luaL_checkoption(L, 1, NULL, names);

	log_level = lv == 0 ? GGML_LOG_LEVEL_CONT + 1 : lv;
	return 0;
}

/*** Get the bytes for n elements of a type.
 * @tparam string type ggml type name
 * @tparam int n element count, a whole number of blocks
 * @treturn int bytes
 * @raise if the type is unknown or n is not a whole number of blocks
 * @function row_size
 */
static int
l_row_size(lua_State *L)
{
	enum ggml_type type = check_type(L, 1);
	lua_Integer n = luaL_checkinteger(L, 2);

	luaL_argcheck(L, n >= 0 && n % ggml_blck_size(type) == 0, 2, "not a whole block");
	lua_pushinteger(L, (lua_Integer)ggml_row_size(type, n));
	return 1;
}

/*** Get monotonic time.
 * @treturn number seconds
 * @function clock
 */
static int
l_clock(lua_State *L)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	lua_pushnumber(L, (lua_Number)ts.tv_sec + (lua_Number)ts.tv_nsec / 1e9);
	return 1;
}

/* host buffers */

/*** Allocate a host buffer.
 * The memory is not cleared. `tensor:to_host` and `tensor:from_host`
 * copy byte ranges between it and a tensor. Offsets are from 0, as
 * in `graph:view`. Lua does not see this memory, so the call first
 * runs a collector step sized for it.
 * @tparam int nbytes the size in bytes
 * @treturn hostbuf the buffer
 * @raise if nbytes is negative or memory runs out
 * @function hostbuf
 */
static int
l_hostbuf(lua_State *L)
{
	lua_Integer n = luaL_checkinteger(L, 1);
	struct hostbuf *hb;

	luaL_argcheck(L, n >= 0 && (lua_Unsigned)n <= SIZE_MAX / 2, 1, "size out of range");
	lua_gc(L, LUA_GCSTEP, n >> 10 < INT_MAX ? (int)(n >> 10) : INT_MAX);
	hb = lua_newuserdatauv(L, sizeof(*hb), 0);
	hb->p = NULL;
	hb->size = 0;
	luaL_setmetatable(L, HOSTBUF_MT);
	hb->p = malloc(n > 0 ? (size_t)n : 1);
	if (hb->p == NULL)
		return luaL_error(L, "hostbuf: out of memory for %I bytes", n);
	hb->size = (size_t)n;
	return 1;
}

/*** Host memory for copies of tensor contents.
 * Garbage collection frees it; `hostbuf:close` frees it at once.
 * @type hostbuf
 */

/*** Free the memory. Later calls raise an error.
 * The `__gc` and `__close` metamethods call this function.
 * @function hostbuf:close
 */
static int
hostbuf_close(lua_State *L)
{
	struct hostbuf *hb = luaL_checkudata(L, 1, HOSTBUF_MT);

	free(hb->p);
	hb->p = NULL;
	hb->size = 0;
	return 0;
}

/*** Get the size. The `#` operator gives the same.
 * @treturn int bytes
 * @function hostbuf:len
 */
static int
hostbuf_len(lua_State *L)
{
	lua_pushinteger(L, (lua_Integer)check_hostbuf(L, 1)->size);
	return 1;
}

/*** Read bytes as a string.
 * @tparam[opt=0] int off first byte
 * @tparam[opt] int n bytes to read (default: from off to the end)
 * @treturn string the bytes
 * @raise if the range is out of bounds
 * @function hostbuf:bytes
 */
static int
hostbuf_bytes(lua_State *L)
{
	struct hostbuf *hb = check_hostbuf(L, 1);
	lua_Integer off = luaL_optinteger(L, 2, 0), n;

	luaL_argcheck(L, off >= 0 && (lua_Unsigned)off <= hb->size, 2, "offset out of range");
	n = luaL_optinteger(L, 3, (lua_Integer)(hb->size - (size_t)off));
	luaL_argcheck(L, n >= 0 && (lua_Unsigned)n <= hb->size - (size_t)off, 3,
	    "count out of range");
	lua_pushlstring(L, hb->p + off, (size_t)n);
	return 1;
}

static const luaL_Reg backend_meta[] = {
	{ "__gc", backend_close },
	{ "__close", backend_close },
	{ NULL, NULL }
};

static const luaL_Reg backend_methods[] = {
	{ "close", backend_close },
	{ "name", backend_name },
	{ "threads", backend_threads },
	{ "cpu", backend_cpu },
	{ "tensor", backend_tensor },
	{ "load", backend_load },
	{ "graph", backend_graph },
	{ NULL, NULL }
};

static const luaL_Reg store_meta[] = {
	{ "__gc", store_close },
	{ "__close", store_close },
	{ NULL, NULL }
};

static const luaL_Reg store_methods[] = {
	{ "close", store_close },
	{ "get", store_get },
	{ "names", store_names },
	{ "size", store_size },
	{ NULL, NULL }
};

static const luaL_Reg graph_meta[] = {
	{ "__gc", graph_close },
	{ "__close", graph_close },
	{ NULL, NULL }
};

static const luaL_Reg graph_methods[] = {
	{ "close", graph_close },
	{ "add", graph_add },
	{ "sub", graph_sub },
	{ "mul", graph_mul },
	{ "div", graph_div },
	{ "scale", graph_scale },
	{ "silu", graph_silu },
	{ "tanh", graph_tanh },
	{ "sigmoid", graph_sigmoid },
	{ "relu", graph_relu },
	{ "elu", graph_elu },
	{ "exp", graph_exp },
	{ "abs", graph_abs },
	{ "neg", graph_neg },
	{ "sin", graph_sin },
	{ "cos", graph_cos },
	{ "sqr", graph_sqr },
	{ "sqrt", graph_sqrt },
	{ "gelu", graph_gelu },
	{ "gelu_erf", graph_gelu_erf },
	{ "conv1d", graph_conv1d },
	{ "rms_norm", graph_rms_norm },
	{ "norm", graph_norm },
	{ "transpose", graph_transpose },
	{ "cont", graph_cont },
	{ "cast", graph_cast },
	{ "reshape", graph_reshape },
	{ "view", graph_view },
	{ "permute", graph_permute },
	{ "mul_mat", graph_mul_mat },
	{ "matmul", graph_matmul },
	{ "get_rows", graph_get_rows },
	{ "set_rows", graph_set_rows },
	{ "cpy", graph_cpy },
	{ "rope", graph_rope },
	{ "soft_max", graph_soft_max },
	{ "flash_attn", graph_flash_attn },
	{ "tile", graph_repeat },
	{ "concat", graph_concat },
	{ "sum_rows", graph_sum_rows },
	{ "mean", graph_mean },
	{ "clamp", graph_clamp },
	{ "argmax", graph_argmax },
	{ "top_k", graph_top_k },
	{ "compute", graph_compute },
	{ "buffer_size", graph_buffer_size },
	{ "measure", graph_measure },
	{ "splits", graph_splits },
	{ NULL, NULL }
};

static const luaL_Reg tensor_meta[] = {
	{ "__gc", tensor_close },
	{ "__close", tensor_close },
	{ "__tostring", tensor_tostring },
	{ NULL, NULL }
};

static const luaL_Reg tensor_methods[] = {
	{ "close", tensor_close },
	{ "set", tensor_set },
	{ "get", tensor_get },
	{ "bytes", tensor_bytes },
	{ "to_host", tensor_to_host },
	{ "from_host", tensor_from_host },
	{ "fill_causal", tensor_fill_causal },
	{ "fill_mask", tensor_fill_mask },
	{ "fill_blocks", tensor_fill_blocks },
	{ "fill_range", tensor_fill_range },
	{ "zero", tensor_zero },
	{ "fill", tensor_fill },
	{ "shape", tensor_shape },
	{ "strides", tensor_strides },
	{ "type", tensor_type },
	{ "nbytes", tensor_nbytes },
	{ "name", tensor_name },
	{ "backend", tensor_backend },
	{ NULL, NULL }
};

static const luaL_Reg hostbuf_meta[] = {
	{ "__gc", hostbuf_close },
	{ "__close", hostbuf_close },
	{ "__len", hostbuf_len },
	{ NULL, NULL }
};

static const luaL_Reg hostbuf_methods[] = {
	{ "close", hostbuf_close },
	{ "len", hostbuf_len },
	{ "bytes", hostbuf_bytes },
	{ NULL, NULL }
};

static const luaL_Reg lib[] = {
	{ "backend", l_backend },
	{ "clock", l_clock },
	{ "devices", l_devices },
	{ "gguf", l_gguf },
	{ "hostbuf", l_hostbuf },
	{ "log", l_log },
	{ "row_size", l_row_size },
	{ "threads", l_threads },
	{ NULL, NULL }
};

static void
newclass(lua_State *L, const char *name, const luaL_Reg *meta, const luaL_Reg *methods)
{
	luaL_newmetatable(L, name);
	luaL_setfuncs(L, meta, 0);
	lua_newtable(L);
	luaL_setfuncs(L, methods, 0);
	lua_setfield(L, -2, "__index");
	lua_pop(L, 1);
}

struct graph *
lgml_check_graph(lua_State *L, int idx)
{
	return check_graph(L, idx);
}

struct tensor *
lgml_operand(lua_State *L, struct graph *g, int idx)
{
	return operand(L, g, idx);
}

struct tensor *
lgml_opt_operand(lua_State *L, struct graph *g, int idx)
{
	return opt_operand(L, g, idx);
}

void
lgml_reserve(lua_State *L, struct graph *g, int n)
{
	reserve(L, g, n);
}

int
lgml_push_node(lua_State *L, struct graph *g, struct ggml_tensor *r, int rank)
{
	return push_node(L, g, r, rank);
}

struct tensor *
lgml_push_leaf(lua_State *L, int sidx, struct ggml_tensor *gt, int rank)
{
	return push_leaf(L, lua_absindex(L, sidx), gt, rank);
}

/* Add methods to the tensor class. */
void
lgml_tensor_methods(lua_State *L, const luaL_Reg *fns)
{
	luaL_getmetatable(L, TENSOR_MT);
	lua_getfield(L, -1, "__index");
	luaL_setfuncs(L, fns, 0);
	lua_pop(L, 2);
}

/* Add ops to the graph class. Op functions take the graph as arg 1. */
void
lgml_graph_ops(lua_State *L, const luaL_Reg *ops)
{
	luaL_getmetatable(L, GRAPH_MT);
	lua_getfield(L, -1, "__index");
	luaL_setfuncs(L, ops, 0);
	lua_pop(L, 2);
}

int luaopen_lgml(lua_State *L);

int
luaopen_lgml(lua_State *L)
{
	ggml_log_set(log_cb, NULL);
	newclass(L, BACKEND_MT, backend_meta, backend_methods);
	newclass(L, STORE_MT, store_meta, store_methods);
	newclass(L, GRAPH_MT, graph_meta, graph_methods);
	newclass(L, TENSOR_MT, tensor_meta, tensor_methods);
	newclass(L, HOSTBUF_MT, hostbuf_meta, hostbuf_methods);
	luaL_newlib(L, lib);
	lgml_open_sample(L);
	lgml_open_tokenize(L);
	lgml_open_safetensors(L);
	lgml_open_audio(L);
	lgml_open_image(L);
	lgml_open_sha256(L);
	lgml_open_vocab(L);
	lgml_open_ops_orpheus(L);
	lgml_open_ops_qwen3tts(L);
	lgml_open_ops_vibevoice(L);
	lgml_open_ops_nn(L);
	lgml_open_train(L);
	lgml_open_mem(L);
	return 1;
}
