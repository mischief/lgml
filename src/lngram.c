/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Lua binding for ngram.c, the PLE n-gram hash embedding of qwen4exp.
 * The table stays on disk. reader:embed hashes each token with the
 * tokens before it, reads the rows and writes the f32 embedding to a
 * hostbuf or tensor.
 */

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <ggml.h>
#include <ggml-backend.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"
#include "ngram.h"

#define NGRAM_MT	"lgml_ngram"

struct reader {
	struct ngram	*r;
};

static struct ngram *
check_reader(lua_State *L, int idx)
{
	struct reader *rd = luaL_checkudata(L, idx, NGRAM_MT);

	if (rd->r == NULL)
		luaL_error(L, "ngram reader closed");
	return rd->r;
}

/* Scratch memory that the collector frees. Pushes a userdata. */
static void *
scratch(lua_State *L, size_t n)
{
	return lua_newuserdatauv(L, n > 0 ? n : 1, 0);
}

/*
 * The integers of the list at idx, at most max of them, from the end.
 * Pushes the scratch array. A nil list has no integers.
 */
static int32_t *
int32_list(lua_State *L, int idx, size_t max, int64_t lo, size_t *np)
{
	lua_Integer len, i, first, v;
	int32_t *a;

	if (lua_isnoneornil(L, idx)) {
		*np = 0;
		return scratch(L, 0);
	}
	luaL_checktype(L, idx, LUA_TTABLE);
	len = luaL_len(L, idx);
	first = (lua_Unsigned)len > max ? len - (lua_Integer)max : 0;
	a = scratch(L, (size_t)(len - first) * sizeof(*a));
	for (i = first; i < len; i++) {
		lua_geti(L, idx, i + 1);
		if (!lua_isinteger(L, -1))
			luaL_argerror(L, idx, "need a list of integers");
		v = lua_tointeger(L, -1);
		if (v < lo || v > INT32_MAX)
			luaL_argerror(L, idx, "token id out of range");
		a[i - first] = (int32_t)v;
		lua_pop(L, 1);
	}
	*np = (size_t)(len - first);
	return a;
}

/*
 * Where to write nfloats f32 values: the hostbuf or tensor at idx, at
 * byte offset off (arg idx + 1). Returns the hostbuf memory, or NULL for
 * a tensor, then the caller writes through tensor_out.
 */
static float *
out_ptr(lua_State *L, int idx, size_t nfloats, struct tensor **tp, size_t *offp)
{
	struct hostbuf *hb;
	struct tensor *t;
	lua_Integer off = luaL_optinteger(L, idx + 1, 0);
	size_t esz;

	luaL_argcheck(L, off >= 0, idx + 1, "offset out of range");
	*offp = (size_t)off;
	*tp = NULL;
	if ((hb = luaL_testudata(L, idx, HOSTBUF_MT)) != NULL) {
		if (hb->p == NULL)
			luaL_argerror(L, idx, "hostbuf closed");
		luaL_argcheck(L, off % (lua_Integer)sizeof(float) == 0, idx + 1,
		    "offset not a multiple of 4");
		luaL_argcheck(L, (size_t)off <= hb->size &&
		    nfloats <= (hb->size - (size_t)off) / sizeof(float), idx,
		    "hostbuf too small");
		return (float *)(void *)(hb->p + off);
	}
	t = lgml_check_tensor(L, idx);
	if (t->t->buffer == NULL || !ggml_is_contiguous(t->t) ||
	    (t->t->type != GGML_TYPE_F32 && t->t->type != GGML_TYPE_F16))
		luaL_argerror(L, idx, "need a contiguous f32 or f16 tensor with storage");
	esz = ggml_type_size(t->t->type);
	luaL_argcheck(L, (size_t)off <= ggml_nbytes(t->t) &&
	    nfloats <= (ggml_nbytes(t->t) - (size_t)off) / esz, idx, "tensor too small");
	*tp = t;
	return NULL;
}

/* Write the n floats of f to the tensor t at byte offset off. */
static void
tensor_out(struct tensor *t, float *f, size_t n, size_t off)
{
	/* The row converter writes a narrower type, so in place is safe. */
	if (t->t->type == GGML_TYPE_F16)
		ggml_fp32_to_fp16_row(f, (ggml_fp16_t *)(void *)f, (int64_t)n);
	ggml_backend_tensor_set(t->t, f, off, n * ggml_type_size(t->t->type));
}

/* Read rows[0..n) into out (arg idx) and return the bytes written. */
static int
read_out(lua_State *L, struct ngram *r, const int64_t *rows, size_t n, int idx)
{
	struct tensor *t;
	size_t off, nf = n * (size_t)r->head_dim;
	float *f = out_ptr(L, idx, nf, &t, &off);

	if (f == NULL)
		f = scratch(L, nf * sizeof(float));
	if (ngram_read(r, rows, n, f) != 0)
		return luaL_error(L, "ngram: read: %s", strerror(errno));
	if (t != NULL)
		tensor_out(t, f, nf, off);
	lua_pushinteger(L, (lua_Integer)(nf * (t != NULL ? ggml_type_size(t->t->type) :
	    sizeof(float))));
	return 1;
}

/*** Open an n-gram embedding table.
 * The reader keeps the table on disk and reads only the rows each call
 * needs. opts fields, all optional:
 *
 * - `meta`: gguf file with the `<arch>.ple.*` keys (default: path). For a
 *   split model, this is the first shard.
 * - `tensor`: table name (default "per_layer_token_embd.weight")
 * - `mode`: "direct" (O_DIRECT, the default), "pread" or "mmap". Direct
 *   mode falls back to "pread" where the system or file system has no
 *   direct I/O; `reader:info().mode` tells which one is in use.
 * - `threads`: parallel reads (default 8)
 * - `align`: direct I/O unit in bytes, a power of two (default: the
 *   first of 512 and 4096 that the file accepts)
 * @tparam string path gguf file that holds the table
 * @tparam[opt] table opts options
 * @treturn[1] ngram_reader the reader
 * @return[2] nil
 * @treturn[2] string error message
 * @function ngram.open
 */
static int
l_ngram_open(lua_State *L)
{
	static const char *const modes[] = { "direct", "pread", "mmap", NULL };
	const char *path = luaL_checkstring(L, 1), *meta = NULL;
	const char *tensor = "per_layer_token_embd.weight";
	struct reader *rd;
	lua_Integer threads = 8, align = 0;
	char err[512];
	int mode = NGRAM_DIRECT;

	if (!lua_isnoneornil(L, 2)) {
		luaL_checktype(L, 2, LUA_TTABLE);
		if (lua_getfield(L, 2, "meta") != LUA_TNIL)
			meta = luaL_checkstring(L, -1);
		if (lua_getfield(L, 2, "tensor") != LUA_TNIL)
			tensor = luaL_checkstring(L, -1);
		if (lua_getfield(L, 2, "mode") != LUA_TNIL)
			mode = luaL_checkoption(L, -1, NULL, modes);
		if (lua_getfield(L, 2, "threads") != LUA_TNIL)
			threads = luaL_checkinteger(L, -1);
		if (lua_getfield(L, 2, "align") != LUA_TNIL)
			align = luaL_checkinteger(L, -1);
		luaL_argcheck(L, threads >= 1 && threads <= 64, 2, "threads must be 1 to 64");
		luaL_argcheck(L, align >= 0 && align <= 1 << 20, 2, "align out of range");
	}
	rd = lua_newuserdatauv(L, sizeof(*rd), 0);
	rd->r = NULL;
	luaL_setmetatable(L, NGRAM_MT);
	rd->r = ngram_open(path, meta, tensor, (enum ngram_mode)mode, (size_t)align,
	    (int)threads, err, sizeof(err));
	if (rd->r == NULL) {
		luaL_pushfail(L);
		lua_pushstring(L, err);
		return 2;
	}
	return 1;
}

/*** A reader of an n-gram embedding table.
 *
 * Interface for the forward pass: `reader:embed(ids, prev, out)` writes
 * one row of `info().dim` f32 values per token, the input of the PLE
 * layer, as llama.cpp's `ple_embd` tensor (dim, n_tokens). A token row
 * is `heads` blocks of `head_dim` values in head order: first the
 * `heads_per_ngram` bigram heads, then the trigram heads. prev holds the
 * tokens before ids[1]; an empty prev is the start of a sequence, where
 * the missing tokens read as EOS. An EOS before a token also ends its
 * context. For a position that holds an image embedding, pass
 * `info().image_token`. One call covers one sequence.
 * @type ngram_reader
 */

/*** Write the embeddings of tokens.
 * @tparam {int,...} ids token ids
 * @tparam {int,...}|nil prev tokens before ids[1], oldest first; only the
 * last `ngram_size` - 1 count. nil or {} at the start of a sequence.
 * @tparam hostbuf|tensor out f32 hostbuf, or contiguous f32 or f16 tensor
 * with storage; #ids * dim values go there
 * @tparam[opt=0] int off byte offset in out
 * @treturn int bytes written
 * @raise on bad arguments or a read error
 * @function ngram_reader:embed
 */
static int
reader_embed(lua_State *L)
{
	struct ngram *r = check_reader(L, 1);
	const struct ngram_params *p = &r->p;
	size_t n, nprev;
	int32_t *ids, *prev;
	int64_t *rows;

	/* Scratch arrays go above the arguments. */
	lua_settop(L, 5);
	ids = int32_list(L, 2, SIZE_MAX, 0, &n);
	prev = int32_list(L, 3, (size_t)p->ngram_size - 1, -1, &nprev);
	rows = scratch(L, n * (size_t)p->n_heads * sizeof(*rows));
	ngram_rows(p, prev, nprev, ids, n, rows);
	return read_out(L, r, rows, n * (size_t)p->n_heads, 4);
}

/*** Get the table rows of tokens, without reading them.
 * Arguments as in `ngram_reader:embed`.
 * @tparam {int,...} ids token ids
 * @tparam {int,...}|nil prev tokens before ids[1], oldest first
 * @treturn {int,...} #ids * heads rows from 0, token-major
 * @function ngram_reader:rows
 */
static int
reader_rows(lua_State *L)
{
	struct ngram *r = check_reader(L, 1);
	const struct ngram_params *p = &r->p;
	size_t i, n, nprev;
	int32_t *ids, *prev;
	int64_t *rows;

	lua_settop(L, 3);
	ids = int32_list(L, 2, SIZE_MAX, 0, &n);
	prev = int32_list(L, 3, (size_t)p->ngram_size - 1, -1, &nprev);
	rows = scratch(L, n * (size_t)p->n_heads * sizeof(*rows));
	ngram_rows(p, prev, nprev, ids, n, rows);
	n *= (size_t)p->n_heads;
	lua_createtable(L, (int)n, 0);
	for (i = 0; i < n; i++) {
		lua_pushinteger(L, rows[i]);
		lua_seti(L, -2, (lua_Integer)i + 1);
	}
	return 1;
}

/*** Read table rows as f32.
 * @tparam {int,...} rows row numbers from 0
 * @tparam hostbuf|tensor out as in `ngram_reader:embed`; #rows * head_dim
 * values go there
 * @tparam[opt=0] int off byte offset in out
 * @treturn int bytes written
 * @raise on bad arguments or a read error
 * @function ngram_reader:read
 */
static int
reader_read(lua_State *L)
{
	struct ngram *r = check_reader(L, 1);
	lua_Integer len, i, v;
	int64_t *rows;

	lua_settop(L, 4);
	luaL_checktype(L, 2, LUA_TTABLE);
	len = luaL_len(L, 2);
	rows = scratch(L, (size_t)len * sizeof(*rows));
	for (i = 0; i < len; i++) {
		lua_geti(L, 2, i + 1);
		v = lua_tointeger(L, -1);
		if (!lua_isinteger(L, -1) || v < 0 || v >= r->nrows)
			luaL_argerror(L, 2, "row out of range");
		rows[i] = v;
		lua_pop(L, 1);
	}
	return read_out(L, r, rows, (size_t)len, 3);
}

static void
setint(lua_State *L, const char *k, lua_Integer v)
{
	lua_pushinteger(L, v);
	lua_setfield(L, -2, k);
}

static void
setnum(lua_State *L, const char *k, lua_Number v)
{
	lua_pushnumber(L, v);
	lua_setfield(L, -2, k);
}

/*** Describe the table.
 * The result has `mode` ("direct", "pread" or "mmap"), `fell_back` (true
 * when direct mode was asked for but is not in use), `type`, `rows`,
 * `head_dim`, `heads`, `dim` (heads * `head_dim`), `ngram_size`,
 * `heads_per_ngram`, `eos`, `image_token`, `row_bytes` and `align`.
 * @treturn table the description
 * @function ngram_reader:info
 */
static int
reader_info(lua_State *L)
{
	static const char *const modes[] = { "direct", "pread", "mmap" };
	struct ngram *r = check_reader(L, 1);

	lua_newtable(L);
	lua_pushstring(L, modes[r->mode]);
	lua_setfield(L, -2, "mode");
	lua_pushboolean(L, r->fell_back);
	lua_setfield(L, -2, "fell_back");
	lua_pushstring(L, ggml_type_name(r->type));
	lua_setfield(L, -2, "type");
	setint(L, "rows", r->nrows);
	setint(L, "head_dim", r->head_dim);
	setint(L, "heads", r->p.n_heads);
	setint(L, "dim", r->p.n_heads * r->head_dim);
	setint(L, "ngram_size", r->p.ngram_size);
	setint(L, "heads_per_ngram", r->p.heads_per_ngram);
	setint(L, "eos", r->p.eos);
	setint(L, "image_token", r->p.image_token);
	setint(L, "row_bytes", (lua_Integer)r->row_size);
	setint(L, "align", (lua_Integer)r->align);
	return 1;
}

/*** Get the read counters since open or the last reset.
 * The result has `calls`, `rows` (rows asked for), `unique` (rows after
 * dedupe), `reads` (read syscalls), `bytes` (bytes read from the file),
 * `read_sec` (time in reads) and `total_sec` (time in reads and
 * dequantization).
 * @tparam[opt=false] boolean reset set the counters to zero after
 * @treturn table the counters
 * @function ngram_reader:stats
 */
static int
reader_stats(lua_State *L)
{
	struct ngram *r = check_reader(L, 1);

	lua_newtable(L);
	setint(L, "calls", (lua_Integer)r->st.calls);
	setint(L, "rows", (lua_Integer)r->st.rows);
	setint(L, "unique", (lua_Integer)r->st.unique);
	setint(L, "reads", (lua_Integer)r->st.reads);
	setint(L, "bytes", (lua_Integer)r->st.bytes);
	setnum(L, "read_sec", r->st.read_sec);
	setnum(L, "total_sec", r->st.total_sec);
	if (lua_toboolean(L, 2))
		memset(&r->st, 0, sizeof(r->st));
	return 1;
}

/*** Close the file. Later calls raise an error.
 * The `__gc` and `__close` metamethods call this function.
 * @function ngram_reader:close
 */
static int
reader_close(lua_State *L)
{
	struct reader *rd = luaL_checkudata(L, 1, NGRAM_MT);

	ngram_close(rd->r);
	rd->r = NULL;
	return 0;
}

static const luaL_Reg reader_meta[] = {
	{ "__gc", reader_close },
	{ "__close", reader_close },
	{ NULL, NULL }
};

static const luaL_Reg reader_methods[] = {
	{ "close", reader_close },
	{ "embed", reader_embed },
	{ "rows", reader_rows },
	{ "read", reader_read },
	{ "info", reader_info },
	{ "stats", reader_stats },
	{ NULL, NULL }
};

void
lgml_open_ngram(lua_State *L)
{
	luaL_newmetatable(L, NGRAM_MT);
	luaL_setfuncs(L, reader_meta, 0);
	lua_newtable(L);
	luaL_setfuncs(L, reader_methods, 0);
	lua_setfield(L, -2, "__index");
	lua_pop(L, 1);
	lua_newtable(L);
	lua_pushcfunction(L, l_ngram_open);
	lua_setfield(L, -2, "open");
	lua_setfield(L, -2, "ngram");
}
