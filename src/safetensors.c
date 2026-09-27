/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * safetensors reader. A file is an 8-byte little-endian header length,
 * a JSON header and raw tensor data. The file is untrusted: every
 * size and offset is checked before use.
 */

/* fseeko, fileno and off_t. */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <ggml.h>
#include <ggml-backend.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

#define ST_MAX_HEADER	((uint64_t)100 << 20)
#define ST_MAX_TENSORS	65536
#define ST_MAX_FILES	1024
#define ST_MAX_ELEMS	((int64_t)1 << 48)

struct st_tensor {
	char			 name[GGML_MAX_NAME];
	enum ggml_type		 type;
	int			 rank;
	int64_t			 ne[GGML_MAX_DIMS];
	uint64_t		 begin;
	uint64_t		 end;
	struct ggml_tensor	*t;
	int			 widen;		/* f16 or bf16 stored as f32 */
};

struct st_file {
	size_t			 n;
	uint64_t		 data_off;
	uint64_t		 size;
	struct st_tensor	 t[];
};

static const struct {
	const char	*name;
	enum ggml_type	 type;
} dtypes[] = {
	{ "F64", GGML_TYPE_F64 },
	{ "F32", GGML_TYPE_F32 },
	{ "F16", GGML_TYPE_F16 },
	{ "BF16", GGML_TYPE_BF16 },
	{ "I64", GGML_TYPE_I64 },
	{ "I32", GGML_TYPE_I32 },
	{ "I16", GGML_TYPE_I16 },
	{ "I8", GGML_TYPE_I8 },
};

static int
map_dtype(const char *s, enum ggml_type *type)
{
	size_t i;

	for (i = 0; i < sizeof(dtypes) / sizeof(dtypes[0]); i++) {
		if (strcmp(dtypes[i].name, s) == 0) {
			*type = dtypes[i].type;
			return 0;
		}
	}
	return -1;
}

/* Read the header length and the file size. */
static void
read_prefix(lua_State *L, const char *path, uint64_t *hlen, uint64_t *fsize)
{
	unsigned char b[8];
	struct stat sb;
	FILE *f;
	int i, ok;

	f = fopen(path, "rb");
	if (f == NULL)
		luaL_error(L, "%s: %s", path, strerror(errno));
	ok = fstat(fileno(f), &sb) == 0 && fread(b, 1, 8, f) == 8;
	fclose(f);
	if (!ok)
		luaL_error(L, "%s: not a safetensors file", path);
	*hlen = 0;
	for (i = 7; i >= 0; i--)
		*hlen = *hlen << 8 | b[i];
	*fsize = (uint64_t)sb.st_size;
	if (*hlen < 2 || *hlen > ST_MAX_HEADER || *hlen > *fsize - 8)
		luaL_error(L, "%s: bad header length", path);
}

/* Read the JSON header into a new userdata on the stack. */
static const char *
read_header(lua_State *L, const char *path, uint64_t hlen)
{
	char *h = lua_newuserdatauv(L, (size_t)hlen, 0);
	FILE *f;
	int ok;

	f = fopen(path, "rb");
	if (f == NULL)
		luaL_error(L, "%s: %s", path, strerror(errno));
	ok = fseeko(f, 8, SEEK_SET) == 0 && fread(h, 1, (size_t)hlen, f) == hlen;
	fclose(f);
	if (!ok)
		luaL_error(L, "%s: short read", path);
	return h;
}

/* Count the keys of the table on top of the stack, up to max + 1. */
static size_t
count_keys(lua_State *L, size_t max)
{
	size_t n = 0;

	lua_pushnil(L);
	while (lua_next(L, -2) != 0) {
		lua_pop(L, 1);
		if (++n > max) {
			lua_pop(L, 1);
			break;
		}
	}
	return n;
}

/*
 * Read the JSON array of up to max integers at idx into v. Returns
 * the count, or -1.
 */
static int
read_ints(lua_State *L, int idx, int64_t *v, int max)
{
	size_t n, i;
	int isnum;

	if (lua_type(L, idx) != LUA_TTABLE)
		return -1;
	lua_pushvalue(L, idx);
	n = count_keys(L, (size_t)max);
	if (n > (size_t)max) {
		lua_pop(L, 1);
		return -1;
	}
	for (i = 0; i < n; i++) {
		isnum = 0;
		if (lua_rawgeti(L, -1, (lua_Integer)i + 1) == LUA_TNUMBER)
			v[i] = (int64_t)lua_tointegerx(L, -1, &isnum);
		lua_pop(L, 1);
		if (!isnum) {
			lua_pop(L, 1);
			return -1;
		}
	}
	lua_pop(L, 1);
	return (int)n;
}

static int
read_shape(lua_State *L, int idx, struct st_tensor *t)
{
	int64_t dims[GGML_MAX_DIMS];
	int n, i;

	n = read_ints(L, idx, dims, GGML_MAX_DIMS);
	if (n < 0)
		return -1;
	for (i = 0; i < n; i++)
		if (dims[i] < 1)
			return -1;
	t->rank = n;
	for (i = 0; i < GGML_MAX_DIMS; i++)
		t->ne[i] = i < n ? dims[n - 1 - i] : 1;
	return 0;
}

static int
read_offsets(lua_State *L, int idx, struct st_tensor *t)
{
	int64_t v[2];

	if (read_ints(L, idx, v, 2) != 2 || v[0] < 0 || v[0] > v[1])
		return -1;
	t->begin = (uint64_t)v[0];
	t->end = (uint64_t)v[1];
	return 0;
}

/* Parse the tensor entry on top of the stack. Returns a message, or NULL. */
static const char *
read_entry(lua_State *L, struct st_tensor *t)
{
	const char *msg = NULL;
	const char *dt;

	if (!lua_istable(L, -1))
		return "tensor entry is not an object";
	lua_getfield(L, -1, "dtype");
	lua_getfield(L, -2, "shape");
	lua_getfield(L, -3, "data_offsets");
	if (lua_isnil(L, -1) || lua_isnil(L, -2) || lua_isnil(L, -3))
		msg = "tensor entry lacks dtype, shape or data_offsets";
	else if (lua_type(L, -3) != LUA_TSTRING || (dt = lua_tostring(L, -3)) == NULL ||
	    map_dtype(dt, &t->type) != 0)
		msg = "unsupported dtype";
	else if (read_shape(L, -2, t) != 0)
		msg = "bad shape (rank 0 to 4, dimensions >= 1)";
	else if (read_offsets(L, -1, t) != 0)
		msg = "bad data_offsets";
	lua_pop(L, 3);
	return msg;
}

/* Check that the entry size matches its data range inside the file. */
static const char *
check_entry(const struct st_tensor *t, uint64_t data_size)
{
	int64_t n = 1;
	int i;

	for (i = 0; i < GGML_MAX_DIMS; i++) {
		if (t->ne[i] > ST_MAX_ELEMS / n)
			return "tensor too large";
		n *= t->ne[i];
	}
	if (t->end > data_size)
		return "data range past end of file";
	if (t->end - t->begin != (uint64_t)n * ggml_type_size(t->type))
		return "data range does not match shape";
	return NULL;
}

/* Copy the string values of the __metadata__ table on top into idx. */
static void
read_meta(lua_State *L, int idx)
{
	lua_pushnil(L);
	while (lua_next(L, -2) != 0) {
		if (lua_type(L, -2) == LUA_TSTRING && lua_type(L, -1) == LUA_TSTRING) {
			lua_pushvalue(L, -2);
			lua_insert(L, -2);
			lua_rawset(L, idx);
		} else {
			lua_pop(L, 1);
		}
	}
}

/* Data order, then name order. */
static int
cmp_tensor(const void *a, const void *b)
{
	const struct st_tensor *x = a, *y = b;

	if (x->begin != y->begin)
		return x->begin < y->begin ? -1 : 1;
	return strcmp(x->name, y->name);
}

/*
 * Parse the file at path. Push a userdata that holds the tensor index,
 * in data order. With meta != 0, add __metadata__ to the table at that
 * index.
 */
static struct st_file *
st_read(lua_State *L, const char *path, int meta)
{
	struct st_file *sf;
	struct st_tensor *t;
	uint64_t hlen, fsize;
	const char *h, *name, *msg;
	size_t n, len;

	read_prefix(L, path, &hlen, &fsize);
	h = read_header(L, path, hlen);
	lgml_json_decode(L, h, (size_t)hlen, path);
	lua_remove(L, -2);
	if (!lua_istable(L, -1))
		luaL_error(L, "%s: JSON header is not an object", path);
	n = count_keys(L, ST_MAX_TENSORS + 1);
	if (n > ST_MAX_TENSORS + 1)
		luaL_error(L, "%s: too many tensors", path);
	sf = lua_newuserdatauv(L, sizeof(*sf) + n * sizeof(sf->t[0]), 0);
	sf->n = 0;
	sf->data_off = 8 + hlen;
	sf->size = fsize - sf->data_off;
	lua_pushnil(L);
	while (lua_next(L, -3) != 0) {
		if (lua_type(L, -2) != LUA_TSTRING)
			luaL_error(L, "%s: JSON header is not an object", path);
		name = lua_tolstring(L, -2, &len);
		if (strcmp(name, "__metadata__") == 0) {
			if (!lua_istable(L, -1))
				luaL_error(L, "%s: bad __metadata__", path);
			if (meta != 0)
				read_meta(L, meta);
			lua_pop(L, 1);
			continue;
		}
		if (len == 0 || len >= GGML_MAX_NAME || strlen(name) != len)
			luaL_error(L, "%s: tensor name empty, over %d bytes or holds NUL", path,
			    GGML_MAX_NAME - 1);
		if (sf->n == ST_MAX_TENSORS)
			luaL_error(L, "%s: too many tensors", path);
		t = &sf->t[sf->n];
		memset(t, 0, sizeof(*t));
		memcpy(t->name, name, len + 1);
		msg = read_entry(L, t);
		if (msg == NULL)
			msg = check_entry(t, sf->size);
		if (msg != NULL)
			luaL_error(L, "%s: %s: %s", path, name, msg);
		sf->n++;
		lua_pop(L, 1);
	}
	qsort(sf->t, sf->n, sizeof(sf->t[0]), cmp_tensor);
	lua_remove(L, -2);
	return sf;
}

static void
push_shape(lua_State *L, const struct st_tensor *t)
{
	int i;

	lua_createtable(L, t->rank, 0);
	for (i = 0; i < t->rank; i++) {
		lua_pushinteger(L, t->ne[t->rank - 1 - i]);
		lua_seti(L, -2, i + 1);
	}
}

/*** Read the header of a safetensors file.
 * The result is { `data_offset`, size, metadata = { key = value }, tensors =
 * { { name, type, shape, offset, size }, ... } }. offset counts from the
 * start of the data. Tensors are in data order. metadata holds the string
 * values of `__metadata__`.
 * @tparam string path safetensors file
 * @treturn table header
 * @raise if the file cannot be read or is not a valid safetensors file
 * @function safetensors
 */
static int
l_safetensors(lua_State *L)
{
	const char *path = luaL_checkstring(L, 1);
	struct st_file *sf;
	size_t i;

	lua_newtable(L);
	lua_newtable(L);
	sf = st_read(L, path, lua_gettop(L));
	lua_pushvalue(L, -2);
	lua_setfield(L, -4, "metadata");
	lua_pushinteger(L, (lua_Integer)sf->data_off);
	lua_setfield(L, -4, "data_offset");
	lua_pushinteger(L, (lua_Integer)sf->size);
	lua_setfield(L, -4, "size");
	lua_createtable(L, (int)sf->n, 0);
	for (i = 0; i < sf->n; i++) {
		lua_createtable(L, 0, 5);
		lua_pushstring(L, sf->t[i].name);
		lua_setfield(L, -2, "name");
		lua_pushstring(L, ggml_type_name(sf->t[i].type));
		lua_setfield(L, -2, "type");
		push_shape(L, &sf->t[i]);
		lua_setfield(L, -2, "shape");
		lua_pushinteger(L, (lua_Integer)sf->t[i].begin);
		lua_setfield(L, -2, "offset");
		lua_pushinteger(L, (lua_Integer)(sf->t[i].end - sf->t[i].begin));
		lua_setfield(L, -2, "size");
		lua_seti(L, -2, (lua_Integer)i + 1);
	}
	lua_setfield(L, -4, "tensors");
	lua_pop(L, 2);
	return 1;
}

/* Copy the data of every tensor in sf from path to the device. */
/* Source bytes per step when widening: the f32 output needs twice as much. */
#define WIDEN_STEP	(LOAD_CHUNK / 3 & ~(size_t)7)

static void
upload(lua_State *L, const char *path, const struct st_file *sf)
{
	const struct st_tensor *t;
	size_t i, n, done, size, step;
	char *chunk, *src;
	FILE *f;
	int e;

	f = fopen(path, "rb");
	e = errno;
	chunk = f != NULL ? malloc(LOAD_CHUNK) : NULL;
	if (chunk == NULL) {
		if (f != NULL) {
			e = ENOMEM;
			fclose(f);
		}
		luaL_error(L, "%s: %s", path, strerror(e));
	}
	for (i = 0; i < sf->n; i++) {
		t = &sf->t[i];
		size = (size_t)(t->end - t->begin);
		step = t->widen ? WIDEN_STEP : LOAD_CHUNK;
		src = t->widen ? chunk + LOAD_CHUNK - WIDEN_STEP : chunk;
		if (fseeko(f, (off_t)(sf->data_off + t->begin), SEEK_SET) != 0)
			goto short_read;
		for (done = 0; done < size; done += n) {
			n = size - done < step ? size - done : step;
			if (fread(src, 1, n, f) != n)
				goto short_read;
			if (!t->widen) {
				ggml_backend_tensor_set(t->t, chunk, done, n);
				continue;
			}
			if (t->type == GGML_TYPE_F16)
				ggml_fp16_to_fp32_row((void *)src, (float *)(void *)chunk,
				    (int64_t)(n / 2));
			else
				ggml_bf16_to_fp32_row((void *)src, (float *)(void *)chunk,
				    (int64_t)(n / 2));
			ggml_backend_tensor_set(t->t, chunk, done * 2, n * 2);
		}
	}
	free(chunk);
	fclose(f);
	return;

short_read:
	free(chunk);
	fclose(f);
	luaL_error(L, "%s: short read", path);
}

/* Push the list of paths from arg 2: a string or a table of strings. */
static int
push_paths(lua_State *L)
{
	lua_Integer i, n;

	if (lua_type(L, 2) == LUA_TSTRING) {
		lua_createtable(L, 1, 0);
		lua_pushvalue(L, 2);
		lua_seti(L, -2, 1);
		return 1;
	}
	luaL_checktype(L, 2, LUA_TTABLE);
	n = luaL_len(L, 2);
	luaL_argcheck(L, n >= 1 && n <= ST_MAX_FILES, 2, "need 1 to 1024 paths");
	lua_createtable(L, (int)n, 0);
	for (i = 1; i <= n; i++) {
		lua_geti(L, 2, i);
		luaL_argcheck(L, lua_type(L, -1) == LUA_TSTRING, 2, "paths must be strings");
		lua_seti(L, -2, i);
	}
	return (int)n;
}

/*
 * Make the ggml tensors of sf in ctx. seen (at idx) maps names to true.
 * With f32_1d, 1-D f16 and bf16 tensors become f32.
 */
static void
make_tensors(lua_State *L, struct ggml_context *ctx, struct st_file *sf, int seen, int f32_1d)
{
	struct st_tensor *t;
	enum ggml_type type;
	size_t i;

	for (i = 0; i < sf->n; i++) {
		t = &sf->t[i];
		if (lua_getfield(L, seen, t->name) != LUA_TNIL)
			luaL_error(L, "tensor '%s' appears twice", t->name);
		lua_pop(L, 1);
		lua_pushboolean(L, 1);
		lua_setfield(L, seen, t->name);
		type = t->type;
		t->widen = f32_1d && t->rank <= 1 &&
		    (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);
		if (t->widen)
			type = GGML_TYPE_F32;
		t->t = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, t->ne);
		ggml_set_name(t->t, t->name);
	}
}

/*** A compute device.
 * Weights, leaf tensors and graphs keep their backend alive, so close
 * order does not matter.
 * @type backend
 */

/*** Load every tensor of one or more safetensors files to the device.
 * Tensor names must be unique across the files.
 * @tparam string|{string,...} paths one path, or a list of 1 to 1024 paths
 * @tparam[opt] table opts `f32_1d`: store 1-D f16 and bf16 tensors as f32
 * @treturn weights the tensors
 * @raise on a bad file, a name that appears twice, no tensors, or if the
 * device is out of memory
 * @function backend:load_safetensors
 */
static int
backend_load_safetensors(lua_State *L)
{
	struct backend *b = lgml_check_backend(L, 1);
	struct ggml_init_params ip;
	struct ggml_context *ctx;
	struct st_file *sf;
	struct store *s;
	int i, nf, list, first, seen, sidx, f32_1d = 0;
	size_t total = 0;

	if (lua_type(L, 3) == LUA_TTABLE) {
		lua_getfield(L, 3, "f32_1d");
		f32_1d = lua_toboolean(L, -1);
		lua_pop(L, 1);
	}

	nf = push_paths(L);
	list = lua_gettop(L);
	luaL_checkstack(L, nf + 8, "too many files");
	first = list + 1;
	for (i = 1; i <= nf; i++) {
		lua_geti(L, list, i);
		sf = st_read(L, lua_tostring(L, -1), 0);
		lua_remove(L, -2);
		total += sf->n;
		if (total > ST_MAX_TENSORS)
			return luaL_error(L, "too many tensors");
	}
	if (total == 0)
		return luaL_error(L, "no tensors");
	ip.mem_size = ggml_tensor_overhead() * total;
	ip.mem_buffer = NULL;
	ip.no_alloc = true;
	ctx = ggml_init(ip);
	if (ctx == NULL)
		return luaL_error(L, "ggml_init failed");
	s = lgml_push_store(L, 1, ctx, NULL);
	sidx = lua_gettop(L);
	lua_newtable(L);
	seen = lua_gettop(L);
	for (i = 0; i < nf; i++)
		make_tensors(L, ctx, lua_touserdata(L, first + i), seen, f32_1d);
	lgml_mem_need(L, ggml_backend_get_default_buffer_type(b->be),
	    ggml_backend_alloc_ctx_tensors_from_buft_size(ctx,
	    ggml_backend_get_default_buffer_type(b->be)), "safetensors weights");
	s->buf = ggml_backend_alloc_ctx_tensors(ctx, b->be);
	if (s->buf == NULL)
		return luaL_error(L, "out of device memory");
	lgml_store_account(s, MEM_WEIGHTS);
	for (i = 0; i < nf; i++) {
		lua_geti(L, list, i + 1);
		upload(L, lua_tostring(L, -1), lua_touserdata(L, first + i));
		lua_pop(L, 1);
	}
	lua_pushvalue(L, sidx);
	return 1;
}

void
lgml_open_safetensors(lua_State *L)
{
	lua_pushcfunction(L, l_safetensors);
	lua_setfield(L, -2, "safetensors");
	luaL_getmetatable(L, BACKEND_MT);
	lua_getfield(L, -1, "__index");
	lua_pushcfunction(L, backend_load_safetensors);
	lua_setfield(L, -2, "load_safetensors");
	lua_pop(L, 2);
}
