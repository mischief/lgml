/*
 * Memory accounting: the bytes of the ggml buffers that lgml owns, by
 * memory (a device name, or "CPU" for host memory) and by part, and a
 * guard that refuses an allocation that does not fit.
 */

#include <stdio.h>
#include <string.h>

#include "lgml.h"

#define MEM_KEYS	8
#define MiB		((size_t)1 << 20)

struct memkey {
	char	 name[64];
	size_t	 cur[MEM_NPART];
	size_t	 peak;		/* the most of all parts at once */
	size_t	 peak_compute;	/* the most compute at once */
};

static struct memkey keys[MEM_KEYS];
static int nkeys;
static size_t guard_margin = 256 * MiB;
static size_t guard_limit;	/* 0: no limit */

static const char *const part_names[MEM_NPART] = { "weights", "tensors", "compute" };

static const char *
buft_name(ggml_backend_buffer_type_t bt)
{
	ggml_backend_dev_t dev;

	if (ggml_backend_buft_is_host(bt))
		return "CPU";
	dev = ggml_backend_buft_get_device(bt);
	return dev != NULL ? ggml_backend_dev_name(dev) : ggml_backend_buft_name(bt);
}

int
lgml_mem_key(ggml_backend_buffer_type_t bt)
{
	const char *name = buft_name(bt);
	int i;

	for (i = 0; i < nkeys; i++)
		if (strcmp(keys[i].name, name) == 0)
			return i;
	if (nkeys == MEM_KEYS)
		return MEM_KEYS - 1;
	snprintf(keys[nkeys].name, sizeof(keys[nkeys].name), "%s", name);
	return nkeys++;
}

static size_t
key_total(const struct memkey *k)
{
	size_t n = 0;
	int p;

	for (p = 0; p < MEM_NPART; p++)
		n += k->cur[p];
	return n;
}

void
lgml_mem_add(int key, int part, size_t add, size_t sub)
{
	struct memkey *k = &keys[key];
	size_t n;

	n = k->cur[part] + add;
	k->cur[part] = sub < n ? n - sub : 0;
	n = key_total(k);
	if (n > k->peak)
		k->peak = n;
	if (k->cur[MEM_COMPUTE] > k->peak_compute)
		k->peak_compute = k->cur[MEM_COMPUTE];
}

void
lgml_store_account(struct store *s, int part)
{
	if (s->mem_bytes > 0)
		lgml_mem_add(s->mem_key, s->mem_part, 0, s->mem_bytes);
	s->mem_part = part;
	s->mem_bytes = 0;
	if (s->buf == NULL)
		return;
	s->mem_key = lgml_mem_key(ggml_backend_buffer_get_type(s->buf));
	s->mem_bytes = ggml_backend_buffer_get_size(s->buf);
	lgml_mem_add(s->mem_key, part, s->mem_bytes, 0);
}

void
lgml_store_unaccount(struct store *s)
{
	if (s->mem_bytes > 0)
		lgml_mem_add(s->mem_key, s->mem_part, 0, s->mem_bytes);
	s->mem_bytes = 0;
}

/*
 * Check that n more bytes fit in bt. Returns NULL, or pushes an error
 * message and returns it. With a limit, the lgml bytes of the memory
 * must stay at most the limit. On a device, n plus the margin must fit
 * in the free memory that ggml reports.
 */
const char *
lgml_mem_check(lua_State *L, ggml_backend_buffer_type_t bt, size_t n, const char *what)
{
	struct memkey *k = &keys[lgml_mem_key(bt)];
	ggml_backend_dev_t dev;
	size_t used = key_total(k), mfree = 0, mtotal = 0;
	lua_Integer need = (lua_Integer)((n + MiB - 1) / MiB);

	if (n == 0)
		return NULL;
	if (guard_limit > 0 && used + n > guard_limit)
		return lua_pushfstring(L, "memory budget: %s needs %I MiB more on %s; %I MiB of "
		    "the %I MiB limit are in use", what, need, k->name, (lua_Integer)(used / MiB),
		    (lua_Integer)(guard_limit / MiB));
	if (ggml_backend_buft_is_host(bt) || (dev = ggml_backend_buft_get_device(bt)) == NULL)
		return NULL;
	ggml_backend_dev_memory(dev, &mfree, &mtotal);
	if (mtotal == 0 || n + guard_margin <= mfree)
		return NULL;
	return lua_pushfstring(L, "memory budget: %s needs %I MiB more on %s; %I MiB are free "
	    "and %I MiB must stay free", what, need, k->name, (lua_Integer)(mfree / MiB),
	    (lua_Integer)(guard_margin / MiB));
}

/* Raise the error of lgml_mem_check, if any. */
void
lgml_mem_need(lua_State *L, ggml_backend_buffer_type_t bt, size_t n, const char *what)
{
	if (lgml_mem_check(L, bt, n, what) != NULL)
		lua_error(L);
}

/*** Get the bytes of the ggml buffers of this process.
 * Each memory is a device name, such as "Vulkan0", or "CPU" for host
 * memory, pinned or not. weights are buffers of `backend:load` and
 * `backend:load_safetensors`, tensors those of `backend:tensor` and
 * other leaves, compute the buffers of graphs.
 * @treturn table `{ [memory] = { weights, tensors, compute, peak,
 * peak_compute } }`; peak is the most of the three at once, and
 * peak_compute the most compute at once, since `memory_reset`
 * @function memory
 */
static int
l_memory(lua_State *L)
{
	int i, p;

	lua_createtable(L, 0, nkeys);
	for (i = 0; i < nkeys; i++) {
		lua_createtable(L, 0, MEM_NPART + 2);
		for (p = 0; p < MEM_NPART; p++) {
			lua_pushinteger(L, (lua_Integer)keys[i].cur[p]);
			lua_setfield(L, -2, part_names[p]);
		}
		lua_pushinteger(L, (lua_Integer)keys[i].peak);
		lua_setfield(L, -2, "peak");
		lua_pushinteger(L, (lua_Integer)keys[i].peak_compute);
		lua_setfield(L, -2, "peak_compute");
		lua_setfield(L, -2, keys[i].name);
	}
	return 1;
}

/*** Set the peaks of `memory` to the bytes in use now.
 * @function memory_reset
 */
static int
l_memory_reset(lua_State *L)
{
	int i;

	(void)L;
	for (i = 0; i < nkeys; i++) {
		keys[i].peak = key_total(&keys[i]);
		keys[i].peak_compute = keys[i].cur[MEM_COMPUTE];
	}
	return 0;
}

/*** Set the memory guard.
 * A load, tensor or graph plan past the guard raises an error that
 * starts with "memory budget:". On a device, the new bytes plus margin
 * must fit the free memory that ggml reports. With a limit, the lgml
 * bytes of each memory stay at most the limit. The guard does not check
 * a graph that runs through the scheduler (weights on a device and CPU).
 * @tparam[opt] table opts `margin`: bytes that stay free on a device
 * (default 256 MiB); `limit`: the most bytes of each memory, 0 for no
 * limit (default 0)
 * @treturn table the settings before the call, `{ margin, limit }`
 * @function memory_guard
 */
static int
l_memory_guard(lua_State *L)
{
	lua_Integer v;

	lua_createtable(L, 0, 2);
	lua_pushinteger(L, (lua_Integer)guard_margin);
	lua_setfield(L, -2, "margin");
	lua_pushinteger(L, (lua_Integer)guard_limit);
	lua_setfield(L, -2, "limit");
	if (lua_isnoneornil(L, 1))
		return 1;
	luaL_checktype(L, 1, LUA_TTABLE);
	if (lua_getfield(L, 1, "margin") != LUA_TNIL) {
		v = luaL_checkinteger(L, -1);
		luaL_argcheck(L, v >= 0, 1, "negative margin");
		guard_margin = (size_t)v;
	}
	lua_pop(L, 1);
	if (lua_getfield(L, 1, "limit") != LUA_TNIL) {
		v = luaL_checkinteger(L, -1);
		luaL_argcheck(L, v >= 0, 1, "negative limit");
		guard_limit = (size_t)v;
	}
	lua_pop(L, 1);
	return 1;
}

/*** Check that more bytes fit on a backend, as the memory guard does.
 * @tparam backend be the backend
 * @tparam int bytes the new bytes
 * @tparam[opt="allocation"] string what the name for the message
 * @treturn[1] boolean true
 * @treturn[2] nil
 * @treturn[2] string the error, which starts with "memory budget:"
 * @function memory_check
 */
static int
l_memory_check(lua_State *L)
{
	struct backend *b = lgml_check_backend(L, 1);
	lua_Integer n = luaL_checkinteger(L, 2);
	const char *what = luaL_optstring(L, 3, "allocation");

	if (n > 0 && lgml_mem_check(L, ggml_backend_get_default_buffer_type(b->be), (size_t)n,
	    what) != NULL) {
		luaL_pushfail(L);
		lua_insert(L, -2);
		return 2;
	}
	lua_pushboolean(L, 1);
	return 1;
}

void
lgml_open_mem(lua_State *L)
{
	static const luaL_Reg fns[] = {
		{ "memory", l_memory },
		{ "memory_reset", l_memory_reset },
		{ "memory_guard", l_memory_guard },
		{ "memory_check", l_memory_check },
		{ NULL, NULL }
	};

	luaL_setfuncs(L, fns, 0);
}
