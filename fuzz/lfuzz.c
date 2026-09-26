/*
 * AFL++ harness: lfuzz target.lua [input ...]
 *
 * The target script returns a function that takes one input string.
 * An error that escapes the function is a bug: the harness aborts, so
 * afl-fuzz records a crash. Targets catch expected errors themselves.
 * Under afl-fuzz the harness runs in persistent mode. With input
 * arguments it runs each file, or each file in a directory, once.
 *
 * Globals for targets: fuzz.file(s [, slot]) writes s to scratch file
 * slot (1 to 4) and returns its path.
 */

/* RTLD_DEFAULT */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <dlfcn.h>
#include <stdint.h>
#include <unistd.h>

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#ifndef __AFL_FUZZ_TESTCASE_LEN
#define AFL_REPLAY_ONLY
#else
/* The AFL++ macros use GNU C. */
#pragma GCC diagnostic ignored "-Wpedantic"
__AFL_FUZZ_INIT();

/* The AFL pass owns these names, so the harness finds them with dlsym. */
static unsigned char **afl_area;
static unsigned int *afl_map_size, *afl_final_loc;
static uint32_t lua_prev;

/*
 * Lua line coverage. Each (chunk, line) is a location. Edges between
 * locations go to the AFL map as in afl-as, above the C edges when
 * AFL_MAP_SIZE makes room.
 */
static void
line_hook(lua_State *L, lua_Debug *ar)
{
	uint32_t lo = 0, span = *afl_map_size, cur;

	if (!lua_getinfo(L, "S", ar))
		return;
	if (*afl_map_size > *afl_final_loc + 4096) {
		lo = *afl_final_loc;
		span = *afl_map_size - lo;
	}
	cur = (uint32_t)(((uintptr_t)ar->source >> 4) * 0x9e3779b1u ^
	    (uint32_t)ar->currentline * 0x85ebca6bu);
	cur ^= cur >> 15;
	(*afl_area)[lo + (cur ^ lua_prev) % span]++;
	lua_prev = cur >> 1;
}
#endif

int luaopen_lgml(lua_State *);

#define NSCRATCH	4

static int scratch_fd[NSCRATCH] = { -1, -1, -1, -1 };
static char scratch[NSCRATCH][64];

static void
scratch_remove(void)
{
	int i;

	for (i = 0; i < NSCRATCH; i++)
		if (scratch_fd[i] >= 0)
			unlink(scratch[i]);
}

/* fuzz.file(s [, slot]): write s to a scratch file and return its path. */
static int
l_file(lua_State *L)
{
	size_t len;
	const char *s = luaL_checklstring(L, 1, &len);
	lua_Integer slot = luaL_optinteger(L, 2, 1) - 1;
	static int registered;

	luaL_argcheck(L, slot >= 0 && slot < NSCRATCH, 2, "slot out of range");
	if (scratch_fd[slot] < 0) {
		snprintf(scratch[slot], sizeof(scratch[slot]), "/tmp/lfuzz.XXXXXX");
		if ((scratch_fd[slot] = mkstemp(scratch[slot])) < 0)
			return luaL_error(L, "mkstemp failed");
		if (!registered++)
			atexit(scratch_remove);
	}
	if (ftruncate(scratch_fd[slot], 0) != 0 ||
	    pwrite(scratch_fd[slot], s, len, 0) != (ssize_t)len)
		return luaL_error(L, "scratch write failed");
	lua_pushstring(L, scratch[slot]);
	return 1;
}

static int
traceback(lua_State *L)
{
	luaL_traceback(L, L, lua_tostring(L, 1), 1);
	return 1;
}

/* Run the target at stack index 2 on s. Abort on an error. */
static void
run(lua_State *L, const char *s, size_t len)
{
	lua_pushvalue(L, 2);
	lua_pushlstring(L, s, len);
	if (lua_pcall(L, 1, 0, 1) != LUA_OK) {
		fprintf(stderr, "lfuzz: %s\n", lua_tostring(L, -1));
		abort();
	}
}

static char *
slurp(const char *path, size_t *len)
{
	FILE *f = fopen(path, "rb");
	char *buf = NULL;
	long n;

	if (f == NULL || fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0 ||
	    fseek(f, 0, SEEK_SET) != 0 || (buf = malloc((size_t)n + 1)) == NULL ||
	    fread(buf, 1, (size_t)n, f) != (size_t)n) {
		fprintf(stderr, "lfuzz: cannot read %s\n", path);
		exit(2);
	}
	fclose(f);
	*len = (size_t)n;
	return buf;
}

/* Run the target on the file at path, or on each file in the directory. */
static void
run_path(lua_State *L, const char *path)
{
	char sub[4096];
	struct dirent *e;
	size_t len;
	char *buf;
	DIR *d;

	if ((d = opendir(path)) == NULL) {
		buf = slurp(path, &len);
		run(L, buf, len);
		free(buf);
		return;
	}
	while ((e = readdir(d)) != NULL) {
		if (e->d_name[0] == '.')
			continue;
		snprintf(sub, sizeof(sub), "%s/%s", path, e->d_name);
		run_path(L, sub);
	}
	closedir(d);
}

int
main(int argc, char **argv)
{
	lua_State *L;
	int i;
#ifndef AFL_REPLAY_ONLY
	size_t len;
	char *buf;
#endif

	if (argc < 2) {
		fprintf(stderr, "usage: lfuzz target.lua [input ...]\n");
		return 2;
	}
	L = luaL_newstate();
	luaL_openlibs(L);
	luaL_requiref(L, "lgml", luaopen_lgml, 0);
	lua_pop(L, 1);
	/* lgml loads cjson on first use. Under afl-fuzz, it must load before the fork server starts. */
	if (luaL_dostring(L, "require(\"cjson\")") != LUA_OK) {
		fprintf(stderr, "lfuzz: %s\n", lua_tostring(L, -1));
		return 2;
	}
	lua_newtable(L);
	lua_pushcfunction(L, l_file);
	lua_setfield(L, -2, "file");
	lua_setglobal(L, "fuzz");

	lua_pushcfunction(L, traceback);
	if (luaL_loadfile(L, argv[1]) != LUA_OK || lua_pcall(L, 0, 1, 1) != LUA_OK) {
		fprintf(stderr, "lfuzz: %s\n", lua_tostring(L, -1));
		return 2;
	}
	if (!lua_isfunction(L, 2)) {
		fprintf(stderr, "lfuzz: %s does not return a function\n", argv[1]);
		return 2;
	}

	if (argc > 2) {
		for (i = 2; i < argc; i++)
			run_path(L, argv[i]);
		lua_close(L);
		return 0;
	}
#ifndef AFL_REPLAY_ONLY
	__AFL_INIT();
	afl_area = dlsym(RTLD_DEFAULT, "__afl_area_ptr");
	afl_map_size = dlsym(RTLD_DEFAULT, "__afl_map_size");
	afl_final_loc = dlsym(RTLD_DEFAULT, "__afl_final_loc");
	if (afl_area != NULL && afl_map_size != NULL && afl_final_loc != NULL)
		lua_sethook(L, line_hook, LUA_MASKLINE, 0);
	else
		fprintf(stderr, "lfuzz: no AFL map symbols; Lua coverage off\n");
	buf = (char *)__AFL_FUZZ_TESTCASE_BUF;
	while (__AFL_LOOP(100000)) {
		len = (size_t)__AFL_FUZZ_TESTCASE_LEN;
		lua_prev = 0;
		run(L, buf, len);
		lua_gc(L, LUA_GCCOLLECT);
	}
#endif
	lua_close(L);
	return 0;
}
