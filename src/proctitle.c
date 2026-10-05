/*** Tensor graphs on ggml backends.
 * @module lgml
 */

/*
 * The process title: the name that ps, top and the system log show in
 * place of the interpreter.
 */

#if defined(__linux__)
#define _GNU_SOURCE
#include <sys/prctl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

#if defined(__linux__)
/*
 * Find the argv strings of the process: fields 48 and 49 of
 * /proc/self/stat, counted from 1. The comm field (2) can hold spaces,
 * so the count starts after its closing parenthesis.
 */
static int
argv_area(unsigned long *start, unsigned long *end)
{
	char buf[4096], *p, *tok, *save;
	FILE *f;
	size_t n;
	int field;

	if ((f = fopen("/proc/self/stat", "r")) == NULL)
		return -1;
	n = fread(buf, 1, sizeof(buf) - 1, f);
	fclose(f);
	buf[n] = '\0';
	if ((p = strrchr(buf, ')')) == NULL)
		return -1;
	field = 2;
	for (tok = strtok_r(p + 1, " ", &save); tok != NULL; tok = strtok_r(NULL, " ", &save)) {
		field++;
		if (field == 48)
			*start = strtoul(tok, NULL, 10);
		else if (field == 49) {
			*end = strtoul(tok, NULL, 10);
			return *end > *start ? 0 : -1;
		}
	}
	return -1;
}
#endif

/*** Set the process title.
 * On Linux it sets the thread name (at most 15 bytes, as top shows it)
 * and writes the title over the command line, cut to its length. On the
 * BSDs it calls setproctitle(3). Elsewhere it does nothing.
 * @tparam string title the title
 * @tparam[opt] string name the short name for the thread name (default
 * the title)
 * @function set_proc_title
 */
static int
l_set_proc_title(lua_State *L)
{
	const char *s = luaL_checkstring(L, 1);
#if defined(__linux__)
	const char *name = luaL_optstring(L, 2, s);
	unsigned long start = 0, end = 0;

	prctl(PR_SET_NAME, name, 0, 0, 0);
	if (argv_area(&start, &end) == 0) {
		char *a = (char *)start;
		size_t len = end - start, k = strlen(s);

		if (k >= len)
			k = len - 1;
		memcpy(a, s, k);
		memset(a + k, 0, len - k);
	}
#elif defined(__OpenBSD__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
	setproctitle("-%s", s);
#else
	(void)s;
#endif
	return 0;
}

void
lgml_open_proctitle(lua_State *L)
{
	lua_pushcfunction(L, l_set_proc_title);
	lua_setfield(L, -2, "set_proc_title");
}
