/*
 * JSON for the C files, on lua-cjson. One strict decoder lives in the
 * registry: no NaN, Infinity or hex numbers, and a small depth limit.
 */

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

#define JSON_KEY	"lgml.json"
#define JSON_DEPTH	64

/* Push the decode function of the strict decoder. */
static void
push_decoder(lua_State *L)
{
	if (lua_getfield(L, LUA_REGISTRYINDEX, JSON_KEY) == LUA_TFUNCTION)
		return;
	lua_pop(L, 1);
	lua_getglobal(L, "require");
	lua_pushliteral(L, "cjson");
	lua_call(L, 1, 1);
	lua_getfield(L, -1, "new");
	lua_call(L, 0, 1);
	lua_getfield(L, -1, "decode_invalid_numbers");
	lua_pushboolean(L, 0);
	lua_call(L, 1, 0);
	lua_getfield(L, -1, "decode_max_depth");
	lua_pushinteger(L, JSON_DEPTH);
	lua_call(L, 1, 0);
	lua_getfield(L, -1, "decode");
	lua_pushvalue(L, -1);
	lua_setfield(L, LUA_REGISTRYINDEX, JSON_KEY);
	lua_replace(L, -3);
	lua_pop(L, 1);
}

/* Decode s and push the value. Raise an error named by what on bad JSON. */
void
lgml_json_decode(lua_State *L, const char *s, size_t len, const char *what)
{
	push_decoder(L);
	lua_pushlstring(L, s, len);
	if (lua_pcall(L, 1, 1, 0) != LUA_OK)
		luaL_error(L, "%s: bad JSON: %s", what, lua_tostring(L, -1));
}
