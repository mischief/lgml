/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Token id to bytes table for byte-level BPE models on Hugging Face:
 * vocab.json (piece -> id) plus added_tokens.json. Normal pieces map
 * back to bytes with the GPT-2 byte to unicode table. Added tokens
 * keep their text and count as special.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

#define VOCAB_MT	"lgml_vocab"
#define VOCAB_MAX_FILE	((off_t)64 << 20)
#define VOCAB_MAX_ID	(1 << 20)
#define VOCAB_MAX_PIECE	4096

struct vocab {
	int		 n;
	char		**piece;
	uint32_t	*len;
	unsigned char	*special;
};

static int16_t cp_to_byte[512];

/* Inverse of the GPT-2 bytes_to_unicode table. */
static void
byte_table_init(void)
{
	int b, n = 0;

	for (b = 0; b < 512; b++)
		cp_to_byte[b] = -1;
	for (b = 0; b < 256; b++) {
		if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)
			cp_to_byte[b] = (int16_t)b;
		else
			cp_to_byte[256 + n++] = (int16_t)b;
	}
}

static struct vocab *
check_vocab(lua_State *L, int idx)
{
	struct vocab *v = luaL_checkudata(L, idx, VOCAB_MT);

	if (v->piece == NULL)
		luaL_error(L, "vocab closed");
	return v;
}

static void
vocab_free(struct vocab *v)
{
	int i;

	if (v->piece != NULL)
		for (i = 0; i < v->n; i++)
			free(v->piece[i]);
	free(v->piece);
	free(v->len);
	free(v->special);
	memset(v, 0, sizeof(*v));
}

/* Read a whole file into a new userdata on the stack. */
static const char *
read_file(lua_State *L, const char *path, size_t *lenp)
{
	struct stat sb;
	char *buf;
	FILE *f;
	int ok;

	if (stat(path, &sb) != 0)
		luaL_error(L, "%s: %s", path, strerror(errno));
	if (sb.st_size <= 0 || sb.st_size > VOCAB_MAX_FILE)
		luaL_error(L, "%s: bad size", path);
	buf = lua_newuserdatauv(L, (size_t)sb.st_size, 0);
	if ((f = fopen(path, "rb")) == NULL)
		luaL_error(L, "%s: %s", path, strerror(errno));
	ok = fread(buf, 1, (size_t)sb.st_size, f) == (size_t)sb.st_size;
	fclose(f);
	if (!ok)
		luaL_error(L, "%s: short read", path);
	*lenp = (size_t)sb.st_size;
	return buf;
}

/* Decode a UTF-8 code point. Returns its length, or 0 if invalid. */
static size_t
utf8_cp(const unsigned char *s, size_t n, uint32_t *cp)
{
	if (s[0] < 0x80) {
		*cp = s[0];
		return 1;
	}
	if ((s[0] & 0xe0) == 0xc0 && n >= 2 && (s[1] & 0xc0) == 0x80) {
		*cp = (uint32_t)(s[0] & 0x1f) << 6 | (s[1] & 0x3f);
		return 2;
	}
	return 0;
}

/* Map piece text to bytes in place. Unknown code points stay as they are. */
static size_t
unbyte(char *s, size_t n)
{
	size_t i = 0, o = 0, k;
	uint32_t cp;

	while (i < n) {
		k = utf8_cp((unsigned char *)s + i, n - i, &cp);
		if (k != 0 && cp < 512 && cp_to_byte[cp] >= 0) {
			s[o++] = (char)cp_to_byte[cp];
			i += k;
		} else {
			s[o++] = s[i++];
		}
	}
	return o;
}

static int
grow(struct vocab *v, int n)
{
	char **p;
	uint32_t *l;
	unsigned char *sp;

	if (n <= v->n)
		return 0;
	p = realloc(v->piece, (size_t)n * sizeof(*p));
	if (p != NULL)
		v->piece = p;
	l = realloc(v->len, (size_t)n * sizeof(*l));
	if (l != NULL)
		v->len = l;
	sp = realloc(v->special, (size_t)n);
	if (sp != NULL)
		v->special = sp;
	if (p == NULL || l == NULL || sp == NULL)
		return -1;
	memset(v->piece + v->n, 0, (size_t)(n - v->n) * sizeof(*p));
	memset(v->len + v->n, 0, (size_t)(n - v->n) * sizeof(*l));
	memset(v->special + v->n, 0, (size_t)(n - v->n));
	v->n = n;
	return 0;
}

/* Parse an object of piece -> id from path into v. */
static void
load_map(lua_State *L, struct vocab *v, const char *path, int special)
{
	const char *buf, *key;
	size_t len, klen;
	lua_Integer id;
	int isnum;
	char *p;

	buf = read_file(L, path, &len);
	lgml_json_decode(L, buf, len, path);
	lua_remove(L, -2);
	if (!lua_istable(L, -1))
		luaL_error(L, "%s: not a JSON object", path);
	lua_pushnil(L);
	while (lua_next(L, -2) != 0) {
		id = lua_tointegerx(L, -1, &isnum);
		if (lua_type(L, -2) != LUA_TSTRING || lua_type(L, -1) != LUA_TNUMBER || !isnum)
			luaL_error(L, "%s: bad entry", path);
		key = lua_tolstring(L, -2, &klen);
		if (klen >= VOCAB_MAX_PIECE)
			luaL_error(L, "%s: piece over %d bytes", path, VOCAB_MAX_PIECE - 1);
		if (id < 0 || id >= VOCAB_MAX_ID)
			luaL_error(L, "%s: id out of range", path);
		if (grow(v, (int)id + 1) != 0 || (p = malloc(klen + 1)) == NULL)
			luaL_error(L, "out of memory");
		memcpy(p, key, klen);
		if (!special)
			klen = unbyte(p, klen);
		p[klen] = '\0';
		free(v->piece[id]);
		v->piece[id] = p;
		v->len[id] = (uint32_t)klen;
		v->special[id] = (unsigned char)special;
		lua_pop(L, 1);
	}
	lua_pop(L, 1);
}

/*** Load a token table for a byte-level BPE model on Hugging Face.
 * Normal pieces map back to bytes with the GPT-2 byte table. Added
 * tokens keep their text and count as special.
 * @tparam string vocab_json path of vocab.json (piece to id)
 * @tparam[opt] string added_tokens_json path of `added_tokens`.json
 * @treturn vocab the table
 * @raise if a file cannot be read or holds bad entries
 * @function vocab
 */
static int
l_vocab(lua_State *L)
{
	const char *vpath = luaL_checkstring(L, 1);
	const char *apath = luaL_optstring(L, 2, NULL);
	struct vocab *v;

	v = lua_newuserdatauv(L, sizeof(*v), 0);
	memset(v, 0, sizeof(*v));
	luaL_setmetatable(L, VOCAB_MT);
	load_map(L, v, vpath, 0);
	if (apath != NULL)
		load_map(L, v, apath, 1);
	if (v->piece == NULL)
		return luaL_error(L, "%s: empty vocab", vpath);
	return 1;
}

/*** Maps token ids to bytes.
 * @type vocab
 */

/*** Free the table.
 * The `__gc` and `__close` metamethods call this function.
 * @function vocab:close
 */
static int
vocab_close(lua_State *L)
{
	vocab_free(luaL_checkudata(L, 1, VOCAB_MT));
	return 0;
}

/*** Get the table size.
 * @treturn int one more than the highest id
 * @function vocab:n
 */
static int
vocab_n(lua_State *L)
{
	lua_pushinteger(L, check_vocab(L, 1)->n);
	return 1;
}

static int
check_id(lua_State *L, const struct vocab *v, int idx)
{
	lua_Integer id = luaL_checkinteger(L, idx);

	return id >= 0 && id < v->n && v->piece[id] != NULL ? (int)id : -1;
}

/*** Get the bytes of a token.
 * @tparam int id token id, from 0
 * @treturn string|nil token bytes, or nil if the id has no token
 * @function vocab:piece
 */
static int
vocab_piece(lua_State *L)
{
	struct vocab *v = check_vocab(L, 1);
	int id = check_id(L, v, 2);

	if (id < 0)
		return 0;
	lua_pushlstring(L, v->piece[id], v->len[id]);
	return 1;
}

/*** Check if a token is special.
 * @tparam int id token id, from 0
 * @treturn bool true for an added token
 * @function vocab:special
 */
static int
vocab_special(lua_State *L)
{
	struct vocab *v = check_vocab(L, 1);
	int id = check_id(L, v, 2);

	lua_pushboolean(L, id >= 0 && v->special[id]);
	return 1;
}

/*** Find the id of a token.
 * Special tokens match first.
 * @tparam string bytes token bytes
 * @treturn int|nil token id, or nil if no token matches
 * @function vocab:id
 */
static int
vocab_id(lua_State *L)
{
	struct vocab *v = check_vocab(L, 1);
	size_t len;
	const char *s = luaL_checklstring(L, 2, &len);
	int pass, i;

	for (pass = 1; pass >= 0; pass--) {
		for (i = 0; i < v->n; i++) {
			if (v->piece[i] != NULL && v->special[i] == pass && v->len[i] == len &&
			    memcmp(v->piece[i], s, len) == 0) {
				lua_pushinteger(L, i);
				return 1;
			}
		}
	}
	return 0;
}

/*** Decode token ids to bytes.
 * @tparam {int,...} ids token ids, from 0
 * @tparam[opt=false] bool special keep special tokens
 * @treturn string bytes
 * @raise on an id with no token
 * @function vocab:decode
 */
static int
vocab_decode(lua_State *L)
{
	struct vocab *v = check_vocab(L, 1);
	int keep = lua_toboolean(L, 3);
	lua_Integer i, n;
	luaL_Buffer b;
	int id;

	luaL_checktype(L, 2, LUA_TTABLE);
	n = luaL_len(L, 2);
	luaL_buffinit(L, &b);
	for (i = 1; i <= n; i++) {
		lua_geti(L, 2, i);
		id = check_id(L, v, -1);
		lua_pop(L, 1);
		if (id < 0)
			return luaL_error(L, "token id out of range at %d", (int)i);
		if (!v->special[id] || keep)
			luaL_addlstring(&b, v->piece[id], v->len[id]);
	}
	luaL_pushresult(&b);
	return 1;
}

static const luaL_Reg vocab_meta[] = {
	{ "__gc", vocab_close },
	{ "__close", vocab_close },
	{ NULL, NULL }
};

static const luaL_Reg vocab_methods[] = {
	{ "close", vocab_close },
	{ "n", vocab_n },
	{ "piece", vocab_piece },
	{ "special", vocab_special },
	{ "id", vocab_id },
	{ "decode", vocab_decode },
	{ NULL, NULL }
};

void
lgml_open_vocab(lua_State *L)
{
	byte_table_init();
	luaL_newmetatable(L, VOCAB_MT);
	luaL_setfuncs(L, vocab_meta, 0);
	lua_newtable(L);
	luaL_setfuncs(L, vocab_methods, 0);
	lua_setfield(L, -2, "__index");
	lua_pop(L, 1);
	lua_pushcfunction(L, l_vocab);
	lua_setfield(L, -2, "vocab");
}
