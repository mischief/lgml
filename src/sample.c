/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Sampling from logits. The chain follows llama.cpp: penalties, top_k,
 * top_p, min_p, temperature, then a draw. Logits stay in C buffers.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"
#include "grammar.h"

#define SAMPLER_MT	"lgml_sampler"
#define TOP_P_START	256	/* first partial sort size for top_p */
#define MASK_TOP	512	/* tokens checked first against a grammar */

struct cand {
	float		 logit;
	int32_t		 id;
};

struct sampler {
	/* Parameters. */
	float		 temp;
	int32_t		 top_k;
	float		 top_p;
	float		 min_p;
	float		 rep;
	float		 freq;
	float		 pres;
	int32_t		 last_n;

	/* logit_bias: nbias pairs of token id and bias. */
	int32_t		*bias_id;
	float		*bias;
	int32_t		 nbias;

	uint64_t	 rng[4];

	/* Buffers, sized for vocab entries. */
	int64_t		 vocab;
	float		*logits;
	void		*raw;		/* f16/bf16 row before conversion */
	struct cand	*cand;
	int32_t		*count;		/* occurrences of each id in history */

	/* Penalty history: a ring of last_n ids. */
	int32_t		*hist;
	int32_t		 hist_len;
	int32_t		 hist_pos;

	/* Grammar; the vocab is in the first uservalue. */
	struct grammar	*g;
	struct gvocab	*gv;
	uint8_t		*mask;

	int		 closed;
};

/* xoshiro256** seeded with splitmix64. */
static uint64_t
splitmix64(uint64_t *x)
{
	uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);

	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}

static uint64_t
rotl(uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

static uint64_t
rng_next(uint64_t s[4])
{
	uint64_t r = rotl(s[1] * 5, 7) * 9;
	uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];
	s[2] ^= t;
	s[3] = rotl(s[3], 45);
	return r;
}

/* Uniform double in [0, 1). */
static double
rng_double(uint64_t s[4])
{
	return (double)(rng_next(s) >> 11) * 0x1.0p-53;
}

static void
rng_seed(uint64_t s[4], uint64_t seed)
{
	int i;

	for (i = 0; i < 4; i++)
		s[i] = splitmix64(&seed);
}

static struct sampler *
check_sampler(lua_State *L, int idx)
{
	struct sampler *s = luaL_checkudata(L, idx, SAMPLER_MT);

	if (s->closed)
		luaL_error(L, "sampler closed");
	return s;
}

static float
opt_number(lua_State *L, const char *key, float def)
{
	float v = def;

	if (lua_getfield(L, 1, key) != LUA_TNIL) {
		if (!lua_isnumber(L, -1))
			luaL_error(L, "option '%s' must be a number", key);
		v = (float)lua_tonumber(L, -1);
	}
	lua_pop(L, 1);
	return v;
}

static int32_t
opt_int(lua_State *L, const char *key, int32_t def)
{
	lua_Integer v = def;
	int isint;

	if (lua_getfield(L, 1, key) != LUA_TNIL) {
		v = lua_tointegerx(L, -1, &isint);
		if (!isint || v < INT32_MIN || v > INT32_MAX)
			luaL_error(L, "option '%s' must be an integer", key);
	}
	lua_pop(L, 1);
	return (int32_t)v;
}

static void
free_buffers(struct sampler *s)
{
	free(s->logits);
	free(s->raw);
	free(s->cand);
	free(s->count);
	s->logits = NULL;
	s->raw = NULL;
	s->cand = NULL;
	s->count = NULL;
	s->vocab = 0;
}

/* Size the buffers for vocab entries and rebuild the counts. */
static void
ensure_vocab(lua_State *L, struct sampler *s, int64_t vocab)
{
	int32_t i, id;

	if (vocab == s->vocab)
		return;
	free_buffers(s);
	s->logits = malloc((size_t)vocab * sizeof(*s->logits));
	s->raw = malloc((size_t)vocab * sizeof(uint16_t));
	s->cand = malloc((size_t)vocab * sizeof(*s->cand));
	s->count = calloc((size_t)vocab, sizeof(*s->count));
	if (s->logits == NULL || s->raw == NULL || s->cand == NULL ||
	    s->count == NULL) {
		free_buffers(s);
		luaL_error(L, "out of memory");
	}
	s->vocab = vocab;
	for (i = 0; i < s->hist_len; i++) {
		id = s->hist[i];
		if (id < vocab)
			s->count[id]++;
	}
}

/* Read opts.logit_bias, a table of token id to bias, into s. */
static void
read_bias(lua_State *L, struct sampler *s)
{
	lua_Integer id;
	int32_t n = 0, k = 0;
	int isint;

	if (lua_getfield(L, 1, "logit_bias") == LUA_TNIL) {
		lua_pop(L, 1);
		return;
	}
	luaL_argcheck(L, lua_istable(L, -1), 1, "logit_bias must be a table");
	lua_pushnil(L);
	while (lua_next(L, -2) != 0) {
		n++;
		lua_pop(L, 1);
	}
	if (n > 0) {
		s->bias_id = malloc((size_t)n * sizeof(*s->bias_id));
		s->bias = malloc((size_t)n * sizeof(*s->bias));
		if (s->bias_id == NULL || s->bias == NULL)
			luaL_error(L, "out of memory");
	}
	lua_pushnil(L);
	while (lua_next(L, -2) != 0 && k < n) {
		id = lua_tointegerx(L, -2, &isint);
		if (!isint || id < 0 || id > INT32_MAX || !lua_isnumber(L, -1))
			luaL_error(L, "logit_bias: need token id keys and number values");
		s->bias_id[k] = (int32_t)id;
		s->bias[k] = (float)lua_tonumber(L, -1);
		k++;
		lua_pop(L, 1);
	}
	s->nbias = k;
	lua_pop(L, 1);
}

/*** Make a token sampler.
 * The chain follows llama.cpp: penalties, `top_k`, `top_p`, `min_p`,
 * temperature, then a draw. opts fields and defaults: temperature (0.8;
 * <= 0 picks the largest logit), `top_k` (40; 0 is off), `top_p` (0.95),
 * `min_p` (0.05), `repeat_penalty` (1), `frequency_penalty` (0),
 * `presence_penalty` (0), `repeat_last_n` (64), seed (from the clock).
 * `logit_bias` is a table of token id to a number that the sampler adds
 * to the logit of that id before the penalties.
 * grammar is GBNF text; tokens must then fit it (see `sampler:sample`).
 * It needs tokenizer, an object with `n_vocab`, `token` and `special`
 * methods; `grammar_root` names the start rule (default "root").
 * @tparam[opt] table opts options
 * @treturn sampler the sampler
 * @raise if an option has the wrong type or range, or the grammar is bad
 * @function sampler
 */
static int
l_sampler(lua_State *L)
{
	struct sampler *s;
	lua_Integer seed;
	int isint;
	const char *text, *root = "root";
	char err[160];
	size_t len;

	if (lua_isnoneornil(L, 1)) {
		lua_settop(L, 0);
		lua_newtable(L);
	}
	luaL_checktype(L, 1, LUA_TTABLE);
	s = lua_newuserdatauv(L, sizeof(*s), 1);
	memset(s, 0, sizeof(*s));
	s->closed = 1;
	luaL_setmetatable(L, SAMPLER_MT);

	/* Defaults are the llama.cpp common defaults. */
	s->temp = opt_number(L, "temperature", 0.8f);
	s->top_k = opt_int(L, "top_k", 40);
	s->top_p = opt_number(L, "top_p", 0.95f);
	s->min_p = opt_number(L, "min_p", 0.05f);
	s->rep = opt_number(L, "repeat_penalty", 1.0f);
	s->freq = opt_number(L, "frequency_penalty", 0.0f);
	s->pres = opt_number(L, "presence_penalty", 0.0f);
	s->last_n = opt_int(L, "repeat_last_n", 64);
	luaL_argcheck(L, s->last_n >= 0, 1, "repeat_last_n must be >= 0");
	luaL_argcheck(L, s->rep > 0, 1, "repeat_penalty must be > 0");

	if (lua_getfield(L, 1, "seed") == LUA_TNIL) {
		seed = (lua_Integer)time(NULL) ^ (lua_Integer)clock() ^
		    (lua_Integer)(uintptr_t)s;
	} else {
		seed = lua_tointegerx(L, -1, &isint);
		if (!isint)
			return luaL_error(L, "option 'seed' must be an integer");
	}
	lua_pop(L, 1);
	rng_seed(s->rng, (uint64_t)seed);

	if (s->last_n > 0) {
		s->hist = malloc((size_t)s->last_n * sizeof(*s->hist));
		if (s->hist == NULL)
			return luaL_error(L, "out of memory");
	}
	s->closed = 0;
	read_bias(L, s);
	if (lua_getfield(L, 1, "grammar") == LUA_TNIL) {
		lua_pop(L, 1);
		return 1;
	}
	text = luaL_checklstring(L, -1, &len);
	if (lua_getfield(L, 1, "grammar_root") != LUA_TNIL)
		root = luaL_checkstring(L, -1);
	if (lua_getfield(L, 1, "tokenizer") == LUA_TNIL)
		return luaL_error(L, "option 'grammar' needs 'tokenizer'");
	s->gv = gvocab_push(L, -1);
	lua_setiuservalue(L, 2, 1);
	if ((s->g = grammar_new(text, len, root, s->gv, 0, err,
	    sizeof(err))) == NULL)
		return luaL_error(L, "grammar: %s", err);
	lua_settop(L, 2);
	return 1;
}

/*** Samples token ids from logits.
 * Logits stay in C buffers.
 * @type sampler
 */

/*** Free the buffers.
 * The `__gc` and `__close` metamethods call this function.
 * @function sampler:close
 */
static int
sampler_close(lua_State *L)
{
	struct sampler *s = luaL_checkudata(L, 1, SAMPLER_MT);

	if (s->closed)
		return 0;
	free_buffers(s);
	free(s->hist);
	s->hist = NULL;
	free(s->bias_id);
	free(s->bias);
	s->bias_id = NULL;
	s->bias = NULL;
	s->nbias = 0;
	grammar_free(s->g);
	s->g = NULL;
	free(s->mask);
	s->mask = NULL;
	s->closed = 1;
	return 0;
}

/*** Add a token id to the penalty history.
 * The history keeps the last `repeat_last_n` ids. With a grammar, the
 * token also advances the grammar.
 * @tparam int id token id, from 0
 * @raise if the id is negative or too large, or does not fit the grammar
 * @function sampler:accept
 */
static int
sampler_accept(lua_State *L)
{
	struct sampler *s = check_sampler(L, 1);
	lua_Integer v = luaL_checkinteger(L, 2);
	int32_t id, old;

	luaL_argcheck(L, v >= 0 && v <= INT32_MAX, 2, "bad token id");
	id = (int32_t)v;
	if (s->g != NULL && grammar_accept_token(s->g, s->gv, id) == -1)
		return luaL_error(L, "grammar: %s", grammar_error(s->g));
	if (s->last_n == 0)
		return 0;
	if (s->hist_len == s->last_n) {
		old = s->hist[s->hist_pos];
		if (old < s->vocab)
			s->count[old]--;
	} else
		s->hist_len++;
	s->hist[s->hist_pos] = id;
	s->hist_pos = (s->hist_pos + 1) % s->last_n;
	if (id < s->vocab)
		s->count[id]++;
	return 0;
}

/*** Clear the penalty history.
 * @function sampler:reset
 */
static int
sampler_reset(lua_State *L)
{
	struct sampler *s = check_sampler(L, 1);

	if (s->count != NULL)
		memset(s->count, 0, (size_t)s->vocab * sizeof(*s->count));
	s->hist_len = 0;
	s->hist_pos = 0;
	return 0;
}

/* Penalize each distinct id in the history once. */
static void
apply_penalties(struct sampler *s)
{
	float *l = s->logits;
	int32_t i, id, c;

	if (s->hist_len == 0 ||
	    (s->rep == 1.0f && s->freq == 0.0f && s->pres == 0.0f))
		return;
	for (i = 0; i < s->hist_len; i++) {
		id = s->hist[i];
		if (id >= s->vocab || (c = s->count[id]) <= 0)
			continue;
		if (l[id] <= 0)
			l[id] *= s->rep;
		else
			l[id] /= s->rep;
		l[id] -= (float)c * s->freq + s->pres;
		s->count[id] = -c;	/* mark as done */
	}
	for (i = 0; i < s->hist_len; i++) {
		id = s->hist[i];
		if (id < s->vocab && s->count[id] < 0)
			s->count[id] = -s->count[id];
	}
}

/* Descending logit, then ascending id. */
static int
cand_cmp(const void *pa, const void *pb)
{
	const struct cand *a = pa, *b = pb;

	if (a->logit != b->logit)
		return a->logit < b->logit ? 1 : -1;
	return (a->id > b->id) - (a->id < b->id);
}

static void
cand_swap(struct cand *a, struct cand *b)
{
	struct cand t = *a;

	*a = *b;
	*b = t;
}

/* Quickselect: move the k largest to c[0..k), unordered. O(n) expected. */
static void
select_top(struct cand *c, int64_t n, int64_t k)
{
	int64_t lo = 0, hi = n - 1, i, j, mid;
	struct cand pivot;

	while (lo < hi) {
		mid = lo + (hi - lo) / 2;
		if (cand_cmp(&c[mid], &c[lo]) < 0)
			cand_swap(&c[mid], &c[lo]);
		if (cand_cmp(&c[hi], &c[lo]) < 0)
			cand_swap(&c[hi], &c[lo]);
		if (cand_cmp(&c[hi], &c[mid]) < 0)
			cand_swap(&c[hi], &c[mid]);
		pivot = c[mid];
		i = lo;
		j = hi;
		while (i <= j) {
			while (cand_cmp(&c[i], &pivot) < 0)
				i++;
			while (cand_cmp(&c[j], &pivot) > 0)
				j--;
			if (i <= j) {
				cand_swap(&c[i], &c[j]);
				i++;
				j--;
			}
		}
		if (k - 1 <= j)
			hi = j;
		else if (k - 1 >= i)
			lo = i;
		else
			break;
	}
}

/* Sort the k largest of c[0..n) into c[0..k), descending. */
static void
sort_top(struct cand *c, int64_t n, int64_t k)
{
	if (k < n)
		select_top(c, n, k);
	qsort(c, (size_t)k, sizeof(*c), cand_cmp);
}

/* Move h[0] down a min-heap of k entries. The root is the worst entry. */
static void
sift_down(struct cand *h, int64_t k)
{
	int64_t i = 0, j;

	for (;;) {
		j = 2 * i + 1;
		if (j >= k)
			break;
		if (j + 1 < k && cand_cmp(&h[j + 1], &h[j]) > 0)
			j++;
		if (cand_cmp(&h[j], &h[i]) <= 0)
			break;
		cand_swap(&h[i], &h[j]);
		i = j;
	}
}

/*
 * Put the k largest of l[0..n) in h[0..k), sorted descending. Most
 * entries fail one compare against the heap root, so this is near O(n).
 */
static void
heap_top(const float *l, int64_t n, struct cand *h, int64_t k)
{
	int64_t i;

	for (i = 0; i < k; i++) {
		h[i].logit = l[i];
		h[i].id = (int32_t)i;
	}
	qsort(h, (size_t)k, sizeof(*h), cand_cmp);
	for (i = 0; i < k / 2; i++)
		cand_swap(&h[i], &h[k - 1 - i]);	/* ascending is a min-heap */
	for (i = k; i < n; i++) {
		/* A tie loses: its id is larger. */
		if (l[i] <= h[0].logit)
			continue;
		h[0].logit = l[i];
		h[0].id = (int32_t)i;
		sift_down(h, k);
	}
	qsort(h, (size_t)k, sizeof(*h), cand_cmp);
}

static float
max_logit(const struct cand *c, int64_t n)
{
	float m = -INFINITY;
	int64_t i;

	for (i = 0; i < n; i++)
		if (c[i].logit > m)
			m = c[i].logit;
	return m;
}

/*
 * Keep the smallest top set with probability mass >= p. Sort a small
 * prefix first and grow it only when the mass is not reached.
 */
static int64_t
apply_top_p(struct cand *c, int64_t n, int sorted, float p)
{
	float m = sorted ? c[0].logit : max_logit(c, n);
	float sum = 0, cum = 0;
	int64_t i, k;

	for (i = 0; i < n; i++)
		sum += expf(c[i].logit - m);
	k = sorted ? n : (n < TOP_P_START ? n : TOP_P_START);
	if (!sorted)
		sort_top(c, n, k);
	for (i = 0; i < n; i++) {
		if (i == k) {
			/* c[0..k) is the sorted top; extend into the tail. */
			k = k * 4 < n ? k * 4 : n;
			sort_top(c + i, n - i, k - i);
		}
		cum += expf(c[i].logit - m) / sum;
		if (cum >= p)
			return i + 1;
	}
	return n;
}

/* Keep ids with probability >= p times the top probability. */
static int64_t
apply_min_p(struct cand *c, int64_t n, int sorted, float p)
{
	float thr;
	int64_t i, j;

	if (sorted) {
		thr = c[0].logit + logf(p);
		for (i = 1; i < n; i++)
			if (c[i].logit < thr)
				break;
		return i;
	}
	thr = max_logit(c, n) + logf(p);
	for (i = j = 0; i < n; i++)
		if (c[i].logit >= thr)
			c[j++] = c[i];
	return j;
}

/* Draw one id from softmax(c). Overwrites each logit with its weight. */
static int32_t
draw(struct sampler *s, struct cand *c, int64_t n)
{
	float m = max_logit(c, n);
	double sum = 0, run = 0, tgt;
	int64_t i, last = 0;

	for (i = 0; i < n; i++) {
		c[i].logit = expf(c[i].logit - m);
		sum += c[i].logit;
	}
	tgt = sum * rng_double(s->rng);
	/* A zero weight, as from a grammar mask, never wins. */
	for (i = 0; i < n; i++) {
		if (c[i].logit <= 0)
			continue;
		run += c[i].logit;
		last = i;
		if (run >= tgt)
			return c[i].id;
	}
	return c[last].id;
}

/* Read one row of the logits tensor into s->logits. */
static void
read_row(lua_State *L, struct sampler *s, int idx)
{
	struct tensor *t = lgml_check_tensor(L, idx);
	struct ggml_tensor *gt = t->t;
	int64_t vocab, rows, row;
	size_t esz, off;

	if (gt->buffer == NULL)
		luaL_error(L, "tensor has no storage; compute it first");
	if (!ggml_is_contiguous(gt))
		luaL_error(L, "tensor not contiguous; use cont");
	if (gt->type != GGML_TYPE_F32 && gt->type != GGML_TYPE_F16 &&
	    gt->type != GGML_TYPE_BF16)
		luaL_error(L, "logits must be f32, f16 or bf16");
	vocab = gt->ne[0];
	rows = ggml_nelements(gt) / vocab;
	row = luaL_optinteger(L, idx + 1, rows);
	luaL_argcheck(L, row >= 1 && row <= rows, idx + 1, "row out of range");
	ensure_vocab(L, s, vocab);
	esz = ggml_type_size(gt->type);
	off = (size_t)(row - 1) * (size_t)vocab * esz;
	if (gt->type == GGML_TYPE_F32) {
		ggml_backend_tensor_get(gt, s->logits, off, (size_t)vocab * esz);
		return;
	}
	ggml_backend_tensor_get(gt, s->raw, off, (size_t)vocab * esz);
	if (gt->type == GGML_TYPE_F16)
		ggml_fp16_to_fp32_row(s->raw, s->logits, vocab);
	else
		ggml_bf16_to_fp32_row(s->raw, s->logits, vocab);
}

static int32_t
argmax(const float *l, int64_t n)
{
	int64_t i, best = 0;
	float m = l[0];

	for (i = 1; i < n; i++) {
		if (l[i] > m) {
			m = l[i];
			best = i;
		}
	}
	return (int32_t)best;
}

/* Run the chain on s->logits. It leaves s->logits as it is. */
static int32_t
chain(struct sampler *s)
{
	struct cand *c;
	int64_t i, n;
	int sorted = 0;

	n = s->vocab;
	if (s->temp <= 0.0f) {
		/* top_k, top_p and min_p always keep the maximum. */
		return argmax(s->logits, n);
	}
	c = s->cand;
	if (s->top_k > 0 && s->top_k < n) {
		heap_top(s->logits, n, c, s->top_k);
		n = s->top_k;
		sorted = 1;
	} else {
		for (i = 0; i < n; i++) {
			c[i].logit = s->logits[i];
			c[i].id = (int32_t)i;
		}
	}
	if (s->top_p < 1.0f) {
		n = apply_top_p(c, n, sorted, s->top_p);
		sorted = 1;
	}
	if (s->min_p > 0.0f)
		n = apply_min_p(c, n, sorted, s->min_p);
	if (s->temp != 1.0f)
		for (i = 0; i < n; i++)
			c[i].logit /= s->temp;
	return draw(s, c, n);
}

/*
 * With top_k, the chain sees only the k best tokens that fit. Find them
 * in the best m tokens and mask the rest. Returns 0 when fewer fit.
 */
static int
mask_top(lua_State *L, struct sampler *s, int64_t k)
{
	int32_t ids[MASK_TOP];
	uint8_t ok[MASK_TOP];
	float keep[MASK_TOP];
	int64_t i, m, nok = 0;

	m = s->vocab < MASK_TOP ? s->vocab : MASK_TOP;
	if (k < 1 || k > m / 4)
		return 0;
	heap_top(s->logits, s->vocab, s->cand, m);
	for (i = 0; i < m; i++)
		ids[i] = s->cand[i].id;
	if (grammar_check(s->g, s->gv, ids, (size_t)m, ok) == -1)
		luaL_error(L, "grammar: %s", grammar_error(s->g));
	for (i = 0; i < m; i++) {
		keep[i] = ok[i] ? s->logits[ids[i]] : -INFINITY;
		nok += ok[i];
	}
	if (nok < k)
		return 0;
	for (i = 0; i < s->vocab; i++)
		s->logits[i] = -INFINITY;
	for (i = 0; i < m; i++)
		s->logits[ids[i]] = keep[i];
	return 1;
}

/*
 * Keep id when it fits the grammar, as llama.cpp does. Else mask each
 * token that does not fit and run the chain again.
 */
static int32_t
constrain(lua_State *L, struct sampler *s, int32_t id)
{
	int64_t i, nok = 0;
	int r;

	if (id < s->gv->n) {
		if ((r = grammar_fits(s->g, s->gv, id)) == -1)
			luaL_error(L, "grammar: %s", grammar_error(s->g));
		if (r == 1)
			return id;
	}
	if (mask_top(L, s, s->temp <= 0.0f ? 1 : s->top_k))
		return chain(s);
	if (s->mask == NULL &&
	    (s->mask = malloc(s->gv->n > 0 ? (size_t)s->gv->n : 1)) == NULL)
		luaL_error(L, "out of memory");
	if (grammar_mask(s->g, s->gv, s->mask) == -1)
		luaL_error(L, "grammar: %s", grammar_error(s->g));
	for (i = 0; i < s->vocab; i++) {
		if (i < s->gv->n && s->mask[i])
			nok++;
		else
			s->logits[i] = -INFINITY;
	}
	if (nok == 0)
		luaL_error(L, "grammar: no token fits");
	return chain(s);
}

/*** Sample a token id from one row of logits.
 * The innermost dimension of logits is the vocabulary. Penalties from the
 * history apply first. With a grammar, the chain runs first; when its
 * token does not fit, each token that does not fit is masked and the
 * chain runs again. An end-of-generation token fits only where the
 * grammar can end.
 * @tparam tensor logits computed, contiguous f32, f16 or bf16 tensor
 * @tparam[opt] int row row number from 1; the default is the last row
 * @treturn int token id, from 0
 * @raise on a bad tensor or row, or when no token fits the grammar
 * @function sampler:sample
 */
static int
sampler_sample(lua_State *L)
{
	struct sampler *s = check_sampler(L, 1);
	int32_t id, i;

	read_row(L, s, 2);
	for (i = 0; i < s->nbias; i++)
		if (s->bias_id[i] < s->vocab)
			s->logits[s->bias_id[i]] += s->bias[i];
	apply_penalties(s);
	id = chain(s);
	if (s->g != NULL)
		id = constrain(L, s, id);
	lua_pushinteger(L, id);
	return 1;
}

/*** Log probabilities from the last `sampler:sample` call.
 * They are the log-softmax of the logits after `logit_bias` and the
 * penalties, before temperature and truncation. When the grammar masked
 * tokens, the masked tokens have -inf.
 * @tparam int id a token id, from 0
 * @tparam[opt=0] int k the number of most likely tokens to list
 * @treturn number the log probability of id
 * @treturn {{int,number},...} the k most likely `{ id, logprob }` pairs,
 * most likely first
 * @raise before the first sample, or when id or k is out of range
 * @function sampler:logprobs
 */
static int
sampler_logprobs(lua_State *L)
{
	struct sampler *s = check_sampler(L, 1);
	lua_Integer id = luaL_checkinteger(L, 2);
	lua_Integer k = luaL_optinteger(L, 3, 0);
	const float *l = s->logits;
	double sum = 0, lse;
	float m = -INFINITY;
	int64_t i;

	if (l == NULL)
		return luaL_error(L, "no logits: sample first");
	luaL_argcheck(L, id >= 0 && id < s->vocab, 2, "bad token id");
	luaL_argcheck(L, k >= 0 && k <= s->vocab, 3, "bad count");
	for (i = 0; i < s->vocab; i++)
		if (l[i] > m)
			m = l[i];
	for (i = 0; i < s->vocab; i++)
		sum += exp((double)l[i] - m);
	lse = m + log(sum);
	lua_pushnumber(L, (double)l[id] - lse);
	lua_createtable(L, (int)k, 0);
	if (k > 0)
		heap_top(l, s->vocab, s->cand, k);
	for (i = 0; i < k; i++) {
		lua_createtable(L, 2, 0);
		lua_pushinteger(L, s->cand[i].id);
		lua_rawseti(L, -2, 1);
		lua_pushnumber(L, (double)s->cand[i].logit - lse);
		lua_rawseti(L, -2, 2);
		lua_rawseti(L, -2, i + 1);
	}
	return 2;
}

/*** List the token ids that fit the grammar now.
 * @treturn {int,...} token ids, from 0, ascending
 * @raise without a grammar
 * @function sampler:allowed
 */
static int
sampler_allowed(lua_State *L)
{
	struct sampler *s = check_sampler(L, 1);
	lua_Integer k = 0;
	int32_t i;

	if (s->g == NULL)
		return luaL_error(L, "sampler has no grammar");
	if (s->mask == NULL &&
	    (s->mask = malloc(s->gv->n > 0 ? (size_t)s->gv->n : 1)) == NULL)
		return luaL_error(L, "out of memory");
	if (grammar_mask(s->g, s->gv, s->mask) == -1)
		return luaL_error(L, "grammar: %s", grammar_error(s->g));
	lua_newtable(L);
	for (i = 0; i < s->gv->n; i++) {
		if (s->mask[i]) {
			lua_pushinteger(L, i);
			lua_rawseti(L, -2, ++k);
		}
	}
	return 1;
}

/*** Test whether the grammar can end here.
 * @treturn boolean true without a grammar, or when the grammar can end
 * @function sampler:can_end
 */
static int
sampler_can_end(lua_State *L)
{
	struct sampler *s = check_sampler(L, 1);

	lua_pushboolean(L, s->g == NULL || grammar_can_end(s->g));
	return 1;
}

/*
 * Philox 4x32-10 normal draws, as the torch CUDA generator makes them
 * and stable-diffusion.cpp copies on the CPU: draw i takes counter
 * (offset, 0, i, 0) and the seed as key, then Box-Muller in f32.
 */
static void
philox(uint32_t c[4], uint32_t k0, uint32_t k1)
{
	uint64_t a, b;
	int r;

	for (r = 0; r < 10; r++) {
		a = (uint64_t)c[0] * 0xD2511F53u;
		b = (uint64_t)c[2] * 0xCD9E8D57u;
		c[0] = (uint32_t)(b >> 32) ^ c[1] ^ k0;
		c[1] = (uint32_t)b;
		c[2] = (uint32_t)(a >> 32) ^ c[3] ^ k1;
		c[3] = (uint32_t)a;
		k0 += 0x9E3779B9u;
		k1 += 0xBB67AE85u;
	}
}

/*** Normal random numbers from the Philox generator.
 * Draw i uses counter (offset, 0, i, 0). The values match the "cuda"
 * RNG of stable-diffusion.cpp and of the AUTOMATIC1111 web UI.
 * @tparam int seed the seed, 64 bits
 * @tparam int n the count, 1 to 2^28
 * @tparam[opt=0] int offset the call count of the generator
 * @treturn string n f32 values as raw bytes
 * @raise if n or offset is out of range
 * @function randn_philox
 */
static int
l_randn_philox(lua_State *L)
{
	uint64_t seed = (uint64_t)luaL_checkinteger(L, 1);
	lua_Integer n = luaL_checkinteger(L, 2);
	lua_Integer off = luaL_optinteger(L, 3, 0);
	const float inv = 2.3283064e-10f, inv2pi = 2.3283064e-10f * 6.2831855f;
	uint32_t c[4];
	luaL_Buffer b;
	float *out, u, v, s;
	lua_Integer i;

	luaL_argcheck(L, n >= 1 && n <= (1 << 28), 2, "count out of range");
	luaL_argcheck(L, off >= 0 && off <= UINT32_MAX, 3, "offset out of range");
	out = (float *)(void *)luaL_buffinitsize(L, &b, (size_t)n * sizeof(float));
	for (i = 0; i < n; i++) {
		c[0] = (uint32_t)off;
		c[1] = 0;
		c[2] = (uint32_t)i;
		c[3] = 0;
		philox(c, (uint32_t)seed, (uint32_t)(seed >> 32));
		u = (float)c[0] * inv + inv / 2;
		v = (float)c[1] * inv2pi + inv2pi / 2;
		/* The reference rounds only s to f32. */
		s = (float)sqrt(-2.0 * log(u));
		out[i] = (float)(s * sin(v));
	}
	luaL_pushresultsize(&b, (size_t)n * sizeof(float));
	return 1;
}

static const luaL_Reg sampler_meta[] = {
	{ "__gc", sampler_close },
	{ "__close", sampler_close },
	{ NULL, NULL }
};

static const luaL_Reg sampler_methods[] = {
	{ "sample", sampler_sample },
	{ "accept", sampler_accept },
	{ "reset", sampler_reset },
	{ "allowed", sampler_allowed },
	{ "logprobs", sampler_logprobs },
	{ "can_end", sampler_can_end },
	{ "close", sampler_close },
	{ NULL, NULL }
};

void
lgml_open_sample(lua_State *L)
{
	luaL_newmetatable(L, SAMPLER_MT);
	luaL_setfuncs(L, sampler_meta, 0);
	lua_newtable(L);
	luaL_setfuncs(L, sampler_methods, 0);
	lua_setfield(L, -2, "__index");
	lua_pop(L, 1);
	lua_pushcfunction(L, l_sampler);
	lua_setfield(L, -2, "sampler");
	lgml_open_grammar(L);
	lua_pushcfunction(L, l_randn_philox);
	lua_setfield(L, -2, "randn_philox");
}
