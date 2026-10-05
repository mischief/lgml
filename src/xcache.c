/*** Expert cache state for mixture-of-experts layers.
 * @module lgml
 */

/*
 * The routed experts of a layer live in host memory, and a device cache
 * holds some of them in slots. This file keeps the host side: the slot
 * of each expert, the pick counts, and a graph op that gives the CPU the
 * ids it must compute. Lua copies experts into slots.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <lua.h>
#include <lauxlib.h>

#include <ggml-cpu.h>

#include "lgml.h"

#define XCACHE_MT	"lgml_expert_cache"

struct xcache;

struct xlayer {
	struct xcache	*x;
	int		 nslots;
	int		*slot;		/* expert -> slot, or -1 */
	int		*expert;	/* slot -> expert, or -1 */
	double		*score;		/* decayed pick counts */
	uint32_t	*pend;		/* picks since the last plan */
	int64_t		 ntok;		/* tokens since the last plan */
	int		 fb;		/* an expert not in a slot, or -1 */
};

struct xcache {
	int		 nl;
	int		 ne;
	struct xlayer	*l;
	uint64_t	 hits;
	uint64_t	 misses;
};

static struct xcache *
check_xcache(lua_State *L, int idx)
{
	struct xcache *x = luaL_checkudata(L, idx, XCACHE_MT);

	if (x->l == NULL)
		luaL_error(L, "expert cache is closed");
	return x;
}

static struct xlayer *
check_layer(lua_State *L, struct xcache *x, int idx)
{
	lua_Integer il = luaL_checkinteger(L, idx);

	luaL_argcheck(L, il >= 1 && il <= x->nl, idx, "layer out of range");
	return &x->l[il - 1];
}

static void
free_layers(struct xcache *x)
{
	int i;

	if (x->l == NULL)
		return;
	for (i = 0; i < x->nl; i++) {
		free(x->l[i].slot);
		free(x->l[i].expert);
		free(x->l[i].score);
		free(x->l[i].pend);
	}
	free(x->l);
	x->l = NULL;
}

/* Find an expert that is not in a slot. */
static void
update_fb(struct xlayer *l)
{
	int e;

	l->fb = -1;
	for (e = 0; e < l->x->ne; e++) {
		if (l->slot[e] < 0) {
			l->fb = e;
			return;
		}
	}
}

/*** Make the expert cache state of a model.
 * Each layer starts with no slots and zero scores.
 * @tparam int n_layer the layer count
 * @tparam int n_expert the routed experts of a layer
 * @treturn expert_cache the state
 * @function expert_cache
 */
static int
l_expert_cache(lua_State *L)
{
	lua_Integer nl = luaL_checkinteger(L, 1), ne = luaL_checkinteger(L, 2);
	struct xcache *x;
	struct xlayer *l;
	int i, e;

	luaL_argcheck(L, nl >= 1 && nl <= 4096, 1, "layer count out of range");
	luaL_argcheck(L, ne >= 1 && ne <= 65536, 2, "expert count out of range");
	x = lua_newuserdatauv(L, sizeof(*x), 0);
	memset(x, 0, sizeof(*x));
	luaL_setmetatable(L, XCACHE_MT);
	x->nl = (int)nl;
	x->ne = (int)ne;
	if ((x->l = calloc((size_t)nl, sizeof(*x->l))) == NULL)
		return luaL_error(L, "out of memory");
	for (i = 0; i < x->nl; i++) {
		l = &x->l[i];
		l->x = x;
		l->slot = malloc((size_t)ne * sizeof(*l->slot));
		l->score = calloc((size_t)ne, sizeof(*l->score));
		l->pend = calloc((size_t)ne, sizeof(*l->pend));
		if (l->slot == NULL || l->score == NULL || l->pend == NULL)
			return luaL_error(L, "out of memory");
		for (e = 0; e < x->ne; e++)
			l->slot[e] = -1;
		l->fb = 0;
	}
	return 1;
}

/*** The state of an expert cache, from `lgml.expert_cache`.
 * Layers count from 1. Experts and slots count from 0.
 * @type expert_cache
 */

static int
xcache_gc(lua_State *L)
{
	free_layers(luaL_checkudata(L, 1, XCACHE_MT));
	return 0;
}

/*** Give a layer n empty slots.
 * The scores stay.
 * @tparam int il the layer
 * @tparam int n the slot count, 0 to the expert count
 * @function expert_cache:resize
 */
static int
xcache_resize(lua_State *L)
{
	struct xcache *x = check_xcache(L, 1);
	struct xlayer *l = check_layer(L, x, 2);
	lua_Integer n = luaL_checkinteger(L, 3);
	int *ex = NULL, e;

	luaL_argcheck(L, n >= 0 && n <= x->ne, 3, "slot count out of range");
	if (n > 0 && (ex = malloc((size_t)n * sizeof(*ex))) == NULL)
		return luaL_error(L, "out of memory");
	free(l->expert);
	l->expert = ex;
	l->nslots = (int)n;
	for (e = 0; e < n; e++)
		ex[e] = -1;
	for (e = 0; e < x->ne; e++)
		l->slot[e] = -1;
	l->fb = 0;
	return 0;
}

/*** Record that a slot holds an expert.
 * The expert leaves its old slot, and the old expert of the slot
 * leaves the cache.
 * @tparam int il the layer
 * @tparam int slot the slot
 * @tparam int expert the expert, or -1 for none
 * @treturn int the old expert of the slot, or -1
 * @function expert_cache:assign
 */
static int
xcache_assign(lua_State *L)
{
	struct xcache *x = check_xcache(L, 1);
	struct xlayer *l = check_layer(L, x, 2);
	lua_Integer s = luaL_checkinteger(L, 3), e = luaL_checkinteger(L, 4);
	int old;

	luaL_argcheck(L, s >= 0 && s < l->nslots, 3, "slot out of range");
	luaL_argcheck(L, e >= -1 && e < x->ne, 4, "expert out of range");
	old = l->expert[s];
	if (old >= 0)
		l->slot[old] = -1;
	if (e >= 0 && l->slot[e] >= 0)
		l->expert[l->slot[e]] = -1;
	l->expert[s] = (int)e;
	if (e >= 0)
		l->slot[e] = (int)s;
	update_fb(l);
	lua_pushinteger(L, old);
	return 1;
}

/*** Get the device tables of a layer.
 * @tparam int il the layer
 * @treturn {int,...} the slot of each expert, or 0 for an expert not in
 * a slot
 * @treturn {number,...} 1 for each expert in a slot, else 0
 * @treturn {int,...} the expert of each slot, or -1
 * @function expert_cache:tables
 */
static int
xcache_tables(lua_State *L)
{
	struct xcache *x = check_xcache(L, 1);
	struct xlayer *l = check_layer(L, x, 2);
	int e;

	lua_createtable(L, x->ne, 0);
	lua_createtable(L, x->ne, 0);
	for (e = 0; e < x->ne; e++) {
		lua_pushinteger(L, l->slot[e] >= 0 ? l->slot[e] : 0);
		lua_rawseti(L, -3, e + 1);
		lua_pushinteger(L, l->slot[e] >= 0);
		lua_rawseti(L, -2, e + 1);
	}
	lua_createtable(L, l->nslots, 0);
	for (e = 0; e < l->nslots; e++) {
		lua_pushinteger(L, l->expert[e]);
		lua_rawseti(L, -2, e + 1);
	}
	return 3;
}

/* Pick j of a row of an i32 (T, K) tensor. */
static int32_t
pick(const struct ggml_tensor *t, int64_t row, int64_t j)
{
	return *(const int32_t *)((const char *)t->data + row * t->nb[1] + j * t->nb[0]);
}

/*
 * Give the CPU ids of a pass. A pick of an expert in a slot becomes a
 * pick that the CPU makes anyway: a miss of the same token, else a miss
 * of the pass, else the fallback. The CPU weight of that pick is 0.
 */
static void
remap_op(struct ggml_tensor *dst, int ith, int nth, void *ud)
{
	struct xlayer *l = ud;
	const struct ggml_tensor *sel = dst->src[0];
	int64_t K = sel->ne[0], T = sel->ne[1], t, j;
	int32_t id, first = -1, rowmiss, *out;
	uint64_t hits = 0, misses = 0;

	(void)nth;
	if (ith != 0)
		return;
	for (t = 0; t < T && first < 0; t++) {
		for (j = 0; j < K; j++) {
			if (l->slot[pick(sel, t, j)] < 0) {
				first = pick(sel, t, j);
				break;
			}
		}
	}
	if (first < 0)
		first = l->fb >= 0 ? l->fb : 0;
	for (t = 0; t < T; t++) {
		out = (int32_t *)((char *)dst->data + t * dst->nb[1]);
		rowmiss = -1;
		for (j = 0; j < K; j++) {
			id = pick(sel, t, j);
			l->pend[id]++;
			if (l->slot[id] < 0) {
				rowmiss = id;
				misses++;
			} else
				hits++;
		}
		for (j = 0; j < K; j++) {
			id = pick(sel, t, j);
			out[j] = l->slot[id] < 0 ? id : rowmiss >= 0 ? rowmiss : first;
		}
	}
	l->ntok += T;
	l->x->hits += hits;
	l->x->misses += misses;
}

/*** Add the CPU ids of a layer to a graph.
 * The op runs on the CPU. It counts the picks of each expert, and the
 * picks that the slots hold. A pick in a slot gets the id of an expert
 * that the CPU computes anyway; give it weight 0 on the CPU.
 * @tparam graph g the graph
 * @tparam int il the layer
 * @tparam tensor sel i32 (T, `n_used`) expert ids of the router
 * @treturn tensor i32 (T, `n_used`) the ids for the CPU
 * @function expert_cache:remap
 */
static int
xcache_remap(lua_State *L)
{
	struct xcache *x = check_xcache(L, 1);
	struct graph *g = lgml_check_graph(L, 2);
	struct xlayer *l = check_layer(L, x, 3);
	struct tensor *sel = lgml_operand(L, g, 4);
	struct ggml_tensor *args[1], *r;

	if (sel->t->type != GGML_TYPE_I32 || sel->t->ne[2] != 1 || sel->t->ne[3] != 1)
		return luaL_error(L, "remap: sel must be i32 (T, n_used)");
	/* The graph keeps the state alive. */
	lua_getiuservalue(L, 2, UV_EXTRA);
	lua_pushvalue(L, 1);
	lua_pushboolean(L, 1);
	lua_rawset(L, -3);
	lua_pop(L, 1);
	args[0] = sel->t;
	lgml_reserve(L, g, 1);
	r = ggml_custom_4d(g->ctx, GGML_TYPE_I32, sel->t->ne[0], sel->t->ne[1], 1, 1, args, 1,
	    remap_op, 1, l);
	return lgml_push_node(L, g, r, 2);
}

/*
 * The CPU part of the experts: only the picks that no slot holds. The
 * first op writes silu(gate x) * up x of each such pick, the second adds
 * down of it times the router weight into the token output.
 */

/* The picks of a pass that no slot holds, as t * K + k. */
static int64_t
misses_of(const struct xlayer *l, const struct ggml_tensor *sel, int64_t *out)
{
	int64_t K = sel->ne[0], T = sel->ne[1], t, j, n = 0;

	for (t = 0; t < T; t++)
		for (j = 0; j < K; j++)
			if (l->slot[pick(sel, t, j)] < 0)
				out[n++] = t * K + j;
	return n;
}

/* Count the picks of a pass, as remap_op does. */
static void
count_picks(struct xlayer *l, const struct ggml_tensor *sel)
{
	int64_t K = sel->ne[0], T = sel->ne[1], t, j;
	uint64_t hits = 0, misses = 0;
	int32_t id;

	for (t = 0; t < T; t++) {
		for (j = 0; j < K; j++) {
			id = pick(sel, t, j);
			l->pend[id]++;
			if (l->slot[id] < 0)
				misses++;
			else
				hits++;
		}
	}
	l->ntok += T;
	l->x->hits += hits;
	l->x->misses += misses;
}

/* Quantize a row of n floats to the vec_dot type of a weight type. */
static const void *
quant_row(enum ggml_type wt, const float *x, int64_t n, void **buf, size_t *cap)
{
	const struct ggml_type_traits_cpu *tr = ggml_get_type_traits_cpu(wt);
	size_t sz;

	if (tr->vec_dot_type == GGML_TYPE_F32)
		return x;
	sz = ggml_row_size(tr->vec_dot_type, n);
	if (sz > *cap) {
		free(*buf);
		if ((*buf = malloc(sz)) == NULL)
			abort();
		*cap = sz;
	}
	ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x, *buf, n);
	return *buf;
}

/* Dot of row r of expert e of w (ne = [n, rows, experts]) with q. */
static float
expert_dot(const struct ggml_tensor *w, int64_t e, int64_t r, const void *q)
{
	float s;

	ggml_get_type_traits_cpu(w->type)->vec_dot((int)w->ne[0], &s, 0,
	    (const char *)w->data + e * w->nb[2] + r * w->nb[1], 0, q, 0, 1);
	return s;
}

/* src: x (T, 1, E) f32, sel i32 (T, K), gate, up (or fused gate_up). */
static void
miss_act_op(struct ggml_tensor *dst, int ith, int nth, void *ud)
{
	struct xlayer *l = ud;
	const struct ggml_tensor *x = dst->src[0], *sel = dst->src[1];
	const struct ggml_tensor *gate = dst->src[2], *up = dst->src[3];
	int64_t K = sel->ne[0], F = dst->ne[0], n, i, i0, i1, per, cur = -1;
	int64_t *ms;
	void *bg = NULL, *bu = NULL;
	size_t cg = 0, cu = 0;
	const void *qg = NULL, *qu = NULL;

	if ((ms = malloc(sizeof(*ms) * (size_t)(sel->ne[0] * sel->ne[1] + 1))) == NULL)
		abort();
	n = misses_of(l, sel, ms);
	per = (n * F + nth - 1) / nth;
	i0 = per * ith;
	i1 = i0 + per < n * F ? i0 + per : n * F;
	for (i = i0; i < i1; i++) {
		int64_t m = ms[i / F], r = i % F, t = m / K;
		int32_t e = pick(sel, t, m % K);
		const float *xr = (const float *)((const char *)x->data + t * x->nb[2]);
		float gv, uv;

		if (t != cur) {
			qg = quant_row(gate->type, xr, x->ne[0], &bg, &cg);
			qu = up != NULL ? quant_row(up->type, xr, x->ne[0], &bu, &cu) : qg;
			cur = t;
		}
		gv = expert_dot(gate, e, r, qg);
		uv = up != NULL ? expert_dot(up, e, r, qu) : expert_dot(gate, e, F + r, qg);
		((float *)((char *)dst->data + m * dst->nb[1]))[r] = gv / (1.0f + expf(-gv)) * uv;
	}
	free(bg);
	free(bu);
	free(ms);
	if (ith == 0)
		count_picks(l, sel);
}

/* src: act (T * K, F) f32, sel i32 (T, K), w f32 (T, K, 1), down. */
static void
miss_down_op(struct ggml_tensor *dst, int ith, int nth, void *ud)
{
	struct xlayer *l = ud;
	const struct ggml_tensor *act = dst->src[0], *sel = dst->src[1];
	const struct ggml_tensor *w = dst->src[2], *down = dst->src[3];
	enum ggml_type vt = ggml_get_type_traits_cpu(down->type)->vec_dot_type;
	int64_t K = sel->ne[0], E = dst->ne[0], T = dst->ne[1], F = act->ne[0];
	int64_t n, i, j, j0, j1, per, t;
	int64_t *ms;
	size_t rs = ggml_row_size(vt, F);
	char *q;

	if ((ms = malloc(sizeof(*ms) * (size_t)(K * T + 1))) == NULL)
		abort();
	n = misses_of(l, sel, ms);
	if ((q = malloc(rs * (size_t)(n + 1))) == NULL)
		abort();
	for (i = 0; i < n; i++) {
		const float *a = (const float *)((const char *)act->data + ms[i] * act->nb[1]);

		if (vt == GGML_TYPE_F32)
			memcpy(q + i * rs, a, rs);
		else
			ggml_get_type_traits_cpu(vt)->from_float(a, q + i * rs, F);
	}
	per = (E + nth - 1) / nth;
	j0 = per * ith;
	j1 = j0 + per < E ? j0 + per : E;
	for (t = 0; t < T; t++) {
		float *o = (float *)((char *)dst->data + t * dst->nb[1]);

		for (j = j0; j < j1; j++)
			o[j] = 0;
	}
	for (i = 0; i < n; i++) {
		int64_t tt = ms[i] / K, k = ms[i] % K;
		int32_t e = pick(sel, tt, k);
		float wt = *(const float *)((const char *)w->data + tt * w->nb[2] + k * w->nb[1]);
		float *o = (float *)((char *)dst->data + tt * dst->nb[1]);

		for (j = j0; j < j1; j++)
			o[j] += wt * expert_dot(down, e, j, q + i * rs);
	}
	free(q);
	free(ms);
}

/*** Add the CPU part of the experts of a layer to a graph.
 * It computes only the picks that no slot holds, with SwiGLU, and sums
 * their outputs times the router weights. It runs on the CPU and counts
 * the picks, as `remap` does.
 * @tparam graph g the graph
 * @tparam int il the layer
 * @tparam tensor x f32 (T, 1, E) expert input, contiguous
 * @tparam tensor sel i32 (T, K) expert ids of the router
 * @tparam tensor w f32 (T, K, 1) router weights
 * @tparam tensor gate the gate experts, or the fused gate and up experts
 * @tparam tensor|nil up the up experts, nil when gate is fused
 * @tparam tensor down the down experts
 * @treturn tensor f32 (T, E)
 * @function expert_cache:misses
 */
static int
xcache_misses(lua_State *L)
{
	struct xcache *xc = check_xcache(L, 1);
	struct graph *g = lgml_check_graph(L, 2);
	struct xlayer *l = check_layer(L, xc, 3);
	struct tensor *x = lgml_operand(L, g, 4);
	struct tensor *sel = lgml_operand(L, g, 5);
	struct tensor *w = lgml_operand(L, g, 6);
	struct tensor *gate = lgml_operand(L, g, 7);
	struct tensor *up = lua_isnoneornil(L, 8) ? NULL : lgml_operand(L, g, 8);
	struct tensor *down = lgml_operand(L, g, 9);
	struct ggml_tensor *a1[4], *a2[4], *act, *r;
	int64_t T = sel->t->ne[1], K = sel->t->ne[0], E = x->t->ne[0];
	int64_t F = up != NULL ? gate->t->ne[1] : gate->t->ne[1] / 2;

	if (sel->t->type != GGML_TYPE_I32 || x->t->type != GGML_TYPE_F32 ||
	    w->t->type != GGML_TYPE_F32 || x->t->ne[2] != T || gate->t->ne[0] != E ||
	    down->t->ne[0] != F || down->t->ne[1] != E || !ggml_is_contiguous(x->t))
		return luaL_error(L, "misses: shapes do not match");
	/* The graph keeps the state alive. */
	lua_getiuservalue(L, 2, UV_EXTRA);
	lua_pushvalue(L, 1);
	lua_pushboolean(L, 1);
	lua_rawset(L, -3);
	lua_pop(L, 1);
	a1[0] = x->t;
	a1[1] = sel->t;
	a1[2] = gate->t;
	a1[3] = up != NULL ? up->t : NULL;
	lgml_reserve(L, g, 2);
	act = ggml_custom_4d(g->ctx, GGML_TYPE_F32, F, T * K, 1, 1, a1, up != NULL ? 4 : 3,
	    miss_act_op, GGML_N_TASKS_MAX, l);
	a2[0] = act;
	a2[1] = sel->t;
	a2[2] = w->t;
	a2[3] = down->t;
	r = ggml_custom_4d(g->ctx, GGML_TYPE_F32, E, T, 1, 1, a2, 4, miss_down_op,
	    GGML_N_TASKS_MAX, l);
	return lgml_push_node(L, g, r, 2);
}

struct cand {
	double	 gain;
	int	 il;
	int	 slot;
	int	 expert;
};

/* The scores that by_score sorts by; plan sets it. */
static const double *sort_scores;

/* Order indices by score, most first. */
static int
by_score(const void *a, const void *b)
{
	int i = *(const int *)a, j = *(const int *)b;
	double x = sort_scores[i], y = sort_scores[j];

	return x < y ? 1 : x > y ? -1 : i - j;
}

static int
by_gain(const void *a, const void *b)
{
	const struct cand *x = a, *y = b;

	if (x->gain != y->gain)
		return x->gain < y->gain ? 1 : -1;
	return x->il != y->il ? x->il - y->il : x->slot - y->slot;
}

/* Fold the picks since the last plan into the scores. */
static void
fold(struct xcache *x, struct xlayer *l, double decay)
{
	double f = l->ntok > 0 ? pow(decay, (double)l->ntok) : 1;
	int e;

	for (e = 0; e < x->ne; e++) {
		l->score[e] = l->score[e] * f + l->pend[e];
		l->pend[e] = 0;
	}
	l->ntok = 0;
}

/*** Plan the next slot changes.
 * First the picks since the last plan go into the scores; old scores
 * fall by `decay` for each token. A change puts an expert in a slot: an
 * empty slot, or one whose expert scores less by a margin. The plan
 * takes the changes of most gain over all layers.
 * @tparam int n the most changes
 * @tparam[opt=1] number decay the factor of a score for each token
 * @tparam[opt=0.25] number margin a change needs the new score more
 * than (1 + margin) times the old one plus 1
 * @treturn {{int,int,int},...} changes `{ il, slot, expert }`, most
 * gain first
 * @function expert_cache:plan
 */
static int
xcache_plan(lua_State *L)
{
	struct xcache *x = check_xcache(L, 1);
	lua_Integer n = luaL_checkinteger(L, 2);
	double decay = luaL_optnumber(L, 3, 1), margin = luaL_optnumber(L, 4, 0.25);
	struct xlayer *l;
	struct cand *c = NULL, *nc;
	size_t nc_len = 0, cap = 0;
	int *in, *out, i, j, e, ni, no;
	double sin, sout, *ss;

	luaL_argcheck(L, decay > 0 && decay <= 1, 3, "decay must be in (0, 1]");
	in = malloc((size_t)x->ne * sizeof(*in));
	out = malloc((size_t)x->ne * sizeof(*out));
	ss = malloc((size_t)x->ne * sizeof(*ss));
	if (in == NULL || out == NULL || ss == NULL) {
		free(in);
		free(out);
		free(ss);
		return luaL_error(L, "out of memory");
	}
	for (i = 0; i < x->nl; i++) {
		l = &x->l[i];
		fold(x, l, decay);
		if (l->nslots == 0 || n <= 0)
			continue;
		ni = no = 0;
		for (e = 0; e < x->ne; e++)
			if (l->slot[e] < 0)
				in[ni++] = e;
		sort_scores = l->score;
		qsort(in, (size_t)ni, sizeof(*in), by_score);
		/* Slots, empty ones first, then the least score. */
		for (j = 0; j < l->nslots; j++) {
			out[no++] = j;
			ss[j] = l->expert[j] < 0 ? HUGE_VAL : -l->score[l->expert[j]];
		}
		sort_scores = ss;
		qsort(out, (size_t)no, sizeof(*out), by_score);
		for (j = 0; j < ni && j < no && j < n; j++) {
			sin = l->score[in[j]];
			if (l->expert[out[j]] < 0)
				sout = -1;
			else
				sout = l->score[l->expert[out[j]]] * (1 + margin) + 1;
			if (sin <= sout)
				break;
			if (nc_len == cap) {
				cap = cap ? cap * 2 : 64;
				if ((nc = realloc(c, cap * sizeof(*c))) == NULL) {
					free(c);
					free(in);
					free(out);
					free(ss);
					return luaL_error(L, "out of memory");
				}
				c = nc;
			}
			c[nc_len].gain = l->expert[out[j]] < 0 ? HUGE_VAL : sin - sout;
			c[nc_len].il = i + 1;
			c[nc_len].slot = out[j];
			c[nc_len].expert = in[j];
			nc_len++;
		}
	}
	free(in);
	free(out);
	free(ss);
	if (nc_len > 0)
		qsort(c, nc_len, sizeof(*c), by_gain);
	if ((lua_Integer)nc_len > n)
		nc_len = (size_t)(n > 0 ? n : 0);
	lua_createtable(L, (int)nc_len, 0);
	for (i = 0; i < (int)nc_len; i++) {
		lua_createtable(L, 3, 0);
		lua_pushinteger(L, c[i].il);
		lua_rawseti(L, -2, 1);
		lua_pushinteger(L, c[i].slot);
		lua_rawseti(L, -2, 2);
		lua_pushinteger(L, c[i].expert);
		lua_rawseti(L, -2, 3);
		lua_rawseti(L, -2, i + 1);
	}
	free(c);
	return 1;
}

/*** Get the scores of a layer.
 * The picks since the last plan count in full.
 * @tparam int il the layer
 * @treturn {number,...} one score for each expert
 * @function expert_cache:scores
 */
static int
xcache_scores(lua_State *L)
{
	struct xcache *x = check_xcache(L, 1);
	struct xlayer *l = check_layer(L, x, 2);
	int e;

	lua_createtable(L, x->ne, 0);
	for (e = 0; e < x->ne; e++) {
		lua_pushnumber(L, l->score[e] + l->pend[e]);
		lua_rawseti(L, -2, e + 1);
	}
	return 1;
}

/*** Set the scores of a layer.
 * @tparam int il the layer
 * @tparam {number,...} scores one score for each expert
 * @function expert_cache:seed
 */
static int
xcache_seed(lua_State *L)
{
	struct xcache *x = check_xcache(L, 1);
	struct xlayer *l = check_layer(L, x, 2);
	int e;

	luaL_checktype(L, 3, LUA_TTABLE);
	luaL_argcheck(L, lua_rawlen(L, 3) == (size_t)x->ne, 3, "need one score for each expert");
	for (e = 0; e < x->ne; e++) {
		lua_rawgeti(L, 3, e + 1);
		l->score[e] = luaL_checknumber(L, -1);
		l->pend[e] = 0;
		lua_pop(L, 1);
	}
	l->ntok = 0;
	return 0;
}

/*** Get the pick counts since the last `expert_cache:reset_stats`.
 * @treturn int picks of experts in a slot
 * @treturn int picks of other experts
 * @function expert_cache:stats
 */
static int
xcache_stats(lua_State *L)
{
	struct xcache *x = check_xcache(L, 1);

	lua_pushinteger(L, (lua_Integer)x->hits);
	lua_pushinteger(L, (lua_Integer)x->misses);
	return 2;
}

/*** Set the pick counts of `expert_cache:stats` to 0.
 * @function expert_cache:reset_stats
 */
static int
xcache_reset_stats(lua_State *L)
{
	struct xcache *x = check_xcache(L, 1);

	x->hits = x->misses = 0;
	return 0;
}

/*** Copy one slice of a leaf into a slice of another.
 * A slice is one index of the outermost dimension, such as one expert of
 * an (E, M, K) weight. Both leaves have the same type and slice size.
 * @tparam int i the slice of this tensor, from 0
 * @tparam tensor src the source leaf
 * @tparam int j the slice of src, from 0
 * @raise if a tensor has no storage, or the types or sizes differ
 * @function tensor:set_slice
 */
static int
tensor_set_slice(lua_State *L)
{
	struct tensor *d = lgml_check_tensor(L, 1), *s = lgml_check_tensor(L, 3);
	lua_Integer i = luaL_checkinteger(L, 2), j = luaL_checkinteger(L, 4);
	const struct ggml_tensor *dt = d->t, *st = s->t;
	size_t nb;
	void *tmp;

	if (dt->buffer == NULL || st->buffer == NULL || dt->data == NULL || st->data == NULL)
		return luaL_error(L, "set_slice: tensors need storage");
	if (dt->type != st->type || !ggml_is_contiguous(dt) || !ggml_is_contiguous(st))
		return luaL_error(L, "set_slice: need contiguous tensors of one type");
	nb = dt->nb[d->rank - 1];
	if (st->nb[s->rank - 1] != nb)
		return luaL_error(L, "set_slice: slice sizes differ");
	luaL_argcheck(L, i >= 0 && i < dt->ne[d->rank - 1], 2, "slice out of range");
	luaL_argcheck(L, j >= 0 && j < st->ne[s->rank - 1], 4, "slice out of range");
	if (ggml_backend_buffer_is_host(st->buffer)) {
		ggml_backend_tensor_set(d->t, (const char *)st->data + (size_t)j * nb,
		    (size_t)i * nb, nb);
		return 0;
	}
	if ((tmp = malloc(nb)) == NULL)
		return luaL_error(L, "out of memory");
	ggml_backend_tensor_get(s->t, tmp, (size_t)j * nb, nb);
	ggml_backend_tensor_set(d->t, tmp, (size_t)i * nb, nb);
	free(tmp);
	return 0;
}

void
lgml_open_xcache(lua_State *L)
{
	static const luaL_Reg meta[] = {
		{ "__gc", xcache_gc },
		{ "__close", xcache_gc },
		{ NULL, NULL }
	};
	static const luaL_Reg methods[] = {
		{ "resize", xcache_resize },
		{ "assign", xcache_assign },
		{ "tables", xcache_tables },
		{ "remap", xcache_remap },
		{ "misses", xcache_misses },
		{ "plan", xcache_plan },
		{ "scores", xcache_scores },
		{ "seed", xcache_seed },
		{ "stats", xcache_stats },
		{ "reset_stats", xcache_reset_stats },
		{ NULL, NULL }
	};
	static const luaL_Reg tfns[] = {
		{ "set_slice", tensor_set_slice },
		{ NULL, NULL }
	};

	luaL_newmetatable(L, XCACHE_MT);
	luaL_setfuncs(L, meta, 0);
	luaL_newlib(L, methods);
	lua_setfield(L, -2, "__index");
	lua_pop(L, 1);
	lgml_tensor_methods(L, tfns);
	lua_pushcfunction(L, l_expert_cache);
	lua_setfield(L, -2, "expert_cache");
}
