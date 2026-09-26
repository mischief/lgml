/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Graph ops for the Qwen3-TTS codec decoder. Activations are (T, C):
 * time outer, channels inner. All convolutions are causal.
 */

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

#define MAX_KERNEL	64
#define MAX_DILATION	4096
#define MAX_STRIDE	64

/* An f32 (T, C) activation with contiguous storage. */
static struct tensor *
check_act(lua_State *L, struct graph *g, int idx)
{
	struct tensor *x = lgml_operand(L, g, idx);
	const struct ggml_tensor *t = x->t;

	if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[2] != 1 ||
	    t->ne[3] != 1)
		luaL_argerror(L, idx, "need a contiguous f32 (T, C) tensor");
	return x;
}

/* An optional f32 vector with n entries. */
static struct tensor *
check_vec(lua_State *L, struct graph *g, int idx, int64_t n)
{
	struct tensor *v = lgml_opt_operand(L, g, idx);

	if (v == NULL)
		return NULL;
	if (v->t->type != GGML_TYPE_F32 || !ggml_is_contiguous(v->t) ||
	    ggml_nelements(v->t) != n || v->t->ne[0] != n)
		luaL_argerror(L, idx, "need an f32 vector with one entry per channel");
	return v;
}

static int
check_weight_type(const struct ggml_tensor *w)
{
	return (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16) &&
	    ggml_is_contiguous(w) && w->ne[3] == 1;
}

/* Rows [row, row + n) of a (T, C) tensor, as a contiguous view. */
static struct ggml_tensor *
rows(struct ggml_context *ctx, struct ggml_tensor *x, int64_t row, int64_t n)
{
	return ggml_view_2d(ctx, x, x->ne[0], n, x->nb[1], (size_t)row * x->nb[1]);
}

/* Pad n zero rows before the first row of x. */
static struct ggml_tensor *
pad_front(struct ggml_context *ctx, struct ggml_tensor *x, int n)
{
	return ggml_pad_ext(ctx, x, 0, 0, n, 0, 0, 0, 0, 0);
}

/*** A graph of ops on one backend.
 * Each op returns a node tensor. Nodes get storage when `graph:compute`
 * runs. An operand is a leaf on the same backend or a node of this graph.
 * The graph keeps its leaves alive until it closes.
 * @type graph
 */

/*** Causal 1-D convolution with stride 1.
 * The input gets (K - 1) * dilation zero steps on the left. Each kernel tap
 * is one `mul_mat`.
 * @tparam tensor x contiguous f32 (T, IC)
 * @tparam tensor w contiguous f16 or f32 (K, OC, IC), K up to 64
 * @tparam[opt] tensor b f32 (OC) bias
 * @tparam[opt=1] int dilation 1 to 4096
 * @treturn tensor (T, OC) node
 * @raise on bad types, shapes or ranges
 * @function graph:qwen3tts_conv
 */
static int
op_conv(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = check_act(L, g, 2);
	struct tensor *w = lgml_operand(L, g, 3);
	struct tensor *b;
	lua_Integer d = luaL_optinteger(L, 5, 1);
	struct ggml_tensor *tw = w->t, *xp, *wk, *y = NULL, *m;
	int64_t ic = x->t->ne[0], t = x->t->ne[1], oc, k, i;

	if (!check_weight_type(tw) || tw->ne[0] != ic)
		return luaL_error(L, "qwen3tts_conv: need w (K, OC, IC) f16 or f32 with IC %d",
		    (int)ic);
	oc = tw->ne[1];
	k = tw->ne[2];
	b = check_vec(L, g, 4, oc);
	luaL_argcheck(L, d >= 1 && d <= MAX_DILATION, 5, "dilation out of range");
	if (k < 1 || k > MAX_KERNEL)
		return luaL_error(L, "qwen3tts_conv: kernel size out of range");
	lgml_reserve(L, g, (k > 1 ? 1 : 0) + 4 * (int)k + (b != NULL ? 1 : 0));
	xp = k > 1 ? pad_front(g->ctx, x->t, (int)((k - 1) * d)) : x->t;
	for (i = 0; i < k; i++) {
		wk = ggml_view_2d(g->ctx, tw, ic, oc, tw->nb[1], (size_t)i * tw->nb[2]);
		m = ggml_mul_mat(g->ctx, wk, rows(g->ctx, xp, i * d, t));
		y = y == NULL ? m : ggml_add(g->ctx, y, m);
	}
	if (b != NULL)
		y = ggml_add(g->ctx, y, b->t);
	return lgml_push_node(L, g, y, 2);
}

/*** Causal depthwise 1-D convolution with stride 1.
 * @tparam tensor x contiguous f32 (T, C)
 * @tparam tensor w contiguous f32 (K, C), K up to 64
 * @tparam[opt] tensor b f32 (C) bias
 * @treturn tensor (T, C) node
 * @raise on bad types, shapes or ranges
 * @function graph:qwen3tts_dwconv
 */
static int
op_dwconv(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = check_act(L, g, 2);
	struct tensor *w = lgml_operand(L, g, 3);
	struct tensor *b;
	struct ggml_tensor *tw = w->t, *xp, *wk, *y = NULL, *m;
	int64_t c = x->t->ne[0], t = x->t->ne[1], k, i;

	if (tw->type != GGML_TYPE_F32 || !ggml_is_contiguous(tw) || tw->ne[0] != c ||
	    tw->ne[2] != 1 || tw->ne[3] != 1)
		return luaL_error(L, "qwen3tts_dwconv: need f32 w (K, C) with C %d", (int)c);
	k = tw->ne[1];
	b = check_vec(L, g, 4, c);
	if (k < 1 || k > MAX_KERNEL)
		return luaL_error(L, "qwen3tts_dwconv: kernel size out of range");
	lgml_reserve(L, g, (k > 1 ? 1 : 0) + 4 * (int)k + (b != NULL ? 1 : 0));
	xp = k > 1 ? pad_front(g->ctx, x->t, (int)(k - 1)) : x->t;
	for (i = 0; i < k; i++) {
		wk = ggml_view_1d(g->ctx, tw, c, (size_t)i * tw->nb[1]);
		m = ggml_mul(g->ctx, rows(g->ctx, xp, i, t), wk);
		y = y == NULL ? m : ggml_add(g->ctx, y, m);
	}
	if (b != NULL)
		y = ggml_add(g->ctx, y, b->t);
	return lgml_push_node(L, g, y, 2);
}

/*** Causal transposed 1-D convolution.
 * The op crops the last K - stride samples.
 * @tparam tensor x contiguous f32 (T, IC)
 * @tparam tensor w contiguous f16 or f32 (OC, K, IC), stride <= K <= 64
 * @tparam tensor|nil b f32 (OC) bias, or nil
 * @tparam int stride 1 to 64
 * @treturn tensor (T * stride, OC) node
 * @raise on bad types, shapes or ranges
 * @function graph:qwen3tts_tconv
 */
static int
op_tconv(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = check_act(L, g, 2);
	struct tensor *w = lgml_operand(L, g, 3);
	struct tensor *b;
	lua_Integer s = luaL_checkinteger(L, 5);
	struct ggml_tensor *tw = w->t, *cols, *y;
	int64_t ic = x->t->ne[0], t = x->t->ne[1], k, oc, full;

	if (!check_weight_type(tw) || tw->ne[0] != ic)
		return luaL_error(L, "qwen3tts_tconv: need w (OC, K, IC) f16 or f32 with IC %d",
		    (int)ic);
	k = tw->ne[1];
	oc = tw->ne[2];
	b = check_vec(L, g, 4, oc);
	luaL_argcheck(L, s >= 1 && s <= MAX_STRIDE, 5, "stride out of range");
	if (k < s || k > MAX_KERNEL)
		return luaL_error(L, "qwen3tts_tconv: need stride <= K <= %d", MAX_KERNEL);
	full = (t - 1) * s + k;
	lgml_reserve(L, g, 7);
	cols = ggml_mul_mat(g->ctx, ggml_reshape_2d(g->ctx, tw, ic, k * oc), x->t);
	y = ggml_col2im_1d(g->ctx, cols, (int)s, (int)oc, 0);
	y = ggml_view_2d(g->ctx, y, t * s, oc, (size_t)full * sizeof(float), 0);
	y = ggml_cont(g->ctx, ggml_transpose(g->ctx, y));
	if (b != NULL)
		y = ggml_add(g->ctx, y, b->t);
	return lgml_push_node(L, g, y, 2);
}

/*** SnakeBeta with log-scale parameters.
 * y = x + exp(-b) * sin(exp(a) * x)^2.
 * @tparam tensor x contiguous f32 (T, C)
 * @tparam tensor a f32 (C)
 * @tparam tensor b f32 (C)
 * @treturn tensor node with the shape of x
 * @raise if a or b is missing or has the wrong shape
 * @function graph:qwen3tts_snake
 */
static int
op_snake(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = check_act(L, g, 2);
	struct tensor *a = check_vec(L, g, 3, x->t->ne[0]);
	struct tensor *b = check_vec(L, g, 4, x->t->ne[0]);
	struct ggml_context *ctx;
	struct ggml_tensor *s;

	if (a == NULL || b == NULL)
		return luaL_error(L, "qwen3tts_snake: need a and b");
	lgml_reserve(L, g, 8);
	ctx = g->ctx;
	s = ggml_sin(ctx, ggml_mul(ctx, x->t, ggml_exp(ctx, a->t)));
	s = ggml_mul(ctx, ggml_sqr(ctx, s), ggml_exp(ctx, ggml_neg(ctx, b->t)));
	return lgml_push_node(L, g, ggml_add(ctx, x->t, s), x->rank);
}

/*** Limit each element to [lo, hi].
 * @tparam tensor x contiguous f32 input
 * @tparam number lo lower bound
 * @tparam number hi upper bound, not less than lo
 * @treturn tensor node with the shape of x
 * @raise if x is not contiguous f32 or hi is less than lo
 * @function graph:qwen3tts_clamp
 */
static int
op_clamp(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = lgml_operand(L, g, 2);
	float lo = (float)luaL_checknumber(L, 3);
	float hi = (float)luaL_checknumber(L, 4);

	luaL_argcheck(L, lo <= hi, 4, "hi must not be less than lo");
	if (x->t->type != GGML_TYPE_F32 || !ggml_is_contiguous(x->t))
		return luaL_error(L, "qwen3tts_clamp: need contiguous f32");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_clamp(g->ctx, x->t, lo, hi), x->rank);
}

/*** Join f32 tensors along one dimension.
 * All other sizes match.
 * @tparam tensor a f32 first part
 * @tparam tensor b f32 second part
 * @tparam int dim torch dimension, from 1
 * @treturn tensor node
 * @raise if dim is out of range, a type is not f32, or other sizes differ
 * @function graph:concat_qwen3tts
 */
static int
op_concat(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	struct tensor *b = lgml_operand(L, g, 3);
	lua_Integer dim = luaL_checkinteger(L, 4);
	int rank = a->rank > b->rank ? a->rank : b->rank;
	int i, gd;

	luaL_argcheck(L, dim >= 1 && dim <= rank, 4, "dimension out of range");
	gd = rank - (int)dim;
	if (a->t->type != GGML_TYPE_F32 || b->t->type != GGML_TYPE_F32)
		return luaL_error(L, "concat_qwen3tts: need f32");
	for (i = 0; i < GGML_MAX_DIMS; i++)
		if (i != gd && a->t->ne[i] != b->t->ne[i])
			return luaL_error(L, "concat_qwen3tts: shapes differ outside dim");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_concat(g->ctx, a->t, b->t, gd), rank);
}

static const luaL_Reg ops[] = {
	{ "qwen3tts_conv", op_conv },
	{ "qwen3tts_dwconv", op_dwconv },
	{ "qwen3tts_tconv", op_tconv },
	{ "qwen3tts_snake", op_snake },
	{ "qwen3tts_clamp", op_clamp },
	{ "concat_qwen3tts", op_concat },
	{ NULL, NULL }
};

void
lgml_open_ops_qwen3tts(lua_State *L)
{
	lgml_graph_ops(L, ops);
}
