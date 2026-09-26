/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Graph ops for the VibeVoice TTS model. Shapes use torch order. The
 * acoustic decoder keeps activations as (T, C): one row per time step.
 */

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

#define VV_MAX_K	(1 << 16)

/* A (T, C) f32 tensor with contiguous storage. */
static void
check_tc(lua_State *L, int idx, const struct ggml_tensor *t)
{
	if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[2] != 1 ||
	    t->ne[3] != 1)
		luaL_argerror(L, idx, "need a contiguous f32 (T, C) tensor");
}

/* An f32 vector of n elements, or nil. */
static void
check_bias(lua_State *L, int idx, const struct tensor *b, int64_t n)
{
	if (b == NULL)
		return;
	if (b->t->type != GGML_TYPE_F32 || !ggml_is_contiguous(b->t) ||
	    ggml_nelements(b->t) != n || b->t->ne[0] != n)
		luaL_argerror(L, idx, "bias must be an f32 vector of the output width");
}

/*** A graph of ops on one backend.
 * Each op returns a node tensor. Nodes get storage when `graph:compute`
 * runs. An operand is a leaf on the same backend or a node of this graph.
 * The graph keeps its leaves alive until it closes.
 * @type graph
 */

/*** Causal 1-D convolution with stride 1.
 * The input gets K - 1 zero steps on the left.
 * @tparam tensor x contiguous f32 (T, IC)
 * @tparam tensor w contiguous (OC, IC, K)
 * @tparam[opt] tensor b f32 (OC) bias
 * @treturn tensor (T, OC) node
 * @raise on bad types or shapes
 * @function graph:vibevoice_conv
 */
static int
vv_conv(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = lgml_operand(L, g, 2);
	struct tensor *w = lgml_operand(L, g, 3);
	struct tensor *b = lgml_opt_operand(L, g, 4);
	const struct ggml_tensor *tw = w->t;
	struct ggml_tensor *xt, *col, *r;
	int64_t k = tw->ne[0], ic = tw->ne[1], oc = tw->ne[2];

	check_tc(L, 2, x->t);
	check_bias(L, 4, b, oc);
	if (!ggml_is_contiguous(tw) || tw->ne[3] != 1 || ic != x->t->ne[0])
		return luaL_error(L, "vibevoice_conv: need contiguous w (OC, IC, K)");
	if (k < 1 || k > VV_MAX_K)
		return luaL_error(L, "vibevoice_conv: kernel size out of range");
	lgml_reserve(L, g, 8);
	xt = ggml_cont(g->ctx, ggml_transpose(g->ctx, x->t));
	if (k > 1)
		xt = ggml_pad_ext(g->ctx, xt, (int)(k - 1), 0, 0, 0, 0, 0, 0, 0);
	col = ggml_im2col(g->ctx, w->t, xt, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
	r = ggml_mul_mat(g->ctx, ggml_reshape_2d(g->ctx, w->t, k * ic, oc),
	    ggml_reshape_2d(g->ctx, col, col->ne[0], col->ne[1]));
	if (b != NULL)
		r = ggml_add(g->ctx, r, b->t);
	return lgml_push_node(L, g, r, 2);
}

/*** Causal depthwise 1-D convolution with stride 1.
 * @tparam tensor x contiguous f32 (T, C)
 * @tparam tensor w contiguous f32 (K, C), one row for each tap, K up to 64
 * @tparam tensor b f32 (C) bias
 * @treturn tensor (T, C) node
 * @raise on bad types, shapes or ranges
 * @function graph:vibevoice_dwconv
 */
static int
vv_dwconv(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = lgml_operand(L, g, 2);
	struct tensor *w = lgml_operand(L, g, 3);
	struct tensor *b = lgml_operand(L, g, 4);
	const struct ggml_tensor *tx = x->t, *tw = w->t;
	struct ggml_tensor *xp, *r = NULL, *term;
	int64_t i, c = tx->ne[0], n = tx->ne[1], k = tw->ne[1];

	check_tc(L, 2, tx);
	check_bias(L, 4, b, c);
	if (tw->type != GGML_TYPE_F32 || !ggml_is_contiguous(tw) || tw->ne[0] != c ||
	    tw->ne[2] != 1 || tw->ne[3] != 1)
		return luaL_error(L, "vibevoice_dwconv: need contiguous f32 w (K, C)");
	if (k < 1 || k > 64)
		return luaL_error(L, "vibevoice_dwconv: kernel size out of range");
	lgml_reserve(L, g, (int)(4 * k + 1));
	xp = ggml_pad_ext(g->ctx, x->t, 0, 0, (int)(k - 1), 0, 0, 0, 0, 0);
	for (i = 0; i < k; i++) {
		term = ggml_mul(g->ctx,
		    ggml_view_2d(g->ctx, xp, c, n, xp->nb[1], (size_t)i * xp->nb[1]),
		    ggml_view_1d(g->ctx, w->t, c, (size_t)i * tw->nb[1]));
		r = r == NULL ? term : ggml_add(g->ctx, r, term);
	}
	r = ggml_add(g->ctx, r, b->t);
	return lgml_push_node(L, g, r, 2);
}

/*** Causal transposed 1-D convolution.
 * The op drops the last K - stride steps.
 * @tparam tensor x contiguous f32 (T, IC)
 * @tparam tensor wp contiguous weight as (OC * K, IC); row oc * K + k is
 * tap k of output channel oc
 * @tparam tensor b f32 (OC) bias
 * @tparam int stride 1 to 64, not more than K
 * @treturn tensor (T * stride, OC) node
 * @raise on bad types, shapes or ranges
 * @function graph:vibevoice_convtr
 */
static int
vv_convtr(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = lgml_operand(L, g, 2);
	struct tensor *w = lgml_operand(L, g, 3);
	struct tensor *b = lgml_operand(L, g, 4);
	lua_Integer s = luaL_checkinteger(L, 5);
	const struct ggml_tensor *tx = x->t, *tw = w->t;
	struct ggml_tensor *col, *y;
	int64_t oc = b->t->ne[0], k, n = tx->ne[1];

	check_tc(L, 2, tx);
	check_bias(L, 4, b, oc);
	luaL_argcheck(L, s >= 1 && s <= 64, 5, "stride out of range");
	if (!ggml_is_contiguous(tw) || tw->ne[2] != 1 || tw->ne[3] != 1 ||
	    tw->ne[0] != tx->ne[0] || oc < 1 || tw->ne[1] % oc != 0)
		return luaL_error(L, "vibevoice_convtr: need contiguous wp (OC * K, IC)");
	k = tw->ne[1] / oc;
	if (k < s)
		return luaL_error(L, "vibevoice_convtr: kernel shorter than stride");
	lgml_reserve(L, g, 7);
	col = ggml_mul_mat(g->ctx, w->t, x->t);
	y = ggml_col2im_1d(g->ctx, col, (int)s, (int)oc, 0);
	if (y->ne[0] != n * s)
		y = ggml_cont(g->ctx, ggml_view_2d(g->ctx, y, n * s, oc, y->nb[1], 0));
	y = ggml_cont(g->ctx, ggml_transpose(g->ctx, y));
	y = ggml_add(g->ctx, y, b->t);
	return lgml_push_node(L, g, y, 2);
}

/*** Largest absolute value in each block of k elements.
 * @tparam tensor x contiguous f32 input with n elements
 * @tparam int k block size; it divides n
 * @treturn tensor (n / k) node
 * @raise if x is not contiguous f32, or k is out of range or does not divide n
 * @function graph:vibevoice_peak
 */
static int
vv_peak(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = lgml_operand(L, g, 2);
	lua_Integer k = luaL_checkinteger(L, 3);
	const struct ggml_tensor *t = x->t;
	int64_t n = ggml_nelements(t);
	struct ggml_tensor *r;

	luaL_argcheck(L, k >= 1 && k <= VV_MAX_K, 3, "block size out of range");
	if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t))
		return luaL_error(L, "vibevoice_peak: need a contiguous f32 tensor");
	if (n % k != 0)
		return luaL_error(L, "vibevoice_peak: size not a multiple of the block");
	lgml_reserve(L, g, 3);
	r = ggml_abs(g->ctx, ggml_reshape_1d(g->ctx, x->t, n));
	r = ggml_pool_1d(g->ctx, r, GGML_OP_POOL_MAX, (int)k, (int)k, 0);
	return lgml_push_node(L, g, r, 1);
}

static const luaL_Reg vv_ops[] = {
	{ "vibevoice_conv", vv_conv },
	{ "vibevoice_dwconv", vv_dwconv },
	{ "vibevoice_convtr", vv_convtr },
	{ "vibevoice_peak", vv_peak },
	{ NULL, NULL }
};

void
lgml_open_ops_vibevoice(lua_State *L)
{
	lgml_graph_ops(L, vv_ops);
}
