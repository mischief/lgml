/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Graph ops for the orpheus TTS model and its SNAC codec. Shapes use
 * torch order. Signals are (C, T): time is the innermost dimension.
 */

#include <stdint.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

/* w as f32. A cast adds one node. */
static struct ggml_tensor *
as_f32(lua_State *L, struct graph *g, struct ggml_tensor *w)
{
	if (w->type == GGML_TYPE_F32)
		return w;
	if (w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_BF16)
		luaL_error(L, "weight must be f32, f16 or bf16");
	lgml_reserve(L, g, 1);
	return ggml_cast(g->ctx, w, GGML_TYPE_F32);
}

static void
check_signal(lua_State *L, const struct ggml_tensor *x, const char *op)
{
	if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
	    x->ne[2] != 1 || x->ne[3] != 1)
		luaL_error(L, "%s: x must be contiguous f32 (C, T)", op);
}

/*** A graph of ops on one backend.
 * Each op returns a node tensor. Nodes get storage when `graph:compute`
 * runs. An operand is a leaf on the same backend or a node of this graph.
 * The graph keeps its leaves alive until it closes.
 * @type graph
 */

/*** Snake activation: x + sin(alpha * x)^2 / alpha.
 * @tparam tensor x contiguous f32 (C, T) signal
 * @tparam tensor alpha contiguous f32, f16 or bf16, one value for each channel
 * @treturn tensor f32 (C, T) node
 * @raise on bad types or shapes
 * @function graph:orpheus_snake
 */
static int
op_snake(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = lgml_operand(L, g, 2);
	struct tensor *a = lgml_operand(L, g, 3);
	struct ggml_tensor *tx = x->t, *ta, *ax, *r;

	check_signal(L, tx, "orpheus_snake");
	if (!ggml_is_contiguous(a->t) || ggml_nelements(a->t) != tx->ne[1])
		return luaL_error(L, "orpheus_snake: alpha needs one value per channel");
	ta = as_f32(L, g, a->t);
	lgml_reserve(L, g, 6);
	ta = ggml_reshape_2d(g->ctx, ta, 1, tx->ne[1]);
	ax = ggml_mul(g->ctx, tx, ta);
	ax = ggml_sqr(g->ctx, ggml_sin(g->ctx, ax));
	r = ggml_add(g->ctx, tx, ggml_div(g->ctx, ax, ta));
	return lgml_push_node(L, g, r, 2);
}

/*** Depthwise 1-D convolution with stride 1.
 * The result length is L + 2 * pad - dilation * (K - 1).
 * @tparam tensor w contiguous f32, f16 or bf16 (C, 1, K)
 * @tparam tensor x contiguous f32 (C, L) signal
 * @tparam[opt=0] int pad zeros on each side, 0 to 65536
 * @tparam[opt=1] int dilation 1 to 65536
 * @treturn tensor f32 (C, OL) node
 * @raise on bad types, shapes or ranges, or if the kernel is longer than
 * the padded input
 * @function graph:conv1d_dw_orpheus
 */
static int
op_conv1d_dw(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *w = lgml_operand(L, g, 2);
	struct tensor *x = lgml_operand(L, g, 3);
	lua_Integer p = luaL_optinteger(L, 4, 0);
	lua_Integer d = luaL_optinteger(L, 5, 1);
	struct ggml_tensor *tw = w->t, *tx = x->t, *wf, *xb, *col, *r;
	int64_t k = tw->ne[0], c = tx->ne[1], l = tx->ne[0], ol;

	luaL_argcheck(L, p >= 0 && p <= 65536, 4, "pad out of range");
	luaL_argcheck(L, d >= 1 && d <= 65536, 5, "dilation out of range");
	check_signal(L, tx, "conv1d_dw_orpheus");
	if (!ggml_is_contiguous(tw) || ggml_nelements(tw) != k * c || tw->ne[3] != 1)
		return luaL_error(L, "conv1d_dw_orpheus: need w (C, 1, K) with C of x");
	ol = l + 2 * p - d * (k - 1);
	if (ol < 1)
		return luaL_error(L, "conv1d_dw_orpheus: kernel larger than padded input");
	wf = as_f32(L, g, tw);
	lgml_reserve(L, g, 5);
	wf = ggml_reshape_4d(g->ctx, wf, k, 1, c, 1);
	xb = ggml_reshape_4d(g->ctx, tx, l, 1, c, 1);
	col = ggml_im2col(g->ctx, wf, xb, 1, 0, (int)p, 0, (int)d, 0, false, GGML_TYPE_F32);
	r = ggml_mul_mat(g->ctx, col, wf);
	r = ggml_reshape_2d(g->ctx, r, r->ne[0], c);
	return lgml_push_node(L, g, r, 2);
}

/*** Transposed 1-D convolution, as torch ConvTranspose1d.
 * The op adds no bias and no output padding.
 * @tparam tensor w contiguous (IC, OC, K)
 * @tparam tensor x contiguous f32 (IC, L) signal
 * @tparam int stride 1 to 65536
 * @tparam[opt=0] int pad 0 to 65536
 * @treturn tensor f32 (OC, (L - 1) * stride + K - 2 * pad) node
 * @raise on bad shapes or ranges
 * @function graph:conv_transpose1d_orpheus
 */
static int
op_conv_transpose1d(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *w = lgml_operand(L, g, 2);
	struct tensor *x = lgml_operand(L, g, 3);
	lua_Integer s = luaL_checkinteger(L, 4);
	lua_Integer p = luaL_optinteger(L, 5, 0);
	struct ggml_tensor *tw = w->t, *tx = x->t, *wp, *xt, *cols, *r;
	int64_t k = tw->ne[0], oc = tw->ne[1], ic = tw->ne[2], l = tx->ne[0];

	luaL_argcheck(L, s >= 1 && s <= 65536, 4, "stride out of range");
	luaL_argcheck(L, p >= 0 && p <= 65536, 5, "pad out of range");
	check_signal(L, tx, "conv_transpose1d_orpheus");
	if (!ggml_is_contiguous(tw) || tw->ne[3] != 1 || ic != tx->ne[1])
		return luaL_error(L, "conv_transpose1d_orpheus: need w (IC, OC, K) with IC of x");
	if (k * oc > INT32_MAX || (l - 1) * s + k - 2 * p < 1)
		return luaL_error(L, "conv_transpose1d_orpheus: bad kernel or pad");
	lgml_reserve(L, g, 6);
	/* Columns (L, OC * K), K innermost, as col2im_1d wants. */
	wp = ggml_reshape_2d(g->ctx, tw, k * oc, ic);
	wp = ggml_cont(g->ctx, ggml_transpose(g->ctx, wp));
	xt = ggml_cont(g->ctx, ggml_transpose(g->ctx, tx));
	cols = ggml_mul_mat(g->ctx, wp, xt);
	r = ggml_col2im_1d(g->ctx, cols, (int)s, (int)oc, (int)p);
	return lgml_push_node(L, g, r, 2);
}

static const luaL_Reg ops[] = {
	{ "orpheus_snake", op_snake },
	{ "conv1d_dw_orpheus", op_conv1d_dw },
	{ "conv_transpose1d_orpheus", op_conv_transpose1d },
	{ NULL, NULL }
};

void
lgml_open_ops_orpheus(lua_State *L)
{
	lgml_graph_ops(L, ops);
}
