/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Graph ops for mixture-of-experts, state-space and diffusion models,
 * and a few general ones. Shapes use torch order; each op notes the
 * ggml layout when it differs.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

typedef struct ggml_tensor *(*unfn)(struct ggml_context *, struct ggml_tensor *);
typedef struct ggml_tensor *(*binfn)(struct ggml_context *, struct ggml_tensor *,
    struct ggml_tensor *);

static void
check_f32(lua_State *L, const struct ggml_tensor *t, const char *op, const char *what)
{
	if (t->type != GGML_TYPE_F32)
		luaL_error(L, "%s: %s must be f32", op, what);
}

static int
unop(lua_State *L, unfn fn, const char *op)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);

	check_f32(L, a->t, op, "a");
	if (!ggml_is_contiguous_rows(a->t))
		return luaL_error(L, "%s: rows not contiguous; use cont", op);
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, fn(g->ctx, a->t), a->rank);
}

/*** A graph of ops on one backend.
 * Each op returns a node tensor. Nodes get storage when `graph:compute`
 * runs. An operand is a leaf on the same backend or a node of this graph.
 * The graph keeps its leaves alive until it closes.
 * @type graph
 */

/*** Elementwise natural log.
 * @tparam tensor a f32 input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if a is not f32 or its rows are not contiguous
 * @function graph:log
 */
static int op_log(lua_State *L) { return unop(L, ggml_log, "log"); }

/*** Elementwise step: 1 where a > 0, else 0.
 * @tparam tensor a f32 input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if a is not f32 or its rows are not contiguous
 * @function graph:step
 */
static int op_step(lua_State *L) { return unop(L, ggml_step, "step"); }

/*** Elementwise softplus: log(1 + exp(a)).
 * @tparam tensor a f32 input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if a is not f32 or its rows are not contiguous
 * @function graph:softplus
 */
static int op_softplus(lua_State *L) { return unop(L, ggml_softplus, "softplus"); }

/*** Elementwise exp(a) - 1.
 * @tparam tensor a f32 input with contiguous rows
 * @treturn tensor node with the shape of a
 * @raise if a is not f32 or its rows are not contiguous
 * @function graph:expm1
 */
static int op_expm1(lua_State *L) { return unop(L, ggml_expm1, "expm1"); }

/*** Running sum along the innermost dimension.
 * @tparam tensor a contiguous f32 input
 * @treturn tensor node with the shape of a
 * @raise if a is not contiguous f32
 * @function graph:cumsum
 */
static int
op_cumsum(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);

	check_f32(L, a->t, "cumsum", "a");
	if (!ggml_is_contiguous(a->t))
		return luaL_error(L, "cumsum: tensor not contiguous; use cont");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_cumsum(g->ctx, a->t), a->rank);
}

/* Gated linear unit on rows that hold [x | gate]. */
static int
glu(lua_State *L, unfn fn, const char *op)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);

	check_f32(L, a->t, op, "a");
	if (a->t->ne[0] % 2 != 0)
		return luaL_error(L, "%s: last dimension must be even", op);
	if (!ggml_is_contiguous_rows(a->t))
		return luaL_error(L, "%s: rows not contiguous; use cont", op);
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, fn(g->ctx, a->t), a->rank);
}

/*** GEGLU on packed input.
 * The innermost dimension holds [x | gate]. The result is gelu(x) * gate,
 * half as wide.
 * @tparam tensor a f32 input with contiguous rows and an even innermost size
 * @treturn tensor node
 * @raise if a is not f32, the innermost size is odd, or rows are not contiguous
 * @function graph:geglu
 */
static int op_geglu(lua_State *L) { return glu(L, ggml_geglu, "geglu"); }

/*** SwiGLU on packed input.
 * The innermost dimension holds [x | gate]. The result is silu(x) * gate,
 * half as wide.
 * @tparam tensor a f32 input with contiguous rows and an even innermost size
 * @treturn tensor node
 * @raise if a is not f32, the innermost size is odd, or rows are not contiguous
 * @function graph:swiglu
 */
static int op_swiglu(lua_State *L) { return glu(L, ggml_swiglu, "swiglu"); }

/* Gated linear unit on separate x and gate tensors. */
static int
glu_split(lua_State *L, binfn fn, const char *op)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	struct tensor *b = lgml_operand(L, g, 3);

	check_f32(L, a->t, op, "x");
	check_f32(L, b->t, op, "gate");
	if (!ggml_are_same_shape(a->t, b->t))
		return luaL_error(L, "%s: x and gate must have the same shape", op);
	if (!ggml_is_contiguous_rows(a->t) || !ggml_is_contiguous_rows(b->t))
		return luaL_error(L, "%s: rows not contiguous; use cont", op);
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, fn(g->ctx, a->t, b->t), a->rank);
}

/*** GEGLU on split input: gelu(x) * gate.
 * @tparam tensor x f32 input with contiguous rows
 * @tparam tensor gate f32 gate with the shape of x
 * @treturn tensor node with the shape of x
 * @raise if an input is not f32, the shapes differ, or rows are not contiguous
 * @function graph:geglu_split
 */
static int op_geglu_split(lua_State *L) { return glu_split(L, ggml_geglu_split, "geglu_split"); }

/*** SwiGLU on split input: silu(x) * gate.
 * @tparam tensor x f32 input with contiguous rows
 * @tparam tensor gate f32 gate with the shape of x
 * @treturn tensor node with the shape of x
 * @raise if an input is not f32, the shapes differ, or rows are not contiguous
 * @function graph:swiglu_split
 */
static int op_swiglu_split(lua_State *L) { return glu_split(L, ggml_swiglu_split, "swiglu_split"); }

/*** Divide each row by its L2 norm.
 * @tparam tensor a f32 input
 * @tparam[opt=1e-6] number eps lower bound on the norm
 * @treturn tensor node with the shape of a
 * @raise if a is not f32
 * @function graph:l2_norm
 */
static int
op_l2_norm(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	float eps = (float)luaL_optnumber(L, 3, 1e-6);

	check_f32(L, a->t, "l2_norm", "a");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_l2_norm(g->ctx, a->t, eps), a->rank);
}

/*** Group normalization without weight or bias.
 * The C channels split into groups. Each group is normalized over its
 * channels and positions.
 * @tparam tensor a contiguous f32 (N, C, H, W) or (C, H, W)
 * @tparam int groups number of groups; it divides C
 * @tparam[opt=1e-6] number eps added to the variance
 * @treturn tensor node with the shape of a
 * @raise if a is not contiguous f32, or groups does not divide C
 * @function graph:group_norm
 */
static int
op_group_norm(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	lua_Integer n = luaL_checkinteger(L, 3);
	float eps = (float)luaL_optnumber(L, 4, 1e-6);

	check_f32(L, a->t, "group_norm", "a");
	luaL_argcheck(L, n >= 1 && n <= a->t->ne[2] && a->t->ne[2] % n == 0, 3,
	    "groups must divide the channel count");
	if (!ggml_is_contiguous(a->t))
		return luaL_error(L, "group_norm: tensor not contiguous; use cont");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_group_norm(g->ctx, a->t, (int)n, eps), a->rank);
}

/*** Indices that sort each row.
 * Indices count from 0.
 * @tparam tensor a f32 input
 * @tparam[opt="asc"] string order "asc" or "desc"
 * @treturn tensor i32 node with the shape of a
 * @raise if a is not f32 or a row is too long
 * @function graph:argsort
 */
static int
op_argsort(lua_State *L)
{
	static const char *const names[] = { "asc", "desc", NULL };
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	int order = luaL_checkoption(L, 3, "asc", names);

	check_f32(L, a->t, "argsort", "a");
	if (a->t->ne[0] > INT32_MAX)
		return luaL_error(L, "argsort: rows too long");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_argsort(g->ctx, a->t,
	    order == 0 ? GGML_SORT_ORDER_ASC : GGML_SORT_ORDER_DESC), a->rank);
}

/*** Indices of the k largest values in each row, largest first.
 * Indices count from 0. The result is a view. Use `graph:top_k` when the
 * order does not matter.
 * @tparam tensor a f32 input
 * @tparam int k 1 to the row length
 * @treturn tensor i32 view node with k entries in each row
 * @raise if a is not f32 or k is out of range
 * @function graph:argsort_top_k
 */
static int
op_argsort_top_k(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	lua_Integer k = luaL_checkinteger(L, 3);

	check_f32(L, a->t, "argsort_top_k", "a");
	luaL_argcheck(L, k >= 1 && k <= a->t->ne[0], 3, "k out of range");
	lgml_reserve(L, g, 2);
	return lgml_push_node(L, g, ggml_argsort_top_k(g->ctx, a->t, (int)k), a->rank);
}

/*** Mixture-of-experts matrix product.
 * w holds one (M, K) matrix for each of E experts. U is 1 or `n_used`. Row
 * (t, j) of the result is w[ids[t, j]] times x[t, j % U]. Ids count from 0.
 * @tparam tensor w (E, M, K) expert weights, not transposed
 * @tparam tensor x (T, U, K) input
 * @tparam tensor ids i32 (T, `n_used`) expert indices
 * @treturn tensor f32 (T, `n_used`, M) node
 * @raise if ids is not i32 or the shapes do not match
 * @function graph:mul_mat_id
 */
static int
op_mul_mat_id(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *w = lgml_operand(L, g, 2);
	struct tensor *x = lgml_operand(L, g, 3);
	struct tensor *ids = lgml_operand(L, g, 4);
	const struct ggml_tensor *tw = w->t, *tx = x->t, *ti = ids->t;

	if (ti->type != GGML_TYPE_I32)
		return luaL_error(L, "mul_mat_id: ids must be i32");
	if (ggml_is_transposed(tw) || tw->ne[3] != 1 || tx->ne[3] != 1 ||
	    ti->ne[2] != 1 || ti->ne[3] != 1)
		return luaL_error(L, "mul_mat_id: need w (E, M, K), x (T, U, K), ids (T, n_used)");
	if (tw->ne[0] != tx->ne[0])
		return luaL_error(L, "mul_mat_id: w and x must share K");
	if (ti->ne[1] != tx->ne[2] || ti->ne[0] % tx->ne[1] != 0)
		return luaL_error(L, "mul_mat_id: ids must be (T, n_used), n_used a multiple of U");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_mul_mat_id(g->ctx, w->t, x->t, ids->t), 3);
}

/*** Add a bias for each selected expert.
 * Row (t, j) of a gets b[ids[t, j]] added. Ids count from 0.
 * @tparam tensor a f32 (T, `n_used`, M) input
 * @tparam tensor b f32 (E, M) biases
 * @tparam tensor ids i32 (T, `n_used`) expert indices
 * @treturn tensor node with the shape of a
 * @raise on bad types or shapes
 * @function graph:add_id
 */
static int
op_add_id(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	struct tensor *b = lgml_operand(L, g, 3);
	struct tensor *ids = lgml_operand(L, g, 4);
	const struct ggml_tensor *ta = a->t, *tb = b->t, *ti = ids->t;

	check_f32(L, ta, "add_id", "a");
	check_f32(L, tb, "add_id", "b");
	if (ti->type != GGML_TYPE_I32)
		return luaL_error(L, "add_id: ids must be i32");
	if (ta->ne[0] != tb->ne[0] || ta->ne[1] != ti->ne[0] || ta->ne[2] != ti->ne[1] ||
	    ta->ne[3] != 1 || tb->ne[2] != 1 || tb->ne[3] != 1)
		return luaL_error(L, "add_id: need a (T, n_used, M), b (E, M), ids (T, n_used)");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_add_id(g->ctx, a->t, b->t, ids->t), a->rank);
}

/*** Causal depthwise convolution for state-space layers.
 * The op adds no padding: x holds `d_conv` - 1 earlier steps and then T new
 * steps, so L = `d_conv` - 1 + T.
 * @tparam tensor x f32 (S, D, L) input
 * @tparam tensor w f32 (D, `d_conv`) kernel
 * @treturn tensor f32 (S, T, D) node
 * @raise on bad types or shapes
 * @function graph:ssm_conv
 */
static int
op_ssm_conv(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = lgml_operand(L, g, 2);
	struct tensor *w = lgml_operand(L, g, 3);
	const struct ggml_tensor *tx = x->t, *tw = w->t;

	check_f32(L, tx, "ssm_conv", "x");
	check_f32(L, tw, "ssm_conv", "w");
	if (tx->ne[3] != 1 || tw->ne[2] != 1 || tw->ne[3] != 1)
		return luaL_error(L, "ssm_conv: need x (S, D, L) and w (D, d_conv)");
	if (tx->ne[1] != tw->ne[1] || tx->ne[0] < tw->ne[0] - 1)
		return luaL_error(L, "ssm_conv: need x (S, D, d_conv - 1 + T) for w (D, d_conv)");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_ssm_conv(g->ctx, x->t, w->t), 3);
}

/*** Gated delta rule linear attention, as in Qwen3-Next.
 * Hk divides H. The op scales q by 1 / sqrt(S). The result holds the
 * outputs, then K state snapshots.
 * @tparam tensor q f32 (B, T, Hk, S) queries with contiguous rows
 * @tparam tensor k f32 (B, T, Hk, S) keys with contiguous rows
 * @tparam tensor v f32 (B, T, H, S) values with contiguous rows
 * @tparam tensor gate contiguous f32 (B, T, H, 1) or (B, T, H, S)
 * @tparam tensor beta contiguous f32 (B, T, H, 1)
 * @tparam tensor state contiguous f32 (B, H, S, S) initial state
 * @tparam[opt=1] int K number of state snapshots, 1 to 64
 * @treturn tensor f32 (T * B + K * S * B, S * H) node
 * @raise on bad types, shapes or layout
 * @function graph:gated_delta_net
 */
static int
op_gated_delta_net(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *q = lgml_operand(L, g, 2);
	struct tensor *k = lgml_operand(L, g, 3);
	struct tensor *v = lgml_operand(L, g, 4);
	struct tensor *ga = lgml_operand(L, g, 5);
	struct tensor *be = lgml_operand(L, g, 6);
	struct tensor *st = lgml_operand(L, g, 7);
	lua_Integer nk = luaL_optinteger(L, 8, 1);
	const struct ggml_tensor *tq = q->t, *tk = k->t, *tv = v->t;
	int64_t sv = tv->ne[0], h = tv->ne[1], nt = tv->ne[2], ns = tv->ne[3];
	int i;

	for (i = 2; i <= 7; i++)
		check_f32(L, lgml_check_tensor(L, i)->t, "gated_delta_net", "every input");
	luaL_argcheck(L, nk >= 1 && nk <= 64, 8, "K out of range");
	if (!ggml_is_contiguous_rows(tq) || !ggml_is_contiguous_rows(tk) ||
	    !ggml_is_contiguous_rows(tv) || !ggml_is_contiguous(ga->t) ||
	    !ggml_is_contiguous(be->t) || !ggml_is_contiguous(st->t))
		return luaL_error(L, "gated_delta_net: inputs not contiguous; use cont");
	if (tq->ne[0] != sv || tk->ne[0] != sv || !ggml_are_same_shape(tq, tk) ||
	    tq->ne[1] < 1 || h % tq->ne[1] != 0 || tq->ne[2] != nt || tq->ne[3] != ns)
		return luaL_error(L, "gated_delta_net: q and k must be (B, T, Hk, S) for v (B, T, H, S)");
	if ((ga->t->ne[0] != 1 && ga->t->ne[0] != sv) || ga->t->ne[1] != h ||
	    ga->t->ne[2] != nt || ga->t->ne[3] != ns)
		return luaL_error(L, "gated_delta_net: gate must be (B, T, H, 1) or (B, T, H, S)");
	if (be->t->ne[0] != 1 || be->t->ne[1] != h || be->t->ne[2] != nt || be->t->ne[3] != ns)
		return luaL_error(L, "gated_delta_net: beta must be (B, T, H, 1)");
	if (st->t->ne[0] != sv || st->t->ne[1] != sv || st->t->ne[2] != h || st->t->ne[3] != ns)
		return luaL_error(L, "gated_delta_net: state must be (B, H, S, S)");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_gated_delta_net(g->ctx, q->t, k->t, v->t, ga->t,
	    be->t, st->t, (int64_t)nk), 2);
}

/*** Pad with zeros.
 * @tparam tensor a f32 input
 * @tparam {{int,int},...} pads one { before, after } pair for each
 * dimension of a, in torch order; each 0 to 65536
 * @treturn tensor node with the rank of a
 * @raise if a is not f32 or pads is malformed
 * @function graph:pad
 */
static int
op_pad(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	int p[GGML_MAX_DIMS][2] = { { 0, 0 } };
	lua_Integer v;
	int i, j, d, n = a->rank;

	check_f32(L, a->t, "pad", "a");
	luaL_checktype(L, 3, LUA_TTABLE);
	luaL_argcheck(L, luaL_len(L, 3) == n, 3, "need one pair per dimension");
	for (i = 0; i < n; i++) {
		lua_geti(L, 3, i + 1);
		luaL_argcheck(L, lua_istable(L, -1), 3, "need { before, after } pairs");
		d = n - 1 - i;
		for (j = 0; j < 2; j++) {
			lua_geti(L, -1, j + 1);
			v = luaL_checkinteger(L, -1);
			luaL_argcheck(L, v >= 0 && v <= 65536, 3, "pad out of range");
			p[d][j] = (int)v;
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
	}
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_pad_ext(g->ctx, a->t, p[0][0], p[0][1], p[1][0],
	    p[1][1], p[2][0], p[2][1], p[3][0], p[3][1]), n);
}

/*** Scale the two innermost dimensions by an integer factor.
 * @tparam tensor a f32 (N, C, H, W) input
 * @tparam int factor 1 to 16
 * @tparam[opt="nearest"] string mode "nearest" or "bilinear"
 * @treturn tensor node with H and W times factor
 * @raise if a is not f32, factor is out of range, or the result is too large
 * @function graph:upscale
 */
static int
op_upscale(lua_State *L)
{
	static const char *const names[] = { "nearest", "bilinear", NULL };
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	lua_Integer f = luaL_checkinteger(L, 3);
	int mode = luaL_checkoption(L, 4, "nearest", names);

	check_f32(L, a->t, "upscale", "a");
	luaL_argcheck(L, f >= 1 && f <= 16, 3, "factor out of range");
	if (a->t->ne[0] * f > (int64_t)1 << 20 || a->t->ne[1] * f > (int64_t)1 << 20)
		return luaL_error(L, "upscale: result too large");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_upscale(g->ctx, a->t, (int)f,
	    mode == 0 ? GGML_SCALE_MODE_NEAREST : GGML_SCALE_MODE_BILINEAR), a->rank);
}

/*** 2-D convolution, as torch conv2d without bias.
 * stride, pad and dilation apply to both axes.
 * @tparam tensor w contiguous (OC, IC, KH, KW)
 * @tparam tensor x contiguous f32 (N, IC, H, W)
 * @tparam[opt=1] int stride 1 to 1024
 * @tparam[opt=0] int pad zeros on each side, 0 to 1024
 * @tparam[opt=1] int dilation 1 to 1024
 * @treturn tensor (N, OC, OH, OW) node
 * @raise on bad shapes, types or ranges, or if the kernel is larger than
 * the padded input
 * @function graph:conv2d
 */
static int
op_conv2d(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *w = lgml_operand(L, g, 2);
	struct tensor *x = lgml_operand(L, g, 3);
	lua_Integer s = luaL_optinteger(L, 4, 1);
	lua_Integer p = luaL_optinteger(L, 5, 0);
	lua_Integer d = luaL_optinteger(L, 6, 1);
	const struct ggml_tensor *tw = w->t, *tx = x->t;
	int i;

	check_f32(L, tx, "conv2d", "x");
	luaL_argcheck(L, s >= 1 && s <= 1024, 4, "stride out of range");
	luaL_argcheck(L, p >= 0 && p <= 1024, 5, "pad out of range");
	luaL_argcheck(L, d >= 1 && d <= 1024, 6, "dilation out of range");
	if (tw->ne[2] != tx->ne[2] || !ggml_is_contiguous(tw) || !ggml_is_contiguous(tx))
		return luaL_error(L, "conv2d: need contiguous w (OC, IC, KH, KW) and x (N, IC, H, W)");
	for (i = 0; i < 2; i++)
		if (tx->ne[i] + 2 * p - d * (tw->ne[i] - 1) - 1 < 0)
			return luaL_error(L, "conv2d: kernel larger than padded input");
	lgml_reserve(L, g, 8);
	return lgml_push_node(L, g, ggml_conv_2d(g->ctx, w->t, x->t, (int)s, (int)s, (int)p,
	    (int)p, (int)d, (int)d), 4);
}

/*** Add nodes to the graph without making them outputs.
 * `graph:compute` runs them too. Use it for writes into leaves, such as a
 * `graph:cpy` that saves recurrent state.
 * @tparam tensor ... nodes of this graph
 * @raise if a tensor is not a node of this graph
 * @function graph:expand
 */
static int
op_expand(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *t;
	int i, n = lua_gettop(L);

	for (i = 2; i <= n; i++) {
		t = lgml_check_tensor(L, i);
		luaL_argcheck(L, t->g == g, i, "not a node of this graph");
		ggml_build_forward_expand(g->gf, t->t);
	}
	return 0;
}

/***
 * @section end
 */

/*** Quantize f32 values with the ggml reference quantizer.
 * @tparam string type ggml type name; types that need an importance
 * matrix do not work
 * @tparam string data f32 values as raw bytes
 * @tparam[opt] int n_per_row values in each row; the default is all values
 * @treturn string quantized bytes
 * @raise if the type is not supported, or `n_per_row` does not divide the
 * values or is not a multiple of the block size
 * @function quantize
 */
static int
l_quantize(lua_State *L)
{
	const char *name = luaL_checkstring(L, 1);
	size_t len, size;
	const char *src = luaL_checklstring(L, 2, &len);
	int64_t n = (int64_t)(len / sizeof(float));
	lua_Integer per = luaL_optinteger(L, 3, n);
	enum ggml_type type = GGML_TYPE_COUNT;
	luaL_Buffer lb;
	float *f;
	int i;

	for (i = 0; i < GGML_TYPE_COUNT; i++)
		if (ggml_type_name((enum ggml_type)i) != NULL &&
		    strcmp(ggml_type_name((enum ggml_type)i), name) == 0)
			type = (enum ggml_type)i;
	if (type == GGML_TYPE_COUNT || ggml_quantize_requires_imatrix(type))
		return luaL_error(L, "quantize: unsupported type '%s'", name);
	luaL_argcheck(L, len % sizeof(float) == 0 && n > 0, 2, "need f32 values");
	luaL_argcheck(L, per >= 1 && n % per == 0 && per % ggml_blck_size(type) == 0, 3,
	    "n_per_row must divide the values and be a multiple of the block size");
	f = malloc(len);
	if (f == NULL)
		return luaL_error(L, "out of memory");
	memcpy(f, src, len);
	size = ggml_row_size(type, per) * (size_t)(n / per);
	ggml_quantize_chunk(type, f, luaL_buffinitsize(L, &lb, size), 0, n / per, per, NULL);
	free(f);
	luaL_pushresultsize(&lb, size);
	return 1;
}

/***
 * Multiply rows by a Walsh-Hadamard matrix. rot is the f32 (n, n)
 * matrix from `tensor:fill("hadamard")` and x is f32 (rows, n). The op
 * carries the ggml Hadamard hint where the backend takes it, and runs
 * the fast transform there; elsewhere it is a plain matmul.
 * @tparam tensor rot the matrix
 * @tparam tensor x the rows
 * @treturn tensor f32 (rows, n)
 * @raise when rot is not square f32 with a power of 2 side, or x does
 * not match
 * @function graph:hadamard
 */
static int
op_hadamard(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *h = lgml_operand(L, g, 2);
	struct tensor *x = lgml_operand(L, g, 3);
	const struct ggml_tensor *th = h->t, *tx = x->t;
	struct ggml_tensor *r;
	int64_t n = th->ne[0];

	check_f32(L, th, "hadamard", "rot");
	check_f32(L, tx, "hadamard", "x");
	if (th->ne[1] != n || th->ne[2] != 1 || th->ne[3] != 1 || (n & (n - 1)) != 0)
		return luaL_error(L, "hadamard: rot must be square with a power of 2 side");
	if (tx->ne[0] != n || !ggml_is_contiguous(tx))
		return luaL_error(L, "hadamard: x must be contiguous with rows of %d", (int)n);
	lgml_reserve(L, g, 1);
	r = ggml_mul_mat(g->ctx, h->t, x->t);
	ggml_mul_mat_set_hint(r, GGML_HINT_SRC0_IS_HADAMARD);
	if (!ggml_backend_supports_op(g->b->be, r))
		ggml_mul_mat_set_hint(r, GGML_HINT_NONE);
	return lgml_push_node(L, g, r, x->rank);
}

static const luaL_Reg ops[] = {
	{ "log", op_log },
	{ "step", op_step },
	{ "softplus", op_softplus },
	{ "expm1", op_expm1 },
	{ "cumsum", op_cumsum },
	{ "geglu", op_geglu },
	{ "swiglu", op_swiglu },
	{ "geglu_split", op_geglu_split },
	{ "swiglu_split", op_swiglu_split },
	{ "l2_norm", op_l2_norm },
	{ "group_norm", op_group_norm },
	{ "argsort", op_argsort },
	{ "argsort_top_k", op_argsort_top_k },
	{ "mul_mat_id", op_mul_mat_id },
	{ "add_id", op_add_id },
	{ "ssm_conv", op_ssm_conv },
	{ "gated_delta_net", op_gated_delta_net },
	{ "pad", op_pad },
	{ "upscale", op_upscale },
	{ "conv2d", op_conv2d },
	{ "hadamard", op_hadamard },
	{ "expand", op_expand },
	{ NULL, NULL }
};

void
lgml_open_ops_nn(lua_State *L)
{
	lgml_graph_ops(L, ops);
	lua_pushcfunction(L, l_quantize);
	lua_setfield(L, -2, "quantize");
}
