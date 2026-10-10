/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Autograd and optimizer steps. A trainable leaf owns a gradient leaf.
 * tensor:backward adds the backward pass to the graph of the loss, and
 * each compute of that graph adds the new gradients to the gradient
 * leaves.
 */

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

/*
 * New ggml tensors that backward can add for each node on the path,
 * with the out_prod operands of lower_out_prod.
 */
#define BACK_PER_NODE	24

/*** A leaf with storage, or a node of a graph.
 * A leaf comes from `backend:tensor` or `weights:get`. A node comes from a
 * graph op and gets storage when `graph:compute` runs. The rank is 1 to 4.
 * @type tensor
 */

/* Push a zeroed f32 leaf with the shape of gt, on the backend of the leaf at tidx. */
static struct tensor *
push_zeros(lua_State *L, int tidx, const struct ggml_tensor *gt, int rank)
{
	struct ggml_init_params ip = { ggml_tensor_overhead(), NULL, true };
	struct ggml_context *ctx;
	struct ggml_tensor *nt;
	ggml_backend_buffer_t buf;
	struct backend *b;
	struct store *s;
	struct tensor *t;

	lua_getiuservalue(L, tidx, UV_PARENT);
	lua_getiuservalue(L, -1, UV_PARENT);
	b = lgml_check_backend(L, -1);
	ctx = ggml_init(ip);
	if (ctx == NULL)
		luaL_error(L, "ggml_init failed");
	nt = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, gt->ne);
	buf = ggml_backend_alloc_ctx_tensors(ctx, b->be);
	if (buf == NULL) {
		ggml_free(ctx);
		luaL_error(L, "out of device memory");
	}
	ggml_backend_buffer_clear(buf, 0);
	s = lgml_push_store(L, -1, ctx, buf);
	/* The store frees itself when its one leaf closes. */
	s->closed = 1;
	t = lgml_push_leaf(L, -1, nt, rank);
	lua_replace(L, -4);
	lua_pop(L, 2);
	return t;
}

/*** Mark a leaf as trainable, or frozen.
 * A trainable leaf is contiguous f32. The first call makes its gradient,
 * a zeroed f32 leaf of the same shape. `tensor:backward` gives gradients
 * to trainable leaves only.
 * @tparam[opt=true] boolean flag false to freeze the leaf
 * @treturn tensor the leaf
 * @raise if the tensor is not a leaf, or is not contiguous f32
 * @function tensor:requires_grad
 */
static int
tensor_requires_grad(lua_State *L)
{
	struct tensor *t = lgml_check_tensor(L, 1);
	int on = lua_isnoneornil(L, 2) || lua_toboolean(L, 2);

	lua_settop(L, 2);
	if (t->s == NULL)
		return luaL_error(L, "requires_grad: need a leaf tensor");
	if (!on) {
		t->t->flags &= ~GGML_TENSOR_FLAG_PARAM;
		lua_settop(L, 1);
		return 1;
	}
	if (t->t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t->t))
		return luaL_error(L, "requires_grad: need a contiguous f32 leaf");
	if (lua_getiuservalue(L, 1, UV_GRAD) == LUA_TNIL) {
		push_zeros(L, 1, t->t, t->rank);
		lua_setiuservalue(L, 1, UV_GRAD);
	}
	ggml_set_param(t->t);
	lua_settop(L, 1);
	return 1;
}

/*** Get the gradient of a trainable leaf.
 * The gradient is an f32 leaf. Each compute of a graph with a backward
 * pass adds to it; `tensor:zero` clears it.
 * @treturn tensor|nil the gradient, or nil if the leaf never was trainable
 * @function tensor:grad
 */
static int
tensor_grad(lua_State *L)
{
	struct tensor *t = lgml_check_tensor(L, 1);

	if (t->s == NULL)
		lua_pushnil(L);
	else
		lua_getiuservalue(L, 1, UV_GRAD);
	return 1;
}

/*** Tell if a leaf is trainable.
 * @treturn boolean true after `tensor:requires_grad`
 * @function tensor:is_param
 */
static int
tensor_is_param(lua_State *L)
{
	struct tensor *t = lgml_check_tensor(L, 1);

	lua_pushboolean(L, (t->t->flags & GGML_TENSOR_FLAG_PARAM) != 0);
	return 1;
}

/* Pointer to node index, for the nodes of one graph. */
struct index {
	const struct ggml_tensor	**keys;
	int				 *vals;
	size_t				  cap;
};

static size_t
hash_ptr(const void *p, size_t cap)
{
	uintptr_t x = (uintptr_t)p;

	x ^= x >> 17;
	x *= 0xed5ad4bbU;
	x ^= x >> 11;
	return (size_t)x % cap;
}

static int
index_init(struct index *ix, struct ggml_cgraph *gf)
{
	int i, n = ggml_graph_n_nodes(gf);
	size_t h;

	ix->cap = (size_t)n * 2 + 1;
	ix->keys = calloc(ix->cap, sizeof(*ix->keys));
	ix->vals = calloc(ix->cap, sizeof(*ix->vals));
	if (ix->keys == NULL || ix->vals == NULL)
		return -1;
	for (i = 0; i < n; i++) {
		h = hash_ptr(ggml_graph_node(gf, i), ix->cap);
		while (ix->keys[h] != NULL)
			h = (h + 1) % ix->cap;
		ix->keys[h] = ggml_graph_node(gf, i);
		ix->vals[h] = i;
	}
	return 0;
}

static int
index_find(const struct index *ix, const struct ggml_tensor *t)
{
	size_t h;

	if (t == NULL)
		return -1;
	for (h = hash_ptr(t, ix->cap); ix->keys[h] != NULL; h = (h + 1) % ix->cap)
		if (ix->keys[h] == t)
			return ix->vals[h];
	return -1;
}

static void
index_free(struct index *ix)
{
	free(ix->keys);
	free(ix->vals);
}

/*
 * The name of an op without a usable gradient in ggml, or NULL. The
 * gradients of mean and add1 fail on a trainable leaf.
 */
static const char *
no_grad_op(const struct ggml_tensor *t)
{
	switch (t->op) {
	case GGML_OP_NONE:
	case GGML_OP_DUP:
	case GGML_OP_ADD:
	case GGML_OP_ACC:
	case GGML_OP_SUB:
	case GGML_OP_MUL:
	case GGML_OP_DIV:
	case GGML_OP_SQR:
	case GGML_OP_SQRT:
	case GGML_OP_LOG:
	case GGML_OP_SIN:
	case GGML_OP_COS:
	case GGML_OP_SUM:
	case GGML_OP_SUM_ROWS:
	case GGML_OP_REPEAT:
	case GGML_OP_REPEAT_BACK:
	case GGML_OP_RMS_NORM:
	case GGML_OP_MUL_MAT:
	case GGML_OP_SCALE:
	case GGML_OP_SET:
	case GGML_OP_CPY:
	case GGML_OP_CONT:
	case GGML_OP_RESHAPE:
	case GGML_OP_VIEW:
	case GGML_OP_PERMUTE:
	case GGML_OP_TRANSPOSE:
	case GGML_OP_GET_ROWS:
	case GGML_OP_DIAG_MASK_INF:
	case GGML_OP_DIAG_MASK_ZERO:
	case GGML_OP_SOFT_MAX:
	case GGML_OP_ROPE:
	case GGML_OP_IM2COL:
	case GGML_OP_POOL_2D:
	case GGML_OP_CROSS_ENTROPY_LOSS:
	case GGML_OP_L2_NORM:
	case GGML_OP_CONCAT:
	case GGML_OP_SSM_CONV:
	case GGML_OP_GATED_DELTA_NET:
		return NULL;
	case GGML_OP_UNARY:
		switch (ggml_get_unary_op(t)) {
		case GGML_UNARY_OP_ABS:
		case GGML_UNARY_OP_SGN:
		case GGML_UNARY_OP_NEG:
		case GGML_UNARY_OP_STEP:
		case GGML_UNARY_OP_RELU:
		case GGML_UNARY_OP_SILU:
		case GGML_UNARY_OP_EXP:
		case GGML_UNARY_OP_EXPM1:
		case GGML_UNARY_OP_SOFTPLUS:
		case GGML_UNARY_OP_SIGMOID:
			return NULL;
		default:
			return ggml_unary_op_name(ggml_get_unary_op(t));
		}
	case GGML_OP_GLU:
		if (ggml_get_glu_op(t) == GGML_GLU_OP_SWIGLU && t->src[1] != NULL)
			return NULL;
		return ggml_glu_op_name(ggml_get_glu_op(t));
	default:
		return ggml_op_desc(t);
	}
}

/* Tell if backward ignores the gradient of src j of t, as ggml does. */
static int
ignore_src(const struct ggml_tensor *t, int j)
{
	if (j == 0 && (t->op == GGML_OP_IM2COL || t->op == GGML_OP_IM2COL_BACK))
		return 1;
	if (j == 0 && t->op == GGML_OP_UNARY && (ggml_get_unary_op(t) == GGML_UNARY_OP_SGN ||
	    ggml_get_unary_op(t) == GGML_UNARY_OP_STEP))
		return 1;
	return j == 1 && (t->op == GGML_OP_CPY || t->op == GGML_OP_GET_ROWS ||
	    t->op == GGML_OP_GET_ROWS_BACK || t->op == GGML_OP_ROPE);
}

/*
 * Check that ggml can build the backward pass of gf to loss: ggml aborts
 * where these checks raise. Return the count of nodes that get a gradient.
 */
static int
check_backward(lua_State *L, struct ggml_cgraph *gf, struct ggml_tensor *loss)
{
	struct index ix;
	struct ggml_tensor *t, *s;
	const char *bad = NULL, *why = NULL;
	char *needs, *anc, name[64];
	int i, j, k, n = ggml_graph_n_nodes(gf), count = 0, params = 0;

	needs = calloc((size_t)n + 1, 1);
	anc = calloc((size_t)n + 1, 1);
	if (needs == NULL || anc == NULL || index_init(&ix, gf) != 0) {
		free(needs);
		free(anc);
		return luaL_error(L, "out of memory");
	}
	/* Nodes that depend on a trainable leaf. */
	for (i = 0; i < n && bad == NULL; i++) {
		t = ggml_graph_node(gf, i);
		if (t->flags & GGML_TENSOR_FLAG_PARAM) {
			needs[i] = 1;
			params++;
			continue;
		}
		if (t->type == GGML_TYPE_I32)
			continue;
		for (j = 0; j < GGML_MAX_SRC; j++) {
			k = index_find(&ix, t->src[j]);
			if (k < 0 || !needs[k] || ignore_src(t, j))
				continue;
			s = t->src[j];
			if (s->type != GGML_TYPE_F32 && s->type != GGML_TYPE_F16) {
				bad = ggml_op_desc(t);
				why = "a gradient input is not f32 or f16";
			}
			needs[i] = 1;
		}
		if (needs[i] && t->view_src != NULL && t->op != GGML_OP_CPY &&
		    t->op != GGML_OP_SET_ROWS && t->op != GGML_OP_VIEW &&
		    t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE &&
		    t->op != GGML_OP_TRANSPOSE) {
			bad = ggml_op_desc(t);
			why = "in-place ops have no gradient";
		}
	}
	/* Nodes on a path to the loss. Nodes come after their sources. */
	k = index_find(&ix, loss);
	if (k >= 0)
		anc[k] = 1;
	for (i = n - 1; i >= 0 && bad == NULL; i--) {
		if (!anc[i])
			continue;
		t = ggml_graph_node(gf, i);
		for (j = 0; j < GGML_MAX_SRC; j++) {
			k = index_find(&ix, t->src[j]);
			if (k >= 0)
				anc[k] = 1;
		}
		if (!needs[i] || t->op == GGML_OP_NONE)
			continue;
		/* The ssm_conv gradient takes a few ops per kernel column. */
		count += t->op == GGML_OP_SSM_CONV ? 4 * (int)t->src[1]->ne[0] : 1;
		if ((why = no_grad_op(t)) != NULL) {
			bad = why;
			why = "the op has no gradient in ggml";
		} else if (t->op == GGML_OP_SOFT_MAX && (k = index_find(&ix, t->src[1])) >= 0 &&
		    needs[k]) {
			bad = "soft_max";
			why = "the mask has no gradient";
		} else if (t->op == GGML_OP_ROPE && (k = index_find(&ix, t->src[2])) >= 0 &&
		    needs[k]) {
			bad = "rope";
			why = "the frequency factors have no gradient";
		} else if (t->op == GGML_OP_MUL_MAT &&
		    (ggml_is_transposed(t->src[0]) || ggml_is_transposed(t->src[1]))) {
			bad = "mul_mat";
			why = "a transposed operand has no gradient; use cont";
		}
	}
	index_free(&ix);
	free(needs);
	free(anc);
	if (bad != NULL) {
		/* ggml names ops in capitals; lgml names them in lower case. */
		for (i = 0; bad[i] != '\0' && i < (int)sizeof(name) - 1; i++)
			name[i] = (char)tolower((unsigned char)bad[i]);
		name[i] = '\0';
		return luaL_error(L, "backward: %s: %s", name, why);
	}
	if (params == 0)
		return luaL_error(L, "backward: no trainable leaf; use requires_grad");
	return count;
}

/* Contiguous f32 copy of a for out_prod. ids holds 0, 1, 2, ... as i32. */
static struct ggml_tensor *
as_f32(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *ids)
{
	if (a->type == GGML_TYPE_F32)
		return ggml_is_contiguous(a) ? a : ggml_cont(ctx, a);
	if (a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16)
		return ggml_cast(ctx, a, GGML_TYPE_F32);
	/* get_rows takes quantized rows on more backends than cpy. */
	if (ids == NULL || a->ne[2] != 1 || a->ne[3] != 1 || a->ne[1] > ids->ne[0])
		return a;
	return ggml_get_rows(ctx, a, ggml_view_1d(ctx, ids, a->ne[1], 0));
}

/* Largest f32 copy of a quantized weight to make for one out_prod. */
#define DEQUANT_MAX	((int64_t)1 << 30)

/*
 * The ggml gradient of mul_mat uses out_prod on quantized, transposed
 * or permuted operands. The CPU out_prod dequantizes a quantized weight
 * once for each gradient row; an f32 copy made once is about twice as
 * fast for a whole step. Weights with a copy over DEQUANT_MAX bytes,
 * such as a large output projection, keep the slow form where the
 * backend has it. A backend without the other forms gets contiguous
 * f32 operands. Return nonzero if a node changed.
 */
static int
lower_out_prod(struct graph *g, struct ggml_cgraph *gf, struct ggml_tensor *ids, int all)
{
	struct ggml_tensor *t, *a, *b;
	int i, lack, changed = 0;

	for (i = 0; i < ggml_graph_n_nodes(gf); i++) {
		t = ggml_graph_node(gf, i);
		if (t->op != GGML_OP_OUT_PROD)
			continue;
		a = t->src[0];
		b = t->src[1];
		lack = all || !ggml_backend_supports_op(g->b->be, t);
		if (lack || (ggml_is_quantized(a->type) && 4 * ggml_nelements(a) <= DEQUANT_MAX))
			a = as_f32(g->ctx, a, ids);
		if (lack)
			b = as_f32(g->ctx, b, NULL);
		if (a != t->src[0] || b != t->src[1]) {
			t->src[0] = a;
			t->src[1] = b;
			changed = 1;
		}
	}
	return changed;
}

/* Rows of the largest quantized out_prod operand. */
static int64_t
lower_rows(struct ggml_cgraph *gf)
{
	struct ggml_tensor *t;
	int64_t rows = 0;
	int i;

	for (i = 0; i < ggml_graph_n_nodes(gf); i++) {
		t = ggml_graph_node(gf, i);
		if (t->op == GGML_OP_OUT_PROD && ggml_is_quantized(t->src[0]->type) &&
		    t->src[0]->ne[1] > rows)
			rows = t->src[0]->ne[1];
	}
	return rows;
}

/*
 * Make a 1-D leaf of n elements with the given data, in a store on the
 * backend of the graph at gidx. The graph holds the store until it is
 * collected.
 */
static struct ggml_tensor *
graph_const(lua_State *L, int gidx, enum ggml_type type, int64_t n, const void *data)
{
	struct ggml_init_params ip = { ggml_tensor_overhead(), NULL, true };
	struct ggml_context *ctx;
	struct ggml_tensor *t;
	ggml_backend_buffer_t buf;
	struct backend *b;

	lua_getiuservalue(L, gidx, UV_PARENT);
	b = lgml_check_backend(L, -1);
	ctx = ggml_init(ip);
	if (ctx == NULL)
		luaL_error(L, "ggml_init failed");
	t = ggml_new_tensor_1d(ctx, type, n);
	buf = ggml_backend_alloc_ctx_tensors(ctx, b->be);
	if (buf == NULL) {
		ggml_free(ctx);
		luaL_error(L, "out of device memory");
	}
	lgml_push_store(L, -1, ctx, buf);
	ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
	lua_getiuservalue(L, gidx, UV_EXTRA);
	lua_pushvalue(L, -2);
	lua_pushboolean(L, 1);
	lua_rawset(L, -3);
	lua_pop(L, 3);
	return t;
}

/* Gradient leaves of the trainable leaves of the graph at gidx, by node index. */
static void
grad_accs(lua_State *L, int gidx, struct ggml_cgraph *gf, struct ggml_tensor **accs)
{
	struct tensor *t, *gt;
	struct index ix;
	int k;

	if (index_init(&ix, gf) != 0)
		luaL_error(L, "out of memory");
	lua_getiuservalue(L, gidx, UV_EXTRA);
	lua_pushnil(L);
	while (lua_next(L, -2) != 0) {
		lua_pop(L, 1);
		t = luaL_testudata(L, -1, TENSOR_MT);
		if (t == NULL || t->t == NULL || t->s == NULL ||
		    !(t->t->flags & GGML_TENSOR_FLAG_PARAM))
			continue;
		k = index_find(&ix, t->t);
		if (k < 0)
			continue;
		lua_getiuservalue(L, -1, UV_GRAD);
		gt = luaL_testudata(L, -1, TENSOR_MT);
		if (gt != NULL && gt->t != NULL)
			accs[k] = gt->t;
		lua_pop(L, 1);
	}
	lua_pop(L, 1);
	index_free(&ix);
}

/* Count the ggml tensors in ctx. */
static int
count_tensors(struct ggml_context *ctx)
{
	struct ggml_tensor *t;
	int n = 0;

	for (t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t))
		n++;
	return n;
}

/*** Add the backward pass of this loss to its graph.
 * The loss is an f32 scalar node. Each later `graph:compute` of the
 * graph runs the forward and the backward pass, and adds the gradient
 * of the loss to the gradient of each trainable leaf the loss depends
 * on. A graph takes one backward pass.
 * Some ops have no gradient in ggml, for example `flash_attn`,
 * `mul_mat_id`, `sigmoid`, `tanh`, `gelu` and `set_rows`. Masks of
 * `soft_max` and frequency factors of `rope` get no gradient.
 * @raise if the tensor is not an f32 scalar node, the graph has a
 * backward pass, no trainable leaf feeds the loss, an op on the path
 * has no gradient, or the graph would pass 1048576 tensors
 * @tparam[opt] table opts options: `lower` true gives every out_prod of
 * the backward pass contiguous f32 operands, as on a backend without
 * the other forms (for tests)
 * @function tensor:backward
 */
static int
tensor_backward(lua_State *L)
{
	struct tensor *t = lgml_check_tensor(L, 1);
	struct graph *g = t->g;
	struct ggml_cgraph *gb, *gn;
	struct ggml_tensor *seed, *ids = NULL, **accs;
	const float one = 1.0f;
	int64_t rows;
	int32_t *v;
	int i, n, need, gidx, before, all = 0;

	if (g == NULL)
		return luaL_error(L, "backward: need a node of a graph");
	if (g->closed)
		return luaL_error(L, "graph closed");
	if (t->t->type != GGML_TYPE_F32 || ggml_nelements(t->t) != 1)
		return luaL_error(L, "backward: the loss must be an f32 scalar");
	if (g->backward)
		return luaL_error(L, "backward: the graph has a backward pass");
	if (lua_istable(L, 2)) {
		lua_getfield(L, 2, "lower");
		all = lua_toboolean(L, -1);
	}
	lua_settop(L, 1);
	lua_getiuservalue(L, 1, UV_PARENT);
	gidx = lua_gettop(L);
	ggml_set_output(t->t);
	ggml_build_forward_expand(g->gf, t->t);
	/* ggml aborts when a context or cgraph is full, so grow both first. */
	need = check_backward(L, g->gf, t->t) * BACK_PER_NODE + 64;
	lgml_graph_room(L, g, need);

	n = ggml_graph_n_nodes(g->gf);
	accs = lua_newuserdatauv(L, (size_t)n * sizeof(*accs), 0);
	memset(accs, 0, (size_t)n * sizeof(*accs));
	grad_accs(L, gidx, g->gf, accs);
	seed = graph_const(L, gidx, GGML_TYPE_F32, 1, &one);
	for (i = 0; i < n; i++)
		if (ggml_graph_node(g->gf, i) == t->t)
			accs[i] = seed;

	ggml_set_loss(t->t);
	gb = lgml_new_cgraph(L, g, g->size, 1);
	ggml_graph_cpy(g->gf, gb);
	g->gf = gb;
	g->grads = 1;
	before = count_tensors(g->ctx);
	ggml_build_backward_expand(g->ctx, gb, accs);
	rows = lower_rows(gb);
	if (rows > 0) {
		v = lua_newuserdatauv(L, (size_t)rows * sizeof(*v), 0);
		for (i = 0; i < rows; i++)
			v[i] = i;
		ids = graph_const(L, gidx, GGML_TYPE_I32, rows, v);
	}
	if (lower_out_prod(g, gb, ids, all)) {
		/* Order the new operands before their users. */
		gn = lgml_new_cgraph(L, g, g->size, 0);
		for (i = 0; i < ggml_graph_n_nodes(gb); i++)
			ggml_build_forward_expand(gn, ggml_graph_node(gb, i));
		g->gf = gn;
		g->grads = 0;
	}
	n = count_tensors(g->ctx) - before;
	g->ntensors += n;
	g->ctx_left -= n;
	g->backward = 1;
	return 0;
}

/* graph ops */

/*** Cross-entropy loss of rows of logits against label distributions.
 * The loss is the mean over rows of -sum(labels * log_softmax(logits)).
 * Each row of labels is a distribution, such as a one-hot row from
 * `tensor:fill_onehot`.
 * @tparam tensor logits contiguous f32 (rows, classes)
 * @tparam tensor labels contiguous f32 with the shape of logits
 * @treturn tensor f32 scalar node, rank 1
 * @raise on bad types or shapes
 * @function graph:cross_entropy_loss
 */
static int
op_cross_entropy_loss(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	struct tensor *b = lgml_operand(L, g, 3);

	if (a->t->type != GGML_TYPE_F32 || b->t->type != GGML_TYPE_F32 ||
	    !ggml_is_contiguous(a->t) || !ggml_is_contiguous(b->t))
		return luaL_error(L, "cross_entropy_loss: need contiguous f32 logits and labels");
	if (!ggml_are_same_shape(a->t, b->t))
		return luaL_error(L, "cross_entropy_loss: logits and labels differ in shape");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_cross_entropy_loss(g->ctx, a->t, b->t), 1);
}

/*** Sum of all elements.
 * @tparam tensor a input
 * @treturn tensor scalar node, rank 1
 * @function graph:sum
 */
static int
op_sum(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);

	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_sum(g->ctx, a->t), 1);
}

/*** Outer product summed over the outer dimension: b^T a.
 * a is (K, M) and b is (K, N); the result is f32 (N, M). Each outer
 * size of b beyond 2 is a multiple of the same size of a.
 * @tparam tensor a (K, M), not transposed
 * @tparam tensor b f32 (K, N)
 * @treturn tensor f32 (N, M) node
 * @raise on a shape mismatch or a transposed a
 * @function graph:out_prod
 */
static int
op_out_prod(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *a = lgml_operand(L, g, 2);
	struct tensor *b = lgml_operand(L, g, 3);
	const struct ggml_tensor *ta = a->t, *tb = b->t;

	if (ta->ne[1] != tb->ne[1] || tb->ne[2] % ta->ne[2] != 0 || tb->ne[3] % ta->ne[3] != 0)
		return luaL_error(L, "out_prod shape mismatch");
	if (ggml_is_transposed(ta) || tb->type != GGML_TYPE_F32)
		return luaL_error(L, "out_prod: need a not transposed and f32 b");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_out_prod(g->ctx, a->t, b->t),
	    a->rank > b->rank ? a->rank : b->rank);
}

/* A trainable leaf with gradient-shaped companions, from args 2 to n. */
static struct tensor *
step_operands(lua_State *L, struct graph *g, int n, struct tensor **ops, const char *name)
{
	struct tensor *p = lgml_operand(L, g, 2);
	int i;

	if (p->s == NULL || !(p->t->flags & GGML_TENSOR_FLAG_PARAM))
		luaL_error(L, "%s: need a trainable leaf", name);
	for (i = 3; i <= n; i++) {
		ops[i - 3] = lgml_operand(L, g, i);
		if (!ggml_are_same_shape(p->t, ops[i - 3]->t) || ops[i - 3]->t->type != GGML_TYPE_F32)
			luaL_error(L, "%s: arg %d must be f32 with the shape of the leaf", name, i);
	}
	return p;
}

static void
check_hparams(lua_State *L, struct tensor *h, int64_t n, const char *name)
{
	if (h->t->type != GGML_TYPE_F32 || ggml_nelements(h->t) != n || !ggml_is_contiguous(h->t))
		luaL_error(L, "%s: hparams must be %d contiguous f32 values", name, (int)n);
}

/*** One AdamW step on a trainable leaf, in place.
 * hparams holds lr, beta1, beta2, eps, weight decay, 1 / (1 - beta1^t)
 * and 1 / (1 - beta2^t). The result is a view of the leaf; compute it,
 * or pass it to `graph:expand`, to do the step.
 * @tparam tensor p trainable f32 leaf
 * @tparam tensor grad f32 gradient with the shape of p
 * @tparam tensor m f32 first moment with the shape of p
 * @tparam tensor v f32 second moment with the shape of p
 * @tparam tensor hparams f32 (7)
 * @treturn tensor view node of p
 * @raise on bad types or shapes, or if p is not trainable
 * @function graph:opt_step_adamw
 */
static int
op_opt_step_adamw(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *o[4], *p = step_operands(L, g, 5, o, "opt_step_adamw");
	struct tensor *h = lgml_operand(L, g, 6);

	check_hparams(L, h, 7, "opt_step_adamw");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_opt_step_adamw(g->ctx, p->t, o[0]->t, o[1]->t, o[2]->t,
	    h->t), p->rank);
}

/*** One SGD step on a trainable leaf, in place.
 * hparams holds lr and weight decay. The result is a view of the leaf;
 * compute it, or pass it to `graph:expand`, to do the step.
 * @tparam tensor p trainable f32 leaf
 * @tparam tensor grad f32 gradient with the shape of p
 * @tparam tensor hparams f32 (2)
 * @treturn tensor view node of p
 * @raise on bad types or shapes, or if p is not trainable
 * @function graph:opt_step_sgd
 */
static int
op_opt_step_sgd(lua_State *L)
{
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *o[1], *p = step_operands(L, g, 3, o, "opt_step_sgd");
	struct tensor *h = lgml_operand(L, g, 4);

	check_hparams(L, h, 2, "opt_step_sgd");
	lgml_reserve(L, g, 1);
	return lgml_push_node(L, g, ggml_opt_step_sgd(g->ctx, p->t, o[0]->t, h->t), p->rank);
}

/*** Fill an f32 leaf of (rows, classes) with one-hot rows.
 * Row i gets 1 at column ids[i] and 0 elsewhere. An id of -1 gives a
 * row of zeros. Ids count from 0.
 * @tparam {int,...} ids one id for each row
 * @raise unless the tensor is a contiguous f32 leaf of rank 1 or 2, or
 * on a bad id
 * @function tensor:fill_onehot
 */
static int
tensor_fill_onehot(lua_State *L)
{
	struct tensor *t = lgml_check_tensor(L, 1);
	const struct ggml_tensor *gt = t->t;
	int64_t i, rows = gt->ne[1], cols = gt->ne[0];
	lua_Integer id;
	float *p;

	if (t->s == NULL || gt->buffer == NULL || gt->type != GGML_TYPE_F32 ||
	    !ggml_is_contiguous(gt) || gt->ne[2] != 1 || gt->ne[3] != 1)
		return luaL_error(L, "fill_onehot: need a contiguous f32 leaf of rank 1 or 2");
	luaL_checktype(L, 2, LUA_TTABLE);
	luaL_argcheck(L, luaL_len(L, 2) == rows, 2, "need one id for each row");
	p = calloc((size_t)(rows * cols), sizeof(*p));
	if (p == NULL)
		return luaL_error(L, "out of memory");
	for (i = 0; i < rows; i++) {
		lua_geti(L, 2, i + 1);
		id = lua_tointeger(L, -1);
		lua_pop(L, 1);
		if (id < -1 || id >= cols) {
			free(p);
			return luaL_error(L, "fill_onehot: id %d out of range", (int)id);
		}
		if (id >= 0)
			p[i * cols + id] = 1.0f;
	}
	ggml_backend_tensor_set((struct ggml_tensor *)gt, p, 0, ggml_nbytes(gt));
	free(p);
	return 0;
}

static const luaL_Reg tensor_fns[] = {
	{ "requires_grad", tensor_requires_grad },
	{ "grad", tensor_grad },
	{ "is_param", tensor_is_param },
	{ "backward", tensor_backward },
	{ "fill_onehot", tensor_fill_onehot },
	{ NULL, NULL }
};

static const luaL_Reg ops[] = {
	{ "cross_entropy_loss", op_cross_entropy_loss },
	{ "sum", op_sum },
	{ "out_prod", op_out_prod },
	{ "opt_step_adamw", op_opt_step_adamw },
	{ "opt_step_sgd", op_opt_step_sgd },
	{ NULL, NULL }
};

void
lgml_open_train(lua_State *L)
{
	lgml_graph_ops(L, ops);
	lgml_tensor_methods(L, tensor_fns);
}
