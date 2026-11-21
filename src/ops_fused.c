/*** Tensor graphs on ggml backends.
 * @module lgml
 */

/*
 * Fused graph ops: one op for a chain of small ones. Each is a ggml
 * custom op with a C function for the CPU and, when the ggml Vulkan
 * backend takes custom shaders, a SPIR-V shader for the device.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <ggml-backend.h>
#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

#ifdef LGML_HAVE_SPV
struct ggml_vk_custom_op {
	const void	*spv;
	size_t		 spv_size;
	uint32_t	 push_size;
	uint32_t	 wg_denoms[3];
	void		(*setup)(const struct ggml_tensor *, void *, uint32_t[3]);
};

typedef void (*vk_register_fn)(ggml_custom_op_t, const struct ggml_vk_custom_op *);

static const uint32_t hc_add_spv[] =
#include "hc_add.spv.h"
;
static const uint32_t hc_mean_spv[] =
#include "hc_mean.spv.h"
;
static const uint32_t rel_mask_spv[] =
#include "rel_mask.spv.h"
;
#endif

/* True when the Vulkan backend runs the fused ops on the device. */
static int vk_fused;

static int register_vk(void);

/*
 * hc_add: d (T, HC, E) = r + o (T, E) * 2 * sigmoid(w (T, HC)).
 * Rows of d split across the threads.
 */
static void
hc_add_op(struct ggml_tensor *d, int ith, int nth, void *ud)
{
	const struct ggml_tensor *r = d->src[0], *o = d->src[1], *w = d->src[2];
	int64_t E = d->ne[0], HC = d->ne[1], rows = d->ne[1] * d->ne[2], row, e;
	int64_t per = (rows + nth - 1) / nth, r0 = per * ith;
	int64_t r1 = r0 + per < rows ? r0 + per : rows;

	(void)ud;
	for (row = r0; row < r1; row++) {
		const float *rr = (const float *)r->data + row * E;
		const float *oo = (const float *)o->data + (row / HC) * E;
		float *dd = (float *)d->data + row * E;
		float g = 2.0f / (1.0f + expf(-((const float *)w->data)[row]));

		for (e = 0; e < E; e++)
			dd[e] = rr[e] + oo[e] * g;
	}
}

#ifdef LGML_HAVE_SPV
static void
hc_add_setup(const struct ggml_tensor *d, void *push, uint32_t el[3])
{
	uint32_t *p = push;

	p[0] = (uint32_t)d->ne[0];
	p[1] = (uint32_t)d->ne[1];
	p[2] = (uint32_t)(d->ne[1] * d->ne[2]);
	el[0] = p[0];
	el[1] = p[2];
	el[2] = 1;
}
#endif

/*** Add a block output to each hyper-connection stream.
 * The result is r + o * 2 * sigmoid(w), with o broadcast over the
 * streams and w over the embedding. All f32 and contiguous.
 * @tparam graph g the graph
 * @tparam tensor r (T, HC, E) the streams
 * @tparam tensor o (T, E) the block output
 * @tparam tensor w (T, HC) the inject logits
 * @treturn tensor (T, HC, E)
 * @function graph:hc_add
 */
static int
g_hc_add(lua_State *L)
{
	register_vk();
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *r = lgml_operand(L, g, 2);
	struct tensor *o = lgml_operand(L, g, 3);
	struct tensor *w = lgml_operand(L, g, 4);
	struct ggml_tensor *args[3], *t;

	if (r->t->type != GGML_TYPE_F32 || o->t->type != GGML_TYPE_F32 || w->t->type != GGML_TYPE_F32 ||
	    !ggml_is_contiguous(r->t) || !ggml_is_contiguous(o->t) || !ggml_is_contiguous(w->t) ||
	    o->t->ne[0] != r->t->ne[0] || ggml_nelements(o->t) * r->t->ne[1] != ggml_nelements(r->t) ||
	    ggml_nelements(w->t) * r->t->ne[0] != ggml_nelements(r->t))
		return luaL_error(L, "hc_add: shapes do not match");
	args[0] = r->t;
	args[1] = o->t;
	args[2] = w->t;
	lgml_reserve(L, g, 1);
	t = ggml_custom_4d(g->ctx, GGML_TYPE_F32, r->t->ne[0], r->t->ne[1], r->t->ne[2], r->t->ne[3],
	    args, 3, hc_add_op, GGML_N_TASKS_MAX, NULL);
	return lgml_push_node(L, g, t, 3);
}

/*
 * hc_mean: d (T, E) = sum over h of x (T, HC, E) * sigmoid(g (T, HC, E)).
 * Elements of d split across the threads.
 */
static void
hc_mean_op(struct ggml_tensor *d, int ith, int nth, void *ud)
{
	const struct ggml_tensor *x = d->src[0], *g = d->src[1];
	int64_t E = d->ne[0], T = d->ne[1], HC = x->ne[1], n = E * T, i, h;
	int64_t per = (n + nth - 1) / nth, i0 = per * ith, i1 = i0 + per < n ? i0 + per : n;
	const float *xx = x->data, *gg = g->data;

	(void)ud;
	for (i = i0; i < i1; i++) {
		int64_t t = i / E, e = i % E;
		float s = 0;

		for (h = 0; h < HC; h++) {
			int64_t k = (t * HC + h) * E + e;

			s += xx[k] / (1.0f + expf(-gg[k]));
		}
		((float *)d->data)[i] = s;
	}
}

#ifdef LGML_HAVE_SPV
static void
hc_mean_setup(const struct ggml_tensor *d, void *push, uint32_t el[3])
{
	uint32_t *p = push;

	p[0] = (uint32_t)d->ne[0];
	p[1] = (uint32_t)d->src[0]->ne[1];
	p[2] = (uint32_t)d->ne[1];
	el[0] = p[0];
	el[1] = p[2];
	el[2] = 1;
}
#endif

/*** Mix the hyper-connection streams.
 * The result is the sum over the streams of x * sigmoid(gate). All f32
 * and contiguous.
 * @tparam graph g the graph
 * @tparam tensor x (T, HC, E) the normed streams
 * @tparam tensor gate (T, HC, E) the gate logits
 * @treturn tensor (T, E)
 * @function graph:hc_mean
 */
static int
g_hc_mean(lua_State *L)
{
	register_vk();
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *x = lgml_operand(L, g, 2);
	struct tensor *gate = lgml_operand(L, g, 3);
	struct ggml_tensor *args[2], *t;

	if (x->t->type != GGML_TYPE_F32 || gate->t->type != GGML_TYPE_F32 ||
	    !ggml_is_contiguous(x->t) || !ggml_is_contiguous(gate->t) || !ggml_are_same_shape(x->t, gate->t))
		return luaL_error(L, "hc_mean: shapes do not match");
	args[0] = x->t;
	args[1] = gate->t;
	lgml_reserve(L, g, 1);
	t = ggml_custom_4d(g->ctx, GGML_TYPE_F32, x->t->ne[0], x->t->ne[2], 1, 1, args, 2, hc_mean_op,
	    GGML_N_TASKS_MAX, NULL);
	return lgml_push_node(L, g, t, 2);
}

/*
 * rel_mask: d (H, M, S) f16 = mask (M, S) + tab (M, H, R) at the distance
 * clamp(qpos[i] - kpos[j], 0, R - 1). Rows of d split across the threads.
 */
static void
rel_mask_op(struct ggml_tensor *d, int ith, int nth, void *ud)
{
	const struct ggml_tensor *tab = d->src[0], *q = d->src[1], *k = d->src[2], *m = d->src[3];
	int64_t S = d->ne[0], M = d->ne[1], H = d->ne[2], R = tab->ne[0], rows = M * H, row, j;
	int64_t per = (rows + nth - 1) / nth, r0 = per * ith, r1 = r0 + per < rows ? r0 + per : rows;
	const float *qq = q->data, *kk = k->data;

	(void)ud;
	for (row = r0; row < r1; row++) {
		int64_t h = row / M, i = row % M;
		const float *t = (const float *)tab->data + (i * H + h) * R;
		const ggml_fp16_t *mm = (const ggml_fp16_t *)m->data + i * S;
		ggml_fp16_t *dd = (ggml_fp16_t *)d->data + row * S;

		for (j = 0; j < S; j++) {
			float dist = qq[i] - kk[j];

			dist = dist < 0 ? 0 : dist > (float)(R - 1) ? (float)(R - 1) : dist;
			dd[j] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(mm[j]) + t[(int64_t)dist]);
		}
	}
}

#ifdef LGML_HAVE_SPV
static void
rel_mask_setup(const struct ggml_tensor *d, void *push, uint32_t el[3])
{
	uint32_t *p = push;

	p[0] = (uint32_t)d->ne[0];
	p[1] = (uint32_t)d->ne[1];
	p[2] = (uint32_t)d->ne[2];
	p[3] = (uint32_t)d->src[0]->ne[0];
	el[0] = p[0];
	el[1] = p[1] * p[2];
	el[2] = 1;
}
#endif

/*** Build the per-head attention mask of a bias by distance.
 * Element (h, i, j) of the result is mask[i][j] + tab[i][h][d], with d the
 * distance qpos[i] - kpos[j] clamped to [0, R - 1].
 * @tparam graph g the graph
 * @tparam tensor tab f32 (M, H, R) contiguous bias by distance
 * @tparam tensor qpos f32 (M) query positions
 * @tparam tensor kpos f32 (S) cell positions
 * @tparam tensor mask f16 (M, S) contiguous mask
 * @treturn tensor f16 (H, M, S)
 * @function graph:rel_mask
 */
static int
g_rel_mask(lua_State *L)
{
	register_vk();
	struct graph *g = lgml_check_graph(L, 1);
	struct tensor *tab = lgml_operand(L, g, 2);
	struct tensor *q = lgml_operand(L, g, 3);
	struct tensor *k = lgml_operand(L, g, 4);
	struct tensor *m = lgml_operand(L, g, 5);
	struct ggml_tensor *args[4], *t;
	int64_t M = m->t->ne[1], S = m->t->ne[0], H = tab->t->ne[1];

	if (tab->t->type != GGML_TYPE_F32 || q->t->type != GGML_TYPE_F32 || k->t->type != GGML_TYPE_F32 ||
	    m->t->type != GGML_TYPE_F16 || !ggml_is_contiguous(tab->t) || !ggml_is_contiguous(q->t) ||
	    !ggml_is_contiguous(k->t) || !ggml_is_contiguous(m->t) || tab->t->ne[2] != M ||
	    ggml_nelements(q->t) != M || ggml_nelements(k->t) != S || ggml_nrows(m->t) != M ||
	    M * H > 65535 || S > INT32_MAX)
		return luaL_error(L, "rel_mask: shapes do not match");
	args[0] = tab->t;
	args[1] = q->t;
	args[2] = k->t;
	args[3] = m->t;
	lgml_reserve(L, g, 1);
	t = ggml_custom_4d(g->ctx, GGML_TYPE_F16, S, M, H, 1, args, 4, rel_mask_op, GGML_N_TASKS_MAX, NULL);
	return lgml_push_node(L, g, t, 3);
}

static const luaL_Reg ops[] = {
	{ "hc_add", g_hc_add },
	{ "hc_mean", g_hc_mean },
	{ "rel_mask", g_rel_mask },
	{ NULL, NULL }
};

/*
 * Give the shaders to the Vulkan backend when it takes them, once. It
 * starts Vulkan, so it waits for the first use: a process that only
 * loads the module (a server, a fake worker) stays fast.
 */
static int
register_vk(void)
{
#ifdef LGML_HAVE_SPV
	static int done;
	ggml_backend_reg_t reg;
	vk_register_fn fn;
	struct ggml_vk_custom_op op;
	void *addr;

	if (done)
		return vk_fused;
	done = 1;
	reg = ggml_backend_reg_by_name("Vulkan");
	if (reg == NULL)
		return 0;
	addr = ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_register_custom");
	if (addr == NULL)
		return 0;
	memcpy(&fn, &addr, sizeof(fn));
	memset(&op, 0, sizeof(op));
	op.push_size = 3 * sizeof(uint32_t);
	op.wg_denoms[0] = 256;
	op.wg_denoms[1] = 1;
	op.wg_denoms[2] = 1;
	op.spv = hc_add_spv;
	op.spv_size = sizeof(hc_add_spv);
	op.setup = hc_add_setup;
	fn(hc_add_op, &op);
	op.spv = hc_mean_spv;
	op.spv_size = sizeof(hc_mean_spv);
	op.setup = hc_mean_setup;
	fn(hc_mean_op, &op);
	op.push_size = 4 * sizeof(uint32_t);
	op.spv = rel_mask_spv;
	op.spv_size = sizeof(rel_mask_spv);
	op.setup = rel_mask_setup;
	fn(rel_mask_op, &op);
	vk_fused = 1;
#endif
	return vk_fused;
}

/*
 * The Vulkan buffer type of pinned host memory that ops on the device read
 * in place, over PCIe; NULL when ggml lacks the lgml patch.
 */
ggml_backend_buffer_type_t
lgml_host_read_buft(void)
{
	ggml_backend_reg_t reg = ggml_backend_reg_by_name("Vulkan");
	ggml_backend_buffer_type_t (*fn)(void);
	void *addr = reg != NULL ?
	    ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_host_read_buffer_type") : NULL;

	if (addr == NULL)
		return NULL;
	memcpy(&fn, &addr, sizeof(fn));
	return fn();
}

/*** Whether Vulkan devices can read host memory in place: tensors from
 * `backend:tensor` with `host_read`.
 * @treturn boolean
 * @function vk_host_read
 */
static int
l_vk_host_read(lua_State *L)
{
	lua_pushboolean(L, lgml_host_read_buft() != NULL);
	return 1;
}

/*** Whether the Vulkan backend runs the fused ops on the device; without
 * it a fused op on a GPU graph runs on the CPU. The first call gives
 * Vulkan the shaders.
 * @treturn boolean
 * @function vk_fused
 */
static int
l_vk_fused(lua_State *L)
{
	lua_pushboolean(L, register_vk());
	return 1;
}

void
lgml_open_ops_fused(lua_State *L)
{
	lgml_graph_ops(L, ops);
	lua_pushcfunction(L, l_vk_host_read);
	lua_setfield(L, -2, "vk_host_read");
	lua_pushcfunction(L, l_vk_fused);
	lua_setfield(L, -2, "vk_fused");
}
