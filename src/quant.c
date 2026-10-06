/*
 * Quantize and dequantize rows on the host with the ggml reference
 * quantizers. For model conversion and its checks.
 */

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <ggml.h>

#include <lua.h>
#include <lauxlib.h>

#include "lgml.h"

#define QUANT_MAX_THREADS	64
#define QUANT_STEP_ROWS		64	/* rows per f32 staging step */

struct qjob {
	enum ggml_type	 type;		/* output type */
	enum ggml_type	 from;		/* input type */
	const char	*src;
	char		*dst;
	int64_t		 n;		/* elements per row */
	int64_t		 row0;
	int64_t		 rows;
	int		 failed;
	pthread_t	 tid;
};

static enum ggml_type
type_arg(lua_State *L, int idx)
{
	const char *name = luaL_checkstring(L, idx);
	const char *tn;
	int i;

	for (i = 0; i < GGML_TYPE_COUNT; i++) {
		tn = ggml_type_name((enum ggml_type)i);
		if (tn != NULL && strcmp(tn, name) == 0)
			return (enum ggml_type)i;
	}
	luaL_error(L, "unknown type '%s'", name);
	return GGML_TYPE_COUNT;
}

/* The types ggml_quantize_chunk writes without an importance matrix. */
static int
can_quantize(enum ggml_type t)
{
	switch (t) {
	case GGML_TYPE_F32:
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16:
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q2_K:
	case GGML_TYPE_Q3_K:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ4_XS:
	case GGML_TYPE_TQ1_0:
	case GGML_TYPE_TQ2_0:
	case GGML_TYPE_MXFP4:
	case GGML_TYPE_NVFP4:
	case GGML_TYPE_Q1_0:
	case GGML_TYPE_Q2_0:
	case GGML_TYPE_IQ2_XXS:
	case GGML_TYPE_IQ2_XS:
	case GGML_TYPE_IQ2_S:
	case GGML_TYPE_IQ3_XXS:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ1_S:
	case GGML_TYPE_IQ1_M:
		return !ggml_quantize_requires_imatrix(t);
	default:
		return 0;
	}
}

static void *
quant_part(void *arg)
{
	struct qjob *j = arg;
	const struct ggml_type_traits *ft = ggml_get_type_traits(j->from);
	size_t in_rs = ggml_row_size(j->from, j->n), out_rs = ggml_row_size(j->type, j->n);
	float *buf = NULL;
	int64_t r, nr;

	if (j->from != GGML_TYPE_F32 &&
	    (buf = malloc((size_t)j->n * QUANT_STEP_ROWS * sizeof(float))) == NULL) {
		j->failed = 1;
		return NULL;
	}
	for (r = j->row0; r < j->row0 + j->rows; r += nr) {
		nr = j->row0 + j->rows - r < QUANT_STEP_ROWS ? j->row0 + j->rows - r : QUANT_STEP_ROWS;
		if (buf != NULL)
			ft->to_float(j->src + (size_t)r * in_rs, buf, nr * j->n);
		ggml_quantize_chunk(j->type, buf != NULL ? buf :
		    (const float *)(const void *)(j->src + (size_t)r * in_rs),
		    j->dst + (size_t)r * out_rs, 0, nr, j->n, NULL);
	}
	free(buf);
	return NULL;
}

/*** Quantize rows with the ggml reference quantizer.
 * Types that need an importance matrix do not work.
 * @tparam string type output type, such as "q8_0", "q4_K" or "bf16"
 * @tparam string data the input values as raw bytes
 * @tparam[opt] int n_per_row values in each row; the default is all values
 * @tparam[opt] table opts `from`: input type "f32" (the default), "f16" or
 * "bf16"; `threads`: worker threads, 1 by default
 * @treturn string quantized bytes
 * @raise if the type is not supported, or `n_per_row` does not divide the
 * values or is not a multiple of the block size
 * @function quantize
 */
static int
l_quantize(lua_State *L)
{
	enum ggml_type type = type_arg(L, 1), from = GGML_TYPE_F32;
	size_t len, in_rs;
	const char *src = luaL_checklstring(L, 2, &len);
	lua_Integer n, nt = 1;
	struct qjob jobs[QUANT_MAX_THREADS];
	luaL_Buffer b;
	int64_t rows, per;
	char *dst;
	int i, failed = 0;

	if (!lua_isnoneornil(L, 4)) {
		luaL_checktype(L, 4, LUA_TTABLE);
		if (lua_getfield(L, 4, "from") != LUA_TNIL)
			from = type_arg(L, -1);
		if (lua_getfield(L, 4, "threads") != LUA_TNIL)
			nt = luaL_checkinteger(L, -1);
		lua_pop(L, 2);
	}
	if (!can_quantize(type))
		return luaL_error(L, "quantize: unsupported type '%s'", lua_tostring(L, 1));
	luaL_argcheck(L, from == GGML_TYPE_F32 || from == GGML_TYPE_F16 ||
	    from == GGML_TYPE_BF16, 4, "from must be f32, f16 or bf16");
	luaL_argcheck(L, len % ggml_type_size(from) == 0 && len > 0, 2, "need whole values");
	n = luaL_optinteger(L, 3, (lua_Integer)(len / ggml_type_size(from)));
	luaL_argcheck(L, n > 0 && n % ggml_blck_size(type) == 0 &&
	    len % ggml_row_size(from, n) == 0, 3,
	    "n_per_row must divide the values and be a multiple of the block size");
	in_rs = ggml_row_size(from, n);
	nt = nt < 1 ? 1 : nt > QUANT_MAX_THREADS ? QUANT_MAX_THREADS : nt;
	rows = (int64_t)(len / in_rs);
	if (rows < nt * QUANT_STEP_ROWS)
		nt = rows / QUANT_STEP_ROWS > 0 ? rows / QUANT_STEP_ROWS : 1;
	dst = luaL_buffinitsize(L, &b, (size_t)rows * ggml_row_size(type, n));
	ggml_quantize_init(type);
	per = (rows + nt - 1) / nt;
	/* Job 0 runs on this thread after the others start. */
	for (i = nt - 1; i >= 0; i--) {
		jobs[i] = (struct qjob){ type, from, src, dst, n, i * per, 0, 0, pthread_self() };
		jobs[i].rows = rows - jobs[i].row0 < per ? rows - jobs[i].row0 : per;
		if (jobs[i].rows <= 0) {
			jobs[i].rows = 0;
			continue;
		}
		if (i == 0 || pthread_create(&jobs[i].tid, NULL, quant_part, &jobs[i]) != 0) {
			jobs[i].tid = pthread_self();
			quant_part(&jobs[i]);
		}
	}
	for (i = 0; i < nt; i++) {
		if (jobs[i].rows > 0 && !pthread_equal(jobs[i].tid, pthread_self()))
			pthread_join(jobs[i].tid, NULL);
		failed |= jobs[i].failed;
	}
	if (failed)
		return luaL_error(L, "quantize: out of memory");
	luaL_pushresultsize(&b, (size_t)rows * ggml_row_size(type, n));
	return 1;
}

/*** Dequantize rows to f32.
 * @tparam string type input type, such as "q8_0", "q4_K" or "bf16"
 * @tparam string data the input rows
 * @tparam int n elements per row
 * @treturn string the rows as f32
 * @raise on a type without a converter or a size that is not whole rows
 * @function dequantize
 */
static int
l_dequantize(lua_State *L)
{
	enum ggml_type type = type_arg(L, 1);
	const struct ggml_type_traits *tt = ggml_get_type_traits(type);
	size_t len, rs;
	const char *src = luaL_checklstring(L, 2, &len);
	lua_Integer n = luaL_checkinteger(L, 3);
	luaL_Buffer b;
	int64_t rows;
	float *dst;

	luaL_argcheck(L, tt->to_float != NULL || type == GGML_TYPE_F32, 1, "no converter");
	luaL_argcheck(L, n > 0 && n % ggml_blck_size(type) == 0, 3, "not a whole block");
	rs = ggml_row_size(type, n);
	luaL_argcheck(L, len % rs == 0, 2, "not whole rows");
	rows = (int64_t)(len / rs);
	dst = (float *)(void *)luaL_buffinitsize(L, &b, (size_t)(rows * n) * sizeof(float));
	if (type == GGML_TYPE_F32)
		memcpy(dst, src, len);
	else
		tt->to_float(src, dst, rows * n);
	luaL_pushresultsize(&b, (size_t)(rows * n) * sizeof(float));
	return 1;
}

/*** Compare two f32 arrays.
 * @tparam string ref the reference values
 * @tparam string got values of the same length
 * @treturn number root mean square of the difference
 * @treturn number largest absolute difference
 * @treturn number root mean square of ref
 * @raise if the lengths differ or are not whole f32 values
 * @function f32_error
 */
static int
l_f32_error(lua_State *L)
{
	size_t la, lb, i, n;
	const char *a = luaL_checklstring(L, 1, &la);
	const char *b = luaL_checklstring(L, 2, &lb);
	double se = 0, sr = 0, mx = 0, d;
	float x, y;

	luaL_argcheck(L, la == lb && la % sizeof(float) == 0, 2, "lengths differ");
	n = la / sizeof(float);
	for (i = 0; i < n; i++) {
		memcpy(&x, a + i * sizeof(float), sizeof(x));
		memcpy(&y, b + i * sizeof(float), sizeof(y));
		d = fabs((double)x - (double)y);
		se += d * d;
		sr += (double)x * x;
		mx = d > mx || isnan(d) ? d : mx;
	}
	lua_pushnumber(L, n > 0 ? sqrt(se / (double)n) : 0);
	lua_pushnumber(L, mx);
	lua_pushnumber(L, n > 0 ? sqrt(sr / (double)n) : 0);
	return 3;
}

void
lgml_open_quant(lua_State *L)
{
	lua_pushcfunction(L, l_quantize);
	lua_setfield(L, -2, "quantize");
	lua_pushcfunction(L, l_dequantize);
	lua_setfield(L, -2, "dequantize");
	lua_pushcfunction(L, l_f32_error);
	lua_setfield(L, -2, "f32_error");
}
