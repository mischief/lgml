/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Lua binding for image.c. An image object holds 8-bit pixels in C
 * memory. Pixels never become Lua values; image:tensor writes them
 * straight into a tensor.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <ggml.h>
#include <ggml-backend.h>

#include <lua.h>
#include <lauxlib.h>

#include "image.h"
#include "lgml.h"

#define IMAGE_MT	"lgml_image"

static struct image *
check_image(lua_State *L, int idx)
{
	struct image *im = luaL_checkudata(L, idx, IMAGE_MT);

	if (im->p == NULL)
		luaL_error(L, "image closed");
	return im;
}

/* Push an empty image object. Fill it before any call that can fail. */
static struct image *
new_image(lua_State *L)
{
	struct image *im = lua_newuserdatauv(L, sizeof(*im), 0);

	memset(im, 0, sizeof(*im));
	luaL_setmetatable(L, IMAGE_MT);
	return im;
}

/*** Decode a PNG or JPEG file.
 * The first bytes pick the format. Samples become 8 bits. PNG images
 * keep their gray, alpha and color channels; palette images become RGB,
 * or RGBA with transparency. JPEG images become gray or RGB; CMYK
 * becomes RGB. The EXIF orientation is not applied. Sides are at most
 * 32768 and the pixel count at most 2^26.
 * @tparam string bytes contents of the file, at most 256 MB
 * @treturn[1] image the image
 * @return[2] nil
 * @treturn[2] string error message
 * @function image
 */
static int
l_image(lua_State *L)
{
	size_t len;
	const char *s = luaL_checklstring(L, 1, &len);
	struct image tmp, *im;
	char err[IMAGE_ERRLEN];

	im = new_image(L);
	if (image_decode(s, len, &tmp, err) != 0) {
		luaL_pushfail(L);
		lua_pushstring(L, err);
		return 2;
	}
	*im = tmp;
	return 1;
}

/*** Make an image from a tensor.
 * Values in [-1, 1] map to 0 to 255, clamped and rounded. C is 1 (gray),
 * 2 (gray and alpha), 3 (RGB) or 4 (RGBA).
 * @tparam tensor t computed, contiguous f32 tensor of shape (N, C, H, W)
 * @tparam[opt=1] int n which image of the batch, 1 to N
 * @treturn image the image
 * @raise on a bad tensor, shape or n
 * @function image_from
 */
static int
l_image_from(lua_State *L)
{
	struct tensor *t = lgml_check_tensor(L, 1);
	const struct ggml_tensor *gt = t->t;
	lua_Integer n = luaL_optinteger(L, 2, 1);
	struct image *im;
	size_t plane, i, k, c;
	float *x, v;

	if (gt->buffer == NULL || gt->type != GGML_TYPE_F32 || !ggml_is_contiguous(gt))
		return luaL_error(L, "image_from needs a computed, contiguous f32 tensor");
	if (gt->ne[2] < 1 || gt->ne[2] > 4)
		return luaL_error(L, "image_from needs (N, C, H, W) with C 1 to 4");
	luaL_argcheck(L, n >= 1 && n <= gt->ne[3], 2, "n out of range");
	im = new_image(L);
	if (image_alloc(im, (int)(gt->ne[0] > IMAGE_MAX_SIDE ? 0 : gt->ne[0]),
	    (int)(gt->ne[1] > IMAGE_MAX_SIDE ? 0 : gt->ne[1]), (int)gt->ne[2]) != 0)
		return luaL_error(L, "image_from: image too large or out of memory");
	c = (size_t)im->c;
	plane = (size_t)im->w * (size_t)im->h;
	if ((x = malloc(plane * c * sizeof(*x))) == NULL)
		return luaL_error(L, "out of memory");
	ggml_backend_tensor_get(gt, x, (size_t)(n - 1) * gt->nb[3], plane * c * sizeof(*x));
	for (k = 0; k < c; k++)
		for (i = 0; i < plane; i++) {
			v = (x[k * plane + i] + 1.0f) * 127.5f;
			v = isnan(v) ? 0 : v < 0 ? 0 : v > 255 ? 255 : v;
			im->p[i * c + k] = (unsigned char)lrintf(v);
		}
	free(x);
	return 1;
}

/*** 8-bit pixels in C memory.
 * The fields width, height and channels are integers. channels is 1
 * (gray), 2 (gray and alpha), 3 (RGB) or 4 (RGBA).
 * @type image
 */

/*** Free the pixels.
 * The `__gc` and `__close` metamethods call this function.
 * @function image:close
 */
static int
image_close(lua_State *L)
{
	image_free(luaL_checkudata(L, 1, IMAGE_MT));
	return 0;
}

/*** Make an f32 tensor of shape (1, 3, H, W) with values in [-1, 1].
 * Gray becomes three equal channels. Alpha blends over a gray
 * background level.
 * @tparam backend be the backend for the tensor
 * @tparam[opt=1] number background background level, 0 (black) to 1 (white)
 * @treturn tensor a new leaf
 * @raise on a bad backend or background, or out of memory
 * @function image:tensor
 */
static int
image_tensor(lua_State *L)
{
	struct image *im = check_image(L, 1);
	lua_Number bg = luaL_optnumber(L, 3, 1);
	size_t plane = (size_t)im->w * (size_t)im->h, i, c = (size_t)im->c;
	struct tensor *t;
	const unsigned char *p;
	float *x, a;
	int k;

	lgml_check_backend(L, 2);
	luaL_argcheck(L, bg >= 0 && bg <= 1, 3, "background out of range");
	lua_getfield(L, 2, "tensor");
	lua_pushvalue(L, 2);
	lua_pushstring(L, "f32");
	lua_createtable(L, 4, 0);
	lua_pushinteger(L, 1);
	lua_rawseti(L, -2, 1);
	lua_pushinteger(L, 3);
	lua_rawseti(L, -2, 2);
	lua_pushinteger(L, im->h);
	lua_rawseti(L, -2, 3);
	lua_pushinteger(L, im->w);
	lua_rawseti(L, -2, 4);
	lua_call(L, 3, 1);
	t = lgml_check_tensor(L, -1);
	if ((x = malloc(plane * 3 * sizeof(*x))) == NULL)
		return luaL_error(L, "out of memory");
	for (i = 0; i < plane; i++) {
		p = im->p + i * c;
		a = c == 2 || c == 4 ? p[c - 1] / 255.0f : 1.0f;
		for (k = 0; k < 3; k++)
			x[(size_t)k * plane + i] = (c < 3 ? p[0] : p[k]) / 255.0f * a +
			    (float)bg * (1 - a);
	}
	for (i = 0; i < plane * 3; i++)
		x[i] = x[i] * 2 - 1;
	ggml_backend_tensor_set(t->t, x, 0, plane * 3 * sizeof(*x));
	free(x);
	return 1;
}

/*** Encode as an 8-bit PNG file with the channels of the image.
 * @treturn string PNG bytes
 * @raise if memory runs out
 * @function image:png
 */
static int
image_png(lua_State *L)
{
	struct image *im = check_image(L, 1);
	size_t len;
	void *buf;

	if ((buf = png_encode(im, &len)) == NULL)
		return luaL_error(L, "png: out of memory");
	lua_pushlstring(L, buf, len);
	free(buf);
	return 1;
}

/*** Encode as a baseline JPEG file.
 * One or two channels give a gray JPEG; three or four give 4:2:0 color.
 * Alpha is dropped.
 * @tparam[opt=90] int quality 1 to 100
 * @treturn string JPEG bytes
 * @raise if quality is out of range or memory runs out
 * @function image:jpeg
 */
static int
image_jpeg(lua_State *L)
{
	struct image *im = check_image(L, 1);
	lua_Integer q = luaL_optinteger(L, 2, 90);
	size_t len;
	void *buf;

	luaL_argcheck(L, q >= 1 && q <= 100, 2, "quality out of range");
	if ((buf = jpeg_encode(im, (int)q, &len)) == NULL)
		return luaL_error(L, "jpeg: out of memory");
	lua_pushlstring(L, buf, len);
	free(buf);
	return 1;
}

/*** Resize to a new width and height.
 * The filter is a triangle, widened to average over the area when the
 * image shrinks.
 * @tparam int w new width, 1 to 32768
 * @tparam int h new height, 1 to 32768
 * @treturn image new image
 * @raise if a size is out of range or memory runs out
 * @function image:resize
 */
static int
image_resize_l(lua_State *L)
{
	struct image *im = check_image(L, 1);
	lua_Integer w = luaL_checkinteger(L, 2), h = luaL_checkinteger(L, 3);
	struct image tmp, *r;

	luaL_argcheck(L, w >= 1 && w <= IMAGE_MAX_SIDE, 2, "width out of range");
	luaL_argcheck(L, h >= 1 && h <= IMAGE_MAX_SIDE, 3, "height out of range");
	r = new_image(L);
	if (image_resize(im, (int)w, (int)h, &tmp) != 0)
		return luaL_error(L, "resize: image too large or out of memory");
	*r = tmp;
	return 1;
}

/*** Cut out a rectangle.
 * @tparam int x left column, from 0
 * @tparam int y top row, from 0
 * @tparam int w width, at least 1
 * @tparam int h height, at least 1
 * @treturn image new image
 * @raise if the rectangle is not inside the image
 * @function image:crop
 */
static int
image_crop(lua_State *L)
{
	struct image *im = check_image(L, 1);
	lua_Integer x = luaL_checkinteger(L, 2), y = luaL_checkinteger(L, 3);
	lua_Integer w = luaL_checkinteger(L, 4), h = luaL_checkinteger(L, 5);
	size_t c = (size_t)im->c, i;
	struct image *r;

	if (x < 0 || y < 0 || w < 1 || h < 1 || x > im->w - w || y > im->h - h)
		return luaL_error(L, "crop: rectangle not inside the image");
	r = new_image(L);
	if (image_alloc(r, (int)w, (int)h, im->c) != 0)
		return luaL_error(L, "out of memory");
	for (i = 0; i < (size_t)h; i++)
		memcpy(r->p + i * (size_t)w * c, im->p + (((size_t)y + i) * (size_t)im->w +
		    (size_t)x) * c, (size_t)w * c);
	return 1;
}

static int
image_index(lua_State *L)
{
	struct image *im = luaL_checkudata(L, 1, IMAGE_MT);
	const char *k = lua_type(L, 2) == LUA_TSTRING ? lua_tostring(L, 2) : NULL;

	if (k != NULL && strcmp(k, "width") == 0)
		lua_pushinteger(L, im->w);
	else if (k != NULL && strcmp(k, "height") == 0)
		lua_pushinteger(L, im->h);
	else if (k != NULL && strcmp(k, "channels") == 0)
		lua_pushinteger(L, im->c);
	else
		lua_rawget(L, lua_upvalueindex(1));
	return 1;
}

static const luaL_Reg image_meta[] = {
	{ "__gc", image_close },
	{ "__close", image_close },
	{ NULL, NULL }
};

static const luaL_Reg image_methods[] = {
	{ "close", image_close },
	{ "tensor", image_tensor },
	{ "png", image_png },
	{ "jpeg", image_jpeg },
	{ "resize", image_resize_l },
	{ "crop", image_crop },
	{ NULL, NULL }
};

void
lgml_open_image(lua_State *L)
{
	luaL_newmetatable(L, IMAGE_MT);
	luaL_setfuncs(L, image_meta, 0);
	lua_newtable(L);
	luaL_setfuncs(L, image_methods, 0);
	lua_pushcclosure(L, image_index, 1);
	lua_setfield(L, -2, "__index");
	lua_pop(L, 1);
	lua_pushcfunction(L, l_image);
	lua_setfield(L, -2, "image");
	lua_pushcfunction(L, l_image_from);
	lua_setfield(L, -2, "image_from");
}
