/*
 * Images in C. libpng and libjpeg-turbo decode and encode. The input
 * is untrusted: the header sizes are checked against our limits before
 * the libraries allocate anything, and the libraries check the rest.
 */

#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>	/* jpeglib.h needs FILE */
#include <stdlib.h>
#include <string.h>

#include <jpeglib.h>
#include <jerror.h>
#include <png.h>

#include "image.h"

void
image_free(struct image *im)
{
	free(im->p);
	im->p = NULL;
	im->w = im->h = im->c = 0;
}

/* Allocate w x h pixels of c channels. Returns -1 on bad sizes or no memory. */
int
image_alloc(struct image *im, int w, int h, int c)
{
	if (w < 1 || h < 1 || w > IMAGE_MAX_SIDE || h > IMAGE_MAX_SIDE || c < 1 || c > 4 ||
	    (size_t)w * (size_t)h > IMAGE_MAX_PIXELS)
		return -1;
	im->p = malloc((size_t)w * (size_t)h * (size_t)c);
	if (im->p == NULL)
		return -1;
	im->w = w;
	im->h = h;
	im->c = c;
	return 0;
}

static int
size_ok(uint32_t w, uint32_t h)
{
	return w >= 1 && h >= 1 && w <= IMAGE_MAX_SIDE && h <= IMAGE_MAX_SIDE &&
	    (size_t)w * h <= IMAGE_MAX_PIXELS;
}

/* PNG */

static const unsigned char png_sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };

static uint32_t
be32(const unsigned char *p)
{
	return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}

static png_uint_32
png_format(int c)
{
	static const png_uint_32 fmt[5] = { 0, PNG_FORMAT_GRAY, PNG_FORMAT_GA, PNG_FORMAT_RGB,
	    PNG_FORMAT_RGBA };

	return fmt[c];
}

/* Copy an error message into err, which holds IMAGE_ERRLEN bytes. */
static int
fail(char *err, const char *msg)
{
	snprintf(err, IMAGE_ERRLEN, "%s", msg != NULL && msg[0] != '\0' ? msg : "bad image");
	return -1;
}

struct membuf {
	const unsigned char	*p;
	size_t			 n;
	size_t			 off;
};

static void
png_mem_read(png_structp png, png_bytep out, size_t len)
{
	struct membuf *m = png_get_io_ptr(png);

	if (len > m->n - m->off)
		png_error(png, "PNG truncated");
	memcpy(out, m->p + m->off, len);
	m->off += len;
}

static void
png_on_error(png_structp png, png_const_charp msg)
{
	fail(png_get_error_ptr(png), msg);
	png_longjmp(png, 1);
}

static void
png_on_warning(png_structp png, png_const_charp msg)
{
	(void)png;
	(void)msg;
}

/*
 * Decode with the full libpng API: the simplified API treats 16-bit
 * samples as linear light, and its 8-bit output would not match the
 * file. Here 16-bit samples round to 8 bits.
 */
static int
png_read(const unsigned char *p, size_t n, struct image *im, char *err)
{
	struct membuf m = { p, n, 0 };
	png_structp png;
	png_infop info;
	unsigned char **volatile rows = NULL;
	png_uint_32 w, h, y;
	int depth, ctype;

	/* The first chunk must be IHDR; check its size before libpng does anything. */
	if (n < 33 || be32(p + 8) != 13 || memcmp(p + 12, "IHDR", 4) != 0)
		return fail(err, "PNG does not start with IHDR");
	if (!size_ok(be32(p + 16), be32(p + 20)))
		return fail(err, "PNG too large or empty");
	png = png_create_read_struct(PNG_LIBPNG_VER_STRING, err, png_on_error, png_on_warning);
	info = png != NULL ? png_create_info_struct(png) : NULL;
	if (info == NULL) {
		png_destroy_read_struct(&png, NULL, NULL);
		return fail(err, "out of memory");
	}
	if (setjmp(png_jmpbuf(png))) {
		free(rows);
		image_free(im);
		png_destroy_read_struct(&png, &info, NULL);
		return -1;
	}
	png_set_read_fn(png, &m, png_mem_read);
	png_set_user_limits(png, IMAGE_MAX_SIDE, IMAGE_MAX_SIDE);
	png_set_chunk_malloc_max(png, (png_alloc_size_t)8 << 20);
	png_read_info(png, info);
	png_get_IHDR(png, info, &w, &h, &depth, &ctype, NULL, NULL, NULL);
	if (!size_ok(w, h))
		png_error(png, "PNG too large or empty");
	if (ctype == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png);
	if (ctype == PNG_COLOR_TYPE_GRAY && depth < 8)
		png_set_expand_gray_1_2_4_to_8(png);
	if (png_get_valid(png, info, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png);
	if (depth == 16)
		png_set_scale_16(png);
	png_set_interlace_handling(png);
	png_read_update_info(png, info);
	if (image_alloc(im, (int)w, (int)h, png_get_channels(png, info)) != 0 ||
	    png_get_rowbytes(png, info) != (size_t)w * (size_t)im->c ||
	    (rows = malloc(h * sizeof(*rows))) == NULL)
		png_error(png, "out of memory");
	for (y = 0; y < h; y++)
		rows[y] = im->p + (size_t)y * w * (size_t)im->c;
	png_read_image(png, rows);
	free(rows);
	png_destroy_read_struct(&png, &info, NULL);
	return 0;
}

/* Encode an 8-bit PNG with the channels of im. Returns a malloc buffer or NULL. */
void *
png_encode(const struct image *im, size_t *outlen)
{
	png_image pi;
	png_alloc_size_t size = 0;
	void *buf;

	if (im->p == NULL || im->c < 1 || im->c > 4)
		return NULL;
	memset(&pi, 0, sizeof(pi));
	pi.version = PNG_IMAGE_VERSION;
	pi.width = (png_uint_32)im->w;
	pi.height = (png_uint_32)im->h;
	pi.format = png_format(im->c);
	if (!png_image_write_to_memory(&pi, NULL, &size, 0, im->p, 0, NULL))
		return NULL;
	if ((buf = malloc(size)) == NULL)
		return NULL;
	if (!png_image_write_to_memory(&pi, buf, &size, 0, im->p, 0, NULL)) {
		free(buf);
		return NULL;
	}
	*outlen = size;
	return buf;
}

/* JPEG */

struct jerr {
	struct jpeg_error_mgr	 pub;
	jmp_buf			 jb;
	char			*err;
};

static void
jpeg_on_error(j_common_ptr c)
{
	struct jerr *e = (struct jerr *)c->err;
	char msg[JMSG_LENGTH_MAX];

	(*c->err->format_message)(c, msg);
	fail(e->err, msg);
	longjmp(e->jb, 1);
}

static void
jpeg_fail(struct jerr *e, const char *msg)
{
	fail(e->err, msg);
	longjmp(e->jb, 1);
}

/*
 * Most warnings leave a usable image, as a web browser shows it. Image
 * data that ends early is an error: libjpeg would fill in the rest, so
 * a few bytes could make 2^26 pixels.
 */
static void
jpeg_on_message(j_common_ptr c, int level)
{
	int code = c->err->msg_code;

	if (level < 0 && (code == JWRN_JPEG_EOF || code == JWRN_HIT_MARKER))
		jpeg_fail((struct jerr *)c->err, "JPEG data ends early");
}

/* Bound the work that a progressive file with many scans can cause. */
static void
jpeg_on_progress(j_common_ptr c)
{
	struct jerr *e = (struct jerr *)c->err;

	if (((j_decompress_ptr)c)->input_scan_number > 500)
		jpeg_fail(e, "JPEG has too many scans");
}

/* CMYK to RGB. Adobe files store inverted values. */
static void
cmyk_to_rgb(const unsigned char *s, unsigned char *d, size_t n, int inverted)
{
	size_t i;
	int k, v, black;

	for (i = 0; i < n; i++, s += 4, d += 3) {
		black = inverted ? s[3] : 255 - s[3];
		for (k = 0; k < 3; k++) {
			v = inverted ? s[k] : 255 - s[k];
			d[k] = (unsigned char)((v * black + 127) / 255);
		}
	}
}

static int
jpeg_read(const unsigned char *p, size_t n, struct image *im, char *err)
{
	struct jpeg_decompress_struct d;
	struct jpeg_progress_mgr prog;
	struct jerr e;
	unsigned char *volatile cmyk = NULL;
	unsigned char *out;
	JSAMPROW row;
	size_t stride;
	int c;

	d.err = jpeg_std_error(&e.pub);
	e.pub.error_exit = jpeg_on_error;
	e.pub.emit_message = jpeg_on_message;
	e.err = err;
	if (setjmp(e.jb)) {
		free(cmyk);
		image_free(im);
		jpeg_destroy_decompress(&d);
		return -1;
	}
	jpeg_create_decompress(&d);
	prog.progress_monitor = jpeg_on_progress;
	d.progress = &prog;
	d.mem->max_memory_to_use = (long)1 << 28;
	jpeg_mem_src(&d, p, (unsigned long)n);
	jpeg_read_header(&d, TRUE);
	if (!size_ok(d.image_width, d.image_height))
		jpeg_fail(&e, "JPEG too large or empty");
	if (d.data_precision != 8)
		jpeg_fail(&e, "JPEG precision other than 8 bits not supported");
	if (d.jpeg_color_space == JCS_GRAYSCALE)
		d.out_color_space = JCS_GRAYSCALE;
	else if (d.jpeg_color_space == JCS_CMYK || d.jpeg_color_space == JCS_YCCK)
		d.out_color_space = JCS_CMYK;
	else
		d.out_color_space = JCS_RGB;
	jpeg_start_decompress(&d);
	c = d.out_color_space == JCS_GRAYSCALE ? 1 : 3;
	if (image_alloc(im, (int)d.output_width, (int)d.output_height, c) != 0 ||
	    (d.out_color_space == JCS_CMYK && (cmyk = malloc((size_t)im->w * 4)) == NULL))
		jpeg_fail(&e, "out of memory");
	stride = (size_t)im->w * (size_t)c;
	while (d.output_scanline < d.output_height) {
		out = im->p + (size_t)d.output_scanline * stride;
		row = cmyk != NULL ? cmyk : out;
		if (jpeg_read_scanlines(&d, &row, 1) != 1)
			break;
		if (cmyk != NULL)
			cmyk_to_rgb(cmyk, out, (size_t)im->w, d.saw_Adobe_marker);
	}
	free(cmyk);
	jpeg_destroy_decompress(&d);
	return 0;
}

/*
 * Encode a baseline JPEG at quality 1 to 100: gray for 1 or 2 channels,
 * else 4:2:0 color. Alpha is dropped. Returns a malloc buffer or NULL.
 */
void *
jpeg_encode(const struct image *im, int quality, size_t *outlen)
{
	struct jpeg_compress_struct cc;
	struct jerr e;
	char err[IMAGE_ERRLEN];
	unsigned char *out = NULL, *tmp;
	unsigned long size = 0;
	size_t c = (size_t)im->c, i, y;
	/* volatile: read after longjmp. */
	volatile int oc = im->c <= 2 ? 1 : 3;
	JSAMPROW row;

	if (im->p == NULL || im->c < 1 || im->c > 4 || quality < 1 || quality > 100)
		return NULL;
	if ((tmp = malloc((size_t)im->w * 3)) == NULL)
		return NULL;
	cc.err = jpeg_std_error(&e.pub);
	e.pub.error_exit = jpeg_on_error;
	e.pub.emit_message = jpeg_on_message;
	e.err = err;
	if (setjmp(e.jb)) {
		jpeg_destroy_compress(&cc);
		free(tmp);
		free(out);
		return NULL;
	}
	jpeg_create_compress(&cc);
	jpeg_mem_dest(&cc, &out, &size);
	cc.image_width = (JDIMENSION)im->w;
	cc.image_height = (JDIMENSION)im->h;
	cc.input_components = oc;
	cc.in_color_space = oc == 1 ? JCS_GRAYSCALE : JCS_RGB;
	jpeg_set_defaults(&cc);
	jpeg_set_quality(&cc, quality, TRUE);
	jpeg_start_compress(&cc, TRUE);
	for (y = 0; y < (size_t)im->h; y++) {
		row = im->p + y * (size_t)im->w * c;
		if (c != (size_t)oc) {
			for (i = 0; i < (size_t)im->w * (size_t)oc; i++)
				tmp[i] = row[i / (size_t)oc * c + i % (size_t)oc];
			row = tmp;
		}
		jpeg_write_scanlines(&cc, &row, 1);
	}
	jpeg_finish_compress(&cc);
	jpeg_destroy_compress(&cc);
	free(tmp);
	*outlen = size;
	return out;
}

/*
 * Decode a PNG or JPEG file, chosen by its first bytes. Returns 0, or
 * -1 with a message in err, which holds IMAGE_ERRLEN bytes.
 */
int
image_decode(const void *buf, size_t n, struct image *im, char *err)
{
	const unsigned char *p = buf;

	memset(im, 0, sizeof(*im));
	if (n > IMAGE_MAX_FILE)
		return fail(err, "image file too large");
	if (n >= 8 && memcmp(p, png_sig, 8) == 0)
		return png_read(p, n, im, err);
	if (n >= 3 && p[0] == 0xff && p[1] == 0xd8 && p[2] == 0xff)
		return jpeg_read(p, n, im, err);
	return fail(err, "not a PNG or JPEG file");
}

/* resize */

struct taps {
	int	*first;		/* first source index of each output */
	int	*n;		/* tap count of each output */
	float	*w;		/* weights, maxn per output */
	int	 maxn;
};

static void
taps_free(struct taps *t)
{
	free(t->first);
	free(t->n);
	free(t->w);
}

/* Triangle filter weights, widened by the scale factor when shrinking. */
static int
taps_make(struct taps *t, int in, int out)
{
	double scale = (double)in / out, support = scale > 1 ? scale : 1, c, sum, v;
	int i, j, lo, hi;

	t->maxn = (int)(2 * support) + 3;
	t->first = malloc((size_t)out * sizeof(int));
	t->n = malloc((size_t)out * sizeof(int));
	t->w = malloc((size_t)out * (size_t)t->maxn * sizeof(float));
	if (t->first == NULL || t->n == NULL || t->w == NULL) {
		taps_free(t);
		return -1;
	}
	for (i = 0; i < out; i++) {
		c = (i + 0.5) * scale - 0.5;
		lo = (int)(c - support) - 1;
		hi = (int)(c + support) + 1;
		lo = lo < 0 ? 0 : lo;
		hi = hi > in - 1 ? in - 1 : hi;
		if (hi - lo + 1 > t->maxn)
			hi = lo + t->maxn - 1;
		sum = 0;
		for (j = lo; j <= hi; j++) {
			v = 1 - (j > c ? j - c : c - j) / support;
			v = v > 0 ? v : 0;
			t->w[(size_t)i * (size_t)t->maxn + (size_t)(j - lo)] = (float)v;
			sum += v;
		}
		if (sum <= 0) {
			/* No tap in reach: take the nearest pixel. */
			j = (int)(c + 0.5);
			lo = j < 0 ? 0 : j > in - 1 ? in - 1 : j;
			hi = lo;
			t->w[(size_t)i * (size_t)t->maxn] = 1;
			sum = 1;
		}
		for (j = lo; j <= hi; j++)
			t->w[(size_t)i * (size_t)t->maxn + (size_t)(j - lo)] /= (float)sum;
		t->first[i] = lo;
		t->n[i] = hi - lo + 1;
	}
	return 0;
}

/* Resize with a triangle filter that averages over the area when shrinking. */
int
image_resize(const struct image *src, int w, int h, struct image *dst)
{
	struct taps tx, ty;
	float *tmp, acc;
	size_t c = (size_t)src->c, x, y, k;
	const float *wt, *col;
	const unsigned char *in;
	unsigned char *out;
	int j;

	memset(dst, 0, sizeof(*dst));
	if (image_alloc(dst, w, h, src->c) != 0)
		return -1;
	tmp = malloc((size_t)w * (size_t)src->h * c * sizeof(float));
	if (tmp == NULL || taps_make(&tx, src->w, w) != 0) {
		free(tmp);
		image_free(dst);
		return -1;
	}
	if (taps_make(&ty, src->h, h) != 0) {
		taps_free(&tx);
		free(tmp);
		image_free(dst);
		return -1;
	}
	/* Rows first into tmp, then columns into dst. */
	for (y = 0; y < (size_t)src->h; y++)
		for (x = 0; x < (size_t)w; x++) {
			wt = tx.w + x * (size_t)tx.maxn;
			in = src->p + (y * (size_t)src->w + (size_t)tx.first[x]) * c;
			for (k = 0; k < c; k++) {
				acc = 0;
				for (j = 0; j < tx.n[x]; j++)
					acc += wt[j] * in[(size_t)j * c + k];
				tmp[(y * (size_t)w + x) * c + k] = acc;
			}
		}
	for (y = 0; y < (size_t)h; y++)
		for (x = 0; x < (size_t)w; x++) {
			wt = ty.w + y * (size_t)ty.maxn;
			col = tmp + ((size_t)ty.first[y] * (size_t)w + x) * c;
			out = dst->p + (y * (size_t)w + x) * c;
			for (k = 0; k < c; k++) {
				acc = 0;
				for (j = 0; j < ty.n[y]; j++)
					acc += wt[j] * col[(size_t)j * (size_t)w * c + k];
				acc = acc < 0 ? 0 : acc > 255 ? 255 : acc;
				out[k] = (unsigned char)(acc + 0.5f);
			}
		}
	taps_free(&tx);
	taps_free(&ty);
	free(tmp);
	return 0;
}

/*
 * Bilinear resize as llama.cpp clip: the corners map to the corners,
 * and each result truncates toward zero.
 */
int
image_resize_bilinear(const struct image *src, int w, int h, struct image *dst)
{
	float xr, yr, px, py, xf, yf, top, bot;
	size_t c = (size_t)src->c, k;
	const unsigned char *p00, *p10, *p01, *p11;
	int x, y, x0, y0, x1, y1;

	memset(dst, 0, sizeof(*dst));
	if (image_alloc(dst, w, h, src->c) != 0)
		return -1;
	xr = w > 1 ? (float)(src->w - 1) / (float)(w - 1) : 0.0f;
	yr = h > 1 ? (float)(src->h - 1) / (float)(h - 1) : 0.0f;
	for (y = 0; y < h; y++) {
		py = (float)y * yr;
		y0 = (int)py < src->h - 1 ? (int)py : src->h - 1;
		y1 = y0 + 1 < src->h - 1 ? y0 + 1 : src->h - 1;
		yf = py - (float)y0;
		for (x = 0; x < w; x++) {
			px = (float)x * xr;
			x0 = (int)px < src->w - 1 ? (int)px : src->w - 1;
			x1 = x0 + 1 < src->w - 1 ? x0 + 1 : src->w - 1;
			xf = px - (float)x0;
			p00 = src->p + ((size_t)y0 * (size_t)src->w + (size_t)x0) * c;
			p10 = src->p + ((size_t)y0 * (size_t)src->w + (size_t)x1) * c;
			p01 = src->p + ((size_t)y1 * (size_t)src->w + (size_t)x0) * c;
			p11 = src->p + ((size_t)y1 * (size_t)src->w + (size_t)x1) * c;
			for (k = 0; k < c; k++) {
				top = (float)p00[k] + ((float)p10[k] - (float)p00[k]) * xf;
				bot = (float)p01[k] + ((float)p11[k] - (float)p01[k]) * xf;
				dst->p[((size_t)y * (size_t)w + (size_t)x) * c + k] =
				    (unsigned char)(top + (bot - top) * yf);
			}
		}
	}
	return 0;
}
