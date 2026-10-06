/* Images in C: PNG and JPEG through libpng and libjpeg-turbo, and resize. */
#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>

#define IMAGE_MAX_SIDE		32768
#define IMAGE_MAX_PIXELS	((size_t)1 << 26)
#define IMAGE_MAX_FILE		((size_t)1 << 28)
#define IMAGE_ERRLEN		96

/* 8-bit pixels, rows top first. c is 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA. */
struct image {
	unsigned char	*p;
	int		 w;
	int		 h;
	int		 c;
};

void		 image_free(struct image *);
int		 image_alloc(struct image *, int, int, int);
int		 image_decode(const void *, size_t, struct image *, char *);
void		*png_encode(const struct image *, size_t *);
void		*jpeg_encode(const struct image *, int, size_t *);
int		 image_resize(const struct image *, int, int, struct image *);
int		 image_resize_bilinear(const struct image *, int, int, struct image *);
int		 image_resize_bicubic(const struct image *, int, int, struct image *);

#endif
