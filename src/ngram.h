/* Reader for a PLE n-gram hash embedding table in a gguf file. */
#ifndef NGRAM_H
#define NGRAM_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include <ggml.h>
#include <gguf.h>

#define NGRAM_MAX_N	8
#define NGRAM_MAX_HEADS	64

/*
 * Hash constants, from the <arch>.ple.* keys. Head h of token t is
 * row (mixed_n % vocab[h]) + offset[h] of the table, with n = 2 + h /
 * heads_per_ngram. See ngram_rows.
 */
struct ngram_params {
	int		 ngram_size;
	int		 heads_per_ngram;
	int		 n_heads;		/* (ngram_size - 1) * heads_per_ngram */
	int32_t		 eos;
	int32_t		 image_token;		/* stand-in for image positions */
	uint64_t	 mult[NGRAM_MAX_N];
	uint64_t	 offset[NGRAM_MAX_HEADS];
	uint64_t	 vocab[NGRAM_MAX_HEADS];
};

enum ngram_mode {
	NGRAM_DIRECT,		/* O_DIRECT reads, past the page cache */
	NGRAM_PREAD,		/* pread through the page cache */
	NGRAM_MMAP		/* copy from a map of the file */
};

struct ngram_stats {
	uint64_t	 calls;
	uint64_t	 rows;		/* rows asked for */
	uint64_t	 unique;	/* rows after dedupe */
	uint64_t	 reads;		/* read syscalls */
	uint64_t	 bytes;		/* bytes read from the file */
	double		 read_sec;	/* time in reads */
	double		 total_sec;	/* time in ngram_read */
};

struct ngram {
	struct ngram_params	 p;
	enum ngram_mode		 mode;
	int			 fell_back;	/* direct mode was asked for but failed */
	int			 fd;		/* buffered */
	int			 dfd;		/* O_DIRECT, or -1 */
	void			*map;
	size_t			 maplen;
	enum ggml_type		 type;
	int64_t			 head_dim;	/* row length in elements */
	int64_t			 nrows;
	size_t			 row_size;	/* row length in bytes */
	off_t			 base;		/* file offset of row 0 */
	size_t			 align;		/* direct I/O offset and size unit */
	int			 threads;
	char			*buf;		/* aligned read buffer */
	size_t			 bufsize;
	void			*ring;		/* struct io_uring, or NULL */
	struct ngram_stats	 st;
};

int		 ngram_params_gguf(struct ngram_params *, const struct gguf_context *,
		    char *, size_t);
void		 ngram_rows(const struct ngram_params *, const int32_t *, size_t,
		    const int32_t *, size_t, int64_t *);
struct ngram	*ngram_open(const char *, const char *, const char *, enum ngram_mode,
		    size_t, int, char *, size_t);
void		 ngram_close(struct ngram *);
int		 ngram_uring(struct ngram *, int);
int		 ngram_read(struct ngram *, const int64_t *, size_t, float *);

#endif
