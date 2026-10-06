/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * GPT-2 style byte-level BPE tokenizer with the gpt2, qwen2 and llama3
 * pre-tokenizers, and the gemma4 BPE over raw UTF-8 with byte fallback.
 * The vocab and merges load into C tables from gguf metadata or from a
 * Hugging Face tokenizer.json. The behavior follows llama.cpp
 * llama-vocab.cpp and unicode.cpp, and Hugging Face tokenizers for
 * tokenizer.json.
 */

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gguf.h>
#include <lua.h>
#include <lauxlib.h>
#include <utf8proc.h>

#include "lgml.h"

#define TOKENIZER_MT	"lgml_tokenizer"

/* Unicode flags, the same bits as llama.cpp unicode_cpt_flags. */
#define UF_UNDEF	0x001
#define UF_NUMBER	0x002
#define UF_LETTER	0x004
#define UF_MARK		0x010
#define UF_WS		0x100

struct uni_range {
	uint32_t	 start;
	uint16_t	 flags;
};

#include "unicode_ranges.h"

static uint16_t uni_bmp[0x10000];
static int uni_ready;

static void
uni_init(void)
{
	size_t i;
	uint32_t c, end;

	if (uni_ready)
		return;
	for (i = 0; i + 1 < UNI_NRANGES; i++) {
		end = uni_ranges[i + 1].start;
		if (end > 0x10000)
			end = 0x10000;
		for (c = uni_ranges[i].start; c < end; c++)
			uni_bmp[c] = uni_ranges[i].flags;
	}
	uni_ready = 1;
}

static uint16_t
uni_flags(uint32_t cp)
{
	size_t lo, hi, mid;

	if (cp < 0x10000)
		return uni_bmp[cp];
	if (cp >= 0x110000)
		return UF_UNDEF;
	lo = 0;
	hi = UNI_NRANGES - 1;
	while (lo < hi) {
		mid = (lo + hi + 1) / 2;
		if (uni_ranges[mid].start <= cp)
			lo = mid;
		else
			hi = mid - 1;
	}
	return uni_ranges[lo].flags;
}

/*
 * Decode one code point with the rules of llama.cpp
 * unicode_cpt_from_utf8. Returns the length, or 0 if invalid.
 */
static size_t
utf8_decode(const unsigned char *s, size_t n, uint32_t *cp)
{
	unsigned char c = s[0];

	if (c < 0x80) {
		*cp = c;
		return 1;
	}
	if (!(c & 0x40))
		return 0;
	if (!(c & 0x20)) {
		if (n < 2 || (s[1] & 0xc0) != 0x80)
			return 0;
		*cp = ((uint32_t)(c & 0x1f) << 6) | (s[1] & 0x3f);
		return 2;
	}
	if (!(c & 0x10)) {
		if (n < 3 || (s[1] & 0xc0) != 0x80 || (s[2] & 0xc0) != 0x80)
			return 0;
		*cp = ((uint32_t)(c & 0x0f) << 12) |
		    ((uint32_t)(s[1] & 0x3f) << 6) | (s[2] & 0x3f);
		return 3;
	}
	if (!(c & 0x08)) {
		if (n < 4 || (s[1] & 0xc0) != 0x80 || (s[2] & 0xc0) != 0x80 ||
		    (s[3] & 0xc0) != 0x80)
			return 0;
		*cp = ((uint32_t)(c & 0x07) << 18) |
		    ((uint32_t)(s[1] & 0x3f) << 12) |
		    ((uint32_t)(s[2] & 0x3f) << 6) | (s[3] & 0x3f);
		return 4;
	}
	return 0;
}

/* Encode cp (at most 0x10ffff) into buf. Returns the length. */
static size_t
utf8_encode(uint32_t cp, unsigned char *buf)
{
	if (cp < 0x80) {
		buf[0] = (unsigned char)cp;
		return 1;
	}
	if (cp < 0x800) {
		buf[0] = (unsigned char)(0xc0 | (cp >> 6));
		buf[1] = (unsigned char)(0x80 | (cp & 0x3f));
		return 2;
	}
	if (cp < 0x10000) {
		buf[0] = (unsigned char)(0xe0 | (cp >> 12));
		buf[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
		buf[2] = (unsigned char)(0x80 | (cp & 0x3f));
		return 3;
	}
	buf[0] = (unsigned char)(0xf0 | (cp >> 18));
	buf[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3f));
	buf[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
	buf[3] = (unsigned char)(0x80 | (cp & 0x3f));
	return 4;
}

/* GPT-2 byte to code point map, and the inverse for cp < 324. */
#define BYTE_CP_MAX	324

static uint16_t byte_cp[256];
static int16_t cp_byte[BYTE_CP_MAX];

static void
byte_map_init(void)
{
	int b, n = 0;

	for (b = 0; b < BYTE_CP_MAX; b++)
		cp_byte[b] = -1;
	for (b = 0; b < 256; b++) {
		if ((b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) ||
		    (b >= 0xae && b <= 0xff))
			byte_cp[b] = (uint16_t)b;
		else
			byte_cp[b] = (uint16_t)(256 + n++);
		cp_byte[byte_cp[b]] = (int16_t)b;
	}
}

/* Token types from tokenizer.ggml.token_type. */
enum {
	TT_NORMAL = 1,
	TT_UNKNOWN = 2,
	TT_CONTROL = 3,
	TT_USER = 4,
	TT_UNUSED = 5,
	TT_BYTE = 6,
};

static const char *const tt_names[] = {
	"undefined", "normal", "unknown", "control", "user", "unused", "byte",
};

struct merge {
	uint64_t	 key;	/* merge_key(left, right); 0 is empty */
	uint32_t	 rank;
	int32_t		 id;
};

struct sym {
	int32_t		 id;
	int32_t		 prev;
	int32_t		 next;
};

struct heapent {
	uint64_t	 key;	/* rank << 32 | left position */
	int32_t		 lid;
	int32_t		 rid;
};

#define MAX_EOG		32
#define MAX_NL		64	/* longest newline run looked up whole */

struct tokenizer {
	int32_t		 n_vocab;
	uint32_t	*off;	/* piece i is pool[off[i] .. off[i + 1]) */
	char		*pool;
	uint8_t		*type;
	int32_t		 byte_id[256];

	struct merge	*merges;
	uint32_t	 mmask;

	int32_t		*spec;	/* special ids, by first byte, then longest */
	uint32_t	 sfirst[257];
	int32_t		 nspec;

	int32_t		 bos, eos, eot, pad, unk;
	int32_t		 eog[MAX_EOG];
	int		 neog;
	int		 add_bos;
	int		 max_digits;	/* digits per number pre-token */
	int		 gpt2;		/* the GPT-2 pre-tokenizer regex */
	int		 nfc;		/* normalize text to NFC first */
	uint8_t		*strip;		/* STRIP_ bits by token, or NULL */
	uint16_t	 letters;	/* flags that count as \p{L}: qwen35 adds \p{M} */
	int32_t		*phash;	/* llama3: normal piece bytes to id + 1 */
	uint32_t	 pmask;

	/* gemma4: one-character tokens and newline runs by raw text. */
	int		 gemma4;
	uint32_t	*ckey;	/* code point + 1; 0 is empty */
	int32_t		*cid;
	uint32_t	 cmask;
	int32_t		 nl_id[MAX_NL + 1];

	/* Scratch buffers for encode. */
	uint32_t	*cp;
	uint16_t	*fl;
	uint32_t	*co;
	size_t		 ccap;
	unsigned char	*sbuf;
	size_t		 scap;
	unsigned char	*ebuf;	/* gemma4: text with spaces escaped */
	size_t		 ecap;
	struct sym	*syms;
	struct heapent	*heap;
	size_t		 wcap;
	int32_t		*out;
	size_t		 nout;
	size_t		 ocap;

	int		 closed;
};

/* Grow *pp to at least n elements of size sz. Returns 0 on failure. */
static int
grow(void *pp, size_t *cap, size_t n, size_t sz)
{
	void **p = pp;
	void *q;
	size_t c = *cap ? *cap : 64;

	if (n <= *cap && *p != NULL)
		return 1;
	while (c < n) {
		if (c > SIZE_MAX / 2)
			return 0;
		c *= 2;
	}
	if (c > SIZE_MAX / sz)
		return 0;
	if ((q = realloc(*p, c * sz)) == NULL)
		return 0;
	*p = q;
	*cap = c;
	return 1;
}

static void
tok_free(struct tokenizer *t)
{
	free(t->off);
	free(t->pool);
	free(t->type);
	free(t->merges);
	free(t->phash);
	free(t->ckey);
	free(t->cid);
	free(t->spec);
	free(t->strip);
	free(t->cp);
	free(t->fl);
	free(t->co);
	free(t->sbuf);
	free(t->ebuf);
	free(t->syms);
	free(t->heap);
	free(t->out);
	memset(t, 0, sizeof(*t));
	t->closed = 1;
}

static uint64_t
hash_bytes(const void *p, size_t n)
{
	const unsigned char *s = p;
	uint64_t h = 1469598103934665603ULL;

	while (n-- > 0)
		h = (h ^ *s++) * 1099511628211ULL;
	return h;
}

static uint32_t
hash_pair(uint64_t key, uint32_t mask)
{
	return (uint32_t)((key * 0x9e3779b97f4a7c15ULL) >> 32) & mask;
}

/*
 * llama3 uses a whole pre-token as one token when the vocab has it
 * (llama.cpp ignore_merges). Index normal pieces by their bytes.
 */
static int
piece_index(struct tokenizer *t)
{
	uint32_t size = 1, i, h;

	while (size < (uint32_t)t->n_vocab * 2)
		size <<= 1;
	if ((t->phash = calloc(size, sizeof(*t->phash))) == NULL)
		return 0;
	t->pmask = size - 1;
	for (i = 0; i < (uint32_t)t->n_vocab; i++) {
		if (t->type[i] != TT_NORMAL || t->off[i + 1] == t->off[i])
			continue;
		h = (uint32_t)hash_bytes(t->pool + t->off[i], t->off[i + 1] - t->off[i]) & t->pmask;
		while (t->phash[h] != 0)
			h = (h + 1) & t->pmask;
		t->phash[h] = (int32_t)i + 1;
	}
	return 1;
}

static int32_t
piece_lookup(const struct tokenizer *t, const unsigned char *w, size_t n)
{
	uint32_t h = (uint32_t)hash_bytes(w, n) & t->pmask;
	int32_t id;

	while ((id = t->phash[h]) != 0) {
		id--;
		if (t->off[id + 1] - t->off[id] == n &&
		    memcmp(t->pool + t->off[id], w, n) == 0)
			return id;
		h = (h + 1) & t->pmask;
	}
	return -1;
}

/* Temporary map from the gguf token text to the token id. */
struct vmap {
	int32_t		*slot;	/* id + 1; 0 is empty */
	uint32_t	 mask;
	const char	**text;
	uint32_t	*len;
};

static int32_t
vmap_find(const struct vmap *m, const char *s, size_t n)
{
	uint32_t i = (uint32_t)hash_bytes(s, n) & m->mask;
	int32_t id;

	while ((id = m->slot[i]) != 0) {
		id--;
		if (m->len[id] == n && memcmp(m->text[id], s, n) == 0)
			return id;
		i = (i + 1) & m->mask;
	}
	return -1;
}

/*
 * gemma4 BPE starts from one symbol per character. Map each token whose
 * raw text is one character to its id, as llama.cpp text_to_token.
 */
static int
char_index(struct tokenizer *t, const struct vmap *vm)
{
	const unsigned char *s;
	uint32_t size = 1, i, h, cp, nc = 0;

	for (i = 0; i < (uint32_t)t->n_vocab; i++) {
		s = (const unsigned char *)vm->text[i];
		if (vm->len[i] > 0 && utf8_decode(s, vm->len[i], &cp) == vm->len[i])
			nc++;
	}
	while (size < nc * 2)
		size <<= 1;
	t->ckey = calloc(size, sizeof(*t->ckey));
	t->cid = calloc(size, sizeof(*t->cid));
	if (t->ckey == NULL || t->cid == NULL)
		return 0;
	t->cmask = size - 1;
	for (i = 0; i < (uint32_t)t->n_vocab; i++) {
		s = (const unsigned char *)vm->text[i];
		if (vm->len[i] == 0 || utf8_decode(s, vm->len[i], &cp) != vm->len[i])
			continue;
		h = hash_pair(cp, t->cmask);
		while (t->ckey[h] != 0 && t->ckey[h] != cp + 1)
			h = (h + 1) & t->cmask;
		if (t->ckey[h] == 0) {
			t->ckey[h] = cp + 1;
			t->cid[h] = (int32_t)i;
		}
	}
	return 1;
}

static int32_t
char_lookup(const struct tokenizer *t, uint32_t cp)
{
	uint32_t h = hash_pair(cp, t->cmask);

	while (t->ckey[h] != 0) {
		if (t->ckey[h] == cp + 1)
			return t->cid[h];
		h = (h + 1) & t->cmask;
	}
	return -1;
}

/* Never 0, so the pair (0, 0) is not an empty slot. */
static uint64_t
merge_key(int32_t a, int32_t b)
{
	return ((uint64_t)(uint32_t)a + 1) << 32 | (uint32_t)b;
}

static int32_t
merge_find(const struct tokenizer *t, int32_t a, int32_t b, uint32_t *rank)
{
	uint64_t key = merge_key(a, b);
	uint32_t i = hash_pair(key, t->mmask);

	for (;;) {
		if (t->merges[i].key == key) {
			*rank = t->merges[i].rank;
			return t->merges[i].id;
		}
		if (t->merges[i].key == 0)
			return -1;
		i = (i + 1) & t->mmask;
	}
}

static uint32_t
pow2_at_least(size_t n)
{
	uint32_t p = 1024;

	while (p < n)
		p *= 2;
	return p;
}

/* Read an integer metadata key. Returns def if absent or not an int. */
static int64_t
meta_int(struct gguf_context *c, const char *key, int64_t def)
{
	int64_t k = gguf_find_key(c, key);

	if (k < 0)
		return def;
	switch (gguf_get_kv_type(c, k)) {
	case GGUF_TYPE_UINT32:
		return gguf_get_val_u32(c, k);
	case GGUF_TYPE_INT32:
		return gguf_get_val_i32(c, k);
	case GGUF_TYPE_UINT64:
		return (int64_t)gguf_get_val_u64(c, k);
	case GGUF_TYPE_INT64:
		return gguf_get_val_i64(c, k);
	case GGUF_TYPE_BOOL:
		return gguf_get_val_bool(c, k);
	default:
		return def;
	}
}

static const char *
meta_str(struct gguf_context *c, const char *key)
{
	int64_t k = gguf_find_key(c, key);

	if (k < 0 || gguf_get_kv_type(c, k) != GGUF_TYPE_STRING)
		return NULL;
	return gguf_get_val_str(c, k);
}

static int32_t
valid_id(const struct tokenizer *t, int64_t id)
{
	return id >= 0 && id < t->n_vocab ? (int32_t)id : -1;
}

/*
 * Token names that llama.cpp makes control tokens. ctl_names lists end
 * of turn names in priority order first. eog_names also end generation.
 */
static const char *const ctl_names[] = {
	"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>",
	"<|endoftext|>", "<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]",
	"<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>",
	"<end_of_utterance>", "<|eom_id|>",
	"<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<PRE>",
	"\xe2\x96\x81" "<PRE>", "<|code_prefix|>", "<|prefix|>",
	"<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<SUF>",
	"\xe2\x96\x81" "<SUF>", "<|code_suffix|>", "<|suffix|>",
	"<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<MID>",
	"\xe2\x96\x81" "<MID>", "<|code_middle|>", "<|middle|>",
	NULL,
};
#define N_EOT_NAMES	10

static const char *const eog_names[] = {
	"<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>",
	"<|flush|>", "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>",
	"<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
	"<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
	"<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>",
	"<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]",
	"<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>",
	"<|file_sep|>",
	NULL,
};

static int
hexval(int c)
{
	if (c >= '0' && c <= '9')
		return c - '0';
	if (c >= 'A' && c <= 'F')
		return c - 'A' + 10;
	if (c >= 'a' && c <= 'f')
		return c - 'a' + 10;
	return -1;
}

/* gemma4: U+2581 back to space, <0xXX> to the byte. */
static void
unescape_piece(const char *s, size_t n, int ty, char *pool, size_t *pos)
{
	size_t i;
	int hi, lo;

	if (ty == TT_BYTE && n == 6 && memcmp(s, "<0x", 3) == 0 &&
	    s[5] == '>' && (hi = hexval(s[3])) >= 0 && (lo = hexval(s[4])) >= 0) {
		pool[(*pos)++] = (char)(hi << 4 | lo);
		return;
	}
	for (i = 0; i < n; i++) {
		if (ty == TT_NORMAL && i + 2 < n && memcmp(s + i, "\xe2\x96\x81", 3) == 0) {
			pool[(*pos)++] = ' ';
			i += 2;
		} else
			pool[(*pos)++] = s[i];
	}
}

/* Byte-decode a gguf token text into pool at *pos. */
static void
decode_piece(const char *s, size_t n, char *pool, size_t *pos)
{
	const unsigned char *u = (const unsigned char *)s;
	size_t i = 0, l;
	uint32_t c;

	while (i < n) {
		l = utf8_decode(u + i, n - i, &c);
		if (l > 0 && c < BYTE_CP_MAX && cp_byte[c] >= 0) {
			pool[(*pos)++] = (char)cp_byte[c];
		} else {
			if (l == 0)
				l = 1;
			memcpy(pool + *pos, u + i, l);
			*pos += l;
		}
		i += l;
	}
}

static int
is_special_type(int ty)
{
	return ty == TT_CONTROL || ty == TT_USER || ty == TT_UNKNOWN;
}

/* qsort has no context argument, so spec_cmp reads this. */
static const struct tokenizer *sort_tok;

static int
spec_cmp(const void *pa, const void *pb)
{
	int32_t a = *(const int32_t *)pa, b = *(const int32_t *)pb;
	const struct tokenizer *t = sort_tok;
	unsigned char fa = (unsigned char)t->pool[t->off[a]];
	unsigned char fb = (unsigned char)t->pool[t->off[b]];
	uint32_t la = t->off[a + 1] - t->off[a];
	uint32_t lb = t->off[b + 1] - t->off[b];

	if (fa != fb)
		return fa < fb ? -1 : 1;
	if (la != lb)
		return la > lb ? -1 : 1;
	return a < b ? -1 : a > b;
}

/*
 * The tokens and merges of a tokenizer, from gguf metadata or from a
 * tokenizer.json table. The source owns the strings. A merge whose
 * left side is NULL does not count.
 */
struct tok_src {
	const char	*model;
	const char	*pre;
	size_t		 n;
	const char	**text;
	uint32_t	*len;
	int32_t		*types;		/* token types, or NULL: all normal */
	uint8_t		*strip;		/* STRIP_ bits by token, or NULL */
	size_t		 nm;
	const char	**ma;		/* left side of each merge */
	const char	**mb;		/* right side */
	uint32_t	*mla;
	uint32_t	*mlb;
	int64_t		 bos, eos, eot, pad, unk;
	int		 add_bos;
	int		 nfc;
};

/* An added token of tokenizer.json takes the space before or after it. */
#define STRIP_LEFT	1
#define STRIP_RIGHT	2

static void
src_free(struct tok_src *s)
{
	free(s->text);
	free(s->len);
	free(s->types);
	free(s->strip);
	free(s->ma);
	free(s->mb);
	free(s->mla);
	free(s->mlb);
}

/* Room for n tokens and nm merges. Returns 0 on failure. */
static int
src_alloc(struct tok_src *s, size_t n, size_t nm)
{
	s->n = n;
	s->nm = nm;
	s->text = calloc(n, sizeof(*s->text));
	s->len = calloc(n, sizeof(*s->len));
	s->types = calloc(n, sizeof(*s->types));
	s->ma = calloc(nm + 1, sizeof(*s->ma));
	s->mb = calloc(nm + 1, sizeof(*s->mb));
	s->mla = calloc(nm + 1, sizeof(*s->mla));
	s->mlb = calloc(nm + 1, sizeof(*s->mlb));
	return s->text != NULL && s->len != NULL && s->types != NULL &&
	    s->ma != NULL && s->mb != NULL && s->mla != NULL && s->mlb != NULL;
}

/* Split merge i "a b" at the first space after the first byte. */
static void
src_merge(struct tok_src *s, size_t i, const char *ms, size_t n)
{
	const char *sp;

	if (n < 3 || (sp = memchr(ms + 1, ' ', n - 1)) == NULL)
		return;
	s->ma[i] = ms;
	s->mla[i] = (uint32_t)(sp - ms);
	s->mb[i] = sp + 1;
	s->mlb[i] = (uint32_t)(n - (size_t)(sp - ms) - 1);
}

static const char *
src_gguf(struct tok_src *s, struct gguf_context *c, const char *def_model,
    const char *def_pre)
{
	const int32_t *types = NULL;
	const char *ms;
	int64_t kt, kty, km;
	size_t n, nm, i;

	if ((s->model = meta_str(c, "tokenizer.ggml.model")) == NULL)
		s->model = def_model;
	if ((s->pre = meta_str(c, "tokenizer.ggml.pre")) == NULL)
		s->pre = def_pre;
	kt = gguf_find_key(c, "tokenizer.ggml.tokens");
	if (kt < 0 || gguf_get_kv_type(c, kt) != GGUF_TYPE_ARRAY ||
	    gguf_get_arr_type(c, kt) != GGUF_TYPE_STRING)
		return "missing tokenizer.ggml.tokens";
	n = gguf_get_arr_n(c, kt);
	if (n < 256 || n > INT32_MAX / 4)
		return "bad vocab size";
	kty = gguf_find_key(c, "tokenizer.ggml.token_type");
	if (kty >= 0) {
		if (gguf_get_kv_type(c, kty) != GGUF_TYPE_ARRAY ||
		    gguf_get_arr_type(c, kty) != GGUF_TYPE_INT32 ||
		    gguf_get_arr_n(c, kty) != n)
			return "bad tokenizer.ggml.token_type";
		types = gguf_get_arr_data(c, kty);
	}
	km = gguf_find_key(c, "tokenizer.ggml.merges");
	if (km < 0 || gguf_get_kv_type(c, km) != GGUF_TYPE_ARRAY ||
	    gguf_get_arr_type(c, km) != GGUF_TYPE_STRING)
		return "missing tokenizer.ggml.merges";
	nm = gguf_get_arr_n(c, km);
	if (nm > UINT32_MAX / 4)
		return "too many merges";
	if (!src_alloc(s, n, nm))
		return "out of memory";
	for (i = 0; i < n; i++) {
		s->text[i] = gguf_get_arr_str(c, kt, i);
		s->len[i] = (uint32_t)strlen(s->text[i]);
		s->types[i] = types != NULL ? types[i] : TT_NORMAL;
	}
	for (i = 0; i < nm; i++) {
		ms = gguf_get_arr_str(c, km, i);
		src_merge(s, i, ms, strlen(ms));
	}
	s->bos = meta_int(c, "tokenizer.ggml.bos_token_id", -1);
	s->eos = meta_int(c, "tokenizer.ggml.eos_token_id", -1);
	s->pad = meta_int(c, "tokenizer.ggml.padding_token_id", -1);
	s->unk = meta_int(c, "tokenizer.ggml.unknown_token_id", -1);
	s->eot = meta_int(c, "tokenizer.ggml.eot_token_id", -1);
	s->add_bos = meta_int(c, "tokenizer.ggml.add_bos_token", 0) != 0;
	s->nfc = meta_int(c, "tokenizer.ggml.normalizer.nfc", 0) != 0;
	return NULL;
}

/* True for nil and the JSON null (a NULL light userdata). */
static int
is_null(lua_State *L, int idx)
{
	return lua_isnil(L, idx) ||
	    (lua_islightuserdata(L, idx) && lua_touserdata(L, idx) == NULL);
}

/* Field k of the table at idx as a string, or NULL. Pops nothing. */
static const char *
field_str(lua_State *L, int idx, const char *k)
{
	const char *s;

	lua_getfield(L, idx, k);
	s = lua_type(L, -1) == LUA_TSTRING ? lua_tostring(L, -1) : NULL;
	lua_pop(L, 1);
	return s;
}

/* Field k of the table at idx as a boolean; def for nil or null. */
static int
field_bool(lua_State *L, int idx, const char *k, int def)
{
	int r;

	lua_getfield(L, idx, k);
	r = is_null(L, -1) ? def : lua_toboolean(L, -1);
	lua_pop(L, 1);
	return r;
}

/* The id of the added token with this content, or -1. */
static int64_t
added_id(lua_State *L, int added, const char *content)
{
	lua_Integer i, n = (lua_Integer)lua_rawlen(L, added), id = -1;
	const char *c;

	for (i = 1; i <= n && id < 0 && content != NULL; i++) {
		lua_rawgeti(L, added, i);
		if (lua_istable(L, -1) && (c = field_str(L, -1, "content")) != NULL &&
		    strcmp(c, content) == 0) {
			lua_getfield(L, -1, "id");
			id = lua_tointeger(L, -1);
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
	}
	return id;
}

/*
 * The special token of a TemplateProcessing single template at its
 * first (last = 0) or last (last = 1) item, or -1.
 */
static int64_t
template_id(lua_State *L, int root, int added, int last)
{
	lua_Integer n, i;
	int64_t id = -1;
	const char *name;

	lua_getfield(L, root, "post_processor");
	if (!lua_istable(L, -1) || (name = field_str(L, -1, "type")) == NULL ||
	    strcmp(name, "TemplateProcessing") != 0) {
		lua_pop(L, 1);
		return -1;
	}
	lua_getfield(L, -1, "single");
	if (lua_istable(L, -1) && (n = (lua_Integer)lua_rawlen(L, -1)) > 0) {
		i = last ? n : 1;
		lua_rawgeti(L, -1, i);
		if (lua_istable(L, -1)) {
			lua_getfield(L, -1, "SpecialToken");
			if (lua_istable(L, -1))
				id = added_id(L, added, field_str(L, -1, "id"));
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
	}
	lua_pop(L, 2);
	return id;
}

/* The gguf pre-tokenizer names of known tokenizer.json Split regexes. */
static const struct {
	const char	*regex;
	const char	*pre;
} split_pre[] = {
	{ "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++"
	    "[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+", "llada-moe" },
};

/*
 * The gguf pre-tokenizer name for the tokenizer.json pre_tokenizer at
 * idx, or NULL. It is ByteLevel with its GPT-2 regex, or a Sequence of a
 * known Split regex and ByteLevel without a regex.
 */
static const char *
json_pre(lua_State *L, int idx)
{
	const char *ty, *re, *pre = NULL;
	size_t i;

	if (!lua_istable(L, idx) || (ty = field_str(L, idx, "type")) == NULL)
		return NULL;
	if (strcmp(ty, "ByteLevel") == 0)
		return field_bool(L, idx, "use_regex", 1) &&
		    !field_bool(L, idx, "add_prefix_space", 0) ? "gpt2" : NULL;
	if (strcmp(ty, "Sequence") != 0)
		return NULL;
	lua_getfield(L, idx, "pretokenizers");
	if (!lua_istable(L, -1) || lua_rawlen(L, -1) != 2) {
		lua_pop(L, 1);
		return NULL;
	}
	lua_rawgeti(L, -1, 1);
	lua_rawgeti(L, -2, 2);
	if (lua_istable(L, -2) && lua_istable(L, -1) &&
	    (ty = field_str(L, -2, "type")) != NULL && strcmp(ty, "Split") == 0 &&
	    (ty = field_str(L, -2, "behavior")) != NULL && strcmp(ty, "Isolated") == 0 &&
	    !field_bool(L, -2, "invert", 0) &&
	    (ty = field_str(L, -1, "type")) != NULL && strcmp(ty, "ByteLevel") == 0 &&
	    !field_bool(L, -1, "use_regex", 1) && !field_bool(L, -1, "add_prefix_space", 0)) {
		lua_getfield(L, -2, "pattern");
		re = lua_istable(L, -1) ? field_str(L, -1, "Regex") : NULL;
		for (i = 0; re != NULL && i < sizeof(split_pre) / sizeof(split_pre[0]); i++)
			if (strcmp(re, split_pre[i].regex) == 0)
				pre = split_pre[i].pre;
		lua_pop(L, 1);
	}
	lua_pop(L, 3);
	return pre;
}

/*
 * The source from a decoded tokenizer.json at root: a BPE model with
 * a pre-tokenizer that json_pre knows, and no normalizer or NFC. The
 * table must stay on the stack while the source is in use.
 */
static const char *
src_json(lua_State *L, int root, struct tok_src *s)
{
	int top = lua_gettop(L), model, vocab, added, merges;
	lua_Integer id, maxid = -1, i, na;
	const char *str, *err = NULL;
	size_t l, nm;
	int isnum, sp;

	s->model = "gpt2";
	s->pre = "gpt2";
	s->bos = s->eos = s->eot = s->pad = s->unk = -1;
	lua_getfield(L, root, "normalizer");
	if (!is_null(L, -1)) {
		if (!lua_istable(L, -1) || (str = field_str(L, -1, "type")) == NULL ||
		    strcmp(str, "NFC") != 0) {
			err = "unsupported normalizer (need none or NFC)";
			goto done;
		}
		s->nfc = 1;
	}
	lua_getfield(L, root, "pre_tokenizer");
	if ((s->pre = json_pre(L, lua_gettop(L))) == NULL) {
		err = "unsupported pre_tokenizer (need ByteLevel with use_regex, or a known "
		    "Split regex then ByteLevel, and no prefix space)";
		goto done;
	}
	lua_getfield(L, root, "model");
	model = lua_gettop(L);
	if (!lua_istable(L, model) || (str = field_str(L, model, "type")) == NULL ||
	    strcmp(str, "BPE") != 0 || field_bool(L, model, "byte_fallback", 0)) {
		err = "unsupported model (need BPE without byte fallback)";
		goto done;
	}
	lua_getfield(L, model, "vocab");
	vocab = lua_gettop(L);
	lua_getfield(L, root, "added_tokens");
	added = lua_gettop(L);
	lua_getfield(L, model, "merges");
	merges = lua_gettop(L);
	if (!lua_istable(L, vocab) || !lua_istable(L, merges)) {
		err = "model has no vocab or merges";
		goto done;
	}
	if (!lua_istable(L, added)) {
		lua_pop(L, 1);
		lua_newtable(L);
	}

	/* The token count is the largest id plus 1. */
	lua_pushnil(L);
	while (lua_next(L, vocab) != 0) {
		id = lua_tointegerx(L, -1, &isnum);
		lua_pop(L, 1);
		if (!isnum || lua_type(L, -1) != LUA_TSTRING || id < 0 || id >= INT32_MAX / 4) {
			lua_pop(L, 1);
			err = "bad vocab entry";
			goto done;
		}
		maxid = id > maxid ? id : maxid;
	}
	na = (lua_Integer)lua_rawlen(L, added);
	for (i = 1; i <= na; i++) {
		if (lua_rawgeti(L, added, i) != LUA_TTABLE) {
			err = "bad added token";
			goto done;
		}
		lua_getfield(L, -1, "id");
		id = lua_tointegerx(L, -1, &isnum);
		lua_pop(L, 1);
		if (!isnum || id < 0 || id >= INT32_MAX / 4 ||
		    field_str(L, -1, "content") == NULL) {
			err = "bad added token";
			goto done;
		}
		lua_pop(L, 1);
		maxid = id > maxid ? id : maxid;
	}
	if (maxid < 255) {
		err = "bad vocab size";
		goto done;
	}
	nm = lua_rawlen(L, merges);
	if (nm > UINT32_MAX / 4 || !src_alloc(s, (size_t)maxid + 1, nm) ||
	    (s->strip = calloc((size_t)maxid + 1, 1)) == NULL) {
		err = "out of memory";
		goto done;
	}

	lua_pushnil(L);
	while (lua_next(L, vocab) != 0) {
		id = lua_tointeger(L, -1);
		lua_pop(L, 1);
		s->text[id] = lua_tolstring(L, -1, &l);
		s->len[id] = (uint32_t)l;
		s->types[id] = TT_NORMAL;
	}
	/* Added tokens split first, as control (special) or user tokens. */
	for (i = 1; i <= na; i++) {
		lua_rawgeti(L, added, i);
		lua_getfield(L, -1, "id");
		id = lua_tointeger(L, -1);
		lua_getfield(L, -2, "content");
		s->text[id] = lua_tolstring(L, -1, &l);
		s->len[id] = (uint32_t)l;
		lua_pop(L, 2);
		s->types[id] = field_bool(L, -1, "special", 0) ? TT_CONTROL : TT_USER;
		sp = field_bool(L, -1, "lstrip", 0) ? STRIP_LEFT : 0;
		sp |= field_bool(L, -1, "rstrip", 0) ? STRIP_RIGHT : 0;
		s->strip[id] = (uint8_t)sp;
		lua_pop(L, 1);
	}
	for (i = 0; i <= maxid; i++) {
		if (s->text[i] == NULL) {
			s->text[i] = "";
			s->types[i] = TT_UNUSED;
		}
	}
	/* A merge is "a b" or ["a", "b"]. */
	for (i = 0; i < (lua_Integer)nm; i++) {
		lua_rawgeti(L, merges, i + 1);
		if (lua_type(L, -1) == LUA_TSTRING) {
			str = lua_tolstring(L, -1, &l);
			src_merge(s, (size_t)i, str, l);
		} else if (lua_istable(L, -1) && lua_rawlen(L, -1) == 2) {
			lua_rawgeti(L, -1, 1);
			lua_rawgeti(L, -2, 2);
			if (lua_type(L, -2) == LUA_TSTRING && lua_type(L, -1) == LUA_TSTRING) {
				s->ma[i] = lua_tolstring(L, -2, &l);
				s->mla[i] = (uint32_t)l;
				s->mb[i] = lua_tolstring(L, -1, &l);
				s->mlb[i] = (uint32_t)l;
			}
			lua_pop(L, 2);
		}
		lua_pop(L, 1);
	}
	s->bos = template_id(L, root, added, 0);
	s->eos = template_id(L, root, added, 1);
	if ((str = field_str(L, model, "unk_token")) != NULL)
		s->unk = added_id(L, added, str);
	lua_getfield(L, root, "padding");
	if (lua_istable(L, -1)) {
		lua_getfield(L, -1, "pad_id");
		id = lua_tointegerx(L, -1, &isnum);
		s->pad = isnum ? id : -1;
		lua_pop(L, 1);
	}
done:
	lua_settop(L, top);
	return err;
}

static const char *
load_merges(struct tokenizer *t, const struct tok_src *src, const struct vmap *vm)
{
	char *cat = NULL;
	size_t i, la, lb, cap = 0;
	int32_t a, b, r;
	uint64_t key;
	uint32_t j;

	t->mmask = pow2_at_least(src->nm + src->nm / 2 + 1) - 1;
	t->merges = calloc((size_t)t->mmask + 1, sizeof(*t->merges));
	if (t->merges == NULL)
		return "out of memory";
	for (i = 0; i < src->nm; i++) {
		if (src->ma[i] == NULL)
			continue;
		la = src->mla[i];
		lb = src->mlb[i];
		a = vmap_find(vm, src->ma[i], la);
		b = vmap_find(vm, src->mb[i], lb);
		if (a < 0 || b < 0)
			continue;
		if (!grow(&cat, &cap, la + lb, 1)) {
			free(cat);
			return "out of memory";
		}
		memcpy(cat, src->ma[i], la);
		memcpy(cat + la, src->mb[i], lb);
		/* A merge whose result is not a token is dropped. */
		if ((r = vmap_find(vm, cat, la + lb)) < 0)
			continue;
		key = merge_key(a, b);
		j = hash_pair(key, t->mmask);
		while (t->merges[j].key != 0 && t->merges[j].key != key)
			j = (j + 1) & t->mmask;
		if (t->merges[j].key == 0) {
			t->merges[j].key = key;
			t->merges[j].rank = (uint32_t)i;
			t->merges[j].id = r;
		}
	}
	free(cat);
	return NULL;
}

static void
load_special(struct tokenizer *t, const struct tok_src *src, const struct vmap *vm)
{
	int32_t id, i;
	int k, e;

	t->bos = valid_id(t, src->bos);
	t->eos = valid_id(t, src->eos);
	t->pad = valid_id(t, src->pad);
	t->unk = valid_id(t, src->unk);
	t->eot = valid_id(t, src->eot);
	t->add_bos = src->add_bos;

	for (k = 0; ctl_names[k] != NULL; k++) {
		id = vmap_find(vm, ctl_names[k], strlen(ctl_names[k]));
		if (id < 0)
			continue;
		if (t->eot < 0 && k < N_EOT_NAMES)
			t->eot = id;
		t->type[id] = TT_CONTROL;
	}
	t->neog = 0;
	for (k = 0; eog_names[k] != NULL; k++) {
		id = vmap_find(vm, eog_names[k], strlen(eog_names[k]));
		if (id < 0 || t->neog >= MAX_EOG)
			continue;
		t->type[id] = TT_CONTROL;
		t->eog[t->neog++] = id;
	}
	for (e = 0; e < 2; e++) {
		id = e == 0 ? t->eos : t->eot;
		if (id < 0)
			continue;
		for (i = 0; i < t->neog && t->eog[i] != id; i++)
			;
		if (i == t->neog && t->neog < MAX_EOG)
			t->eog[t->neog++] = id;
	}

	/* With <|tool_response> (gemma4), </s> is a normal token. */
	if (vmap_find(vm, "<|tool_response>", 16) < 0 ||
	    (id = vmap_find(vm, "</s>", 4)) < 0)
		return;
	for (i = 0; i < t->neog && t->eog[i] != id; i++)
		;
	if (i == t->neog)
		return;
	memmove(t->eog + i, t->eog + i + 1, (t->neog - i - 1) * sizeof(*t->eog));
	t->neog--;
	t->type[id] = TT_NORMAL;
}

static const char *
load_spec(struct tokenizer *t)
{
	int32_t i, j;
	int k;

	t->nspec = 0;
	for (i = 0; i < t->n_vocab; i++)
		if (is_special_type(t->type[i]) && t->off[i + 1] > t->off[i])
			t->nspec++;
	if ((t->spec = calloc((size_t)t->nspec + 1, sizeof(*t->spec))) == NULL)
		return "out of memory";
	t->nspec = 0;
	for (i = 0; i < t->n_vocab; i++)
		if (is_special_type(t->type[i]) && t->off[i + 1] > t->off[i])
			t->spec[t->nspec++] = i;
	sort_tok = t;
	qsort(t->spec, (size_t)t->nspec, sizeof(*t->spec), spec_cmp);
	sort_tok = NULL;
	j = 0;
	for (k = 0; k <= 256; k++) {
		while (j < t->nspec &&
		    (unsigned char)t->pool[t->off[t->spec[j]]] < k)
			j++;
		t->sfirst[k] = (uint32_t)j;
	}
	return NULL;
}

static const char *
tok_load(struct tokenizer *t, const struct tok_src *src)
{
	struct vmap vm = { NULL, 0, NULL, NULL };
	const char *model = src->model, *pre = src->pre, *err = NULL;
	static const char hex[] = "0123456789ABCDEF";
	unsigned char enc[4];
	char nl[MAX_NL], bt[6] = { '<', '0', 'x', 0, 0, '>' };
	size_t n = src->n, i, l, total, pos;
	uint32_t h;
	int k;

	if (model != NULL && strcmp(model, "gemma4") == 0)
		t->gemma4 = 1;
	else if (model == NULL || strcmp(model, "gpt2") != 0)
		return "tokenizer.ggml.model is not gpt2 or gemma4";
	if (t->gemma4)
		t->max_digits = 0;
	else if (pre != NULL && (strcmp(pre, "qwen2") == 0 ||
	    strcmp(pre, "deepseek-r1-qwen") == 0 || strcmp(pre, "qwen35") == 0 ||
	    strcmp(pre, "bailingmoe") == 0 || strcmp(pre, "bailingmoe2") == 0 ||
	    strcmp(pre, "llada-moe") == 0))
		t->max_digits = 1;
	else if (pre != NULL && (strcmp(pre, "llama-bpe") == 0 ||
	    strcmp(pre, "llama3") == 0))
		t->max_digits = 3;
	else if (pre != NULL && (strcmp(pre, "gpt2") == 0 || strcmp(pre, "gpt-2") == 0))
		t->gpt2 = 1;
	else
		return "unsupported tokenizer.ggml.pre (need gpt2, qwen2, qwen35 or llama-bpe)";
	t->letters = UF_LETTER;
	if (pre != NULL && strcmp(pre, "qwen35") == 0)
		t->letters |= UF_MARK;
	t->nfc = src->nfc;
	if (n < 256 || n > INT32_MAX / 4)
		return "bad vocab size";
	t->n_vocab = (int32_t)n;

	vm.mask = pow2_at_least(n * 2) - 1;
	vm.slot = calloc((size_t)vm.mask + 1, sizeof(*vm.slot));
	vm.text = src->text;
	vm.len = src->len;
	t->type = calloc(n, 1);
	t->off = calloc(n + 1, sizeof(*t->off));
	if (src->strip != NULL && (t->strip = malloc(n)) != NULL)
		memcpy(t->strip, src->strip, n);
	if (vm.slot == NULL || t->type == NULL || t->off == NULL ||
	    (src->strip != NULL && t->strip == NULL)) {
		err = "out of memory";
		goto done;
	}

	/* First text wins if the vocab has duplicates. */
	total = 0;
	for (i = 0; i < n; i++) {
		l = vm.len[i];
		if (l > UINT32_MAX / 4 || total > UINT32_MAX / 2) {
			err = "vocab too large";
			goto done;
		}
		total += l;
		t->type[i] = TT_NORMAL;
		if (src->types != NULL && src->types[i] >= TT_NORMAL && src->types[i] <= TT_BYTE)
			t->type[i] = (uint8_t)src->types[i];
		h = (uint32_t)hash_bytes(vm.text[i], l) & vm.mask;
		while (vm.slot[h] != 0 && (vm.len[vm.slot[h] - 1] != l ||
		    memcmp(vm.text[vm.slot[h] - 1], vm.text[i], l) != 0))
			h = (h + 1) & vm.mask;
		if (vm.slot[h] == 0)
			vm.slot[h] = (int32_t)i + 1;
	}

	/* gemma4 byte tokens are <0xXX>; a missing one drops the byte. */
	for (k = 0; k < 256 && t->gemma4; k++) {
		bt[3] = hex[k >> 4];
		bt[4] = hex[k & 15];
		t->byte_id[k] = vmap_find(&vm, bt, sizeof(bt));
	}
	for (k = 0; k < 256 && !t->gemma4; k++) {
		l = utf8_encode(byte_cp[k], enc);
		t->byte_id[k] = vmap_find(&vm, (const char *)enc, l);
		/* Encode makes valid UTF-8, which has no C0, C1 or F5 to FF. */
		if (t->byte_id[k] < 0 && k != 0xc0 && k != 0xc1 && k < 0xf5) {
			err = "vocab lacks a byte token";
			goto done;
		}
	}
	if ((err = load_merges(t, src, &vm)) != NULL)
		goto done;
	load_special(t, src, &vm);
	if (t->gemma4) {
		t->add_bos = 1;
		memset(nl, '\n', sizeof(nl));
		for (k = 1; k <= MAX_NL; k++)
			t->nl_id[k] = vmap_find(&vm, nl, (size_t)k);
		if (!char_index(t, &vm)) {
			err = "out of memory";
			goto done;
		}
	}

	/*
	 * Pieces: byte-decoded (gpt2) or unescaped (gemma4) for normal
	 * tokens, raw text otherwise.
	 */
	if ((t->pool = malloc(total + 1)) == NULL) {
		err = "out of memory";
		goto done;
	}
	pos = 0;
	for (i = 0; i < n; i++) {
		t->off[i] = (uint32_t)pos;
		if (t->gemma4) {
			unescape_piece(vm.text[i], vm.len[i], t->type[i],
			    t->pool, &pos);
		} else if (t->type[i] == TT_NORMAL) {
			decode_piece(vm.text[i], vm.len[i], t->pool, &pos);
		} else {
			memcpy(t->pool + pos, vm.text[i], vm.len[i]);
			pos += vm.len[i];
		}
	}
	t->off[n] = (uint32_t)pos;
	err = load_spec(t);
	if (err == NULL && t->max_digits == 3 && !piece_index(t))
		err = "out of memory";

done:
	free(vm.slot);
	return err;
}

static int
out_push(struct tokenizer *t, int32_t id)
{
	if (t->nout == t->ocap &&
	    !grow(&t->out, &t->ocap, t->nout + 1, sizeof(*t->out)))
		return 0;
	t->out[t->nout++] = id;
	return 1;
}

static void
heap_push(struct heapent *h, size_t *n, struct heapent e)
{
	size_t i = (*n)++, p;

	while (i > 0) {
		p = (i - 1) / 2;
		if (h[p].key <= e.key)
			break;
		h[i] = h[p];
		i = p;
	}
	h[i] = e;
}

static struct heapent
heap_pop(struct heapent *h, size_t *n)
{
	struct heapent top = h[0], last = h[--(*n)];
	size_t i = 0, c;

	while ((c = 2 * i + 1) < *n) {
		if (c + 1 < *n && h[c + 1].key < h[c].key)
			c++;
		if (last.key <= h[c].key)
			break;
		h[i] = h[c];
		i = c;
	}
	h[i] = last;
	return top;
}

static void
add_pair(const struct tokenizer *t, struct heapent *h, size_t *nh,
    const struct sym *s, int32_t l)
{
	struct heapent e;
	uint32_t rank;
	int32_t r;

	if (l < 0 || (r = s[l].next) < 0 || s[l].id < 0 || s[r].id < 0)
		return;
	if (merge_find(t, s[l].id, s[r].id, &rank) < 0)
		return;
	e.key = ((uint64_t)rank << 32) | (uint32_t)l;
	e.lid = s[l].id;
	e.rid = s[r].id;
	heap_push(h, nh, e);
}

/* Room for n symbols and their pair queue. */
static int
grow_syms(struct tokenizer *t, size_t n)
{
	size_t c1, c2;

	if (n > INT32_MAX / 4)
		return 0;
	if (n > t->wcap) {
		c1 = c2 = t->wcap;
		if (!grow(&t->syms, &c1, n, sizeof(*t->syms)) ||
		    !grow(&t->heap, &c2, 3 * n, sizeof(*t->heap)))
			return 0;
		t->wcap = c1 < c2 / 3 ? c1 : c2 / 3;
	}
	return 1;
}

/*
 * BPE over the n symbols in syms. Merge the lowest rank first, then the
 * leftmost pair. A queued pair is stale if either side changed. A
 * symbol with id -1 takes part in no merge.
 */
static void
bpe_merge(struct tokenizer *t, size_t n)
{
	struct sym *s = t->syms;
	struct heapent *h = t->heap, e;
	size_t i, nh = 0;
	uint32_t rank;
	int32_t l, r, id;

	for (i = 0; i < n; i++) {
		s[i].prev = (int32_t)i - 1;
		s[i].next = i + 1 < n ? (int32_t)i + 1 : -1;
	}
	for (i = 0; i + 1 < n; i++)
		add_pair(t, h, &nh, s, (int32_t)i);
	while (nh > 0) {
		e = heap_pop(h, &nh);
		l = (int32_t)(uint32_t)e.key;
		r = s[l].next;
		if (s[l].id != e.lid || r < 0 || s[r].id != e.rid)
			continue;
		if ((id = merge_find(t, e.lid, e.rid, &rank)) < 0)
			continue;
		s[l].id = id;
		s[l].next = s[r].next;
		if (s[r].next >= 0)
			s[s[r].next].prev = l;
		s[r].id = -1;
		add_pair(t, h, &nh, s, s[l].prev);
		add_pair(t, h, &nh, s, l);
	}
}

/* Byte-level BPE over one pre-token. */
static int
bpe_word(struct tokenizer *t, const unsigned char *w, size_t n)
{
	size_t i;
	int32_t l, id;

	if (n == 1)
		return out_push(t, t->byte_id[w[0]]);
	if (t->phash != NULL && (id = piece_lookup(t, w, n)) >= 0)
		return out_push(t, id);
	if (!grow_syms(t, n))
		return 0;
	for (i = 0; i < n; i++)
		t->syms[i].id = t->byte_id[w[i]];
	bpe_merge(t, n);
	for (l = 0; l >= 0; l = t->syms[l].next)
		if (!out_push(t, t->syms[l].id))
			return 0;
	return 1;
}

/*
 * gemma4 BPE over code points a to b of the decoded text s. Symbols
 * start as characters. A character with no token becomes <0xXX> byte
 * tokens, and a byte with no token is dropped, as in llama.cpp.
 */
static int
bpe_chars(struct tokenizer *t, const unsigned char *s, size_t a, size_t b)
{
	size_t n = b - a, i;
	uint32_t j;
	int32_t l;

	if (!grow_syms(t, n))
		return 0;
	for (i = 0; i < n; i++)
		t->syms[i].id = char_lookup(t, t->cp[a + i]);
	bpe_merge(t, n);
	for (l = 0; l >= 0; l = t->syms[l].next) {
		if (t->syms[l].id >= 0) {
			if (!out_push(t, t->syms[l].id))
				return 0;
			continue;
		}
		for (j = t->co[a + l]; j < t->co[a + l + 1]; j++)
			if (t->byte_id[s[j]] >= 0 &&
			    !out_push(t, t->byte_id[s[j]]))
				return 0;
	}
	return 1;
}

static int
grow_cp(struct tokenizer *t, size_t n)
{
	size_t c1 = t->ccap, c2 = t->ccap, c3 = t->ccap;

	if (n <= t->ccap)
		return 1;
	if (!grow(&t->cp, &c1, n, sizeof(*t->cp)) ||
	    !grow(&t->fl, &c2, c1, sizeof(*t->fl)) ||
	    !grow(&t->co, &c3, c1, sizeof(*t->co)))
		return 0;
	t->ccap = c1;
	return 1;
}

/*
 * Decode text into code points, flags and byte offsets. Invalid or
 * non-canonical UTF-8 becomes canonical text in sbuf, as in llama.cpp.
 * Returns the code point count, or -1 on memory failure.
 */
static long
decode_text(struct tokenizer *t, const unsigned char **ps, size_t *pn)
{
	const unsigned char *s = *ps;
	size_t n = *pn, i, l, k, m = 0;
	unsigned char buf[4];
	uint32_t c;
	int pass;

	for (pass = 0; pass < 2; pass++) {
		if (!grow_cp(t, n + 1))
			return -1;
		m = 0;
		for (i = 0; i < n; i += l) {
			l = utf8_decode(s + i, n - i, &c);
			if (l == 0 || c > 0x10ffff || utf8_encode(c, buf) != l)
				break;
			t->cp[m] = c;
			t->fl[m] = uni_flags(c);
			t->co[m] = (uint32_t)i;
			m++;
		}
		if (i >= n)
			break;
		if (pass == 1 || n > SIZE_MAX / 4 ||
		    !grow(&t->sbuf, &t->scap, 3 * n, 1))
			return -1;
		/* One invalid byte becomes U+FFFD, three bytes. */
		k = 0;
		for (i = 0; i < n; i += l) {
			l = utf8_decode(s + i, n - i, &c);
			if (l == 0 || c > 0x10ffff) {
				l = 1;
				c = 0xfffd;
			}
			k += utf8_encode(c, t->sbuf + k);
		}
		s = t->sbuf;
		n = k;
		if (n > UINT32_MAX - 1)
			return -1;
	}
	t->co[m] = (uint32_t)n;
	*ps = s;
	*pn = n;
	return (long)m;
}

/*
 * gemma4 pre-tokenizer: spaces become U+2581, then the regex
 * [^\n]+|[\n]+ splits the text. A newline run that is a token is
 * used whole.
 */
static int
encode_gemma4(struct tokenizer *t, const unsigned char *s, size_t n)
{
	size_t i, k, np, pos, end;
	long m;
	int nl;

	for (i = k = 0; i < n; i++)
		k += s[i] == ' ' ? 3 : 1;
	if (n > UINT32_MAX / 4 || !grow(&t->ebuf, &t->ecap, k + 1, 1))
		return 0;
	for (i = k = 0; i < n; i++) {
		if (s[i] == ' ') {
			memcpy(t->ebuf + k, "\xe2\x96\x81", 3);
			k += 3;
		} else
			t->ebuf[k++] = s[i];
	}
	s = t->ebuf;
	n = k;
	if ((m = decode_text(t, &s, &n)) < 0)
		return 0;
	np = (size_t)m;
	for (pos = 0; pos < np; pos = end) {
		nl = t->cp[pos] == '\n';
		for (end = pos + 1; end < np && (t->cp[end] == '\n') == nl; end++)
			;
		k = end - pos;
		if (nl && k <= MAX_NL && t->nl_id[k] >= 0) {
			if (!out_push(t, t->nl_id[k]))
				return 0;
		} else if (!bpe_chars(t, s, pos, end))
			return 0;
	}
	return 1;
}

#define OOR	0xffffffffU

static uint32_t
lower_ascii(uint32_t c)
{
	return c >= 'A' && c <= 'Z' ? c + 32 : c;
}

/*
 * The GPT-2 pre-tokenizer as a scanner, the same as llama.cpp
 * unicode_regex_split_custom_gpt2. Hugging Face ByteLevel with
 * use_regex uses this regex:
 * 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|
 * \s+(?!\S)|\s+
 */
static int
encode_gpt2(struct tokenizer *t, const unsigned char *s, size_t n)
{
	const uint32_t *cp;
	const uint16_t *fl;
	uint32_t c, c1, c2;
	uint16_t f2;
	size_t pos, prev, np, nws;
	long m;

	if (n > UINT32_MAX / 4)
		return 0;
	if ((m = decode_text(t, &s, &n)) < 0)
		return 0;
	np = (size_t)m;
	cp = t->cp;
	fl = t->fl;

#define FLG(p)	((p) < np ? fl[p] : 0)
#define EMIT(end) do {							\
	if ((end) > prev && !bpe_word(t, s + t->co[prev],		\
	    t->co[end] - t->co[prev]))					\
		return 0;						\
	prev = (end);							\
} while (0)

	pos = prev = 0;
	while (pos < np) {
		c = cp[pos];
		if (c == '\'' && pos + 1 < np) {
			c1 = cp[pos + 1];
			if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
				pos += 2;
				EMIT(pos);
				continue;
			}
			if (pos + 2 < np) {
				c2 = cp[pos + 2];
				if ((c1 == 'r' && c2 == 'e') ||
				    (c1 == 'v' && c2 == 'e') ||
				    (c1 == 'l' && c2 == 'l')) {
					pos += 3;
					EMIT(pos);
					continue;
				}
			}
		}
		f2 = c == ' ' ? FLG(pos + 1) : fl[pos];
		if (f2 & UF_LETTER) {
			pos += c == ' ';
			while (FLG(pos) & UF_LETTER)
				pos++;
			EMIT(pos);
			continue;
		}
		if (f2 & UF_NUMBER) {
			pos += c == ' ';
			while (FLG(pos) & UF_NUMBER)
				pos++;
			EMIT(pos);
			continue;
		}
		if (!(f2 & (UF_WS | UF_LETTER | UF_NUMBER)) && f2 != 0) {
			pos += c == ' ';
			while (!(FLG(pos) & (UF_WS | UF_LETTER | UF_NUMBER)) && FLG(pos) != 0)
				pos++;
			EMIT(pos);
			continue;
		}
		nws = 0;
		while (FLG(pos + nws) & UF_WS)
			nws++;
		if (nws > 1 && pos + nws < np)
			pos += nws - 1;
		else
			pos += nws > 0 ? nws : 1;
		EMIT(pos);
	}
#undef FLG
#undef EMIT
	return 1;
}

/*
 * The qwen2 and llama3 pre-tokenizers as a scanner, the same as llama.cpp
 * unicode_regex_split_custom_qwen2 and _llama3. llama3 takes \p{N}{1,3}
 * where qwen2 takes \p{N}. qwen35 counts \p{M} as \p{L}. Regex:
 * (?i:'s|'t|'re|'ve|'m|'ll|'d)|
 * [^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|
 * \s+(?!\S)|\s+
 */
static int
encode_text(struct tokenizer *t, const unsigned char *s, size_t n)
{
	const uint32_t *cp;
	const uint16_t *fl;
	uint32_t c, c1, c2;
	uint16_t f, f2;
	size_t pos, prev, np, nws, last;
	long m;

	if (t->gemma4)
		return encode_gemma4(t, s, n);
	if (t->gpt2)
		return encode_gpt2(t, s, n);
	if (n > UINT32_MAX / 4)
		return 0;
	if ((m = decode_text(t, &s, &n)) < 0)
		return 0;
	np = (size_t)m;
	cp = t->cp;
	fl = t->fl;

#define CPT(p)	((p) < np ? cp[p] : OOR)
#define FLG(p)	((p) < np ? fl[p] : 0)
#define EMIT(end) do {							\
	if ((end) > prev && !bpe_word(t, s + t->co[prev],		\
	    t->co[end] - t->co[prev]))					\
		return 0;						\
	prev = (end);							\
} while (0)

	pos = prev = 0;
	while (pos < np) {
		c = cp[pos];
		f = fl[pos];

		if (c == '\'' && pos + 1 < np) {
			c1 = lower_ascii(cp[pos + 1]);
			if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
				pos += 2;
				EMIT(pos);
				continue;
			}
			if (pos + 2 < np) {
				c2 = lower_ascii(cp[pos + 2]);
				if ((c1 == 'r' && c2 == 'e') ||
				    (c1 == 'v' && c2 == 'e') ||
				    (c1 == 'l' && c2 == 'l')) {
					pos += 3;
					EMIT(pos);
					continue;
				}
			}
		}

		if (!(c == '\r' || c == '\n' || (f & UF_NUMBER)) &&
		    ((f & t->letters) || (FLG(pos + 1) & t->letters))) {
			pos++;
			while (FLG(pos) & t->letters)
				pos++;
			EMIT(pos);
			continue;
		}

		if (f & UF_NUMBER) {
			last = pos + (size_t)t->max_digits;
			pos++;
			while (pos < last && (FLG(pos) & UF_NUMBER))
				pos++;
			EMIT(pos);
			continue;
		}

		f2 = c == ' ' ? FLG(pos + 1) : f;
		if (!(f2 & (UF_WS | t->letters | UF_NUMBER)) && f != 0) {
			pos += c == ' ';
			while (!(f2 & (UF_WS | t->letters | UF_NUMBER)) && f2 != 0) {
				pos++;
				f2 = FLG(pos);
			}
			c2 = CPT(pos);
			while (c2 == '\r' || c2 == '\n') {
				pos++;
				c2 = CPT(pos);
			}
			EMIT(pos);
			continue;
		}

		nws = 0;
		last = 0;
		while (FLG(pos + nws) & UF_WS) {
			c2 = cp[pos + nws];
			if (c2 == '\r' || c2 == '\n')
				last = pos + nws + 1;
			nws++;
		}
		if (last > 0)
			pos = last;
		else if (nws > 1 && pos + nws < np)
			pos += nws - 1;
		else
			pos += nws > 0 ? nws : 1;
		EMIT(pos);
	}
#undef CPT
#undef FLG
#undef EMIT
	return 1;
}

/* Lua API */

static struct tokenizer *
check_tok(lua_State *L, int idx)
{
	struct tokenizer *t = luaL_checkudata(L, idx, TOKENIZER_MT);

	if (t->closed)
		luaL_error(L, "tokenizer is closed");
	return t;
}

static int
opt_bool(lua_State *L, int idx, const char *key, int def)
{
	int r = def;

	if (lua_istable(L, idx)) {
		if (lua_getfield(L, idx, key) != LUA_TNIL)
			r = lua_toboolean(L, -1);
		lua_pop(L, 1);
	}
	return r;
}

/* The most bytes of a tokenizer.json. */
#define JSON_MAX_FILE	((long)256 << 20)

/* True when the file at path starts with '{' after white space. */
static int
is_json_file(const char *path)
{
	FILE *f;
	int c;

	if ((f = fopen(path, "rb")) == NULL)
		return 0;
	while ((c = getc(f)) == ' ' || c == '\t' || c == '\r' || c == '\n')
		;
	fclose(f);
	return c == '{';
}

/* Read the file into a new userdata on the stack. Returns NULL on error. */
static const char *
read_json_file(lua_State *L, const char *path, size_t *lenp)
{
	char *buf;
	FILE *f;
	long sz;
	int ok;

	if ((f = fopen(path, "rb")) == NULL)
		return NULL;
	if (fseek(f, 0, SEEK_END) != 0 || (sz = ftell(f)) <= 0 || sz > JSON_MAX_FILE ||
	    fseek(f, 0, SEEK_SET) != 0) {
		fclose(f);
		return NULL;
	}
	buf = lua_newuserdatauv(L, (size_t)sz, 0);
	ok = fread(buf, 1, (size_t)sz, f) == (size_t)sz;
	fclose(f);
	if (!ok) {
		lua_pop(L, 1);
		return NULL;
	}
	*lenp = (size_t)sz;
	return buf;
}

/*** Load a BPE tokenizer from gguf metadata or a tokenizer.json.
 * From gguf, it supports the gpt2 model with the gpt2, qwen2,
 * deepseek-r1-qwen, llama-bpe, llama3, bailingmoe, bailingmoe2 or
 * llada-moe pre-tokenizer, and the gemma4 model, as llama.cpp does.
 * opts.model and opts.pre stand in for keys the file does not have.
 * The lgml key `tokenizer.ggml.normalizer.nfc` turns on NFC.
 *
 * A file that starts with `{` is a Hugging Face tokenizer.json. It
 * needs a BPE model without byte fallback, the ByteLevel pre-tokenizer
 * with its GPT-2 regex or a known Split regex followed by ByteLevel,
 * and no normalizer or NFC. Added tokens split
 * first: special ones are control tokens, others user tokens.
 * `lstrip` and `rstrip` take ASCII white space. The first and last
 * special tokens of a TemplateProcessing post-processor are bos and
 * eos; `encode` never adds them.
 * @tparam string path gguf file or tokenizer.json
 * @tparam[opt] table opts fallback model and pre names (gguf only)
 * @treturn[1] tokenizer the tokenizer
 * @return[2] nil
 * @treturn[2] string error message
 * @raise on a tokenizer.json that is not valid JSON
 * @function tokenizer
 */
static int
l_tokenizer(lua_State *L)
{
	const char *path = luaL_checkstring(L, 1);
	struct gguf_init_params gp = { true, NULL };
	struct gguf_context *c;
	struct tokenizer *t;
	struct tok_src src;
	const char *err, *buf, *def_model = NULL, *def_pre = NULL;
	size_t len;

	/* opts.model and opts.pre stand in for keys the file lacks. */
	if (lua_istable(L, 2)) {
		lua_getfield(L, 2, "model");
		def_model = lua_tostring(L, -1);
		lua_getfield(L, 2, "pre");
		def_pre = lua_tostring(L, -1);
		lua_pop(L, 2);
	}

	t = lua_newuserdatauv(L, sizeof(*t), 0);
	memset(t, 0, sizeof(*t));
	t->closed = 1;
	luaL_setmetatable(L, TOKENIZER_MT);
	memset(&src, 0, sizeof(src));
	if (is_json_file(path)) {
		if ((buf = read_json_file(L, path, &len)) == NULL) {
			luaL_pushfail(L);
			lua_pushfstring(L, "%s: cannot read the file", path);
			return 2;
		}
		lgml_json_decode(L, buf, len, path);
		lua_remove(L, -2);
		if (!lua_istable(L, -1))
			err = "not a JSON object";
		else if ((err = src_json(L, lua_gettop(L), &src)) == NULL) {
			t->closed = 0;
			err = tok_load(t, &src);
		}
		src_free(&src);
		lua_pop(L, 1);
	} else {
		if ((c = gguf_init_from_file(path, gp)) == NULL) {
			luaL_pushfail(L);
			lua_pushfstring(L, "%s: not a gguf file", path);
			return 2;
		}
		if ((err = src_gguf(&src, c, def_model, def_pre)) == NULL) {
			t->closed = 0;
			err = tok_load(t, &src);
		}
		src_free(&src);
		gguf_free(c);
	}
	if (err != NULL) {
		if (!t->closed)
			tok_free(t);
		luaL_pushfail(L);
		lua_pushfstring(L, "%s: %s", path, err);
		return 2;
	}
	return 1;
}

static int
is_ascii_space(unsigned char c)
{
	return c == ' ' || (c >= '\t' && c <= '\r');
}

/*
 * Split out special tokens, leftmost and then longest first. Without
 * opts.special, only user-defined tokens split, as in llama.cpp. A
 * token with STRIP_LEFT or STRIP_RIGHT takes the white space next to it.
 */
static int
encode_all(struct tokenizer *t, const unsigned char *s, size_t n,
    int special)
{
	size_t pos = 0, seg = 0, sl = 0, end_seg;
	uint32_t k, end;
	int32_t id = -1;
	int strip;

	while (pos < n) {
		end = t->sfirst[s[pos] + 1];
		for (k = t->sfirst[s[pos]]; k < end; k++) {
			id = t->spec[k];
			sl = t->off[id + 1] - t->off[id];
			if (sl <= n - pos &&
			    (special || t->type[id] == TT_USER) &&
			    memcmp(s + pos, t->pool + t->off[id], sl) == 0)
				break;
		}
		if (k == end) {
			pos++;
			continue;
		}
		strip = t->strip != NULL ? t->strip[id] : 0;
		end_seg = pos;
		while ((strip & STRIP_LEFT) && end_seg > seg && is_ascii_space(s[end_seg - 1]))
			end_seg--;
		if (end_seg > seg && !encode_text(t, s + seg, end_seg - seg))
			return 0;
		if (!out_push(t, id))
			return 0;
		pos += sl;
		while ((strip & STRIP_RIGHT) && pos < n && is_ascii_space(s[pos]))
			pos++;
		seg = pos;
	}
	return n > seg ? encode_text(t, s + seg, n - seg) : 1;
}

/*
 * NFC of s with utf8proc. Returns a malloc'ed copy in *out, or NULL
 * when s stays as it is: text with no code point at or above U+0300 is
 * NFC, and text that is not valid UTF-8 is not changed.
 */
static int
nfc(const unsigned char *s, size_t n, unsigned char **out, size_t *outn)
{
	utf8proc_uint8_t *dst = NULL;
	utf8proc_ssize_t r;
	size_t i;

	*out = NULL;
	*outn = n;
	/* Lead bytes below 0xcc encode only code points below U+0300. */
	for (i = 0; i < n && s[i] < 0xcc; i++)
		;
	if (i == n || n > SIZE_MAX / 8)
		return 1;
	r = utf8proc_map(s, (utf8proc_ssize_t)n, &dst, UTF8PROC_STABLE | UTF8PROC_COMPOSE);
	if (r == UTF8PROC_ERROR_NOMEM)
		return 0;
	if (r < 0)
		return 1;
	*out = dst;
	*outn = (size_t)r;
	return 1;
}

/*** Normalize text to Unicode NFC, with utf8proc.
 * Text that is not valid UTF-8 comes back as it is.
 * @tparam string s the text
 * @treturn string the NFC text
 * @raise if memory runs out
 * @function nfc
 */
static int
l_nfc(lua_State *L)
{
	const unsigned char *s;
	unsigned char *o;
	size_t n, on;

	s = (const unsigned char *)luaL_checklstring(L, 1, &n);
	if (!nfc(s, n, &o, &on))
		return luaL_error(L, "nfc: out of memory");
	if (o == NULL) {
		lua_settop(L, 1);
		return 1;
	}
	lua_pushlstring(L, (const char *)o, on);
	free(o);
	return 1;
}

/*** Encodes text to token ids and back.
 * @type tokenizer
 */

/*** Encode text to token ids.
 * A tokenizer.json with the NFC normalizer normalizes the text first.
 * Special tokens split first, leftmost and then longest. With
 * opts.special false, only user-defined tokens split.
 * @tparam string text UTF-8 text
 * @tparam[opt] table opts special (default true), `add_bos` (default from
 * the file)
 * @treturn {int,...} token ids, from 0
 * @raise if memory runs out
 * @function tokenizer:encode
 */
static int
l_encode(lua_State *L)
{
	struct tokenizer *t = check_tok(L, 1);
	const unsigned char *s;
	unsigned char *norm = NULL;
	size_t n, i;
	int special, add_bos, ok;

	s = (const unsigned char *)luaL_checklstring(L, 2, &n);
	special = opt_bool(L, 3, "special", 1);
	add_bos = opt_bool(L, 3, "add_bos", t->add_bos);

	t->nout = 0;
	if (add_bos && t->bos >= 0 && !out_push(t, t->bos))
		return luaL_error(L, "encode: out of memory");
	if (t->nfc && !nfc(s, n, &norm, &n))
		return luaL_error(L, "encode: out of memory");
	ok = encode_all(t, norm != NULL ? norm : s, n, special);
	free(norm);
	if (!ok)
		return luaL_error(L, "encode: out of memory");
	if (t->nout > INT_MAX)
		return luaL_error(L, "encode: too many tokens");
	lua_createtable(L, (int)t->nout, 0);
	for (i = 0; i < t->nout; i++) {
		lua_pushinteger(L, t->out[i]);
		lua_rawseti(L, -2, (lua_Integer)i + 1);
	}
	return 1;
}

/*** Decode token ids to text.
 * Unused tokens never decode. With opts.special false, control and
 * unknown tokens do not decode.
 * @tparam {int,...} ids token ids, from 0
 * @tparam[opt] table opts special (default true)
 * @treturn string bytes of the text
 * @raise on an id that is out of range
 * @function tokenizer:decode
 */
static int
l_decode(lua_State *L)
{
	struct tokenizer *t = check_tok(L, 1);
	lua_Integer i, n, id;
	luaL_Buffer b;
	int special, isnum;

	luaL_checktype(L, 2, LUA_TTABLE);
	special = opt_bool(L, 3, "special", 1);
	n = luaL_len(L, 2);
	luaL_buffinit(L, &b);
	for (i = 1; i <= n; i++) {
		lua_rawgeti(L, 2, i);
		id = lua_tointegerx(L, -1, &isnum);
		lua_pop(L, 1);
		if (!isnum || id < 0 || id >= t->n_vocab)
			return luaL_error(L, "decode: bad token id at index %d",
			    (int)i);
		if (t->type[id] == TT_UNUSED || (!special &&
		    (t->type[id] == TT_CONTROL || t->type[id] == TT_UNKNOWN)))
			continue;
		luaL_addlstring(&b, t->pool + t->off[id],
		    t->off[id + 1] - t->off[id]);
	}
	luaL_pushresult(&b);
	return 1;
}

/*** Get the text and type of one token.
 * @tparam int id token id, from 0
 * @treturn string token bytes
 * @treturn string type: "normal", "unknown", "control", "user", "unused",
 * "byte" or "undefined"
 * @raise if the id is out of range
 * @function tokenizer:token
 */
static int
l_token(lua_State *L)
{
	struct tokenizer *t = check_tok(L, 1);
	lua_Integer id = luaL_checkinteger(L, 2);

	luaL_argcheck(L, id >= 0 && id < t->n_vocab, 2, "bad token id");
	lua_pushlstring(L, t->pool + t->off[id], t->off[id + 1] - t->off[id]);
	lua_pushstring(L, tt_names[t->type[id]]);
	return 2;
}

/*** Get the vocabulary size.
 * @treturn int number of tokens
 * @function tokenizer:n_vocab
 */
static int
l_n_vocab(lua_State *L)
{
	struct tokenizer *t = check_tok(L, 1);

	lua_pushinteger(L, t->n_vocab);
	return 1;
}

static void
set_id(lua_State *L, const char *key, int32_t id)
{
	if (id >= 0) {
		lua_pushinteger(L, id);
		lua_setfield(L, -2, key);
	}
}

/*** Get the special tokens.
 * The result is { bos, eos, eot, pad, unk, `add_bos`, eog, tokens }. An id
 * field is absent when the file has no such token. eog lists the ids that
 * end generation. tokens maps the text of each special token to its id.
 * @treturn table special tokens
 * @function tokenizer:special
 */
static int
l_special(lua_State *L)
{
	struct tokenizer *t = check_tok(L, 1);
	int32_t i, id;

	lua_createtable(L, 0, 8);
	set_id(L, "bos", t->bos);
	set_id(L, "eos", t->eos);
	set_id(L, "eot", t->eot);
	set_id(L, "pad", t->pad);
	set_id(L, "unk", t->unk);
	lua_pushboolean(L, t->add_bos);
	lua_setfield(L, -2, "add_bos");
	lua_createtable(L, t->neog, 0);
	for (i = 0; i < t->neog; i++) {
		lua_pushinteger(L, t->eog[i]);
		lua_rawseti(L, -2, i + 1);
	}
	lua_setfield(L, -2, "eog");
	lua_createtable(L, 0, t->nspec);
	for (i = 0; i < t->nspec; i++) {
		id = t->spec[i];
		lua_pushlstring(L, t->pool + t->off[id],
		    t->off[id + 1] - t->off[id]);
		lua_pushinteger(L, id);
		lua_rawset(L, -3);
	}
	lua_setfield(L, -2, "tokens");
	return 1;
}

/*** Free the tokenizer tables.
 * The `__gc` and `__close` metamethods call this function.
 * @function tokenizer:close
 */
static int
l_close(lua_State *L)
{
	struct tokenizer *t = luaL_checkudata(L, 1, TOKENIZER_MT);

	if (!t->closed)
		tok_free(t);
	return 0;
}

static const luaL_Reg tok_meta[] = {
	{ "__gc", l_close },
	{ "__close", l_close },
	{ NULL, NULL }
};

static const luaL_Reg tok_methods[] = {
	{ "encode", l_encode },
	{ "decode", l_decode },
	{ "token", l_token },
	{ "n_vocab", l_n_vocab },
	{ "special", l_special },
	{ "close", l_close },
	{ NULL, NULL }
};

void
lgml_open_tokenize(lua_State *L)
{
	uni_init();
	byte_map_init();
	luaL_newmetatable(L, TOKENIZER_MT);
	luaL_setfuncs(L, tok_meta, 0);
	lua_newtable(L);
	luaL_setfuncs(L, tok_methods, 0);
	lua_setfield(L, -2, "__index");
	lua_pop(L, 1);
	lua_pushcfunction(L, l_tokenizer);
	lua_setfield(L, -2, "tokenizer");
	lua_pushcfunction(L, l_nfc);
	lua_setfield(L, -2, "nfc");
}
