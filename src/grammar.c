/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * GBNF grammars, as in llama.cpp. The parser builds flat rule tables.
 * The matcher is a pushdown automaton over UTF-8 code points: it keeps
 * the set of parse stacks, and each stack ends at a terminal element.
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lua.h>
#include <lauxlib.h>

#include "grammar.h"

#define GRAMMAR_MT	"lgml_grammar"
#define GVOCAB_MT	"lgml_gvocab"
#define GVOCAB_CACHE	"lgml.gvocab"

/* Bounds for hostile grammars. */
#define MAX_RULES	16384
#define MAX_ELEMS	(1 << 20)
#define MAX_NEST	64
#define MAX_REPEAT	2000		/* as llama.cpp */
#define MAX_DEPTH	1024		/* elements in one stack */
#define MAX_STACKS	4096
#define MAX_SEEN	(1 << 16)
#define MAX_WORK	(1 << 24)	/* steps and stack words per call */
#define MAX_TOKCP	1024		/* code points in one token */

#define NO_TOKEN	UINT32_MAX
#define NO_MAX		UINT64_MAX

enum {
	G_END,		/* end of a rule */
	G_ALT,		/* start of another alternative */
	G_REF,		/* rule reference */
	G_CHAR,		/* code point, or start of a class */
	G_NOT,		/* start of a negated class */
	G_UPPER,	/* upper end of a range */
	G_CALT,		/* another code point in a class */
	G_ANY,		/* any code point */
	G_TOK,		/* token id */
	G_TOKNOT	/* any token but the id */
};

struct gel {
	uint32_t	 type;
	uint32_t	 value;
};

struct gvec {
	struct gel	*v;
	size_t		 n;
	size_t		 cap;
};

struct gcand;

struct scratch {
	struct gcand	*p;
	size_t		 cap;
};

/* A set of stacks. Each stack is a list of element indexes. */
struct sref {
	uint32_t	 off;
	uint32_t	 len;
	uint32_t	 hash;
	uint32_t	 slot;
};

struct sset {
	uint32_t	*pool;
	size_t		 np;
	size_t		 capp;
	struct sref	*v;
	size_t		 n;
	size_t		 cap;
	uint32_t	*ht;	/* index + 1; 0 is empty */
	size_t		 htcap;
};

struct grammar {
	struct gel	*el;
	uint32_t	 nel;
	uint32_t	*roff;	/* first element of each rule */
	uint32_t	 nrules;

	struct sset	 cur;
	struct sset	 start;	/* cur after grammar_new */
	struct sset	 nxt;
	struct sset	 a;
	struct sset	 b;
	struct sset	 seen;
	uint32_t	*work;
	size_t		 nwork;
	size_t		 capwork;
	uint32_t	*sbuf;	/* stack under construction in advance */
	uint32_t	*abuf;	/* input stack for advance */
	struct scratch	*scr;	/* candidate buffers by depth */
	struct gcand	*cbuf;
	struct gcand	*rbuf;
	uint32_t	*cpv;
	int8_t		*cnr;
	size_t		 ccap;
	const uint32_t	*cpbase;
	const int32_t	*cids;
	uint64_t	 budget;

	uint32_t	 pv;	/* partial UTF-8: value so far */
	int		 nr;	/* partial UTF-8: bytes left, or -1 */

	char		 err[160];
};

struct gparse {
	const char		*src;
	const struct gvocab	*voc;
	int			 lax;
	char			**names;
	struct gvec		*rules;
	uint32_t		 nsym;
	uint32_t		 capsym;
	uint32_t		*ht;	/* symbol + 1 by name hash */
	uint32_t		 htcap;
	size_t			 nel;
	int			 depth;
	char			 err[160];
};

/*
 * A token on its way through the matcher: its next code point, at
 * g->cpbase[cp], and its index in the checked list. A 0 code point ends
 * the token; g->cnr and g->cpv hold its partial UTF-8 state.
 */
struct gcand {
	uint32_t	 cp;
	uint32_t	 idx;
};

static int	spend(struct grammar *, uint64_t);
static int	advance(struct grammar *, const uint32_t *, uint32_t,
		    struct sset *);
static int	reject_all(struct grammar *, unsigned, struct sset *,
		    const struct gcand *, size_t, struct gcand *, size_t *);

static void	set_err(char *, size_t, const char *, ...)
		    __attribute__((__format__(__printf__, 3, 4)));

static void
set_err(char *err, size_t n, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(err, n, fmt, ap);
	va_end(ap);
}

static uint32_t
hash_words(const uint32_t *w, uint32_t n)
{
	uint32_t h = 2166136261u ^ n;
	uint32_t i;

	for (i = 0; i < n; i++)
		h = (h ^ w[i]) * 16777619u;
	return h;
}

static uint32_t
hash_bytes(const char *s, size_t n)
{
	uint32_t h = 2166136261u;
	size_t i;

	for (i = 0; i < n; i++)
		h = (h ^ (unsigned char)s[i]) * 16777619u;
	return h;
}

/* realloc for n elements of size sz, with an overflow check. */
static void *
resize(void *p, size_t n, size_t sz)
{
	if (sz != 0 && n > SIZE_MAX / sz)
		return NULL;
	return realloc(p, n * sz);
}

/* Grow *pp to hold at least n elements of size sz. 0 on success. */
static int
grow(void *pp, size_t *cap, size_t n, size_t sz)
{
	void **p = pp;
	size_t c = *cap ? *cap : 16;
	void *q;

	if (n <= *cap)
		return 0;
	while (c < n)
		c *= 2;
	if ((q = resize(*p, c, sz)) == NULL)
		return -1;
	*p = q;
	*cap = c;
	return 0;
}

/* Stack sets */

static void
ss_free(struct sset *s)
{
	free(s->pool);
	free(s->v);
	free(s->ht);
	memset(s, 0, sizeof(*s));
}

static void
ss_clear(struct sset *s)
{
	size_t i;

	for (i = 0; i < s->n; i++)
		s->ht[s->v[i].slot] = 0;
	s->n = 0;
	s->np = 0;
}

static void
ss_swap(struct sset *a, struct sset *b)
{
	struct sset t = *a;

	*a = *b;
	*b = t;
}

static const uint32_t *
ss_get(const struct sset *s, size_t i, uint32_t *len)
{
	*len = s->v[i].len;
	return s->pool + s->v[i].off;
}

static int
ss_rehash(struct sset *s, size_t cap)
{
	uint32_t *ht;
	size_t i, j;

	if ((ht = calloc(cap, sizeof(*ht))) == NULL)
		return -1;
	for (i = 0; i < s->n; i++) {
		j = s->v[i].hash & (cap - 1);
		while (ht[j] != 0)
			j = (j + 1) & (cap - 1);
		ht[j] = (uint32_t)i + 1;
		s->v[i].slot = (uint32_t)j;
	}
	free(s->ht);
	s->ht = ht;
	s->htcap = cap;
	return 0;
}

/*
 * Add a stack. Returns its index when new, -2 when the set has it, and
 * -1 on error. e must not point into s.
 */
static long
ss_add(struct grammar *g, struct sset *s, const uint32_t *e, uint32_t len,
    size_t limit)
{
	uint32_t h = hash_words(e, len), k;
	size_t j;

	if (s->htcap == 0 || (s->n + 1) * 2 > s->htcap) {
		if (ss_rehash(s, s->htcap ? s->htcap * 2 : 64) == -1)
			goto nomem;
	}
	for (j = h & (s->htcap - 1); (k = s->ht[j]) != 0;
	    j = (j + 1) & (s->htcap - 1)) {
		k--;
		if (s->v[k].hash == h && s->v[k].len == len &&
		    memcmp(s->pool + s->v[k].off, e, len * sizeof(*e)) == 0)
			return -2;
	}
	if (s->n >= limit) {
		set_err(g->err, sizeof(g->err), "too many parse stacks");
		return -1;
	}
	if (spend(g, len + 1) == -1)
		return -1;
	if (grow(&s->v, &s->cap, s->n + 1, sizeof(*s->v)) == -1 ||
	    grow(&s->pool, &s->capp, s->np + len + 1, sizeof(*s->pool)) == -1)
		goto nomem;
	memcpy(s->pool + s->np, e, len * sizeof(*e));
	s->v[s->n].off = (uint32_t)s->np;
	s->v[s->n].len = len;
	s->v[s->n].hash = h;
	s->v[s->n].slot = (uint32_t)j;
	s->ht[j] = (uint32_t)s->n + 1;
	s->np += len;
	return (long)s->n++;
nomem:
	set_err(g->err, sizeof(g->err), "out of memory");
	return -1;
}

/* Make dst a copy of src. */
static int
copy_set(struct grammar *g, struct sset *dst, const struct sset *src)
{
	const uint32_t *stk;
	uint32_t len;
	size_t i;

	ss_clear(dst);
	for (i = 0; i < src->n; i++) {
		stk = ss_get(src, i, &len);
		if (ss_add(g, dst, stk, len, MAX_STACKS) == -1)
			return -1;
	}
	return 0;
}

/* Parser */

static int
is_word(int c)
{
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
	    (c >= '0' && c <= '9') || c == '-';
}

static int
is_digit(int c)
{
	return c >= '0' && c <= '9';
}

static int
is_end(const struct gel *e)
{
	return e->type == G_END || e->type == G_ALT;
}

static const char *
skip_space(const char *s, int nl)
{
	while (*s == ' ' || *s == '\t' || *s == '#' ||
	    (nl && (*s == '\r' || *s == '\n'))) {
		if (*s == '#') {
			while (*s != '\0' && *s != '\r' && *s != '\n')
				s++;
		} else
			s++;
	}
	return s;
}

/* Record an error with the text at s. Returns NULL. */
static const char *
fail(struct gparse *p, const char *msg, const char *s)
{
	char at[24];
	size_t i;

	for (i = 0; i < sizeof(at) - 1 && s[i] != '\0'; i++)
		at[i] = (s[i] >= 0x20 && s[i] < 0x7f) ? s[i] : '?';
	at[i] = '\0';
	if (p->err[0] == '\0')
		set_err(p->err, sizeof(p->err), "%s at '%s'", msg, at);
	return NULL;
}

static int
push(struct gparse *p, struct gvec *r, uint32_t type, uint32_t value)
{
	if (p->nel >= MAX_ELEMS) {
		fail(p, "grammar too large", "");
		return -1;
	}
	if (grow(&r->v, &r->cap, r->n + 1, sizeof(*r->v)) == -1) {
		fail(p, "out of memory", "");
		return -1;
	}
	r->v[r->n].type = type;
	r->v[r->n].value = value;
	r->n++;
	p->nel++;
	return 0;
}

static int
push_all(struct gparse *p, struct gvec *r, const struct gel *e, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++)
		if (push(p, r, e[i].type, e[i].value) == -1)
			return -1;
	return 0;
}

static long
find_symbol(struct gparse *p, const char *s, size_t n, uint32_t *slot)
{
	uint32_t j, k;

	for (j = hash_bytes(s, n) & (p->htcap - 1); (k = p->ht[j]) != 0;
	    j = (j + 1) & (p->htcap - 1)) {
		k--;
		if (strlen(p->names[k]) == n && memcmp(p->names[k], s, n) == 0)
			return k;
	}
	*slot = j;
	return -1;
}

static int
rehash_symbols(struct gparse *p, uint32_t cap)
{
	uint32_t *ht, i, j;
	const char *nm;

	if ((ht = calloc(cap, sizeof(*ht))) == NULL)
		return -1;
	for (i = 0; i < p->nsym; i++) {
		if ((nm = p->names[i]) == NULL)
			continue;
		j = hash_bytes(nm, strlen(nm)) & (cap - 1);
		while (ht[j] != 0)
			j = (j + 1) & (cap - 1);
		ht[j] = i + 1;
	}
	free(p->ht);
	p->ht = ht;
	p->htcap = cap;
	return 0;
}

/* A new symbol. With a name that exists, the name stays with the old one. */
static long
new_symbol(struct gparse *p, const char *s, size_t n)
{
	size_t cap = p->capsym;
	uint32_t slot;
	long old;
	char *nm;

	if (p->nsym >= MAX_RULES) {
		fail(p, "too many rules", "");
		return -1;
	}
	if ((p->nsym + 1) * 2 > p->htcap &&
	    rehash_symbols(p, p->htcap ? p->htcap * 2 : 64) == -1)
		goto nomem;
	if (grow(&p->names, &cap, p->nsym + 1, sizeof(*p->names)) == -1)
		goto nomem;
	cap = p->capsym;
	if (grow(&p->rules, &cap, p->nsym + 1, sizeof(*p->rules)) == -1)
		goto nomem;
	p->capsym = (uint32_t)cap;
	if ((nm = malloc(n + 1)) == NULL)
		goto nomem;
	memcpy(nm, s, n);
	nm[n] = '\0';
	p->names[p->nsym] = nm;
	memset(&p->rules[p->nsym], 0, sizeof(p->rules[0]));
	old = find_symbol(p, s, n, &slot);
	if (old < 0)
		p->ht[slot] = p->nsym + 1;
	return p->nsym++;
nomem:
	fail(p, "out of memory", "");
	return -1;
}

static long
get_symbol(struct gparse *p, const char *s, size_t n)
{
	uint32_t slot;
	long id;

	if (p->htcap > 0 && (id = find_symbol(p, s, n, &slot)) >= 0)
		return id;
	return new_symbol(p, s, n);
}

/* A generated symbol: the base name, "_" and the symbol count. */
static long
gen_symbol(struct gparse *p, uint32_t base)
{
	char nm[96];
	int n;

	n = snprintf(nm, sizeof(nm), "%.64s_%u", p->names[base], p->nsym);
	if (n < 0 || (size_t)n >= sizeof(nm))
		n = sizeof(nm) - 1;
	return new_symbol(p, nm, (size_t)n);
}

/* Take ownership of r as the body of rule id. */
static void
add_rule(struct gparse *p, uint32_t id, struct gvec *r)
{
	free(p->rules[id].v);
	p->rules[id] = *r;
	memset(r, 0, sizeof(*r));
}

static const char *
parse_hex(struct gparse *p, const char *s, int size, uint32_t *v)
{
	int i, c;

	*v = 0;
	for (i = 0; i < size; i++) {
		c = s[i];
		if (c >= '0' && c <= '9')
			*v = (*v << 4) + (uint32_t)(c - '0');
		else if (c >= 'a' && c <= 'f')
			*v = (*v << 4) + (uint32_t)(c - 'a' + 10);
		else if (c >= 'A' && c <= 'F')
			*v = (*v << 4) + (uint32_t)(c - 'A' + 10);
		else
			return fail(p, "expecting hex digits", s);
	}
	return s + size;
}

/* Decode one UTF-8 sequence of grammar text, as llama.cpp does. */
static const char *
decode_src(const char *s, uint32_t *cp)
{
	static const int len_of[16] = {
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4
	};
	unsigned char c = (unsigned char)*s;
	int len = len_of[c >> 4];
	const char *end = s + len;
	uint32_t v = c & ((1u << (8 - len)) - 1);

	for (s++; s < end && *s != '\0'; s++)
		v = (v << 6) + ((unsigned char)*s & 0x3f);
	*cp = v;
	return s;
}

static const char *
parse_char(struct gparse *p, const char *s, uint32_t *cp)
{
	if (*s == '\\') {
		switch (s[1]) {
		case 'x':
			return parse_hex(p, s + 2, 2, cp);
		case 'u':
			return parse_hex(p, s + 2, 4, cp);
		case 'U':
			return parse_hex(p, s + 2, 8, cp);
		case 't':
			*cp = '\t';
			return s + 2;
		case 'r':
			*cp = '\r';
			return s + 2;
		case 'n':
			*cp = '\n';
			return s + 2;
		case '\\':
		case '"':
		case '[':
		case ']':
			*cp = (unsigned char)s[1];
			return s + 2;
		default:
			return fail(p, "unknown escape", s);
		}
	}
	if (*s == '\0')
		return fail(p, "unexpected end of input", s);
	return decode_src(s, cp);
}

/* Parse a decimal count of at most 9 digits. */
static const char *
parse_count(struct gparse *p, const char *s, uint64_t *v)
{
	const char *start = s;

	*v = 0;
	while (is_digit(*s)) {
		if (s - start >= 9)
			return fail(p, "number too large", start);
		*v = *v * 10 + (uint64_t)(*s - '0');
		s++;
	}
	if (s == start)
		return fail(p, "expecting an integer", s);
	return s;
}

/* <[id]> or <text>, where text is the bytes of one token. */
static const char *
parse_token(struct gparse *p, const char *s, uint32_t *id)
{
	const struct gvocab *v = p->voc;
	const char *start = s, *gt;
	uint64_t n;
	size_t len;
	int32_t i;

	if (*s != '<')
		return fail(p, "expecting '<'", s);
	s++;
	if (*s == '[') {
		if ((s = parse_count(p, s + 1, &n)) == NULL)
			return NULL;
		if (s[0] != ']' || s[1] != '>')
			return fail(p, "expecting ']>'", s);
		*id = (uint32_t)n;
		return s + 2;
	}
	if ((gt = strchr(s, '>')) == NULL)
		return fail(p, "expecting '>'", s);
	len = (size_t)(gt + 1 - start);
	*id = NO_TOKEN;
	for (i = 0; v != NULL && i < v->n; i++) {
		if (v->off[i + 1] - v->off[i] == len &&
		    memcmp(v->bytes + v->off[i], start, len) == 0) {
			*id = (uint32_t)i;
			break;
		}
	}
	if (*id == NO_TOKEN && !p->lax)
		return fail(p, v ? "unknown token" : "no vocab for token",
		    start);
	return gt + 1;
}

/*
 * Rewrite the last item S of r for S{min,max}, as llama.cpp does:
 * S{m,n} is m copies of S, then an optional chain of n - m more. S{m,}
 * is m copies, then S' ::= S S' | "".
 */
static int
repeat(struct gparse *p, struct gvec *r, size_t last, uint64_t min,
    uint64_t max, uint64_t *nprev, uint32_t name)
{
	struct gvec rec = { NULL, 0, 0 };
	struct gel *prev = NULL;
	size_t np = r->n - last;
	uint64_t total = 1, nopt, i;
	long id = 0, lastid = 0;
	int nomax = max == NO_MAX, rv = -1;

	if (np == 0) {
		fail(p, "expecting item before repetition", "");
		return -1;
	}
	if (!nomax && max < min) {
		fail(p, "repetition max below min", "");
		return -1;
	}
	if (!nomax && max > 0)
		total = max;
	else if (min > 0)
		total = min;
	if (*nprev * total >= MAX_REPEAT) {
		fail(p, "repetition too large", "");
		return -1;
	}
	if ((prev = resize(NULL, np, sizeof(*prev))) == NULL) {
		fail(p, "out of memory", "");
		return -1;
	}
	memcpy(prev, r->v + last, np * sizeof(*prev));
	if (min == 0) {
		r->n = last;
		p->nel -= np;
	}
	for (i = 1; i < min; i++)
		if (push_all(p, r, prev, np) == -1)
			goto out;
	nopt = nomax ? 1 : max - min;
	for (i = 0; i < nopt; i++) {
		if ((id = gen_symbol(p, name)) == -1)
			goto out;
		if (push_all(p, &rec, prev, np) == -1)
			goto out;
		if ((i > 0 || nomax) &&
		    push(p, &rec, G_REF, (uint32_t)(nomax ? id : lastid)) == -1)
			goto out;
		if (push(p, &rec, G_ALT, 0) == -1 ||
		    push(p, &rec, G_END, 0) == -1)
			goto out;
		add_rule(p, (uint32_t)id, &rec);
		lastid = id;
	}
	if (nopt > 0 && push(p, r, G_REF, (uint32_t)lastid) == -1)
		goto out;
	*nprev *= total;
	rv = 0;
out:
	free(rec.v);
	free(prev);
	return rv;
}

static const char *parse_alternates(struct gparse *, const char *,
		    uint32_t, uint32_t, int);

static const char *
parse_sequence(struct gparse *p, const char *s, uint32_t name, struct gvec *r,
    int nested)
{
	size_t last = r->n;
	uint64_t nprev = 1, min, max;
	uint32_t cp, type, before;
	const char *e;
	long id;

	while (*s != '\0') {
		if (*s == '"') {
			s++;
			last = r->n;
			nprev = 1;
			while (*s != '"') {
				if ((s = parse_char(p, s, &cp)) == NULL ||
				    push(p, r, G_CHAR, cp) == -1)
					return NULL;
			}
			s = skip_space(s + 1, nested);
		} else if (*s == '[') {
			s++;
			type = G_CHAR;
			if (*s == '^') {
				s++;
				type = G_NOT;
			}
			last = r->n;
			nprev = 1;
			while (*s != ']') {
				if ((s = parse_char(p, s, &cp)) == NULL ||
				    push(p, r, r->n > last ? G_CALT : type,
				    cp) == -1)
					return NULL;
				if (s[0] != '-' || s[1] == ']')
					continue;
				if ((s = parse_char(p, s + 1, &cp)) == NULL ||
				    push(p, r, G_UPPER, cp) == -1)
					return NULL;
			}
			s = skip_space(s + 1, nested);
		} else if (*s == '<' || *s == '!') {
			type = G_TOK;
			if (*s == '!') {
				type = G_TOKNOT;
				s++;
			}
			if ((s = parse_token(p, s, &cp)) == NULL)
				return NULL;
			last = r->n;
			nprev = 1;
			if (push(p, r, type, cp) == -1)
				return NULL;
			s = skip_space(s, nested);
		} else if (is_word((unsigned char)*s)) {
			for (e = s; is_word((unsigned char)*e); e++)
				;
			if ((id = get_symbol(p, s, (size_t)(e - s))) == -1)
				return NULL;
			s = skip_space(e, nested);
			last = r->n;
			nprev = 1;
			if (push(p, r, G_REF, (uint32_t)id) == -1)
				return NULL;
		} else if (*s == '(') {
			s = skip_space(s + 1, 1);
			before = p->nsym;
			if ((id = gen_symbol(p, name)) == -1)
				return NULL;
			s = parse_alternates(p, s, name, (uint32_t)id, 1);
			if (s == NULL)
				return NULL;
			nprev = p->nsym - before > 1 ? p->nsym - before : 1;
			last = r->n;
			if (push(p, r, G_REF, (uint32_t)id) == -1)
				return NULL;
			if (*s != ')')
				return fail(p, "expecting ')'", s);
			s = skip_space(s + 1, nested);
		} else if (*s == '.') {
			last = r->n;
			nprev = 1;
			if (push(p, r, G_ANY, 0) == -1)
				return NULL;
			s = skip_space(s + 1, nested);
		} else if (*s == '*' || *s == '+' || *s == '?') {
			min = *s == '+' ? 1 : 0;
			max = *s == '?' ? 1 : NO_MAX;
			s = skip_space(s + 1, nested);
			if (repeat(p, r, last, min, max, &nprev, name) == -1)
				return fail(p, "bad repetition", s);
		} else if (*s == '{') {
			s = skip_space(s + 1, nested);
			if ((s = parse_count(p, s, &min)) == NULL)
				return NULL;
			s = skip_space(s, nested);
			max = NO_MAX;
			if (*s == '}') {
				max = min;
			} else if (*s == ',') {
				s = skip_space(s + 1, nested);
				if (is_digit(*s)) {
					if ((s = parse_count(p, s, &max)) ==
					    NULL)
						return NULL;
					s = skip_space(s, nested);
				}
				if (*s != '}')
					return fail(p, "expecting '}'", s);
			} else
				return fail(p, "expecting ',' or '}'", s);
			s = skip_space(s + 1, nested);
			if (min > MAX_REPEAT || (max != NO_MAX &&
			    max > MAX_REPEAT))
				return fail(p, "too many repetitions", s);
			if (repeat(p, r, last, min, max, &nprev, name) == -1)
				return fail(p, "bad repetition", s);
		} else
			break;
	}
	return s;
}

static const char *
parse_alternates(struct gparse *p, const char *s, uint32_t name, uint32_t id,
    int nested)
{
	struct gvec r = { NULL, 0, 0 };

	if (++p->depth > MAX_NEST)
		return fail(p, "groups nest too deep", s);
	s = parse_sequence(p, s, name, &r, nested);
	while (s != NULL && *s == '|') {
		if (push(p, &r, G_ALT, 0) == -1)
			s = NULL;
		else
			s = parse_sequence(p, skip_space(s + 1, 1), name, &r,
			    nested);
	}
	if (s != NULL && push(p, &r, G_END, 0) == -1)
		s = NULL;
	if (s != NULL)
		add_rule(p, id, &r);
	free(r.v);
	p->depth--;
	return s;
}

static const char *
parse_rule(struct gparse *p, const char *s)
{
	const char *e = s;
	long id;

	while (is_word((unsigned char)*e))
		e++;
	if (e == s)
		return fail(p, "expecting name", s);
	if ((id = get_symbol(p, s, (size_t)(e - s))) == -1)
		return NULL;
	s = skip_space(e, 0);
	if (s[0] != ':' || s[1] != ':' || s[2] != '=')
		return fail(p, "expecting ::=", s);
	s = parse_alternates(p, skip_space(s + 3, 1), (uint32_t)id,
	    (uint32_t)id, 0);
	if (s == NULL)
		return NULL;
	if (*s == '\r')
		s += s[1] == '\n' ? 2 : 1;
	else if (*s == '\n')
		s++;
	else if (*s != '\0')
		return fail(p, "expecting newline or end", s);
	return skip_space(s, 1);
}

static void
parse_free(struct gparse *p)
{
	uint32_t i;

	for (i = 0; i < p->nsym; i++) {
		free(p->names[i]);
		free(p->rules[i].v);
	}
	free(p->names);
	free(p->rules);
	free(p->ht);
}

/* Matcher */

/* Does c match the class at e? Sets *after past the class. */
static int
match_char(const struct gel *e, uint32_t c, const struct gel **after)
{
	int pos = e->type == G_CHAR || e->type == G_ANY, found = 0;

	do {
		if (e[1].type == G_UPPER) {
			found = found || (e->value <= c && c <= e[1].value);
			e += 2;
		} else if (e->type == G_ANY) {
			found = 1;
			e++;
		} else {
			found = found || e->value == c;
			e++;
		}
	} while (e->type == G_CALT);
	if (after != NULL)
		*after = e;
	return found == pos;
}

/* Can some end of a partial UTF-8 sequence match the class at e? */
static int
match_partial(const struct gel *e, uint32_t pv, int nr)
{
	int pos = e->type == G_CHAR || e->type == G_ANY;
	uint32_t lo, hi;

	if (nr < 0 || (nr == 1 && pv < 2))
		return 0;
	lo = pv << (nr * 6);
	hi = lo | ((1u << (nr * 6)) - 1);
	if (lo == 0) {
		if (nr == 2)
			lo = 1u << 11;
		else if (nr == 3)
			lo = 1u << 16;
	}
	do {
		if (e[1].type == G_UPPER) {
			if (e->value <= hi && lo <= e[1].value)
				return pos;
			e += 2;
		} else if (e->type == G_ANY) {
			return 1;
		} else {
			if (lo <= e->value && e->value <= hi)
				return pos;
			e++;
		}
	} while (e->type == G_CALT);
	return !pos;
}

static int
match_token(const struct gel *e, int32_t id)
{
	if (id < 0)
		return 0;
	if (e->type == G_TOK)
		return e->value == (uint32_t)id;
	return e->value != (uint32_t)id;
}

static int
spend(struct grammar *g, uint64_t n)
{
	if (g->budget < n) {
		set_err(g->err, sizeof(g->err), "grammar match too costly");
		return -1;
	}
	g->budget -= n;
	return 0;
}

static int
push_work(struct grammar *g, uint32_t i)
{
	if (grow(&g->work, &g->capwork, g->nwork + 1, sizeof(*g->work)) == -1) {
		set_err(g->err, sizeof(g->err), "out of memory");
		return -1;
	}
	g->work[g->nwork++] = i;
	return 0;
}

/*
 * Expand rule references at the top of stk until each stack ends at a
 * terminal, and add the results to out. stk must not point into seen,
 * sbuf or out.
 */
static int
advance(struct grammar *g, const uint32_t *stk, uint32_t len,
    struct sset *out)
{
	const struct gel *top;
	uint32_t i, sub, n, top_i;
	long j;

	ss_clear(&g->seen);
	g->nwork = 0;
	if ((j = ss_add(g, &g->seen, stk, len, MAX_SEEN)) < 0 ||
	    push_work(g, (uint32_t)j) == -1)
		return -1;
	while (g->nwork > 0) {
		if (spend(g, 1) == -1)
			return -1;
		i = g->work[--g->nwork];
		len = g->seen.v[i].len;
		memcpy(g->sbuf, g->seen.pool + g->seen.v[i].off,
		    len * sizeof(*g->sbuf));
		if (len == 0) {
			if (ss_add(g, out, g->sbuf, 0, MAX_STACKS) == -1)
				return -1;
			continue;
		}
		top_i = g->sbuf[len - 1];
		top = &g->el[top_i];
		if (top->type != G_REF) {
			if (ss_add(g, out, g->sbuf, len, MAX_STACKS) == -1)
				return -1;
			continue;
		}
		sub = g->roff[top->value];
		for (;;) {
			n = len - 1;
			if (!is_end(top + 1))
				g->sbuf[n++] = top_i + 1;
			if (!is_end(&g->el[sub]))
				g->sbuf[n++] = sub;
			if (n > MAX_DEPTH) {
				set_err(g->err, sizeof(g->err),
				    "parse stack too deep");
				return -1;
			}
			j = ss_add(g, &g->seen, g->sbuf, n, MAX_SEEN);
			if (j == -1 || (j >= 0 &&
			    push_work(g, (uint32_t)j) == -1))
				return -1;
			while (!is_end(&g->el[sub]))
				sub++;
			if (g->el[sub].type != G_ALT)
				break;
			sub++;
		}
	}
	return 0;
}

/* Pop the terminal at the top of stk, push next, and advance into out. */
static int
step(struct grammar *g, const uint32_t *stk, uint32_t len,
    const struct gel *next, struct sset *out)
{
	uint32_t n = len - 1;

	memcpy(g->abuf, stk, n * sizeof(*stk));
	if (!is_end(next))
		g->abuf[n++] = (uint32_t)(next - g->el);
	return advance(g, g->abuf, n, out);
}

/* Match one code point against every stack of in, into out. */
static int
accept_cp(struct grammar *g, struct sset *in, uint32_t cp, struct sset *out)
{
	const struct gel *top, *after;
	const uint32_t *stk;
	uint32_t len;
	size_t i;

	for (i = 0; i < in->n; i++) {
		stk = ss_get(in, i, &len);
		if (len == 0)
			continue;
		top = &g->el[stk[len - 1]];
		if (top->type == G_TOK || top->type == G_TOKNOT)
			continue;
		if (match_char(top, cp, &after) &&
		    step(g, stk, len, after, out) == -1)
			return -1;
	}
	return 0;
}

/*
 * Decode s after a partial sequence, as llama.cpp does. Writes the code
 * points and a 0 to out. On invalid input out is empty and *nr is -1.
 */
static void
decode(const char *s, size_t n, uint32_t *pv, int *nr, uint32_t *out)
{
	static const int left[16] = {
		1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4
	};
	const unsigned char *u = (const unsigned char *)s;
	uint32_t v = *pv;
	int r = *nr, start = *nr;
	size_t i = 0, k = 0;

	while (i < n && r > 0) {
		if ((u[i] >> 6) != 2)
			goto bad;
		v = (v << 6) + (u[i] & 0x3f);
		i++;
		r--;
	}
	if (start > 0 && r == 0)
		out[k++] = v;
	while (i < n) {
		r = left[u[i] >> 4] - 1;
		if (r < 0)
			goto bad;
		v = u[i] & ((1u << (7 - r)) - 1);
		i++;
		while (i < n && r > 0) {
			v = (v << 6) + (u[i] & 0x3f);
			i++;
			r--;
		}
		if (r == 0)
			out[k++] = v;
	}
	out[k] = 0;
	*pv = v;
	*nr = r;
	return;
bad:
	out[0] = 0;
	*pv = 0;
	*nr = -1;
}

/*
 * Advance the stacks by a token: id is its token id or -1, cps its code
 * points. Returns -2 and keeps the stacks when nothing fits.
 */
static int
accept_cps(struct grammar *g, int32_t id, const uint32_t *cps, uint32_t pv,
    int nr)
{
	const struct gel *top;
	const uint32_t *stk;
	uint32_t len;
	size_t i;

	g->budget = MAX_WORK;
	ss_clear(&g->nxt);
	ss_clear(&g->a);
	for (i = 0; i < g->cur.n; i++) {
		stk = ss_get(&g->cur, i, &len);
		if (len == 0)
			continue;
		top = &g->el[stk[len - 1]];
		if (top->type == G_TOK || top->type == G_TOKNOT) {
			if (match_token(top, id) &&
			    step(g, stk, len, top + 1, &g->nxt) == -1)
				return -1;
		} else if (ss_add(g, &g->a, stk, len, MAX_STACKS) == -1)
			return -1;
	}
	for (; *cps != 0 && g->a.n > 0; cps++) {
		ss_clear(&g->b);
		if (accept_cp(g, &g->a, *cps, &g->b) == -1)
			return -1;
		ss_swap(&g->a, &g->b);
	}
	for (i = 0; i < g->a.n; i++) {
		stk = ss_get(&g->a, i, &len);
		if (ss_add(g, &g->nxt, stk, len, MAX_STACKS) == -1)
			return -1;
	}
	/* A partial sequence must be able to end in a match. */
	if (nr > 0) {
		ss_clear(&g->b);
		for (i = 0; i < g->nxt.n; i++) {
			stk = ss_get(&g->nxt, i, &len);
			if (len == 0)
				continue;
			top = &g->el[stk[len - 1]];
			if (top->type != G_TOK && top->type != G_TOKNOT &&
			    match_partial(top, pv, nr) &&
			    ss_add(g, &g->b, stk, len, MAX_STACKS) == -1)
				return -1;
		}
		ss_swap(&g->nxt, &g->b);
	}
	if (g->nxt.n == 0) {
		set_err(g->err, sizeof(g->err), "input does not fit");
		return -2;
	}
	ss_swap(&g->cur, &g->nxt);
	g->pv = pv;
	g->nr = nr;
	return 0;
}

int
grammar_accept_bytes(struct grammar *g, const char *s, size_t n)
{
	uint32_t *cps, pv = g->pv;
	int nr = g->nr, rv;

	if (n == 0)
		return 0;
	if (memchr(s, '\0', n) != NULL) {
		set_err(g->err, sizeof(g->err), "NUL byte in input");
		return -1;
	}
	if ((cps = resize(NULL, n + 1, sizeof(*cps))) == NULL) {
		set_err(g->err, sizeof(g->err), "out of memory");
		return -1;
	}
	decode(s, n, &pv, &nr, cps);
	if (nr < 0) {
		set_err(g->err, sizeof(g->err), "invalid UTF-8");
		rv = -1;
	} else
		rv = accept_cps(g, -1, cps, pv, nr) < 0 ? -1 : 0;
	free(cps);
	return rv;
}

int
grammar_accept_token(struct grammar *g, const struct gvocab *v, int32_t id)
{
	uint32_t pv = g->pv;
	int nr = g->nr;

	if (id < 0 || id >= v->n || (v->flag[id] & GV_BAD)) {
		set_err(g->err, sizeof(g->err), "token %d does not fit", id);
		return -1;
	}
	if (v->flag[id] & GV_EOG) {
		if (grammar_can_end(g))
			return 0;
		set_err(g->err, sizeof(g->err), "grammar cannot end here");
		return -1;
	}
	decode(v->bytes + v->off[id], v->off[id + 1] - v->off[id], &pv, &nr,
	    v->cps);
	if (nr < 0) {
		set_err(g->err, sizeof(g->err), "token %d does not fit", id);
		return -1;
	}
	if (accept_cps(g, id, v->cps, pv, nr) < 0) {
		if (strcmp(g->err, "input does not fit") == 0)
			set_err(g->err, sizeof(g->err), "token %d does not fit",
			    id);
		return -1;
	}
	return 0;
}

/* A candidate buffer of at least n entries, kept for reuse. */
static struct gcand *
scratch(struct grammar *g, unsigned depth, int which, size_t n)
{
	struct scratch *sc;
	struct gcand *p;
	size_t cap;

	if (depth > MAX_TOKCP + 1) {
		set_err(g->err, sizeof(g->err), "token too long");
		return NULL;
	}
	if (g->scr == NULL &&
	    (g->scr = calloc((MAX_TOKCP + 2) * 3, sizeof(*g->scr))) == NULL)
		goto nomem;
	sc = &g->scr[depth * 3 + which];
	if (n <= sc->cap)
		return sc->p;
	cap = n < 64 ? 64 : n;
	if ((p = resize(sc->p, cap, sizeof(*p))) == NULL)
		goto nomem;
	sc->p = p;
	sc->cap = cap;
	return p;
nomem:
	set_err(g->err, sizeof(g->err), "out of memory");
	return NULL;
}

/* Filter c by one stack. Rejected candidates go to rej. */
static int
reject_stack(struct grammar *g, unsigned depth, const uint32_t *stk,
    uint32_t len, const struct gcand *c, size_t n, struct gcand *rej,
    size_t *nrej)
{
	struct sset next = { 0 };
	struct gcand *nc, *sub;
	const struct gel *top, *after;
	uint32_t *copy = g->abuf, cp, k, last = 0;
	const uint32_t *cps = g->cpbase;
	size_t i, nn = 0, ns = 0;
	int rv = -1, last_ok = 0;

	*nrej = 0;
	if (spend(g, n) == -1)
		return -1;
	if (len == 0) {
		for (i = 0; i < n; i++)
			if (cps[c[i].cp] != 0 || g->cnr[c[i].idx] != 0)
				rej[(*nrej)++] = c[i];
		return 0;
	}
	top = &g->el[stk[len - 1]];
	/* A token element takes a whole token, not the rest of one. */
	if (top->type == G_TOK || top->type == G_TOKNOT) {
		for (i = 0; i < n; i++) {
			if (cps[c[i].cp] == 0 ? g->cnr[c[i].idx] != 0 :
			    depth > 0 || !match_token(top, g->cids[c[i].idx]))
				rej[(*nrej)++] = c[i];
		}
		return 0;
	}
	if ((nc = scratch(g, depth, 1, n)) == NULL)
		return -1;
	for (i = 0; i < n; i++) {
		if ((cp = cps[c[i].cp]) == 0) {
			k = c[i].idx;
			if (g->cnr[k] != 0 && !match_partial(top, g->cpv[k],
			    g->cnr[k]))
				rej[(*nrej)++] = c[i];
		} else if (cp == last ? last_ok : (last = cp,
		    last_ok = match_char(top, cp, NULL))) {
			nc[nn] = c[i];
			nc[nn++].cp++;
		} else
			rej[(*nrej)++] = c[i];
	}
	if (nn == 0)
		return 0;
	match_char(top, 0, &after);
	memcpy(copy, stk, (len - 1) * sizeof(*copy));
	i = len - 1;
	if (!is_end(after))
		copy[i++] = (uint32_t)(after - g->el);
	if ((sub = scratch(g, depth, 2, nn)) == NULL ||
	    advance(g, copy, (uint32_t)i, &next) == -1 ||
	    reject_all(g, depth + 1, &next, nc, nn, sub, &ns) == -1)
		goto out;
	for (i = 0; i < ns; i++) {
		rej[*nrej] = sub[i];
		rej[(*nrej)++].cp--;
	}
	rv = 0;
out:
	ss_free(&next);
	return rv;
}

/* Candidates that no stack of s accepts go to rej. */
static int
reject_all(struct grammar *g, unsigned depth, struct sset *s,
    const struct gcand *c, size_t n, struct gcand *rej, size_t *nrej)
{
	const struct gcand *in = c;
	struct gcand *tmp, *out;
	const uint32_t *stk;
	uint32_t len;
	size_t i, nt = n;

	if (n == 0 || s->n == 0) {
		memmove(rej, c, n * sizeof(*c));
		*nrej = n;
		return 0;
	}
	if ((tmp = scratch(g, depth, 0, n)) == NULL)
		return -1;
	out = rej;
	for (i = 0; i < s->n && nt > 0; i++) {
		stk = ss_get(s, i, &len);
		if (reject_stack(g, depth, stk, len, in, nt, out, &nt) == -1)
			return -1;
		in = out;
		out = out == rej ? tmp : rej;
	}
	if (in != rej)
		memcpy(rej, in, nt * sizeof(*rej));
	*nrej = nt;
	return 0;
}

int
grammar_reset(struct grammar *g)
{
	g->budget = MAX_WORK;
	g->pv = 0;
	g->nr = 0;
	return copy_set(g, &g->cur, &g->start);
}

int
grammar_can_end(const struct grammar *g)
{
	size_t i;

	for (i = 0; i < g->cur.n; i++)
		if (g->cur.v[i].len == 0)
			return 1;
	return 0;
}

size_t
grammar_nstacks(const struct grammar *g)
{
	return g->cur.n;
}

const char *
grammar_error(const struct grammar *g)
{
	return g->err;
}

int
grammar_check(struct grammar *g, const struct gvocab *v, const int32_t *ids,
    size_t n, uint8_t *ok)
{
	struct gcand *c, *rej;
	uint32_t *cp = v->cps, pv, *npv;
	size_t i, nc = 0, nrej;
	int8_t *nnr;
	int32_t id;
	int nr, end = grammar_can_end(g);

	if (n > g->ccap) {
		if ((c = resize(g->cbuf, n, sizeof(*c))) != NULL)
			g->cbuf = c;
		if ((rej = resize(g->rbuf, n, sizeof(*rej))) != NULL)
			g->rbuf = rej;
		if ((npv = resize(g->cpv, n, sizeof(*npv))) != NULL)
			g->cpv = npv;
		if ((nnr = resize(g->cnr, n, sizeof(*nnr))) != NULL)
			g->cnr = nnr;
		if (c == NULL || rej == NULL || npv == NULL || nnr == NULL) {
			set_err(g->err, sizeof(g->err), "out of memory");
			return -1;
		}
		g->ccap = n;
	}
	c = g->cbuf;
	rej = g->rbuf;
	g->cids = ids;
	g->cpbase = g->nr == 0 ? v->dcp : v->cps;
	for (i = 0; i < n; i++) {
		id = ids[i];
		ok[i] = 0;
		if (id < 0 || id >= v->n || (v->flag[id] & GV_BAD))
			continue;
		if (v->flag[id] & GV_EOG) {
			ok[i] = (uint8_t)end;
			continue;
		}
		c[nc].idx = (uint32_t)i;
		if (g->nr == 0) {
			/* The common case: no partial sequence. */
			c[nc].cp = v->doff[id];
			g->cpv[i] = v->dpv[id];
			g->cnr[i] = v->dnr[id];
		} else {
			pv = g->pv;
			nr = g->nr;
			decode(v->bytes + v->off[id], v->off[id + 1] -
			    v->off[id], &pv, &nr, cp);
			c[nc].cp = (uint32_t)(cp - v->cps);
			g->cpv[i] = pv;
			g->cnr[i] = (int8_t)nr;
			while (*cp != 0)
				cp++;
			cp++;
		}
		ok[i] = 1;
		nc++;
	}
	g->budget = MAX_WORK;
	if (reject_all(g, 0, &g->cur, c, nc, rej, &nrej) == -1)
		return -1;
	for (i = 0; i < nrej; i++)
		ok[rej[i].idx] = 0;
	return 0;
}

int
grammar_fits(struct grammar *g, const struct gvocab *v, int32_t id)
{
	uint8_t ok;

	if (grammar_check(g, v, &id, 1, &ok) == -1)
		return -1;
	return ok;
}

int
grammar_mask(struct grammar *g, const struct gvocab *v, uint8_t *ok)
{
	return grammar_check(g, v, v->ids, (size_t)v->n, ok);
}

/* Build */

/* Does a rule have an empty alternative? */
static int
rule_empty(const struct grammar *g, uint32_t r)
{
	const struct gel *e = &g->el[g->roff[r]];
	int start = 1;

	for (;; e++) {
		if (is_end(e)) {
			if (start)
				return 1;
			if (e->type == G_END)
				return 0;
			start = 1;
		} else
			start = 0;
	}
}

struct lrframe {
	uint32_t	 rule;
	uint32_t	 pos;
	uint32_t	 wait;	/* rule to check on return, or NO_TOKEN */
	int		 rec;
};

/*
 * Find left recursion, as llama.cpp does: follow the leftmost reference
 * of each alternative, and the next one while the rules can be empty.
 * Returns the rule that recurses, -1 when none, or -2 on no memory.
 */
static long
left_recursion(const struct grammar *g)
{
	struct lrframe *st;
	uint8_t *state, *empty;	/* state: 1 in progress, 2 done */
	const struct gel *e;
	struct lrframe *f;
	uint32_t r, c, n;
	long rv = -1;

	st = resize(NULL, g->nrules, sizeof(*st));
	state = calloc(g->nrules, 1);
	empty = calloc(g->nrules, 1);
	if (st == NULL || state == NULL || empty == NULL) {
		rv = -2;
		goto out;
	}
	for (r = 0; r < g->nrules; r++)
		empty[r] = (uint8_t)rule_empty(g, r);
	for (r = 0; r < g->nrules && rv == -1; r++) {
		if (state[r] != 0)
			continue;
		n = 0;
		st[n++] = (struct lrframe){ r, g->roff[r], NO_TOKEN, 1 };
		state[r] = 1;
		while (n > 0 && rv == -1) {
			f = &st[n - 1];
			if (f->wait != NO_TOKEN) {
				if (!empty[f->wait])
					f->rec = 0;
				f->wait = NO_TOKEN;
			}
			e = &g->el[f->pos];
			if (e->type == G_END) {
				state[f->rule] = 2;
				n--;
				continue;
			}
			f->pos++;
			if (e->type == G_ALT) {
				f->rec = 1;
			} else if (e->type == G_REF && f->rec) {
				c = e->value;
				if (state[c] == 1)
					rv = c;
				else if (state[c] == 2) {
					if (!empty[c])
						f->rec = 0;
				} else {
					f->wait = c;
					state[c] = 1;
					st[n++] = (struct lrframe){ c,
					    g->roff[c], NO_TOKEN, 1 };
				}
			} else
				f->rec = 0;
		}
	}
out:
	free(st);
	free(state);
	free(empty);
	return rv;
}

void
grammar_free(struct grammar *g)
{
	size_t i;

	if (g == NULL)
		return;
	free(g->el);
	free(g->roff);
	ss_free(&g->cur);
	ss_free(&g->start);
	ss_free(&g->nxt);
	ss_free(&g->a);
	ss_free(&g->b);
	ss_free(&g->seen);
	free(g->work);
	free(g->sbuf);
	free(g->abuf);
	if (g->scr != NULL) {
		for (i = 0; i < (MAX_TOKCP + 2) * 3; i++)
			free(g->scr[i].p);
		free(g->scr);
	}
	free(g->cbuf);
	free(g->rbuf);
	free(g->cpv);
	free(g->cnr);
	free(g);
}

/* Flatten the parsed rules into g. */
static int
flatten(struct gparse *p, struct grammar *g)
{
	uint32_t i, k;
	const struct gel *e;

	g->nrules = p->nsym;
	g->roff = resize(NULL, p->nsym ? p->nsym : 1, sizeof(*g->roff));
	g->el = resize(NULL, p->nel ? p->nel : 1, sizeof(*g->el));
	g->sbuf = resize(NULL, MAX_DEPTH + 2, sizeof(*g->sbuf));
	g->abuf = resize(NULL, MAX_DEPTH + 2, sizeof(*g->abuf));
	if (g->roff == NULL || g->el == NULL || g->sbuf == NULL ||
	    g->abuf == NULL) {
		fail(p, "out of memory", "");
		return -1;
	}
	for (i = 0; i < p->nsym; i++) {
		if (p->rules[i].n == 0) {
			set_err(p->err, sizeof(p->err),
			    "undefined rule '%.64s'", p->names[i]);
			return -1;
		}
		g->roff[i] = g->nel;
		for (k = 0; k < p->rules[i].n; k++) {
			e = &p->rules[i].v[k];
			if (e->type == G_REF && e->value >= p->nsym) {
				fail(p, "bad rule reference", "");
				return -1;
			}
			g->el[g->nel++] = *e;
		}
	}
	return 0;
}

struct grammar *
grammar_new(const char *src, size_t len, const char *root,
    const struct gvocab *voc, int lax, char *err, size_t errlen)
{
	struct gparse p;
	struct grammar *g = NULL;
	const char *s;
	uint32_t slot, pos;
	long r, root_id;

	memset(&p, 0, sizeof(p));
	p.src = src;
	p.voc = voc;
	p.lax = lax;
	if (strlen(src) != len) {
		set_err(err, errlen, "NUL byte in grammar");
		return NULL;
	}
	if ((g = calloc(1, sizeof(*g))) == NULL) {
		set_err(err, errlen, "out of memory");
		return NULL;
	}
	s = skip_space(src, 1);
	while (s != NULL && *s != '\0')
		s = parse_rule(&p, s);
	if (s == NULL || flatten(&p, g) == -1)
		goto bad;
	if (p.nsym == 0 || (root_id = find_symbol(&p, root, strlen(root),
	    &slot)) < 0) {
		set_err(p.err, sizeof(p.err), "grammar has no rule '%.64s'",
		    root);
		goto bad;
	}
	if ((r = left_recursion(g)) == -2) {
		fail(&p, "out of memory", "");
		goto bad;
	} else if (r >= 0) {
		set_err(p.err, sizeof(p.err), "left recursion in rule '%.64s'",
		    p.names[r]);
		goto bad;
	}
	r = root_id;
	g->budget = MAX_WORK;
	pos = g->roff[r];
	for (;;) {
		g->abuf[0] = pos;
		if (advance(g, g->abuf, is_end(&g->el[pos]) ? 0 : 1,
		    &g->cur) == -1) {
			set_err(p.err, sizeof(p.err), "%s", g->err);
			goto bad;
		}
		while (!is_end(&g->el[pos]))
			pos++;
		if (g->el[pos].type != G_ALT)
			break;
		pos++;
	}
	if (copy_set(g, &g->start, &g->cur) == -1) {
		set_err(p.err, sizeof(p.err), "%s", g->err);
		goto bad;
	}
	parse_free(&p);
	return g;
bad:
	set_err(err, errlen, "%s", p.err[0] ? p.err : "bad grammar");
	parse_free(&p);
	grammar_free(g);
	return NULL;
}

/* Lua API */

static void
gvocab_free(struct gvocab *v)
{
	free(v->bytes);
	free(v->off);
	free(v->flag);
	free(v->cps);
	free(v->dcp);
	free(v->doff);
	free(v->dpv);
	free(v->dnr);
	free(v->ids);
	memset(v, 0, sizeof(*v));
}

static int
gvocab_gc(lua_State *L)
{
	gvocab_free(luaL_checkudata(L, 1, GVOCAB_MT));
	return 0;
}

/* Call method name of the object at idx with nargs arguments on top. */
static void
call_method(lua_State *L, int idx, const char *name, int nargs, int nres)
{
	lua_getfield(L, idx, name);
	if (lua_type(L, -1) != LUA_TFUNCTION)
		luaL_error(L, "tokenizer has no method '%s'", name);
	lua_insert(L, -nargs - 1);
	lua_pushvalue(L, idx);
	lua_insert(L, -nargs - 1);
	lua_call(L, nargs + 1, nres);
}

static void
gvocab_build(lua_State *L, int tk, struct gvocab *v)
{
	lua_Integer n, id;
	size_t len, total = 0, cap = 0;
	const char *piece, *kind;
	int isint;

	call_method(L, tk, "n_vocab", 0, 1);
	n = lua_tointegerx(L, -1, &isint);
	lua_pop(L, 1);
	if (!isint || n < 0 || n > INT32_MAX / 2)
		luaL_error(L, "bad vocabulary size");
	v->off = resize(NULL, (size_t)n + 1, sizeof(*v->off));
	v->flag = calloc((size_t)n + 1, 1);
	if (v->off == NULL || v->flag == NULL)
		luaL_error(L, "out of memory");
	v->off[0] = 0;
	for (id = 0; id < n; id++) {
		lua_pushinteger(L, id);
		call_method(L, tk, "token", 1, 2);
		piece = luaL_checklstring(L, -2, &len);
		kind = lua_tostring(L, -1);
		if (len == 0 || len > MAX_TOKCP || memchr(piece, 0, len) ||
		    (kind != NULL && strcmp(kind, "unused") == 0)) {
			v->flag[id] |= GV_BAD;
			len = 0;
		}
		if (total + len > UINT32_MAX / 2)
			luaL_error(L, "vocabulary too large");
		if (grow(&v->bytes, &cap, total + len + 1, 1) == -1)
			luaL_error(L, "out of memory");
		memcpy(v->bytes + total, piece, len);
		total += len;
		v->off[id + 1] = (uint32_t)total;
		lua_pop(L, 2);
	}
	v->n = (int32_t)n;
	v->cps = resize(NULL, total + (size_t)n + 1, sizeof(*v->cps));
	v->dcp = resize(NULL, total + (size_t)n + 1, sizeof(*v->dcp));
	v->doff = resize(NULL, (size_t)n + 1, sizeof(*v->doff));
	v->dpv = resize(NULL, (size_t)n + 1, sizeof(*v->dpv));
	v->dnr = resize(NULL, (size_t)n + 1, sizeof(*v->dnr));
	if (v->cps == NULL || v->dcp == NULL || v->doff == NULL ||
	    v->dpv == NULL || v->dnr == NULL)
		luaL_error(L, "out of memory");
	/* Decode each token once, from a clean state. */
	for (total = 0, id = 0; id < n; id++) {
		uint32_t pv = 0;
		int nr = 0;

		v->doff[id] = (uint32_t)total;
		decode(v->bytes + v->off[id], v->off[id + 1] - v->off[id],
		    &pv, &nr, v->dcp + total);
		v->dpv[id] = pv;
		v->dnr[id] = (int8_t)nr;
		while (v->dcp[total] != 0)
			total++;
		total++;
	}
	if ((v->ids = resize(NULL, (size_t)n + 1, sizeof(*v->ids))) == NULL)
		luaL_error(L, "out of memory");
	for (id = 0; id < n; id++)
		v->ids[id] = (int32_t)id;
	call_method(L, tk, "special", 0, 1);
	if (lua_getfield(L, -1, "eog") == LUA_TTABLE) {
		for (id = 1; lua_rawgeti(L, -1, id) != LUA_TNIL; id++) {
			n = lua_tointegerx(L, -1, &isint);
			if (isint && n >= 0 && n < v->n)
				v->flag[n] = GV_EOG;
			lua_pop(L, 1);
		}
		lua_pop(L, 1);
	}
	lua_pop(L, 2);
}

struct gvocab *
gvocab_push(lua_State *L, int tk)
{
	struct gvocab *v;

	tk = lua_absindex(L, tk);
	if (lua_getfield(L, LUA_REGISTRYINDEX, GVOCAB_CACHE) != LUA_TTABLE) {
		lua_pop(L, 1);
		lua_newtable(L);
		lua_createtable(L, 0, 1);
		lua_pushliteral(L, "k");
		lua_setfield(L, -2, "__mode");
		lua_setmetatable(L, -2);
		lua_pushvalue(L, -1);
		lua_setfield(L, LUA_REGISTRYINDEX, GVOCAB_CACHE);
	}
	lua_pushvalue(L, tk);
	if (lua_rawget(L, -2) == LUA_TUSERDATA) {
		v = luaL_checkudata(L, -1, GVOCAB_MT);
		lua_remove(L, -2);
		return v;
	}
	lua_pop(L, 1);
	v = lua_newuserdatauv(L, sizeof(*v), 0);
	memset(v, 0, sizeof(*v));
	luaL_setmetatable(L, GVOCAB_MT);
	gvocab_build(L, tk, v);
	lua_pushvalue(L, tk);
	lua_pushvalue(L, -2);
	lua_rawset(L, -4);
	lua_remove(L, -2);
	return v;
}

static struct grammar **
check_grammar(lua_State *L, int idx)
{
	struct grammar **gp = luaL_checkudata(L, idx, GRAMMAR_MT);

	if (*gp == NULL)
		luaL_error(L, "grammar closed");
	return gp;
}

/*** Parse a GBNF grammar, as llama.cpp reads it.
 * opts fields: root (the start rule, default "root"), tokenizer (to
 * resolve `<text>` tokens), lax (true: unknown `<text>` tokens parse as
 * tokens that never match).
 * @tparam string text the grammar
 * @tparam[opt] table opts options
 * @treturn[1] grammar the grammar
 * @return[2] nil
 * @treturn[2] string the error message
 * @function grammar
 */
static int
l_grammar(lua_State *L)
{
	const struct gvocab *v = NULL;
	struct grammar **gp;
	const char *src, *root = "root";
	char err[160];
	size_t len;
	int lax = 0;

	src = luaL_checklstring(L, 1, &len);
	if (lua_istable(L, 2)) {
		if (lua_getfield(L, 2, "root") != LUA_TNIL)
			root = luaL_checkstring(L, -1);
		if (lua_getfield(L, 2, "lax") != LUA_TNIL)
			lax = lua_toboolean(L, -1);
		if (lua_getfield(L, 2, "tokenizer") != LUA_TNIL)
			v = gvocab_push(L, -1);
	}
	gp = lua_newuserdatauv(L, sizeof(*gp), 0);
	*gp = NULL;
	luaL_setmetatable(L, GRAMMAR_MT);
	if ((*gp = grammar_new(src, len, root, v, lax, err,
	    sizeof(err))) == NULL) {
		luaL_pushfail(L);
		lua_pushstring(L, err);
		return 2;
	}
	return 1;
}

/*** A grammar and its parse state.
 * @type grammar
 */

/*** Advance the grammar by text.
 * A partial UTF-8 sequence at the end waits for the next call.
 * @tparam string bytes the text
 * @treturn[1] boolean true
 * @treturn[2] boolean false when the text does not fit; the state stays
 * @treturn[2] string the error message
 * @function grammar:accept
 */
static int
grammar_l_accept(lua_State *L)
{
	struct grammar *g = *check_grammar(L, 1);
	size_t n;
	const char *s = luaL_checklstring(L, 2, &n);

	if (grammar_accept_bytes(g, s, n) == -1) {
		lua_pushboolean(L, 0);
		lua_pushstring(L, grammar_error(g));
		return 2;
	}
	lua_pushboolean(L, 1);
	return 1;
}

/*** Test whether the input so far is a complete match.
 * @treturn boolean true when the grammar can end here
 * @function grammar:can_end
 */
static int
grammar_l_can_end(lua_State *L)
{
	lua_pushboolean(L, grammar_can_end(*check_grammar(L, 1)));
	return 1;
}

/*** Count the parse stacks.
 * @treturn int the number of stacks
 * @function grammar:stacks
 */
static int
grammar_l_stacks(lua_State *L)
{
	lua_pushinteger(L, (lua_Integer)grammar_nstacks(*check_grammar(L, 1)));
	return 1;
}

/*** Free the grammar.
 * The `__gc` and `__close` metamethods call this function.
 * @function grammar:close
 */
static int
grammar_l_close(lua_State *L)
{
	struct grammar **gp = luaL_checkudata(L, 1, GRAMMAR_MT);

	grammar_free(*gp);
	*gp = NULL;
	return 0;
}

static const luaL_Reg grammar_meta[] = {
	{ "__gc", grammar_l_close },
	{ "__close", grammar_l_close },
	{ NULL, NULL }
};

static const luaL_Reg grammar_methods[] = {
	{ "accept", grammar_l_accept },
	{ "can_end", grammar_l_can_end },
	{ "stacks", grammar_l_stacks },
	{ "close", grammar_l_close },
	{ NULL, NULL }
};

void
lgml_open_grammar(lua_State *L)
{
	luaL_newmetatable(L, GVOCAB_MT);
	lua_pushcfunction(L, gvocab_gc);
	lua_setfield(L, -2, "__gc");
	lua_pop(L, 1);
	luaL_newmetatable(L, GRAMMAR_MT);
	luaL_setfuncs(L, grammar_meta, 0);
	lua_newtable(L);
	luaL_setfuncs(L, grammar_methods, 0);
	lua_setfield(L, -2, "__index");
	lua_pop(L, 1);
	lua_pushcfunction(L, l_grammar);
	lua_setfield(L, -2, "grammar");
}
