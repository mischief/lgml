/* GBNF grammars for constrained sampling. Not a public API. */
#ifndef GRAMMAR_H
#define GRAMMAR_H

#include <stddef.h>
#include <stdint.h>

#include <lua.h>

struct grammar;
struct gvocab;

/* Token bytes for grammar matching, from a tokenizer. */
struct gvocab {
	int32_t		 n;
	char		*bytes;
	uint32_t	*off;	/* token i: bytes[off[i] .. off[i + 1]) */
	uint8_t		*flag;
	uint32_t	*cps;	/* decode scratch: bytes + n + 1 entries */
	uint32_t	*dcp;	/* code points of each token, clean start */
	uint32_t	*doff;
	uint32_t	*dpv;	/* partial UTF-8 state after each token */
	int8_t		*dnr;
	int32_t		*ids;	/* 0 to n - 1 */
};

#define GV_EOG		0x01	/* ends generation */
#define GV_BAD		0x02	/* empty, NUL or too long */

/*
 * Parse GBNF text and build the start stacks. v resolves <name> tokens;
 * with lax set, unknown names parse as tokens that never match.
 * Returns NULL and a message in err on failure.
 */
struct grammar	*grammar_new(const char *, size_t, const char *,
		    const struct gvocab *, int, char *, size_t);
void		 grammar_free(struct grammar *);
const char	*grammar_error(const struct grammar *);
int		 grammar_can_end(const struct grammar *);
/* Go back to the start state. 0 on success, -1 on error. */
int		 grammar_reset(struct grammar *);
size_t		 grammar_nstacks(const struct grammar *);

/* 0 on success, -1 when the input does not fit or on error. */
int		 grammar_accept_bytes(struct grammar *, const char *, size_t);
int		 grammar_accept_token(struct grammar *, const struct gvocab *,
		    int32_t);

/* 1 when the token fits, 0 when not, -1 on error. */
int		 grammar_fits(struct grammar *, const struct gvocab *,
		    int32_t);

/* Set ok[i] to 1 when token ids[i] fits, else 0. -1 on error. */
int		 grammar_check(struct grammar *, const struct gvocab *,
		    const int32_t *, size_t, uint8_t *);

/* Set ok[i] to 1 for each token that fits, else 0. -1 on error. */
int		 grammar_mask(struct grammar *, const struct gvocab *,
		    uint8_t *);

/* Push the vocab of the tokenizer at idx, built once and cached. */
struct gvocab	*gvocab_push(lua_State *, int);

void		 lgml_open_grammar(lua_State *);

#endif
