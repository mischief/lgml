/* Shared between the lgml C files. Not a public API. */
#ifndef LGML_H
#define LGML_H

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <lua.h>
#include <lauxlib.h>

#define BACKEND_MT	"lgml_backend"
#define STORE_MT	"lgml_weights"
#define GRAPH_MT	"lgml_graph"
#define TENSOR_MT	"lgml_tensor"
#define HOSTBUF_MT	"lgml_hostbuf"

#define GRAPH_SIZE	16384	/* tensors per graph; a 41-layer MoE needs about 4000 */
#define LOAD_CHUNK	((size_t)64 << 20)

/* Uservalue slots. */
#define UV_PARENT	1	/* tensor: store or graph; store, graph: backend */
#define UV_EXTRA	2	/* store: name -> tensor; graph: set of leaves */
#define UV_GRAD		2	/* leaf tensor: its gradient leaf, or nil */

struct backend {
	ggml_backend_t		 be;
	int			 refs;
	int			 closed;
};

struct store {
	struct backend		*b;
	struct ggml_context	*ctx;
	ggml_backend_buffer_t	 buf;
	int			 refs;
	int			 closed;
};

struct graph {
	struct backend		*b;
	struct ggml_context	*ctx;
	struct ggml_cgraph	*gf;
	ggml_gallocr_t		 galloc;
	int			 size;		/* the most ggml tensors */
	int			 ntensors;
	int			 plan_nodes;	/* nodes at the last plan; -1 before */
	int			 plan_tensors;	/* ntensors at the last plan */
	int			 backward;	/* 1 after tensor:backward */
	int			 refs;
	int			 closed;
};

struct hostbuf {
	char			*p;
	size_t			 size;
};

/* A leaf has s. A node has g. */
struct tensor {
	struct ggml_tensor	*t;
	struct store		*s;
	struct graph		*g;
	int			 rank;
};

struct backend	*lgml_check_backend(lua_State *, int);
struct tensor	*lgml_check_tensor(lua_State *, int);
struct store	*lgml_push_store(lua_State *, int, struct ggml_context *,
		    ggml_backend_buffer_t);
struct tensor	*lgml_push_leaf(lua_State *, int, struct ggml_tensor *, int);
void		 lgml_tensor_methods(lua_State *, const luaL_Reg *);

/* Each adds its functions to the module table on top of the stack. */
void		 lgml_open_sample(lua_State *);
void		 lgml_open_tokenize(lua_State *);
void		 lgml_open_safetensors(lua_State *);
void		 lgml_open_audio(lua_State *);
void		 lgml_open_image(lua_State *);
void		 lgml_open_vocab(lua_State *);
void		 lgml_open_ops_orpheus(lua_State *);
void		 lgml_open_ops_qwen3tts(lua_State *);
void		 lgml_open_ops_vibevoice(lua_State *);
void		 lgml_open_ops_nn(lua_State *);
void		 lgml_open_train(lua_State *);

/*
 * For op files. An op checks its graph with lgml_check_graph, takes
 * operands with lgml_operand, calls lgml_reserve before each new ggml
 * tensor, checks shapes, then returns lgml_push_node.
 */
struct graph	*lgml_check_graph(lua_State *, int);
struct tensor	*lgml_operand(lua_State *, struct graph *, int);
struct tensor	*lgml_opt_operand(lua_State *, struct graph *, int);
void		 lgml_reserve(lua_State *, struct graph *, int);
int		 lgml_push_node(lua_State *, struct graph *, struct ggml_tensor *, int);
void		 lgml_graph_ops(lua_State *, const luaL_Reg *);

/* Decode JSON text and push the value. Raises an error on bad JSON. */
void		 lgml_json_decode(lua_State *, const char *, size_t, const char *);

#endif
