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

#define GRAPH_SIZE	16384	/* first tensor room of a graph; a 41-layer MoE needs about 4000 */
#define GRAPH_MAX	(1 << 20)	/* the most tensors a graph grows to */
#define LOAD_CHUNK	((size_t)64 << 20)

/* Uservalue slots. */
#define UV_PARENT	1	/* tensor: store or graph; store, graph: backend */
#define UV_EXTRA	2	/* store: name -> tensor; graph: set of leaves */
#define UV_GRAD		2	/* leaf tensor: its gradient leaf, or nil */
#define UV_HOST		3	/* store: the store of its CPU part, or nil */
#define UV_CPU		1	/* backend: its CPU backend, or nil */

/*
 * A backend opened with a CPU backend (opts.cpu) computes graphs that use
 * weights on both through a ggml_backend_sched.
 */
struct backend {
	ggml_backend_t		 be;
	struct backend		*cpu;		/* its CPU backend, or NULL */
	int			 threads;	/* CPU threads; 0 when the backend has none */
	int			 refs;
	int			 closed;
};

struct store {
	struct backend		*b;
	struct ggml_context	*ctx;
	ggml_backend_buffer_t	 buf;
	int			 refs;
	int			 closed;
	int			 mem_key;	/* see lgml_store_account */
	int			 mem_part;
	size_t			 mem_bytes;
};

/*
 * A graph grows: new tensors go into ctx, and a full ctx gets a larger
 * successor. A full gf moves to a larger cgraph. ctxs holds every
 * context, for tensors and for cgraphs.
 */
struct graph {
	struct backend		*b;
	struct ggml_context	*ctx;
	struct ggml_context	**ctxs;
	int			 nctx;
	int			 ctxcap;
	int			 ctx_left;	/* tensor slots left in ctx */
	struct ggml_cgraph	*gf;
	int			 grads;		/* gf has gradient arrays */
	ggml_gallocr_t		 galloc;
	int			 size;		/* node and leaf slots of gf */
	int			 ntensors;
	int			 nleaves;	/* leaf operands from other stores */
	int			 plan_nodes;	/* nodes at the last plan; -1 before */
	int			 plan_tensors;	/* ntensors at the last plan */
	int			 mixed;		/* has leaves on the CPU backend of b */
	int			 use_sched;	/* the last plan went through sched */
	ggml_backend_sched_t	 sched;		/* for mixed graphs, or NULL */
	int			 sched_size;	/* the graph size of sched */
	struct ggml_tensor	**saved;	/* node and its srcs, before sched */
	int			 nsaved;	/* nodes in saved */
	int			 backward;	/* 1 after tensor:backward */
	int			 nouts;		/* open output node tensors */
	int			 mem_key[2];	/* compute buffers: backend, CPU */
	size_t			 mem_bytes[2];
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
	int			 out;		/* a node counted in g->nouts */
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
void		 lgml_open_sha256(lua_State *);
void		 lgml_sha256(const void *, size_t, unsigned char[32]);
void		 lgml_open_vocab(lua_State *);
void		 lgml_open_ops_orpheus(lua_State *);
void		 lgml_open_ops_qwen3tts(lua_State *);
void		 lgml_open_ops_vibevoice(lua_State *);
void		 lgml_open_ops_nn(lua_State *);
void		 lgml_open_train(lua_State *);
void		 lgml_open_ngram(lua_State *);

/*
 * For op files. An op checks its graph with lgml_check_graph, takes
 * operands with lgml_operand, calls lgml_reserve before each new ggml
 * tensor, checks shapes, then returns lgml_push_node. lgml_reserve can
 * change g->ctx, so read g->ctx after it.
 */
struct graph	*lgml_check_graph(lua_State *, int);
struct tensor	*lgml_operand(lua_State *, struct graph *, int);
struct tensor	*lgml_opt_operand(lua_State *, struct graph *, int);
void		 lgml_reserve(lua_State *, struct graph *, int);
void		 lgml_graph_room(lua_State *, struct graph *, int);
struct ggml_cgraph *lgml_new_cgraph(lua_State *, struct graph *, int, int);
int		 lgml_push_node(lua_State *, struct graph *, struct ggml_tensor *, int);
void		 lgml_graph_ops(lua_State *, const luaL_Reg *);

/* Memory accounting, in mem.c. Parts of lgml_mem_add. */
enum { MEM_WEIGHTS, MEM_TENSORS, MEM_COMPUTE, MEM_NPART };
int		 lgml_mem_key(ggml_backend_buffer_type_t);
void		 lgml_mem_add(int, int, size_t, size_t);
void		 lgml_store_account(struct store *, int);
void		 lgml_store_unaccount(struct store *);
const char	*lgml_mem_check(lua_State *, ggml_backend_buffer_type_t, size_t, const char *);
void		 lgml_mem_need(lua_State *, ggml_backend_buffer_type_t, size_t, const char *);
void		 lgml_open_mem(lua_State *);

/* Decode JSON text and push the value. Raises an error on bad JSON. */
void		 lgml_json_decode(lua_State *, const char *, size_t, const char *);

#endif
