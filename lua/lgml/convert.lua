--- Convert a Hugging Face safetensors model to gguf.
-- An architecture table maps the tensor names and config.json keys of a
-- model_type to llama.cpp gguf names and metadata. Tensors stream from
-- the shards to the output, one row chunk or one expert at a time.
-- @module lgml.convert
local lgml = require("lgml")
local json = require("lgml.json")
local writer = require("lgml.gguf_writer")

local M = {}

-- Source bytes per read when a tensor streams in row chunks.
local CHUNK = 64 << 20

local function read_json(path)
	local f, err = io.open(path, "rb")
	if not f then
		return nil, err
	end
	local s = f:read("a")
	f:close()
	return json.decode(s)
end

local function isnull(v)
	return v == nil or v == json.null
end

-- The leading layers whose moe_layer_freq entry is 0.
local function leading_dense(c)
	local n = 0
	for _, v in ipairs(c.moe_layer_freq or {}) do
		if v ~= 0 then
			break
		end
		n = n + 1
	end
	return n
end

--- Architectures by config.json model_type.
-- tensors lists { pattern, gguf name } in output order: the first capture
-- of a per-layer pattern is the layer, and with `stack` a second capture
-- is the expert, so the experts stack into one tensor, expert-major.
-- meta(config) gives { key, type, value } with keys under the arch name.
-- @table arch
M.arch = {}

M.arch.llada_moe_v2 = {
	name = "llada-moe-v2",
	tensors = {
		{ "^model%.embed_tokens%.weight$", "token_embd.weight" },
		{ "^model%.layers%.(%d+)%.input_layernorm%.weight$", "blk.%d.attn_norm.weight" },
		{ "^model%.layers%.(%d+)%.self_attn%.q_proj%.weight$", "blk.%d.attn_q.weight" },
		{ "^model%.layers%.(%d+)%.self_attn%.k_proj%.weight$", "blk.%d.attn_k.weight" },
		{ "^model%.layers%.(%d+)%.self_attn%.v_proj%.weight$", "blk.%d.attn_v.weight" },
		{ "^model%.layers%.(%d+)%.self_attn%.q_norm%.weight$", "blk.%d.attn_q_norm.weight" },
		{ "^model%.layers%.(%d+)%.self_attn%.k_norm%.weight$", "blk.%d.attn_k_norm.weight" },
		{ "^model%.layers%.(%d+)%.self_attn%.o_proj%.weight$", "blk.%d.attn_output.weight" },
		{ "^model%.layers%.(%d+)%.post_attention_layernorm%.weight$", "blk.%d.ffn_norm.weight" },
		{ "^model%.layers%.(%d+)%.mlp%.gate%.weight$", "blk.%d.ffn_gate_inp.weight" },
		{ "^model%.layers%.(%d+)%.mlp%.experts%.(%d+)%.gate_proj%.weight$",
		    "blk.%d.ffn_gate_exps.weight", stack = "num_experts" },
		{ "^model%.layers%.(%d+)%.mlp%.experts%.(%d+)%.up_proj%.weight$",
		    "blk.%d.ffn_up_exps.weight", stack = "num_experts" },
		{ "^model%.layers%.(%d+)%.mlp%.experts%.(%d+)%.down_proj%.weight$",
		    "blk.%d.ffn_down_exps.weight", stack = "num_experts" },
		{ "^model%.layers%.(%d+)%.shared_expert%.gate_proj%.weight$", "blk.%d.ffn_gate_shexp.weight" },
		{ "^model%.layers%.(%d+)%.shared_expert%.up_proj%.weight$", "blk.%d.ffn_up_shexp.weight" },
		{ "^model%.layers%.(%d+)%.shared_expert%.down_proj%.weight$", "blk.%d.ffn_down_shexp.weight" },
		{ "^model%.layers%.(%d+)%.mlp%.gate_proj%.weight$", "blk.%d.ffn_gate.weight" },
		{ "^model%.layers%.(%d+)%.mlp%.up_proj%.weight$", "blk.%d.ffn_up.weight" },
		{ "^model%.layers%.(%d+)%.mlp%.down_proj%.weight$", "blk.%d.ffn_down.weight" },
		{ "^model%.norm%.weight$", "output_norm.weight" },
		{ "^lm_head%.weight$", "output.weight" },
	},
	meta = function(c)
		assert(not c.moe_router_enable_expert_bias, "llada_moe_v2: router expert bias not supported")
		assert(c.moe_router_score_function == nil or c.moe_router_score_function == "softmax",
		    "llada_moe_v2: router score function must be softmax")
		assert(isnull(c.router_num_group), "llada_moe_v2: grouped routing not supported")
		assert(isnull(c.rope_scaling), "llada_moe_v2: rope scaling not supported")
		local hd = c.head_dim or c.hidden_size // c.num_attention_heads
		return {
			{ "vocab_size", "u32", c.vocab_size },
			{ "context_length", "u32", c.max_position_embeddings },
			{ "embedding_length", "u32", c.hidden_size },
			{ "block_count", "u32", c.num_hidden_layers },
			{ "leading_dense_block_count", "u32", leading_dense(c) },
			{ "feed_forward_length", "u32", c.dense_intermediate_size },
			{ "attention.head_count", "u32", c.num_attention_heads },
			{ "attention.head_count_kv", "u32", c.num_key_value_heads },
			{ "attention.key_length", "u32", hd },
			{ "attention.value_length", "u32", hd },
			{ "attention.layer_norm_rms_epsilon", "f32", c.rms_norm_eps },
			{ "attention.causal", "bool", false },
			{ "rope.freq_base", "f32", c.rope_theta },
			{ "rope.dimension_count", "u32", math.floor(hd * (c.partial_rotary_factor or 1)) },
			{ "expert_count", "u32", c.num_experts },
			{ "expert_used_count", "u32", c.num_experts_per_tok },
			{ "expert_shared_count", "u32", c.shared_expert_intermediate_size and 1 or 0 },
			{ "expert_feed_forward_length", "u32", c.expert_intermediate_size },
			{ "expert_shared_feed_forward_length", "u32", c.shared_expert_intermediate_size or 0 },
			{ "expert_weights_scale", "f32", c.routed_scaling_factor or 1 },
			{ "expert_weights_norm", "bool", c.norm_topk_prob and true or false },
			{ "expert_gating_func", "u32", 1 },
		}
	end,
	-- Keys outside the arch namespace.
	extra = function(c)
		return {
			{ "diffusion.shift_logits", "bool", false },
			{ "tokenizer.ggml.mask_token_id", "u32", c.mask_token_id },
		}
	end,
	-- Chat turns end with this token; the reference sampler stops on it.
	eot = "<|role_end|>",
}

-- ggml file types by output type name.
local FILE_TYPE = { f32 = 0, f16 = 1, bf16 = 32, q8_0 = 7, q4_K_M = 15, q6_K = 18, q5_K_M = 17 }

-- llama.cpp gives these layers more bits in the _M mixes.
local function more_bits(i, n)
	return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2
end

--- Output type policies: policy(t) gives the type of tensor t, which
-- has name, shape, layer and `n_layer`. 1-D tensors and the router stay
-- f32 in every policy. The _M mixes follow llama.cpp's llama-quantize.
-- @table policy
M.policy = {}

local function keep_f32(t)
	return #t.shape == 1 or t.name:find("ffn_gate_inp", 1, true) ~= nil
end

local function uniform(ty)
	return function(t)
		return keep_f32(t) and "f32" or ty
	end
end

M.policy.f32 = uniform("f32")
M.policy.f16 = uniform("f16")
M.policy.bf16 = uniform("bf16")
M.policy.q8_0 = uniform("q8_0")
M.policy.q6_K = uniform("q6_K")

local function k_mix(base, high)
	return function(t)
		if keep_f32(t) then
			return "f32"
		elseif t.name == "output.weight" then
			return "q6_K"
		elseif t.layer and (t.name:find("attn_v.weight", 1, true) or t.name:find("ffn_down", 1, true)) and
		    more_bits(t.layer, t.n_layer) then
			return high
		end
		return base
	end
end

M.policy.q4_K_M = k_mix("q4_K", "q6_K")
M.policy.q5_K_M = k_mix("q5_K", "q6_K")

-- K-quants need rows of whole 256-blocks; others fall back to q8_0.
local function fit(ty, n)
	local blk = (ty:find("_K", 1, true) and 256) or (ty:match("^q") and 32) or 1
	if n % blk == 0 then
		return ty
	end
	return n % 32 == 0 and "q8_0" or "f16"
end

--- The gguf pre-tokenizer names of known tokenizer.json Split regexes.
-- @table pre_regex
M.pre_regex = {
	["'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++" ..
	    "[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"] = "llada-moe",
}

local function pre_name(pt)
	if type(pt) ~= "table" then
		return nil
	elseif pt.type == "ByteLevel" and pt.use_regex ~= false then
		return "gpt2"
	elseif pt.type == "Sequence" and #pt.pretokenizers == 2 then
		local s, b = pt.pretokenizers[1], pt.pretokenizers[2]
		if s.type == "Split" and type(s.pattern) == "table" and b.type == "ByteLevel" and
		    b.use_regex == false then
			return M.pre_regex[s.pattern.Regex]
		end
	end
	return nil
end

-- The content of a tokenizer_config.json token entry.
local function token_text(v)
	if type(v) == "table" then
		return v.content
	end
	return type(v) == "string" and v or nil
end

--- Build the tokenizer metadata from tokenizer.json and tokenizer_config.json.
-- The vocabulary holds config.vocab_size tokens; ids without a token
-- become unused [PADn] tokens, as llama.cpp writes them.
-- @tparam string dir the model directory
-- @tparam table cfg config.json
-- @tparam[opt] table arch the architecture entry
-- @treturn table list of { key, type, value }
-- @raise on a tokenizer that is not a byte-level BPE
function M.vocab(dir, cfg, arch)
	local tj = assert(read_json(dir .. "/tokenizer.json"))
	local tc = read_json(dir .. "/tokenizer_config.json") or {}
	local m = tj.model
	assert(m and m.type == "BPE" and not m.byte_fallback, "convert: need a byte-level BPE tokenizer")
	local pre = pre_name(tj.pre_tokenizer)
	assert(pre, "convert: unknown pre_tokenizer " .. tostring(json.encode(tj.pre_tokenizer)))
	local norm = tj.normalizer
	assert(isnull(norm) or norm.type == "NFC", "convert: unsupported normalizer")
	local text, ty, maxid = {}, {}, -1
	for s, id in pairs(m.vocab) do
		text[id], ty[id] = s, 1
		maxid = math.max(maxid, id)
	end
	local byname = {}
	for _, a in ipairs(tj.added_tokens or {}) do
		text[a.id], ty[a.id] = a.content, a.special and 3 or 4
		maxid = math.max(maxid, a.id)
	end
	local n = math.max(cfg.vocab_size or 0, maxid + 1)
	local tokens, types = {}, {}
	for i = 0, n - 1 do
		tokens[i + 1] = text[i] or ("[PAD" .. i .. "]")
		types[i + 1] = ty[i] or 5
		byname[tokens[i + 1]] = byname[tokens[i + 1]] or i
	end
	local merges = {}
	for i, mg in ipairs(m.merges) do
		merges[i] = type(mg) == "table" and (mg[1] .. " " .. mg[2]) or mg
	end
	local function id_of(v)
		local s = token_text(v)
		return s and byname[s]
	end
	local function int(v)
		return math.type(v) == "integer" and v or nil
	end
	local bos = id_of(tc.bos_token) or int(cfg.bos_token_id)
	local eos = int(cfg.eos_token_id) or id_of(tc.eos_token)
	local pad = int(cfg.pad_token_id) or id_of(tc.pad_token)
	local kv = {
		{ "tokenizer.ggml.model", "string", "gpt2" },
		{ "tokenizer.ggml.pre", "string", pre },
		{ "tokenizer.ggml.tokens", { "array", "string" }, tokens },
		{ "tokenizer.ggml.token_type", { "array", "i32" }, types },
		{ "tokenizer.ggml.merges", { "array", "string" }, merges },
	}
	local function add(k, t, v)
		if v ~= nil then
			kv[#kv + 1] = { k, t, v }
		end
	end
	add("tokenizer.ggml.bos_token_id", "u32", bos)
	add("tokenizer.ggml.eos_token_id", "u32", eos)
	add("tokenizer.ggml.padding_token_id", "u32", pad)
	add("tokenizer.ggml.eot_token_id", "u32", arch and arch.eot and byname[arch.eot])
	add("tokenizer.ggml.add_bos_token", "bool", tc.add_bos_token == true)
	add("tokenizer.ggml.add_eos_token", "bool", tc.add_eos_token == true)
	if not isnull(norm) then
		add("tokenizer.ggml.normalizer.nfc", "bool", true)
	end
	local ct = tc.chat_template
	if type(ct) == "string" then
		add("tokenizer.chat_template", "string", ct)
	elseif type(ct) == "table" then
		for _, e in ipairs(ct) do
			add(e.name == "default" and "tokenizer.chat_template" or
			    "tokenizer.chat_template." .. e.name, "string", e.template)
		end
	end
	return kv
end

-- The source tensors of dir by name: { path, offset, size, type, shape }.
local function sources(dir)
	local files = {}
	local idx = read_json(dir .. "/model.safetensors.index.json")
	if idx then
		local seen = {}
		for _, f in pairs(idx.weight_map) do
			if not seen[f] then
				seen[f] = true
				files[#files + 1] = f
			end
		end
		table.sort(files)
	else
		files[1] = "model.safetensors"
	end
	local src = {}
	for _, f in ipairs(files) do
		local path = dir .. "/" .. f
		local h = lgml.safetensors(path)
		for _, t in ipairs(h.tensors) do
			assert(not src[t.name], "convert: tensor " .. t.name .. " appears twice")
			src[t.name] = { path = path, offset = h.data_offset + t.offset, size = t.size,
			    type = t.type, shape = t.shape }
		end
	end
	return src
end

local function same_shape(a, b)
	if #a ~= #b then
		return false
	end
	for i = 1, #a do
		if a[i] ~= b[i] then
			return false
		end
	end
	return true
end

--- Plan the output tensors of a model directory.
-- Each entry has name, shape (torch order), layer, `n_layer` and parts:
-- the source tensors in output order, one per expert when stacked.
-- @tparam string dir the model directory
-- @tparam table cfg config.json
-- @tparam table arch the architecture entry
-- @treturn table the plan
-- @raise on a source tensor that no pattern maps, or a missing expert
function M.plan(dir, cfg, arch)
	local src = sources(dir)
	local n_layer = cfg.num_hidden_layers
	local out, byname = {}, {}
	for name, s in pairs(src) do
		local hit
		for pi, p in ipairs(arch.tensors) do
			local c1, c2 = name:match(p[1])
			if c1 then
				hit = true
				local layer = p[2]:find("%d", 1, true) and math.tointeger(c1) or nil
				local gname = layer and p[2]:format(layer) or p[2]
				local e = byname[gname]
				if not e then
					local order = layer and layer or (pi == 1 and -1 or math.huge)
					e = { name = gname, layer = layer, n_layer = n_layer, parts = {},
					    order = order, pi = pi, stack = p.stack and cfg[p.stack] }
					byname[gname] = e
					out[#out + 1] = e
				end
				if e.stack then
					local x = math.tointeger(c2)
					assert(x and x < e.stack, "convert: " .. name .. ": bad expert")
					e.parts[x + 1] = s
				else
					e.parts[1] = s
				end
				break
			end
		end
		assert(hit, "convert: no gguf name for " .. name)
	end
	for _, e in ipairs(out) do
		local n = e.stack or 1
		local first = assert(e.parts[1], "convert: " .. e.name .. ": missing part 1")
		for i = 1, n do
			local p = assert(e.parts[i], "convert: " .. e.name .. ": missing expert " .. (i - 1))
			assert(p.type == first.type and same_shape(p.shape, first.shape),
			    "convert: " .. e.name .. ": experts differ in shape or type")
		end
		e.src_type = first.type
		e.shape = { table.unpack(first.shape) }
		if e.stack then
			table.insert(e.shape, 1, e.stack)
		end
	end
	table.sort(out, function(a, b)
		if a.order ~= b.order then
			return a.order < b.order
		end
		return a.pi < b.pi
	end)
	return out
end

-- Source files open once each; closing the reader closes them.
local Reader = {}
Reader.__index = Reader

function Reader:read(path, off, size)
	local f = self[path]
	if not f then
		f = assert(io.open(path, "rb"))
		self[path] = f
	end
	assert(f:seek("set", off))
	local s = f:read(size)
	assert(s and #s == size, path .. ": short read")
	return s
end

function Reader:__close()
	for k, f in pairs(self) do
		f:close()
		self[k] = nil
	end
end

-- Encode rows of n elements from type from to type to.
local function encode(s, from, to, n, threads)
	if from == to then
		return s
	elseif to == "f32" then
		return lgml.dequantize(from, s, n)
	end
	return lgml.quantize(to, s, n, { from = from, threads = threads })
end

--- Convert a Hugging Face model directory to a gguf file.
-- opts: dir, out, type ("bf16", "f16", "f32", "q8_0", "q6_K", "q4_K_M" or
-- "q5_K_M"), name (general.name), threads, and log(fmt, ...) for progress.
-- @tparam table opts the options
-- @treturn table the plan, with the output type of each tensor
-- @raise on a bad model directory or a write error
function M.convert(opts)
	local dir = assert(opts.dir, "convert: need dir")
	local cfg = assert(read_json(dir .. "/config.json"))
	local arch = assert(M.arch[cfg.model_type], "convert: unknown model_type " .. tostring(cfg.model_type))
	local otype = opts.type or "bf16"
	local policy = assert(M.policy[otype], "convert: unknown output type " .. otype)
	local threads = opts.threads or lgml.threads()
	local log = opts.log or function() end
	local plan = M.plan(dir, cfg, arch)
	local quant = false
	for _, e in ipairs(plan) do
		e.type = fit(policy(e), e.shape[#e.shape])
		quant = quant or not (e.type == "f32" or e.type == "f16" or e.type == "bf16")
	end
	local a = arch.name
	local kv = {
		{ "general.architecture", "string", a },
		{ "general.type", "string", "model" },
		{ "general.name", "string", opts.name or dir:match("([^/]+)/*$") },
		{ "general.file_type", "u32", assert(FILE_TYPE[otype]) },
	}
	if quant then
		kv[#kv + 1] = { "general.quantization_version", "u32", 2 }
	end
	for _, e in ipairs(arch.meta(cfg)) do
		kv[#kv + 1] = { a .. "." .. e[1], e[2], e[3] }
	end
	for _, e in ipairs(arch.extra and arch.extra(cfg) or {}) do
		kv[#kv + 1] = e
	end
	for _, e in ipairs(M.vocab(dir, cfg, arch)) do
		kv[#kv + 1] = e
	end

	local rd <close> = setmetatable({}, Reader)
	local ts = {}
	local t0 = lgml.clock()
	for i, e in ipairs(plan) do
		ts[i] = { e.name, e.type, e.shape, function(write)
			local n = e.shape[#e.shape]
			local rowb = lgml.row_size(e.src_type, n)
			local step = math.max(1, CHUNK // rowb) * rowb
			for _, p in ipairs(e.parts) do
				for off = 0, p.size - 1, step do
					local s = rd:read(p.path, p.offset + off, math.min(step, p.size - off))
					write(encode(s, e.src_type, e.type, n, threads))
				end
			end
			log("%4d/%d %-32s %-5s %s %.0fs", i, #plan, e.name, e.type,
			    table.concat(e.shape, "x"), lgml.clock() - t0)
		end }
	end
	local out = assert(opts.out, "convert: need out")
	assert(writer.write(out, kv, ts))
	return plan
end

return M
