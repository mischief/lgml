-- Hugging Face to gguf conversion on a tiny synthetic llada_moe_v2
-- directory: names, shapes, types, metadata, bf16 bytes, expert order,
-- quantization error and the tokenizer. Output is TAP.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local json = require("lgml.json")
local writer = require("lgml.gguf_writer")
local convert = require("lgml.convert")
local bpe_vocab = require("bpe_vocab")

local n = 0
local failed = false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local function writefile(path, s)
	local f = assert(io.open(path, "wb"))
	f:write(s)
	f:close()
end

local function readat(path, off, size)
	local f = assert(io.open(path, "rb"))
	f:seek("set", off)
	local s = f:read(size)
	f:close()
	return s
end

-- A safetensors file of { name, dtype, shape, bytes }.
local function write_st(path, list)
	local hdr, off, parts = { __metadata__ = { format = "pt" } }, 0, {}
	for i, t in ipairs(list) do
		hdr[t[1]] = { dtype = t[2], shape = json.array(t[3]),
			data_offsets = json.array({ off, off + #t[4] }) }
		off = off + #t[4]
		parts[i] = t[4]
	end
	local h = assert(json.encode(hdr))
	h = h .. string.rep(" ", -#h % 8)
	writefile(path, string.pack("<I8", #h) .. h .. table.concat(parts))
end

-- Random bf16 values with a per-tensor scale.
local function bf16(count, scale)
	local p = {}
	for i = 1, count do
		p[i] = string.pack("<f", (math.random() - 0.5) * scale)
	end
	return lgml.quantize("bf16", table.concat(p))
end

local dir = os.tmpname()
os.remove(dir)
assert(os.execute("mkdir -p '" .. dir .. "'"))

-- The tokenizer: the tiny byte-level vocab with the LLaDA-MoE split regex.
local tj, ids = bpe_vocab.hf_json()
local REGEX = "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++" ..
    "[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
tj.pre_tokenizer = { type = "Sequence", pretokenizers = json.array({
	{ type = "Split", pattern = { Regex = REGEX }, behavior = "Isolated", invert = false },
	{ type = "ByteLevel", add_prefix_space = false, trim_offsets = true, use_regex = false },
}) }
tj.post_processor = { type = "ByteLevel", add_prefix_space = true, trim_offsets = false,
	use_regex = true }
writefile(dir .. "/tokenizer.json", assert(json.encode(tj)))
local TEMPLATE = "{% for m in messages %}<role>{{ m.role }}</role>{{ m.content }}{% endfor %}"
writefile(dir .. "/tokenizer_config.json", assert(json.encode({
	add_bos_token = false, add_eos_token = false, bos_token = "[CLS]",
	eos_token = { content = "[SEP]" }, chat_template = TEMPLATE })))
local n_vocab = ids["[MASK]"] + 5

local H, HD, NE, FF = 256, 128, 4, 256
local cfg = {
	model_type = "llada_moe_v2", num_hidden_layers = 2, hidden_size = H,
	num_attention_heads = 2, num_key_value_heads = 1, head_dim = HD,
	dense_intermediate_size = 512, expert_intermediate_size = FF,
	shared_expert_intermediate_size = FF, num_experts = NE, num_experts_per_tok = 2,
	max_position_embeddings = 4096, rms_norm_eps = 1e-5, rope_theta = 500000,
	partial_rotary_factor = 1, routed_scaling_factor = 2.5, norm_topk_prob = true,
	moe_layer_freq = json.array({ 1, 1 }), moe_router_enable_expert_bias = false,
	moe_router_score_function = "softmax", router_num_group = json.null,
	rope_scaling = json.null, vocab_size = n_vocab, mask_token_id = ids["[MASK]"],
	eos_token_id = ids["[SEP]"], pad_token_id = ids["[PAD]"], tie_word_embeddings = false,
}
writefile(dir .. "/config.json", assert(json.encode(cfg)))

-- Source tensors in two shards; layer 1's experts span both.
math.randomseed(7)
local src, shards, wmap = {}, { {}, {} }, {}
local function add(shard, name, shape, scale)
	local count = 1
	for _, d in ipairs(shape) do
		count = count * d
	end
	local b = bf16(count, scale or 0.1)
	src[name] = b
	table.insert(shards[shard], { name, "BF16", shape, b })
	wmap[name] = ("model-%05d-of-00002.safetensors"):format(shard)
end
add(1, "model.embed_tokens.weight", { n_vocab, H })
for l = 0, 1 do
	local p = "model.layers." .. l .. "."
	local s = l + 1
	add(s, p .. "input_layernorm.weight", { H }, 2)
	add(s, p .. "post_attention_layernorm.weight", { H }, 2)
	add(s, p .. "self_attn.q_proj.weight", { 2 * HD, H })
	add(s, p .. "self_attn.k_proj.weight", { HD, H })
	add(s, p .. "self_attn.v_proj.weight", { HD, H })
	add(s, p .. "self_attn.o_proj.weight", { H, 2 * HD })
	add(s, p .. "self_attn.q_norm.weight", { HD }, 2)
	add(s, p .. "self_attn.k_norm.weight", { HD }, 2)
	add(s, p .. "mlp.gate.weight", { NE, H })
	-- Reverse expert order in the file, so stacking must sort by id.
	for e = NE - 1, 0, -1 do
		local es = l == 1 and e % 2 + 1 or s
		add(es, p .. "mlp.experts." .. e .. ".gate_proj.weight", { FF, H })
		add(es, p .. "mlp.experts." .. e .. ".up_proj.weight", { FF, H })
		add(es, p .. "mlp.experts." .. e .. ".down_proj.weight", { H, FF })
	end
	add(s, p .. "shared_expert.gate_proj.weight", { FF, H })
	add(s, p .. "shared_expert.up_proj.weight", { FF, H })
	add(s, p .. "shared_expert.down_proj.weight", { H, FF })
end
add(2, "model.norm.weight", { H }, 2)
add(2, "lm_head.weight", { n_vocab, H })
for i = 1, 2 do
	write_st(dir .. ("/model-%05d-of-00002.safetensors"):format(i), shards[i])
end
writefile(dir .. "/model.safetensors.index.json",
    assert(json.encode({ metadata = { total_size = 0 }, weight_map = wmap })))

-- The expected gguf names and their sources.
local expect = {
	["token_embd.weight"] = { "model.embed_tokens.weight" },
	["output_norm.weight"] = { "model.norm.weight" },
	["output.weight"] = { "lm_head.weight" },
}
local LAYER = {
	attn_norm = "input_layernorm", ffn_norm = "post_attention_layernorm",
	attn_q = "self_attn.q_proj", attn_k = "self_attn.k_proj", attn_v = "self_attn.v_proj",
	attn_output = "self_attn.o_proj", attn_q_norm = "self_attn.q_norm",
	attn_k_norm = "self_attn.k_norm", ffn_gate_inp = "mlp.gate",
	ffn_gate_shexp = "shared_expert.gate_proj", ffn_up_shexp = "shared_expert.up_proj",
	ffn_down_shexp = "shared_expert.down_proj",
}
for l = 0, 1 do
	for g, h in pairs(LAYER) do
		expect[("blk.%d.%s.weight"):format(l, g)] = { ("model.layers.%d.%s.weight"):format(l, h) }
	end
	for _, w in ipairs({ "gate", "up", "down" }) do
		local parts = {}
		for e = 0, NE - 1 do
			parts[e + 1] = ("model.layers.%d.mlp.experts.%d.%s_proj.weight"):format(l, e, w)
		end
		expect[("blk.%d.ffn_%s_exps.weight"):format(l, w)] = parts
	end
end

local function source_bytes(name)
	local p = {}
	for i, s in ipairs(expect[name]) do
		p[i] = src[s]
	end
	return table.concat(p)
end

local function convert_to(ty)
	local out = os.tmpname()
	local good, err = pcall(convert.convert, { dir = dir, out = out, type = ty, name = "tiny",
		threads = 2 })
	ok(good, ty .. ": converts" .. (good and "" or ": " .. tostring(err)))
	local info = good and lgml.gguf(out)
	ok(info ~= nil, ty .. ": reads back")
	return out, info
end

-- bf16: names, shapes, types, metadata, exact bytes.
local out, info = convert_to("bf16")
local byname, count = {}, 0
for _, t in ipairs(info.tensors) do
	byname[t.name] = t
	count = count + 1
end
local nexp = 0
local missing = {}
for name in pairs(expect) do
	nexp = nexp + 1
	if not byname[name] then
		missing[#missing + 1] = name
	end
end
ok(count == nexp and #missing == 0, "bf16: tensor names " .. table.concat(missing, " "))
ok(info.tensors[1].name == "token_embd.weight" and info.tensors[#info.tensors].name == "output.weight",
    "bf16: token_embd first, output last")
local function shape_is(name, s)
	local t = byname[name]
	return t ~= nil and table.concat(t.shape, ",") == table.concat(s, ",")
end
ok(shape_is("blk.1.ffn_gate_exps.weight", { NE, FF, H }) and
    shape_is("blk.1.ffn_down_exps.weight", { NE, H, FF }) and
    shape_is("blk.0.attn_k.weight", { HD, H }) and shape_is("token_embd.weight", { n_vocab, H }) and
    shape_is("blk.0.attn_q_norm.weight", { HD }), "bf16: shapes")
ok(byname["blk.0.attn_q.weight"].type == "bf16" and byname["blk.0.ffn_up_exps.weight"].type == "bf16" and
    byname["blk.0.attn_norm.weight"].type == "f32" and byname["blk.1.ffn_gate_inp.weight"].type == "f32" and
    byname["output.weight"].type == "bf16", "bf16: types, norms and router f32")

local m = info.meta
local a = "llada-moe-v2."
ok(m["general.architecture"] == "llada-moe-v2" and m["general.name"] == "tiny" and
    m["general.file_type"] == 32 and m["general.quantization_version"] == nil, "bf16: general keys")
ok(m[a .. "block_count"] == 2 and m[a .. "embedding_length"] == H and
    m[a .. "context_length"] == 4096 and m[a .. "feed_forward_length"] == 512 and
    m[a .. "attention.head_count"] == 2 and m[a .. "attention.head_count_kv"] == 1 and
    m[a .. "attention.key_length"] == HD and m[a .. "attention.value_length"] == HD and
    m[a .. "rope.dimension_count"] == HD and m[a .. "rope.freq_base"] == 500000 and
    math.abs(m[a .. "attention.layer_norm_rms_epsilon"] - 1e-5) < 1e-10 and
    m[a .. "attention.causal"] == false and m[a .. "vocab_size"] == n_vocab and
    m[a .. "leading_dense_block_count"] == 0, "bf16: attention and size keys")
ok(m[a .. "expert_count"] == NE and m[a .. "expert_used_count"] == 2 and
    m[a .. "expert_shared_count"] == 1 and m[a .. "expert_feed_forward_length"] == FF and
    m[a .. "expert_shared_feed_forward_length"] == FF and m[a .. "expert_weights_scale"] == 2.5 and
    m[a .. "expert_weights_norm"] == true and m[a .. "expert_gating_func"] == 1, "bf16: expert keys")
ok(m["diffusion.shift_logits"] == false and m["tokenizer.ggml.mask_token_id"] == ids["[MASK]"],
    "bf16: diffusion keys")
local toks, types = m["tokenizer.ggml.tokens"], m["tokenizer.ggml.token_type"]
ok(#toks == n_vocab and toks[n_vocab] == "[PAD" .. (n_vocab - 1) .. "]" and types[n_vocab] == 5 and
    types[ids["[MASK]"] + 1] == 3 and types[ids["  "] + 1] == 4 and types[1] == 1,
    "bf16: tokens, types and padding")
ok(m["tokenizer.ggml.model"] == "gpt2" and m["tokenizer.ggml.pre"] == "llada-moe" and
    #m["tokenizer.ggml.merges"] == #tj.model.merges and
    m["tokenizer.ggml.merges"][2] == table.concat(tj.model.merges[2], " ") and
    m["tokenizer.ggml.bos_token_id"] == ids["[CLS]"] and m["tokenizer.ggml.eos_token_id"] == ids["[SEP]"] and
    m["tokenizer.ggml.padding_token_id"] == ids["[PAD]"] and m["tokenizer.ggml.add_bos_token"] == false and
    m["tokenizer.ggml.normalizer.nfc"] == true and m["tokenizer.chat_template"] == TEMPLATE,
    "bf16: tokenizer keys")

local function raw(path, inf, name)
	for _, t in ipairs(inf.tensors) do
		if t.name == name then
			return readat(path, inf.data_offset + t.offset, t.size)
		end
	end
end
local exact = true
for name, t in pairs(byname) do
	local got = raw(out, info, name)
	local want = source_bytes(name)
	if t.type == "f32" then
		want = lgml.dequantize("bf16", want, #want // 2)
	end
	if got ~= want then
		exact = false
		io.stderr:write("bf16 mismatch: ", name, "\n")
	end
end
ok(exact, "bf16: every tensor matches its source bytes, experts in id order")

-- The tokenizer from the gguf encodes as the tokenizer.json does.
do
	local tg <close> = assert(lgml.tokenizer(out))
	local tjs <close> = assert(lgml.tokenizer(dir .. "/tokenizer.json"))
	local same = true
	for _, s in ipairs({ "hello world", "123 12345 x's", "[CLS]hello[SEP]", "x  y\n\n z",
		"caf\u{e9} e\u{301}", "<role>user</role>hi" }) do
		local x, y = tg:encode(s), tjs:encode(s)
		if table.concat(x, ",") ~= table.concat(y, ",") then
			same = false
			io.stderr:write(("tokenizer differs on %q: %s / %s\n"):format(s,
			    table.concat(x, ","), table.concat(y, ",")))
		end
	end
	ok(same, "tokenizer: gguf and tokenizer.json encode the same")
	ok(table.concat(tg:encode("1234"), ",") ==
	    table.concat({ ids["1"], ids["2"], ids["3"], ids["4"] }, ","), "tokenizer: single digits")
end
os.remove(out)

-- Quantized outputs: types by policy and round-trip error.
local BOUND = { q8_0 = 0.01, q6_K = 0.02, q4_K = 0.08 }
for _, ty in ipairs({ "q8_0", "q4_K_M" }) do
	local qout, qinfo = convert_to(ty)
	local tmap, worst, typed = {}, {}, true
	for _, t in ipairs(qinfo.tensors) do
		tmap[t.name] = t.type
		local want = ty == "q8_0" and "q8_0" or "q4_K"
		if #t.shape == 1 or t.name:find("ffn_gate_inp") then
			want = "f32"
		elseif ty == "q4_K_M" and (t.name == "output.weight" or
		    t.name:find("^blk%.1%.attn_v") or t.name:find("^blk%.1%.ffn_down")) then
			want = "q6_K"
		end
		if t.type ~= want then
			typed = false
			io.stderr:write(("%s: %s is %s, want %s\n"):format(ty, t.name, t.type, want))
		end
		if t.type ~= "f32" then
			local got = raw(qout, qinfo, t.name)
			local ref = lgml.dequantize("bf16", source_bytes(t.name), t.shape[#t.shape])
			local rmse, _, rms = lgml.f32_error(ref, lgml.dequantize(t.type, got, t.shape[#t.shape]))
			worst[t.type] = math.max(worst[t.type] or 0, rmse / rms)
		end
	end
	ok(typed, ty .. ": types by policy")
	ok(qinfo.meta["general.file_type"] == (ty == "q8_0" and 7 or 15) and
	    qinfo.meta["general.quantization_version"] == 2, ty .. ": file type")
	local within, desc = true, {}
	for k, v in pairs(worst) do
		within = within and v < BOUND[k]
		desc[#desc + 1] = ("%s %.4f"):format(k, v)
	end
	table.sort(desc)
	ok(within and next(worst) ~= nil, ty .. ": relative error " .. table.concat(desc, ", "))
	os.remove(qout)
end

-- The writer streams data functions and checks their size.
do
	local p = os.tmpname()
	local q = lgml.quantize("q8_0", string.rep(string.pack("<f", 1.5), 64), 32)
	local good = writer.write(p, { { "general.architecture", "string", "x" } }, {
		{ "a", "q8_0", { 2, 32 }, function(write)
			write(q:sub(1, 10))
			write(q:sub(11))
		end },
		{ "b", "f32", { 3 }, string.pack("<fff", 1, 2, 3) },
	})
	local i = lgml.gguf(p)
	ok(good and i and i.tensors[1].type == "q8_0" and i.tensors[1].size == #q and
	    readat(p, i.data_offset + i.tensors[1].offset, #q) == q and
	    readat(p, i.data_offset + i.tensors[2].offset, 12) == string.pack("<fff", 1, 2, 3),
	    "writer: streamed and in-memory tensors")
	ok(not pcall(writer.write, p, {}, { { "a", "q8_0", { 2, 32 }, function(write) write("x") end } }),
	    "writer: a short data function is an error")
	ok(writer.size("q4_K", { 3, 512 }) == 3 * lgml.row_size("q4_K", 512) and
	    writer.size("bf16", { 4, 5 }) == 40, "writer: sizes")
	os.remove(p)
end

-- An unmapped source tensor is an error.
do
	local list = { table.unpack(shards[2]) }
	list[#list + 1] = { "model.extra.weight", "BF16", { 2 }, bf16(2, 1) }
	write_st(dir .. "/model-00002-of-00002.safetensors", list)
	wmap["model.extra.weight"] = "model-00002-of-00002.safetensors"
	writefile(dir .. "/model.safetensors.index.json",
	    assert(json.encode({ metadata = { total_size = 0 }, weight_map = wmap })))
	local p = os.tmpname()
	local good, err = pcall(convert.convert, { dir = dir, out = p, type = "bf16" })
	ok(not good and tostring(err):find("model.extra.weight", 1, true), "unmapped tensor is an error")
	os.remove(p)
end

os.execute("rm -rf '" .. dir .. "'")
print("1.." .. n)
os.exit(failed and 1 or 0)
