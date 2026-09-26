--- Qwen3-TTS talker and code predictor (CustomVoice gguf from CrispASR).
-- The talker emits codebook 0 of each 12.5 Hz frame. The code
-- predictor then emits codebooks 1 to 15 of the same frame.
--
-- A loaded talker has these fields: hp and cp_hp (hyperparameters), sp
-- (special token ids), speakers (lowercase name to { name, id, dialect },
-- plus list), languages (lowercase name to codec id) and kind.
-- @classmod lgml.qwen3tts.talker
local lgml = require("lgml")
local block = require("lgml.qwen3tts.block")

local M = {}
M.__index = M

--- Talker KV cache rows.
M.N_CTX = 2048
--- Code predictor KV cache rows.
M.CP_CTX = 16

-- Hyperparameters of the talker or the code predictor (pre). Both
-- share the head size, norm and rope settings of the talker.
local function hparams(meta, pre)
	local function k(s)
		return meta["qwen3tts." .. pre .. "." .. s]
	end
	local function t(s)
		return meta["qwen3tts.talker." .. s]
	end
	return {
		n_layer = k("n_layers"), n_embd = k("d_model"),
		n_head = k("n_heads"), n_head_kv = k("n_kv_heads"),
		head_dim = t("head_dim"), eps = t("rms_norm_eps"),
		rope_base = t("rope_theta"), n_ctx_orig = t("max_pos"),
		n_vocab = k("vocab_size"), n_groups = k("n_code_groups"),
	}
end

-- Token ids from the gguf metadata.
local function specials(meta)
	local function k(s)
		return assert(meta["qwen3tts." .. s], s)
	end
	return {
		im_start = k("im_start_token_id"), im_end = k("im_end_token_id"),
		assistant = k("assistant_token_id"), newline = 198,
		tts_bos = k("tts_bos_token_id"), tts_eos = k("tts_eos_token_id"),
		tts_pad = k("tts_pad_token_id"),
		codec_bos = k("talker.codec_bos_id"), codec_eos = k("talker.codec_eos_token_id"),
		codec_pad = k("talker.codec_pad_id"), nothink = k("talker.codec_nothink_id"),
		think = k("talker.codec_think_id"), think_bos = k("talker.codec_think_bos_id"),
		think_eos = k("talker.codec_think_eos_id"),
	}
end

local function speakers(meta)
	local names = meta["qwen3tts.spk_names"] or {}
	local ids = meta["qwen3tts.spk_token_ids"] or {}
	local dialects = meta["qwen3tts.spk_dialect_token_ids"] or {}
	local s = { list = {} }
	for i, name in ipairs(names) do
		s[name:lower()] = { name = name, id = ids[i], dialect = dialects[i] or 0 }
		s.list[i] = name
	end
	return s
end

local function languages(meta)
	local names = meta["qwen3tts.codec_language_names"] or {}
	local ids = meta["qwen3tts.codec_language_ids"] or {}
	local l = {}
	for i, name in ipairs(names) do
		l[name:lower()] = ids[i]
	end
	return l
end

local function load_weights(m, w)
	local function get(name)
		return assert(w:get(name), name)
	end
	m.text_embd = get("talker.token_embd_text.weight")
	m.fc1_w, m.fc1_b = get("talker.text_proj.fc1.weight"), get("talker.text_proj.fc1.bias")
	m.fc2_w, m.fc2_b = get("talker.text_proj.fc2.weight"), get("talker.text_proj.fc2.bias")
	m.codec_embd = get("talker.token_embd.weight")
	m.codec_head = get("talker.output.weight")
	m.norm = get("talker.output_norm.weight")
	m.layers = {}
	for i = 0, m.hp.n_layer - 1 do
		m.layers[i + 1] = block.layer(w, "talker.blk." .. i .. ".")
	end
	m.cp_norm = get("code_pred.output_norm.weight")
	m.cp_layers, m.cp_embd, m.cp_head = {}, {}, {}
	for i = 0, m.cp_hp.n_layer - 1 do
		m.cp_layers[i + 1] = block.layer(w, "code_pred.blk." .. i .. ".")
	end
	for i = 1, m.hp.n_groups - 1 do
		m.cp_embd[i] = get("code_pred.token_embd." .. (i - 1) .. ".weight")
		m.cp_head[i] = get("code_pred.output." .. (i - 1) .. ".weight")
	end
end

--- Text projection of text token embeddings: text_proj(text_embd(ids)).
-- @tparam lgml.graph g graph
-- @tparam lgml.tensor ids i32 text token ids (n)
-- @treturn lgml.tensor embeddings (n, d)
function M:text_proj(g, ids)
	local h = g:add(g:mul_mat(self.fc1_w, g:get_rows(self.text_embd, ids)), self.fc1_b)
	return g:add(g:mul_mat(self.fc2_w, g:silu(h)), self.fc2_b)
end

--- Keep the text-side embedding of tts_pad in the leaf self.tts_pad.
-- Each generated frame adds it to the talker input. load calls it.
function M:init_pad()
	local ids <close> = self.be:tensor("i32", { 1 }, { self.sp.tts_pad })
	local g <close> = self.be:graph()
	self.tts_pad = self.be:tensor("f32", { 1, self.hp.n_embd })
	g:compute(g:cpy(self:text_proj(g, ids), self.tts_pad))
end

--- Load a talker gguf.
-- @tparam lgml.backend be backend
-- @string path gguf path
-- @treturn lgml.qwen3tts.talker the talker
-- @raise when the file is not a qwen3tts talker gguf
-- @static
function M.load(be, path)
	local info = assert(lgml.gguf(path))
	local meta = info.meta
	if meta["general.architecture"] ~= "qwen3tts" then
		error(path .. ": not a qwen3tts talker gguf")
	end
	local w = assert(be:load(path))
	local m = setmetatable({
		be = be, w = w, meta = meta,
		hp = hparams(meta, "talker"), cp_hp = hparams(meta, "code_pred"),
		sp = specials(meta), speakers = speakers(meta), languages = languages(meta),
		kind = meta["qwen3tts.tts_model_type"],
	}, M)
	load_weights(m, w)
	m.cache = block.cache(be, m.hp, m.hp.n_layer, M.N_CTX)
	m.cp_cache = block.cache(be, m.cp_hp, m.cp_hp.n_layer, M.CP_CTX)
	m.one = block.step_leaves(be)
	m.cp_one = block.step_leaves(be)
	m.hidden = be:tensor("f32", { 1, m.hp.n_embd })
	m.id = be:tensor("i32", { 1 })
	m.codes = be:tensor("i32", { m.hp.n_groups })
	m.bias = m:suppress_bias(false)
	m.bias_noeos = m:suppress_bias(true)
	m:init_pad()
	return m
end

--- Logit bias for codebook 0.
-- The ids from vocab - 1024 up are control ids. The bias blocks all of
-- them except codec_eos, and codec_eos too when noeos.
-- @bool noeos also block codec_eos
-- @treturn lgml.tensor f32 bias (vocab)
function M:suppress_bias(noeos)
	local n = self.hp.n_vocab
	local b = {}
	for i = 1, n do
		b[i] = 0
	end
	for i = n - 1024, n - 1 do
		if i ~= self.sp.codec_eos or noeos then
			b[i + 1] = -math.huge
		end
	end
	return self.be:tensor("f32", { n }, b)
end

--- Free the weights.
function M:close()
	self.w:close()
end

--- Prompt rows for CustomVoice.
-- @tparam {int,...} text text token ids
-- @int spk speaker token id
-- @tparam[opt] int lang codec language id, or nil for auto
-- @treturn {int,...} text-side ids
-- @treturn {int,...} codec ids
-- @treturn {int,...} 0/1 row mask for the codec side
function M:prompt(text, spk, lang)
	local sp = self.sp
	local pre = lang and { sp.think, sp.think_bos, lang, sp.think_eos } or
	    { sp.nothink, sp.think_bos, sp.think_eos }
	local tids, cids, mask = { sp.im_start, sp.assistant, sp.newline }, { 0, 0, 0 }, { 0, 0, 0 }
	local function row(t, c)
		tids[#tids + 1], cids[#cids + 1], mask[#mask + 1] = t, c, 1
	end
	pre[#pre + 1] = spk
	pre[#pre + 1] = sp.codec_pad
	for i, c in ipairs(pre) do
		row(i < #pre and sp.tts_pad or sp.tts_bos, c)
	end
	for _, t in ipairs(text) do
		row(t, sp.codec_pad)
	end
	row(sp.tts_eos, sp.codec_pad)
	row(sp.tts_pad, sp.codec_bos)
	return tids, cids, mask
end

--- Talker forward on x at cache.n_past.
-- The last hidden row goes into self.hidden. The call computes g and
-- advances cache.n_past.
-- @tparam lgml.graph g graph
-- @tparam lgml.tensor x input (n, d)
-- @int n row count
-- @tparam lgml.tensor bias logit bias from suppress_bias
-- @treturn lgml.tensor logits (1, vocab) with the bias added
function M:talker(g, x, n, bias)
	local inp, leaves = block.inputs(self.be, self.cache, n, self.one)
	local h = block.forward(g, x, self.layers, self.norm, self.hp, inp)
	local logits = g:add(g:mul_mat(self.codec_head, h), bias)
	g:compute(logits, g:cpy(h, self.hidden))
	self.cache.n_past = inp.n_kv
	for _, t in ipairs(leaves) do
		t:close()
	end
	return logits
end

--- Prefill the prompt from the start of the cache.
-- The caller closes the graph after it reads the logits.
-- @tparam {int,...} tids text-side ids from prompt
-- @tparam {int,...} cids codec ids from prompt
-- @tparam {int,...} mask codec row mask from prompt
-- @treturn lgml.graph the graph
-- @treturn lgml.tensor logits of frame 0 (codec_eos blocked)
function M:prefill(tids, cids, mask)
	local be, n = self.be, #tids
	local t <close> = be:tensor("i32", { n }, tids)
	local c <close> = be:tensor("i32", { n }, cids)
	local m <close> = be:tensor("f32", { n, 1 }, mask)
	self.cache.n_past = 0
	local g = be:graph()
	local x = g:add(self:text_proj(g, t), g:mul(g:get_rows(self.codec_embd, c), m))
	return g, self:talker(g, x, n, self.bias_noeos)
end

--- Talker step for the frame in codes.
-- The caller closes the graph after it reads the logits.
-- @tparam {int,...} codes the n_groups codes of the last frame
-- @bool allow_eos allow codec_eos in the logits
-- @treturn lgml.graph the graph
-- @treturn lgml.tensor logits of the next frame
function M:step(codes, allow_eos)
	self.codes:set(codes)
	local g = self.be:graph()
	local x = self.tts_pad
	for i = 1, self.hp.n_groups do
		local id = g:view(self.codes, { 1 }, nil, (i - 1) * 4)
		local tab = i == 1 and self.codec_embd or self.cp_embd[i - 1]
		x = g:add(g:get_rows(tab, id), x)
	end
	return g, self:talker(g, x, 1, allow_eos and self.bias or self.bias_noeos)
end

--- Code predictor step i.
-- Step 0 reads the talker hidden state and the embedding of codebook 0.
-- Step i reads the embedding of the code from step i - 1.
-- The caller closes the graph after it reads the logits.
-- @int i step, 0 to n_groups - 2
-- @int prev code from the previous step (codebook 0 for step 0)
-- @treturn lgml.graph the graph
-- @treturn lgml.tensor logits for codebook i + 1
function M:cp_step(i, prev)
	local be, c = self.be, self.cp_cache
	self.id:set({ prev })
	local g = be:graph()
	local x, n
	if i == 0 then
		c.n_past = 0
		x = g:concat_qwen3tts(self.hidden, g:get_rows(self.codec_embd, self.id), 1)
		n = 2
	else
		x = g:get_rows(self.cp_embd[i], self.id)
		n = 1
	end
	local inp, leaves = block.inputs(be, c, n, self.cp_one)
	local h = block.forward(g, x, self.cp_layers, self.cp_norm, self.cp_hp, inp)
	local logits = g:mul_mat(self.cp_head[i + 1], h)
	g:compute(logits)
	c.n_past = inp.n_kv
	for _, t in ipairs(leaves) do
		t:close()
	end
	return g, logits
end

return M
