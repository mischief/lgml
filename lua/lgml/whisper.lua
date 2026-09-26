--- Whisper encoder-decoder on lgml, from Hugging Face safetensors.
-- Greedy decoding without timestamps. Audio and mel stay in C; Lua
-- only builds graphs and handles token ids.
-- @module lgml.whisper
local lgml = require("lgml")
local json = require("lgml.json")

local M = {}
M.__index = M

--- Input sample rate in Hz (16000).
M.RATE = 16000
--- Samples in one 30 s window at 16 kHz (480000).
M.N_SAMPLES = 480000
--- Mel frames in one window (3000).
M.N_FRAMES = 3000

--- Language codes in Whisper token order, with names.
-- The list is flat: code, name, code, name, and so on.
M.LANGUAGES = {
	"en", "english", "zh", "chinese", "de", "german", "es", "spanish",
	"ru", "russian", "ko", "korean", "fr", "french", "ja", "japanese",
	"pt", "portuguese", "tr", "turkish", "pl", "polish", "ca", "catalan",
	"nl", "dutch", "ar", "arabic", "sv", "swedish", "it", "italian",
	"id", "indonesian", "hi", "hindi", "fi", "finnish", "vi", "vietnamese",
	"he", "hebrew", "uk", "ukrainian", "el", "greek", "ms", "malay",
	"cs", "czech", "ro", "romanian", "da", "danish", "hu", "hungarian",
	"ta", "tamil", "no", "norwegian", "th", "thai", "ur", "urdu",
	"hr", "croatian", "bg", "bulgarian", "lt", "lithuanian", "la", "latin",
	"mi", "maori", "ml", "malayalam", "cy", "welsh", "sk", "slovak",
	"te", "telugu", "fa", "persian", "lv", "latvian", "bn", "bengali",
	"sr", "serbian", "az", "azerbaijani", "sl", "slovenian", "kn", "kannada",
	"et", "estonian", "mk", "macedonian", "br", "breton", "eu", "basque",
	"is", "icelandic", "hy", "armenian", "ne", "nepali", "mn", "mongolian",
	"bs", "bosnian", "kk", "kazakh", "sq", "albanian", "sw", "swahili",
	"gl", "galician", "mr", "marathi", "pa", "punjabi", "si", "sinhala",
	"km", "khmer", "sn", "shona", "yo", "yoruba", "so", "somali",
	"af", "afrikaans", "oc", "occitan", "ka", "georgian", "be", "belarusian",
	"tg", "tajik", "sd", "sindhi", "gu", "gujarati", "am", "amharic",
	"yi", "yiddish", "lo", "lao", "uz", "uzbek", "fo", "faroese",
	"ht", "haitian creole", "ps", "pashto", "tk", "turkmen", "nn", "nynorsk",
	"mt", "maltese", "sa", "sanskrit", "lb", "luxembourgish", "my", "myanmar",
	"bo", "tibetan", "tl", "tagalog", "mg", "malagasy", "as", "assamese",
	"tt", "tatar", "haw", "hawaiian", "ln", "lingala", "ha", "hausa",
	"ba", "bashkir", "jw", "javanese", "su", "sundanese", "yue", "cantonese",
}

local function exists(path)
	local f = io.open(path, "rb")
	if f then
		f:close()
	end
	return f ~= nil
end

local function read_json(path)
	local f = assert(io.open(path, "rb"))
	local s = f:read("a")
	f:close()
	return assert(json.decode(s))
end

--- Test for a Hugging Face Whisper model directory.
-- This is `lgml.detect.is_whisper`.
-- @function is_model
-- @tparam string dir the model directory
-- @treturn boolean true when config.json has model_type "whisper"
M.is_model = require("lgml.detect").is_whisper

local function weight_files(dir)
	local idx = dir .. "/model.safetensors.index.json"
	if not exists(idx) then
		return dir .. "/model.safetensors"
	end
	local seen, list = {}, {}
	for _, f in pairs(read_json(idx).weight_map) do
		if not seen[f] then
			seen[f] = true
			list[#list + 1] = dir .. "/" .. f
		end
	end
	table.sort(list)
	return list
end

local function hparams(cfg)
	return {
		n_mels = cfg.num_mel_bins, d = cfg.d_model,
		enc_layers = cfg.encoder_layers, enc_heads = cfg.encoder_attention_heads,
		dec_layers = cfg.decoder_layers, dec_heads = cfg.decoder_attention_heads,
		n_audio_ctx = cfg.max_source_positions, n_text_ctx = cfg.max_target_positions,
		n_vocab = cfg.vocab_size, eps = 1e-5,
	}
end

local function getter(w)
	return function(name)
		return assert(w:get(name), "missing tensor " .. name)
	end
end

local function lin(get, p, bias)
	return { w = get(p .. ".weight"), b = bias ~= false and get(p .. ".bias") or nil }
end

local function norm(get, p)
	return { w = get(p .. ".weight"), b = get(p .. ".bias") }
end

-- Self-attention and MLP weights of one layer with prefix p.
local function self_attn_weights(get, p)
	return {
		ln1 = norm(get, p .. "self_attn_layer_norm"),
		q = lin(get, p .. "self_attn.q_proj"), k = lin(get, p .. "self_attn.k_proj", false),
		v = lin(get, p .. "self_attn.v_proj"), o = lin(get, p .. "self_attn.out_proj"),
		fc1 = lin(get, p .. "fc1"), fc2 = lin(get, p .. "fc2"),
	}
end

local function load_encoder(get, hp)
	local p = "model.encoder."
	local e = {
		conv1 = lin(get, p .. "conv1"), conv2 = lin(get, p .. "conv2"),
		pos = get(p .. "embed_positions.weight"), ln_post = norm(get, p .. "layer_norm"),
		layers = {},
	}
	for i = 0, hp.enc_layers - 1 do
		local q = p .. "layers." .. i .. "."
		local L = self_attn_weights(get, q)
		L.ln2 = norm(get, q .. "final_layer_norm")
		e.layers[i + 1] = L
	end
	return e
end

local function load_decoder(get, hp)
	local p = "model.decoder."
	local d = {
		tok = get(p .. "embed_tokens.weight"), pos = get(p .. "embed_positions.weight"),
		ln = norm(get, p .. "layer_norm"), layers = {},
	}
	for i = 0, hp.dec_layers - 1 do
		local q = p .. "layers." .. i .. "."
		local L = self_attn_weights(get, q)
		L.ln2 = norm(get, q .. "encoder_attn_layer_norm")
		L.cq = lin(get, q .. "encoder_attn.q_proj")
		L.ck = lin(get, q .. "encoder_attn.k_proj", false)
		L.cv = lin(get, q .. "encoder_attn.v_proj")
		L.co = lin(get, q .. "encoder_attn.out_proj")
		L.ln3 = norm(get, q .. "final_layer_norm")
		d.layers[i + 1] = L
	end
	return d
end

-- Special token ids and the language table.
local function load_tokens(v)
	local function id(s)
		return assert(v:id(s), "vocab lacks " .. s)
	end
	local t = {
		sot = id("<|startoftranscript|>"), eot = id("<|endoftext|>"),
		transcribe = id("<|transcribe|>"), translate = id("<|translate|>"),
		notimestamps = id("<|notimestamps|>"), prev = id("<|startofprev|>"),
		space = v:id(" "),
		lang = {}, lang_code = {}, lang_name = {},
	}
	for i = 1, #M.LANGUAGES, 2 do
		local code, name = M.LANGUAGES[i], M.LANGUAGES[i + 1]
		local lid = v:id("<|" .. code .. "|>")
		if lid then
			t.lang[code], t.lang[name] = lid, lid
			t.lang_code[lid], t.lang_name[lid] = code, name
		end
	end
	return t
end

-- An f32 bias over the vocabulary: 0 where keep(id) is true, else -inf.
local function bias(be, n, keep)
	local b = {}
	for i = 1, n do
		b[i] = keep(i - 1) and 0 or -math.huge
	end
	return be:tensor("f32", { n }, b)
end

--- Load a Whisper model from a Hugging Face directory.
-- The directory holds config.json, vocab.json, optional
-- added_tokens.json, and model.safetensors or a sharded index.
-- `encode_text` also needs merges.txt. The
-- function also allocates the decoder state (see `Whisper:alloc_state`).
-- @tparam lgml.backend be the backend that holds the weights
-- @tparam string dir the model directory
-- @treturn Whisper the model
-- @raise when config.json is not a Whisper config, or a file or
-- tensor is missing
function M.load(be, dir)
	local cfg = read_json(dir .. "/config.json")
	assert(cfg.model_type == "whisper", dir .. ": not a Whisper model")
	local hp = hparams(cfg)
	local w = assert(be:load_safetensors(weight_files(dir), { f32_1d = true }))
	local added = dir .. "/added_tokens.json"
	local vocab = lgml.vocab(dir .. "/vocab.json", exists(added) and added or nil)
	local get = getter(w)
	local m = setmetatable({
		be = be, w = w, hp = hp, vocab = vocab, dir = dir,
		enc = load_encoder(get, hp), dec = load_decoder(get, hp),
		tok = load_tokens(vocab),
	}, M)
	m:alloc_state()
	return m
end

--- A loaded Whisper model.
-- Fields: `be`, `w` (weights), `hp` (hyperparameters), `vocab`, `dir`,
-- `tok` (special token ids and language tables), and the state
-- leaves `mel`, `ck`, `cv`, `sk`, `sv`.
-- @type Whisper

--- Allocate the leaves that persist across calls.
-- These are the cross K/V, the self-attention cache, the mel input
-- and the vocabulary biases. The biases suppress all special tokens
-- except end of text, as whisper.cpp does without timestamps. The
-- first token also may not be blank.
function M:alloc_state()
	local be, hp, tok = self.be, self.hp, self.tok
	self.ck, self.cv, self.sk, self.sv = {}, {}, {}, {}
	for i = 1, hp.dec_layers do
		self.ck[i] = be:tensor("f16", { hp.n_audio_ctx, hp.d })
		self.cv[i] = be:tensor("f16", { hp.n_audio_ctx, hp.d })
		self.sk[i] = be:tensor("f16", { hp.n_text_ctx, hp.d })
		self.sv[i] = be:tensor("f16", { hp.n_text_ctx, hp.d })
	end
	self.mel = be:tensor("f32", { hp.n_mels, M.N_FRAMES })
	self.bias_next = bias(be, hp.n_vocab, function(i)
		return i < tok.sot
	end)
	self.bias_first = bias(be, hp.n_vocab, function(i)
		return i < tok.sot and i ~= tok.eot and i ~= tok.space
	end)
	self.bias_lang = bias(be, hp.n_vocab, function(i)
		return tok.lang_code[i] ~= nil
	end)
end

--- Close the weights and the vocabulary.
function M:close()
	self.w:close()
	self.vocab:close()
end

local function layer_norm(g, x, p, eps)
	return g:add(g:mul(g:norm(x, eps), p.w), p.b)
end

local function linear(g, x, p)
	local y = g:mul_mat(p.w, x)
	if p.b then
		y = g:add(y, p.b)
	end
	return y
end

local function mlp(g, x, L)
	return linear(g, g:gelu_erf(linear(g, x, L.fc1)), L.fc2)
end

-- q is (n, H, D) f32; k and v are (m, H, D) f16. Returns (n, H * D).
local function attend(g, q, k, v, mask, n, H, D)
	q = g:permute(q, { 2, 1, 3 })
	k = g:permute(k, { 2, 1, 3 })
	v = g:permute(v, { 2, 1, 3 })
	local o = g:flash_attn(q, k, v, mask, 1 / math.sqrt(D))
	return g:reshape(o, { n, H * D })
end

local function conv_block(g, x, p, stride)
	local oc = p.w:shape()[1]
	return g:gelu_erf(g:add(g:conv1d(p.w, x, stride, 1), g:reshape(p.b, { oc, 1 })))
end

local function encoder_layer(g, x, L, hp)
	local n, H = hp.n_audio_ctx, hp.enc_heads
	local D = hp.d // H
	local h = layer_norm(g, x, L.ln1, hp.eps)
	local q = g:reshape(linear(g, h, L.q), { n, H, D })
	local k = g:cast(g:reshape(linear(g, h, L.k), { n, H, D }), "f16")
	local v = g:cast(g:reshape(linear(g, h, L.v), { n, H, D }), "f16")
	x = g:add(x, linear(g, attend(g, q, k, v, nil, n, H, D), L.o))
	return g:add(x, mlp(g, layer_norm(g, x, L.ln2, hp.eps), L))
end

--- Encode the mel window in `self.mel`.
-- The encoder writes the cross K/V leaves of each decoder layer.
function M:encode()
	local hp, e = self.hp, self.enc
	local g <close> = self.be:graph()
	local x = conv_block(g, self.mel, e.conv1, 1)
	x = conv_block(g, x, e.conv2, 2)
	x = g:add(g:cont(g:transpose(x)), g:cast(e.pos, "f32"))
	for _, L in ipairs(e.layers) do
		x = encoder_layer(g, x, L, hp)
	end
	x = layer_norm(g, x, e.ln_post, hp.eps)
	local outs = {}
	for i, L in ipairs(self.dec.layers) do
		outs[#outs + 1] = g:cpy(linear(g, x, L.ck), self.ck[i])
		outs[#outs + 1] = g:cpy(linear(g, x, L.cv), self.cv[i])
	end
	g:compute(table.unpack(outs))
end

-- Self-attention with the cache: write rows, attend over n_kv rows.
local function self_attn(g, h, L, il, inp, self, hp)
	local n, H = inp.n, hp.dec_heads
	local D = hp.d // H
	local q = g:reshape(linear(g, h, L.q), { n, H, D })
	local kw = g:set_rows(self.sk[il], linear(g, h, L.k), inp.rows)
	local vw = g:set_rows(self.sv[il], linear(g, h, L.v), inp.rows)
	local row, hd = lgml.row_size("f16", hp.d), lgml.row_size("f16", D)
	local k = g:view(kw, { inp.n_kv, H, D }, { row, hd })
	local v = g:view(vw, { inp.n_kv, H, D }, { row, hd })
	return linear(g, attend(g, q, k, v, inp.mask, n, H, D), L.o)
end

local function cross_attn(g, h, L, il, inp, self, hp)
	local n, H, m = inp.n, hp.dec_heads, hp.n_audio_ctx
	local D = hp.d // H
	local q = g:reshape(linear(g, h, L.cq), { n, H, D })
	local k = g:reshape(self.ck[il], { m, H, D })
	local v = g:reshape(self.cv[il], { m, H, D })
	return linear(g, attend(g, q, k, v, nil, n, H, D), L.co)
end

--- Make the decoder graph inputs for ids at n_past.
-- @tparam {int,...} ids token ids
-- @tparam int n_past count of cached tokens before ids
-- @treturn table inputs: `n`, `n_kv`, and the leaves `ids`, `pos`,
-- `rows` and `mask` (causal)
function M:inputs(ids, n_past)
	local be, n = self.be, #ids
	local inp = {
		n = n, n_kv = n_past + n,
		ids = be:tensor("i32", { n }, ids), pos = be:tensor("i32", { n }),
		rows = be:tensor("i64", { n }), mask = be:tensor("f16", { n, n_past + n }),
	}
	inp.pos:fill_range(n_past)
	inp.rows:fill_range(n_past)
	inp.mask:fill_causal(n_past)
	return inp
end

--- Run ids at n_past through the decoder.
-- The keys and values of ids go to the self-attention cache.
-- @tparam {int,...} ids token ids
-- @tparam int n_past count of cached tokens before ids
-- @tparam lgml.tensor bias_t f32 bias over the vocabulary
-- @treturn int the argmax of the last logits plus bias_t
-- @raise "text context full" when n_past + #ids exceeds the text context
function M:decode(ids, n_past, bias_t)
	local hp, d = self.hp, self.dec
	if n_past + #ids > hp.n_text_ctx then
		error("text context full")
	end
	local inp = self:inputs(ids, n_past)
	local g <close> = self.be:graph()
	local x = g:add(g:get_rows(d.tok, inp.ids), g:get_rows(d.pos, inp.pos))
	for il, L in ipairs(d.layers) do
		x = g:add(x, self_attn(g, layer_norm(g, x, L.ln1, hp.eps), L, il, inp, self, hp))
		x = g:add(x, cross_attn(g, layer_norm(g, x, L.ln2, hp.eps), L, il, inp, self, hp))
		x = g:add(x, mlp(g, layer_norm(g, x, L.ln3, hp.eps), L))
	end
	if inp.n > 1 then
		x = g:view(x, { 1, hp.d }, nil, (inp.n - 1) * lgml.row_size("f32", hp.d))
	end
	x = layer_norm(g, x, d.ln, hp.eps)
	local best = g:argmax(g:add(g:mul_mat(d.tok, x), bias_t))
	g:compute(best)
	local id = best:get()[1]
	for _, t in pairs(inp) do
		if type(t) == "userdata" then
			t:close()
		end
	end
	return id
end

--- Detect the language of the encoded window.
-- @treturn int the most likely language token id
function M:detect_language()
	return self:decode({ self.tok.sot }, 0, self.bias_lang)
end

--- Look up a language token.
-- @tparam string s a language code or name; case does not matter
-- @treturn ?int the language token id, or nil
function M:language_id(s)
	return self.tok.lang[string.lower(s)]
end

-- text encoding

-- GPT-2 bytes_to_unicode: the piece character of each byte.
local function byte_chars()
	local c, n = {}, 0
	for b = 0, 255 do
		if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or b >= 174 then
			c[b] = utf8.char(b)
		else
			c[b] = utf8.char(256 + n)
			n = n + 1
		end
	end
	return c
end

local SPACE = { [" "] = true, ["\t"] = true, ["\n"] = true, ["\r"] = true, ["\v"] = true, ["\f"] = true }

-- The class of one UTF-8 character for the GPT-2 pre-tokenizer: "s"
-- (space), "L" (letter), "N" (digit) or "P" (other). Non-ASCII
-- characters count as letters, except common spaces and punctuation.
local function class(ch)
	if SPACE[ch] then
		return "s"
	elseif #ch == 1 then
		return ch:match("%a") and "L" or ch:match("%d") and "N" or "P"
	end
	local cp = utf8.codepoint(ch)
	if cp == 0xa0 or cp == 0x3000 or (cp >= 0x2000 and cp <= 0x200a) then
		return "s"
	elseif (cp >= 0xa1 and cp <= 0xbf) or cp == 0xd7 or cp == 0xf7 or (cp >= 0x2010 and cp <= 0x2bff) or
	    (cp >= 0x3001 and cp <= 0x303f) or (cp >= 0xff01 and cp <= 0xff0f) then
		return "P"
	end
	return "L"
end

local CONTRACTIONS = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" }

-- Splits text as the GPT-2 regex does: contractions, then an optional
-- space and a run of one class, then whitespace. A whitespace run
-- before a word leaves its last character for that word.
local function pretokens(text)
	local ch, cl = {}, {}
	for c in text:gmatch(utf8.charpattern) do
		ch[#ch + 1], cl[#cl + 1] = c, class(c)
	end
	local out, i, n = {}, 1, #ch
	while i <= n do
		local j
		if ch[i] == "'" then
			for _, c in ipairs(CONTRACTIONS) do
				local k = i + #c - 1
				if k <= n and table.concat(ch, "", i, k) == c then
					j = k
					break
				end
			end
		end
		if not j then
			local k = i
			if ch[i] == " " and i < n and cl[i + 1] ~= "s" then
				k = i + 1
			end
			j = k
			while j < n and cl[j + 1] == cl[k] do
				j = j + 1
			end
			if cl[k] == "s" and j < n and j > i then
				j = j - 1
			end
		end
		out[#out + 1] = table.concat(ch, "", i, j)
		i = j + 1
	end
	return out
end

local function pair_key(a, b)
	return #a .. " " .. a .. b
end

-- Byte-level BPE of one pre-token: merge the pair of lowest rank until
-- no pair has a rank.
local function bpe(word, ranks)
	local syms = {}
	for c in word:gmatch(".") do
		syms[#syms + 1] = c
	end
	while #syms > 1 do
		local best, a, b
		for i = 1, #syms - 1 do
			local r = ranks[pair_key(syms[i], syms[i + 1])]
			if r and (not best or r < best) then
				best, a, b = r, syms[i], syms[i + 1]
			end
		end
		if not best then
			break
		end
		local out, i = {}, 1
		while i <= #syms do
			if i < #syms and syms[i] == a and syms[i + 1] == b then
				out[#out + 1] = a .. b
				i = i + 2
			else
				out[#out + 1] = syms[i]
				i = i + 1
			end
		end
		syms = out
	end
	return syms
end

--- Makes a GPT-2 byte-level BPE encoder for a Hugging Face vocabulary.
-- The pre-tokenizer follows the GPT-2 regex. It has no Unicode tables:
-- non-ASCII characters count as letters, except common spaces and
-- punctuation. Special tokens in the text are encoded as plain text.
-- @tparam lgml.vocab vocab the vocabulary
-- @tparam string merges_path the path of merges.txt
-- @treturn[1] function a function(text) that returns token ids
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.encoder(vocab, merges_path)
	local f, err = io.open(merges_path, "rb")
	if not f then
		return nil, err
	end
	local back = {}
	for b, c in pairs(byte_chars()) do
		back[utf8.codepoint(c)] = string.char(b)
	end
	local function unpiece(s)
		local out = {}
		for _, cp in utf8.codes(s) do
			out[#out + 1] = back[cp] or ""
		end
		return table.concat(out)
	end
	local ranks, rank = {}, 0
	for line in f:lines() do
		local a, b = line:match("^(%S+) (%S+)$")
		if a and not line:find("^#version") then
			rank = rank + 1
			ranks[pair_key(unpiece(a), unpiece(b))] = rank
		end
	end
	f:close()
	local ids = {}
	for i = 0, vocab:n() - 1 do
		local p = vocab:piece(i)
		if p and not vocab:special(i) and not ids[p] then
			ids[p] = i
		end
	end
	local clean = require("lgml.tokenizer").clean
	return function(text)
		local out = {}
		for _, w in ipairs(pretokens(clean(text))) do
			for _, s in ipairs(bpe(w, ranks)) do
				if ids[s] then
					out[#out + 1] = ids[s]
				else
					for c in s:gmatch(".") do
						out[#out + 1] = ids[c]
					end
				end
			end
		end
		return out
	end
end

--- Encodes text to token ids with the merges.txt of the model.
-- @tparam string text the text
-- @treturn {int,...} the token ids
-- @raise when the model directory has no merges.txt
function M:encode_text(text)
	if not self.encode_fn then
		self.encode_fn = assert(M.encoder(self.vocab, self.dir .. "/merges.txt"))
	end
	return self.encode_fn(text)
end

--- Build a decoder prompt.
-- The prompt is the optional previous text after <|startofprev|>, then
-- the task tokens. The previous text keeps at most its last
-- n_text_ctx / 2 - 1 tokens.
-- @tparam int lang language token id
-- @tparam[opt] {int,...} prev token ids of the previous window
-- @treturn {int,...} prompt token ids
function M:prompt(lang, prev)
	local tok, p = self.tok, {}
	if prev and #prev > 0 then
		local keep = math.min(#prev, self.hp.n_text_ctx // 2 - 1)
		p[1] = tok.prev
		table.move(prev, #prev - keep + 1, #prev, 2, p)
	end
	table.move({ tok.sot, lang, tok.transcribe, tok.notimestamps }, 1, 4, #p + 1, p)
	return p
end

--- Decode the encoded window greedily.
-- @tparam table opts options:
-- opts.lang: language token id, or nil to detect it.
-- opts.prev: token ids of the previous window.
-- opts.max_tokens: token limit (default n_text_ctx / 2).
-- opts.step: function called after each token; false stops.
-- @treturn {int,...} the text token ids
-- @treturn int the language token id
function M:decode_window(opts)
	local tok = self.tok
	local lang = opts.lang or self:detect_language()
	local prompt = self:prompt(lang, opts.prev)
	local max = math.min(opts.max_tokens or self.hp.n_text_ctx // 2,
	    self.hp.n_text_ctx - #prompt)
	local out = {}
	local id = self:decode(prompt, 0, self.bias_first)
	while id ~= tok.eot and #out < max do
		out[#out + 1] = id
		if opts.step and opts.step() == false then
			break
		end
		if #out >= max then
			break
		end
		id = self:decode({ id }, #prompt + #out - 1, self.bias_next)
	end
	return out, lang
end

--- Transcribe 16 kHz audio in sequential 30 s windows.
-- Each window sees the text of the one before as its prompt.
-- @tparam lgml.audio audio the input audio at 16 kHz
-- @tparam[opt] table opts options as for `decode_window` (lang,
-- max_tokens, step), plus:
-- opts.cancelled: function; true stops after the current window.
-- opts.prompt: token ids of an initial prompt (see `encode_text`). The
-- first window sees them as its previous text.
-- @treturn table result: `text`, `language` (code), `language_name`,
-- `duration` (seconds) and `segments` (list of `id`, `start`, `end`,
-- `text`)
-- @raise when the audio rate is not 16 kHz
function M:transcribe(audio, opts)
	opts = opts or {}
	assert(audio:rate() == M.RATE, "audio must be 16 kHz")
	local n = audio:len()
	local segs, texts, lang, prev = {}, {}, opts.lang, opts.prompt
	for start = 0, math.max(n - 1, 0), M.N_SAMPLES do
		audio:mel(self.mel, start)
		self:encode()
		local ids
		ids, lang = self:decode_window({
			lang = lang, prev = prev, max_tokens = opts.max_tokens, step = opts.step,
		})
		prev = ids
		local text = self.vocab:decode(ids):gsub("^%s+", ""):gsub("%s+$", "")
		segs[#segs + 1] = {
			id = #segs, start = start / M.RATE,
			["end"] = math.min(start + M.N_SAMPLES, n) / M.RATE, text = text,
		}
		if text ~= "" then
			texts[#texts + 1] = text
		end
		if opts.cancelled and opts.cancelled() then
			break
		end
	end
	return {
		text = table.concat(texts, " "), language = self.tok.lang_code[lang],
		language_name = self.tok.lang_name[lang], duration = n / M.RATE, segments = segs,
	}
end

return M
