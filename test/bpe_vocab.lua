-- Tiny BPE vocabs as gguf metadata. In the byte-level vocab, ids 0 to
-- 255 are the bytes, then merge results, then special tokens.
local gguf = require("gguf")

local M = {}

-- GPT-2 bytes_to_unicode: byte value to piece text.
local function byte_pieces()
	local p, n = {}, 0
	for b = 0, 255 do
		if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or b >= 174 then
			p[b] = utf8.char(b)
		else
			p[b] = utf8.char(256 + n)
			n = n + 1
		end
	end
	return p
end

M.merges = {
	"h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "l d", "Ġwor ld",
	"1 2", "12 3", "\u{100} \u{100}",
}

-- In the vocab, but no merge makes it.
M.whole = { "xyz" }

M.special = {
	{ "<|im_start|>", 3 }, { "<|im_end|>", 3 }, { "<think>", 4 },
}

-- Returns the kv list for pre ("qwen2" or "llama-bpe") and the
-- piece -> id table.
function M.kv(pre)
	local tokens, types, ids = {}, {}, {}
	local function add(piece, ty)
		tokens[#tokens + 1] = piece
		types[#types + 1] = ty
		ids[piece] = #tokens - 1
	end
	local bp = byte_pieces()
	for b = 0, 255 do
		add(bp[b], 1)
	end
	for _, m in ipairs(M.merges) do
		add((m:gsub(" ", "")), 1)
	end
	for _, w in ipairs(M.whole) do
		add(w, 1)
	end
	for _, s in ipairs(M.special) do
		add(s[1], s[2])
	end
	local kv = {
		{ "tokenizer.ggml.model", gguf.STRING, "gpt2" },
		{ "tokenizer.ggml.pre", gguf.STRING, pre },
		{ "tokenizer.ggml.tokens", { gguf.ARRAY, gguf.STRING }, tokens },
		{ "tokenizer.ggml.token_type", { gguf.ARRAY, gguf.INT32 }, types },
		{ "tokenizer.ggml.merges", { gguf.ARRAY, gguf.STRING }, M.merges },
		{ "tokenizer.ggml.eos_token_id", gguf.UINT32, ids["<|im_end|>"] },
	}
	return kv, ids
end

-- A tiny gemma4 vocab: BPE over characters, U+2581 for space, and
-- <0xXX> byte tokens. <0xFF> is missing; valid UTF-8 never needs it.
M.gemma4_merges = {
	"h e", "l l", "he ll", "hell o", "\u{2581} w", "o r", "\u{2581}w or",
	"l d", "\u{2581}wor ld", "\n \n",
}

M.gemma4_chars = {
	"\u{2581}", "h", "e", "l", "o", "w", "r", "d", "\n", "x", "y", "z", "我",
}

M.gemma4_special = {
	{ "<pad>", 3 }, { "<eos>", 3 }, { "<bos>", 3 }, { "<unk>", 3 },
	{ "<|turn>", 3 }, { "<turn|>", 3 }, { "<|tool_response>", 4 },
	{ '<|"|>', 4 }, { "</s>", 1 },
}

-- Returns the gemma4 kv list and the piece -> id table.
function M.gemma4_kv()
	local tokens, types, ids = {}, {}, {}
	local function add(piece, ty)
		tokens[#tokens + 1] = piece
		types[#types + 1] = ty
		ids[piece] = #tokens - 1
	end
	for _, s in ipairs(M.gemma4_special) do
		add(s[1], s[2])
	end
	for b = 0, 254 do
		add(string.format("<0x%02X>", b), 6)
	end
	for _, c in ipairs(M.gemma4_chars) do
		add(c, 1)
	end
	for _, m in ipairs(M.gemma4_merges) do
		add((m:gsub(" ", "")), 1)
	end
	add("\n\n\n", 1)
	add("xyz", 1)
	local kv = {
		{ "tokenizer.ggml.model", gguf.STRING, "gemma4" },
		{ "tokenizer.ggml.tokens", { gguf.ARRAY, gguf.STRING }, tokens },
		{ "tokenizer.ggml.token_type", { gguf.ARRAY, gguf.INT32 }, types },
		{ "tokenizer.ggml.merges", { gguf.ARRAY, gguf.STRING }, M.gemma4_merges },
		{ "tokenizer.ggml.bos_token_id", gguf.UINT32, ids["<bos>"] },
		{ "tokenizer.ggml.eos_token_id", gguf.UINT32, ids["<eos>"] },
		{ "tokenizer.ggml.add_bos_token", gguf.BOOL, false },
	}
	return kv, ids
end

-- Added tokens of the tokenizer.json vocab, as ModernBERT has them.
M.hf_added = {
	{ content = "  ", special = false }, { content = "[UNK]", special = true },
	{ content = "[CLS]", special = true }, { content = "[SEP]", special = true },
	{ content = "[PAD]", special = true }, { content = "[MASK]", special = true, lstrip = true },
}

-- Returns a Hugging Face tokenizer.json table for the byte-level vocab
-- with the ByteLevel pre-tokenizer, NFC and a [CLS] A [SEP] template,
-- and the piece -> id table. words adds pieces (byte-level text) with
-- no merge. Odd merges are strings, even ones pairs.
function M.hf_json(words)
	local json = require("lgml.json")
	local vocab, ids, n = {}, {}, 0
	local function add(piece)
		if not ids[piece] then
			vocab[piece], ids[piece], n = n, n, n + 1
		end
	end
	local bp = byte_pieces()
	for b = 0, 255 do
		if b ~= 0xc0 and b ~= 0xc1 and b < 0xf5 then
			add(bp[b])
		end
	end
	local merges = json.array()
	for i, m in ipairs(M.merges) do
		add((m:gsub(" ", "")))
		merges[i] = i % 2 == 1 and m or json.array({ m:match("^(%S+) (%S+)$") })
	end
	for _, w in ipairs(M.whole) do
		add(w)
	end
	for _, w in ipairs(words or {}) do
		add(w)
	end
	local added = json.array()
	for i, a in ipairs(M.hf_added) do
		add(a.content)
		added[i] = { id = ids[a.content], content = a.content, special = a.special,
			lstrip = a.lstrip or false, rstrip = false, normalized = not a.special,
			single_word = false }
	end
	local function special(name)
		return { SpecialToken = { id = name, type_id = 0 } }
	end
	return {
		version = "1.0", truncation = json.null, padding = json.null,
		added_tokens = added,
		normalizer = { type = "NFC" },
		pre_tokenizer = { type = "ByteLevel", add_prefix_space = false, trim_offsets = true,
			use_regex = true },
		post_processor = { type = "TemplateProcessing",
			single = json.array({ special("[CLS]"), { Sequence = { id = "A", type_id = 0 } },
				special("[SEP]") }) },
		decoder = { type = "ByteLevel" },
		model = { type = "BPE", dropout = json.null, unk_token = json.null,
			byte_fallback = false, vocab = vocab, merges = merges },
	}, ids
end

return M
