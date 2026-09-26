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

return M
