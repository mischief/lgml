-- Tokenizer encode and decode on a tiny vocab of each pre-tokenizer.
-- Valid UTF-8 must round-trip; the streaming decoder must agree.
-- gemma4 decodes U+2581 as a space. The tokenizer.json vocab
-- normalizes to NFC, and [MASK] takes the white space before it.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "../test/?.lua;" .. package.path
local lgml = require("lgml")
local gguf = require("gguf")
local json = require("lgml.json")
local bpe_vocab = require("bpe_vocab")
local tokenizer = require("lgml.tokenizer")

local tks = {}
for _, pre in ipairs({ "qwen2", "llama-bpe" }) do
	tks[#tks + 1] = assert(lgml.tokenizer(fuzz.file(gguf.encode((bpe_vocab.kv(pre))))))
end
local gemma4 = assert(lgml.tokenizer(fuzz.file(gguf.encode((bpe_vocab.gemma4_kv())))))
tks[#tks + 1] = gemma4
local hf = assert(lgml.tokenizer(fuzz.file(assert(json.encode((bpe_vocab.hf_json()))))))
tks[#tks + 1] = hf

return function(s)
	for _, tk in ipairs(tks) do
		for _, special in ipairs({ true, false }) do
			local want = s
			if tk == gemma4 then
				want = s:gsub("\u{2581}", " ")
			elseif tk == hf then
				want = lgml.nfc(s)
				if special then
					want = want:gsub("[ \t-\r]+%[MASK%]", "[MASK]")
				end
			end
			local ids = tk:encode(s, { special = special, add_bos = false })
			local back = tk:decode(ids)
			if utf8.len(s) then
				assert(back == want, "round trip")
			end
			local dec = tokenizer.decoder(tk)
			local parts = {}
			for _, id in ipairs(ids) do
				parts[#parts + 1] = dec:push(id)
			end
			parts[#parts + 1] = dec:flush()
			assert(table.concat(parts) == tokenizer.clean(back), "streaming decoder")
		end
	end
end
