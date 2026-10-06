-- Tokenizer tests. Output is TAP. A tiny vocab built here always runs.
-- LGML_VOCAB_DIR names the llama.cpp vocab test files (models/ in its
-- tree); its qwen35 and gemma-4 vocabs also give the chat and special
-- token checks. No test needs model weights.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local lgml = require("lgml")
local gguf = require("gguf")
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

local function skip(name, why)
	n = n + 1
	print("ok " .. n .. " - " .. name .. " # SKIP " .. why)
end

local function readfile(path)
	local f = io.open(path, "rb")
	if not f then
		return nil
	end
	local s = f:read("a")
	f:close()
	return s
end

local function same(a, b)
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

local dir = os.getenv("LGML_VOCAB_DIR") or ""

-- Split the .inp file on "\n__ggml_vocab_test__\n", as test-tokenizer-0 does.
local function cases(inp)
	local out, pos, sep = {}, 1, "\n__ggml_vocab_test__\n"
	while pos <= #inp do
		local a, b = inp:find(sep, pos, true)
		if not a then
			out[#out + 1] = inp:sub(pos)
			break
		end
		out[#out + 1] = inp:sub(pos, a - 1)
		pos = b + 1
	end
	return out
end

local function vocab_test(name)
	local path = dir .. "/ggml-vocab-" .. name .. ".gguf"
	local inp = readfile(path .. ".inp")
	local outf = readfile(path .. ".out")
	if dir == "" then
		skip(name .. " vocab", "LGML_VOCAB_DIR not set")
		return
	elseif not inp or not outf or not readfile(path) then
		skip(name .. " vocab", "no files in " .. dir)
		return
	end
	local tk <close> = assert(lgml.tokenizer(path))
	ok(tk:n_vocab() > 0, name .. ": loaded")
	local texts = cases(inp)
	local want = {}
	for line in outf:gmatch("([^\n]*)\n") do
		local ids = {}
		for w in line:gmatch("%d+") do
			ids[#ids + 1] = tonumber(w)
		end
		want[#want + 1] = ids
	end
	ok(#texts == #want, name .. ": case count " .. #texts)
	local bad = 0
	for i, text in ipairs(texts) do
		local got = tk:encode(text, { special = false, add_bos = false })
		if not same(got, want[i] or {}) then
			bad = bad + 1
			io.stderr:write(string.format("case %d %q\n  got  %s\n  want %s\n",
			    i, text, table.concat(got, " "),
			    table.concat(want[i] or {}, " ")))
		end
		local back = tk:decode(got)
		if back ~= text and utf8.len(text) then
			bad = bad + 1
			io.stderr:write(string.format("case %d: decode %q\n", i, back))
		end
	end
	ok(bad == 0, name .. ": all cases match and round-trip")
end

-- A tiny vocab built here, so these checks need no outside files.
local function synth_test(pre)
	local kv, id = bpe_vocab.kv(pre)
	local path = os.tmpname()
	gguf.write(path, kv)
	local tk <close> = assert(lgml.tokenizer(path))
	os.remove(path)
	local name = "tiny " .. pre
	ok(tk:n_vocab() == 256 + #bpe_vocab.merges + #bpe_vocab.whole + #bpe_vocab.special,
	    name .. ": loaded")
	ok(same(tk:encode("hello world"), { id.hello, id["Ġworld"] }), name .. ": merges")
	local digits = pre == "qwen2" and { id["1"], id["2"], id["3"] } or { id["123"] }
	ok(same(tk:encode("123"), digits), name .. ": digit groups")
	local xyz = pre == "qwen2" and { id.x, id.y, id.z } or { id.xyz }
	ok(same(tk:encode("xyz"), xyz), name .. ": whole-word lookup")
	ok(same(tk:encode("<|im_start|>hello<|im_end|>"),
	    { id["<|im_start|>"], id.hello, id["<|im_end|>"] }), name .. ": special tokens")
	ok(#tk:encode("<|im_start|>", { special = false }) > 1, name .. ": special as text")
	ok(tk:special().eos == id["<|im_end|>"], name .. ": eos")
	ok(same(tk:encode("\0\0\0"), { id["\u{100}\u{100}"], id["\u{100}"] }),
	    name .. ": merge of token 0 with itself")
	math.randomseed(7)
	local rt = true
	for _ = 1, 2000 do
		local b = {}
		for i = 1, math.random(0, 40) do
			b[i] = string.char(math.random(0, 255))
		end
		local s = table.concat(b)
		if utf8.len(s) and tk:decode(tk:encode(s)) ~= s then
			rt = false
			io.stderr:write(string.format("%s: round trip %q\n", name, s))
		end
	end
	ok(rt, name .. ": random bytes round-trip")
end

-- The tiny gemma4 vocab: character BPE, U+2581 spaces, byte fallback.
local function synth_gemma4()
	local kv, id = bpe_vocab.gemma4_kv()
	local path = os.tmpname()
	gguf.write(path, kv)
	local tk <close> = assert(lgml.tokenizer(path))
	os.remove(path)
	local name = "tiny gemma4"
	local nb = { add_bos = false }
	local sp = tk:special()
	ok(sp.bos == id["<bos>"] and sp.add_bos, name .. ": add_bos forced on")
	ok(same(tk:encode("hello world"), { id["<bos>"], id.hello, id["\u{2581}world"] }),
	    name .. ": merges and bos")
	ok(same(tk:encode(" world", nb), { id["\u{2581}world"] }), name .. ": space escape")
	ok(same(tk:encode("\u{2581}world", nb), { id["\u{2581}world"] }), name .. ": literal U+2581")
	ok(same(tk:encode("xyz", nb), { id.x, id.y, id.z }), name .. ": no whole-word lookup")
	ok(same(tk:encode("\n\n\n", nb), { id["\n\n\n"] }), name .. ": newline run whole")
	ok(same(tk:encode("\n\n\n\n", nb), { id["\n\n"], id["\n\n"] }), name .. ": newline run merges")
	ok(same(tk:encode("hel\nlo", nb), { id.he, id.l, id["\n"], id.l, id.o }),
	    name .. ": newline splits")
	ok(same(tk:encode("Q\u{e9}我", nb), { id["<0x51>"], id["<0xC3>"], id["<0xA9>"], id["我"] }),
	    name .. ": byte fallback")
	ok(same(tk:encode("a\xffb", nb), { id["<0x61>"], id["<0xEF>"], id["<0xBF>"], id["<0xBD>"],
	    id["<0x62>"] }), name .. ": invalid utf-8")
	ok(tk:decode({ id.hello, id["\u{2581}world"], id["<0xC3>"], id["<0xA9>"] }) == "hello world\u{e9}",
	    name .. ": decode")
	ok(same(tk:encode("<|turn>hello<turn|>", nb), { id["<|turn>"], id.hello, id["<turn|>"] }),
	    name .. ": special tokens")
	ok(same(tk:encode('<|"|>', { add_bos = false, special = false }), { id['<|"|>'] }),
	    name .. ": user token splits without special")
	local eog = {}
	for _, e in ipairs(sp.eog) do
		eog[e] = true
	end
	ok(eog[id["<eos>"]] and eog[id["<turn|>"]] and eog[id["<|tool_response>"]] and
	    not eog[id["</s>"]], name .. ": eog")
	ok(select(2, tk:token(id["</s>"])) == "normal", name .. ": </s> is normal")
	math.randomseed(11)
	local rt = true
	local dec = require("lgml.tokenizer")
	for _ = 1, 2000 do
		local b = {}
		for i = 1, math.random(0, 40) do
			b[i] = string.char(math.random(0, 255))
		end
		local s = table.concat(b):gsub("\xe2\x96\x81", "")
		if utf8.len(s) then
			local ids = tk:encode(s, nb)
			local d, parts = dec.decoder(tk), {}
			for _, i in ipairs(ids) do
				parts[#parts + 1] = d:push(i)
			end
			parts[#parts + 1] = d:flush()
			if tk:decode(ids) ~= s or table.concat(parts) ~= s then
				rt = false
				io.stderr:write(string.format("%s: round trip %q\n", name, s))
			end
		end
	end
	ok(rt, name .. ": random bytes round-trip")
end


-- A tiny tokenizer.json: ByteLevel with the GPT-2 regex, NFC, and
-- added tokens that split first.
local function synth_json()
	local json = require("lgml.json")
	local t, id = bpe_vocab.hf_json()
	local path = os.tmpname()
	local f = assert(io.open(path, "wb"))
	f:write(assert(json.encode(t)))
	f:close()
	local tk <close> = assert(lgml.tokenizer(path))
	local name = "tiny tokenizer.json"
	ok(tk:n_vocab() == id["[MASK]"] + 1, name .. ": loaded")
	ok(same(tk:encode("hello world"), { id.hello, id["Ġworld"] }), name .. ": merges")
	ok(same(tk:encode("123 12"), { id["123"], id["Ġ"], id["12"] }), name .. ": digit runs")
	ok(same(tk:encode("x's"), { id.x, id["'"], id.s }), name .. ": contraction")
	ok(same(tk:encode("[CLS]hello[SEP]"), { id["[CLS]"], id.hello, id["[SEP]"] }),
	    name .. ": special tokens")
	ok(same(tk:encode("x  y"), { id.x, id["  "], id.y }), name .. ": added space run")
	ok(same(tk:encode("x [MASK] y"), { id.x, id["[MASK]"], id["Ġ"], id.y }),
	    name .. ": lstrip")
	local e = "\u{e9}"
	ok(same(tk:encode("e\u{301}"), { id[utf8.char(0xc3)], id[utf8.char(0xa9)] }) and
	    tk:decode(tk:encode("e\u{301}")) == e, name .. ": NFC")
	local sp = tk:special()
	ok(sp.bos == id["[CLS]"] and sp.eos == id["[SEP]"] and sp.tokens["[MASK]"] == id["[MASK]"],
	    name .. ": template bos and eos")
	math.randomseed(11)
	local rt = true
	for _ = 1, 2000 do
		local b = {}
		for i = 1, math.random(0, 40) do
			b[i] = string.char(math.random(0, 127))
		end
		local s = table.concat(b)
		if tk:decode(tk:encode(s)) ~= s:gsub(" +%[MASK%]", "[MASK]") then
			rt = false
			io.stderr:write(string.format("%s: round trip %q\n", name, s))
		end
	end
	ok(rt, name .. ": random ASCII round-trips")
	t.normalizer = { type = "NFKC" }
	f = assert(io.open(path, "wb"))
	f:write(assert(json.encode(t)))
	f:close()
	local bad, err = lgml.tokenizer(path)
	ok(bad == nil and err:find("normalizer"), name .. ": other normalizers fail")
	os.remove(path)
end

-- NFC with utf8proc: composition, canonical order, Hangul, exclusions.
local function nfc_test()
	ok(lgml.nfc("abc") == "abc", "nfc: ASCII")
	ok(lgml.nfc("e\u{301}") == "\u{e9}", "nfc: compose")
	ok(lgml.nfc("a\u{302}\u{323}") == "\u{1ead}", "nfc: canonical order")
	ok(lgml.nfc("\u{1100}\u{1161}\u{11a8}") == "\u{ac01}", "nfc: Hangul")
	ok(lgml.nfc("\u{2126}") == "\u{3a9}", "nfc: singleton")
	ok(lgml.nfc("\u{958}") == "\u{915}\u{93c}", "nfc: composition exclusion")
	ok(lgml.nfc("\xff\u{301}") == "\xff\u{301}", "nfc: invalid UTF-8 unchanged")
end

synth_test("qwen2")
synth_test("llama-bpe")
synth_gemma4()
synth_json()
nfc_test()

vocab_test("qwen2")
vocab_test("qwen35")
vocab_test("llama-bpe")
vocab_test("gemma-4")

-- The vocab file of a llama.cpp test, or nil after a skip.
local function vocab_file(name)
	local path = dir .. "/ggml-vocab-" .. name .. ".gguf"
	if dir == "" then
		skip(name .. " special tokens", "LGML_VOCAB_DIR not set")
		return nil
	end
	local f = io.open(path, "rb")
	if not f then
		skip(name .. " special tokens", "no " .. path)
		return nil
	end
	f:close()
	return path
end

-- Chat and special tokens of the Qwen vocab. It has the chat control
-- tokens; user tokens such as <think> are [PADn] placeholders there.
local qwen = vocab_file("qwen35")
if qwen then
	local tk <close> = assert(lgml.tokenizer(qwen))
	local sp = tk:special()
	ok(sp.tokens["<|im_start|>"] == 151644, "im_start id")
	ok(sp.eos == 151643 and sp.eot == 151645, "eos and eot")
	local ids = tk:encode("<|im_start|>user\nhi<|im_end|>\n")
	ok(same(ids, { 151644, 872, 198, 6023, 151645, 198 }), "chat encode")
	local plain = tk:encode("<|im_start|>", { special = false })
	ok(#plain > 1, "control token as text without special")
	local user = tk:encode("[PAD151667]", { special = false })
	ok(same(user, { 151667 }), "user token splits without special")
	local text = "<|im_start|>assistant\n<think>\nok</think>\n\n" ..
	    "<tool_call>\n{\"name\": \"f\"}\n</tool_call><|im_end|>"
	ok(tk:decode(tk:encode(text)) == text, "chat round-trip")
	ok(tk:decode(tk:encode(text), { special = false }):find("im_start") == nil,
	    "decode drops control tokens")
	local piece, kind = tk:token(151644)
	ok(piece == "<|im_start|>" and kind == "control", "token piece")
	ok(tk:token(9707) == "Hello", "normal piece")
	ok(same(tk:encode("a\xffb"), tk:encode("a\u{fffd}b")), "invalid utf-8")
	ok(#tk:encode("") == 0, "empty text")
	ok(same(tk:encode("</s>"), { 128247 }), "forced control token")

	-- Random bytes must not crash. Valid UTF-8 must round-trip.
	math.randomseed(42)
	local rt = true
	for _ = 1, 2000 do
		local b = {}
		for i = 1, math.random(0, 40) do
			b[i] = string.char(math.random(0, 255))
		end
		local s = table.concat(b)
		local back = tk:decode(tk:encode(s))
		if utf8.len(s) and back ~= s then
			rt = false
		end
	end
	ok(rt, "random bytes")

	-- The streaming decoder emits only complete UTF-8.
	local dec = require("lgml.tokenizer").decoder(tk)
	local emoji = "a 😶‍🌫️ 我想 🦙!"
	local parts, whole = {}, true
	for _, id in ipairs(tk:encode(emoji)) do
		local piece = dec:push(id)
		whole = whole and utf8.len(piece) ~= nil
		parts[#parts + 1] = piece
	end
	parts[#parts + 1] = dec:flush()
	ok(whole and table.concat(parts) == emoji, "streaming decoder")
end

-- Special tokens of the Gemma-4 vocab. Ids are from llama.cpp.
local g4 = vocab_file("gemma-4")
if g4 then
	local tk <close> = assert(lgml.tokenizer(g4))
	local sp = tk:special()
	ok(sp.bos == 2 and sp.eos == 1 and sp.pad == 0 and sp.add_bos, "gemma4: bos, eos, pad")
	table.sort(sp.eog)
	ok(same(sp.eog, { 1, 50, 106 }), "gemma4: eog is <eos>, <|tool_response>, <turn|>")
	ok(sp.tokens["<|turn>"] == 105 and sp.tokens["<|tool_call>"] == 48 and
	    sp.tokens['<|"|>'] == 52 and sp.tokens["<|channel>"] == 100, "gemma4: special ids")
	ok(select(2, tk:token(212)) == "normal", "gemma4: </s> is normal")
	local text = "<|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
	ok(same(tk:encode(text), { 2, 105, 2364, 107, 2202, 106, 107, 105, 4368, 107, 100,
	    45518, 107, 101 }), "gemma4: chat encode")
	ok(tk:decode(tk:encode(text, { add_bos = false })) == text, "gemma4: chat round-trip")
	ok(same(tk:encode("<|turn>", { special = false, add_bos = false }), { 236820, 236909, 887, 236813 }),
	    "gemma4: control token as text without special")
end

print("1.." .. n)
os.exit(failed and 1 or 0)
