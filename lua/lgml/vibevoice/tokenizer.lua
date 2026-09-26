--- Byte-level BPE for the VibeVoice gguf.
-- The file has tokens but no merges. A merge takes the pair whose
-- joined piece has the lowest id. Qwen2 ids follow merge order, so this
-- matches the merge list.
-- @classmod lgml.vibevoice.tokenizer
local M = {}
M.__index = M

-- GPT-2 byte to unicode table: byte -> UTF-8 string.
local function byte_table()
	local t, n = {}, 0
	for b = 0, 255 do
		local keep = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or
		    (b >= 174 and b <= 255)
		if keep then
			t[b] = utf8.char(b)
		else
			t[b] = utf8.char(256 + n)
			n = n + 1
		end
	end
	return t
end

local BYTES = byte_table()

--- Make a tokenizer.
-- @tparam {string,...} tokens list of pieces; id i is at index i + 1
-- @treturn lgml.vibevoice.tokenizer the tokenizer
-- @static
function M.new(tokens)
	local id = {}
	for i, s in ipairs(tokens) do
		if id[s] == nil then
			id[s] = i - 1
		end
	end
	return setmetatable({ id = id, cache = {}, ncache = 0 }, M)
end

-- Character classes for the Qwen2 pre-tokenizer. Code points above
-- 0x7f count as letters, except general punctuation and CJK marks.
local function class(cp)
	if cp < 0x80 then
		local c = string.char(cp)
		if c:match("%a") then
			return "L"
		elseif c:match("%d") then
			return "N"
		elseif c == "\r" or c == "\n" then
			return "NL"
		elseif c:match("%s") then
			return "S"
		end
		return "P"
	end
	if (cp >= 0x2000 and cp <= 0x206f) or (cp >= 0x3000 and cp <= 0x303f) or
	    (cp >= 0xff00 and cp <= 0xff0f) or cp == 0xa0 or cp == 0xab or cp == 0xbb then
		return cp == 0xa0 and "S" or "P"
	end
	return "L"
end

local function is_space(k)
	return k == "S" or k == "NL"
end

local CONTRACTIONS = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" }

local function contraction(ch, i)
	if ch[i] ~= "'" then
		return nil
	end
	for _, c in ipairs(CONTRACTIONS) do
		local ok = true
		for j = 2, #c do
			local x = ch[i + j - 1]
			if x == nil or x:lower() ~= c:sub(j, j) then
				ok = false
				break
			end
		end
		if ok then
			return i + #c
		end
	end
	return nil
end

local function run(kind, i, pred)
	while kind[i] ~= nil and pred(kind[i]) do
		i = i + 1
	end
	return i
end

local function letter(k)
	return k == "L"
end

local function punct(k)
	return k == "P"
end

local function newline(k)
	return k == "NL"
end

-- End (exclusive) of the whitespace piece that starts at i.
local function space_piece(kind, i)
	local j = run(kind, i, is_space)
	local last_nl
	for x = i, j - 1 do
		if kind[x] == "NL" then
			last_nl = x
		end
	end
	if last_nl then
		return last_nl + 1
	end
	if kind[j] == nil or j - i == 1 then
		return j
	end
	return j - 1
end

-- End (exclusive) of the piece that starts at i.
local function piece_end(ch, kind, i)
	local e = contraction(ch, i)
	if e then
		return e
	end
	local k = kind[i]
	if k == "L" then
		return run(kind, i, letter)
	end
	if k ~= "N" and k ~= "NL" and kind[i + 1] == "L" then
		return run(kind, i + 1, letter)
	end
	if k == "N" then
		return i + 1
	end
	local j = i
	if ch[j] == " " and kind[j + 1] == "P" then
		j = j + 1
	end
	if kind[j] == "P" then
		return run(kind, run(kind, j, punct), newline)
	end
	return space_piece(kind, i)
end

--- Split text into Qwen2 pre-tokenizer pieces.
-- @string text UTF-8 text
-- @treturn {string,...} the pieces
-- @static
function M.pieces(text)
	local ch, kind = {}, {}
	for _, cp in utf8.codes(text) do
		ch[#ch + 1] = utf8.char(cp)
		kind[#kind + 1] = class(cp)
	end
	local out, i = {}, 1
	while i <= #ch do
		local e = piece_end(ch, kind, i)
		out[#out + 1] = table.concat(ch, "", i, e - 1)
		i = e
	end
	return out
end

--- Append the token ids of one pre-tokenizer piece to out.
-- The tokenizer caches the ids of up to 65536 pieces.
-- @string piece a piece from pieces
-- @tparam {int,...} out list to append to
-- @raise when a merged symbol is not in the vocabulary
function M:bpe(piece, out)
	local ids = self.cache[piece]
	if ids == nil then
		local sym = {}
		for i = 1, #piece do
			sym[i] = BYTES[piece:byte(i)]
		end
		while #sym > 1 do
			local best, bid
			for i = 1, #sym - 1 do
				local v = self.id[sym[i] .. sym[i + 1]]
				if v ~= nil and (bid == nil or v < bid) then
					best, bid = i, v
				end
			end
			if best == nil then
				break
			end
			sym[best] = sym[best] .. table.remove(sym, best + 1)
		end
		ids = {}
		for i, s in ipairs(sym) do
			ids[i] = assert(self.id[s], "token not in vocab")
		end
		if self.ncache < 65536 then
			self.cache[piece] = ids
			self.ncache = self.ncache + 1
		end
	end
	table.move(ids, 1, #ids, #out + 1, out)
end

--- Token ids of text.
-- @string text valid UTF-8 text
-- @treturn {int,...} token ids
function M:encode(text)
	local out = {}
	for _, p in ipairs(M.pieces(text)) do
		self:bpe(p, out)
	end
	return out
end

return M
