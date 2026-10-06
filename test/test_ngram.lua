-- lgml.ngram on a small IQ4_NL table. The expected rows come from the
-- segment form of the hash in the reference model. Output is TAP.
local lgml = require("lgml")
local gguf = dofile((arg[0]:match("^(.*)/") or ".") .. "/gguf.lua")

local n = 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		os.exit(1)
	end
end

local DIM, ROWS, HEADS, EOS, IMG = 64, 3001, 4, 2, 77
local MULT = { 1000003, 999983, 7919 }
local VOCAB, OFF = {}, {}
for h = 1, HEADS do
	VOCAB[h] = 700 + h
	OFF[h] = h == 1 and 0 or OFF[h - 1] + VOCAB[h - 1]
end

-- IQ4_NL: an f16 scale, then 16 bytes; element j is the low nibble of
-- byte j, element j + 16 the high nibble.
local KV = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 }
local SCALE = { [0] = { 0x3c00, 1.0 }, { 0x3800, 0.5 }, { 0x3400, 0.25 } }

local function block(row, b)
	local q = {}
	for j = 0, 15 do
		q[j + 1] = (row * 31 + b * 7 + j * 13) & 255
	end
	return SCALE[(row + b) % 3], q
end

local function row_bytes(row)
	local s = {}
	for b = 0, DIM // 32 - 1 do
		local d, q = block(row, b)
		s[#s + 1] = string.pack("<I2", d[1]) .. string.char(table.unpack(q))
	end
	return table.concat(s)
end

local function row_values(row)
	local y = {}
	for b = 0, DIM // 32 - 1 do
		local d, q = block(row, b)
		for j = 0, 15 do
			y[b * 32 + j + 1] = d[2] * KV[(q[j + 1] & 15) + 1]
			y[b * 32 + j + 17] = d[2] * KV[(q[j + 1] >> 4) + 1]
		end
	end
	return y
end

local data = {}
for r = 0, ROWS - 1 do
	data[r + 1] = row_bytes(r)
end
local path = os.tmpname()
gguf.write(path, {
	{ "general.architecture", gguf.STRING, "t" },
	{ "t.ple.ngram_size", gguf.UINT32, 3 },
	{ "t.ple.heads_per_ngram", gguf.UINT32, 2 },
	{ "t.ple.eos_token_id", gguf.UINT32, EOS },
	{ "t.ple.image_token_id", gguf.UINT32, IMG },
	{ "t.ple.layer_multipliers", { gguf.ARRAY, gguf.UINT64 }, MULT },
	{ "t.ple.head_offsets", { gguf.ARRAY, gguf.UINT64 }, OFF },
	{ "t.ple.head_vocab_sizes", { gguf.ARRAY, gguf.UINT64 }, VOCAB },
}, { { "per_layer_token_embd.weight", { ROWS, DIM }, table.concat(data), 20 } })
local nometa = os.tmpname()
gguf.write(nometa, { { "general.architecture", gguf.STRING, "t" } },
	{ { "per_layer_token_embd.weight", { ROWS, DIM }, table.concat(data), 20 } })

-- The reference shifts a token history in which an EOS at or before
-- position p - s ends shift s at p; positions before the start are EOS.
local function want_rows(ids, prev)
	local hist = { EOS, EOS }
	for i = math.max(1, #prev - 1), #prev do
		hist[#hist + 1] = prev[i]
	end
	hist = { table.unpack(hist, #hist - 1) }
	for _, t in ipairs(ids) do
		hist[#hist + 1] = t
	end
	local rows = {}
	for p = 3, #hist do
		local start = 1
		for q = 1, p - 1 do
			if hist[q] == EOS then
				start = q + 1
			end
		end
		local ctx = {}
		for s = 0, 2 do
			ctx[s] = p - s >= start and hist[p - s] or EOS
		end
		for ng = 2, 3 do
			local mixed = ctx[0] * MULT[1]
			for j = 1, ng - 1 do
				mixed = mixed ~ (ctx[j] * MULT[j + 1])
			end
			for g = 1, 2 do
				local h = (ng - 2) * 2 + g
				rows[#rows + 1] = mixed % VOCAB[h] + OFF[h]
			end
		end
	end
	return rows
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

local function floats(s)
	local t = {}
	for i = 1, #s // 4 do
		t[i] = string.unpack("<f", s, (i - 1) * 4 + 1)
	end
	return t
end

local ids = { 5, 9, 5, 9, 5, 1, EOS, 7, 7, 7, 7, 3, IMG, 2999 }
local prev = { 11, 12, 13 }

local r <close> = assert(lgml.ngram.open(path))
local info = r:info()
ok(info.heads == HEADS and info.head_dim == DIM and info.dim == HEADS * DIM and
	info.rows == ROWS and info.type == "iq4_nl" and info.ngram_size == 3 and
	info.heads_per_ngram == 2 and info.eos == EOS and info.image_token == IMG and
	info.row_bytes == 36, "info")
print("# mode " .. info.mode .. (info.fell_back and " (fell back)" or ""))

ok(same(r:rows(ids, prev), want_rows(ids, prev)), "rows with prev")
ok(same(r:rows(ids), want_rows(ids, {})), "rows at sequence start")
ok(same(r:rows(ids, {}), r:rows(ids, nil)), "empty prev is nil prev")
ok(same(r:rows({ 4 }, { 5, EOS, 6 }), want_rows({ 4 }, { 5, EOS, 6 })), "EOS in prev")
-- The rows of a split run match those of the whole run.
local whole = r:rows(ids, prev)
local tail = r:rows({ table.unpack(ids, 6) }, { table.unpack(ids, 1, 5) })
ok(same({ table.unpack(whole, 5 * HEADS + 1) }, tail), "split run")

local nt = #ids
local hb = lgml.hostbuf(nt * HEADS * DIM * 4 + 8)
ok(r:embed(ids, prev, hb, 8) == nt * HEADS * DIM * 4, "embed into a hostbuf")
local got = floats(hb:bytes(8))
local want = {}
for _, row in ipairs(whole) do
	for _, v in ipairs(row_values(row)) do
		want[#want + 1] = v
	end
end
ok(same(got, want), "embed values match the table rows")
local st = r:stats(true)
ok(st.calls == 1 and st.rows == nt * HEADS and st.unique < st.rows and st.reads >= 1 and
	st.bytes >= st.unique * 36, "stats")
ok(r:stats().calls == 0, "stats reset")

local rr = lgml.hostbuf(2 * DIM * 4)
r:read({ ROWS - 1, 0 }, rr)
local edge = row_values(ROWS - 1)
for _, v in ipairs(row_values(0)) do
	edge[#edge + 1] = v
end
ok(same(floats(rr:bytes()), edge), "read first and last rows")

for _, mode in ipairs({ "pread", "mmap" }) do
	local m <close> = assert(lgml.ngram.open(path, { mode = mode, threads = 3 }))
	ok(m:info().mode == mode, mode .. " mode")
	local b = lgml.hostbuf(nt * HEADS * DIM * 4)
	m:embed(ids, prev, b)
	ok(same(floats(b:bytes()), want), mode .. " matches direct")
end

-- With and without the io_uring.
print("# io_uring " .. (info.uring and "in use" or "not in use"))
for _, mode in ipairs({ "direct", "pread" }) do
	local m <close> = assert(lgml.ngram.open(path, { mode = mode, uring = false, threads = 2 }))
	ok(m:info().uring == false, mode .. " without io_uring")
	local b = lgml.hostbuf(nt * HEADS * DIM * 4)
	m:embed(ids, prev, b)
	ok(same(floats(b:bytes()), want), mode .. " without io_uring matches")
end
local mm <close> = assert(lgml.ngram.open(path, { mode = "mmap" }))
ok(mm:info().uring == false, "mmap has no io_uring")

local sep <close> =assert(lgml.ngram.open(nometa, { meta = path, mode = "pread" }))
local b2 = lgml.hostbuf(nt * HEADS * DIM * 4)
sep:embed(ids, prev, b2)
ok(same(floats(b2:bytes()), want), "keys from a separate meta file")

local be <close> = assert(lgml.backend("CPU"))
local t <close> = be:tensor("f32", { nt, HEADS * DIM })
ok(r:embed(ids, prev, t) == nt * HEADS * DIM * 4, "embed into a tensor")
ok(same(t:get(), want), "tensor values")
local th <close> = be:tensor("f16", { nt, HEADS * DIM })
ok(r:embed(ids, prev, th) == nt * HEADS * DIM * 2, "embed into an f16 tensor")
ok(same(th:get(), want), "f16 tensor values")
local t1 <close> = be:tensor("f32", { 2, HEADS * DIM })
r:embed({ ids[1] }, prev, t1, HEADS * DIM * 4)
ok(same({ table.unpack(t1:get(), HEADS * DIM + 1) }, { table.unpack(want, 1, HEADS * DIM) }),
	"embed at an offset")

ok(not pcall(r.embed, r, ids, prev, t1), "small tensor raises")
ok(not pcall(r.embed, r, { -1 }, {}, hb), "negative id raises")
ok(not pcall(r.embed, r, ids, prev, lgml.hostbuf(16)), "small hostbuf raises")
ok(not pcall(r.read, r, { ROWS }, hb), "row past the end raises")
local x, err = lgml.ngram.open(nometa)
ok(x == nil and err:find("ple", 1, true) ~= nil, "no ple keys")
local badkey = os.tmpname()
gguf.write(badkey, { { "t.ple.ngram_size", gguf.STRING, "3" } },
	{ { "per_layer_token_embd.weight", { ROWS, DIM }, table.concat(data), 20 } })
x, err = lgml.ngram.open(badkey)
ok(x == nil and err:find("not an integer", 1, true) ~= nil, "string key is an error")
os.remove(badkey)
x, err = lgml.ngram.open(path, { tensor = "nope" })
ok(x == nil and err:find("no tensor", 1, true) ~= nil, "missing tensor")
x = lgml.ngram.open(path .. ".missing")
ok(x == nil, "missing file")
r:close()
ok(not pcall(r.info, r), "closed reader raises")

os.remove(path)
os.remove(nometa)
print("1.." .. n)
