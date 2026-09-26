-- safetensors, vocab, conv1d and audio tests, then Whisper model tests.
-- Output is TAP. Set LGML_WHISPER to a Hugging Face Whisper model
-- directory and LGML_WHISPER_WAV to jfk.wav to run the model tests.
local lgml = require("lgml")
local json = require("lgml.json")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function skip(name)
	n = n + 1
	print("ok " .. n .. " # SKIP " .. name)
end

local function near(got, want, tol)
	if #got ~= #want then
		return false
	end
	for i = 1, #want do
		if math.abs(got[i] - want[i]) > (tol or 1e-5) then
			io.stderr:write("got {", table.concat(got, ", "), "}\n")
			return false
		end
	end
	return true
end

local function write(path, data)
	local f = assert(io.open(path, "wb"))
	f:write(data)
	f:close()
end

local tmp = os.tmpname()

-- safetensors

-- entries: { name, dtype, shape, bytes }. extra is added to the header.
local function st_bytes(entries, meta, extra)
	local h, data, off = {}, {}, 0
	for _, e in ipairs(entries) do
		h[e[1]] = { dtype = e[2], shape = json.array(e[3]), data_offsets = { off, off + #e[4] } }
		data[#data + 1] = e[4]
		off = off + #e[4]
	end
	h.__metadata__ = meta
	local s = json.encode(h) .. (extra or "")
	return string.pack("<I8", #s) .. s .. table.concat(data)
end

local function f32(...)
	return string.pack("<" .. string.rep("f", select("#", ...)), ...)
end

local function u16(...)
	return string.pack("<" .. string.rep("I2", select("#", ...)), ...)
end

local ENTRIES = {
	{ "a", "F32", { 3 }, f32(1, 2, 3) },
	{ "m.h", "F16", { 2, 2 }, u16(0x3c00, 0x4000, 0xc000, 0x3800) },
	{ "b", "BF16", { 2 }, u16(0x3f80, 0xbfc0) },
	{ "h1", "F16", { 2 }, u16(0x3c00, 0x3800) },
	{ "i", "I32", { 2 }, string.pack("<i4i4", 7, -9) },
}

local function test_index()
	write(tmp, st_bytes(ENTRIES, { format = "pt", note = "x\"y" }))
	local info = lgml.safetensors(tmp)
	ok(info.metadata.format == "pt" and info.metadata.note == 'x"y', "safetensors metadata")
	local by = {}
	for _, t in ipairs(info.tensors) do
		by[t.name] = t
	end
	ok(#info.tensors == 5 and by["m.h"].type == "f16" and near(by["m.h"].shape, { 2, 2 }) and
	    by.a.size == 12 and by.b.offset == 20, "safetensors tensor index")
end

local function test_load(be)
	write(tmp, st_bytes(ENTRIES))
	local w <close> = be:load_safetensors(tmp)
	ok(near(w:get("a"):get(), { 1, 2, 3 }), "load f32")
	ok(near(w:get("m.h"):get(), { 1, 2, -2, 0.5 }) and near(w:get("m.h"):shape(), { 2, 2 }),
	    "load f16 matrix")
	ok(near(w:get("b"):get(), { 1, -1.5 }) and w:get("b"):type() == "bf16", "load bf16")
	ok(near(w:get("i"):get(), { 7, -9 }), "load i32")
	local w2 <close> = be:load_safetensors({ tmp }, { f32_1d = true })
	ok(w2:get("h1"):type() == "f32" and near(w2:get("h1"):get(), { 1, 0.5 }) and
	    w2:get("b"):type() == "f32" and w2:get("m.h"):type() == "f16", "f32_1d widens 1-D halfs")
	local tmp2 = os.tmpname()
	write(tmp2, st_bytes({ { "z", "F32", { 1 }, f32(5) } }))
	local w3 <close> = be:load_safetensors({ tmp, tmp2 })
	ok(near(w3:get("z"):get(), { 5 }) and near(w3:get("a"):get(), { 1, 2, 3 }), "load two files")
	ok(not pcall(be.load_safetensors, be, { tmp, tmp }), "duplicate names across files rejected")
	os.remove(tmp2)
end

local function rejects(be, data, name)
	write(tmp, data)
	local good = pcall(lgml.safetensors, tmp) or pcall(be.load_safetensors, be, tmp)
	ok(not good, name)
end

local function test_bad(be)
	local good = st_bytes(ENTRIES)
	rejects(be, good:sub(1, 7), "short file rejected")
	rejects(be, string.pack("<I8", 1 << 40) .. "{}", "huge header length rejected")
	rejects(be, good:sub(1, #good - 1), "data past end rejected")
	rejects(be, st_bytes({ { "a", "F32", { 4 }, f32(1, 2, 3) } }), "size mismatch rejected")
	rejects(be, st_bytes({ { "a", "F32", { 1, 1, 1, 1, 1 }, f32(1) } }), "rank 5 rejected")
	rejects(be, st_bytes({ { "a", "F32", { 0 }, "" } }), "zero dimension rejected")
	rejects(be, st_bytes({ { "a", "C64", { 1 }, f32(1, 2) } }), "unknown dtype rejected")
	rejects(be, st_bytes({ { string.rep("n", 64), "F32", { 1 }, f32(1) } }), "long name rejected")
	rejects(be, st_bytes(ENTRIES, nil, " x"), "trailing garbage rejected")
	local s = '{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]'
	rejects(be, string.pack("<I8", #s) .. s .. f32(1), "unterminated JSON rejected")
	s = '{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}'
	rejects(be, string.pack("<I8", #s) .. s .. f32(1, 2), "reversed offsets rejected")
	s = string.rep("[", 200) .. string.rep("]", 200)
	s = '{"__metadata__":{"x":' .. s .. '}}'
	rejects(be, string.pack("<I8", #s) .. s, "deep nesting rejected")
	write(tmp, string.pack("<I8", 2) .. "{}")
	ok(#lgml.safetensors(tmp).tensors == 0 and not pcall(be.load_safetensors, be, tmp),
	    "empty index reads; loading no tensors is an error")
end

-- conv1d: w (2, 1, 3), x (1, 5), pad 1.
local function test_conv(be, wtype)
	local tag = be:name() .. " " .. wtype .. ": "
	local w <close> = be:tensor(wtype, { 2, 1, 3 }, { 1, 0, -1, 1, 1, 1 })
	local x <close> = be:tensor("f32", { 1, 5 }, { 1, 2, 3, 4, 5 })
	local g <close> = be:graph()
	local y1 = g:conv1d(w, x, 1, 1)
	local y2 = g:conv1d(w, x, 2, 1)
	g:compute(y1, y2)
	ok(near(y1:shape(), { 2, 5 }) and near(y1:get(), { -2, -2, -2, -2, 4, 3, 6, 9, 12, 9 }),
	    tag .. "conv1d stride 1")
	ok(near(y2:get(), { -2, -2, 4, 3, 9, 9 }), tag .. "conv1d stride 2")
	ok(not pcall(g.conv1d, g, w, x, 1, 0, 3), tag .. "conv1d kernel too large is an error")
end

local function test_binop_types(be)
	local a <close> = be:tensor("f32", { 2 }, { 1, 2 })
	local h <close> = be:tensor("f16", { 2 }, { 1, 2 })
	local g <close> = be:graph()
	ok(not pcall(g.add, g, a, h), "add f32 + f16 is an error, not an abort")
end

-- vocab

local function test_vocab()
	local vp, ap = os.tmpname(), os.tmpname()
	write(vp, '{"a": 0, "\\u0120b": 1, "\\u010a": 2}')
	write(ap, '{"<|x|>": 3}')
	local v <close> = lgml.vocab(vp, ap)
	ok(v:decode({ 0, 1, 2, 3 }) == "a b\n" and v:decode({ 0, 3 }, true) == "a<|x|>",
	    "vocab decodes byte-level pieces")
	ok(v:id(" b") == 1 and v:id("<|x|>") == 3 and v:id("zz") == nil and v:special(3) and
	    not v:special(1), "vocab id and special")
	ok(not pcall(v.decode, v, { 9 }), "vocab rejects unknown ids")
	write(vp, '{"a": 0, "b": }')
	ok(not pcall(lgml.vocab, vp), "vocab rejects bad JSON")
	os.remove(vp)
	os.remove(ap)
end

-- audio

local function tone_wav(rate, count)
	local s = {}
	for i = 0, count - 1 do
		s[#s + 1] = string.pack("<i2", math.floor(8000 * math.sin(i * 0.3)))
	end
	local data = table.concat(s)
	return "RIFF" .. string.pack("<I4", 36 + #data) .. "WAVEfmt " ..
	    string.pack("<I4I2I2I4I4I2I2", 16, 1, 1, rate, rate * 2, 2, 16) ..
	    "data" .. string.pack("<I4", #data) .. data
end

local function test_audio(be)
	local a <close> = assert(lgml.audio(tone_wav(16000, 1600)))
	ok(a:rate() == 16000 and a:len() == 1600 and math.abs(a:duration() - 0.1) < 1e-9,
	    "audio from WAV")
	local b <close> = a:resample(44100)
	local c <close> = a:concat(a, a)
	ok(b:len() == 4410 and b:rate() == 44100 and c:len() == 4800, "resample and concat")
	local d <close> = assert(lgml.audio(b:wav(2, "f32")))
	ok(d:rate() == 44100 and d:len() == 4410, "WAV encode round trip")
	local r1, r2 = lgml.audio("junk")
	ok(r1 == nil and r2:find("RIFF"), "junk is not audio")
	local mel <close> = be:tensor("f32", { 128, 3000 })
	a:mel(mel)
	local v = mel:get()
	local lo, hi = math.huge, -math.huge
	for i = 1, #v do
		lo, hi = math.min(lo, v[i]), math.max(hi, v[i])
	end
	ok(#v == 384000 and lo >= hi - 2.0001 and hi < 3, "mel fills the tensor")
	ok(not pcall(b.mel, b, mel), "mel needs 16 kHz")
	local bad <close> = be:tensor("f16", { 128, 3000 })
	ok(not pcall(a.mel, a, bad), "mel needs f32")
end

-- model

local JFK = "And so, my fellow Americans, ask not what your country can do for you, " ..
    "ask what you can do for your country."

local function count(s, pat)
	local k = 0
	for _ in s:gmatch(pat) do
		k = k + 1
	end
	return k
end

local function test_model(dir, wav)
	local whisper = require("lgml.whisper")
	local be = assert(lgml.backend())
	local m = whisper.load(be, dir)
	ok(m.hp.n_mels >= 80 and m.tok.sot ~= nil, "whisper loads")
	if not wav then
		for _, name in ipairs({ "jfk", "language", "44.1 kHz stereo", "long", "engine" }) do
			skip(name .. ": LGML_WHISPER_WAV not set")
		end
		return
	end
	local f = assert(io.open(wav, "rb"))
	local bytes = f:read("a")
	f:close()
	local a = assert(lgml.audio(bytes))
	if a:rate() ~= 16000 then
		a = a:resample(16000)
	end
	local r = m:transcribe(a, { lang = m:language_id("en") })
	ok(r.text == JFK, "jfk transcript matches whisper-cli")
	r = m:transcribe(a)
	ok(r.language == "en" and r.text == JFK, "language detected as en")
	local st = assert(lgml.audio(a:resample(44100):wav(2)))
	r = m:transcribe(st:resample(16000))
	ok(r.text == JFK, "44.1 kHz stereo transcript")
	r = m:transcribe(a:concat(a, a, a))
	ok(count(r.text, "ask not what your country") == 4 and #r.segments == 2 and
	    math.abs(r.duration - 4 * a:duration()) < 1e-6, "long audio in 30 s windows")
	m:close()
	local eng = require("lgml.stt_engine").new({ model = dir })
	local res, msg, status = eng:transcribe({ language = "xx" }, bytes)
	ok(res == nil and status == 400 and msg:find("language") and
	    eng:transcribe({}, bytes).text == JFK and
	    select(3, eng:transcribe({}, "junk")) == 400, "engine transcribe")
end

local dev = assert(lgml.backend("CPU"))
test_index()
test_load(dev)
test_bad(dev)
for _, d in ipairs(lgml.devices()) do
	local be <close> = assert(lgml.backend(d.name))
	test_conv(be, "f32")
	test_conv(be, "f16")
end
test_binop_types(dev)
test_vocab()
test_audio(dev)
os.remove(tmp)

local dir = os.getenv("LGML_WHISPER")
if dir and dir ~= "" then
	local wav = os.getenv("LGML_WHISPER_WAV")
	test_model(dir, wav ~= "" and wav or nil)
else
	skip("whisper model: LGML_WHISPER not set")
end
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
