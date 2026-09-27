-- Tests for /v1/embeddings: the request checks, the reply shape,
-- dimensions and base64 on the fake engine, and lgml.embed_engine on a
-- tiny qwen3 model of random weights on the CPU. TAP output.
local json = require("lgml.json")
local lgml = require("lgml")
local base64 = require("lgml.base64")
local httpd = require("lgml.server.httpd")
local openai = require("lgml.server.openai")
local embeddings = require("lgml.server.embeddings")
local fake = require("lgml.server.fake_engine")
local detect = require("lgml.detect")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local tiny = require("tiny")
local gguf = require("gguf")
local inproc = require("inproc")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function norm(v)
	local s = 0
	for _, x in ipairs(v) do
		s = s + x * x
	end
	return math.sqrt(s)
end

-- request checks

local function check(b)
	return embeddings.check(b)
end

local function test_check()
	local r = check({ input = "hi", model = "m", user = "u" })
	ok(r and #r.input == 1 and r.input[1] == "hi" and r.encoding_format == "float", "one string")
	r = check({ input = json.array({ "a", "b" }) })
	ok(r and #r.input == 2 and r.input[2] == "b", "array of strings")
	r = check({ input = json.array({ 1, 2, 3 }) })
	ok(r and #r.input == 1 and #r.input[1] == 3, "one token list")
	r = check({ input = json.array({ json.array({ 1 }), json.array({ 2, 3 }) }) })
	ok(r and #r.input == 2 and r.input[2][2] == 3, "token lists")
	r = check({ input = "x", dimensions = 4, encoding_format = "base64", model = json.null })
	ok(r and r.dimensions == 4 and r.encoding_format == "base64" and r.model == nil,
	    "dimensions, base64, and null reads as absent")
	local function bad(b, param, code, name)
		local good, msg, p, c = check(b)
		ok(not good and p == param and c == code and type(msg) == "string", name)
	end
	bad({ input = "x", foo = 1 }, "foo", "unknown_parameter", "unknown field")
	bad({ input = "x", encoding_format = "int8" }, "encoding_format", "invalid_value", "bad encoding_format")
	bad({ input = "x", dimensions = 0 }, "dimensions", "invalid_value", "dimensions 0")
	bad({ input = "x", dimensions = 1.5 }, "dimensions", "invalid_value", "dimensions not an integer")
	bad({ model = "m" }, "input", "invalid_value", "input is required")
	bad({ input = "" }, "input", "invalid_value", "empty string")
	bad({ input = json.array() }, "input", "invalid_value", "empty array")
	bad({ input = json.array({ "a", "" }) }, "input", "invalid_value", "empty string in an array")
	bad({ input = json.array({ "a", 1 }) }, "input", "invalid_value", "string and token id")
	bad({ input = json.array({ 1, -2 }) }, "input", "invalid_value", "negative token id")
	bad({ input = json.array({ json.array() }) }, "input", "invalid_value", "empty token list")
	bad({ input = 5 }, "input", "invalid_value", "number input")
	local many = json.array()
	for i = 1, embeddings.MAX_INPUTS + 1 do
		many[i] = "x"
	end
	bad({ input = many }, "input", "invalid_value", "too many inputs")
	ok(not check(json.array()), "an array body")
end

-- reply helpers

local function test_reply()
	local v = embeddings.truncate({ 3, 4, 12 }, 2)
	ok(#v == 2 and math.abs(v[1] - 0.6) < 1e-12 and math.abs(v[2] - 0.8) < 1e-12, "truncate scales to 1")
	local e = embeddings.encode({ 1, -0.5, 0.25 })
	local raw = base64.decode(e)
	ok(#raw == 12 and string.unpack("<f", raw, 5) == -0.5, "base64 of little-endian float32")
end

-- HTTP on the fake engine

local function test_http()
	local eng = fake.new({ delay = 0 })
	local srv = httpd.new()
	openai.install(srv, eng)
	local texts = json.array({ "one", "two words", "three" })
	local st, b, h = inproc.json(srv, "POST", "/v1/embeddings", { input = texts, model = "fake" })
	ok(st == 200 and b.object == "list" and b.model == "fake" and #b.data == 3, "reply shape")
	local good = true
	for i, d in ipairs(b.data) do
		good = good and d.object == "embedding" and d.index == i - 1 and #d.embedding == fake.EMBED_DIM and
		    math.abs(norm(d.embedding) - 1) < 1e-9
	end
	ok(good, "data entries in input order, length 1")
	ok(b.usage.prompt_tokens > 0 and b.usage.total_tokens == b.usage.prompt_tokens, "usage")
	ok(h["x-lgml-prompt-n"] == tostring(b.usage.prompt_tokens) and h["x-lgml-elapsed-ms"] ~= nil,
	    "x-lgml headers")

	local st2, b2 = inproc.json(srv, "POST", "/embeddings", { input = texts, dimensions = 3 })
	good = st2 == 200
	for i, d in ipairs(b2.data or {}) do
		local full = b.data[i].embedding
		local want = embeddings.truncate(full, 3)
		good = good and #d.embedding == 3 and math.abs(norm(d.embedding) - 1) < 1e-9
		for k = 1, 3 do
			good = good and math.abs(d.embedding[k] - want[k]) < 1e-9 and d.embedding[k] * full[k] >= 0
		end
	end
	ok(good, "dimensions keeps the first values, length 1 again; /embeddings alias")

	local st3, b3 = inproc.json(srv, "POST", "/v1/embeddings", { input = texts, encoding_format = "base64" })
	good = st3 == 200
	for i, d in ipairs(b3.data or {}) do
		local raw = base64.decode(d.embedding)
		good = good and type(d.embedding) == "string" and #raw == 4 * fake.EMBED_DIM
		for k = 1, fake.EMBED_DIM do
			local x = string.unpack("<f", raw, 4 * k - 3)
			good = good and x == string.unpack("<f", string.pack("<f", b.data[i].embedding[k]))
		end
	end
	ok(good, "base64 round trip gives the float32 values")

	local st4, b4 = inproc.json(srv, "POST", "/v1/embeddings", { input = "x", dimensions = fake.EMBED_DIM + 1 })
	ok(st4 == 400 and b4.error.param == "dimensions" and b4.error.type == "invalid_request_error",
	    "dimensions above the model gives 400")
	local st5, b5 = inproc.json(srv, "POST", "/v1/embeddings", { input = "x", foo = true })
	ok(st5 == 400 and b5.error.code == "unknown_parameter" and b5.error.param == "foo", "unknown field gives 400")
	local st6, b6 = inproc.json(srv, "POST", "/v1/embeddings", { input = "fail" })
	ok(st6 == 500 and b6.error.message == "fake failure" and b6.error.type == "server_error",
	    "engine error in the OpenAI shape")
	local st7 = inproc.request(srv, "POST", "/v1/embeddings", "{")
	ok(st7 == 400, "bad JSON gives 400")
	local st8, b8 = inproc.json(srv, "POST", "/v1/embeddings", { input = json.array({ 7, 8 }) })
	ok(st8 == 200 and b8.usage.prompt_tokens == 2, "token ids")

	local bare = setmetatable({ embed = false }, { __index = eng })
	local srv2 = httpd.new()
	openai.install(srv2, bare)
	local st9 = inproc.json(srv2, "POST", "/v1/embeddings", { input = "x" })
	ok(st9 == 404, "no embed method gives 404")
end

-- lgml.embed_engine on a tiny model

local function tiny_model(kv)
	local path = tiny.tmp(".gguf")
	tiny.write(path, "qwen3", { n_ctx = 256, kv = kv })
	return path
end

local EMBED_KV = {
	{ "qwen3.pooling_type", gguf.UINT32, 3 },
	{ "tokenizer.ggml.add_eos_token", gguf.BOOL, true },
}

local function test_engine()
	local path = tiny_model(EMBED_KV)
	local chat = tiny_model()
	ok(detect.is_embedding(path) and not detect.is_embedding(chat), "pooling_type makes an embedding model")
	ok(table.concat(detect.modes({ model = path }), ",") == "embedding" and
	    table.concat(detect.modes({ model = chat }), ",") == "completion,decide", "modes")
	os.remove(chat)

	local ee = require("lgml.embed_engine")
	-- An f32 cache and soft_max attention keep the results exact, so the
	-- checks below can compare them closely.
	local e = ee.new({ model = path, backend = "CPU", id = "tiny", flash_attn = false,
	    cache_type_k = "f32", cache_type_v = "f32" })
	ok(e:models()[1].id == "tiny" and e:props().n_ctx == 256, "models and props")
	local text = "hello there, general kenobi"
	local ids = e.tk:encode(text)
	local r = assert(e:embed({ input = { text, "he" } }))
	ok(#r.data == 2 and #r.data[1] == tiny.E and math.abs(norm(r.data[1]) - 1) < 1e-5, "vector length 1")
	ok(r.usage.prompt_tokens == #ids + 1 + #e.tk:encode("he") + 1, "each text gets the end token")

	-- A token list with the end token gives the same vector.
	local with_eos = table.move(ids, 1, #ids, 1, {})
	with_eos[#with_eos + 1] = e.tk:special().eos
	local r2 = assert(e:embed({ input = { with_eos } }))
	local d = 0
	for k = 1, tiny.E do
		d = math.max(d, math.abs(r2.data[1][k] - r.data[1][k]))
	end
	ok(d < 1e-6, "token ids run as given")

	-- Inputs in one packed pass give the vectors of passes of at most
	-- 4 tokens through the cache.
	local texts = { ("abcdefghij"):rep(8), "hello", "general kenobi" }
	local packed = assert(e:embed({ input = texts })).data
	local ub = e.ubatch
	e.ubatch = 4
	local chunked = assert(e:embed({ input = texts })).data
	local passes = e:passes({ { 1, 2 }, { 3 }, { 4, 5, 6 }, { 7, 8, 9, 10, 11 }, { 12 } })
	e.ubatch = ub
	d = 0
	for i = 1, #texts do
		for k = 1, tiny.E do
			d = math.max(d, math.abs(packed[i][k] - chunked[i][k]))
		end
	end
	ok(d < 1e-4, ("a packed pass matches chunked passes (max diff %.2g)"):format(d))
	-- Lengths 2, 1, 3, 5 and 1 with 4 tokens per pass: { 1, 2 }, { 3, 5 }
	-- and 4 alone.
	local shape = {}
	for i, p in ipairs(passes) do
		shape[i] = table.concat(p, ",") .. (p.alone and "a" or "")
	end
	ok(table.concat(shape, " ") == "1,2 3,5 4a", "passes: " .. table.concat(shape, " "))

	-- The pooled row is the normed hidden state of the last token.
	local m = e.model
	local toks = e.tk:encode("abc")
	local g = m.be:graph()
	local _, taps = m:forward(g, m:inputs(toks, nil, {}))
	g:compute(taps.result_norm)
	local all = taps.result_norm:get()
	local pooled = m:embed(e.cache, toks, {})
	d = 0
	for k = 1, tiny.E do
		d = math.max(d, math.abs(pooled[k] - all[(#toks - 1) * tiny.E + k]))
	end
	g:close()
	ok(d < 1e-4, ("pooled output is the last row of the final norm (max diff %.2g)"):format(d))

	local _, msg, status = e:embed({ input = { ("x"):rep(300) } })
	ok(status == 400 and msg:find("context is 256", 1, true), "an input past the context gives 400")
	_, msg, status = e:embed({ input = { { e.tk:n_vocab() } } })
	ok(status == 400 and msg:find("not a token id", 1, true), "a bad token id gives 400")
	_, _, status = e:embed({ input = { "x" } }, { cancelled = function() return true end })
	ok(status == 499, "cancel gives 499")

	local srv = httpd.new()
	openai.install(srv, e)
	local st, b = inproc.json(srv, "POST", "/v1/embeddings", { input = json.array({ "a", "b" }), dimensions = 16 })
	ok(st == 200 and b.model == "tiny" and #b.data[2].embedding == 16, "HTTP on the engine")
	local _, mb = inproc.json(srv, "GET", "/v1/models")
	ok(table.concat(mb.data[1].capabilities, ",") == "embedding", "capability embedding")
	os.remove(path)

	local mean = tiny_model({ { "qwen3.pooling_type", gguf.UINT32, 1 } })
	local good, err = pcall(ee.new, { model = mean, backend = "CPU" })
	ok(not good and tostring(err):find("pooling type 1", 1, true), "mean pooling does not load")
	os.remove(mean)
end

test_check()
test_reply()
test_http()
test_engine()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
