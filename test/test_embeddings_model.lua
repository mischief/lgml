-- lgml.embed_engine on a real Qwen3-Embedding gguf. TAP output.
-- LGML_EMBED_MODEL is the gguf path; the test skips when it is empty.
-- LGML_BACKEND picks the backend (default: the best one). The scores
-- of the queries and documents must match the model card within 0.02.
-- The speed lines are TAP comments.
local lgml = require("lgml")
local json = require("lgml.json")
local httpd = require("lgml.server.httpd")
local openai = require("lgml.server.openai")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local inproc = require("inproc")

local path = os.getenv("LGML_EMBED_MODEL")
if not path or path == "" then
	print("ok 1 # SKIP LGML_EMBED_MODEL not set")
	print("1..1")
	os.exit(0)
end

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function dot(a, b)
	local s = 0
	for i = 1, #a do
		s = s + a[i] * b[i]
	end
	return s
end

local e = require("lgml.embed_engine").new({ model = path, backend = os.getenv("LGML_BACKEND"), n_ctx = 4096 })

-- The example of the Qwen3-Embedding-4B model card and its scores.
local TASK = "Given a web search query, retrieve relevant passages that answer the query"
local function query(q)
	return ("Instruct: %s\nQuery:%s"):format(TASK, q)
end
local texts = {
	query("What is the capital of China?"), query("Explain gravity"),
	"The capital of China is Beijing.",
	"Gravity is a force that attracts two bodies towards each other. It gives weight to physical " ..
	    "objects and is responsible for the movement of planets around the sun.",
}
local CARD = { { 0.7534, 0.1147 }, { 0.0320, 0.6258 } }

local r = assert(e:embed({ input = texts }))
local good = #r.data == 4
for _, v in ipairs(r.data) do
	good = good and #v == e.model.hp.n_embd and math.abs(math.sqrt(dot(v, v)) - 1) < 1e-4
end
ok(good, ("%d vectors of %d values with length 1"):format(#r.data, e.model.hp.n_embd))
local worst = 0
for q = 1, 2 do
	local row = {}
	for d = 1, 2 do
		local s = dot(r.data[q], r.data[2 + d])
		row[d] = ("%.4f (card %.4f)"):format(s, CARD[q][d])
		worst = math.max(worst, math.abs(s - CARD[q][d]))
	end
	print("# query " .. q .. ": " .. table.concat(row, ", "))
end
ok(worst < 0.02, ("scores match the model card (max diff %.4f)"):format(worst))

-- dimensions through HTTP: the first values, length 1 again.
local srv = httpd.new()
openai.install(srv, e)
local st, b = inproc.json(srv, "POST", "/v1/embeddings", { input = json.array({ texts[1], texts[3] }),
	dimensions = 1024 })
local a, c = b.data and b.data[1].embedding, b.data and b.data[2].embedding
ok(st == 200 and #a == 1024 and math.abs(math.sqrt(dot(a, a)) - 1) < 1e-6, "dimensions 1024")
print(("# 1024 dimensions: query 1 and document 1 score %.4f"):format(dot(a, c)))

-- Speed: one text of about 2000 tokens, then 16 of them.
local para = "The history of the river valley spans many centuries of farming, trade and war. " ..
    "Towns grew along its banks, and each spring the floods brought new soil to the fields. "
local long = {}
for i = 1, 16 do
	local t = { ("Document %d. "):format(i) }
	while #e.tk:encode(table.concat(t)) < 2000 do
		t[#t + 1] = para
	end
	long[i] = table.concat(t)
end
-- The time of e:embed on inputs, with a TAP comment.
local function timed(name, inputs)
	local t0 = lgml.clock()
	local res = assert(e:embed({ input = inputs }))
	local s = lgml.clock() - t0
	print(("# %s: %d inputs, %d tokens in %.3f s, %.0f tok/s, %.1f ms per input"):format(name, #inputs,
	    res.usage.prompt_tokens, s, res.usage.prompt_tokens / s, s * 1000 / #inputs))
	return res
end
timed("warm up", { long[1] })
local one = timed("one long input", { long[1] })
local all = timed("long inputs", long)
ok(#all.data == 16 and dot(all.data[1], one.data[1]) > 0.9999,
    "an input gives the same vector alone and in a list")

-- Short inputs share packed passes. Each input alone through the cache
-- gives nearly the same vector.
local short = {}
for i = 1, 64 do
	short[i] = query(("question %d: how do I learn the violin as an adult?"):format(i))
end
timed("warm up", short)
local packed = timed("short inputs, packed", short)
e.packable = false
local alone = timed("short inputs, one pass each", short)
e.packable = true
local least = 1
for i = 1, #short do
	least = math.min(least, dot(packed.data[i], alone.data[i]))
end
ok(least > 0.999, ("packed passes match passes of one input (least cosine %.5f)"):format(least))

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
