-- lgml.finetune and bin/lgml-finetune on a tiny qwen3 model: chat and text
-- sample masks, packing, and short training runs whose adapters change
-- the output. Output is TAP.
local lgml = require("lgml")
local chat = require("lgml.chat")
local tokenizer = require("lgml.tokenizer")
local transformer = require("lgml.transformer")
local lora = require("lgml.lora")
local finetune = require("lgml.finetune")
local json = require("lgml.json")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local tiny = require("tiny")

local n = 0
local failed = false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local files = {}
local function tmp(suffix)
	local p = tiny.tmp(suffix)
	files[#files + 1] = p
	return p
end

local path = tmp(".gguf")
tiny.write(path, "qwen3")
local info = assert(lgml.gguf(path))
local family = chat.load(info.meta)
local tk = assert(tokenizer.open(path))

-- The trained text of a sample.
local function trained(s)
	local ids = {}
	for i, id in ipairs(s.ids) do
		if s.train[i] then
			ids[#ids + 1] = id
		end
	end
	return tk:decode(ids)
end

do
	local msgs = {
		{ role = "system", content = "Be brief." },
		{ role = "user", content = "Hi?" },
		{ role = "assistant", content = "Hello there." },
	}
	local ss = finetune.encode(tk, family, msgs)
	ok(#ss == 1, "one sample for one turn")
	ok(trained(ss[1]) == "Hello there.<|im_end|>\n", ("the answer and end of turn train: %q"):format(
	    trained(ss[1])))
	ok(tk:decode(ss[1].ids) == family.render(msgs, { enable_thinking = false }),
	    "the sample holds the whole render")

	local multi = {
		{ role = "user", content = "One?" },
		{ role = "assistant", content = "1" },
		{ role = "user", content = "Two?" },
		{ role = "assistant", content = "2" },
	}
	ss = finetune.encode(tk, family, multi)
	ok(#ss == 2, "an earlier turn that renders in another form gets its own sample")
	ok(trained(ss[1]) == "2<|im_end|>\n" and trained(ss[2]) == "1<|im_end|>\n",
	    "each turn trains once")
	local none = finetune.encode(tk, family, { { role = "user", content = "x" } })
	ok(#none == 0, "a conversation without an answer gives no sample")
end

do
	local s = function(len, m)
		local t = { ids = {}, train = {} }
		for i = 1, len do
			t.ids[i], t.train[i] = 65, i > len - m
		end
		return t
	end
	local mbs, cut = finetune.pack({ s(10, 3), s(10, 3), s(10, 3), s(40, 30) }, 32)
	ok(#mbs == 2 and mbs[1].n == 30 and mbs[1].m == 9 and mbs[2].n == 32 and cut == 1,
	    "pack fills micro-batches and cuts long samples")
	mbs = finetune.pack({ s(10, 3), s(10, 3), s(10, 3) }, 64, 2)
	ok(#mbs == 2 and #mbs[1].samples == 2, "pack keeps the batch limit")
end

-- Packed samples do not see each other: the loss of a micro-batch of
-- two samples is the mean of their losses alone.
do
	local be <close> = assert(lgml.backend("CPU"))
	local m = transformer.load(be, path)
	local a = finetune.encode(tk, family, {
		{ role = "user", content = "Name a color." }, { role = "assistant", content = "Blue." } })[1]
	local b = finetune.encode(tk, family, {
		{ role = "user", content = "And a fruit?" }, { role = "assistant", content = "Pear!" } })[1]
	local la = finetune.loss(m, finetune.pack({ a }, 256)[1])
	local lb = finetune.loss(m, finetune.pack({ b }, 256)[1])
	local both = finetune.pack({ a, b }, 256)
	local ma, mb = both[1].m, 0
	ma = finetune.pack({ a }, 256)[1].m
	mb = finetune.pack({ b }, 256)[1].m
	local lab = finetune.loss(m, both[1])
	local want = (la * ma + lb * mb) / (ma + mb)
	ok(#both == 1 and math.abs(lab - want) < 1e-4, ("packed loss %.5f, separate %.5f"):format(lab,
	    want))
	local good, err = pcall(finetune.check, { hp = { arch = "qwen35" }, recurrent = true })
	ok(not good and tostring(err):find("gated delta net", 1, true), "hybrid models raise")
	m:close()
end

-- The command line tool trains an adapter that changes the answer.
do
	local data = tmp(".jsonl")
	local f = assert(io.open(data, "w"))
	local convo = {
		{ role = "user", content = "Who are you?" },
		{ role = "assistant", content = "I am lgml." },
	}
	for _ = 1, 4 do
		f:write(json.encode({ messages = convo }), "\n")
	end
	f:close()
	local out = tmp(".gguf")
	local lua = arg[-1] or "lua5.4"
	local cmd = ("%s %s/../bin/lgml-finetune -m %s -d %s -o %s -b CPU -r 8 --lr 1e-2 " ..
	    "-e 25 --grad-accum 2 --batch 1 --eval-split 0.25 --save-every 20 " ..
	    "--targets attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down,output 2>&1"):format(
	    lua, here, path, data, out)
	local p = assert(io.popen(cmd))
	local log = p:read("a")
	local closed = p:close()
	local losses, evals = {}, {}
	for l in log:gmatch("step %d+ epoch %d+ loss ([%d.]+)") do
		losses[#losses + 1] = tonumber(l)
	end
	for l in log:gmatch("eval step %d+ loss ([%d.]+)") do
		evals[#evals + 1] = tonumber(l)
	end
	ok(closed and #losses == 50, "the tool runs 50 steps")
	if not closed then
		io.stderr:write(log)
	end
	ok(#losses > 0 and losses[#losses] < losses[1] / 10, ("train loss %.3f -> %.4f"):format(
	    losses[1] or 0, losses[#losses] or 0))
	ok(#evals == 25 and evals[#evals] < evals[1], ("eval loss %.3f -> %.4f"):format(evals[1] or 0,
	    evals[#evals] or 0))
	local ck = io.open(out:gsub("%.gguf$", "") .. "-step40.gguf")
	ok(ck ~= nil, "a checkpoint at step 40")
	if ck then
		ck:close()
		files[#files + 1] = out:gsub("%.gguf$", "") .. "-step40.gguf"
		files[#files + 1] = out:gsub("%.gguf$", "") .. "-step20.gguf"
	end

	-- Greedy answers with and without the adapter.
	local be <close> = assert(lgml.backend("CPU"))
	local m = transformer.load(be, path)
	local prompt = tk:encode(family.render({ convo[1] }, { add_generation_prompt = true,
	    enable_thinking = false }))
	local function answer()
		local cache = m:cache(256)
		local ids = { m:step(cache, prompt) }
		for _ = 1, 12 do
			ids[#ids + 1] = m:step(cache, { ids[#ids] })
		end
		return tk:decode(ids)
	end
	local base = answer()
	m:set_adapters({ lora.load(m, out) })
	local tuned = answer()
	ok(tuned:sub(1, 10) == "I am lgml." and base:sub(1, 10) ~= "I am lgml.",
	    ("the adapter changes the answer: %q -> %q"):format(base, tuned))
	m:close()
end

-- Plain text: chunks of the context size, every token trains.
do
	local text = "/* file: add.c */\nint add(int a, int b) { return a + b; }\n"
	local ids = tk:encode(text, { add_bos = false })
	local ss = finetune.encode_text(tk, text, 16)
	local count, all = 0, true
	for _, s in ipairs(ss) do
		count = count + #s.ids
		for _, t in ipairs(s.train) do
			all = all and t
		end
	end
	ok(#ss == (#ids + 15) // 16 and count == #ids and all and #ss[1].ids == 16,
	    ("text splits into %d chunks of 16 tokens"):format(#ss))

	local mixed = tmp(".jsonl")
	local f = assert(io.open(mixed, "w"))
	f:write(json.encode({ text = text }), "\n\n")
	f:write(json.encode({ messages = { { role = "user", content = "a" },
	    { role = "assistant", content = "b" } } }), "\n")
	f:close()
	local recs = finetune.read_jsonl(mixed)
	ok(#recs == 2 and recs[1].text == text and recs[2].messages, "chat format takes mixed lines")
	local good, err = pcall(finetune.read_jsonl, mixed, "text")
	ok(not good and tostring(err):find(":3:", 1, true), "text format needs text on every line")

	-- The tool learns to complete a source file from its header.
	local data = tmp(".jsonl")
	f = assert(io.open(data, "w"))
	for _ = 1, 3 do
		f:write(json.encode({ text = text }), "\n")
	end
	f:close()
	local out = tmp(".gguf")
	local cmd = ("%s %s/../bin/lgml-finetune -m %s -d %s -o %s -b CPU --format text -c 32 " ..
	    "--lr 1e-2 -e 30 --targets attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down,output " ..
	    "2>&1"):format(arg[-1] or "lua5.4", here, path, data, out)
	local p = assert(io.popen(cmd))
	local log = p:read("a")
	local closed = p:close()
	local losses = {}
	for l in log:gmatch("step %d+ epoch %d+ loss ([%d.]+)") do
		losses[#losses + 1] = tonumber(l)
	end
	ok(closed and #losses > 0 and losses[#losses] < losses[1] / 10,
	    ("text train loss %.3f -> %.4f"):format(losses[1] or 0, losses[#losses] or 0))
	if not closed then
		io.stderr:write(log)
	end
	local be <close> = assert(lgml.backend("CPU"))
	local m = transformer.load(be, path)
	m:set_adapters({ lora.load(m, out) })
	local head = tk:encode("/* file: add.c */\n", { add_bos = false })
	local cache = m:cache(128)
	local got = { m:step(cache, head) }
	for _ = 1, 7 do
		got[#got + 1] = m:step(cache, { got[#got] })
	end
	local s = tk:decode(got)
	ok(s == ("int add(int a, int b) { return a + b; }\n"):sub(1, #s) and #s >= 8,
	    ("the adapter completes the file: %q"):format(s))
	m:close()
end

for _, p in ipairs(files) do
	os.remove(p)
end
print("1.." .. n)
os.exit(failed and 1 or 0)
