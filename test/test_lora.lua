-- LoRA adapters on tiny models: inference against merged weights, file
-- round trips (gguf and PEFT), replay, and training a tiny adapter to
-- overfit one sequence. Output is TAP.
local lgml = require("lgml")
local transformer = require("lgml.transformer")
local lora = require("lgml.lora")
local optim = require("lgml.optim")
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

local be <close> = assert(lgml.backend(os.getenv("LGML_DEVICE") or "CPU"))
-- GPU backends accumulate in f16, so whole-pass comparisons get more room.
local LOOSE = (os.getenv("LGML_DEVICE") or "CPU") == "CPU" and 1e-3 or 3e-2
local files = {}
local function tmp(suffix)
	local p = tiny.tmp(suffix)
	files[#files + 1] = p
	return p
end

local function logits_of(m, ids)
	local inp = m:inputs(ids, nil, {})
	local g <close> = be:graph()
	local out = m:forward(g, inp)
	g:compute(out)
	return out:get()
end

local function maxdiff(a, b)
	local d = #a == #b and 0 or math.huge
	for i = 1, #a do
		d = math.max(d, math.abs(a[i] - b[i]))
	end
	return d
end

-- W + s * B A for each adapter weight, on the host. w is { shape, values }.
local function merge(byname, list)
	local out = {}
	for name, t in pairs(byname) do
		out[name] = table.move(t[2], 1, #t[2], 1, {})
	end
	for _, ad in ipairs(list) do
		for _, name in ipairs(ad.names) do
			local p = ad.pairs[name]
			local a, b = p.a:get(), p.b:get()
			local r, K = table.unpack(p.a:shape())
			local M = p.b:shape()[1]
			local s = ad:scale_of(name)
			local w = out[name]
			for i = 0, M - 1 do
				for k = 0, K - 1 do
					local acc = 0
					for j = 0, r - 1 do
						acc = acc + b[i * r + j + 1] * a[j * K + k + 1]
					end
					w[i * K + k + 1] = w[i * K + k + 1] + s * acc
				end
			end
		end
	end
	return out
end

local ids = { 104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100 }

-- Inference: a new adapter changes nothing, a trained one matches the
-- merged weights, and files round trip.
do
	local path = tmp(".gguf")
	local byname = tiny.write(path, "qwen3")
	local m = transformer.load(be, path)
	local base = logits_of(m, ids)
	local ad = lora.new(m, { rank = 4, alpha = 8 })
	ok(#ad.names == 2 * 7, "new adapter targets 7 weights in each layer")
	m:set_adapters({ ad })
	ok(maxdiff(logits_of(m, ids), base) == 0, "zero B leaves the logits")
	for i, name in ipairs(ad.names) do
		ad.pairs[name].b:fill("normal", 0, 0.2, 100 + i)
	end
	local with = logits_of(m, ids)
	ok(maxdiff(with, base) > 1e-2, "the adapter changes the logits")

	local mpath = tmp(".gguf")
	tiny.write(mpath, "qwen3", { override = merge(byname, { ad }) })
	local merged = transformer.load(be, mpath)
	local d = maxdiff(with, logits_of(merged, ids))
	ok(d < LOOSE, ("adapter matches merged weights (%.2e)"):format(d))
	merged:close()

	local apath = tmp(".gguf")
	assert(ad:save(apath, { ["general.name"] = "test" }))
	local meta = lgml.gguf(apath).meta
	ok(meta["general.type"] == "adapter" and meta["adapter.type"] == "lora" and
	    meta["adapter.lora.alpha"] == 8 and meta["general.architecture"] == "qwen3",
	    "saved gguf has the llama.cpp adapter keys")
	local back = lora.load(m, apath)
	m:set_adapters({ back })
	ok(maxdiff(logits_of(m, ids), with) == 0, "saved adapter loads back")

	back.scale = 0
	m:set_adapters({ back })
	ok(maxdiff(logits_of(m, ids), base) == 0, "scale 0 gives the base logits")

	-- Two adapters add up.
	local ad2 = lora.new(m, { rank = 2, alpha = 2, targets = { "attn_v", "ffn_down" }, seed = 9 })
	for i, name in ipairs(ad2.names) do
		ad2.pairs[name].b:fill("normal", 0, 0.3, 200 + i)
	end
	ad2.scale = 0.5
	m:set_adapters({ ad, ad2 })
	local both = logits_of(m, ids)
	local mpath2 = tmp(".gguf")
	tiny.write(mpath2, "qwen3", { override = merge(byname, { ad, ad2 }) })
	local merged2 = transformer.load(be, mpath2)
	d = maxdiff(both, logits_of(merged2, ids))
	ok(d < LOOSE, ("two adapters match merged weights (%.2e)"):format(d))
	merged2:close()

	-- Decode with replay against a new graph per pass.
	local function decode(model)
		local cache = model:cache(64)
		local out = { model:step(cache, ids) }
		for _ = 1, 6 do
			out[#out + 1] = model:step(cache, { out[#out] })
		end
		return table.concat(out, " ")
	end
	local noreplay = transformer.load(be, path, { replay = false })
	noreplay:set_adapters({ ad, ad2 })
	local a, b = decode(m), decode(noreplay)
	ok(a == b, "replay with adapters: " .. a)
	m:set_adapters({ ad })
	local c = decode(m)
	ok(c ~= a, "a new adapter set builds a new replay graph")
	local c0 = { n_past = 0, n_ctx = 64 }
	local k1 = m:replay_key(c0, 1, {})
	ad.scale = 0.9
	m:set_adapters({ ad })
	ok(k1 ~= m:replay_key(c0, 1, {}), "the replay key holds the adapter scales")
	ad.scale = 1
	noreplay:close()

	local bad = lora.new(m, { rank = 2 })
	bad.names[#bad.names + 1] = "blk.0.ffn_gate_exps.weight"
	bad.pairs["blk.0.ffn_gate_exps.weight"] = bad.pairs[bad.names[1]]
	local good, err = pcall(m.set_adapters, m, { bad })
	ok(not good and tostring(err):find("ffn_gate_exps", 1, true), "unknown target raises")
	m:set_adapters({})
	ok(maxdiff(logits_of(m, ids), base) == 0, "an empty list removes the adapters")
	m:close()
end

-- Write a PEFT adapter: safetensors with Hugging Face names, and the config.
local HF = {
	attn_q = "self_attn.q_proj", attn_k = "self_attn.k_proj", attn_v = "self_attn.v_proj",
	attn_output = "self_attn.o_proj", ffn_gate = "mlp.gate_proj", ffn_up = "mlp.up_proj",
	ffn_down = "mlp.down_proj",
}
local function write_peft(dir, ad, hp, arch)
	os.execute("mkdir -p '" .. dir .. "'")
	local header, blobs, off = {}, {}, 0
	local function put(name, t, rows)
		local v = t:get()
		local sh = t:shape()
		if rows then
			-- gguf row order back to Hugging Face order for llama Q and K.
			local d, r, out = sh[1] // rows, sh[2], {}
			for h = 0, rows - 1 do
				for j = 0, d // 2 - 1 do
					for x = 0, 1 do
						local src, dst = h * d + j * 2 + x, h * d + x * (d // 2) + j
						for c = 1, r do
							out[dst * r + c] = v[src * r + c]
						end
					end
				end
			end
			v = out
		end
		local parts = {}
		for i, x in ipairs(v) do
			parts[i] = string.pack("<f", x)
		end
		local data = table.concat(parts)
		header[name] = { dtype = "F32", shape = sh, data_offsets = { off, off + #data } }
		blobs[#blobs + 1] = data
		off = off + #data
	end
	for _, name in ipairs(ad.names) do
		local il, t = name:match("^blk%.(%d+)%.(.-)%.weight$")
		local mod = "base_model.model.model.layers." .. il .. "." .. HF[t]
		local rows
		if arch == "llama" and t == "attn_q" then
			rows = hp.n_head
		elseif arch == "llama" and t == "attn_k" then
			rows = hp.n_head_kv
		end
		put(mod .. ".lora_A.weight", ad.pairs[name].a)
		put(mod .. ".lora_B.weight", ad.pairs[name].b, rows)
	end
	local h = json.encode(header)
	local f = assert(io.open(dir .. "/adapter_model.safetensors", "wb"))
	f:write(string.pack("<I8", #h), h, table.concat(blobs))
	f:close()
	f = assert(io.open(dir .. "/adapter_config.json", "w"))
	f:write(json.encode({ r = 4, lora_alpha = ad.alpha, peft_type = "LORA" }))
	f:close()
end

for _, arch in ipairs({ "llama", "qwen3" }) do
	local path = tmp(".gguf")
	tiny.write(path, arch)
	local m = transformer.load(be, path)
	local ad = lora.new(m, { rank = 4, alpha = 16 })
	for i, name in ipairs(ad.names) do
		ad.pairs[name].b:fill("normal", 0, 0.2, 300 + i)
	end
	m:set_adapters({ ad })
	local want = logits_of(m, ids)
	local dir = tmp("-peft")
	write_peft(dir, ad, m.hp, arch)
	local pa = lora.load(m, dir)
	m:set_adapters({ pa })
	local d = maxdiff(logits_of(m, ids), want)
	ok(d < 1e-5 and #pa.names == #ad.names, ("%s: PEFT adapter loads (%.2e)"):format(arch, d))
	os.remove(dir .. "/adapter_model.safetensors")
	os.remove(dir .. "/adapter_config.json")
	os.remove(dir)
	m:close()
end

-- Train a tiny adapter on a q8_0 model until it recalls one sequence.
-- The frozen head of a model this small caps each target probability,
-- so the adapter also takes the output projection.
do
	local path = tmp(".gguf")
	local _, nv = tiny.write(path, "qwen3", { quant = "q8_0" })
	local m = transformer.load(be, path)
	local targets = table.move(lora.TARGETS, 1, #lora.TARGETS, 1, {})
	targets[#targets + 1] = "output"
	local ad = lora.new(m, { rank = 8, alpha = 16, targets = targets })
	m:set_adapters({ ad })
	local seq = { 72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33, 32, 76, 111, 82, 65 }
	local inp = m:inputs(seq, nil, { hidden = m.hp.n_layer })
	local rows, targets = {}, {}
	for i = 1, #seq - 1 do
		rows[i], targets[i] = i - 1, seq[i + 1]
	end
	local sel = be:tensor("i32", { #rows }, rows)
	local labels = be:tensor("f32", { #rows, nv })
	labels:fill_onehot(targets)
	local g <close> = be:graph(8192)
	local x = m:forward(g, inp)
	local logits = transformer.head(g, m.hp, m.output_norm, m.output, g:get_rows(x, sel), m.lora_out)
	local loss = g:cross_entropy_loss(logits, labels)
	loss:backward()
	local opt = optim.adamw(ad:params(), { lr = 1e-2, weight_decay = 0 })
	ok(#ad.names == 2 * 7 + 1, "the adapter takes the output projection")
	local curve = {}
	for step = 1, 80 do
		opt:zero_grad()
		g:compute(loss)
		curve[step] = loss:get()[1]
		opt:step()
	end
	io.stderr:write("overfit loss:")
	for step = 1, #curve, 8 do
		io.stderr:write((" %d:%.3f"):format(step, curve[step]))
	end
	io.stderr:write((" %d:%.4f\n"):format(#curve, curve[#curve]))
	ok(curve[1] > 4 and curve[#curve] < 0.05, ("overfit: loss %.3f -> %.4f"):format(curve[1],
	    curve[#curve]))
	-- The base weights stay as they were.
	local plain = transformer.load(be, path)
	local w0, w1 = plain.w:get("blk.0.attn_q.weight"):bytes(), m.w:get("blk.0.attn_q.weight"):bytes()
	ok(w0 == w1, "the base weights do not change")
	plain:close()
	-- Greedy decode now gives the sequence back.
	local cache = m:cache(64)
	local out = { m:step(cache, { seq[1] }) }
	for i = 2, #seq - 1 do
		out[i] = m:step(cache, { out[i - 1] })
	end
	local want = table.concat(seq, " ", 2)
	ok(table.concat(out, " ") == want, "the trained adapter recalls the sequence")
	m:close()
end

-- The chat engine with adapters: requests change scales, and the prompt
-- store keeps each adapter set apart.
do
	local engine = require("lgml.engine")
	local getopt = require("lgml.getopt")
	local path = tmp(".gguf")
	tiny.write(path, "qwen3")
	local m = transformer.load(be, path)
	local ad = lora.new(m, { rank = 4, alpha = 8 })
	for i, name in ipairs(ad.names) do
		ad.pairs[name].b:fill("normal", 0, 0.3, 400 + i)
	end
	local apath = tmp(".gguf")
	assert(ad:save(apath))
	m:close()

	local function req(e, msgs, extra)
		local parts, r = {}, {}
		local sink = {}
		function sink.content(_, t)
			parts[#parts + 1] = t
			return true
		end
		sink.reasoning, sink.tool_arguments = sink.content, sink.content
		function sink.tool_call()
			return 0
		end
		function sink.finish(_, reason, usage, timings)
			r.reason, r.usage, r.timings = reason, usage, timings
		end
		function sink.cancelled()
			return false
		end
		local body = { messages = msgs, max_tokens = 6, top_k = 1,
		    chat_template_kwargs = { enable_thinking = false } }
		for k, v in pairs(extra or {}) do
			body[k] = v
		end
		local done
		coroutine.wrap(function()
			r.res, r.err, r.status = e:chat(body, sink)
			done = true
		end)()
		while not done do
			require("luv").run("once")
		end
		r.text = table.concat(parts)
		return r
	end
	local sys = { role = "system", content = ("You are a small test model. "):rep(3) }
	local msgs = { sys, { role = "user", content = "Say something." } }
	local plain = engine.new({ model = path, backend = "CPU", n_ctx = 256, flash_attn = false })
	local e = engine.new({ model = path, backend = "CPU", n_ctx = 256, flash_attn = false,
	    lora = { { path = apath, scale = 1 } } })
	local p = e:props().lora
	ok(#p == 1 and p[1].id == 0 and p[1].path == apath and p[1].scale == 1, "props list the adapter")
	local base = req(plain, msgs).text
	local a1 = req(e, msgs)
	ok(a1.text ~= base, ("the adapter changes the answer: %q -> %q"):format(base, a1.text))
	local off = req(e, msgs, { lora = { { id = 0, scale = 0 } } })
	ok(off.text == base and off.timings.cache_n == 0,
	    "scale 0 gives the base answer and does not reuse the adapter cache")
	local a2 = req(e, msgs)
	ok(a2.text == a1.text and a2.timings.restored_n > 0,
	    ("back to the adapter restores its store entry (%d tokens)"):format(a2.timings.restored_n))
	local bad = req(e, msgs, { lora = { { id = 3, scale = 1 } } })
	ok(bad.res == nil and bad.status == 400, "a bad adapter id gives 400")
	ok(e:lora_adapters({ { id = 0, scale = 0 } }) and e:props().lora[1].scale == 0,
	    "lora_adapters sets the global scale that props shows")
	ok(req(e, msgs).text == base, "a request with no lora uses the global scales")
	ok(req(e, msgs, { lora = { { id = 0, scale = 1 } } }).text == a1.text and e:props().lora[1].scale == 0,
	    "a request lora applies to that request only")
	e:set_lora(nil)
	ok(e.lora_key == "0", "no lora list (as decide gives) keeps the global scales")
	ok(e:lora_adapters({ { id = 0, scale = 1 } }) and req(e, msgs, { lora = {} }).text == base,
	    "adapters that a request lora does not name get scale 0")
	local _, lmsg, lstatus = e:lora_adapters({ { id = 1, scale = 1 } })
	ok(lstatus == 400 and lmsg:find("id from 0 to 0", 1, true) and e:props().lora[1].scale == 1,
	    "lora_adapters rejects a bad id and keeps the scales")
	local o = getopt.parse({ "--lora", "a.gguf:0.5", "--lora=b" }, "", { ["lora="] = false },
	    { lora = true })
	ok(o and #o.lora == 2 and o.lora[1] == "a.gguf:0.5" and o.lora[2] == "b",
	    "getopt collects repeated options")
	local f, s = lora.parse_spec("dir/a.gguf:0.5")
	local f2, s2 = lora.parse_spec("dir/b.gguf")
	ok(f == "dir/a.gguf" and s == 0.5 and f2 == "dir/b.gguf" and s2 == 1, "parse file[:scale]")
end

for _, p in ipairs(files) do
	os.remove(p)
end
print("1.." .. n)
os.exit(failed and 1 or 0)
