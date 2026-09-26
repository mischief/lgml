-- LoRA adapters on tiny models: inference against merged weights,
-- replay, and training a tiny adapter to overfit one sequence. Output
-- is TAP.
local lgml = require("lgml")
local transformer = require("lgml.transformer")
local lora = require("lgml.lora")
local optim = require("lgml.optim")

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

-- Inference: a new adapter changes nothing, and a trained one matches
-- the merged weights.
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
	ok(d < 1e-3, ("adapter matches merged weights (%.2e)"):format(d))
	merged:close()

	ad.scale = 0
	m:set_adapters({ ad })
	ok(maxdiff(logits_of(m, ids), base) == 0, "scale 0 gives the base logits")
	ad.scale = 1
	m:set_adapters({ ad })

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
	ok(d < 1e-3, ("two adapters match merged weights (%.2e)"):format(d))
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

for _, p in ipairs(files) do
	os.remove(p)
end
print("1.." .. n)
os.exit(failed and 1 or 0)
