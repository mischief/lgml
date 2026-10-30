-- Granite on tiny models: with its scales at their llama values it gives
-- the llama logits, and each scale changes them as llama.cpp applies it.
-- Output is TAP.
local lgml = require("lgml")
local transformer = require("lgml.transformer")

local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local tiny = require("tiny")
local gguf_mod = require("gguf")

local n, failed = 0, false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local be <close> = assert(lgml.backend("CPU"))
local files = {}

local function logits_of(path, ids)
	local m = transformer.load(be, path)
	local inp = m:inputs(ids, nil, {})
	local g <close> = be:graph()
	local out = m:forward(g, inp)
	g:compute(out)
	return out:get(), m.hp
end

local function maxdiff(a, b)
	local d = #a == #b and 0 or math.huge
	for i = 1, #a do
		d = math.max(d, math.abs(a[i] - b[i]))
	end
	return d
end

-- The weights of tiny.write come from a seed of the arch name length;
-- this seed gives granite the weights of llama.
local SEED = (#"llama" - #"granite") * 7919

local function granite(kv)
	local path = tiny.tmp(".gguf")
	files[#files + 1] = path
	local list = {}
	for k, v in pairs(kv) do
		list[#list + 1] = { "granite." .. k, gguf_mod.FLOAT32, v }
	end
	tiny.write(path, "granite", { seed = SEED, kv = list })
	return path
end

local ids = { 5, 17, 42, 99, 3, 64 }
local lpath = tiny.tmp(".gguf")
files[#files + 1] = lpath
tiny.write(lpath, "llama")
local base = logits_of(lpath, ids)

local D = tiny.D
local plain = { ["attention.scale"] = 1 / math.sqrt(D), embedding_scale = 1, residual_scale = 1, logit_scale = 1 }
local g1, hp = logits_of(granite(plain), ids)
ok(hp.res_scale == nil and hp.embd_scale == nil and hp.logit_scale == nil, "scales of 1 add no ops")
ok(maxdiff(g1, base) < 1e-5, "granite with llama scales gives the llama logits")

local half = {}
for k, v in pairs(plain) do
	half[k] = v
end
half.logit_scale = 2
local g2 = logits_of(granite(half), ids)
local d = 0
for i = 1, #base do
	d = math.max(d, math.abs(g2[i] * 2 - base[i]))
end
ok(d < 1e-5, "logit_scale divides the logits")

for _, key in ipairs({ "attention.scale", "embedding_scale", "residual_scale" }) do
	local t = {}
	for k, v in pairs(plain) do
		t[k] = v
	end
	t[key] = key == "attention.scale" and 1 / D or 0.5
	ok(maxdiff(logits_of(granite(t), ids), base) > 1e-3, key .. " changes the logits")
end

for _, p in ipairs(files) do
	os.remove(p)
end
print("1.." .. n)
os.exit(failed and 1 or 0)
