-- Qwen3-TTS: codec ops, text split, engine. Output is TAP. Set
-- LGML_QWEN3TTS to a directory with the talker, codec and tokenizer
-- gguf files to run the model tests. LGML_QWEN3TTS_BACKEND picks the
-- backend (default: best).
local lgml = require("lgml")
local qwen3tts = require("lgml.qwen3tts")

local n = 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		os.exit(1)
	end
end

local function skip(name)
	n = n + 1
	print("ok " .. n .. " # SKIP " .. name)
end

local function rnd(k)
	local t = {}
	for i = 1, k do
		t[i] = math.random() * 2 - 1
	end
	return t
end

local function maxdiff(a, b)
	local m = 0
	for i = 1, #b do
		m = math.max(m, math.abs(a[i] - b[i]))
	end
	return m
end

math.randomseed(7)
local be = assert(lgml.backend("CPU"))
local T, IC, OC = 9, 5, 4
local xd = rnd(T * IC)
local x = be:tensor("f32", { T, IC }, xd)

-- Causal conv with dilation: x (T, IC), w (K, OC, IC).
do
	local K, D = 3, 2
	local wd, bd = rnd(K * OC * IC), rnd(OC)
	local g <close> = be:graph()
	local y = g:qwen3tts_conv(x, be:tensor("f32", { K, OC, IC }, wd),
	    be:tensor("f32", { OC }, bd), D)
	g:compute(y)
	local want = {}
	for t = 0, T - 1 do
		for o = 0, OC - 1 do
			local s = bd[o + 1]
			for k = 0, K - 1 do
				local ti = t - (K - 1 - k) * D
				for i = 0, ti >= 0 and IC - 1 or -1 do
					s = s + wd[(k * OC + o) * IC + i + 1] * xd[ti * IC + i + 1]
				end
			end
			want[#want + 1] = s
		end
	end
	ok(maxdiff(y:get(), want) < 1e-5, "qwen3tts_conv matches a direct sum")
end

-- Causal transposed conv: w (OC, K, IC), output (T * stride, OC).
do
	local K, S = 4, 2
	local wd = rnd(OC * K * IC)
	local g <close> = be:graph()
	local y = g:qwen3tts_tconv(x, be:tensor("f32", { OC, K, IC }, wd), nil, S)
	g:compute(y)
	local want = {}
	for t = 0, T * S - 1 do
		for o = 0, OC - 1 do
			local s = 0
			for ti = 0, T - 1 do
				local k = t - ti * S
				for i = 0, (k >= 0 and k < K) and IC - 1 or -1 do
					s = s + wd[(o * K + k) * IC + i + 1] * xd[ti * IC + i + 1]
				end
			end
			want[#want + 1] = s
		end
	end
	local s = y:shape()
	ok(s[1] == T * S and s[2] == OC and maxdiff(y:get(), want) < 1e-5,
	    "qwen3tts_tconv matches a direct sum")
end

-- Depthwise conv and SnakeBeta.
do
	local K = 3
	local wd, a, b = rnd(K * IC), rnd(IC), rnd(IC)
	local g <close> = be:graph()
	local y = g:qwen3tts_dwconv(x, be:tensor("f32", { K, IC }, wd))
	local z = g:qwen3tts_snake(x, be:tensor("f32", { IC }, a), be:tensor("f32", { IC }, b))
	g:compute(y, z)
	local wy, wz = {}, {}
	for t = 0, T - 1 do
		for c = 0, IC - 1 do
			local s = 0
			for k = 0, K - 1 do
				local ti = t - (K - 1 - k)
				s = s + (ti >= 0 and wd[k * IC + c + 1] * xd[ti * IC + c + 1] or 0)
			end
			local v = xd[t * IC + c + 1]
			wy[#wy + 1] = s
			wz[#wz + 1] = v + math.exp(-b[c + 1]) * math.sin(math.exp(a[c + 1]) * v) ^ 2
		end
	end
	ok(maxdiff(y:get(), wy) < 1e-5, "qwen3tts_dwconv matches a direct sum")
	ok(maxdiff(z:get(), wz) < 1e-5, "qwen3tts_snake matches the formula")
end

do
	local g <close> = be:graph()
	local y = g:concat_qwen3tts(x, g:qwen3tts_clamp(x, -0.1, 0.1), 1)
	g:compute(y)
	local v = y:get()
	local joined = #v == 2 * T * IC and math.abs(v[1] - xd[1]) < 1e-6
	ok(joined and math.abs(v[T * IC + 1]) <= 0.1 + 1e-6, "concat_qwen3tts and qwen3tts_clamp")
	ok(not pcall(g.qwen3tts_conv, g, x, be:tensor("f32", { 3, OC, IC + 1 })),
	    "qwen3tts_conv rejects a weight of the wrong shape")
end

do
	local long = ("Word, "):rep(100) .. "end. Short one! And a third?"
	local runs = qwen3tts.split(long)
	local fits = true
	for _, r in ipairs(runs) do
		fits = fits and #r <= qwen3tts.MAX_CHARS
	end
	ok(#runs == 2 and fits, "split keeps runs under MAX_CHARS")
	runs = qwen3tts.split("One. Two!  Three?")
	ok(#runs == 1 and runs[1] == "One. Two!  Three?", "split keeps short text whole")
end

local dir = os.getenv("LGML_QWEN3TTS")
if not dir or dir == "" then
	for _, name in ipairs({ "engine loads", "speak makes speech", "speed 2 is refused",
	    "unknown voice is refused", "cancel stops the request" }) do
		skip(name .. " (LGML_QWEN3TTS not set)")
	end
	print("1.." .. n)
	return
end

local tts = require("lgml.tts.qwen3tts")
local backend = os.getenv("LGML_QWEN3TTS_BACKEND")
local okload, e = pcall(tts.new, { model = dir, backend = backend ~= "" and backend or nil })
ok(okload, "engine loads")

local r = e:speak({ input = "Hello there, this is a test.", voice = "alloy", seed = 3 })
local d = r and r.audio and r.audio:duration() or 0
ok(r and r.audio:rate() == 24000 and d > 0.5 and d < 8, "speak makes speech")

local r2, _, st2 = e:speak({ input = "Hello.", speed = 2 })
ok(r2 == nil and st2 == 400, "speed 2 is refused")

local r3, _, st3 = e:speak({ input = "Hello.", voice = "nobody" })
ok(r3 == nil and st3 == 400, "unknown voice is refused")

local r4, m4 = e:speak({ input = "This request stops at once." }, {
	cancelled = function() return true end,
})
ok(r4 == nil and m4 == "cancelled", "cancel stops the request")

print("1.." .. n)
