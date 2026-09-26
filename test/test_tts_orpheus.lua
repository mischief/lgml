-- Orpheus TTS: codec ops, tokenizer, talker, SNAC and the engine.
-- Output is TAP. Set LGML_ORPHEUS to a directory with the orpheus LM
-- gguf and snac-24khz.gguf to run the model tests. The SNAC and talker
-- checks use CrispASR reference dumps in diff-harness-ref/ when found.
local lgml = require("lgml")

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

local function close(a, b, tol)
	if #a ~= #b then
		return false
	end
	for i = 1, #a do
		if math.abs(a[i] - b[i]) > tol then
			io.stderr:write(("at %d: %g ~= %g\n"):format(i, a[i], b[i]))
			return false
		end
	end
	return true
end

local function cosine(a, b)
	local d, na, nb = 0, 0, 0
	for i = 1, #b do
		d, na, nb = d + a[i] * b[i], na + a[i] ^ 2, nb + b[i] ^ 2
	end
	return d / math.sqrt(na * nb)
end

local function seq(len, f)
	local t = {}
	for i = 1, len do
		t[i] = f(i)
	end
	return t
end

local function run(be, f)
	local g <close> = be:graph()
	local r = f(g)
	g:compute(r)
	return r:get()
end

-- Codec ops against plain Lua, on the CPU.
do
	local be <close> = assert(lgml.backend("CPU"))
	local C, T, K = 3, 10, 3
	local xs = seq(C * T, function(i) return math.sin(i * 0.7) * 2 end)
	local x = be:tensor("f32", { C, T }, xs)

	local al = { 0.5, 1, 1.5 }
	local got = run(be, function(g)
		return g:orpheus_snake(x, be:tensor("f32", { C }, al))
	end)
	local want = seq(C * T, function(i)
		local a = al[(i - 1) // T + 1]
		return xs[i] + math.sin(a * xs[i]) ^ 2 / a
	end)
	ok(close(got, want, 1e-5), "orpheus_snake")

	local ws = seq(C * K, function(i) return (i % 5) - 2 end)
	local p, d = 2, 2
	got = run(be, function(g)
		return g:conv1d_dw_orpheus(be:tensor("f32", { C, 1, K }, ws), x, p, d)
	end)
	want = {}
	local ol = T + 2 * p - d * (K - 1)
	for c = 0, C - 1 do
		for t = 0, ol - 1 do
			local s = 0
			for k = 0, K - 1 do
				local j = t - p + k * d
				if j >= 0 and j < T then
					s = s + ws[c * K + k + 1] * xs[c * T + j + 1]
				end
			end
			want[#want + 1] = s
		end
	end
	ok(close(got, want, 1e-4), "conv1d_dw_orpheus")

	local OC, S, P, K2 = 2, 3, 1, 5
	local wt = seq(C * OC * K2, function(i) return math.cos(i) end)
	got = run(be, function(g)
		return g:conv_transpose1d_orpheus(be:tensor("f32", { C, OC, K2 }, wt), x, S, P)
	end)
	local tl = (T - 1) * S + K2 - 2 * P
	want = seq(OC * tl, function() return 0 end)
	for ic = 0, C - 1 do
		for oc = 0, OC - 1 do
			for t = 0, T - 1 do
				for k = 0, K2 - 1 do
					local o = t * S + k - P
					if o >= 0 and o < tl then
						local w = wt[(ic * OC + oc) * K2 + k + 1]
						local i = oc * tl + o + 1
						want[i] = want[i] + w * xs[ic * T + t + 1]
					end
				end
			end
		end
	end
	ok(close(got, want, 1e-4), "conv_transpose1d_orpheus")
end

-- Engine helpers without a model.
do
	local orpheus = require("lgml.tts.orpheus")
	ok(orpheus.voice("alloy") == "tara" and orpheus.voice("Leo") == "leo" and
	    orpheus.voice(nil) == "tara" and orpheus.voice("nobody") == nil, "voice names")
	local parts = orpheus.split(("One sentence here. "):rep(30))
	local fits = #parts > 1
	for _, s in ipairs(parts) do
		fits = fits and #s <= 200 and s:match("%.$") ~= nil
	end
	ok(fits, "split at sentence ends")
end

local dir = os.getenv("LGML_ORPHEUS")
if not dir or dir == "" then
	for _, s in ipairs({ "tokenizer", "talker", "snac reference", "speak" }) do
		skip(s .. " (LGML_ORPHEUS not set)")
	end
	print("1.." .. n)
	return
end

local orpheus = require("lgml.tts.orpheus")
local backend = os.getenv("LGML_BACKEND") or "Vulkan0"
local e = orpheus.new({ model = dir, backend = backend })

-- "tara: Hey there, my name is Tara." from the Orpheus prompt.
local body = { 83, 5169, 25, 28653, 1070, 11, 856, 836, 374, 70797, 13 }
local prompt = e:prompt("tara", "Hey there, my name is Tara.")
local want = { 128259, 128000 }
table.move(body, 1, #body, 3, want)
table.move({ 128009, 128260, 128261, 128257 }, 1, 4, #want + 1, want)
ok(close(prompt, want, 0), "tokenizer")

do
	e.cache.n_past = 0
	local logits = e:eval(prompt)
	local smp <close> = lgml.sampler({ temperature = 0 })
	local best = smp:sample(logits)
	ok(best == 129214, "talker first code")
end

local function file(p)
	local f = io.open(p, "rb")
	return f and f:close() and p
end

local ref = file(dir .. "/diff-harness-ref/orpheus-snac-ref.gguf")
if ref then
	local r <close> = e.be:load(ref)
	local codes = {}
	for k = 0, 2 do
		local c = r:get("snac_codes_" .. k):get()
		codes[k + 1] = seq(#c, function(i) return math.tointeger(c[i]) end)
	end
	local pcm = e.codec:decode(codes)
	ok(cosine(pcm:get(), r:get("snac_pcm"):get()) > 0.9999, "snac reference")
	pcm:close()
else
	skip("snac reference (no diff-harness-ref)")
end

local uv = require("luv")
local res
coroutine.wrap(function()
	res = { e:speak({ input = "Hello there.", voice = "tara", seed = 1 },
	    { cancelled = function() return false end }) }
end)()
uv.run()
local d = res[1] and res[1].audio:duration() or 0
ok(d > 0.3 and d < 5 and res[1].audio:rate() == 24000, "speak")
print("1.." .. n)
