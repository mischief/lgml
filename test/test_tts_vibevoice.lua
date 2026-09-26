-- VibeVoice ops, noise, schedule and tokenizer tests, then model and
-- engine tests. Output is TAP. Set LGML_VIBEVOICE to the model gguf or
-- to a directory with the model and vibevoice-voice-*.gguf presets.
local lgml = require("lgml")
local rng = require("lgml.vibevoice.rng")
local diffusion = require("lgml.vibevoice.diffusion")
local tokenizer = require("lgml.vibevoice.tokenizer")

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

local function maxdiff(a, b)
	if #a ~= #b then
		return math.huge
	end
	local m = 0
	for i = 1, #a do
		m = math.max(m, math.abs(a[i] - b[i]))
	end
	return m
end

local function rand(r, k)
	local t = {}
	for i = 1, k do
		t[i] = r:uniform() * 2 - 1
	end
	return t
end

-- Causal conv reference: x (T, IC), w (OC, IC, K), b (OC).
local function conv_ref(x, w, b, T, IC, OC, K)
	local y = {}
	for t = 0, T - 1 do
		for o = 0, OC - 1 do
			local s = b[o + 1]
			for i = 0, IC - 1 do
				for k = 0, K - 1 do
					local ti = t - (K - 1) + k
					if ti >= 0 then
						s = s + w[(o * IC + i) * K + k + 1] * x[ti * IC + i + 1]
					end
				end
			end
			y[t * OC + o + 1] = s
		end
	end
	return y
end

-- Transposed conv reference: w (IC, OC, K), output cropped to T * S.
local function convtr_ref(x, w, b, T, IC, OC, K, S)
	local y = {}
	for i = 1, T * S * OC do
		y[i] = b[(i - 1) % OC + 1]
	end
	for t = 0, T - 1 do
		for i = 0, IC - 1 do
			for o = 0, OC - 1 do
				for k = 0, K - 1 do
					local to = t * S + k
					if to < T * S then
						local j = to * OC + o + 1
						y[j] = y[j] + x[t * IC + i + 1] * w[(i * OC + o) * K + k + 1]
					end
				end
			end
		end
	end
	return y
end

local function test_ops(be)
	local name = be:name()
	local r = rng.new(1)
	local T, IC, OC, K, S = 6, 3, 4, 3, 2
	local x, w, b = rand(r, T * IC), rand(r, OC * IC * K), rand(r, OC)
	local tx = be:tensor("f32", { T, IC }, x)
	local tb = be:tensor("f32", { OC }, b)
	local g0 <close> = be:graph()
	local g <close> = be:graph()
	local y1 = g:vibevoice_conv(tx, be:tensor("f32", { OC, IC, K }, w), tb)
	local dw, db = rand(r, K * IC), rand(r, IC)
	local y2 = g:vibevoice_dwconv(tx, be:tensor("f32", { K, IC }, dw), be:tensor("f32", { IC }, db))
	local wt = rand(r, IC * OC * 4)
	local wp = g0:reshape(g0:cont(g0:permute(be:tensor("f32", { IC, OC, 4 }, wt), { 2, 3, 1 })),
	    { OC * 4, IC })
	local wpl = be:tensor("f32", { OC * 4, IC })
	g0:compute(g0:cpy(wp, wpl))
	local g2 <close> = be:graph()
	local y3 = g2:vibevoice_convtr(tx, wpl, tb, S)
	local y4 = g2:vibevoice_peak(tx, 3)
	g:compute(y1, y2)
	g2:compute(y3, y4)
	ok(maxdiff(y1:get(), conv_ref(x, w, b, T, IC, OC, K)) < 1e-5, name .. " vibevoice_conv")
	local dwf = {}
	for o = 0, IC - 1 do
		for k = 0, K - 1 do
			dwf[(o * IC + o) * K + k + 1] = dw[k * IC + o + 1]
		end
	end
	for i = 1, IC * IC * K do
		dwf[i] = dwf[i] or 0
	end
	ok(maxdiff(y2:get(), conv_ref(x, dwf, db, T, IC, IC, K)) < 1e-5, name .. " vibevoice_dwconv")
	ok(maxdiff(y3:get(), convtr_ref(x, wt, b, T, IC, OC, 4, S)) < 1e-5, name .. " vibevoice_convtr")
	local pk = {}
	for i = 0, #x // 3 - 1 do
		pk[i + 1] = math.max(math.abs(x[i * 3 + 1]), math.abs(x[i * 3 + 2]), math.abs(x[i * 3 + 3]))
	end
	ok(maxdiff(y4:get(), pk) < 1e-6, name .. " vibevoice_peak")
	local g3 <close> = be:graph()
	ok(not pcall(g3.vibevoice_convtr, g3, tx, wpl, tb, 8), name .. " convtr rejects stride > kernel")
end

local function test_rng()
	local a, b = rng.new(42):normal(64), rng.new(42):normal(64)
	ok(maxdiff(a, b) == 0 and math.abs(a[1] - 1.9269152879715) < 1e-6 and
	    math.abs(rng.new(7):normal(16)[1] - a[1]) > 1e-3, "torch-style normal noise")
end

local function test_schedule()
	local s = diffusion.schedule(20, 1000)
	ok(#s == 20 and s[1].t == 999 and s[2].t == 949 and s[20].t == 50 and
	    s[1].C == 0 and s[20].C == 0 and s[10].C ~= 0, "DPM-Solver++ schedule")
end

local function test_pieces()
	local p = tokenizer.pieces("Hello, world! It's 42  ok\n")
	ok(table.concat(p, "|") == "Hello|,| world|!| It|'s| |4|2| | ok|\n", "pre-tokenizer pieces")
end

local function test_chunks(vv)
	local stub = {}
	function stub.encode(_, s)
		local t = {}
		for w in s:gmatch("%S+") do
			t[#t + 1] = w
		end
		return t
	end
	local text = "One two three. Four five six. Seven eight nine ten eleven twelve."
	local a = table.concat(vv.chunks(stub, text, 6), "|")
	local b = table.concat(vv.chunks(stub, text, 4), "|")
	ok(a == "One two three. Four five six.|Seven eight nine ten eleven twelve." and
	    b == "One two three.|Four five six.|Seven eight nine ten|eleven twelve.", "sentence chunks")
end

local function test_model(path, vv)
	local model = require("lgml.vibevoice.model")
	local e = vv.new({ model = path })
	local m = e.model
	local ids = m.tk:encode("Hello, how are you today?")
	ok(table.concat(ids, " ") == "9707 11 1246 525 498 3351 30", "tokenizer ids")
	ids[#ids + 1] = model.NEWLINE
	local v = m:voice(e:resolve_voice("en-Carter_man"))
	local frames = m:generate(ids, v, { seed = 42, cfg = 3.0, steps = 20 })
	local lat = m.io.latents:get()
	local finite = true
	for i = 1, frames * 64 do
		finite = finite and lat[i] == lat[i] and math.abs(lat[i]) < 100
	end
	ok(frames >= 6 and frames <= 20 and finite, "generate latents (" .. tostring(frames) .. " frames)")

	if m.be:name() ~= "CPU" then
		local cpu = model.load(assert(lgml.backend("CPU")), m.path)
		local f2 = cpu:generate(ids, cpu:voice(v.path), { seed = 42, cfg = 3.0, steps = 20 })
		local l2 = cpu.io.latents:get()
		local d = 0
		for i = 1, 64 do
			d = math.max(d, math.abs(lat[i] - l2[i]))
		end
		ok(f2 == frames and d < 0.2, ("CPU and %s agree (frame 0 diff %.3g)"):format(m.be:name(), d))
		cpu.w:close()
	else
		skip("cross-backend check: no GPU")
	end

	local req = { input = "Hello, how are you today?", voice = "alloy", seed = 3 }
	local r1 = assert(e:speak(req))
	local r2 = assert(e:speak(req))
	local d = r1.audio:duration()
	ok(r1.audio:rate() == 24000 and d > 0.6 and d < 4 and r1.audio:wav() == r2.audio:wav(),
	    ("speak is deterministic (%.2f s)"):format(d))
	ok(select(3, e:speak({ input = "x", speed = 1.5 })) == 400 and
	    select(3, e:speak({ input = "x", voice = "nobody" })) == 400 and
	    select(3, e:speak({ input = "  " })) == 400 and
	    e:speak({ input = "x", speed = 1 }) ~= nil, "speak rejects speed, voice and empty input")
	local _, _, st = e:speak({ input = "This one is cancelled." }, {
		cancelled = function()
			return true
		end,
	})
	ok(st == 499, "speak stops when cancelled")
	e:close()
end

for _, dev in ipairs(lgml.devices()) do
	local be <close> = lgml.backend(dev.name)
	if be then
		test_ops(be)
	end
end
test_rng()
test_schedule()
test_pieces()
test_chunks(require("lgml.tts.vibevoice"))

local path = os.getenv("LGML_VIBEVOICE")
if path and path ~= "" then
	test_model(path, require("lgml.tts.vibevoice"))
else
	skip("vibevoice model: LGML_VIBEVOICE not set")
end
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
