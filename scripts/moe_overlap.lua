-- Time MoE layers of gemma-4-26B-A4B size with the experts on the
-- device, on the CPU, or split, to see if the scheduler runs the two
-- parts at the same time. A dense product stands in for attention.
-- Usage: lua5.4 scripts/moe_overlap.lua [device] [tokens] [layers]
-- Environment: QT, NE, REPS, HALVES (device experts of the splits),
-- STATIC (layer counts with all experts on the device), LOGDBG.
local lgml = require("lgml")
if os.getenv("LOGDBG") then
	lgml.log("debug")
end

local dev = arg[1] or "Vulkan0"
local T = tonumber(arg[2]) or 1
local NL = tonumber(arg[3]) or 30
local E, F, NE, K = 2816, 704, tonumber(os.getenv("NE")) or 128, 8
local QT = os.getenv("QT") or "q4_0"
local REPS = tonumber(os.getenv("REPS")) or 20

local be <close> = assert(lgml.backend(dev, { cpu = true }))
local cb = be:cpu()

-- Random bytes of one (rows, cols) matrix in type QT.
local function qmat(rows, cols, seed)
	local f = cb:tensor("f32", { rows, cols })
	f:fill("normal", 0, 0.02, seed)
	return lgml.quantize(QT, f:bytes(), cols)
end

-- The experts go through a gguf file, so that they load as weights:
-- the scheduler runs an op where its weights are.
local QID = { q4_0 = 2, q8_0 = 8, f16 = 1 }
local path = os.tmpname()
do
	local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
	package.path = here .. "../test/?.lua;" .. package.path
	local gguf = require("gguf")
	local gb, ub, db = qmat(F, E, 11), qmat(F, E, 12), qmat(E, F, 13)
	local ts = {
		{ "gate", { NE, F, E }, gb:rep(NE), QID[QT] },
		{ "up", { NE, F, E }, ub:rep(NE), QID[QT] },
		{ "down", { NE, E, F }, db:rep(NE), QID[QT] },
		{ "att", { E, E }, qmat(E, E, 5), QID[QT] },
	}
	-- A router for each layer, so that layers pick other experts.
	for l = 1, NL do
		local r = cb:tensor("f32", { NE, E })
		r:fill("normal", 0, 0.1, 100 + l)
		ts[#ts + 1] = { "inp." .. l, { NE, E }, r:bytes() }
	end
	gguf.write(path, {}, ts)
end
local gstore = assert(be:load(path))
local cstore = assert(be:load(path, { cpu = { gate = true, up = true, down = true } }))
os.remove(path)

-- An (n, rows, cols) expert tensor on the device, from the first n of w.
local function experts(w, n, rows, cols)
	local t = be:tensor(QT, { n, rows, cols }, nil, { weights = true })
	t:set(w:bytes():sub(1, n * rows * lgml.row_size(QT, cols)))
	return t
end

local x0 = be:tensor("f32", { T, E })
x0:fill("normal", 0, 1, 3)

-- The FFN of the experts in w for ids (T, K); the sum over K.
local function ffn(g, w, h, ids)
	local x3 = g:reshape(h, { T, 1, E })
	local gate = g:mul_mat_id(w.gate, x3, ids)
	local up = g:mul_mat_id(w.up, x3, ids)
	local ex = g:mul_mat_id(w.down, g:swiglu_split(gate, up), ids)
	local st = ex:strides()
	local out = g:view(ex, { T, E }, { st[1] })
	for j = 1, K - 1 do
		out = g:add(out, g:view(ex, { T, E }, { st[1] }, j * st[2]))
	end
	return out
end

local function weights(st)
	return { gate = st:get("gate"), up = st:get("up"), down = st:get("down") }
end

local function cache(w, n)
	return { gate = experts(w.gate, n, F, E), up = experts(w.up, n, F, E),
	    down = experts(w.down, n, E, F) }
end

-- A map of expert ids: keep[id] gives id's slot, else dummy.
local function idmap(b, keep, dummy)
	local v = {}
	for i = 0, NE - 1 do
		v[i + 1] = keep(i) or dummy
	end
	local t = b:tensor("i32", { NE, 1 }, nil, { weights = true })
	t:set(v)
	return t
end

local function remap(g, map, sel)
	return g:reshape(g:get_rows(map, g:reshape(g:cont(sel), { T * K })), { T, K })
end

local function build(mode, gw, cw, half)
	local g = be:graph()
	local x = x0
	local gmap, cmap
	if mode ~= "gpu" and mode ~= "cpu" and mode ~= "static" then
		-- Experts below half are on the device; slot half is all zero.
		gmap = idmap(be, function(i) return i < half and i or nil end, half)
		cmap = idmap(be, function(i) return i >= half and i or nil end, half)
	end
	for l = 1, NL do
		-- A dense product on the device stands in for attention.
		x = g:add(x, g:scale(g:mul_mat(cstore:get("att"), g:rms_norm(x, 1e-6)), 0.1))
		local h = g:rms_norm(x, 1e-6)
		local sel = g:argsort_top_k(g:soft_max(g:mul_mat(cstore:get("inp." .. l), h)), K)
		local out
		if mode == "gpu" then
			out = ffn(g, gw, h, sel)
		elseif mode == "cpu" then
			out = ffn(g, cw, h, sel)
		elseif mode == "static" then
			out = ffn(g, l <= half and gw or cw, h, sel)
		elseif mode == "base" then
			out = g:scale(h, 0)
		elseif mode == "gpart" then
			out = ffn(g, gw, h, remap(g, gmap, sel))
		elseif mode == "cpart" then
			out = ffn(g, cw, h, remap(g, cmap, sel))
		else
			local co = ffn(g, cw, h, remap(g, cmap, sel))
			local go = ffn(g, gw, h, remap(g, gmap, sel))
			out = g:add(co, go)
		end
		x = g:add(x, g:scale(out, 0.1))
	end
	return g, x
end

local function time(name, g, x)
	g:compute(x)
	local t0 = lgml.clock()
	for _ = 1, REPS do
		g:compute(x)
	end
	local dt = (lgml.clock() - t0) / REPS
	print(("%-12s %8.2f ms/pass  splits %d"):format(name, dt * 1e3, g:splits()))
	return dt
end

print(("%s T=%d layers=%d %s E=%d F=%d experts=%d used=%d threads=%d"):format(dev, T, NL, QT,
    E, F, NE, K, cb:threads()))
local gw_all = weights(gstore)
local cw_all = weights(cstore)
local g, x = build("gpu", gw_all, nil)
time("all-gpu", g, x)
g:close()
g, x = build("cpu", nil, cw_all)
time("all-cpu", g, x)
g:close()
g, x = build("base", nil, nil, 1)
local base = time("base", g, x)
g:close()
for k in (os.getenv("STATIC") or ""):gmatch("%d+") do
	-- The experts of the first k layers on the device, as n_cpu_moe does.
	g, x = build("static", gw_all, cw_all, tonumber(k))
	time(("static %d/%d"):format(tonumber(k), NL), g, x)
	g:close()
end
local halves = {}
for v in (os.getenv("HALVES") or "32,64,96"):gmatch("%d+") do
	halves[#halves + 1] = tonumber(v)
end
for _, half in ipairs(halves) do
	-- The device cache holds `half` experts and a zero slot.
	local gw = cache(gw_all, half + 1)
	for _, t in pairs(gw) do
		local rb = lgml.row_size(QT, t:shape()[3]) * t:shape()[2]
		local b = t:bytes()
		t:set(b:sub(1, rb * half) .. ("\0"):rep(rb))
	end
	local tt = {}
	for _, m in ipairs({ "gpart", "cpart", "split" }) do
		g, x = build(m, gw, cw_all, half)
		tt[m] = time(("%s %d/%d"):format(m, half, NE), g, x)
		g:close()
	end
	print(("  serial sum %.2f ms, overlap bound %.2f ms; device experts alone %.2f ms"):format(
	    (tt.gpart + tt.cpart) * 1e3, math.max(tt.gpart, tt.cpart) * 1e3, (tt.gpart - base) * 1e3))
	for _, t in pairs(gw) do
		t:close()
	end
end
