-- Basic ops on every device. Output is TAP.
local lgml = require("lgml")

local n = 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		os.exit(1)
	end
end

local function near(got, want, name)
	local good = #got == #want
	for i = 1, #want do
		if not good or math.abs(got[i] - want[i]) > 1e-3 then
			good = false
			break
		end
	end
	if not good then
		io.stderr:write(name, ": got {", table.concat(got, ", "), "}\n")
	end
	ok(good, name)
end

local devs = lgml.devices()
ok(#devs >= 1, "devices listed")

for _, d in ipairs(devs) do
	local be <close> = assert(lgml.backend(d.name))
	local tag = d.name .. ": "

	local a <close> = be:tensor("f32", { 2, 3 }, { 1, 2, 3, 4, 5, 6 })
	local b <close> = be:tensor("f32", { 3 }, { 10, 20, 30 })
	ok(a:type() == "f32", tag .. "type")
	near(a:shape(), { 2, 3 }, tag .. "shape")
	near(a:get(), { 1, 2, 3, 4, 5, 6 }, tag .. "round trip")

	local g <close> = be:graph()
	local s = g:add(a, b)
	local m = g:mul(a, b)
	local k = g:scale(a, 0.5)
	g:compute(s, m, k)
	near(s:get(), { 11, 22, 33, 14, 25, 36 }, tag .. "add broadcast")
	near(m:get(), { 10, 40, 90, 40, 100, 180 }, tag .. "mul broadcast")
	near(k:get(), { 0.5, 1, 1.5, 2, 2.5, 3 }, tag .. "scale")

	-- (2,3) @ (3,2) = (2,2)
	local y <close> = be:tensor("f32", { 3, 2 }, { 1, 0, 0, 1, 1, 1 })
	local p = g:matmul(a, y)
	g:compute(p)
	near(p:shape(), { 2, 2 }, tag .. "matmul shape")
	near(p:get(), { 4, 5, 10, 11 }, tag .. "matmul")

	local h <close> = be:tensor("f16", { 4 }, { 1, 2, 3, 4 })
	near(h:get(), { 1, 2, 3, 4 }, tag .. "f16 round trip")

	ok(not pcall(g.add, g, a, y), tag .. "bad broadcast is an error")
	ok(not pcall(p.get, g:add(p, p)), tag .. "uncomputed node is an error")
end

-- A graph that grows after a compute must plan its memory again.
for _, d in ipairs(devs) do
	local be <close> = assert(lgml.backend(d.name))
	local g <close> = be:graph()
	local a <close> = be:tensor("f32", { 64, 64 })
	-- 1/8 keeps values steady, so f16 math on the GPU does not overflow.
	a:fill("const", 1 / 8, 0)
	local x = g:add(a, a)
	for _ = 1, 5 do
		g:compute(x)
		x = g:matmul(g:mul(x, a), a)
	end
	g:compute(x)
	ok(math.abs(x:get()[1] - 0.25) < 1e-3, d.name .. ": graph grows between computes")
end

-- Close the backend first; tensors must stay usable.
do
	local be = assert(lgml.backend("CPU"))
	local a = be:tensor("f32", { 2 }, { 7, 8 })
	be:close()
	near(a:get(), { 7, 8 }, "leaf outlives closed backend")
	ok(not pcall(be.tensor, be, "f32", { 1 }), "closed backend is an error")
	a:close()
	ok(not pcall(a.get, a), "closed tensor is an error")
end

print("1.." .. n)
