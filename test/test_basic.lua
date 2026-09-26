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

ok(lgml.sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" and
    lgml.sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" and
    lgml.sha256(string.rep("a", 1000000)) ==
    "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "sha256 digests")
-- Lengths at the edges of the padding block.
local edges = {
	[55] = "d5e285683cd4efc02d021a5c62014694958901005d6f71e89e0989fac77e4072",
	[56] = "04c26261370ee7541549d16dee320c723e3fd14671e66a099afe0a377c16888e",
	[64] = "7ce100971f64e7001e8fe5a51973ecdfe1ced42befe7ee8d5fd6219506b5393c",
	[119] = "000b48d4edf0fa7bee3c6236ecd2785baa5db4eeb8bb54341b029e0d9fa5fb0c",
}
local good = true
for len, want in pairs(edges) do
	good = good and lgml.sha256(string.rep("x", len)) == want
end
ok(good, "sha256 at block edges")

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

-- CPU threads: a default from the machine or LGML_THREADS, per backend.
do
	local def = lgml.threads()
	ok(math.type(def) == "integer" and def >= 1 and def <= 1024, "default thread count")
	local env = tonumber(os.getenv("LGML_THREADS") or "")
	ok(not env or def == env, "LGML_THREADS sets the default")
	local be <close> = assert(lgml.backend("CPU"))
	ok(be:threads() == def, "a CPU backend gets the default")
	local b3 <close> = assert(lgml.backend("CPU", { threads = 3 }))
	ok(b3:threads() == 3 and b3:threads(2) == 2 and b3:threads() == 2, "threads option and set")
	local a <close> = b3:tensor("f32", { 2 }, { 1, 2 })
	local g <close> = b3:graph()
	local s = g:add(a, a)
	g:compute(s)
	near(s:get(), { 2, 4 }, "compute after a thread change")
	ok(not pcall(lgml.backend, "CPU", { threads = 0 }) and not pcall(b3.threads, b3, 2000),
	    "a bad thread count is an error")
	for _, d in ipairs(devs) do
		if d.type == "gpu" then
			local gb <close> = assert(lgml.backend(d.name, { threads = 2 }))
			ok(gb:threads() == nil, d.name .. ": a GPU backend has no thread count")
		end
	end
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
