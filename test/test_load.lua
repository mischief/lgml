-- backend:load reads the same bytes through the ring as through stdio.
-- Output is TAP.
local lgml = require("lgml")
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local gguf = require("gguf")

local n = 0
local failed = false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

local I8 = 24

-- A 64 KiB block of pseudo-random bytes.
local seed = 12345
local parts = {}
for i = 1, 8192 do
	seed = (seed * 6364136223846793005 + 1442695040888963407) & 0x7fffffffffffffff
	parts[i] = string.pack("<i8", seed)
end
local block = table.concat(parts)

-- n bytes, different for each tag and each 64 KiB.
local function bytes(tag, len)
	local out, i = {}, 0
	while #out * 65536 < len do
		i = i + 1
		out[i] = string.pack("<I4I4", tag, i) .. block:sub(9)
	end
	return table.concat(out):sub(1, len)
end

local function t(name, len, typ)
	if typ == gguf.F16 then
		return { name, { len // 2 }, bytes(#name * 7919 + len, len), gguf.F16 }
	end
	return { name, { len }, bytes(#name * 7919 + len, len), I8 }
end

local MiB = 1 << 20
local kv = { { "general.architecture", gguf.STRING, "test" } }
-- Odd sizes; a tensor across read chunks; a large skipped tensor that
-- splits the reads and a small one read through; a 1-byte tensor at
-- the end of the file.
local first = {
	t("a", 5), t("b", 26, gguf.F16), t("big", 20 * MiB + 77), t("skip.big", 3 * MiB),
	t("c", 4097), t("skip.small", 100), t("host", 17 * MiB + 3), t("d", 33), t("e", 1),
}
local second = { t("f", 16 * MiB - 5), t("g", 9), t("skip.tail", 2 * MiB + 1), t("h", 64) }
local p1, p2 = "test_load-00001-of-00002.gguf", "test_load-00002-of-00002.gguf"
gguf.write(p1, kv, first)
gguf.write(p2, kv, second)

local skip = { ["skip.big"] = true, ["skip.small"] = true, ["skip.tail"] = true }

local function load(be, path, uring)
	local w = assert(be:load(path, { cpu = { host = true }, skip = skip, uring = uring }))
	local info = lgml.load_info()
	local got = {}
	for _, name in ipairs(w:names()) do
		got[name] = w:get(name):bytes()
	end
	w:close()
	return got, info
end

local function check(be, path, ts)
	local old, oi = load(be, path, false)
	local new, ni = load(be, path, true)
	path = be:name() .. " " .. path
	ok(oi.mode == "stdio", path .. ": uring false reads with stdio")
	print("# " .. path .. ": mode " .. ni.mode .. ", " .. ni.reads .. " reads, "
	    .. ni.bytes .. " bytes, " .. ni.dma .. " by the device")
	if be:name():find("Vulkan") and ni.mode ~= "stdio" then
		ok(ni.dma > 0, path .. ": the device copies from the read buffers")
	end
	local same, want = true, true
	for _, x in ipairs(ts) do
		local name = x[1]
		if skip[name] then
			same = same and new[name] == nil and old[name] == nil
		else
			same = same and new[name] == old[name]
			want = want and new[name] == x[3]
		end
	end
	ok(same, path .. ": ring and stdio load the same bytes")
	ok(want, path .. ": the bytes match the file")
end

-- A GPU copies its tensors straight from the read buffers.
local backends = { assert(lgml.backend("CPU", { cpu = true, threads = 2 })) }
local okv, vk = pcall(lgml.backend, "Vulkan0", { cpu = true, threads = 2 })
if okv and vk then
	backends[2] = vk
end
for _, be in ipairs(backends) do
	check(be, p1, first)
	check(be, p2, second)
	be:close()
end
os.remove(p1)
os.remove(p2)

print("1.." .. n)
if failed then
	os.exit(1)
end
