-- Print the metadata and tensor summary of a gguf file, through
-- lgml.gguf (no weights are read).
-- Usage: lua5.4 scripts/gguf_info.lua file.gguf [pattern]
-- With pattern (a Lua pattern), it also lists the tensors whose names
-- match, with type, shape and size. Arrays show their length and first
-- values. For a split model, give the first shard.
local root = ((arg and arg[0] or ""):match("^(.*)/[^/]*$") or ".") .. "/.."
package.path = root .. "/lua/?.lua;" .. root .. "/lua/?/init.lua;" .. package.path
package.cpath = root .. "/" .. (os.getenv("LGML_BUILD") or "build") .. "/?.so;" .. package.cpath
local lgml = require("lgml")

local path, pattern = arg[1], arg[2]
if not path then
	io.stderr:write("usage: lua5.4 scripts/gguf_info.lua file.gguf [pattern]\n")
	os.exit(2)
end
local g, err = lgml.gguf(path)
if not g then
	io.stderr:write(tostring(err), "\n")
	os.exit(1)
end

local function gib(n)
	return ("%.2f GiB"):format(n / 1073741824)
end

-- Long strings, such as the chat template, show their length and start.
local function show(v)
	if type(v) == "string" and (#v > 120 or v:find("\n")) then
		return ("(%d bytes) %q..."):format(#v, v:sub(1, 60):gsub("\n", " "))
	elseif type(v) ~= "table" then
		return tostring(v)
	end
	local head = {}
	for i = 1, math.min(#v, 4) do
		head[i] = type(v[i]) == "string" and ("%q"):format(v[i]) or tostring(v[i])
	end
	return ("[%d] %s%s"):format(#v, table.concat(head, ", "), #v > 4 and ", ..." or "")
end

print(("gguf v%d, %d tensors, data at %d"):format(g.version, #g.tensors, g.data_offset))
local keys = {}
for k in pairs(g.meta) do
	keys[#keys + 1] = k
end
table.sort(keys)
for _, k in ipairs(keys) do
	print(("  %s = %s"):format(k, show(g.meta[k])))
end

local by, total = {}, 0
for _, t in ipairs(g.tensors) do
	local e = by[t.type] or { type = t.type, n = 0, bytes = 0 }
	by[t.type] = e
	e.n, e.bytes = e.n + 1, e.bytes + t.size
	total = total + t.size
end
local list = {}
for _, e in pairs(by) do
	list[#list + 1] = e
end
table.sort(list, function(a, b) return a.bytes > b.bytes end)
print(("tensors by type (%s in all):"):format(gib(total)))
for _, e in ipairs(list) do
	print(("  %-8s %5d  %s"):format(e.type, e.n, gib(e.bytes)))
end

if pattern then
	for _, t in ipairs(g.tensors) do
		if t.name:find(pattern) then
			print(("  %-48s %-8s %-20s %s"):format(t.name, t.type, table.concat(t.shape, "x"), gib(t.size)))
		end
	end
end
