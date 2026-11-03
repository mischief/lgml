-- Merge luacov stats files into one. File names that differ only in
-- "." and ".." parts, such as test/../lua/lgml/x.lua and lua/lgml/x.lua,
-- count as one file.
-- Usage: lua5.4 scripts/luacov_merge.lua out in...
local stats = require("luacov.stats")

local function clean(path)
	local abs = path:sub(1, 1) == "/"
	local out = {}
	for part in path:gmatch("[^/]+") do
		if part == ".." and #out > 0 and out[#out] ~= ".." then
			out[#out] = nil
		elseif part ~= "." then
			out[#out + 1] = part
		end
	end
	return (abs and "/" or "") .. table.concat(out, "/")
end

local out = assert(arg[1], "usage: luacov_merge.lua out in...")
local merged = {}
for i = 2, #arg do
	for name, d in pairs(stats.load(arg[i]) or {}) do
		local key = clean(name)
		local m = merged[key]
		if not m then
			merged[key] = d
		else
			for line, hits in pairs(d) do
				if type(line) == "number" then
					m[line] = (m[line] or 0) + hits
				end
			end
			m.max = math.max(m.max, d.max)
			m.max_hits = math.max(m.max_hits or 0, d.max_hits or 0)
		end
	end
end
stats.save(out, merged)
