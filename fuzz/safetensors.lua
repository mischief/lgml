-- safetensors files: the index, then a load to the CPU backend.
local lgml = require("lgml")

local be = assert(lgml.backend("CPU"))

return function(s)
	local path = fuzz.file(s)
	if pcall(lgml.safetensors, path) then
		local ok, w = pcall(be.load_safetensors, be, path, { f32_1d = #s % 2 == 1 })
		if ok then
			w:close()
		end
	end
end
