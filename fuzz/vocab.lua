-- vocab.json, then added_tokens.json after the first NUL byte.
local lgml = require("lgml")

return function(s)
	local a, b = s:match("^(.-)%z(.*)$")
	local ok, v
	if a then
		ok, v = pcall(lgml.vocab, fuzz.file(a, 1), fuzz.file(b, 2))
	else
		ok, v = pcall(lgml.vocab, fuzz.file(s, 1))
	end
	if not ok then
		return
	end
	local ids = {}
	for id = 0, math.min(v:n(), 4096) - 1 do
		if v:piece(id) then
			ids[#ids + 1] = id
		end
		v:special(id)
	end
	v:decode(ids, true)
	v:id("a")
	v:close()
end
