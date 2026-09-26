-- WAV files: decode, write back, resample and mel.
local lgml = require("lgml")

local be = assert(lgml.backend("CPU"))
local mel = be:tensor("f32", { 16, 80 })

return function(s)
	local a = lgml.audio(s)
	if not a then
		return
	end
	local back = assert(lgml.audio(a:wav(1, "f32")), "own WAV does not decode")
	assert(back:len() == a:len(), "WAV round trip changes length")
	if a:len() <= 1 << 16 then
		-- Too few samples to resample is an error.
		local ok, r = pcall(a.resample, a, 16000)
		if ok then
			r:mel(mel, #s % (r:len() + 1))
			r:close()
		end
	end
	back:close()
	a:close()
end
