-- Check a gguf from lgml-convert against its Hugging Face directory:
-- float tensors match the source bytes, quantized tensors give their
-- error, and the gguf tokenizer encodes as tokenizer.json does.
-- usage: lua5.4 scripts/convert_check.lua hf-dir file.gguf [every]
-- With every > 1, only every nth tensor is compared.
local lgml = require("lgml")
local json = require("lgml.json")
local convert = require("lgml.convert")

local dir, path, every = arg[1], arg[2], tonumber(arg[3] or 1)
assert(dir and path, "usage: convert_check.lua hf-dir file.gguf [every]")
local f = assert(io.open(dir .. "/config.json"))
local cfg = assert(json.decode(f:read("a")))
f:close()
local plan = convert.plan(dir, cfg, convert.arch[cfg.model_type])
local info = assert(lgml.gguf(path))
local byname = {}
for _, t in ipairs(info.tensors) do
	byname[t.name] = t
end

local files = {}
local function read(p, off, size)
	local h = files[p]
	if not h then
		h = assert(io.open(p, "rb"))
		files[p] = h
	end
	assert(h:seek("set", off))
	local s = h:read(size)
	assert(s and #s == size, p .. ": short read")
	return s
end

local stats, bad, checked = {}, 0, 0
local t0 = lgml.clock()
for i, e in ipairs(plan) do
	local t = assert(byname[e.name], "missing " .. e.name)
	assert(table.concat(t.shape, "x") == table.concat(e.shape, "x"), e.name .. ": shape")
	if (i - 1) % every == 0 then
		local n = e.shape[#e.shape]
		local in_row, out_row = lgml.row_size(e.src_type, n), lgml.row_size(t.type, n)
		local rows = math.max(1, (64 << 20) // in_row)
		local out_off = info.data_offset + t.offset
		local st = stats[t.type] or { se = 0, sr = 0, n = 0, worst = 0, worst_name = "", max = 0 }
		stats[t.type] = st
		local tse, tsr, tn = 0, 0, 0
		for _, p in ipairs(e.parts) do
			for off = 0, p.size - 1, rows * in_row do
				local size = math.min(rows * in_row, p.size - off)
				local s = read(p.path, p.offset + off, size)
				local g = read(path, out_off, size // in_row * out_row)
				out_off = out_off + #g
				if t.type == e.src_type then
					if s ~= g then
						bad = bad + 1
						print("mismatch " .. e.name)
					end
				else
					local ref = lgml.dequantize(e.src_type, s, n)
					local rmse, mx, rms = lgml.f32_error(ref, lgml.dequantize(t.type, g, n))
					local k = #ref // 4
					tse, tsr, tn = tse + rmse * rmse * k, tsr + rms * rms * k, tn + k
					st.max = math.max(st.max, mx)
				end
			end
		end
		if tn > 0 then
			st.se, st.sr, st.n = st.se + tse, st.sr + tsr, st.n + tn
			local rel = math.sqrt(tse / tsr)
			if rel > st.worst then
				st.worst, st.worst_name = rel, e.name
			end
		end
		st.count = (st.count or 0) + 1
		checked = checked + 1
	end
end
for _, h in pairs(files) do
	h:close()
end
print(("checked %d of %d tensors in %.0fs, %d byte mismatches"):format(checked, #plan,
    lgml.clock() - t0, bad))
local names = {}
for k in pairs(stats) do
	names[#names + 1] = k
end
table.sort(names)
for _, k in ipairs(names) do
	local st = stats[k]
	if st.n > 0 then
		print(("%-5s %3d tensors  rel rmse %.5f  worst %.5f (%s)  max abs %.4g"):format(k, st.count,
		    math.sqrt(st.se / st.sr), st.worst, st.worst_name, st.max))
	else
		print(("%-5s %3d tensors  exact"):format(k, st.count))
	end
end

-- The tokenizer, on plain text and on chat template text.
local tg <close> = assert(lgml.tokenizer(path))
local tj <close> = assert(lgml.tokenizer(dir .. "/tokenizer.json"))
local texts = {
	"Hello world! 1234567 numbers, don't stop. It's 3.14159.",
	"  leading spaces\n\n\ttabs and\r\nCRLF   trailing   ",
	"中文测试，日本語のテキスト、한국어 텍스트. Ünïcödé café naïve e\u{301}",
	"emoji 🎉🚀 and symbols ∑∫√ <>&\"' {\"json\": [1, 2, {\"a\": null}]}",
	"def f(x):\n    return x ** 2  # comment\n",
	"<role>SYSTEM</role>You are a helpful AI assistant.\ndetailed thinking off<|role_end|>" ..
	    "<role>HUMAN</role>Lily can run 12 kilometers per hour for 4 hours. How far?<|role_end|>" ..
	    "<role>ASSISTANT</role>",
	"<role>ASSISTANT</role>\n<think>\nplan\n</think>\n\nanswer<tool_call>\n{\"name\": \"f\", " ..
	    "\"arguments\": {}}\n</tool_call><|role_end|><role>OBSERVATION</role>\n<tool_response>\n" ..
	    "ok\n</tool_response><|role_end|><|mask|><|mask|><|endoftext|>",
}
local same = 0
for _, s in ipairs(texts) do
	local a, b = tg:encode(s), tj:encode(s)
	if table.concat(a, ",") == table.concat(b, ",") then
		same = same + 1
	else
		print(("tokenizer differs on %q"):format(s))
	end
end
print(("tokenizer: %d of %d texts encode the same (vocab %d / %d)"):format(same, #texts,
    tg:n_vocab(), tj:n_vocab()))
os.exit(bad == 0 and same == #texts and 0 or 1)
