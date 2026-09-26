-- Gemma 4 image input on a real model against llama.cpp. Output is TAP.
-- LGML_GEMMA4_VISION: a directory with a gemma4 gguf and mmproj*.gguf.
-- LGML_VISION_IMAGE: a PNG or JPEG. LGML_MTMD_EMBD: a scripts/mtmd_embd.c
-- binary, for the encoder output. LGML_MTMD_CLI: llama-mtmd-cli, for a
-- greedy answer. The model runs on LGML_DEVICE (default CPU); run this
-- file with lua5.4 directly for Vulkan.
local lgml = require("lgml")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function skip(name)
	n = n + 1
	print("ok " .. n .. " # SKIP " .. name)
end

local function done()
	print("1.." .. n)
	os.exit(failed == 0 and 0 or 1)
end

local function env(k)
	local v = os.getenv(k)
	return v ~= "" and v or nil
end

local dir, image = env("LGML_GEMMA4_VISION"), env("LGML_VISION_IMAGE")
if not dir or not image then
	skip("LGML_GEMMA4_VISION or LGML_VISION_IMAGE not set")
	done()
end

local function sh(s)
	return "'" .. s:gsub("'", "'\\''") .. "'"
end

local model, mmproj
local ls = assert(io.popen("ls -1 " .. sh(dir)))
for name in ls:lines() do
	if name:match("%.gguf$") then
		if name:lower():match("^mmproj") then
			mmproj = dir .. "/" .. name
		else
			model = dir .. "/" .. name
		end
	end
end
ls:close()
assert(model and mmproj, dir .. ": need a model gguf and an mmproj gguf")
local f = assert(io.open(image, "rb"))
local bytes = f:read("a")
f:close()
local dev = env("LGML_DEVICE") or "CPU"
local tag = "gemma4 vision " .. dev .. ": "

-- The relative rms error of embeddings a against ref, and the worst
-- cosine of a row.
local function compare(a, ref, rows, E)
	local se, sr, worst = 0, 0, 1
	for r = 0, rows - 1 do
		local dot, a2, b2 = 0, 0, 0
		for i = 1, E do
			local off = (r * E + i - 1) * 4 + 1
			local x, y = string.unpack("<f", a, off), string.unpack("<f", ref, off)
			se, sr = se + (x - y) ^ 2, sr + y * y
			dot, a2, b2 = dot + x * y, a2 + x * x, b2 + y * y
		end
		worst = math.min(worst, dot / math.sqrt(a2 * b2))
	end
	return math.sqrt(se / sr), worst
end

-- Encoder output against llama.cpp, on the pixels that llama.cpp
-- decoded: a few image tokens react strongly to one-level pixel
-- changes, and JPEG decoders differ by that much.
local mtmd = env("LGML_MTMD_EMBD")
if not mtmd then
	skip("LGML_MTMD_EMBD not set")
else
	local be <close> = assert(lgml.backend(dev))
	local enc = require("lgml.vision").load(be, mmproj)
	local out, rgb = os.tmpname(), os.tmpname()
	local p = assert(io.popen(("MTMD_NO_FA=1 nice %s %s %s %s %s %s"):format(sh(mtmd), sh(model),
	    sh(mmproj), sh(image), sh(out), sh(rgb))))
	local line = p:read("l") or ""
	p:close()
	local rn, re, w, h = line:match("^(%d+) (%d+) (%d+) (%d+)")
	w, h = tonumber(w), tonumber(h)
	local ref = assert(io.open(out, "rb")):read("a")
	local raw = assert(io.open(rgb, "rb")):read("a")
	os.remove(out)
	os.remove(rgb)
	local planes = {}
	for k = 1, 3 do
		local v = {}
		for i = 0, w * h - 1 do
			v[i + 1] = string.pack("<f", raw:byte(i * 3 + k) / 255 * 2 - 1)
		end
		planes[k] = table.concat(v)
	end
	local t <close> = be:tensor("f32", { 1, 3, h, w }, table.concat(planes))
	local same = lgml.image_from(t)
	local t0 = lgml.clock()
	local buf, count = assert(enc:encode(same))
	local secs = lgml.clock() - t0
	local E = enc.hp.n_out
	ok(tonumber(rn) == count and tonumber(re) == E and #ref == buf:len(),
	    ("%s%d image tokens as llama.cpp (%s)"):format(tag, count, line))
	local rel, worst = compare(buf:bytes(), ref, count, E)
	io.stderr:write(("%sencode %.3f s, relative rms error %.3g, worst row cosine %.5f\n"):format(
	    tag, secs, rel, worst))
	ok(rel < 0.02 and worst > 0.99, ("%sembeddings match llama.cpp (rel rms %.3g, cosine %.5f)")
	    :format(tag, rel, worst))
	local own = assert(enc:encode(bytes))
	rel, worst = compare(own:bytes(), ref, count, E)
	io.stderr:write(("%swith our own decode: relative rms error %.3g, worst row cosine %.5f\n")
	    :format(tag, rel, worst))
	own:close()
	buf:close()
end

-- A greedy answer through the engine against llama-mtmd-cli. The CLI
-- template turns thinking on, so the engine does too.
local cli = env("LGML_MTMD_CLI")
if not cli then
	skip("LGML_MTMD_CLI not set")
	done()
end
local uv = require("luv")
local base64 = require("lgml.base64")
local question = "Describe this image in one sentence."
local N = 48
local p = assert(io.popen(("nice %s -m %s --mmproj %s --image %s -p %s --temp 0 -n %d -ngl 0 " ..
    "--no-mmproj-offload -c 4096 --jinja 2>/dev/null"):format(sh(cli), sh(model), sh(mmproj),
    sh(image), sh(question), N)))
local want = p:read("a")
p:close()
want = want:gsub("^%s*<|channel>thought\n", ""):gsub("%s+$", "")

local e = require("lgml.engine").new({ model = model, mmproj = mmproj, backend = dev, n_ctx = 4096 })
local mime = bytes:sub(1, 4) == "\137PNG" and "image/png" or "image/jpeg"
local got, usage, timings = {}, nil, nil
local sink = {}
function sink.reasoning(_, t)
	got[#got + 1] = t
	return true
end
sink.content = sink.reasoning
function sink.tool_call()
	return 0
end
sink.tool_arguments = sink.reasoning
function sink.finish(_, _, u, tm)
	usage, timings = u, tm
end
function sink.cancelled()
	return false
end
local req = {
	messages = { { role = "user", content = {
		{ type = "image_url", image_url = { url = "data:" .. mime .. ";base64," .. base64.encode(bytes) } },
		{ type = "text", text = question } } } },
	max_tokens = N, temperature = 0, chat_template_kwargs = { enable_thinking = true },
}
local res, err
coroutine.wrap(function()
	res, err = e:chat(req, sink)
end)()
while res == nil and err == nil do
	uv.run("once")
end
local text = table.concat(got):gsub("%s+$", "")
io.stderr:write(("%sours %q\n%sllama %q\n"):format(tag, text, tag, want))
ok(res == true, tag .. "chat runs " .. tostring(err))
-- The parser drops the channel tags; compare the words.
local function words(s)
	return (s:gsub("<[^>]*>", " "):gsub("%s+", " "):gsub("^ ", ""):gsub(" $", ""))
end
ok(#want > 0 and words(text) == words(want), tag .. "greedy answer matches llama-mtmd-cli")
if timings then
	io.stderr:write(("%sprompt %d tokens in %.0f ms (%.1f t/s, image encode included), " ..
	    "%d tokens at %.1f t/s\n"):format(tag, usage.prompt_tokens, timings.prompt_ms,
	    timings.prompt_per_second, timings.predicted_n, timings.predicted_per_second))
end
done()
