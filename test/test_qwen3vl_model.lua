-- Clef-Flash image input on the real model against llama.cpp. Output is
-- TAP. LGML_CLEF_VISION: a directory with a Clef gguf and mmproj*.gguf.
-- LGML_VISION_IMAGE: a PNG. LGML_MTMD_HIDDEN: a scripts/mtmd_hidden.c
-- binary; LGML_CLEF_QWEN35: the Clef backbone as a qwen35 gguf, which
-- llama.cpp runs with images. LGML_MTMD_CLI: llama-mtmd-cli, for a greedy
-- answer. The model runs on LGML_DEVICE (default CPU).
local lgml = require("lgml")
local uv = require("luv")
local json = require("lgml.json")
local base64 = require("lgml.base64")
local systemone = require("lgml.systemone")
local head = require("lgml.decide_head")

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

local dir, image = env("LGML_CLEF_VISION"), env("LGML_VISION_IMAGE")
if not dir or not image then
	skip("LGML_CLEF_VISION or LGML_VISION_IMAGE not set")
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
local bytes = assert(io.open(image, "rb")):read("a")
local dev = env("LGML_DEVICE") or "CPU"
local tag = "clef vision " .. dev .. ": "

-- The relative rms error of rows of a against ref, and the worst cosine.
local function compare(a, ref, rows, E)
	local se, sr, worst = 0, 0, 1
	for _, r in ipairs(rows) do
		local dot, a2, b2 = 0, 0, 0
		for i = 1, E do
			local off = ((r - 1) * E + i - 1) * 4 + 1
			local x, y = string.unpack("<f", a, off), string.unpack("<f", ref, off)
			se, sr = se + (x - y) ^ 2, sr + y * y
			dot, a2, b2 = dot + x * y, a2 + x * x, b2 + y * y
		end
		worst = math.min(worst, dot / math.sqrt(a2 * b2))
	end
	return math.sqrt(se / sr), worst
end

local function run(fn)
	local res
	coroutine.wrap(function()
		res = table.pack(pcall(fn))
	end)()
	while not res do
		uv.run("once")
	end
	assert(res[1], res[2])
	return table.unpack(res, 2, res.n)
end

local e = require("lgml.engine").new({ model = model, mmproj = mmproj, backend = dev, n_ctx = 8192 })
local body = {
	state = "Look at the attached photo.",
	questions = setmetatable({
		animal = { type = "choice", instructions = "Which animal is in the photo?",
			criteria = { cat = "a cat", dog = "a dog", bird = "a bird", none = "no animal" } },
		is_cat = { type = "noul", instructions = "Is there a cat in the photo?" },
	}, { __jsontype = "object", __order = { "animal", "is_cat" } }),
	images = { base64.encode(bytes) },
}
local req = assert(systemone.check(json.encode(body)))
local res = run(function()
	return e:decide(req, {})
end)
ok(res and res.answers.animal, tag .. "decide runs with an image")
io.stderr:write(("%sanimal %s, is_cat %.4f\n"):format(tag, json.encode(res.answers.animal.probabilities),
    res.answers.is_cat.yes))

-- The Clef prompt and hidden states, ours and llama.cpp's.
local tool, ref_model = env("LGML_MTMD_HIDDEN"), env("LGML_CLEF_QWEN35")
if not tool or not ref_model then
	skip("LGML_MTMD_HIDDEN or LGML_CLEF_QWEN35 not set")
else
	local rd, buf, small
	run(function()
		e.images = {}
		local keys = assert(e:prepare_images(req.images))
		rd = head.render_clef(e.tk, req, nil, head.clef_media(e, keys))
		local sink = { computed = 0, cancelled = function()
			return false
		end }
		buf = assert(head.clef_hidden(e, rd.ids, sink))
		-- Passes of 32 tokens split the image.
		local ub = e.ubatch
		e.ubatch = 32
		small = assert(head.clef_hidden(e, rd.ids, sink))
		e.ubatch = ub
		e:drop_images()
	end)
	local words, rows, img = {}, {}, {}
	local i = 1
	while i <= #rd.ids do
		local key = e.image_key(rd.ids[i])
		if key then
			words[#words + 1] = "img:" .. image
			while e.image_key(rd.ids[i + 1] or 0) == key do
				img[#img + 1] = i
				i = i + 1
			end
			img[#img + 1] = i
		else
			words[#words + 1] = tostring(rd.ids[i])
			rows[#rows + 1] = i
		end
		i = i + 1
	end
	local tmp = os.tmpname()
	assert(io.open(tmp .. ".txt", "w")):write(table.concat(words, " "), "\n")
	local p = assert(io.popen(("nice %s %s %s %s %s.f32 %s.embd"):format(sh(tool), sh(ref_model), sh(mmproj),
	    sh(tmp .. ".txt"), sh(tmp), sh(tmp))))
	local out = p:read("a")
	p:close()
	local ref = assert(io.open(tmp .. ".f32", "rb")):read("a")
	local refe = assert(io.open(tmp .. ".embd", "rb")):read("a")
	for _, s in ipairs({ "", ".txt", ".f32", ".embd" }) do
		os.remove(tmp .. s)
	end
	local E = e.model.hp.n_embd
	ok(#ref == buf:len(), ("%s%d prompt rows as llama.cpp (%s)"):format(tag, #rd.ids, out:gsub("\n", " ")))
	-- The encoder alone.
	local enc = assert(e.vision:encode(bytes))
	local all = {}
	for r = 1, #img do
		all[r] = r
	end
	local rel, worst = compare(enc:bytes(), refe, all, E)
	io.stderr:write(("%simage embeddings: relative rms error %.3g, worst row cosine %.5f\n"):format(tag, rel, worst))
	ok(rel < 0.05 and worst > 0.95, ("%simage embeddings match llama.cpp (rel rms %.3g)"):format(tag, rel))
	rel, worst = compare(buf:bytes(), ref, rows, E)
	local irel, iworst = compare(buf:bytes(), ref, img, E)
	io.stderr:write(("%shidden states: text rows rel %.3g cos %.5f; image rows rel %.3g cos %.5f\n"):format(tag,
	    rel, worst, irel, iworst))
	ok(rel < 0.05 and worst > 0.98, ("%stext hidden states match llama.cpp (rel rms %.3g)"):format(tag, rel))
	-- Other pass sizes take other kernels; llama.cpp with and without
	-- flash attention differs by about 3% here.
	local srel = compare(small:bytes(), buf:bytes(), rows, E)
	ok(srel < 0.05, ("%ssmall passes give close hidden states (rel rms %.3g)"):format(tag, srel))
	-- The head on both hidden states.
	local rbuf = lgml.hostbuf(#ref)
	rbuf:set(0, ref)
	local ours = head.clef_logits(e.be, e.decide_head, rd, buf, E)
	local theirs = head.clef_logits(e.be, e.decide_head, rd, rbuf, E)
	local function softmax(s)
		local m, z, q = -math.huge, 0, {}
		for _, v in ipairs(s) do
			m = math.max(m, v)
		end
		for j, v in ipairs(s) do
			q[j] = math.exp(v - m)
			z = z + q[j]
		end
		for j = 1, #q do
			q[j] = q[j] / z
		end
		return q
	end
	local worstp = 0
	for qi, q in ipairs(rd.questions) do
		local a, b = softmax(ours[qi]), softmax(theirs[qi])
		local s = {}
		for j, o in ipairs(q.options) do
			worstp = math.max(worstp, math.abs(a[j] - b[j]))
			s[#s + 1] = ("%s %.4f/%.4f"):format(o.key, a[j], b[j])
		end
		io.stderr:write(("%s%s lgml/llama.cpp: %s\n"):format(tag, q.id, table.concat(s, ", ")))
	end
	ok(worstp < 0.01, ("%sprobabilities match those from llama.cpp's hidden states (max diff %.4f)"):format(tag,
	    worstp))
end

-- A greedy answer through the engine against llama-mtmd-cli, with the
-- Clef backbone as qwen35. The CLI template turns thinking on.
local cli = env("LGML_MTMD_CLI")
if not cli or not ref_model then
	skip("LGML_MTMD_CLI or LGML_CLEF_QWEN35 not set")
	done()
end
local question = "Describe this image in one sentence."
local N = 32
local p = assert(io.popen(("nice %s -m %s --mmproj %s --image %s -p %s --temp 0 -n %d -ngl 0 " ..
    "--no-mmproj-offload -c 4096 --jinja 2>/dev/null"):format(sh(cli), sh(ref_model), sh(mmproj),
    sh(image), sh(question), N)))
local want = p:read("a"):gsub("%s+$", "")
p:close()
local got = {}
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
function sink.finish() end
function sink.cancelled()
	return false
end
local r, err = run(function()
	return e:chat({
		messages = { { role = "user", content = {
			{ type = "image_url", image_url = { url = "data:image/png;base64," .. base64.encode(bytes) } },
			{ type = "text", text = question } } } },
		max_tokens = N, temperature = 0, chat_template_kwargs = { enable_thinking = true },
	}, sink)
end)
local function words(s)
	return (s:gsub("<[^>]*>", " "):gsub("%s+", " "):gsub("^ ", ""):gsub(" $", ""))
end
local text = table.concat(got)
io.stderr:write(("%sours %q\n%sllama %q\n"):format(tag, text, tag, want))
ok(r == true, tag .. "chat runs " .. tostring(err))
ok(#want > 0 and words(text) == words(want), tag .. "greedy answer matches llama-mtmd-cli")
done()
