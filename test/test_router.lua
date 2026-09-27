-- Tests for lgml.server.router with fake engines. TAP output.
local router = require("lgml.server.router")
local fake = require("lgml.server.fake_engine")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

-- An engine that only transcribes.
local stt = { id = "stt" }
function stt.models() return { { id = "whisper", context_length = 448 } } end
function stt.props() return { n_ctx = 448, total_slots = 1 } end
function stt.transcribe() return { text = "from stt" } end

local a = fake.new({ model = "a", n_ctx = 111 })
local b = fake.new({ model = "b", n_ctx = 222 })
local r = router.new({ stt, a, b })

local ids = {}
for _, m in ipairs(r:models()) do
	ids[#ids + 1] = m.id
end
ok(table.concat(ids, ",") == "whisper,a,b", "models from every engine")
ok(r:pick("b", "chat") == b, "chat by model id")
local pe, pid, pwhy = r:pick("nope", "chat")
ok(pe == nil and pid == nil and pwhy == "unknown", "an unknown model picks no engine")
local _, nmsg, ncode = r:chat({ model = "nope", messages = { { role = "user", content = "hi" } } })
ok(ncode == 404 and nmsg:find("model nope not found", 1, true) and nmsg:find("a, b", 1, true),
    "chat with an unknown model gives 404 and names the chat models")
local _, cmsg, ccode = r:chat({ model = "whisper", messages = { { role = "user", content = "hi" } } })
ok(ccode == 400 and cmsg:find("whisper does not serve chat", 1, true),
    "chat with a transcription model gives 400")
ok(r:pick("", "chat") == a, "an empty model goes to first chat engine")
ok(r:pick(nil, "chat") == a, "no model goes to first chat engine")
ok(r:pick("whisper", "transcribe") == stt and r:pick(nil, "chat") ~= stt,
    "chat skips transcribe-only engines")
ok(r:props().n_ctx == 111, "props from first chat engine")
ok(r:transcribe({ model = "whisper" }, "").text == "from stt", "transcribe routes")
local _, lmsg, lcode = r:load("a")
local _, umsg, ucode = r:unload("nope")
ok(lcode == 400 and ucode == 400 and lmsg == require("lgml.server.engine").NO_LOAD and umsg == lmsg,
    "load and unload give 400 when no model loads on demand")
ok(r:model_for(nil, "chat") == "a" and r:model_for("nope", "chat") == nil and
    r:model_for("b", "chat") == "b" and r:model_for("whisper", "chat") == nil,
    "model_for gives the model that serves chat, and nil for an unknown or wrong model")
ok(r:model_for(nil, "transcribe") == "whisper" and router.new({ stt }):model_for(nil, "chat") == nil,
    "model_for gives nil when no model serves the method")
-- LoRA lists, as llama-server reads them.
do
	local api = require("lgml.server.engine")
	local s = api.lora_scales({ { id = 1, scale = 0.5 } }, 3)
	ok(s and s[1] == 0 and s[2] == 0.5 and s[3] == 0, "adapters that a list does not name get 0")
	ok(not api.lora_scales({ { id = 3, scale = 1 } }, 3) and not api.lora_scales({ { id = 0 } }, 3) and
	    not api.lora_scales("x", 3) and api.lora_scales({}, 0), "lora_scales checks ids and scales")
	local req = { messages = {} }
	local w = api.with_lora(req, { 0.25, 1 })
	ok(w ~= req and req.lora == nil and w.lora[1].id == 0 and w.lora[1].scale == 0.25 and w.lora[2].id == 1,
	    "with_lora adds the global scales to a copy")
	local own = { lora = {} }
	ok(api.with_lora(own, { 1 }) == own and api.with_lora(req, nil) == req, "with_lora keeps a request lora")
	local set
	local la = fake.new({ model = "la" })
	function la.lora_adapters(_, list) set = list return true end
	local rl = router.new({ stt, la })
	ok(rl:lora_adapters({ { id = 0, scale = 1 } }) and set and set[1].scale == 1,
	    "the router sets the adapters of the first chat engine")
	local _, msg, code = router.new({ stt, a }):lora_adapters({ { id = 0, scale = 1 } })
	ok(code == 400 and msg:find("no adapters") and router.new({ stt, a }):lora_adapters({}),
	    "an engine without adapters takes only an empty list")
end

-- An engine that only makes images.
local img = {}
function img.models() return { { id = "zimage", context_length = 0 } } end
function img.props() return { n_ctx = 0, total_slots = 1 } end
function img.generate(_, req) return { seeds = { req.seed } } end
local r2 = router.new({ stt, img })
ok(r2:pick(nil, "generate") == img and r2:pick("whisper", "generate") == nil,
    "generate skips engines without generate")
ok(r2:generate({ model = "zimage", seed = 9 }).seeds[1] == 9, "generate routes")
ok(select(3, router.new({ stt }):generate({ prompt = "x" })) == 400,
    "no image engine gives 400")

-- Chat with images goes to a model with vision.
local function vis(id, vision)
	local e = {}
	function e.models() return { { id = id, context_length = 8, vision = vision } } end
	function e.props() return { n_ctx = 8, total_slots = 1 } end
	function e.chat() return id end
	return e
end
local img_msgs = { { role = "user", content = {
	{ type = "image_url", image_url = { url = "data:image/png;base64,AA==" } },
	{ type = "text", text = "what is this?" } } } }
local r3 = router.new({ vis("text", false), vis("eyes", true) })
ok(r3:chat({ messages = img_msgs }) == "eyes", "images with no model go to a vision model")
local _, _, icode = r3:chat({ model = "nope", messages = img_msgs })
ok(icode == 404, "images with an unknown model give 404")
ok(r3:chat({ messages = { { role = "user", content = "hi" } } }) == "text", "text stays on the first chat model")
ok(r3:model_for(nil, "chat", true) == "eyes" and r3:model_for("text", "chat", true) == nil,
    "model_for follows images to a vision model")
local _, msg, code = r3:chat({ model = "text", messages = img_msgs })
ok(code == 400 and msg:find("models with vision: eyes", 1, true), "a named model without vision names the vision models")
_, msg, code = router.new({ vis("text", false) }):chat({ messages = img_msgs })
ok(code == 400 and msg:find("no model has an mmproj", 1, true), "no vision model gives 400")

-- Decisions with images go to a model with vision that can decide.
local function dvis(id, vision)
	local e = vis(id, vision)
	function e.decide(_, req) return { model = id, images = req.images and #req.images or 0 } end
	return e
end
local chat_eyes = vis("chateyes", true)
local r6 = router.new({ chat_eyes, dvis("text", false), dvis("eyes", true) })
local imgs = { { bytes = "\137PNG", mime = "image/png" } }
ok(r6:decide({ images = imgs }).model == "eyes", "decide images with no model go to a vision decide model")
ok(r6:decide({ model = "jevify-latest", images = imgs }).model == "eyes",
    "decide images for jevify-latest go to a vision decide model")
ok(r6:decide({ model = "jevify-latest" }).model == "text" and r6:decide({}).model == "text",
    "text decisions keep the first decide model")
ok(r6:decide({ model = "eyes", images = imgs }).images == 1, "a named vision model gets the images")
_, msg, code = r6:decide({ model = "text", images = imgs })
ok(code == 400 and msg:find("not supported by text; models with vision: eyes$"),
    "a named decide model without vision names the vision decide models: " .. tostring(msg))
_, msg, code = router.new({ chat_eyes, dvis("text", false) }):decide({ images = imgs })
ok(code == 400 and msg:find("no model has an mmproj", 1, true),
    "a chat-only vision model does not serve decisions with images")
ok(r6:decide({ images = {} }).model == "text", "an empty image list is text")

-- Modes of config specs, without loading.
local detect = require("lgml.detect")
local function modes(spec)
	return table.concat(detect.modes(spec), ",")
end
local wdir = os.tmpname()
os.remove(wdir)
os.execute("mkdir " .. wdir)
local cf = assert(io.open(wdir .. "/config.json", "w"))
cf:write('{"model_type": "whisper"}')
cf:close()
ok(modes({ model = "x.gguf" }) == "completion,decide", "gguf is a chat model")
ok(modes({ model = "x.gguf", mmproj = "p.gguf" }) == "completion,multimodal,decide", "mmproj adds multimodal")
ok(modes({ model = wdir }) == "transcription" and modes({ model = "d", stt = true }) == "transcription",
    "whisper directory or stt is transcription")
ok(modes({ model = "d", engine = "lgml.tts.orpheus" }) == "speech", "tts engine is speech")
ok(modes({ model = "d", engine = "lgml.image.engine" }) == "image_generation", "image engine")
ok(modes({ model = "d", engine = "lgml.embed_engine" }) == "embedding", "embedding engine")
ok(modes({ model = "d", engine = "lgml.diffusion_engine" }) == "completion", "diffusion chat has no decide")
ok(modes({ model = "d", engine = "my.engine", modes = { "speech", "completion" } }) == "completion,speech",
    "modes in the spec win")
ok(not pcall(detect.modes, { model = "d", engine = "my.engine" }), "unknown engine needs modes")
ok(not pcall(detect.modes, { model = "d", modes = { "chat" } }), "unknown mode name")
os.remove(wdir .. "/config.json")
os.remove(wdir)

-- Capabilities per mode, and picks from modes alone.
local caps = {}
for _, m in ipairs(r:models()) do
	caps[m.id] = table.concat(m.capabilities, ",")
end
ok(caps.whisper == "transcription", "transcribe-only engine has transcription")
ok(caps.a == "completion,decide,transcription,speech,image_generation,embedding",
    "fake engine has every mode but multimodal")
local r4 = router.new({ vis("text", false), vis("eyes", true) })
local c4 = r4:models()
ok(table.concat(c4[1].capabilities, ",") == "completion" and
    table.concat(c4[2].capabilities, ",") == "completion,multimodal", "vision adds multimodal")
-- A model that is not loaded: its modes come from modes(), and models()
-- has no load to wait for.
local function cold(id, modes)
	local e = { loads = 0 }
	function e.models() return { { id = id, capabilities = modes } } end
	function e.modes() return modes end
	function e.props() return { n_ctx = 0, total_slots = 1 } end
	function e.ready() error("ready must not run") end
	function e.speak(self) self.loads = self.loads + 1 return { wav = "RIFF" } end
	function e.chat(self) self.loads = self.loads + 1 return true end
	return e
end
local tts, cv = cold("tts", { "speech" }), cold("cv", { "completion", "multimodal", "decide" })
local r5 = router.new({ tts, cv })
ok(r5:pick(nil, "speak") == tts and r5:pick("cv", "speak") == nil, "speech goes by modes")
ok(r5:pick(nil, "chat") == cv and r5:pick("tts", "decide") == nil, "chat and decide go by modes")
ok(r5:chat({ messages = img_msgs }) == true and cv.loads == 1, "images go to a multimodal model by modes")
ok(tts.loads == 0 and select(3, router.new({ tts }):chat({ messages = {} })) == 400,
    "no chat model gives 400")

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
