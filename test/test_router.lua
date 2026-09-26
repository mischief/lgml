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
ok(r:pick("nope", "chat") == a, "unknown model goes to first chat engine")
ok(r:pick(nil, "chat") == a, "no model goes to first chat engine")
ok(r:pick("whisper", "transcribe") == stt and r:pick(nil, "chat") ~= stt,
    "chat skips transcribe-only engines")
ok(r:props().n_ctx == 111, "props from first chat engine")
ok(r:transcribe({ model = "whisper" }, "").text == "from stt", "transcribe routes")
-- An engine that only makes images.
local img = {}
function img.models() return { { id = "zimage", context_length = 0 } } end
function img.props() return { n_ctx = 0, total_slots = 1 } end
function img.generate(_, req) return { seeds = { req.seed } } end
local r2 = router.new({ stt, img })
ok(r2:pick(nil, "generate") == img and r2:pick("whisper", "generate") == img,
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
ok(r3:chat({ model = "nope", messages = img_msgs }) == "eyes", "images with an unknown model go to a vision model")
ok(r3:chat({ messages = { { role = "user", content = "hi" } } }) == "text", "text stays on the first chat model")
local _, msg, code = r3:chat({ model = "text", messages = img_msgs })
ok(code == 400 and msg:find("models with vision: eyes", 1, true), "a named model without vision names the vision models")
_, msg, code = router.new({ vis("text", false) }):chat({ messages = img_msgs })
ok(code == 400 and msg:find("no model has an mmproj", 1, true), "no vision model gives 400")

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
ok(caps.a == "completion,decide,transcription,speech,image_generation", "fake engine has every mode but multimodal")
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
ok(r5:pick(nil, "speak") == tts and r5:pick("cv", "speak") == tts, "speech goes by modes")
ok(r5:pick(nil, "chat") == cv and r5:pick("tts", "decide") == cv, "chat and decide go by modes")
ok(r5:chat({ messages = img_msgs }) == true and cv.loads == 1, "images go to a multimodal model by modes")
ok(tts.loads == 0 and select(3, router.new({ tts }):chat({ messages = {} })) == 400,
    "no chat model gives 400")

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
