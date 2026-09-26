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

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
