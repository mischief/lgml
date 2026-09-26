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

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
