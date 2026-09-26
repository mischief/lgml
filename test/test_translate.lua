-- Tests for lgml.server.translate: ids, images, endings, signatures,
-- field checks and the block sink. TAP output.
local json = require("lgml.json")
local base64 = require("lgml.base64")
local translate = require("lgml.server.translate")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

local function test_ids()
	local a, b = translate.id("msg_"), translate.id("msg_")
	ok(a:match("^msg_[%w]+$") and #a == 28 and a ~= b, "id: prefix and 24 base62 characters")
	ok(translate.hex_id("x-", 8):match("^x%-%x%x%x%x%x%x%x%x$"), "hex_id")
end

local function test_images()
	local p = assert(translate.image_part("image/png", PNG))
	ok(p.type == "image_url" and p.image_url.url == "data:image/png;base64," .. PNG, "a base64 PNG")
	ok(translate.image_part(nil, PNG) ~= nil, "the media type is optional")
	local _, err = translate.image_part("image/jpeg", PNG)
	ok(err and err:find("image/png", 1, true), "a media type that does not match the bytes: " .. tostring(err))
	_, err = translate.image_part("image/png", "data:image/png;base64," .. PNG)
	ok(err and err:find("data URL", 1, true), "data with a data: prefix")
	_, err = translate.image_part("image/gif", base64.encode("GIF89a\1\0\1\0\0\0\0"))
	ok(err and err:find("image/gif is not supported", 1, true), "GIF names the supported types")
	_, err = translate.image_part(nil, base64.encode("hello world, not an image"))
	ok(err and err:find("not a PNG", 1, true), "bytes that are no image")
	ok(translate.sniff("RIFF\0\0\0\0WEBPVP8 ") == "image/webp", "sniff WebP")
	ok(translate.url_image("data:image/png;base64," .. PNG) ~= nil, "a data URL")
	_, err = translate.url_image("https://example.com/cat.png")
	ok(err and err:find("send base64", 1, true), "an https URL gives an error")
end

local function test_ending()
	local e = translate.ending
	ok(e(nil, 0) == "stop", "no finish reads as stop")
	ok(e({ reason = "stop", usage = {} }, 0) == "stop", "stop")
	ok(e({ reason = "stop", usage = {} }, 1) == "tool_calls", "stop with calls")
	ok(e({ reason = "stop", usage = { stop_sequence = "x" } }, 0) == "stop_sequence", "a stop string")
	ok(e({ reason = "length", usage = { completion_tokens = 10 } }, 0, 10) == "length", "the token limit")
	ok(e({ reason = "length", usage = { completion_tokens = 7 } }, 0, 10) == "context", "a full context")
	local c = translate.counts({ prompt_tokens = 9, cached_tokens = 4, completion_tokens = 3 })
	ok(c.prompt == 9 and c.cached == 4 and c.output == 3 and c.reasoning == 0, "counts")
end

local function test_signatures()
	local mac = translate.hmac("Jefe", "what do ya want for nothing?")
	local hex = mac:gsub(".", function(ch)
		return ("%02x"):format(ch:byte())
	end)
	ok(hex == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843", "HMAC-SHA256 RFC 4231 case 2")
	local key = translate.load_key()
	ok(#key == 32, "an in-memory key")
	local sig = translate.sign(key, "m", "I think.")
	ok(translate.ours(sig) and sig:match("^lgml1%.%.[%w_-]+$"), "a signature without the text")
	ok(translate.verify(key, "m", sig, "I think.") == "I think.", "it verifies with the text")
	ok(translate.verify(key, "m", sig, "I think!") == nil, "changed text fails")
	ok(translate.verify(key, "other", sig, "I think.") == nil, "another model fails")
	sig = translate.sign(key, "m", "héllo\n", true)
	ok(translate.verify(key, "m", sig, "") == "héllo\n", "the payload holds the text")
	ok(not translate.ours("EqQBCgIYAhIM"), "a foreign signature")
	ok(translate.unb64url(translate.b64url("\0\1\2\255")) == "\0\1\2\255", "base64url round trip")
	local dir = os.tmpname()
	os.remove(dir)
	local k1 = translate.load_key(dir .. "/state")
	local k2 = translate.load_key(dir .. "/state")
	ok(#k1 == 32 and k1 == k2, "the key file keeps the key")
	os.remove(dir .. "/state/signing.key")
	os.remove(dir .. "/state")
	os.remove(dir)
end

local function test_fields()
	local fields = {
		{ "a", "number", function(v) return v > 0 end, "a positive number" },
		{ "b", "string|table" },
	}
	local function unknown(path)
		return path .. ": Extra inputs are not permitted"
	end
	local ctx = translate.context()
	local obj = { a = 1, b = "x", c = json.null }
	ok(translate.check_fields(obj, fields, "", ctx, unknown) and obj.c == nil, "good fields; null reads as absent")
	local good, err, path = translate.check_fields({ a = -1 }, fields, "x.", ctx, unknown)
	ok(not good and path == "x.a" and err == "x.a: must be a positive number", "a bad value names the path")
	good, err, path = translate.check_fields({ z = 1 }, fields, "", ctx, unknown)
	ok(not good and path == "z" and err == "z: Extra inputs are not permitted", "reject: an unknown field")
	translate.set_unknown_fields("ignore")
	obj = { z = 1, a = 2 }
	good = translate.check_fields(obj, fields, "", ctx, unknown)
	ok(good and obj.z == nil and ctx:headers({})["x-lgml-ignored"] == "z", "ignore: the header names the field")
	translate.set_unknown_fields("reject")
	ctx:ignore("z")
	ctx:ignore("metadata")
	ok(ctx:headers({})["x-lgml-ignored"] == "z,metadata", "each ignored field once, in order")
end

local function test_blocks()
	local events = {}
	local h = {}
	for _, name in ipairs({ "prompt", "open", "delta", "close" }) do
		h[name] = function(_, a, b)
			local what = type(a) == "table" and a.kind .. a.index or tostring(a)
			events[#events + 1] = name .. ":" .. what .. (b and ":" .. tostring(b) or "")
			return true
		end
	end
	local res = { s = { gone = function() return false end } }
	local s = translate.blocks(res, h)
	s:prompt(10, 2)
	s:reasoning("a")
	s:reasoning("b")
	s:content("")
	s:content("c")
	local i = s:tool_call("f")
	s:tool_arguments(i, "{}")
	s:finish("stop", { prompt_tokens = 10 })
	ok(table.concat(events, " ") == "prompt:10:2 open:reasoning0 delta:reasoning0:a delta:reasoning0:b " ..
	    "close:reasoning0 open:text1 delta:text1:c close:text1 open:tool2 delta:tool2:{} close:tool2",
	    "the handler sees each block open, grow and close: " .. table.concat(events, " "))
	ok(#s.list == 3 and translate.text(s.list[1]) == "ab" and s.calls[1].name == "f" and i == 0,
	    "the blocks and the calls")
	ok(s.prompt_n == 10 and s.cached_n == 2 and s.finished.reason == "stop", "prompt size and finish")
	ok(not pcall(s.content, s, "x"), "output after finish raises")
end

test_ids()
test_images()
test_ending()
test_signatures()
test_fields()
test_blocks()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
