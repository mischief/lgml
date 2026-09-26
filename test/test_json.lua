-- Tests for lgml.json, the lua-cjson wrapper. TAP output.
local json = require("lgml.json")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

ok(json.decode("null") == json.null, "null sentinel")
ok(json.encode(json.null) == "null", "encode null")
ok(math.type(json.decode("42")) == "integer", "whole number decodes as integer")
ok(math.type(json.decode("2.0")) == "integer", "2.0 decodes as integer")
ok(json.decode("0.5") == 0.5, "float stays float")
ok(json.decode('{"a":[1,{"b":7}]}').a[2].b == 7 and
    math.type(json.decode('{"a":[1,{"b":7}]}').a[2].b) == "integer", "nested integers")
ok(json.encode(json.decode("[]")) == "[]", "empty array round-trips")
ok(json.encode(json.decode("{}")) == "{}", "empty object round-trips")
ok(json.isarray(json.decode("[1]")), "decoded array is marked")
ok(json.encode(json.array()) == "[]", "array marker")
ok(json.encode(json.object()) == "{}", "object marker")
ok(json.encode({ 1, 2, 3 }) == "[1,2,3]", "array")
ok(json.encode("a/b") == '"a/b"', "slash not escaped")
ok(json.decode('"\\u00e9\\ud83d\\ude00"') == "\u{e9}\u{1f600}", "unicode escapes")
ok(json.encode("\u{e9}") == '"\u{e9}"', "utf-8 passes through")
ok(json.encode(0.1) == "0.1", "encode 0.1")
local v, err = json.decode("{bad")
ok(v == nil and type(err) == "string", "bad json gives nil and message")
v, err = json.decode(string.rep("[", 5000) .. string.rep("]", 5000))
ok(v == nil and type(err) == "string", "deep nesting rejected")
ok(json.encode(0 / 0) == nil, "nan rejected")
local big = { messages = {} }
for i = 1, 20000 do
	big.messages[i] = { role = "user", content = string.rep("x", 40) }
end
local s = json.encode(big)
ok(#json.decode(s).messages == 20000, "1 MB round trip")

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
