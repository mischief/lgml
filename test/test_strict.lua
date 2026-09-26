-- lgml.strict: undefined reads and undeclared writes fail. Output is TAP.
local strict = require("lgml.strict")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = failed + 1
	end
end

strict.on()
strict.on()
ok(getmetatable(_G).__strict ~= nil, "on twice keeps one metatable")
top_level = 1
ok(top_level == 1, "the main chunk sets a global")
local good, err = pcall(function() return no_such_global end)
ok(not good and err:find("undefined global 'no_such_global'", 1, true), "undefined read is an error")
good, err = pcall(function() made_in_function = 1 end)
ok(not good and err:find("undeclared global 'made_in_function'", 1, true), "write in a function is an error")
ok(rawget(_G, "made_in_function") == nil, "the failed write sets nothing")
strict.declare("optional_setting")
ok(optional_setting == nil, "a declared global reads as nil")
good = pcall(function() optional_setting = 2 end)
ok(good and optional_setting == 2, "a declared global can be set in a function")
ok(strict.get("still_missing") == nil, "get reads without an error")
ok(_G[1] == nil, "a key that is not a string reads as nil")
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
