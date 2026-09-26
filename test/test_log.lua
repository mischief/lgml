-- Tests for lgml.log. TAP output.
local log = require("lgml.log")
local base64 = require("lgml.base64")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local lines = {}
log.set_output(function(l) lines[#lines + 1] = l end)

-- levels
ok(log.level == log.INFO and log.level_name() == "info", "default level is info")
ok(log.parse_level("DEBUG") == 4 and log.parse_level("5") == 5 and log.parse_level(3) == 3,
    "levels parse from names and numbers")
local bad, err = log.parse_level("loud")
ok(bad == nil and err:find("bad log level"), "a bad level gives an error")
ok(not log.set_level("nope") and log.level == log.INFO, "a bad level does not change the level")

local L = log.new("test")
L:debug("hidden")
L:info("shown", "k", 1)
ok(#lines == 1 and lines[1]:find(" msg=shown k=1$"), "info prints, debug does not")
ok(lines[1]:find("^ts=%d%d%d%d%-%d%d%-%d%dT%d%d:%d%d:%d%d%.%d%d%dZ level=info comp=test msg="),
    "line starts with ts, level and comp")
log.set_level("trace")
L:trace("t")
ok(#lines == 2 and lines[2]:find("level=trace"), "trace prints at trace")
log.set_level("error")
L:warn("w")
L:error("e")
ok(#lines == 3 and lines[3]:find("level=error"), "only error at error")
ok(L:on(log.ERROR) and not L:on(log.WARN) and log.on(log.ERROR), "on checks the level")

-- A call under the level does not format its values.
local calls = 0
local probe = setmetatable({}, { __tostring = function() calls = calls + 1 return "p" end })
log.set_level("info")
L:debug("x", "v", probe, "f", function() calls = calls + 1 return "f" end)
ok(calls == 0, "a debug call at info formats nothing")
L:info("x", "f", function() calls = calls + 1 return "lazy value" end)
ok(calls == 1 and lines[#lines]:find('f="lazy value"', 1, true), "a function value runs when it prints")

-- formatting and quoting
ok(log.value("plain") == "plain", "a plain value has no quotes")
ok(log.value("two words") == '"two words"', "a value with a space is quoted")
ok(log.value('a"b\\c') == '"a\\"b\\\\c"', "quotes and backslashes escape")
ok(log.value("x=y") == '"x=y"', "a value with = is quoted")
ok(log.value("l1\nl2\t\1") == '"l1\\nl2\\t\\x01"', "control characters escape")
ok(log.value("") == '""', "an empty value is quoted")
ok(log.value(3) == "3" and log.value(2.5) == "2.5" and log.value(1 / 3) == "0.3333", "numbers")
ok(log.value(true) == "true", "booleans")
ok(log.value({ a = 1 }) == '"{\\"a\\":1}"', "a table prints as JSON")

L:info("skip nil", "a", nil, "b", 2)
ok(lines[#lines]:find("msg=\"skip nil\" b=2$"), "a nil value is left out")

-- cutting
local long = string.rep("x", log.MAX_VALUE + 100)
local v = log.value(long)
ok(#v < #long and v:find("...[+100 bytes]", 1, true), "a long value is cut with a marker")
local uni = string.rep("é", log.MAX_VALUE)
ok(utf8.len((log.value(uni):gsub("%.%.%.%[.*$", ""))) ~= nil, "a cut keeps whole UTF-8 characters")
ok(log.value(log.full(long)) == long, "a full value is not cut")

-- blobs
local png = "\137PNG\r\n\26\n" .. string.rep("\0\1\2\3", 300)
local url = "data:image/png;base64," .. base64.encode(png)
local s = log.elide("see " .. url .. " end")
ok(s:find("^see <data image/png " .. #png .. "B sha256:%x%x%x%x%x%x%x%x> end$"),
    "a data URL shows kind, size and hash")
ok(not log.value({ url = url }):find("iVBOR", 1, true), "a data URL in a table is elided")
local raw = base64.encode(string.rep("\255\1", 400))
ok(log.elide(raw):find("^<base64 800B sha256:"), "a long base64 run is elided")
ok(log.elide("short aGVsbG8=") == "short aGVsbG8=", "a short base64 run stays")
ok(log.value(log.full("x " .. url)):find("<data image/png", 1, true), "a full value is elided too")

-- request context
coroutine.wrap(function()
	log.bind("abc123")
	log.note("model", "m1")
	L:info("in request")
	ok(lines[#lines]:find(" rid=abc123 msg=\"in request\"", 1, true), "lines carry the request id")
	ok(log.notes().model == "m1", "notes belong to the request")
	L:info("override", "rid", "other")
	ok(lines[#lines]:find(" rid=other ", 1, true), "a rid field overrides the context")
end)()
L:info("outside")
ok(not lines[#lines]:find("rid="), "no request id outside a request")
ok(#log.new_id() == 8, "request ids have 8 hex digits")

-- worker prefix
log.set_prefix("worker:m1")
log.new("worker"):info("w")
ok(lines[#lines]:find(" comp=worker:m1 ", 1, true), "the worker component is the prefix")
log.new("engine"):info("e")
ok(lines[#lines]:find(" comp=worker:m1/engine ", 1, true), "other components get the prefix")
log.set_prefix(nil)

-- worker arguments, file output and dumps
local dir = os.tmpname()
os.remove(dir)
assert(os.execute("mkdir " .. dir))
local path = dir .. "/log.txt"
ok(log.open(path), "open a log file")
log.set_output(nil)
log.set_level("debug")
L:debug("to file")
log.set_dump_dir(dir)
local a = table.concat(log.args(), " ")
ok(a == ("--log-level debug --log-file %s --dump-prompts %s"):format(path, dir), "worker arguments")
log.open(nil)
local f = assert(io.open(path))
local text = f:read("a")
f:close()
ok(text:find("msg=\"to file\"", 1, true) and text:sub(-1) == "\n", "lines append to the file")
local dumped
coroutine.wrap(function()
	log.bind("r1")
	dumped = log.dump("chat", "prompt " .. url)
end)()
f = io.open(dumped or "")
text = f and f:read("a")
if f then
	f:close()
end
ok(dumped == dir .. "/r1.chat.txt" and text and text:find("^prompt <data image/png"),
    "dump writes the elided text to a file named by request id")
log.set_dump_dir(nil)
os.remove(path)
os.remove(dir .. "/r1.chat.txt")
os.remove(dir)

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
