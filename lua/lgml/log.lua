--- Log lines for lgml-server and its workers.
-- Each event is one line of `key=value` fields:
--
--     ts=2026-09-26T10:00:00.123Z level=info comp=http rid=5f3a09c1 msg=access status=200
--
-- `ts` is the UTC time, `comp` the component (such as `http`, `pool`,
-- `router`, or `worker:<model id>/engine` in a worker) and `rid` the
-- request id of the running coroutine, when it has one (see `bind`).
-- A value with a space, a quote, `=` or a control character is quoted,
-- with `\"`, `\\`, `\n`, `\r`, `\t` and `\xNN` escapes. Data URLs and
-- long base64 runs never print in full: they show as
-- `<data image/png 1234B sha256:0a1b2c3d>` or `<base64 1234B sha256:...>`.
-- A value longer than `MAX_VALUE` bytes is cut and ends with
-- `...[+N bytes]`, unless it is wrapped with `full`.
--
-- The levels are error, warn, info, debug and trace. A call below the
-- current level returns before it formats anything. A function value
-- runs only when the line prints, so it can compute a costly value.
-- @module lgml.log

local M = {}

--- Level numbers. A higher number logs more.
-- @field ERROR 1
-- @field WARN 2
-- @field INFO 3
-- @field DEBUG 4
-- @field TRACE 5
M.ERROR, M.WARN, M.INFO, M.DEBUG, M.TRACE = 1, 2, 3, 4, 5
--- Level names by number.
M.NAMES = { "error", "warn", "info", "debug", "trace" }
local BY_NAME = { error = 1, warn = 2, info = 3, debug = 4, trace = 5 }

--- The current level number. Read it for a fast check; change it
-- with `set_level`.
M.level = M.INFO
--- The longest value in bytes before it is cut.
M.MAX_VALUE = 2048
--- The shortest base64 run that is elided.
M.BLOB = 200

local prefix
local out_fn, out_file, out_path
local dump_dir

-- Request contexts by coroutine; weak keys, so a dead coroutine goes.
local ctx = setmetatable({}, { __mode = "k" })

--- Reads a level name or number.
-- @tparam string|int v a name such as `"debug"`, or 1 to 5
-- @treturn[1] int the level number
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.parse_level(v)
	local n = BY_NAME[type(v) == "string" and v:lower() or ""] or math.tointeger(tonumber(v))
	if not n or n < M.ERROR or n > M.TRACE then
		return nil, ("bad log level %s: use error, warn, info, debug or trace"):format(tostring(v))
	end
	return n
end

--- Sets the level.
-- @tparam string|int v a level name or number
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.set_level(v)
	local n, err = M.parse_level(v)
	if not n then
		return nil, err
	end
	M.level = n
	return true
end

--- The name of the current level.
-- @treturn string the name
function M.level_name()
	return M.NAMES[M.level]
end

--- Tells if a level logs now.
-- @tparam int level a level number, such as `DEBUG`
-- @treturn boolean true when lines of that level print
function M.on(level)
	return level <= M.level
end

--- Sends lines to a function in place of stderr.
-- @tparam ?function fn fn(line) gets each line without its newline; nil
-- goes back to stderr (or the log file)
function M.set_output(fn)
	out_fn = fn
end

--- Appends lines to a file in place of stderr.
-- @tparam ?string path the file; nil goes back to stderr
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.open(path)
	if out_file then
		out_file:close()
		out_file, out_path = nil, nil
	end
	if not path then
		return true
	end
	local f, err = io.open(path, "a")
	if not f then
		return nil, err
	end
	f:setvbuf("line")
	out_file, out_path = f, path
	return true
end

--- Opens the log file again, for log rotation.
-- @treturn[1] boolean true, also with no log file
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.reopen()
	if not out_path then
		return true
	end
	return M.open(out_path)
end

--- The log file path, or nil for stderr.
-- @treturn ?string the path
function M.file()
	return out_path
end

--- Sets the component prefix of this process.
-- A worker sets `"worker:<model id>"`. Then the component `worker`
-- prints as the prefix, and other components as `prefix/component`.
-- @tparam ?string p the prefix
function M.set_prefix(p)
	prefix = p
end

--- Sets the directory for `dump`.
-- @tparam ?string dir the directory; nil turns dumps off
function M.set_dump_dir(dir)
	dump_dir = dir
end

--- The directory for `dump`, or nil.
-- @treturn ?string the directory
function M.dump_dir()
	return dump_dir
end

--- The worker arguments for the current settings.
-- They are `--log-level`, and `--log-file` and `--dump-prompts` when set.
-- @treturn {string,...} the arguments
function M.args()
	local a = { "--log-level", M.level_name() }
	if out_path then
		a[#a + 1], a[#a + 2] = "--log-file", out_path
	end
	if dump_dir then
		a[#a + 1], a[#a + 2] = "--dump-prompts", dump_dir
	end
	return a
end

-- request context

--- Makes a request id: 8 hex digits.
-- @treturn string the id
function M.new_id()
	return ("%04x%04x"):format(math.random(0, 0xffff), math.random(0, 0xffff))
end

--- Binds a request id to a coroutine.
-- Each line from that coroutine then has `rid`. The context also holds
-- notes for the access line (see `note`).
-- @tparam ?string rid the request id; nil removes the context
-- @tparam[opt] thread co the coroutine (default: the running one)
-- @treturn ?table the context `{ rid, notes }`
function M.bind(rid, co)
	co = co or coroutine.running()
	if rid == nil then
		ctx[co] = nil
		return nil
	end
	local c = { rid = rid, notes = {} }
	ctx[co] = c
	return c
end

--- The request id of a coroutine.
-- @tparam[opt] thread co the coroutine (default: the running one)
-- @treturn ?string the id
function M.request_id(co)
	local c = ctx[co or coroutine.running()]
	return c and c.rid
end

--- Sets a note on the request of the running coroutine.
-- The access line of the request shows the notes. It does nothing
-- outside a request.
-- @tparam string k the key
-- @param v the value
function M.note(k, v)
	local c = ctx[coroutine.running()]
	if c then
		c.notes[k] = v
	end
end

--- The notes of the request of a coroutine.
-- @tparam[opt] thread co the coroutine (default: the running one)
-- @treturn table the notes; empty outside a request
function M.notes(co)
	local c = ctx[co or coroutine.running()]
	return c and c.notes or {}
end

-- values

local sha256

-- A sha256 hex prefix of s, or nil without the lgml C module.
local function hash(s)
	if sha256 == nil then
		local ok, lgml = pcall(require, "lgml")
		sha256 = ok and type(lgml) == "table" and lgml.sha256 or false
	end
	return sha256 and sha256(s):sub(1, 8) or nil
end

--- A short text for a blob: kind, size and a sha256 prefix.
-- @tparam string kind such as `"image/png"` or `"base64"`
-- @tparam string bytes the bytes
-- @treturn string the text, such as `<image/png 1234B sha256:0a1b2c3d>`
function M.blob(kind, bytes)
	local h = hash(bytes)
	return ("<%s %dB%s>"):format(kind, #bytes, h and (" sha256:" .. h) or "")
end

local function b64decode(s)
	local ok, v = pcall(require("lgml.base64").decode, s)
	return ok and v or nil
end

-- A data URL or base64 run as a blob text; the size and hash are of
-- the decoded bytes when they decode.
local function b64blob(kind, text)
	local clean = text:gsub("\\", "")
	local bytes = b64decode(clean)
	if not bytes then
		return ("<%s %d chars>"):format(kind, #clean)
	end
	return M.blob(kind, bytes)
end

--- Replaces data URLs and long base64 runs in s with blob texts.
-- @tparam string s the text
-- @treturn string the text
function M.elide(s)
	if #s < 16 then
		return s
	end
	if s:find("data:", 1, true) then
		s = s:gsub("data:([%w.+-]+/[%w.+-]+)[^,\"%s]-;base64,([%w+/=\\]+)", function(mime, b)
			return "<data " .. b64blob(mime, b):sub(2)
		end)
	end
	if #s >= M.BLOB then
		s = s:gsub("[%w+/\\]+=*", function(run)
			if #run >= M.BLOB and not run:find("^[%a]+$") then
				return b64blob("base64", run)
			end
		end)
	end
	return s
end

local FULL = {}

--- Wraps a string so its value is never cut.
-- Blobs in it are still elided.
-- @tparam string s the text
-- @treturn table the wrapped text
function M.full(s)
	return setmetatable({ s }, FULL)
end

local function cut(s, max)
	if #s <= max then
		return s
	end
	local n = max
	-- Do not end inside a UTF-8 sequence.
	while n > 0 and (s:byte(n + 1) or 0) & 0xC0 == 0x80 do
		n = n - 1
	end
	return ("%s...[+%d bytes]"):format(s:sub(1, n), #s - n)
end

-- A copy of a table with strings elided, for JSON.
local function redact(t, depth)
	if depth > 20 then
		return "<deep>"
	end
	local o = {}
	for k, v in pairs(t) do
		if type(v) == "string" then
			v = M.elide(v)
		elseif type(v) == "table" then
			v = redact(v, depth + 1)
		elseif type(v) == "function" or type(v) == "userdata" or type(v) == "thread" then
			v = tostring(v)
		end
		o[k] = v
	end
	return setmetatable(o, getmetatable(t))
end

local function to_text(v)
	local t = type(v)
	if t == "function" then
		v = v()
		t = type(v)
	end
	if t == "string" then
		return M.elide(v), false
	elseif t == "number" then
		if math.type(v) == "integer" then
			return tostring(v), false
		end
		return (("%.4f"):format(v):gsub("0+$", ""):gsub("%.$", "")), false
	elseif t == "table" and getmetatable(v) == FULL then
		return M.elide(v[1]), true
	elseif t == "table" and getmetatable(v) == nil or t == "table" and require("lgml.json").isarray(v) then
		local s = require("lgml.json").encode(redact(v, 0))
		return s or tostring(v), false
	end
	return tostring(v), false
end

local ESC = { ["\\"] = "\\\\", ['"'] = '\\"', ["\n"] = "\\n", ["\r"] = "\\r", ["\t"] = "\\t" }

--- Formats one value as it prints in a line.
-- @param v the value: a string, number, boolean, table (as JSON), a
-- `full` string, or a function that returns one of these
-- @treturn string the text, quoted when needed
function M.value(v)
	local s, whole = to_text(v)
	if not whole then
		s = cut(s, M.MAX_VALUE)
	end
	if s ~= "" and not s:find('[%s"=\\%c]') then
		return s
	end
	return '"' .. s:gsub('[\\"%c]', function(c)
		return ESC[c] or ("\\x%02x"):format(c:byte())
	end) .. '"'
end

local clock
local last_sec, last_stamp

local function stamp()
	if clock == nil then
		local ok, uv = pcall(require, "luv")
		clock = ok and uv.gettimeofday or false
	end
	local sec, usec
	if clock then
		sec, usec = clock()
	else
		sec, usec = os.time(), 0
	end
	if sec ~= last_sec then
		last_sec, last_stamp = sec, os.date("!%Y-%m-%dT%H:%M:%S", sec)
	end
	return ("%s.%03dZ"):format(last_stamp, usec // 1000)
end

local function component(comp)
	if not prefix then
		return comp
	elseif comp == "worker" then
		return prefix
	end
	return prefix .. "/" .. comp
end

--- Formats one line. It does not check the level.
-- The fields after msg are key, value pairs; a nil value is left out.
-- A `rid` key overrides the request id of the coroutine.
-- @tparam int level the level number
-- @tparam string comp the component
-- @tparam string msg the event name or message
-- @param ... key, value, key, value, ...
-- @treturn string the line, without a newline
function M.format(level, comp, msg, ...)
	local rid = M.request_id()
	local t = { "ts=" .. stamp(), "level=" .. M.NAMES[level], "comp=" .. M.value(component(comp)) }
	local rest = {}
	for i = 1, select("#", ...), 2 do
		local k, v = select(i, ...)
		if k == "rid" then
			rid = v
		elseif v ~= nil then
			rest[#rest + 1] = k .. "=" .. M.value(v)
		end
	end
	if rid then
		t[#t + 1] = "rid=" .. M.value(rid)
	end
	t[#t + 1] = "msg=" .. M.value(msg)
	table.move(rest, 1, #rest, #t + 1, t)
	return table.concat(t, " ")
end

--- Prints one line when level is on.
-- @tparam int level the level number
-- @tparam string comp the component
-- @tparam string msg the event name or message
-- @param ... key, value pairs
function M.write(level, comp, msg, ...)
	if level > M.level then
		return
	end
	local line = M.format(level, comp, msg, ...)
	if out_fn then
		out_fn(line)
	elseif out_file then
		out_file:write(line, "\n")
	else
		io.stderr:write(line, "\n")
	end
end

--- Writes text to a file of the dump directory.
-- The file is `<dir>/<rid>.<kind>.txt`, with the request id of the
-- running coroutine (`norid` outside a request). Text appends, so one
-- request can dump several prompts. Blobs in text are elided.
-- @tparam string kind the kind, such as `"chat"` or `"decide"`
-- @tparam string text the text
-- @treturn ?string the file path, or nil with no dump directory or on
-- a write error
function M.dump(kind, text)
	if not dump_dir then
		return nil
	end
	local path = ("%s/%s.%s.txt"):format(dump_dir, M.request_id() or "norid", kind)
	local f = io.open(path, "a")
	if not f then
		return nil
	end
	f:write(M.elide(text))
	f:close()
	return path
end

-- loggers

local L = {}
L.__index = L

--- Makes a logger for a component.
-- @tparam string comp the component, such as `"http"`
-- @treturn Logger the logger
function M.new(comp)
	return setmetatable({ comp = comp }, L)
end

--- A logger of one component.
-- Each method takes msg and key, value pairs, as `format`.
-- @type Logger

--- Tells if a level logs now.
-- @tparam int level a level number
-- @treturn boolean true when lines of that level print
function L.on(_, level)
	return level <= M.level
end

--- Logs at error level.
-- @tparam string msg the event
-- @param ... key, value pairs
function L:error(msg, ...)
	if M.level >= 1 then
		M.write(1, self.comp, msg, ...)
	end
end

--- Logs at warn level.
-- @tparam string msg the event
-- @param ... key, value pairs
function L:warn(msg, ...)
	if M.level >= 2 then
		M.write(2, self.comp, msg, ...)
	end
end

--- Logs at info level.
-- @tparam string msg the event
-- @param ... key, value pairs
function L:info(msg, ...)
	if M.level >= 3 then
		M.write(3, self.comp, msg, ...)
	end
end

--- Logs at debug level.
-- @tparam string msg the event
-- @param ... key, value pairs
function L:debug(msg, ...)
	if M.level >= 4 then
		M.write(4, self.comp, msg, ...)
	end
end

--- Logs at trace level.
-- @tparam string msg the event
-- @param ... key, value pairs
function L:trace(msg, ...)
	if M.level >= 5 then
		M.write(5, self.comp, msg, ...)
	end
end

return M
