-- Render chat fixtures with a running llama-server.
-- usage: lua5.4 scripts/chat_fixtures.lua dir [url]
-- Each dir/NAME.json is an /apply-template request. The prompt that
-- the server returns is written to dir/NAME.txt. Needs curl.
local dir = assert(arg[1], "usage: chat_fixtures.lua dir [url]")
local url = (arg[2] or "http://127.0.0.1:8080") .. "/apply-template"

local function shell_quote(s)
	return "'" .. s:gsub("'", "'\\''") .. "'"
end

-- The prompt field of the server reply. The reply is JSON; the prompt
-- is the one string value, so a small decoder is enough.
local function prompt_of(body)
	local s = body:match('^%s*{%s*"prompt"%s*:%s*"(.*)"%s*}%s*$')
	if not s then
		return nil
	end
	-- One pass, so an escaped backslash before "u" stays a backslash.
	local esc = { n = "\n", t = "\t", r = "\r", b = "\b", f = "\f", ['"'] = '"', ["\\"] = "\\", ["/"] = "/" }
	return (s:gsub("\\(.)(%x?%x?%x?%x?)", function(c, hex)
		if c == "u" and #hex == 4 then
			return utf8.char(tonumber(hex, 16))
		end
		return (esc[c] or c) .. hex
	end))
end

local list = io.popen("ls " .. shell_quote(dir))
local n = 0
for name in list:lines() do
	local base = name:match("^(.*)%.json$")
	if base then
		local cmd = ("curl -sf -H 'Content-Type: application/json' --data-binary @%s %s"):format(
		    shell_quote(dir .. "/" .. name), shell_quote(url))
		local p = io.popen(cmd)
		local body = p:read("a")
		p:close()
		local prompt = prompt_of(body)
		if not prompt then
			io.stderr:write(base, ": bad reply: ", body:sub(1, 200), "\n")
		else
			local f = assert(io.open(dir .. "/" .. base .. ".txt", "wb"))
			f:write(prompt)
			f:close()
			n = n + 1
		end
	end
end
list:close()
print(n .. " fixtures rendered")
