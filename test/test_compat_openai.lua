-- OpenAI Chat Completions compatibility: the field table of docs/api.md
-- against the request checks and the openai-python request fields.
-- TAP output.

local json = require("lgml.json")
local chat_request = require("lgml.server.chat_request")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local root = (arg[0]:match("^(.*)/[^/]*$") or ".") .. "/.."

local function read(path)
	local f = assert(io.open(root .. "/" .. path, "rb"))
	local s = f:read("a")
	f:close()
	return s
end

local function fixture(name)
	return assert(json.decode(read("test/fixtures/openai/" .. name)))
end

-- The rows of the markdown table after heading, as lists of cells.
local function table_rows(md, heading)
	local at = md:find("\n" .. heading .. "\n", 1, true)
	assert(at, "no heading " .. heading)
	local rows, started = {}, false
	for line in md:sub(at + #heading + 2):gmatch("([^\n]*)\n") do
		if line:match("^|") then
			started = true
			local cells = {}
			for cell in line:gmatch("|%s*([^|]-)%s*%f[|]") do
				cells[#cells + 1] = cell
			end
			if not cells[1]:match("^%-%-%-") and cells[1] ~= "Field" then
				rows[#rows + 1] = cells
			end
		elseif started then
			break
		end
	end
	return rows
end

-- A deep copy, so each check sees fresh tables.
local function copy(v)
	if type(v) ~= "table" then
		return v
	end
	local t = {}
	for k, x in pairs(v) do
		t[k] = copy(x)
	end
	return setmetatable(t, getmetatable(v))
end

local function body(extra)
	local b = { messages = json.array({ { role = "user", content = "Hi" } }) }
	for k, v in pairs(extra or {}) do
		b[k] = copy(v)
	end
	return b
end

local ACCEPTED = { supported = true, limited = true, ["no-op"] = true }

-- Fields that the good value of a field needs, when lgml accepts it.
local COMPANIONS = {
	tool_choice = { tools = { { type = "function", ["function"] = { name = "f" } } } },
	top_logprobs = { logprobs = true },
}

local function test_field_table()
	local md = read("docs/api.md")
	local rows = table_rows(md, "### Request fields")
	local docs = {}
	for _, r in ipairs(rows) do
		local name = r[1]:match("^`([%w_]+)`$")
		ok(name ~= nil, "request row names a field: " .. r[1])
		if name then
			docs[name] = { label = r[2], status = r[3] }
		end
	end
	local fields = {}
	for _, c in ipairs(chat_request.FIELDS) do
		fields[c[1]] = c
		local d = docs[c[1]]
		ok(d and d.status == c.status and d.label == c.label,
		    ("%s: docs say %s %s, the checks %s %s"):format(c[1], d and d.label, d and d.status,
		    c.label, c.status))
		ok(chat_request.STATUS[c.status] ~= nil, c[1] .. " has a known status")
	end
	for name in pairs(docs) do
		ok(fields[name] ~= nil, name .. " of the docs table is in FIELDS")
	end

	local sdk = fixture("request_fields.json").fields
	ok(#sdk >= 36, "the SDK list has the request fields")
	for _, f in ipairs(sdk) do
		local c = fields[f.name]
		ok(c ~= nil and c.label == "OpenAI", f.name .. ": OpenAI field in FIELDS")
		if c then
			local extra = {}
			if ACCEPTED[c.status] then
				for k, v in pairs(COMPANIONS[f.name] or {}) do
					extra[k] = v
				end
			end
			extra[f.name] = f.good
			local b = f.name == "messages" and { messages = copy(f.good) } or body(extra)
			local good, err, param = chat_request.check(b)
			if ACCEPTED[c.status] then
				ok(good == true, ("%s (%s) accepts %s: %s"):format(f.name, c.status, json.encode(f.good),
				    tostring(err)))
			else
				ok(not good and param == f.name, ("%s (%s) gives 400 for %s: %s"):format(f.name,
				    c.status, json.encode(f.good), tostring(err)))
			end
			if f.limit then
				good, err, param = chat_request.check(body({ [f.name] = f.limit }))
				ok(c.status == "limited" and not good and param == f.name,
				    ("%s: the limit value gives 400: %s"):format(f.name, tostring(err)))
			end
		end
	end
	local _, err, param = chat_request.check(body({ nope = 1 }))
	ok(param == "nope" and err:find("unknown field"), "an unknown field gives 400")
end

local function test_messages()
	local function msg_check(m)
		return chat_request.check({ messages = json.array({ m }) })
	end
	local good = {
		{ role = "user", content = "hi", name = "ann" },
		{ role = "user", content = json.array({ { type = "text", text = "hi" },
			{ type = "image_url", image_url = { url = "data:image/png;base64,AA", detail = "low" } } }) },
		{ role = "assistant", content = json.null, refusal = json.null, annotations = json.array(),
			tool_calls = json.array({ { id = "call_1", type = "function",
				["function"] = { name = "f", arguments = "{}" } } }) },
		{ role = "tool", tool_call_id = "call_1", content = json.array({ { type = "text", text = "ok" } }) },
		{ role = "developer", content = "Be brief." },
	}
	for i, m in ipairs(good) do
		local r, err = msg_check(m)
		ok(r == true, ("message %d is accepted: %s"):format(i, tostring(err)))
	end
	local bad = {
		{ { role = "function", name = "f", content = "x" }, "role" },
		{ { role = "user", content = json.array({ { type = "input_audio",
			input_audio = { data = "AA", format = "wav" } } }) }, "input_audio" },
		{ { role = "user", content = json.array({ { type = "file", file = { file_id = "f" } } }) }, "file" },
		{ { role = "assistant", content = "x", audio = { id = "a" } }, "audio" },
		{ { role = "assistant", content = "x", refusal = "no" }, "refusal" },
		{ { role = "user", content = "x", extra = 1 }, "extra" },
	}
	for _, c in ipairs(bad) do
		local r, err, param = msg_check(c[1])
		ok(not r and param == "messages" and err:find(c[2], 1, true),
		    ("message with %s gives 400: %s"):format(c[2], tostring(err)))
	end
end

test_field_table()
test_messages()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
