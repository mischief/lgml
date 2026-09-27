-- Tests for docs/api.md and the OpenAPI document against the routes
-- that lgml.server.openai installs. TAP output.

local json = require("lgml.json")
local server = require("lgml.server.httpd")
local openai = require("lgml.server.openai")
local openapi = require("lgml.server.openapi")
local systemone = require("lgml.systemone")
local fake = require("lgml.server.fake_engine")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function read(path)
	local f = assert(io.open(path, "rb"))
	local s = f:read("a")
	f:close()
	return s
end

local root = (arg[0]:match("^(.*)/[^/]*$") or ".") .. "/.."
local api_md = read(root .. "/docs/api.md")

local srv = server.new()
openai.install(srv, fake.new())
local routes = srv.routes

-- The document after a round trip through JSON, as a client sees it.
local function spec()
	return assert(json.decode(assert(json.encode(openapi.document(routes)))))
end

local doc = spec()
local schemas = doc.components.schemas

local function sorted(set)
	local t = {}
	for k in pairs(set) do
		t[#t + 1] = k
	end
	table.sort(t)
	return table.concat(t, ",")
end

local function list(a)
	local t = table.move(a, 1, #a, 1, {})
	table.sort(t)
	return table.concat(t, ",")
end

local function props(name)
	return schemas[name] and schemas[name].properties or {}
end

-- routes in the docs and the document

local function test_routes()
	local count = 0
	for path, methods in pairs(routes) do
		ok(api_md:find("`" .. path .. "`", 1, true) ~= nil, "docs/api.md names " .. path)
		for method in pairs(methods) do
			count = count + 1
			local op = doc.paths[path] and doc.paths[path][method:lower()]
			ok(op ~= nil and not op["x-lgml-undocumented"],
			    ("openapi has %s %s"):format(method, path))
		end
	end
	ok(count > 20, "openai.install registers the routes")
	local extra = 0
	for path in pairs(doc.paths) do
		if not routes[path] then
			extra = extra + 1
		end
	end
	ok(extra == 0, "openapi has no path that the server lacks")
end

-- document structure

-- Calls fn on each table in v, with its JSON pointer.
local function walk(v, at, fn)
	if type(v) ~= "table" then
		return
	end
	fn(v, at)
	for k, x in pairs(v) do
		walk(x, at .. "/" .. tostring(k), fn)
	end
end

local function test_structure()
	ok(doc.openapi == "3.1.0", "openapi is 3.1.0")
	ok(type(doc.info.title) == "string" and type(doc.info.version) == "string", "info has title and version")
	local ids, dup, missing = {}, false, {}
	for path, item in pairs(doc.paths) do
		for method, op in pairs(item) do
			if ids[op.operationId] then
				dup = true
			end
			ids[op.operationId] = true
			if type(op.responses) ~= "table" or next(op.responses) == nil then
				missing[#missing + 1] = method .. " " .. path
			end
			if method == "post" and not (op.requestBody and op.requestBody.content) then
				missing[#missing + 1] = method .. " " .. path .. " body"
			end
		end
	end
	ok(not dup, "operation ids are unique")
	ok(#missing == 0, "each operation has responses, each POST a body: " .. table.concat(missing, "; "))
	local bad = {}
	walk(doc, "#", function(t, at)
		local r = t["$ref"]
		if r then
			local name = r:match("^#/components/schemas/(.+)$")
			if not (name and schemas[name]) then
				bad[#bad + 1] = at .. " -> " .. tostring(r)
			end
		end
		if t.type == "object" and t.required then
			for _, k in ipairs(t.required) do
				if not (t.properties and t.properties[k]) then
					bad[#bad + 1] = at .. " requires " .. k
				end
			end
		end
	end)
	ok(#bad == 0, "refs resolve and required fields exist: " .. table.concat(bad, "; "))
end

-- validator fields

local function test_fields()
	local chat = props("ChatRequest")
	for _, c in ipairs(openai.CHAT_FIELDS) do
		ok(chat[c[1]] ~= nil, "ChatRequest has " .. c[1])
		if c[4] then
			ok(chat[c[1]].description and chat[c[1]].description:find(c[4], 1, true),
			    "ChatRequest " .. c[1] .. " carries the check text")
		end
	end
	ok(chat.messages and list(schemas.ChatRequest.required) == "messages", "ChatRequest requires messages")
	ok(chat.tools.type == "array" and chat.stream_options.type == "object", "table fields map to array or object")
	for _, k in ipairs({ "grammar", "response_format", "json_schema" }) do
		ok(chat[k] ~= nil, "ChatRequest has " .. k)
	end
	ok(list(props("ChatMessage").role.enum) == sorted(openai.MESSAGE_ROLES), "message roles")

	local gen, edit = props("ImageRequest"), props("ImageEditRequest")
	for _, c in ipairs(openai.IMAGE_FIELDS) do
		local v = c[2] == "number" and 1 or "x"
		local r, msg = openai.check_image({ prompt = "x", [c[1]] = v })
		local edit_only = not r and msg:find("for edits only", 1, true)
		ok(edit[c[1]] ~= nil, "ImageEditRequest has " .. c[1])
		ok((gen[c[1]] ~= nil) == not edit_only, "ImageRequest has " .. c[1] .. " when generations take it")
	end
	ok(gen.prompt and gen.size and edit.image, "image prompt, size and input image")

	local names = {}
	for _, c in ipairs(openai.EMBEDDING_FIELDS) do
		names[c[1]] = true
	end
	ok(sorted(props("EmbeddingRequest")) == sorted(names) and
	    schemas.EmbeddingRequest.additionalProperties == false and
	    list(schemas.EmbeddingRequest.required) == "input", "EmbeddingRequest fields")
	ok(list(props("EmbeddingRequest").encoding_format.enum) == "base64,float", "embedding formats")

	ok(list(props("SpeechRequest").response_format.enum) == sorted(openai.SPEECH_FORMATS), "speech formats")
	ok(list(props("TranscriptionRequest").response_format.enum) == sorted(openai.TRANSCRIPTION_FORMATS),
	    "transcription formats")

	local f = systemone.fields
	ok(sorted(props("SystemOneRequest")) == sorted(f.request) and
	    schemas.SystemOneRequest.additionalProperties == false, "SystemOneRequest fields")
	ok(sorted(props("SystemOneQuestion")) == sorted(f.question), "SystemOneQuestion fields")
	ok(sorted(props("SystemOneExtension")) == sorted(f.lgml), "SystemOneExtension fields")
	ok(list(props("SystemOneQuestion").type.enum) == sorted(f.types), "question types")
end

-- A field added to a check table shows up with no change to openapi.
local function test_new_fields()
	systemone.fields.request.test_field = true
	openai.CHAT_FIELDS[#openai.CHAT_FIELDS + 1] = { "test_field", "boolean" }
	local d = spec()
	systemone.fields.request.test_field = nil
	openai.CHAT_FIELDS[#openai.CHAT_FIELDS] = nil
	local s = d.components.schemas
	ok(s.SystemOneRequest.properties.test_field ~= nil, "new systemone field appears")
	ok(s.ChatRequest.properties.test_field and s.ChatRequest.properties.test_field.type == "boolean",
	    "new chat field appears with its type")
end

local function test_headers()
	local h = doc.paths["/v1/audio/speech"].post.responses["200"].headers
	ok(h["x-lgml-sample-rate"] and h["x-lgml-channels"], "speech lists the sample rate and channel headers")
end

test_routes()
test_structure()
test_fields()
test_new_fields()
test_headers()
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
