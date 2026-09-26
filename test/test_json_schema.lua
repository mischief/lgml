-- JSON schema to grammar tests, on the llama.cpp cases. Output is TAP.
local js = require("lgml.json_schema")
local json = require("lgml.json")
local cases = dofile((arg[0]:match("^(.*)/") or ".") .. "/json_schema_cases.lua")

-- The C grammar parser, when the build has it.
local grammar
do
	local ok, lgml = pcall(require, "lgml")
	if ok and type(lgml) == "table" then
		grammar = lgml.grammar
	end
end

local n, failed = 0, 0
local function ok(cond, name, detail)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = failed + 1
		if detail then
			io.stderr:write(detail, "\n")
		end
	end
end

local function skip(name, why)
	n = n + 1
	print("ok " .. n .. " - " .. name .. " # SKIP " .. why)
end

-- The trim of the C++ test: outer space, and space at line starts.
local function trim(s)
	s = s:gsub("^[ \n\r\t]+", ""):gsub("[ \n\r\t]+$", "")
	return (s:gsub("^[ \t]+", ""):gsub("\n[ \t]+", "\n"))
end

local function parses(text)
	if not grammar then
		return true
	end
	local good, g = pcall(grammar, text)
	return good and g ~= nil and g ~= false
end

-- Cases that this port does not try to match, with the reason.
local skips = {}

for _, c in ipairs(cases) do
	if skips[c.name] then
		skip(c.name, skips[c.name])
	else
		local schema, derr = js.decode(c.schema)
		assert(schema ~= nil, derr)
		local text, err = js.to_grammar(schema)
		if c.ok then
			local same = text ~= nil and trim(text) == trim(c.grammar)
			ok(same, c.name, not same and ("# EXPECTED:\n%s\n# ACTUAL:\n%s"):format(c.grammar, text or err))
			if same and grammar then
				ok(parses(text), c.name .. " (grammar parses)")
			end
		else
			ok(text == nil and err:find("^JSON schema conversion failed:\n"), c.name .. " (fails)", text)
		end
	end
end

-- decode
local function order(t)
	return table.concat(getmetatable(t).__order, ",")
end

do
	local v = assert(js.decode('{"b": 1, "a": [1.0, 2, null, true], "c": {"z": "\\u00e9\\ud83d\\ude00", "y": -0.5}}'))
	ok(order(v) == "b,a,c", "decode keeps key order")
	ok(order(v.c) == "z,y", "decode keeps nested key order")
	ok(math.type(v.b) == "integer" and math.type(v.a[1]) == "float", "decode integer and float")
	ok(json.isarray(v.a) and #v.a == 4 and v.a[3] == json.null and v.a[4] == true, "decode array and null")
	ok(v.c.z == "\u{e9}\u{1f600}" and v.c.y == -0.5, "decode escapes")
	local e = assert(js.decode('{"a": 1, "b": 2, "a": 3}'))
	ok(order(e) == "a,b" and e.a == 3, "decode repeated key")
	ok(json.isarray(assert(js.decode("[]"))), "decode empty array")
end

for _, bad in ipairs({ "", "{", "[1,]", "01", "1 2", '{"a" 1}', '"\\ud800"', '"\\x"', '"a\nb"',
	'"\xff"', "tru", "[1e999]", "{1: 2}", "-", "1." }) do
	local v, err = js.decode(bad)
	local name = ("decode rejects %q"):format(bad):gsub("\\\n", "\\n"):gsub("[\128-\255]", function(c)
		return ("\\x%02x"):format(c:byte())
	end)
	ok(v == nil and type(err) == "string", name)
end
ok(js.decode(nil) == nil, "decode rejects nil")

do
	local deep = string.rep("[", 200) .. string.rep("]", 200)
	ok(js.decode(deep) ~= nil, "decode depth 200")
	local v, err = js.decode("[" .. deep .. "]")
	ok(v == nil and err:find("deep"), "decode depth 201 fails")
end

-- to_grammar
local function g(text)
	return js.to_grammar(assert(js.decode(text)))
end

ok(js.json_object == g('{"type": "object"}'), "json_object")

do
	local text, err = g('{"$ref": "https://example.com/schema.json#/definitions/a"}')
	ok(text == nil and err:find("Unsupported ref: https://", 1, true), "remote ref fails")
	text, err = g('{"$ref": "#/definitions/nope", "definitions": {}}')
	ok(text == nil and err:find("Error resolving ref"), "missing ref fails")
	text, err = g('{"$ref": "#/items/1", "items": [{"type": "string"}, {"type": "boolean"}]}')
	ok(text and text:find("root ::= boolean\n", 1, true), "ref with array index", text or err)
end

do
	local text, err = g('{"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/a"}}}')
	ok(text ~= nil, "ref cycle ends", err)
	text, err = g('{"allOf": [{"$ref": "#/$defs/a"}], "$defs": {"a": {"$ref": "#/$defs/a"}}}')
	ok(text == nil and err:find("Recursive ref"), "allOf ref cycle fails")
	text, err = g([[{"$ref": "#/$defs/node", "$defs": {"node": {"type": "object",
		"properties": {"v": {"type": "integer"},
		"next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}}}}}]])
	ok(text and text:find("ref-defs-node ::= ", 1, true), "recursive list", err)
	ok(text and parses(text), "recursive list parses")
end

do
	local text, warn = g('{"type": "string", "pattern": "^(?=a)b$"}')
	ok(text and #warn == 1 and warn[1] == "Unsupported pattern syntax", "lookahead warns")
	local _, w2 = g('{"type": "string"}')
	ok(#w2 == 0, "no warnings")
	for _, p in ipairs({ "^]$", "^*a$", "^a{x}$", "^[a$", "abc", "^a{1,2,3}$" }) do
		local t, err = js.to_grammar({ type = "string", pattern = p })
		ok(t == nil and err:find("^JSON schema conversion failed"), ("pattern %q fails"):format(p))
	end
end

do
	local text = g('{"enum": [1.0, 1e20, 0.0001, 1e-05, -0.0, 1e15, 1e14, 0.1, "a\\u0001\\"b\\u007f"]}')
	local want = [[root ::= ("1.0" | "1e+20" | "0.0001" | "1e-05" | "-0.0" | "1e+15" | ]]
		.. [["100000000000000.0" | "0.1" | "\"a\\u0001\\\"b]] .. "\127" .. [[\"")]]
	ok(text and text:find(want, 1, true), "enum dump matches nlohmann", text)
end

do
	-- lgml.json.decode has no key order: keys go in byte order.
	local s = assert(json.decode('{"properties": {"b": {"type": "string"}, "a": {"type": "string"}}, "required": ["a", "b"]}'))
	local text = assert(js.to_grammar(s))
	ok(text:find('root ::= "{" space a-kv "," space b-kv space "}"', 1, true), "unordered keys sort")
	ok(js.to_grammar({ type = "integer", minimum = 3 }) ~= nil, "plain Lua table")
end

-- Hostile input: all fail with an error, and none raise.
do
	local deep = {}
	local t = deep
	for _ = 1, 300 do
		t.items = {}
		t = t.items
	end
	local text, err = js.to_grammar(deep)
	ok(text == nil and err:find("deep"), "deep schema fails")

	local cyc = { type = "array" }
	cyc.items = cyc
	text, err = js.to_grammar(cyc)
	ok(text == nil and err:find("deep"), "cyclic table fails")

	local big = json.array({})
	for i = 1, 70000 do
		big[i] = string.rep("x", 16) .. i
	end
	text, err = js.to_grammar({ enum = big })
	ok(text == nil and err:find("larger than"), "huge grammar fails")

	local s = '{"type": "string"}'
	for _ = 1, 40 do
		s = '{"type": ["object", "object"], "properties": {"a": ' .. s .. "}}"
	end
	text, err = g(s)
	ok(text == nil and (err:find("steps") or err:find("larger than")), "exponential schema fails")

	local props = {}
	local order = {}
	for i = 1, 3000 do
		local k = "a" .. string.rep("!", i % 7) .. string.rep("?", i // 7)
		props[k] = { type = "integer" }
		order[#order + 1] = k
	end
	text, err = js.to_grammar({ properties = setmetatable(props, { __order = order }) })
	ok(text ~= nil or err:find("larger than"), "colliding names end", err)

	text, err = js.to_grammar({ type = "integer", minimum = -math.maxinteger })
	ok(text == nil and err, "bad integer range fails")
	text, err = js.to_grammar({ type = "string", format = 5 })
	ok(text == nil and err, "bad format fails")
	text, err = js.to_grammar({ const = "\xff" })
	ok(text == nil and err:find("UTF%-8"), "bad UTF-8 fails")
	text, err = js.to_grammar({ type = "kaboom" })
	ok(text == nil and err == 'JSON schema conversion failed:\nUnrecognized schema: {"type":"kaboom"}', "error text")
end

print("1.." .. n)
if failed > 0 then
	os.exit(1)
end
