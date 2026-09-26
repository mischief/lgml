-- GBNF grammars and JSON schemas. Input: grammar text, a NUL, then
-- bytes that drive the matcher. Input that starts with "{" is a JSON
-- schema instead; its grammar then drives the matcher. A bad pattern
-- can give a grammar that does not parse, as in llama.cpp. Bad grammars
-- give nil and a message; match limits raise "grammar: " errors.
local lgml = require("lgml")
local json_schema = require("lgml.json_schema")

-- A tiny vocab: ASCII, multi-byte and partial UTF-8 pieces, a special
-- token, and end of generation.
local pieces = { "{", "}", "[", "]", "\"", ":", ",", " ", "\n", "a", "b", "0", "1", "-",
	"ab", "\"a\"", "true", "null", "ä", "\xc3", "\xa4", "\xe2\x82", "\xac", "😀",
	"\xf0\x9f", "\x98\x80", "<s>", "\\", "\\n", "x y", "", "</s>" }
local tk = {}
function tk.n_vocab()
	return #pieces
end
function tk.token(_, id)
	return pieces[id + 1], "normal"
end
function tk.special()
	return { eog = { #pieces - 1 } }
end

local be = assert(lgml.backend("CPU"))

-- Errors from match limits are expected; others are bugs.
local function limited(err)
	if not tostring(err):find("grammar: ", 1, true) then
		error(err, 0)
	end
end

local function drive(text, input)
	local g, err = lgml.grammar(text, { lax = true })
	if not g then
		assert(type(err) == "string", "error is a string")
		return false
	end
	local i = 1
	while i <= #input do
		local n = input:byte(i) % 7 + 1
		local r, e = g:accept(input:sub(i + 1, i + n))
		assert(r == true or (r == false and type(e) == "string"), "accept result")
		assert(type(g:can_end()) == "boolean" and g:stacks() >= 0, "state")
		i = i + n + 1
	end
	g:close()
	local good, s = pcall(lgml.sampler, { grammar = text, tokenizer = tk, seed = #input,
		temperature = #input % 3 == 0 and 0 or 1, top_k = #input % 5 })
	if not good then
		return true
	end
	for step = 1, math.min(#input, 64) do
		local ok, allowed = pcall(s.allowed, s)
		if not ok then
			limited(allowed)
			return true
		end
		if #allowed == 0 then
			break
		end
		local id = allowed[input:byte(step) % #allowed + 1]
		if step % 8 == 0 then
			local l = {}
			for k = 1, #pieces do
				l[k] = ((input:byte((step + k) % #input + 1) or 0) % 17) - 8
			end
			local x = be:tensor("f32", { 1, #pieces }, l)
			local sok, sid = pcall(s.sample, s, x)
			x:close()
			if not sok then
				limited(sid)
				return true
			end
			local fits = false
			for _, a in ipairs(allowed) do
				fits = fits or a == sid
			end
			assert(fits, "sampled token fits the grammar")
			id = sid
		end
		local aok, aerr = pcall(s.accept, s, id)
		if not aok then
			limited(aerr)
			assert(not tostring(aerr):find("does not fit"), "an allowed token fits")
			return true
		end
		if id == #pieces - 1 then
			break
		end
	end
	s:close()
	return true
end

return function(s)
	if s:sub(1, 1) == "{" then
		local v = json_schema.decode(s)
		if type(v) ~= "table" then
			return
		end
		local text, err = json_schema.to_grammar(v)
		if not text then
			assert(type(err) == "string", "schema error is a string")
			return
		end
		drive(text, s)
		return
	end
	local text, input = s:match("^(.-)%z(.*)$")
	drive(text or s, input or "")
end
