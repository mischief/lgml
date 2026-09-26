--- Grammars for chat requests.
-- A request can carry raw GBNF in `grammar` (a llama-server extension)
-- or a `response_format`. This module turns either into GBNF text,
-- checks it, and wraps it so that a reasoning block stays free.
-- @module lgml.grammar
local M = {}

local function is_table(v)
	return type(v) == "table"
end

--- Finds the GBNF text of a chat request.
-- `response_format` is `{ type = "text" }`, `{ type = "json_object" }`
-- (any JSON object, or `schema` when it has one) or `{ type =
-- "json_schema", json_schema = { schema = ... } }`. A top-level
-- `json_schema` field works as in llama-server. A request with both a
-- grammar and a schema is an error.
-- @tparam table req the chat request
-- @treturn[1] ?string the GBNF text, or nil when the request has none
-- @treturn[2] nil
-- @treturn[2] string an error message
-- @treturn[2] string the bad parameter name
function M.from_request(req)
	local g, rf, schema = req.grammar, req.response_format, req.json_schema
	if g ~= nil and type(g) ~= "string" then
		return nil, "grammar must be a string", "grammar"
	end
	if rf ~= nil then
		if not is_table(rf) or type(rf.type) ~= "string" then
			return nil, "response_format must be an object with a type", "response_format"
		end
		if rf.type == "json_object" then
			schema = is_table(rf.schema) and rf.schema or { type = "object" }
		elseif rf.type == "json_schema" then
			local js = rf.json_schema
			schema = is_table(js) and js.schema or rf.schema
			if not is_table(schema) then
				return nil, "response_format.json_schema.schema must be an object",
				    "response_format"
			end
		elseif rf.type ~= "text" then
			return nil, "response_format.type must be text, json_object or json_schema",
			    "response_format"
		end
	end
	if schema ~= nil then
		if g ~= nil then
			return nil, "use either grammar or a JSON schema, not both", "grammar"
		end
		if not is_table(schema) then
			return nil, "json_schema must be an object", "json_schema"
		end
		local text, err = require("lgml.json_schema").to_grammar(schema)
		if not text then
			return nil, err, rf and "response_format" or "json_schema"
		end
		return text
	end
	return g
end

--- Checks GBNF text without a tokenizer.
-- Token names such as `<think>` are not checked here.
-- @tparam string text the grammar
-- @tparam[opt="root"] string root the start rule
-- @treturn[1] boolean true
-- @treturn[2] nil
-- @treturn[2] string the error message
function M.check(text, root)
	local lgml = require("lgml")
	local g, err = lgml.grammar(text, { lax = true, root = root })
	if not g then
		return nil, "grammar: " .. err
	end
	g:close()
	return true
end

-- A GBNF literal or class member for byte c, as a hex escape.
local function hex(c)
	return ("\\x%02X"):format(c)
end

--- Wraps a grammar so that a reasoning block before the answer is free.
-- The block is `open`, any text without `close`, then `close`; the
-- answer may follow after white space. The rules match the text with a
-- string-search automaton, one rule per matched byte of close.
-- @tparam string text the grammar of the answer, with a `root` rule
-- @tparam string open the tag that opens the block, ASCII
-- @tparam string close the tag that ends the block, ASCII
-- @tparam boolean opened true when the prompt already opened the block
-- @treturn string the new grammar text
-- @treturn string the start rule name
function M.reasoning(text, open, close, opened)
	assert(#close > 0 and not close:find("[\128-\255]"), "close tag must be ASCII")
	local n = #close
	local rules = {}
	local function state(k)
		return "lgml-think-" .. k
	end
	-- next[k][c]: the state after byte c in state k (KMP).
	local fail = { [0] = 0, [1] = 0 }
	for k = 2, n do
		local f = fail[k - 1]
		while f > 0 and close:byte(f + 1) ~= close:byte(k) do
			f = fail[f]
		end
		if close:byte(f + 1) == close:byte(k) then
			f = f + 1
		end
		fail[k] = f
	end
	local function next_state(k, c)
		while k > 0 and close:byte(k + 1) ~= c do
			k = fail[k]
		end
		if close:byte(k + 1) == c then
			return k + 1
		end
		return 0
	end
	local bytes = {}
	for i = 1, n do
		bytes[close:byte(i)] = true
	end
	for k = 0, n - 1 do
		local alts, stay = {}, {}
		for c in pairs(bytes) do
			local t = next_state(k, c)
			if t == n then
				alts[#alts + 1] = ('"%s" lgml-answer'):format(hex(c))
			elseif t > 0 then
				alts[#alts + 1] = ('"%s" %s'):format(hex(c), state(t))
			end
			if t ~= 0 then
				stay[#stay + 1] = hex(c)
			end
		end
		table.sort(alts)
		table.sort(stay)
		alts[#alts + 1] = ("[^%s] %s"):format(table.concat(stay), state(0))
		rules[#rules + 1] = ("%s ::= %s"):format(state(k), table.concat(alts, " | "))
	end
	local lit = open:gsub(".", function(c)
		return hex(c:byte())
	end)
	local root = opened and state(0) or ('"%s" %s | lgml-answer'):format(lit, state(0))
	rules[#rules + 1] = "lgml-answer ::= [ \\t\\r\\n]* root"
	rules[#rules + 1] = "lgml-root ::= " .. root
	return text .. "\n" .. table.concat(rules, "\n") .. "\n", "lgml-root"
end

return M
