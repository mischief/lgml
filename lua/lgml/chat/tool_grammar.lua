--- Grammars that force tool calls, for `tool_choice` `"required"` or a
-- named function.
-- Each function takes OpenAI function tools and gives GBNF text and its
-- start rule. The grammar allows only calls in the syntax of the chat
-- template, to the given tools, with nothing around them but white
-- space. With `parallel`, it allows one or more calls, else one.
-- @module lgml.chat.tool_grammar
local grammar = require("lgml.grammar")
local json_schema = require("lgml.json_schema")

local M = {}

local function fn(t)
	return type(t) == "table" and (t["function"] or t) or {}
end

-- The calls rule: one call, or calls with white space between them.
local function calls(parallel)
	local rules = "lgml-ws ::= [ \\t\\r\\n]*\n"
	if parallel then
		return rules .. "lgml-tools ::= lgml-call (lgml-ws lgml-call)*\n"
	end
	return rules .. "lgml-tools ::= lgml-call\n"
end

-- The function name alternatives, as a group of literals.
local function names(tools)
	local alts = {}
	for i, t in ipairs(tools) do
		alts[i] = grammar.literal(fn(t).name)
	end
	return "(" .. table.concat(alts, " | ") .. ")"
end

local function ordered(t, order)
	return setmetatable(t, { __order = order })
end

--- The grammar of hermes calls (Qwen3):
-- `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`.
-- The arguments follow the parameter schema of the tool.
-- @tparam {table,...} tools OpenAI function tools
-- @tparam[opt] table opts `parallel`: allow more than one call
-- @treturn[1] string the GBNF text
-- @treturn[1] string the start rule
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.hermes(tools, opts)
	local alts = {}
	local root = { anyOf = alts }
	for i, t in ipairs(tools) do
		local f = fn(t)
		local params = type(f.parameters) == "table" and f.parameters or { type = "object", properties = {} }
		-- Local refs ("#/$defs/...") resolve against the root schema.
		for _, k in ipairs({ "$defs", "definitions" }) do
			if type(params[k]) == "table" then
				root[k] = root[k] or {}
				for name, d in pairs(params[k]) do
					root[k][name] = d
				end
			end
		end
		alts[i] = {
			type = "object",
			properties = ordered({ name = { const = f.name }, arguments = params }, { "name", "arguments" }),
			required = { "name", "arguments" },
		}
	end
	local text, err = json_schema.to_grammar(root)
	if not text then
		return nil, err
	end
	return text .. calls(opts and opts.parallel) ..
	    'lgml-call ::= "<tool_call>\\n" root "\\n</tool_call>"\n', "lgml-tools"
end

-- The parameter names of a tool, as a group of literals, or nil when
-- the schema lists none.
local function param_names(f)
	local p = type(f.parameters) == "table" and f.parameters.properties
	local list = {}
	for k in pairs(type(p) == "table" and p or {}) do
		if type(k) == "string" then
			list[#list + 1] = grammar.literal(k)
		end
	end
	if #list == 0 then
		return nil
	end
	table.sort(list)
	return "(" .. table.concat(list, " | ") .. ")"
end

--- The grammar of Qwen3.5 XML calls:
-- `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>`.
-- Parameter names come from the schema of the tool; values are free
-- text.
-- @tparam {table,...} tools OpenAI function tools
-- @tparam[opt] table opts `parallel`: allow more than one call
-- @treturn string the GBNF text
-- @treturn string the start rule
function M.qwen35(tools, opts)
	local rules, alts = {}, {}
	for i, t in ipairs(tools) do
		local f = fn(t)
		local keys = param_names(f) or '[^>\\n]+'
		alts[i] = ("%s %s lgml-params-%d"):format(grammar.literal(f.name), grammar.literal(">\n"), i)
		rules[#rules + 1] = ('lgml-params-%d ::= ("<parameter=" %s ">\\n" lgml-value-0)*\n'):format(i, keys)
	end
	rules[#rules + 1] = grammar.until_text("lgml-value", "\n</parameter>\n")
	rules[#rules + 1] = ('lgml-call ::= "<tool_call>\\n<function=" (%s) "</function>\\n</tool_call>"\n')
	    :format(table.concat(alts, " | "))
	return table.concat(rules) .. calls(opts and opts.parallel), "lgml-tools"
end

--- The grammar of Gemma 4 calls: `<|tool_call>call:NAME{...}<tool_call|>`.
-- The text in the braces is free.
-- @tparam {table,...} tools OpenAI function tools
-- @tparam[opt] table opts `parallel`: allow more than one call
-- @treturn string the GBNF text
-- @treturn string the start rule
function M.gemma4(tools, opts)
	local rules = {
		('lgml-call ::= "<|tool_call>call:" %s "{" lgml-args-0\n'):format(names(tools)),
		grammar.until_text("lgml-args", "}<tool_call|>"),
	}
	return table.concat(rules) .. calls(opts and opts.parallel), "lgml-tools"
end

return M
