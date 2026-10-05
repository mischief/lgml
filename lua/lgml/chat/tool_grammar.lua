--- Grammars of tool calls: forced calls for `tool_choice` `"required"`
-- or a named function, and lazy grammars for `"auto"`.
-- Each function takes OpenAI function tools and gives GBNF text, its
-- start rule and the trigger strings. The grammar allows only calls in
-- the syntax of the chat template, to the given tools, with nothing
-- around them but white space. With `parallel`, it allows one or more
-- calls, else one. With `lazy`, it allows one call that starts at a
-- trigger; the sampler starts it there (see `lgml.sampler`).
-- @module lgml.chat.tool_grammar
local grammar = require("lgml.grammar")
local json_schema = require("lgml.json_schema")
local call = require("lgml.chat.qwen35_call")

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
-- @tparam[opt] table opts `parallel`: allow more than one call; `lazy`:
-- one call from a trigger
-- @treturn[1] string the GBNF text
-- @treturn[1] string the start rule
-- @treturn[1] {string,...} the triggers: `<tool_call>`
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
	return text .. calls(opts and opts.parallel and not opts.lazy) ..
	    'lgml-call ::= "<tool_call>\\n" root "\\n</tool_call>"\n', "lgml-tools", { "<tool_call>" }
end

-- Property names of a schema object, in source order when the decoder
-- kept it, else sorted.
local function prop_keys(props)
	local mt = getmetatable(props)
	local order = mt and type(mt.__order) == "table" and mt.__order
	local list = {}
	if order then
		for _, k in ipairs(order) do
			if type(k) == "string" and props[k] ~= nil then
				list[#list + 1] = k
			end
		end
		return list
	end
	for k in pairs(props) do
		if type(k) == "string" then
			list[#list + 1] = k
		end
	end
	table.sort(list)
	return list
end

-- The rule of required args in any order, as llama.cpp permute builds
-- it: one rule per set of args left. More than 6 args stay in order.
local function permute(rules, prefix, args)
	if #args == 0 then
		return ""
	elseif #args == 1 or #args > 6 then
		return table.concat(args, " ")
	end
	local function left(mask)
		if mask == 0 then
			return ""
		end
		local name = ("%s-%d"):format(prefix, mask)
		if not rules[name] then
			rules[name] = true
			local alts = {}
			for i, a in ipairs(args) do
				local bit = 1 << (i - 1)
				if mask & bit ~= 0 then
					alts[#alts + 1] = (a .. " " .. left(mask & ~bit)):gsub("%s+$", "")
				end
			end
			rules[#rules + 1] = ("%s ::= %s\n"):format(name, table.concat(alts, " | "))
		end
		return name
	end
	return left((1 << #args) - 1)
end

-- The rules of the parameters of tool i; returns the args rule name.
-- A string value is raw text up to "\n</parameter>\n"; another value
-- follows the JSON schema of its parameter.
local function xml_params(rules, i, f)
	local params = type(f.parameters) == "table" and f.parameters or {}
	local props = type(params.properties) == "table" and params.properties or {}
	local req = {}
	for _, k in ipairs(type(params.required) == "table" and params.required or {}) do
		req[k] = true
	end
	local need, opt = {}, {}
	for j, k in ipairs(prop_keys(props)) do
		local s, arg = props[k], ("lgml-t%d-a%d"):format(i, j)
		local value = "lgml-value-0"
		if not call.is_string(s, params) then
			-- A copy with the defs of the tool, for local refs.
			local sub = {}
			if type(s) == "table" then
				for key, v in pairs(s) do
					sub[key] = v
				end
				setmetatable(sub, getmetatable(s))
			end
			for _, d in ipairs({ "$defs", "definitions" }) do
				if sub[d] == nil then
					sub[d] = params[d]
				end
			end
			local text, err = json_schema.to_grammar(sub)
			if not text then
				return nil, err
			end
			rules[#rules + 1] = grammar.rename(text, arg .. "-")
			value = arg .. '-root "\\n</parameter>\\n"'
		end
		rules[#rules + 1] = ('%s ::= "<parameter=" %s ">\\n" %s\n'):format(arg, grammar.literal(k), value)
		local list = req[k] and need or opt
		list[#list + 1] = arg
	end
	local name = ("lgml-t%d-args"):format(i)
	local body = permute(rules, name .. "-req", need)
	if #opt > 0 then
		body = body .. " (" .. table.concat(opt, " | ") .. ")*"
	end
	rules[#rules + 1] = ("%s ::= %s\n"):format(name, body == "" and '""' or body)
	return name
end

--- The grammar of Qwen3.5 XML calls:
-- `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>`.
-- As in llama.cpp, the required parameters come once each in any
-- order, then optional ones. A string value is free text; another
-- value follows the parameter schema. The lazy grammar also takes a
-- call without `<tool_call>\n` and `</tool_call>` (Qwen3-Coder).
-- @tparam {table,...} tools OpenAI function tools
-- @tparam[opt] table opts `parallel`: allow more than one call; `lazy`:
-- one call from a trigger
-- @treturn[1] string the GBNF text
-- @treturn[1] string the start rule
-- @treturn[1] {string,...} the triggers: `<tool_call>`, then
-- `<function=NAME>` for each tool
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.qwen35(tools, opts)
	local rules, alts, triggers = {}, {}, { "<tool_call>" }
	for i, t in ipairs(tools) do
		local f = fn(t)
		triggers[#triggers + 1] = "<function=" .. f.name .. ">"
		local args, err = xml_params(rules, i, f)
		if not args then
			return nil, err
		end
		alts[i] = ("%s %s %s"):format(grammar.literal(f.name), grammar.literal(">\n"), args)
	end
	rules[#rules + 1] = grammar.until_text("lgml-value", "\n</parameter>\n")
	rules[#rules + 1] = ('lgml-body ::= "<function=" (%s) "</function>\\n"\n'):format(table.concat(alts, " | "))
	if opts and opts.lazy then
		rules[#rules + 1] = 'lgml-call ::= "<tool_call>\\n" lgml-body "</tool_call>" | lgml-body\n'
	else
		rules[#rules + 1] = 'lgml-call ::= "<tool_call>\\n" lgml-body "</tool_call>"\n'
	end
	return table.concat(rules) .. calls(opts and opts.parallel and not opts.lazy), "lgml-tools", triggers
end

--- The grammar of Gemma 4 calls: `<|tool_call>call:NAME{...}<tool_call|>`.
-- The text in the braces is free.
-- @tparam {table,...} tools OpenAI function tools
-- @tparam[opt] table opts `parallel`: allow more than one call; `lazy`:
-- one call from a trigger
-- @treturn string the GBNF text
-- @treturn string the start rule
-- @treturn {string,...} the triggers: `<|tool_call>`
function M.gemma4(tools, opts)
	local rules = {
		('lgml-call ::= "<|tool_call>call:" %s "{" lgml-args-0\n'):format(names(tools)),
		grammar.until_text("lgml-args", "}<tool_call|>"),
	}
	return table.concat(rules) .. calls(opts and opts.parallel and not opts.lazy), "lgml-tools",
	    { "<|tool_call>" }
end

return M
