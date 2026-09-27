--- JSON schema to GBNF grammar.
-- A port of llama.cpp common/json-schema-to-grammar.cpp. The output
-- matches llama.cpp for the same schema and key order.
--
-- Object key order comes from the metatable field `__order` (as
-- `decode` gives). Keys of other tables go in sorted order. Arrays
-- carry the `lgml.json` array metatable, or are plain sequences.
-- Null is `lgml.json.null`.
--
-- Only local `$ref` pointers ("#/...") resolve. Nesting depth, visit
-- count and grammar size have limits, so a hostile schema gives an
-- error, not a hang.
-- @module lgml.json_schema
local json = require("lgml.json")

local M = {}

local null = json.null
local MAX_DEPTH = 200
local MAX_OUT = 1 << 20
local MAX_VISITS = 100000
local INT_MAX = 0x7fffffff
local I64_MIN, I64_MAX = math.mininteger, math.maxinteger

local SPACE_RULE = [[| " " | "\n"{1,2} [ \t]{0,20}]]

-- Each rule: { content, deps }.
local PRIMITIVE_RULES = {
	boolean = { [[("true" | "false")]], {} },
	["decimal-part"] = { "[0-9]{1,16}", {} },
	["integral-part"] = { "[0] | [1-9] [0-9]{0,15}", {} },
	number = {
		[[("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?]],
		{ "integral-part", "decimal-part" },
	},
	integer = { [[("-"? integral-part)]], { "integral-part" } },
	value = {
		"object | array | string | number | boolean | null",
		{ "object", "array", "string", "number", "boolean", "null" },
	},
	object = {
		[["{" space ( string ":" space value ("," space string ":" space value)* )? space "}"]],
		{ "string", "value" },
	},
	array = { [["[" space ( value ("," space value)* )? space "]"]], { "value" } },
	uuid = {
		[["\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" ]]
			.. [=[[0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\""]=],
		{},
	},
	char = { [=[[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})]=], {} },
	string = { [["\"" char* "\""]], { "char" } },
	null = { [["null"]], {} },
}

local STRING_FORMAT_RULES = {
	date = {
		[=[[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )]=],
		{},
	},
	time = {
		[=[([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ]=]
			.. [=[( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )]=],
		{},
	},
	["date-time"] = { [[date "T" time]], { "date", "time" } },
	["date-string"] = { [["\"" date "\""]], { "date" } },
	["time-string"] = { [["\"" time "\""]], { "time" } },
	["date-time-string"] = { [["\"" date-time "\""]], { "date-time" } },
}

local RESERVED = { root = true }
for k in pairs(PRIMITIVE_RULES) do
	RESERVED[k] = true
end
for k in pairs(STRING_FORMAT_RULES) do
	RESERVED[k] = true
end

local function set(s)
	local t = {}
	for c in s:gmatch(".") do
		t[c] = true
	end
	return t
end

local NON_LITERAL = set("|.()[]{}*+?")
local ESCAPED_IN_REGEXPS = set("^$.[](){}|*+?")

-- Byte order, as std::map<std::string> sorts.
local function before(a, b)
	for i = 1, math.min(#a, #b) do
		local x, y = a:byte(i), b:byte(i)
		if x ~= y then
			return x < y
		end
	end
	return #a < #b
end

local function kind_of(v)
	if v == nil or v == null then
		return "null"
	end
	local t = type(v)
	if t == "table" then
		if json.isarray(v) then
			return "array"
		end
		local mt = getmetatable(v)
		if mt and mt.__order then
			return "object"
		end
		local n = #v
		if n > 0 then
			local c = 0
			for _ in pairs(v) do
				c = c + 1
			end
			if c == n then
				return "array"
			end
		end
		return "object"
	elseif t == "number" then
		return math.type(v) == "integer" and "integer" or "float"
	end
	return t
end

-- Object keys: `__order` first, then the rest in byte order.
local function keys(t)
	local mt = getmetatable(t)
	local order = type(mt) == "table" and mt.__order
	local out, seen = {}, {}
	if type(order) == "table" then
		for _, k in ipairs(order) do
			if t[k] ~= nil and not seen[k] then
				seen[k] = true
				out[#out + 1] = k
			end
		end
	end
	local rest = {}
	for k in pairs(t) do
		if not seen[k] then
			rest[#rest + 1] = k
		end
	end
	table.sort(rest, function(a, b)
		return before(tostring(a), tostring(b))
	end)
	table.move(rest, 1, #rest, #out + 1, out)
	return out
end

-- Float text as nlohmann::json dump writes it.
local function dump_float(x)
	if x ~= x or x == math.huge or x == -math.huge then
		return "null"
	end
	local sign = (x < 0 or 1 / x < 0) and "-" or ""
	if x == 0 then
		return sign .. "0.0"
	end
	x = math.abs(x)
	local d, n
	for p = 1, 17 do
		local s = string.format("%." .. (p - 1) .. "e", x)
		if tonumber(s) == x then
			local m, e = s:match("^(%d[%.%d]*)e([-+]%d+)$")
			d, n = m:gsub("%.", ""):gsub("0+$", ""), tonumber(e) + 1
			break
		end
	end
	local k = #d
	if k <= n and n <= 15 then
		return sign .. d .. string.rep("0", n - k) .. ".0"
	elseif 0 < n and n <= 15 then
		return sign .. d:sub(1, n) .. "." .. d:sub(n + 1)
	elseif -4 < n and n <= 0 then
		return sign .. "0." .. string.rep("0", -n) .. d
	end
	local m = k == 1 and d or d:sub(1, 1) .. "." .. d:sub(2)
	local e = n - 1
	return string.format("%s%se%s%02d", sign, m, e < 0 and "-" or "+", math.abs(e))
end

local dump_escapes = {
	['"'] = '\\"',
	["\\"] = "\\\\",
	["\b"] = "\\b",
	["\f"] = "\\f",
	["\n"] = "\\n",
	["\r"] = "\\r",
	["\t"] = "\\t",
}

local function dump_string(s)
	if not utf8.len(s) then
		error({ fatal = "invalid UTF-8 in string" }, 0)
	end
	return '"' .. s:gsub('[\0-\31"\\]', function(c)
		return dump_escapes[c] or string.format("\\u%04x", c:byte())
	end) .. '"'
end

-- Compact JSON as nlohmann::ordered_json dump. It raises
-- { toolong = true } past limit bytes.
local function dump(v, limit)
	local buf, size = {}, 0
	local function put(s)
		size = size + #s
		if size > limit then
			error({ toolong = true }, 0)
		end
		buf[#buf + 1] = s
	end
	local function go(x, depth)
		if depth > MAX_DEPTH then
			error({ fatal = "value nesting too deep" }, 0)
		end
		local k = kind_of(x)
		if k == "null" then
			put("null")
		elseif k == "boolean" then
			put(tostring(x))
		elseif k == "integer" then
			put(tostring(x))
		elseif k == "float" then
			put(dump_float(x))
		elseif k == "string" then
			put(dump_string(x))
		elseif k == "array" then
			put("[")
			for i = 1, #x do
				if i > 1 then
					put(",")
				end
				go(x[i], depth + 1)
			end
			put("]")
		elseif k == "object" then
			put("{")
			for i, key in ipairs(keys(x)) do
				if i > 1 then
					put(",")
				end
				put(dump_string(tostring(key)))
				put(":")
				go(x[key], depth + 1)
			end
			put("}")
		else
			error({ fatal = "cannot encode a " .. k }, 0)
		end
	end
	go(v, 0)
	return table.concat(buf)
end

-- Dump for an error message, cut to 256 bytes.
local function brief(v)
	local ok, s = pcall(dump, v, 256)
	if ok then
		return s
	end
	return "..."
end

local literal_escapes = { ["\r"] = "\\r", ["\n"] = "\\n", ['"'] = '\\"', ["\\"] = "\\\\" }

local function format_literal(s)
	return '"' .. s:gsub('[\r\n"\\]', literal_escapes) .. '"'
end

local function build_repetition(item, min, max, sep)
	local has_max = max ~= INT_MAX
	if max == 0 then
		return ""
	end
	if min == 0 and max == 1 then
		return item .. "?"
	end
	if sep == nil or sep == "" then
		if min == 1 and not has_max then
			return item .. "+"
		end
		if min == 0 and not has_max then
			return item .. "*"
		end
		return item .. "{" .. min .. "," .. (has_max and tostring(max) or "") .. "}"
	end
	local r = item .. " " .. build_repetition("(" .. sep .. " " .. item .. ")",
		min == 0 and 0 or min - 1, has_max and max - 1 or max)
	if min == 0 then
		r = "(" .. r .. ")?"
	end
	return r
end

-- Integer range rules; appends to out.
local function build_min_max_int(minv, maxv, out, decimals_left, top_level)
	local has_min = minv ~= I64_MIN
	local has_max = maxv ~= I64_MAX
	local function w(s)
		out[#out + 1] = s
	end
	local function digit_range(from, to)
		if from == to then
			w("[" .. string.char(from) .. "]")
		else
			w("[" .. string.char(from) .. "-" .. string.char(to) .. "]")
		end
	end
	local function more_digits(lo, hi)
		w("[0-9]")
		if lo == hi and lo == 1 then
			return
		end
		w("{" .. lo)
		if hi ~= lo then
			w(",")
			if hi ~= INT_MAX then
				w(tostring(hi))
			end
		end
		w("}")
	end
	local function uniform_range(from, to)
		local i = 0
		while i < #from and i < #to and from:byte(i + 1) == to:byte(i + 1) do
			i = i + 1
		end
		if i > 0 then
			w('"' .. from:sub(1, i) .. '"')
		end
		if i >= #from or i >= #to then
			return
		end
		if i > 0 then
			w(" ")
		end
		local fi, ti = from:byte(i + 1), to:byte(i + 1)
		local sub_len = #from - i - 1
		if sub_len == 0 then
			w("[" .. string.char(fi) .. "-" .. string.char(ti) .. "]")
			return
		end
		local from_sub, to_sub = from:sub(i + 2), to:sub(i + 2)
		local zeros, nines = string.rep("0", sub_len), string.rep("9", sub_len)
		local reached = false
		w("(")
		if from_sub == zeros then
			digit_range(fi, ti - 1)
			w(" ")
			more_digits(sub_len, sub_len)
		else
			w("[" .. string.char(fi) .. "] (")
			uniform_range(from_sub, nines)
			w(")")
			if fi < ti - 1 then
				w(" | ")
				if to_sub == nines then
					digit_range(fi + 1, ti)
					reached = true
				else
					digit_range(fi + 1, ti - 1)
				end
				w(" ")
				more_digits(sub_len, sub_len)
			end
		end
		if not reached then
			w(" | ")
			digit_range(ti, ti)
			w(" ")
			uniform_range(zeros, to_sub)
		end
		w(")")
	end

	if has_min and has_max then
		if minv < 0 and maxv < 0 then
			w('"-" (')
			build_min_max_int(-maxv, -minv, out, decimals_left, true)
			w(")")
			return
		end
		if minv < 0 then
			w('"-" (')
			build_min_max_int(0, -minv, out, decimals_left, true)
			w(") | ")
			minv = 0
		end
		local min_s, max_s = tostring(minv), tostring(maxv)
		for digits = #min_s, #max_s - 1 do
			uniform_range(min_s, string.rep("9", digits))
			min_s = "1" .. string.rep("0", digits)
			w(" | ")
		end
		uniform_range(min_s, max_s)
		return
	end

	local less = math.max(decimals_left - 1, 1)
	if has_min then
		if minv < 0 then
			w('"-" (')
			build_min_max_int(I64_MIN, -minv, out, decimals_left, false)
			w(") | [0] | [1-9] ")
			more_digits(0, decimals_left - 1)
		elseif minv == 0 then
			if top_level then
				w("[0] | [1-9] ")
				more_digits(0, less)
			else
				more_digits(1, decimals_left)
			end
		elseif minv <= 9 then
			local c = 48 + minv
			local start = top_level and 49 or 48
			if c > start then
				digit_range(start, c - 1)
				w(" ")
				more_digits(1, less)
				w(" | ")
			end
			digit_range(c, 57)
			w(" ")
			more_digits(0, less)
		else
			local min_s = tostring(minv)
			local len, c = #min_s, min_s:byte(1)
			if c > 49 then
				digit_range(top_level and 49 or 48, c - 1)
				w(" ")
				more_digits(len, less)
				w(" | ")
			end
			digit_range(c, c)
			w(" (")
			build_min_max_int(math.tointeger(tonumber(min_s:sub(2))), I64_MAX, out, less, false)
			w(")")
			if c < 57 then
				w(" | ")
				digit_range(c + 1, 57)
				w(" ")
				more_digits(len - 1, less)
			end
		end
		return
	end

	if has_max then
		if maxv >= 0 then
			if top_level then
				w('"-" [1-9] ')
				more_digits(0, less)
				w(" | ")
			end
			build_min_max_int(0, maxv, out, decimals_left, true)
		else
			w('"-" (')
			build_min_max_int(-maxv, I64_MAX, out, decimals_left, false)
			w(")")
		end
		return
	end
	error({ fatal = "At least one of min_value or max_value must be set" }, 0)
end

local function split(s, sep)
	local parts, start = {}, 1
	while true do
		local a = s:find(sep, start, true)
		if not a then
			break
		end
		parts[#parts + 1] = s:sub(start, a - 1)
		start = a + #sep
	end
	parts[#parts + 1] = s:sub(start)
	return parts
end

-- std::stoi: leading space, sign, digits, then anything.
local function stoi(s)
	local d = s:match("^%s*([-+]?%d+)")
	local v = d and math.tointeger(tonumber(d))
	if v and v >= -INT_MAX - 1 and v <= INT_MAX then
		return v
	end
end

-- Signed char order, as std::map<char> sorts on x86.
local function char_before(a, b)
	local x, y = a:byte(), b:byte()
	x = x >= 128 and x - 256 or x
	y = y >= 128 and y - 256 or y
	return x < y
end

local function convert(schema)
	local rules = { space = SPACE_RULE }
	local size = #"space ::= \n" + #SPACE_RULE
	local free, where = {}, {}
	local refs, resolving = {}, {}
	local errors, warnings = {}, {}
	local visits = 0
	local kinds = {}

	local function fatal(msg)
		error({ fatal = msg }, 0)
	end

	local function kind(v)
		if type(v) ~= "table" then
			return kind_of(v)
		end
		local k = kinds[v]
		if not k then
			k = kind_of(v)
			kinds[v] = k
		end
		return k
	end

	-- Elements as a C++ range-for over a json value sees them.
	local function elems(v)
		local k = kind(v)
		if k == "array" then
			return v
		elseif k == "object" then
			local out = {}
			for i, key in ipairs(keys(v)) do
				out[i] = v[key]
			end
			return out
		elseif k == "null" then
			return {}
		end
		return { v }
	end

	-- Key and value pairs as json::items() gives them.
	local function items(v)
		local k, out = kind(v), {}
		if k == "object" then
			for i, key in ipairs(keys(v)) do
				out[i] = { tostring(key), v[key] }
			end
		elseif k == "array" then
			for i = 1, #v do
				out[i] = { tostring(i - 1), v[i] }
			end
		elseif k ~= "null" then
			out[1] = { "", v }
		end
		return out
	end

	local function constant(v)
		local ok, s = pcall(dump, v, MAX_OUT)
		if not ok then
			if type(s) == "table" and s.toolong then
				fatal("grammar is larger than " .. MAX_OUT .. " bytes")
			end
			error(s, 0)
		end
		return format_literal(s)
	end

	-- where[base][rule] is the least i with rules[base .. i] == rule.
	local function store(key, rule)
		if rules[key] ~= nil then
			return
		end
		size = size + #key + #rule + 6
		if size > MAX_OUT then
			fatal("grammar is larger than " .. MAX_OUT .. " bytes")
		end
		rules[key] = rule
		local p = #key
		while p > 0 and p > #key - 7 and key:byte(p) >= 48 and key:byte(p) <= 57 do
			p = p - 1
		end
		for s = p + 1, #key do
			local d = key:sub(s)
			if d == "0" or d:byte(1) ~= 48 then
				local base, j = key:sub(1, s - 1), tonumber(d)
				local w = where[base]
				if not w then
					w = {}
					where[base] = w
				end
				if w[rule] == nil or j < w[rule] then
					w[rule] = j
				end
			end
		end
	end

	-- Same result as the linear search in llama.cpp _add_rule.
	local function add_rule(name, rule)
		local esc = name:gsub("[^a-zA-Z0-9%-]+", "-")
		local cur = rules[esc]
		if cur == nil or cur == rule then
			store(esc, rule)
			return esc
		end
		local f = free[esc] or 0
		while rules[esc .. f] ~= nil do
			f = f + 1
		end
		free[esc] = f
		local e = where[esc] and where[esc][rule]
		local key = esc .. ((e and e < f) and e or f)
		store(key, rule)
		return key
	end

	local function add_primitive(name, rule)
		local n = add_rule(name, rule[1])
		for _, dep in ipairs(rule[2]) do
			local r = PRIMITIVE_RULES[dep] or STRING_FORMAT_RULES[dep]
			if not r then
				errors[#errors + 1] = "Rule " .. dep .. " not known"
			elseif rules[dep] == nil then
				add_primitive(dep, r)
			end
		end
		return n
	end

	local function to_int(v, what)
		local k = kind(v)
		if k == "boolean" then
			return v and 1 or 0
		elseif k == "integer" or k == "float" then
			local t = v >= 0 and math.floor(v) or math.ceil(v)
			t = math.tointeger(t)
			if t then
				return math.max(-INT_MAX - 1, math.min(INT_MAX, t))
			end
		end
		errors[#errors + 1] = what .. " must be a number: " .. brief(v)
		return 0
	end

	local function to_i64(v, what)
		local k = kind(v)
		if k == "boolean" then
			return v and 1 or 0
		elseif k == "integer" then
			return v
		elseif k == "float" and v > -2^63 and v < 2^63 then
			return math.tointeger(v >= 0 and math.floor(v) or math.ceil(v))
		end
		fatal(what .. " must be a number: " .. brief(v))
	end

	local function visit_pattern(pattern, name)
		if type(pattern) ~= "string" then
			errors[#errors + 1] = "Pattern must be a string"
			return ""
		end
		if not (pattern:sub(1, 1) == "^" and pattern:sub(-1) == "$") then
			errors[#errors + 1] = "Pattern must start with '^' and end with '$'"
			return ""
		end
		local sp = pattern:sub(2, -2)
		local len = #sp
		local i = 0
		local sub_ids, nsub = {}, 0
		local function at(k)
			return sp:sub(k + 1, k + 1)
		end
		local function to_rule(s, lit)
			return lit and '"' .. s .. '"' or s
		end
		-- The pattern matches the JSON string content, so a dot and a
		-- negated class leave out the quote, the backslash and the
		-- control characters. Else the grammar allows invalid JSON.
		local UNSAFE = '\\"\\\\\\x00-\\x1F'
		local function get_dot()
			return add_rule("dot", "[^" .. UNSAFE .. "]")
		end

		local function transform(depth)
			if depth > MAX_DEPTH then
				fatal("pattern nesting too deep")
			end
			local start = i
			local seq = {}

			-- Join the sequence; merge runs of literals.
			local function join_seq()
				local out, lit = {}, {}
				for _, item in ipairs(seq) do
					if item[2] then
						lit[#lit + 1] = item[1]
					else
						if #lit > 0 then
							out[#out + 1] = '"' .. table.concat(lit) .. '"'
							lit = {}
						end
						out[#out + 1] = item[1]
					end
				end
				if #lit > 0 then
					out[#out + 1] = '"' .. table.concat(lit) .. '"'
				end
				return table.concat(out, " "), false
			end

			while i < len do
				local c = at(i)
				if c == "." then
					seq[#seq + 1] = { get_dot(), false }
					i = i + 1
				elseif c == "(" then
					i = i + 1
					local skip = false
					if i < len and at(i) == "?" then
						if i + 1 < len and at(i + 1) == ":" then
							i = i + 2
						else
							-- Lookaround: skip to the matching ')'.
							warnings[#warnings + 1] = "Unsupported pattern syntax"
							local d = 1
							while i < len and d > 0 do
								if at(i) == "\\" and i + 1 < len then
									i = i + 2
								else
									if at(i) == "(" then
										d = d + 1
									elseif at(i) == ")" then
										d = d - 1
									end
									i = i + 1
								end
							end
							skip = true
						end
					end
					if not skip then
						seq[#seq + 1] = { "(" .. to_rule(transform(depth + 1)) .. ")", false }
					end
				elseif c == ")" then
					i = i + 1
					if start > 0 and at(start - 1) ~= "("
						and (start < 2 or at(start - 2) ~= "?" or at(start - 1) ~= ":") then
						errors[#errors + 1] = "Unbalanced parentheses"
					end
					return join_seq()
				elseif c == "[" then
					local b = { "[" }
					i = i + 1
					while i < len and at(i) ~= "]" do
						if at(i) == "\\" then
							b[#b + 1] = sp:sub(i + 1, i + 2)
							i = i + 2
						else
							b[#b + 1] = at(i)
							i = i + 1
						end
					end
					if i >= len then
						errors[#errors + 1] = "Unbalanced square brackets"
					end
					if b[2] == "^" then
						b[#b + 1] = UNSAFE
					end
					b[#b + 1] = "]"
					i = i + 1
					seq[#seq + 1] = { table.concat(b), false }
				elseif c == "|" then
					seq[#seq + 1] = { "|", false }
					i = i + 1
				elseif c == "*" or c == "+" or c == "?" then
					local last = seq[#seq]
					if last then
						seq[#seq] = { to_rule(last[1], last[2]) .. c, false }
					else
						errors[#errors + 1] = "Quantifier without operand in pattern"
					end
					i = i + 1
				elseif c == "{" then
					local b = { "{" }
					i = i + 1
					while i < len and at(i) ~= "}" do
						b[#b + 1] = at(i)
						i = i + 1
					end
					if i >= len then
						errors[#errors + 1] = "Unbalanced curly brackets"
					end
					b[#b + 1] = "}"
					i = i + 1
					local curly = table.concat(b)
					local nums = split(curly:sub(2, -2), ",")
					local min_times, max_times = 0, INT_MAX
					if #nums == 1 then
						min_times = stoi(nums[1])
						max_times = min_times
					elseif #nums ~= 2 then
						errors[#errors + 1] = "Wrong number of values in curly brackets"
					else
						if nums[1] ~= "" then
							min_times = stoi(nums[1])
						end
						if nums[2] ~= "" then
							max_times = stoi(nums[2])
						end
					end
					if min_times == nil or max_times == nil then
						errors[#errors + 1] = "Invalid number in curly brackets"
						return "", false
					end
					local last = seq[#seq]
					if not last then
						errors[#errors + 1] = "Quantifier without operand in pattern"
					else
						local sub = last[1]
						if last[2] then
							sub = '"' .. sub .. '"'
						else
							local id = sub_ids[sub]
							if not id then
								nsub = nsub + 1
								id = add_rule(name .. "-" .. nsub, sub)
								sub_ids[sub] = id
							end
							sub = id
						end
						seq[#seq] = { build_repetition(sub, min_times, max_times, ""), false }
					end
				else
					local lit = {}
					while i < len do
						local ch = at(i)
						if ch == "\\" and i < len - 1 then
							local nx = at(i + 1)
							if ESCAPED_IN_REGEXPS[nx] then
								lit[#lit + 1] = nx
							else
								lit[#lit + 1] = ch .. nx
							end
							i = i + 2
						elseif ch == '"' then
							-- An escaped quote in the JSON text.
							lit[#lit + 1] = '\\\\\\"'
							i = i + 1
						elseif not NON_LITERAL[ch] and (i == len - 1 or #lit == 0
							or at(i + 1) == "." or not NON_LITERAL[at(i + 1)]) then
							lit[#lit + 1] = ch
							i = i + 1
						else
							break
						end
					end
					if #lit > 0 then
						seq[#seq + 1] = { table.concat(lit), true }
					else
						-- A stray ']' or '}'.
						errors[#errors + 1] = "Unexpected '" .. c .. "' in pattern"
						i = i + 1
					end
				end
			end
			return join_seq()
		end
		return add_rule(name, '"\\"" (' .. to_rule(transform(0)) .. ') "\\""')
	end

	-- A string rule that matches none of strs.
	local function not_strings(strs)
		local trie = { kids = {} }
		for _, s in ipairs(strs) do
			local node = trie
			for k = 1, #s do
				local ch = s:sub(k, k)
				local nx = node.kids[ch]
				if not nx then
					nx = { kids = {} }
					node.kids[ch] = nx
				end
				node = nx
			end
			node.last = true
		end
		local char_rule = add_primitive("char", PRIMITIVE_RULES.char)
		local out = { '["] ( ' }
		local function walk(node)
			local ks = {}
			for ch in pairs(node.kids) do
				ks[#ks + 1] = ch
			end
			table.sort(ks, char_before)
			for idx, ch in ipairs(ks) do
				if idx > 1 then
					out[#out + 1] = " | "
				end
				out[#out + 1] = "[" .. ch .. "]"
				local kid = node.kids[ch]
				if next(kid.kids) then
					out[#out + 1] = " ("
					walk(kid)
					out[#out + 1] = ")"
				elseif kid.last then
					out[#out + 1] = " " .. char_rule .. "+"
				end
			end
			if #ks > 0 then
				out[#out + 1] = ' | [^"' .. table.concat(ks) .. "] " .. char_rule .. "*"
			end
		end
		walk(trie)
		out[#out + 1] = " )"
		if not trie.last then
			out[#out + 1] = "?"
		end
		out[#out + 1] = ' ["]'
		return table.concat(out)
	end

	local visit

	local function resolve_ref(ref, depth)
		local h = ref:find("#", 1, true)
		local frag = h and ref:sub(h + 1) or ref
		local ref_name = "ref" .. frag:gsub("[^a-zA-Z0-9%-]+", "-")
		if rules[ref_name] == nil and not resolving[ref] then
			resolving[ref] = true
			ref_name = visit(refs[ref], ref_name, depth)
			resolving[ref] = nil
		end
		return ref_name
	end

	local function build_object_rule(props, required, name, additional, depth)
		local req, opt, kv, names = {}, {}, {}, {}
		local pre = name == "" and "" or name .. "-"
		for _, p in ipairs(props) do
			local pname = p[1]
			local prule = visit(p[2], pre .. pname, depth)
			kv[pname] = add_rule(pre .. pname .. "-kv",
				format_literal(dump_string(pname)) .. ' space ":" space ' .. prule)
			if required[pname] then
				req[#req + 1] = pname
			else
				opt[#opt + 1] = pname
			end
			names[#names + 1] = pname
		end
		local ak = kind(additional)
		if (ak == "boolean" and additional) or ak == "object" then
			local sub = pre .. "additional"
			local value_rule = ak == "object" and visit(additional, sub .. "-value", depth)
				or add_primitive("value", PRIMITIVE_RULES.value)
			local key_rule = #names == 0 and add_primitive("string", PRIMITIVE_RULES.string)
				or add_rule(sub .. "-k", not_strings(names))
			kv["*"] = add_rule(sub .. "-kv", key_rule .. ' ":" space ' .. value_rule)
			opt[#opt + 1] = "*"
		end

		local r = { '"{" space ' }
		for i, k in ipairs(req) do
			if i > 1 then
				r[#r + 1] = ' "," space '
			end
			r[#r + 1] = kv[k]
		end
		if #opt > 0 then
			r[#r + 1] = " ("
			if #req > 0 then
				r[#r + 1] = ' "," space ( '
			end
			local function piece(j, first_optional)
				local k = opt[j]
				local comma = '( "," space ' .. kv[k] .. " )"
				if first_optional then
					return comma .. (k == "*" and "*" or "?")
				end
				return kv[k] .. (k == "*" and " " .. comma .. "*" or "")
			end
			-- tail[j] names the rule for opt[j+1..]. Build the last one first.
			local tail = {}
			for j = #opt - 1, 1, -1 do
				local body = piece(j + 1, true)
				if tail[j + 1] then
					body = body .. " " .. tail[j + 1]
				end
				tail[j] = add_rule(pre .. opt[j] .. "-rest", body)
			end
			for i = 1, #opt do
				if i > 1 then
					r[#r + 1] = " | "
				end
				r[#r + 1] = piece(i, false) .. (tail[i] and " " .. tail[i] or "")
			end
			if #req > 0 then
				r[#r + 1] = " )"
			end
			r[#r + 1] = " )?"
		end
		r[#r + 1] = ' space "}"'
		return table.concat(r)
	end

	local function union_rule(name, alts, depth)
		local out = {}
		for i, alt in ipairs(alts) do
			out[i] = visit(alt, name .. (name == "" and "alternative-" or "-") .. (i - 1), depth)
		end
		return table.concat(out, " | ")
	end

	local EMPTY = {}

	function visit(s, name, depth)
		depth = depth + 1
		if depth > MAX_DEPTH then
			fatal("schema nesting too deep")
		end
		visits = visits + 1
		if visits > MAX_VISITS then
			fatal("schema needs more than " .. MAX_VISITS .. " steps")
		end
		local sk = kind(s)
		local o = sk == "object" and s or EMPTY
		local stype = o.type
		local tk = kind(stype)
		local tnull = tk == "null"
		local fmt = ""
		if o.format ~= nil then
			if type(o.format) == "string" then
				fmt = o.format
			else
				errors[#errors + 1] = "format must be a string: " .. brief(o.format)
			end
		end
		local rule_name = RESERVED[name] and name .. "-" or (name == "" and "root" or name)

		if o["$ref"] ~= nil then
			if type(o["$ref"]) ~= "string" then
				errors[#errors + 1] = "Unsupported ref: " .. brief(o["$ref"])
				return ""
			end
			return add_rule(rule_name, resolve_ref(o["$ref"], depth))
		end
		if o.oneOf ~= nil or o.anyOf ~= nil then
			local alts = o.oneOf
			if alts == nil then
				alts = o.anyOf
			end
			if kind(alts) ~= "array" then
				errors[#errors + 1] = "oneOf and anyOf must be arrays: " .. brief(alts)
				return ""
			end
			return add_rule(rule_name, union_rule(name, alts, depth))
		end
		if tk == "array" then
			local alts = {}
			for i, t in ipairs(stype) do
				local copy = setmetatable({}, getmetatable(s))
				for k, v in pairs(s) do
					copy[k] = v
				end
				copy.type = t
				kinds[copy] = "object"
				alts[i] = copy
			end
			return add_rule(rule_name, union_rule(name, alts, depth))
		end
		if o.const ~= nil then
			return add_rule(rule_name, constant(o.const))
		end
		if o.enum ~= nil then
			local vals = {}
			for i, v in ipairs(elems(o.enum)) do
				vals[i] = constant(v)
			end
			return add_rule(rule_name, "(" .. table.concat(vals, " | ") .. ")")
		end
		if (tnull or stype == "object")
			and (o.properties ~= nil or (o.additionalProperties ~= nil and o.additionalProperties ~= true)) then
			local required = {}
			if kind(o.required) == "array" then
				for _, v in ipairs(o.required) do
					if type(v) == "string" then
						required[v] = true
					end
				end
			end
			local props = o.properties ~= nil and items(o.properties) or {}
			return add_rule(rule_name, build_object_rule(props, required, name, o.additionalProperties, depth))
		end
		if (tnull or stype == "object" or stype == "string") and o.allOf ~= nil then
			local required, props = {}, {}
			local counts, active = {}, {}
			local function add_component(comp, is_required, cdepth)
				if cdepth > MAX_DEPTH then
					fatal("allOf nesting too deep")
				end
				if kind(comp) ~= "object" then
					return
				end
				if comp["$ref"] ~= nil then
					local ref = comp["$ref"]
					if active[ref] then
						errors[#errors + 1] = "Recursive ref in allOf: " .. brief(ref)
						return
					end
					active[ref] = true
					add_component(refs[ref], is_required, cdepth + 1)
					active[ref] = nil
				elseif comp.properties ~= nil then
					for _, p in ipairs(items(comp.properties)) do
						props[#props + 1] = p
						if is_required then
							required[p[1]] = true
						end
					end
				elseif comp.enum ~= nil then
					for _, v in ipairs(elems(comp.enum)) do
						local r = constant(v)
						counts[r] = (counts[r] or 0) + 1
					end
				end
			end
			local all = elems(o.allOf)
			for _, t in ipairs(all) do
				if kind(t) == "object" and t.anyOf ~= nil then
					for _, tt in ipairs(elems(t.anyOf)) do
						add_component(tt, false, depth)
					end
				else
					add_component(t, true, depth)
				end
			end
			if next(counts) then
				local hit = {}
				for r, n in pairs(counts) do
					if n == #all then
						hit[#hit + 1] = r
					end
				end
				if #hit > 0 then
					table.sort(hit, before)
					return add_rule(rule_name, "(" .. table.concat(hit, " | ") .. ")")
				end
			end
			return add_rule(rule_name, build_object_rule(props, required, name, nil, depth))
		end
		if (tnull or stype == "array") and (o.items ~= nil or o.prefixItems ~= nil) then
			local its = o.items
			if its == nil then
				its = o.prefixItems
			end
			local pre = name == "" and "" or name .. "-"
			if kind(its) == "array" then
				local r = { '"[" space ' }
				for i, it in ipairs(its) do
					if i > 1 then
						r[#r + 1] = ' "," space '
					end
					r[#r + 1] = visit(it, pre .. "tuple-" .. (i - 1), depth)
				end
				r[#r + 1] = ' space "]"'
				return add_rule(rule_name, table.concat(r))
			end
			local item_rule = visit(its, pre .. "item", depth)
			local min_items = o.minItems ~= nil and to_int(o.minItems, "minItems") or 0
			local max_items = INT_MAX
			if kind(o.maxItems) == "integer" then
				max_items = math.max(-INT_MAX - 1, math.min(INT_MAX, o.maxItems))
			end
			return add_rule(rule_name, '"[" space '
				.. build_repetition(item_rule, min_items, max_items, '"," space') .. ' space "]"')
		end
		if (tnull or stype == "string") and o.pattern ~= nil then
			return visit_pattern(o.pattern, rule_name)
		end
		if (tnull or stype == "string") and (fmt == "uuid" or fmt:match("^uuid[1-5]$")) then
			return add_primitive(rule_name == "root" and "root" or fmt, PRIMITIVE_RULES.uuid)
		end
		if (tnull or stype == "string") and STRING_FORMAT_RULES[fmt .. "-string"] then
			local prim = fmt .. "-string"
			return add_rule(rule_name, add_primitive(prim, STRING_FORMAT_RULES[prim]))
		end
		if stype == "string" and (o.minLength ~= nil or o.maxLength ~= nil) then
			local char_rule = add_primitive("char", PRIMITIVE_RULES.char)
			local min_len = o.minLength ~= nil and to_int(o.minLength, "minLength") or 0
			local max_len = o.maxLength ~= nil and to_int(o.maxLength, "maxLength") or INT_MAX
			return add_rule(rule_name, '"\\"" ' .. build_repetition(char_rule, min_len, max_len) .. ' "\\""')
		end
		if stype == "integer" and (o.minimum ~= nil or o.exclusiveMinimum ~= nil
			or o.maximum ~= nil or o.exclusiveMaximum ~= nil) then
			local minv, maxv = I64_MIN, I64_MAX
			if o.minimum ~= nil then
				minv = to_i64(o.minimum, "minimum")
			elseif o.exclusiveMinimum ~= nil then
				minv = to_i64(o.exclusiveMinimum, "exclusiveMinimum") + 1
			end
			if o.maximum ~= nil then
				maxv = to_i64(o.maximum, "maximum")
			elseif o.exclusiveMaximum ~= nil then
				maxv = to_i64(o.exclusiveMaximum, "exclusiveMaximum") - 1
			end
			local out = { "(" }
			build_min_max_int(minv, maxv, out, 16, true)
			out[#out + 1] = ")"
			return add_rule(rule_name, table.concat(out))
		end
		local empty = sk == "null" or ((sk == "object" or sk == "array") and next(s) == nil)
		if empty or stype == "object" then
			return add_rule(rule_name, add_primitive("object", PRIMITIVE_RULES.object))
		end
		if tnull and sk == "object" then
			return add_rule(rule_name, add_primitive("value", PRIMITIVE_RULES.value))
		end
		if type(stype) ~= "string" or not PRIMITIVE_RULES[stype] then
			errors[#errors + 1] = "Unrecognized schema: " .. brief(s)
			return ""
		end
		return add_primitive(rule_name == "root" and "root" or stype, PRIMITIVE_RULES[stype])
	end

	-- Record the target of each local $ref in refs.
	local function resolve_refs(root)
		local function walk(n, depth)
			if depth > MAX_DEPTH then
				fatal("schema nesting too deep")
			end
			local k = kind(n)
			if k == "array" then
				for i = 1, #n do
					walk(n[i], depth + 1)
				end
				return
			elseif k ~= "object" then
				return
			end
			local ref = n["$ref"]
			if ref == nil then
				for _, key in ipairs(keys(n)) do
					walk(n[key], depth + 1)
				end
				return
			end
			if type(ref) ~= "string" or ref:sub(1, 2) ~= "#/" then
				errors[#errors + 1] = "Unsupported ref: " .. (type(ref) == "string" and ref or brief(ref))
				return
			end
			if refs[ref] ~= nil then
				return
			end
			local target = root
			local toks = split(ref:sub(2), "/")
			for t = 2, #toks do
				local sel = toks[t]
				local tk = kind(target)
				if tk == "object" and target[sel] ~= nil then
					target = target[sel]
				elseif tk == "array" then
					local d = sel:match("^%s*%+?(%d+)")
					local idx = d and math.tointeger(tonumber(d)) or #target
					if idx >= #target then
						errors[#errors + 1] = "Error resolving ref " .. ref .. ": " .. sel .. " not in " .. brief(target)
						return
					end
					target = target[idx + 1]
				else
					errors[#errors + 1] = "Error resolving ref " .. ref .. ": " .. sel .. " not in " .. brief(target)
					return
				end
			end
			refs[ref] = target
		end
		walk(root, 0)
	end

	resolve_refs(schema)
	visit(schema, "", 0)
	if #errors > 0 then
		return nil, "JSON schema conversion failed:\n" .. table.concat(errors, "\n")
	end
	local names = {}
	for k in pairs(rules) do
		names[#names + 1] = k
	end
	table.sort(names, before)
	local out = {}
	for i, k in ipairs(names) do
		out[i] = k .. " ::= " .. rules[k] .. "\n"
	end
	return table.concat(out), warnings
end

--- Convert a JSON schema to a GBNF grammar.
-- The rules come in byte order of rule name, one per line, as
-- "name ::= rule". The start rule is `root`.
-- @tparam table schema the schema, from `decode` or `lgml.json.decode`
-- @treturn string the grammar text
-- @treturn {string,...} warnings, such as "Unsupported pattern syntax"
-- for a lookahead; the list can be empty
-- @return[2] nil on failure
-- @treturn[2] string the error message
function M.to_grammar(schema)
	local ok, text, warnings = pcall(convert, schema)
	if ok then
		return text, warnings
	end
	local msg = type(text) == "table" and text.fatal or tostring(text)
	return nil, "JSON schema conversion failed:\n" .. msg
end

local unescapes = {
	['"'] = '"',
	["\\"] = "\\",
	["/"] = "/",
	b = "\b",
	f = "\f",
	n = "\n",
	r = "\r",
	t = "\t",
}

--- Decode JSON text and keep the key order of objects.
-- Each object gets a metatable `{ __order = { "k1", "k2", ... } }`.
-- Arrays get the `lgml.json` array metatable. Null is `lgml.json.null`.
-- Integers decode as integers and other numbers as floats, so 1.0
-- stays a float. For a repeated key, the last value wins at the
-- first position. Nesting deeper than 200 is an error.
-- @tparam string text the JSON text
-- @treturn any the value
-- @return[2] nil on bad input
-- @treturn[2] string the error message
function M.decode(text)
	if type(text) ~= "string" then
		return nil, "json: text must be a string"
	end
	if not utf8.len(text) then
		return nil, "json: invalid UTF-8"
	end
	local pos = 1
	local function fail(msg)
		error({ msg = string.format("json: %s at byte %d", msg, pos) }, 0)
	end
	local function ws()
		pos = text:find("[^ \t\n\r]", pos) or #text + 1
	end
	local function hex4(p)
		local h = text:sub(p, p + 3)
		if not h:match("^%x%x%x%x$") then
			pos = p
			fail("bad \\u escape")
		end
		return tonumber(h, 16)
	end
	local function str()
		pos = pos + 1
		local buf = {}
		while true do
			local a = text:find('[\0-\31"\\]', pos)
			if not a then
				fail("unterminated string")
			end
			buf[#buf + 1] = text:sub(pos, a - 1)
			local c = text:byte(a)
			if c == 34 then
				pos = a + 1
				return table.concat(buf)
			elseif c ~= 92 then
				pos = a
				fail("control character in string")
			end
			local e = text:sub(a + 1, a + 1)
			if unescapes[e] then
				buf[#buf + 1] = unescapes[e]
				pos = a + 2
			elseif e == "u" then
				local cp = hex4(a + 2)
				pos = a + 6
				if cp >= 0xD800 and cp <= 0xDBFF then
					if text:sub(pos, pos + 1) ~= "\\u" then
						fail("lone surrogate")
					end
					local lo = hex4(pos + 2)
					if lo < 0xDC00 or lo > 0xDFFF then
						fail("bad surrogate pair")
					end
					cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
					pos = pos + 6
				elseif cp >= 0xDC00 and cp <= 0xDFFF then
					fail("lone surrogate")
				end
				buf[#buf + 1] = utf8.char(cp)
			else
				pos = a
				fail("bad escape")
			end
		end
	end
	local function number()
		local p = pos
		if text:byte(p) == 45 then
			p = p + 1
		end
		local d = text:match("^%d+", p)
		if not d or (#d > 1 and d:byte(1) == 48) then
			fail("bad number")
		end
		p = p + #d
		local float = false
		if text:byte(p) == 46 then
			local f = text:match("^%d+", p + 1)
			if not f then
				fail("bad number")
			end
			p = p + 1 + #f
			float = true
		end
		local c = text:byte(p)
		if c == 101 or c == 69 then
			local ex = text:match("^[-+]?%d+", p + 1)
			if not ex then
				fail("bad number")
			end
			p = p + 1 + #ex
			float = true
		end
		local v = tonumber(text:sub(pos, p - 1))
		if float then
			if v == math.huge or v == -math.huge then
				fail("number out of range")
			end
		end
		pos = p
		return v
	end
	local value
	function value(depth)
		ws()
		local c = text:byte(pos)
		if c == 123 or c == 91 then
			if depth >= MAX_DEPTH then
				fail("nesting too deep")
			end
			pos = pos + 1
			ws()
			if c == 91 then
				local arr = {}
				if text:byte(pos) == 93 then
					pos = pos + 1
					return json.array(arr)
				end
				while true do
					arr[#arr + 1] = value(depth + 1)
					ws()
					local d = text:byte(pos)
					pos = pos + 1
					if d == 93 then
						return json.array(arr)
					elseif d ~= 44 then
						pos = pos - 1
						fail("expected ',' or ']'")
					end
				end
			end
			local obj, order = {}, {}
			if text:byte(pos) == 125 then
				pos = pos + 1
				return setmetatable(obj, { __order = order })
			end
			while true do
				ws()
				if text:byte(pos) ~= 34 then
					fail("expected a string key")
				end
				local k = str()
				ws()
				if text:byte(pos) ~= 58 then
					fail("expected ':'")
				end
				pos = pos + 1
				local v = value(depth + 1)
				if obj[k] == nil then
					order[#order + 1] = k
				end
				obj[k] = v
				ws()
				local d = text:byte(pos)
				pos = pos + 1
				if d == 125 then
					return setmetatable(obj, { __order = order })
				elseif d ~= 44 then
					pos = pos - 1
					fail("expected ',' or '}'")
				end
			end
		elseif c == 34 then
			return str()
		elseif c == 45 or (c and c >= 48 and c <= 57) then
			return number()
		elseif text:sub(pos, pos + 3) == "true" then
			pos = pos + 4
			return true
		elseif text:sub(pos, pos + 4) == "false" then
			pos = pos + 5
			return false
		elseif text:sub(pos, pos + 3) == "null" then
			pos = pos + 4
			return null
		elseif c == nil then
			fail("unexpected end of text")
		end
		fail("unexpected character")
	end
	local ok, v = pcall(function()
		local r = value(0)
		ws()
		if pos <= #text then
			fail("trailing characters")
		end
		return r
	end)
	if ok then
		return v
	end
	return nil, type(v) == "table" and v.msg or tostring(v)
end

--- The grammar for response_format `json_object`.
-- It is the grammar of the schema `{ type = "object" }`.
M.json_object = assert(M.to_grammar({ type = "object" }))

return M
