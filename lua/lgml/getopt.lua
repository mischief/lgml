--- POSIX-style getopt with long options, in pure Lua.
-- In optstring, a letter followed by ':' takes an argument.
--
-- The long table maps a long name to a short letter, or to false for a
-- long-only flag. A long-only name that ends in '=' takes an argument.
-- Results use the short letter as key, else the long name.
-- Parsing stops at "--" or at the first operand.
-- @module lgml.getopt

local M = {}

local function shorts(optstring)
	local t = {}
	for c, colon in optstring:gmatch("(%w)(:?)") do
		t[c] = colon == ":"
	end
	return t
end

local function longs(spec, sh)
	local t = {}
	for name, alias in pairs(spec or {}) do
		local base, eq = name:match("^([^=]+)(=?)$")
		local key = alias or base
		local takes = eq == "="
		if alias then
			takes = sh[alias]
			if takes == nil then
				error("getopt: long option " .. base ..
				    " maps to unknown -" .. alias)
			end
		end
		t[base] = { key = key, arg = takes }
	end
	return t
end

-- Matches a unique prefix of a long option name.
local function find_long(lt, name)
	if lt[name] then
		return lt[name]
	end
	local hit
	for k, v in pairs(lt) do
		if k:sub(1, #name) == name then
			if hit then
				return nil, "option --" .. name .. " is ambiguous"
			end
			hit = v
		end
	end
	if not hit then
		return nil, "unknown option --" .. name
	end
	return hit
end

local function parse_long(lt, args, i, opts)
	local name, val = args[i]:match("^%-%-([^=]+)=(.*)$")
	name = name or args[i]:sub(3)
	local o, err = find_long(lt, name)
	if not o then
		return nil, err
	end
	if not o.arg then
		if val then
			return nil, "option --" .. name .. " takes no argument"
		end
		opts[o.key] = true
		return i + 1
	end
	if not val then
		i = i + 1
		val = args[i]
		if val == nil then
			return nil, "option --" .. name .. " needs an argument"
		end
	end
	opts[o.key] = val
	return i + 1
end

local function parse_short(sh, args, i, opts)
	local s = args[i]
	local j = 2
	while j <= #s do
		local c = s:sub(j, j)
		local takes = sh[c]
		if takes == nil then
			return nil, "unknown option -" .. c
		end
		if not takes then
			opts[c] = true
			j = j + 1
		else
			local val = s:sub(j + 1)
			if val == "" then
				i = i + 1
				val = args[i]
				if val == nil then
					return nil, "option -" .. c ..
					    " needs an argument"
				end
			end
			opts[c] = val
			return i + 1
		end
	end
	return i + 1
end

--- Parse a command line.
-- A flag gets the value true. An option with an argument gets the string.
-- A long option matches any unique prefix of its name. It takes its
-- argument as `--name=value` or as the next word.
-- @tparam {string,...} args a list such as the global arg
-- @tparam[opt] string optstring short options, such as "hn:"
-- @tparam[opt] table long long options, such as `{ help = "h", ["seed="] = false }`
-- @treturn table the options, keyed by short letter or long name
-- @treturn {string,...} the operands
-- @return[2] nil on an unknown, ambiguous, or incomplete option
-- @treturn[2] string the error message
-- @raise when a long option maps to a letter not in optstring
function M.parse(args, optstring, long)
	local sh = shorts(optstring or "")
	local lt = longs(long, sh)
	local opts, rest = {}, {}
	local i, err = 1, nil
	while i <= #args do
		local a = args[i]
		if a == "--" then
			i = i + 1
			break
		elseif a:sub(1, 2) == "--" then
			i, err = parse_long(lt, args, i, opts)
		elseif a:sub(1, 1) == "-" and #a > 1 then
			i, err = parse_short(sh, args, i, opts)
		else
			break
		end
		if not i then
			return nil, err
		end
	end
	for k = i, #args do
		rest[#rest + 1] = args[k]
	end
	return opts, rest
end

return M
