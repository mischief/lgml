--- Strict globals.
-- After `on`, reading a global that has no value is an error, and so is
-- setting a new global outside the main chunk. A missing `local` or a
-- misspelled name then fails at the line that has it, not later as a
-- call on nil. Set globals on purpose with `declare`.
-- @module lgml.strict
local M = {}

-- The main chunk (the script itself) may still set globals; library
-- code may not.
local function main_chunk(level)
	local info = debug.getinfo(level + 1, "S")
	return info == nil or info.what == "main" or info.what == "C"
end

--- Turn strict globals on for `_G`. A second call does nothing.
-- @raise when `_G` already has a metatable of another kind
function M.on()
	local mt = getmetatable(_G)
	if mt and mt.__strict then
		return
	end
	assert(mt == nil, "strict: _G already has a metatable")
	local declared = {}
	setmetatable(_G, {
		__strict = declared,
		__index = function(_, k)
			if declared[k] or type(k) ~= "string" then
				return nil
			end
			error(("undefined global '%s'"):format(k), 2)
		end,
		__newindex = function(t, k, v)
			if not declared[k] and not main_chunk(2) then
				error(("assignment to undeclared global '%s'"):format(tostring(k)), 2)
			end
			declared[k] = true
			rawset(t, k, v)
		end,
	})
end

--- Declare globals, so code may set and read them while they are nil.
-- @tparam string ... the names
function M.declare(...)
	local mt = getmetatable(_G)
	for i = 1, select("#", ...) do
		local k = select(i, ...)
		if mt and mt.__strict then
			mt.__strict[k] = true
		end
	end
end

--- Tell if a global has a value, without an error when it has none.
-- @tparam string k the name
-- @return the value, or nil
function M.get(k)
	return rawget(_G, k)
end

return M
