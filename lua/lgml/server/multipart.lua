--- multipart/form-data parser.
-- A parsed form is `{ fields = {name = value}, files = {name = file},
-- parts = {...} }`. A part is `{ name, filename, type, data }`. A part
-- with a filename is a file.
-- @module lgml.server.multipart

local M = {}

--- The most parts in one form.
M.max_parts = 256

local function quoted(v, i)
	local buf, j = {}, i + 1
	while j <= #v do
		local c = v:sub(j, j)
		if c == '"' then
			break
		elseif c == "\\" then
			j = j + 1
			c = v:sub(j, j)
		end
		buf[#buf + 1] = c
		j = j + 1
	end
	return table.concat(buf), j + 1
end

--- Splits a header value into its first token and its parameters.
-- For example, `form-data; name="a"` gives `"form-data"` and
-- `{ name = "a" }`.
-- @tparam string v the header value
-- @treturn string the first token, in lowercase
-- @treturn table the parameters, with lowercase names
function M.params(v)
	local t = {}
	local first = v:match("^%s*([^;%s]*)")
	local i = v:find(";", 1, true)
	while i do
		local _, e, key = v:find("^;%s*([^=;%s]+)%s*=%s*", i)
		if key then
			local val
			if v:sub(e + 1, e + 1) == '"' then
				val, i = quoted(v, e + 1)
			else
				val = v:match("^[^;]*", e + 1)
				i = e + 1 + #val
				val = val:match("^(.-)%s*$")
			end
			t[key:lower()] = val
		end
		i = v:find(";", i + (key and 0 or 1), true)
	end
	return first:lower(), t
end

--- Returns the boundary parameter of a Content-Type value.
-- @tparam ?string ctype the Content-Type value
-- @treturn ?string the boundary, or nil when the type is not
-- multipart/form-data or the boundary is not valid
function M.boundary(ctype)
	local kind, p = M.params(ctype or "")
	local b = p.boundary
	if kind ~= "multipart/form-data" or not b or #b == 0 or #b > 70 then
		return nil
	end
	return b
end

-- Reads name and filename from a Content-Disposition value.
local function disposition(v)
	local kind, p = M.params(v)
	if kind ~= "form-data" then
		return nil
	end
	return p.name, p.filename
end

local function part_headers(head)
	local h = {}
	for line in (head .. "\r\n"):gmatch("([^\r\n]*)\r\n") do
		local k, v = line:match("^([^:]+):%s*(.-)%s*$")
		if not k then
			return nil
		end
		h[k:lower()] = v
	end
	return h
end

local function add(res, p)
	res.parts[#res.parts + 1] = p
	if p.filename then
		res.files[p.name] = p
	else
		res.fields[p.name] = p.data
	end
end

--- Parses body with boundary b.
-- @tparam string body the body
-- @tparam string b the boundary
-- @treturn[1] table the form
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.parse(body, b)
	local delim = "--" .. b
	local res = { fields = {}, files = {}, parts = {} }
	local s, e = body:find(delim, 1, true)
	if s ~= 1 then
		s, e = body:find("\r\n" .. delim, 1, true)
	end
	if not s then
		return nil, "multipart: no boundary"
	end
	local sep = "\r\n" .. delim
	local pos = e + 1
	while true do
		if body:sub(pos, pos + 1) == "--" then
			return res
		end
		if body:sub(pos, pos + 1) ~= "\r\n" then
			return nil, "multipart: bad boundary line"
		end
		if #res.parts >= M.max_parts then
			return nil, "multipart: too many parts"
		end
		local hs, he = body:find("\r\n\r\n", pos, true)
		if not hs then
			return nil, "multipart: bad part header"
		end
		local h = hs > pos + 2 and part_headers(body:sub(pos + 2, hs - 1))
		    or {}
		if not h then
			return nil, "multipart: bad part header"
		end
		local name, filename = disposition(h["content-disposition"] or "")
		if not name then
			return nil, "multipart: part has no name"
		end
		local ds, de = body:find(sep, he + 1, true)
		if not ds then
			return nil, "multipart: unterminated part"
		end
		add(res, {
			name = name, filename = filename,
			type = h["content-type"], data = body:sub(he + 1, ds - 1),
		})
		pos = de + 1
	end
end

--- Parses the body of an HTTP request.
-- @tparam table req the request
-- @treturn[1] table the form
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.from_request(req)
	local b = M.boundary(req.headers["content-type"])
	if not b then
		return nil, "multipart: not multipart/form-data"
	end
	return M.parse(req.body, b)
end

return M
