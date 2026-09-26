--- Base64 (RFC 4648) with padding.
-- @module lgml.base64
local M = {}

local ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

-- Two output characters for each 12-bit value.
local pair = {}
for i = 0, 4095 do
	local a, b = i >> 6, i & 63
	pair[i] = ALPHA:sub(a + 1, a + 1) .. ALPHA:sub(b + 1, b + 1)
end

local value = {}
for i = 1, 64 do
	value[ALPHA:byte(i)] = i - 1
end

--- Encode bytes.
-- @tparam string s the bytes
-- @treturn string the text
function M.encode(s)
	local out, n = {}, #s
	local full = n - n % 3
	local i = 1
	while i + 11 <= full do
		local a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4 = s:byte(i, i + 11)
		local v1 = a1 << 16 | b1 << 8 | c1
		local v2 = a2 << 16 | b2 << 8 | c2
		local v3 = a3 << 16 | b3 << 8 | c3
		local v4 = a4 << 16 | b4 << 8 | c4
		out[#out + 1] = pair[v1 >> 12] .. pair[v1 & 4095] .. pair[v2 >> 12] .. pair[v2 & 4095] ..
		    pair[v3 >> 12] .. pair[v3 & 4095] .. pair[v4 >> 12] .. pair[v4 & 4095]
		i = i + 12
	end
	while i <= full do
		local a, b, c = s:byte(i, i + 2)
		local v = a << 16 | b << 8 | c
		out[#out + 1] = pair[v >> 12] .. pair[v & 4095]
		i = i + 3
	end
	local r = n - full
	if r == 1 then
		local v = s:byte(n) << 4
		out[#out + 1] = pair[v] .. "=="
	elseif r == 2 then
		local a, b = s:byte(n - 1, n)
		local v = (a << 8 | b) << 2
		out[#out + 1] = pair[v >> 6] .. ALPHA:sub((v & 63) + 1, (v & 63) + 1) .. "="
	end
	return table.concat(out)
end

--- Decode text.
-- Whitespace is skipped.
-- @tparam string s the text
-- @treturn[1] string the bytes
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.decode(s)
	s = s:gsub("%s", "")
	local pad = s:match("=*$")
	if #s % 4 ~= 0 or #pad > 2 then
		return nil, "base64: bad length"
	end
	local out, acc, bits = {}, 0, 0
	for i = 1, #s - #pad do
		local v = value[s:byte(i)]
		if not v then
			return nil, "base64: bad character"
		end
		acc, bits = (acc << 6) | v, bits + 6
		if bits >= 8 then
			bits = bits - 8
			out[#out + 1] = string.char((acc >> bits) & 255)
		end
	end
	return table.concat(out)
end

return M
