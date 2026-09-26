--- Images in OpenAI chat messages.
-- A content part `{ type = "image_url", image_url = { url = ... } }`
-- holds an image as a data URL. `extract` takes the images out of the
-- messages and puts a marker in their place, as llama-server does: a
-- message with an image becomes one string, its text parts joined by
-- newlines, with no newline next to a marker.
-- @module lgml.chat.media
local base64 = require("lgml.base64")

local M = {}

--- The largest image in bytes, after base64 decode.
M.MAX_BYTES = 32 << 20

--- Decode a data URL that holds an image.
-- Only `data:image/...;base64,...` URLs work; http and https URLs give
-- an error.
-- @tparam string url the URL
-- @treturn[1] string the image bytes
-- @treturn[1] string the media type, such as `"image/png"`
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.data_url(url)
	if type(url) ~= "string" then
		return nil, "image url must be a string"
	end
	local scheme = url:match("^(%a[%w+.-]*):")
	scheme = scheme and scheme:lower()
	if scheme == "http" or scheme == "https" then
		return nil, "image url must be a data URL; http and https URLs are not supported"
	elseif scheme ~= "data" then
		return nil, "image url must be a data URL"
	end
	local head, data = url:match("^[^:]+:([^,]*),(.*)$")
	if not head then
		return nil, "data URL has no comma"
	end
	local params = {}
	for p in head:gmatch("[^;]+") do
		params[#params + 1] = p:lower()
	end
	local mime = table.remove(params, 1) or ""
	if not mime:match("^image/[%w.+-]+$") then
		return nil, "data URL must hold an image type, such as image/png"
	elseif params[#params] ~= "base64" then
		return nil, "data URL must be base64"
	elseif #data > M.MAX_BYTES // 3 * 4 + 4 then
		return nil, "image too large"
	end
	local bytes, err = base64.decode(data)
	if not bytes then
		return nil, "data URL: " .. err
	elseif #bytes == 0 then
		return nil, "data URL holds no data"
	end
	return bytes, mime
end

-- The URL of an image_url part, or nil and an error.
local function part_url(p)
	local u = p.image_url
	if type(u) == "table" then
		u = u.url
	end
	if type(u) ~= "string" then
		return nil, "image_url.url must be a string"
	end
	return u
end

--- Tell if messages hold image parts.
-- @tparam table messages the chat messages
-- @treturn boolean true when a content part has type "image_url"
function M.has_images(messages)
	for _, m in ipairs(messages) do
		if type(m) == "table" and type(m.content) == "table" then
			for _, p in ipairs(m.content) do
				if type(p) == "table" and p.type == "image_url" then
					return true
				end
			end
		end
	end
	return false
end

--- Take the images out of chat messages.
-- Messages without images stay as they are.
-- @tparam table messages the chat messages
-- @tparam string marker the text that takes the place of each image
-- @treturn[1] table new messages; a message with images is a copy with
-- string content
-- @treturn[1] {table,...} the images in order: `{ bytes, mime }`
-- @treturn[2] nil
-- @treturn[2] string an error message, with the message and part index
function M.extract(messages, marker)
	local out, images = {}, {}
	for i, m in ipairs(messages) do
		local c = type(m) == "table" and m.content
		if type(c) ~= "table" or not M.has_images({ m }) then
			out[i] = m
		else
			local text, len, last_marker = {}, 0, false
			for j, p in ipairs(c) do
				local at = ("messages[%d].content[%d]"):format(i - 1, j - 1)
				if type(p) ~= "table" then
					return nil, at .. " is not an object"
				end
				local piece
				if p.type == "text" then
					piece = type(p.text) == "string" and p.text or ""
					if not last_marker and len > 0 then
						piece = "\n" .. piece
					end
					last_marker = false
				elseif p.type == "image_url" then
					local url, err = part_url(p)
					local bytes, mime
					if url then
						bytes, mime = M.data_url(url)
					end
					if not bytes then
						return nil, at .. ": " .. (err or mime)
					end
					images[#images + 1] = { bytes = bytes, mime = mime }
					piece, last_marker = marker, true
				end
				if piece then
					text[#text + 1] = piece
					len = len + #piece
				end
			end
			local copy = {}
			for k, v in pairs(m) do
				copy[k] = v
			end
			copy.content = table.concat(text)
			out[i] = copy
		end
	end
	return out, images
end

return M
