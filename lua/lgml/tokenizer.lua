--- Tokenizer helpers on top of the C tokenizer (`lgml.tokenizer`).
-- @module lgml.tokenizer
local lgml = require("lgml")

local M = {}

--- Open the tokenizer of a gguf file.
-- @tparam string path path to the gguf file
-- @return the C tokenizer object
-- @return[2] nil on failure
-- @treturn[2] string the error message
function M.open(path)
	return lgml.tokenizer(path)
end

-- Length of the complete UTF-8 prefix of s. A cut sequence at the end
-- stays out. Invalid bytes count as complete.
local function complete_len(s)
	local n = #s
	for i = n, math.max(1, n - 3), -1 do
		local c = s:byte(i)
		if c < 0x80 then
			return n
		elseif c >= 0xc0 then
			local need = c >= 0xf0 and 4 or c >= 0xe0 and 3 or 2
			if n - i + 1 < need then
				return i - 1
			end
			return n
		end
	end
	return n
end

--- Replace each invalid UTF-8 byte with U+FFFD.
-- @tparam string s the input bytes
-- @treturn string valid UTF-8 text
function M.clean(s)
	local out, i = {}, 1
	while true do
		local n, bad = utf8.len(s, i)
		if n then
			out[#out + 1] = s:sub(i)
			return table.concat(out)
		end
		out[#out + 1] = s:sub(i, bad - 1)
		out[#out + 1] = "\u{fffd}"
		i = bad + 1
	end
end

--- Make a streaming decoder for generated tokens.
-- The decoder holds back an incomplete UTF-8 sequence at the end of the
-- text until the next token completes it. Unused tokens give no text.
-- @param tk the C tokenizer object
-- @tparam[opt] table opts options
-- opts.special: false drops control and unknown tokens (default true)
-- opts.keep: control token pieces to keep with special false, as a set,
-- such as the tool call tags of a chat family
-- @treturn Decoder a new decoder
function M.decoder(tk, opts)
	local special = not (opts and opts.special == false)
	local keep = opts and opts.keep or {}
	local pending = ""
	local d = {}

	--- A streaming token decoder, from `decoder`.
	-- @type Decoder

	--- Add one token.
	-- @function push
	-- @tparam integer id the token id
	-- @treturn string the new complete text, may be empty
	function d.push(_, id)
		local piece, kind = tk:token(id)
		if kind == "unused" or (not special and
		    (kind == "control" or kind == "unknown") and not keep[piece]) then
			return ""
		end
		pending = pending .. piece
		local k = complete_len(pending)
		local out = pending:sub(1, k)
		pending = pending:sub(k + 1)
		return M.clean(out)
	end

	--- Take the rest of the held text.
	-- The call clears the decoder state.
	-- @treturn string the held text, with invalid bytes replaced
	-- @function flush
	function d.flush(_)
		local out = pending
		pending = ""
		return M.clean(out)
	end

	return d
end

return M
