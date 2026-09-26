--- Llama-3 dense decoder (gguf architecture "llama").
-- Attention has no q and k norms and uses normal rope with optional
-- llama3 frequency factors. `tokenizer` wraps the C llama-bpe
-- tokenizer. The model is `lgml.transformer`; see there for the model
-- methods.
-- @module lgml.llama
local lgml = require("lgml")
local transformer = require("lgml.transformer")

local M = {}

--- Read the config from gguf metadata. This is `lgml.transformer.config`.
-- @function hparams
-- @tparam table meta the gguf metadata
-- @treturn table the config
M.hparams = transformer.config

--- Load a model. This is `lgml.transformer.load`.
-- @function load
-- @param be the lgml backend
-- @tparam string path path to the gguf file
-- @tparam[opt] table opts options for `lgml.transformer.load`
-- @treturn lgml.transformer.Model the model
M.load = transformer.load

--- The C tokenizer (llama-bpe pre-tokenizer).
-- encode adds no BOS and does not parse special tokens, unless opts say
-- so.
-- @tparam string path path to the gguf file
-- @treturn Tokenizer the tokenizer
-- @return[2] nil on failure
-- @treturn[2] string the error message
-- @function tokenizer
function M.tokenizer(path)
	local c, err = lgml.tokenizer(path)
	if not c then
		return nil, err
	end
	local sp = c:special()
	local t = { c = c, bos = sp.bos, eos = sp.eos }
	--- A llama tokenizer, from `tokenizer`.
	-- Fields: c (the C tokenizer), bos and eos (token ids).
	-- @type Tokenizer

	--- Encode text as token ids.
	-- @function encode
	-- @tparam string text the text
	-- @tparam[opt] table opts options
	-- opts.special: true parses special tokens (default false)
	-- opts.add_bos: true adds BOS (default false)
	-- @treturn {integer,...} the token ids
	function t.encode(_, text, opts)
		opts = opts or {}
		return c:encode(text, { special = opts.special or false, add_bos = opts.add_bos or false })
	end
	--- Look up the id of a token piece.
	-- @function token_id
	-- @tparam string piece the token text
	-- @treturn ?integer the id, or nil when absent
	function t.token_id(_, piece)
		return sp.tokens[piece]
	end
	return t
end

return M
