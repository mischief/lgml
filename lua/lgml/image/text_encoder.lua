--- Qwen3 text encoder for Z-Image.
-- The prompt goes into the chat template that stable-diffusion.cpp
-- uses for Z-Image: a user turn, then an open assistant turn. The
-- template parts and the prompt are tokenized one at a time and joined,
-- as the reference does. The encoder runs the decoder causally, with
-- no LM head, and gives the hidden state after layer 35 of 36 (the
-- second to last of the Hugging Face hidden states) for every token,
-- the template tokens included.
-- @module lgml.image.text_encoder
local tokenizer = require("lgml.tokenizer")
local transformer = require("lgml.transformer")

local M = {}
M.__index = M

--- Template text before the prompt.
M.PREFIX = "<|im_start|>user\n"
--- Template text after the prompt.
M.SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
--- Layers to run: the output is the hidden state after this many.
M.LAYERS = 35
--- RoPE base. The reference uses the Qwen3-4B base value for every
-- file; Qwen3-4B-Instruct-2507 files give 5e6.
M.ROPE_BASE = 1e6

--- Load the encoder.
-- @param be the lgml backend
-- @tparam string path the Qwen3 gguf file
-- @treturn Encoder the encoder
-- @raise when the model or its tokenizer does not load
function M.load(be, path)
	local model = transformer.load(be, path, { replay = false })
	assert(model.hp.n_layer >= M.LAYERS, path .. ": too few layers")
	model.hp.rope_base = M.ROPE_BASE
	for _, L in ipairs(model.layers) do
		L.base = M.ROPE_BASE
	end
	return setmetatable({ be = be, model = model, tk = assert(tokenizer.open(path)) }, M)
end

--- A text encoder, from `load`.
-- Fields: be, model (an `lgml.transformer` model) and tk.
-- @type Encoder

--- Close the weights and the tokenizer.
function M:close()
	self.model:close()
	self.tk:close()
end

--- Token ids of a prompt in the template.
-- @tparam string prompt the prompt
-- @treturn {integer,...} the ids
function M:tokens(prompt)
	local ids = {}
	for _, s in ipairs({ M.PREFIX, prompt, M.SUFFIX }) do
		for _, id in ipairs(self.tk:encode(s, { add_bos = false })) do
			ids[#ids + 1] = id
		end
	end
	return ids
end

--- Encode a prompt.
-- @tparam string prompt the prompt
-- @treturn string the hidden states (n, n_embd) as raw f32 bytes
-- @treturn integer n, the token count
-- @treturn integer n_embd
function M:encode(prompt)
	local ids = self:tokens(prompt)
	local inp = self.model:inputs(ids, nil, { hidden = M.LAYERS })
	local g = self.be:graph()
	local h = self.model:forward(g, inp)
	g:compute(h)
	local bytes = h:bytes()
	g:close()
	for _, t in ipairs({ inp.ids, inp.pos, inp.mask }) do
		t:close()
	end
	return bytes, #ids, self.model.hp.n_embd
end

return M
