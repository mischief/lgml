-- JSON request bodies: lgml.json, the OpenAI checks and the chat
-- templates. Bad input must give nil and a message, not an error.
local json = require("lgml.json")
local openai = require("lgml.server.openai")
local qwen3 = require("lgml.chat.qwen3")
local gemma4 = require("lgml.chat.gemma4")
local qwen35 = require("lgml.chat.qwen35")

-- Qwen3.5 render raises where its Jinja template raises. Those errors
-- start with "chat: "; any other error is a bug.
local function qwen35_render(v, opts)
	local good, err = pcall(qwen35.render, v.messages, opts)
	if not good and not tostring(err):find("^chat: ") then
		error(err, 0)
	end
end

return function(s)
	local v = json.decode(s)
	if v == nil then
		return
	end
	local text = json.encode(v)
	if text then
		assert(json.decode(text) ~= nil, "encoded JSON does not decode")
	end
	openai.check_speech(v)
	if openai.check_chat(v) then
		qwen3.render(v.messages, { tools = v.tools, add_generation_prompt = true })
		gemma4.render(v.messages, { tools = v.tools, add_generation_prompt = true })
		gemma4.render(v.messages, { tools = v.tools, continue_final_message = true })
		qwen35_render(v, { tools = v.tools, add_generation_prompt = true })
		qwen35_render(v, { tools = v.tools, continue_final_message = true, variant = "effort",
			enable_thinking = #s % 2 == 0, preserve_thinking = false })
		qwen35.new_parser({ tools = v.tools }):finish()
	end
end
