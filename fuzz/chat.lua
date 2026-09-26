-- The Hermes, Gemma-4 and Qwen3.5 output parsers on model text, fed
-- in pieces. The first byte sets the piece size.
local qwen3 = require("lgml.chat.qwen3")
local gemma4 = require("lgml.chat.gemma4")
local qwen35 = require("lgml.chat.qwen35")

-- Qwen3.5 tools: one with a schema, one name without.
local QT = { { type = "function", ["function"] = { name = "get_weather", parameters = {
	type = "object", properties = { city = { type = "string" }, days = { type = "integer" },
		opts = { anyOf = { { type = "object" }, { type = "null" } } } },
	required = { "city" } } } }, "f" }

return function(s)
	if #s == 0 then
		return
	end
	local step = s:byte(1) % 16 + 1
	local think = s:byte(1) >= 128
	local opts = { tools = { "f", "get_weather" }, in_think = think }
	local parsers = { qwen3.new_parser(opts), gemma4.new_parser(opts),
		qwen35.new_parser({ tools = QT, in_think = think }) }
	for _, p in ipairs(parsers) do
		for i = 2, #s, step do
			p:feed(s:sub(i, i + step - 1))
		end
		p:finish()
	end
end
