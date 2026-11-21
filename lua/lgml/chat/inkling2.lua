--- Inkling 2 chat template and output parser.
-- Each message part is `<role token>[name]<content token>text<|end_message|>`.
-- A system message before the first other message sets the thinking
-- effort. The model writes its own turn headers, so there is no
-- generation prompt; it ends a turn with `<|content_model_end_sampling|>`.
-- @module lgml.chat.inkling2
local hermes = require("lgml.chat.hermes")
local tojson = require("lgml.chat.tojson")
local json = require("lgml.json")

local M = {}

local ROLE = {
	user = "<|message_user|>", assistant = "<|message_model|>", system = "<|message_system|>",
	tool = "<|message_tool|>",
}
local TEXT, THINK, END = "<|content_text|>", "<|content_thinking|>", "<|end_message|>"
local CALL, DONE = "<|content_invoke_tool_json|>", "<|content_model_end_sampling|>"
local IMAGE = "<|content_image|><|unused_200054|>"

--- Effort values by name, as the template maps them.
M.EFFORT = { none = 0.0, minimal = 0.1, low = 0.2, medium = 0.7, high = 0.9, xhigh = 0.99, max = 0.99 }

--- Control tokens that the output decoder keeps for the parser.
M.preserved = {
	[ROLE.assistant] = true, [TEXT] = true, [THINK] = true, [END] = true, [CALL] = true, [DONE] = true,
	["<|content_invoke_tool_text|>"] = true,
}

--- The image tokens: one placeholder per image, as the template writes it.
M.image = { begin = "<|content_image|>", token = "<|unused_200054|>", finish = "" }

-- JSON with the given separators, as Jinja tojson with sort_keys.
local function compact(v)
	local k = type(v) == "table" and tojson.kind(v) or nil
	if k == "array" then
		local parts = {}
		for i = 1, #v do
			parts[i] = compact(v[i])
		end
		return "[" .. table.concat(parts, ",") .. "]"
	elseif k == "object" then
		local parts = {}
		for i, key in ipairs(tojson.keys(v)) do
			parts[i] = tojson.string(tostring(key)) .. ":" .. compact(v[key])
		end
		return "{" .. table.concat(parts, ",") .. "}"
	end
	return tojson.encode(v)
end

-- Keys in sorted order, for sort_keys and dictsort.
local function sorted(t)
	local mt = getmetatable(t)
	local keys = {}
	for key in pairs(t) do
		keys[#keys + 1] = key
	end
	table.sort(keys, function(a, b)
		return tostring(a) < tostring(b)
	end)
	return keys, mt
end

-- A copy of v whose objects list their keys sorted.
local function sort_keys(v)
	if type(v) ~= "table" or json.isarray(v) or tojson.kind(v) ~= "object" then
		if type(v) == "table" and tojson.kind(v) == "array" then
			local out = setmetatable({}, getmetatable(v))
			for i = 1, #v do
				out[i] = sort_keys(v[i])
			end
			return out
		end
		return v
	end
	local keys = sorted(v)
	local out = setmetatable({}, { __order = keys, __jsontype = "object" })
	for _, key in ipairs(keys) do
		out[key] = sort_keys(v[key])
	end
	return out
end

-- The effort line, from the template's emit_thinking_effort.
local function effort_line(opts)
	local e = opts.reasoning_effort or opts.effort_hint
	if opts.enable_thinking == false then
		e = "none"
	end
	local num = 0.9
	if type(e) == "string" then
		num = M.EFFORT[e:match("^%s*(.-)%s*$")]
		if not num then
			error("chat: unknown reasoning_effort: " .. e, 0)
		end
	elseif type(e) == "number" then
		num = e
	end
	if num < 0 or num > 0.99 then
		error("chat: reasoning_effort must be in [0.0, 0.99]", 0)
	end
	local s = num == 0 and "0" or tojson.float(num)
	return ROLE.system .. TEXT .. "Thinking effort level: " .. s .. END
end

local function tool_specs(tools)
	local specs = {}
	for i, t in ipairs(tools) do
		local fn = type(t["function"]) == "table" and t["function"] or t
		specs[i] = sort_keys({
			description = type(fn.description) == "string" and fn.description or "",
			name = fn.name,
			parameters = type(fn.parameters) == "table" and next(fn.parameters) ~= nil and fn.parameters or
			    setmetatable({}, { __jsontype = "object" }),
			type = type(t.type) == "string" and t.type ~= "" and t.type or "function",
		})
	end
	return ROLE.system .. "tool_declare<|content_xml|>" .. compact(specs) .. END
end

local function args_table(args)
	if args == nil or args == json.null or args == "" then
		return {}
	end
	if type(args) == "string" then
		local ok, v = pcall(json.decode, args)
		if not ok or type(v) ~= "table" then
			error("chat: tool call arguments are not a JSON object", 0)
		end
		args = v
	end
	if type(args) ~= "table" or tojson.kind(args) == "array" then
		error("chat: tool call arguments must be an object", 0)
	end
	return args
end

local function render_calls(m, out)
	for _, tc in ipairs(m.tool_calls) do
		local fn = type(tc["function"]) == "table" and tc["function"] or tc
		local name = fn.name
		if type(name) ~= "string" then
			error("chat: tool call function name must be a string", 0)
		end
		local args = args_table(fn.arguments)
		local parts = {}
		for i, key in ipairs((sorted(args))) do
			parts[i] = tojson.string(tostring(key)) .. ":" .. compact(args[key])
		end
		out[#out + 1] = ROLE.assistant .. name .. CALL .. '{"name":' .. tojson.string(name) .. ',"args":{' ..
		    table.concat(parts, ",") .. "}}" .. END
	end
end

-- The name a tool message carries: its own, or that of the call it answers.
local function tool_name(messages, m)
	if type(m.name) == "string" and m.name ~= "" then
		return m.name
	end
	local id = m.tool_call_id
	if type(id) ~= "string" or id == "" then
		return ""
	end
	local found = ""
	for _, p in ipairs(messages) do
		if p.role == "assistant" and type(p.tool_calls) == "table" then
			for _, tc in ipairs(p.tool_calls) do
				local fn = type(tc["function"]) == "table" and tc["function"] or {}
				if tc.id == id and type(fn.name) == "string" then
					found = fn.name
				end
			end
		end
	end
	return found
end

local function has_calls(m)
	return type(m.tool_calls) == "table" and #m.tool_calls > 0
end

local function render_message(messages, m, out)
	local role = m.role == "developer" and "system" or m.role
	local head = ROLE[role]
	if not head then
		error("chat: unknown message role: " .. tostring(m.role), 0)
	end
	if role == "tool" then
		head = head .. tool_name(messages, m)
	elseif role == "assistant" and type(m.name) == "string" then
		head = head .. m.name
	end
	local reasoning = type(m.reasoning_content) == "string" and m.reasoning_content ~= "" and m.reasoning_content
	if role == "assistant" and reasoning then
		out[#out + 1] = head .. THINK .. reasoning .. END
	end
	local c = m.content
	if type(c) == "string" and c:find(IMAGE, 1, true) then
		-- An image marker from lgml.chat.media gets a block of its own.
		local i = 1
		while i <= #c do
			local j = c:find(IMAGE, i, true)
			if j == i then
				out[#out + 1] = head .. IMAGE .. END
				i = i + #IMAGE
			else
				local e = j and j - 1 or #c
				out[#out + 1] = head .. TEXT .. c:sub(i, e) .. END
				i = e + 1
			end
		end
	elseif type(c) == "string" then
		if c ~= "" or not (role == "assistant" and (has_calls(m) or reasoning)) then
			out[#out + 1] = head .. TEXT .. c .. END
		end
	elseif type(c) == "table" and c ~= json.null then
		for _, part in ipairs(c) do
			if type(part) == "string" then
				out[#out + 1] = head .. TEXT .. part .. END
			elseif part.type == nil or part.type == "text" or part.type == "input_text" then
				out[#out + 1] = head .. TEXT .. (type(part.text) == "string" and part.text or "") .. END
			elseif part.type == "image" or part.type == "input_image" or part.type == "image_url" then
				if role ~= "user" and role ~= "tool" then
					error("chat: image content is only supported for user and tool messages", 0)
				end
				out[#out + 1] = head .. IMAGE .. END
			else
				error("chat: unsupported content part type: " .. tostring(part.type), 0)
			end
		end
	end
	if role == "assistant" and has_calls(m) then
		render_calls(m, out)
	end
	if role == "assistant" then
		out[#out + 1] = DONE
	end
end

--- Renders messages as prompt text.
-- @tparam {table,...} messages OpenAI chat messages
-- @tparam[opt] table opts the options:
--
-- - `tools`: OpenAI tools list
-- - `enable_thinking`: false sets the effort to none (default true)
-- - `reasoning_effort`: an effort name from `EFFORT`, or a number in
--   [0, 0.99] (default 0.9)
-- - `effort_hint`: the effort when reasoning_effort is not set
-- - `continue_final_message`: continue the text of a final assistant
--   message
-- @treturn string the prompt text
-- @raise for an unknown role or effort, or bad tool call arguments
function M.render(messages, opts)
	opts = opts or {}
	local out = {}
	if type(opts.tools) == "table" and #opts.tools > 0 then
		out[1] = tool_specs(opts.tools)
	end
	local last = messages[#messages]
	local cont = opts.continue_final_message and last and last.role == "assistant"
	local n = cont and #messages - 1 or #messages
	local effort = false
	for i = 1, n do
		local m = messages[i]
		if not effort and m.role ~= "system" and m.role ~= "developer" then
			out[#out + 1] = effort_line(opts)
			effort = true
		end
		render_message(messages, m, out)
	end
	if not effort then
		out[#out + 1] = effort_line(opts)
	end
	if cont then
		local c = type(last.content) == "string" and last.content or ""
		out[#out + 1] = ROLE.assistant .. TEXT .. c
	end
	return table.concat(out)
end

-- The parser: a subclass of the hermes parser with the Inkling tags.
local P = setmetatable({}, { __index = hermes.P })
P.__index = P
P.call_tag = CALL

--- Makes an output parser.
-- @tparam[opt] table opts the options: `tools`, tool names or OpenAI
-- tools; calls to other names become content
-- @treturn lgml.chat.hermes.Parser the parser
function M.new_parser(opts)
	opts = opts or {}
	return setmetatable({
		state = "start", buf = "", hold = "", hold_kind = "content", lead = true, ncalls = 0,
		names = hermes.name_set(opts.tools),
	}, P)
end

local TAGS = { ROLE.assistant, TEXT, THINK, END, CALL, DONE }

-- Length of the longest suffix of s that is a proper prefix of a tag.
local function partial(s)
	for k = math.min(#s, 31), 1, -1 do
		local tail = s:sub(-k)
		for _, tag in ipairs(TAGS) do
			if k < #tag and tag:sub(1, k) == tail then
				return k
			end
		end
	end
	return 0
end

-- The first tag in buf: its position and text.
local function first_tag(buf)
	local at, tag
	for _, t in ipairs(TAGS) do
		local i = buf:find(t, 1, true)
		if i and (not at or i < at) then
			at, tag = i, t
		end
	end
	return at, tag
end

-- Between parts: a turn header names the next part. Text outside a
-- part is content.
function P:step_start(final)
	local buf = self.buf
	local at, tag = first_tag(buf)
	if not at then
		local k = final and 0 or partial(buf)
		self.state = "loose"
		return self:step_loose(final, k)
	end
	if at > 1 then
		self.state = "loose"
		return true
	end
	self.buf = buf:sub(#tag + 1)
	if tag == ROLE.assistant then
		self.state = "header"
	elseif tag == TEXT then
		self:switch("content")
	elseif tag == THINK then
		self:switch("reasoning")
	elseif tag == CALL then
		self.state = "call_json"
		self.call_name, self.json = "", ""
	end
	return true
end

-- Text that no part header opened, up to the next tag.
function P:step_loose(final, k)
	local buf = self.buf
	local at = first_tag(buf)
	if not at then
		k = k or (final and 0 or partial(buf))
		self:emit("content", buf:sub(1, #buf - k))
		self.buf = buf:sub(#buf - k + 1)
		return false
	end
	self:emit("content", buf:sub(1, at - 1))
	self.buf = buf:sub(at)
	self.state = "start"
	return true
end

-- After <|message_model|>: an optional name, then the content token.
function P:step_header(final)
	local buf = self.buf
	local at, tag = first_tag(buf)
	if not at then
		if final then
			self.buf = ""
		end
		return false
	end
	local name = buf:sub(1, at - 1)
	self.buf = buf:sub(at + #tag)
	if tag == TEXT then
		self:switch("content")
	elseif tag == THINK then
		self:switch("reasoning")
	elseif tag == CALL then
		self.state = "call_json"
		self.call_name, self.json = name, ""
	else
		self.buf = tag .. self.buf
		self.state = "start"
	end
	return true
end

function P:step_content(final)
	if not self:scan("content", END, final) then
		return false
	end
	self:switch("start")
	return true
end

function P:step_reasoning(final)
	if not self:scan("reasoning", END, final) then
		return false
	end
	self:switch("start")
	return true
end

-- A call: {"name": ..., "args": {...}} up to <|end_message|>.
function P:step_call_json(final)
	local buf = self.buf
	local i = buf:find(END, 1, true)
	if not i and not final then
		local k = partial(buf)
		self.json = self.json .. buf:sub(1, #buf - k)
		self.buf = buf:sub(#buf - k + 1)
		return false
	end
	local body = self.json .. (i and buf:sub(1, i - 1) or buf)
	self.buf = i and buf:sub(i + #END) or ""
	local ok, v = pcall(json.decode, body)
	local name = ok and type(v) == "table" and (v.name or self.call_name) or nil
	if name == "" or (self.names and not self.names[name]) then
		name = nil
	end
	if not name then
		self.state, self.lead = "content", false
		self:emit("content", body)
		self:switch("start")
		return true
	end
	local args = v.args or v.arguments
	if args == nil or args == json.null then
		args = setmetatable({}, { __jsontype = "object" })
	end
	self.ncalls = self.ncalls + 1
	self:drop_hold()
	self:push({ type = "tool_call_begin", index = self.ncalls - 1, name = name })
	self:push({ type = "tool_call_args", index = self.ncalls - 1, text = json.encode(args) })
	self:push({ type = "tool_call_end", index = self.ncalls - 1 })
	self:switch("start")
	return true
end

return M
