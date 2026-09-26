-- Writes the seed corpus: lua5.4 fuzz/seeds.lua [dir]. Each target
-- gets a few small valid inputs and near misses.
local dir = arg[1] or ((debug.getinfo(1, "S").source:match("^@(.*/)") or "./") .. "corpus")

local function put(target, name, s)
	os.execute(("mkdir -p '%s/%s'"):format(dir, target))
	local f = assert(io.open(("%s/%s/%s"):format(dir, target, name), "wb"))
	f:write(s)
	f:close()
end

-- json
put("json", "chat", [[{"model":"m","messages":[{"role":"system","content":"s"},]] ..
    [[{"role":"user","content":[{"type":"text","text":"hi"}]},]] ..
    [[{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function",]] ..
    [["function":{"name":"f","arguments":"{\"a\":1}"}}]},{"role":"tool","tool_call_id":"c",]] ..
    [["content":"ok"}],"tools":[{"type":"function","function":{"name":"f",]] ..
    [["description":"d","parameters":{"type":"object","properties":{"a":{"type":"integer"}}}}}],]] ..
    [["stream":true,"max_tokens":16,"temperature":0.5}]])
put("json", "image", [[{"model":"m","messages":[{"role":"user","content":[]] ..
    [[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}},]] ..
    [[{"type":"image_url","image_url":"data:image/jpeg;base64,/9j/"},{"type":"text","text":"b"}]}]}]])
put("json", "data-url", [["data:image/png;base64,iVBORw0KGgo="]])
put("json", "speech", [[{"model":"vibevoice","input":"hello","voice":"alloy","speed":1.0,"response_format":"wav"}]])
put("json", "scalars", [=[[1,-2.5e3,"é😀",true,false,null,{},[]]]=])
put("json", "number", "5")
put("json", "tools", [[{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"w?"},{"role":"assistant","content":"","reasoning_content":"r","tool_calls":[{"id":"a","type":"function","function":{"name":"w","arguments":"{\"c\": [1.5, {\"d\": null}]}"}}]},{"role":"tool","tool_call_id":"a","content":"ok"}],"tools":[{"type":"function","function":{"name":"w","description":"d","parameters":{"type":"object","properties":{"c":{"type":"array","items":{"type":"object","properties":{"d":{"type":"string","enum":["x"]}}}},"o":{"type":"object","nullable":true}},"required":["c"]}}}]}]])
put("json", "qwen35", [[{"model":"m","messages":[{"role":"developer","content":"d"},{"role":"system","content":" s "},]] ..
    [[{"role":"user","content":"w?"},{"role":"assistant","content":"<think>r</think>x","tool_calls":[{"type":"function",]] ..
    [["function":{"name":"w","arguments":{"c":[1.5,{"d":null}],"s":"t","b":true}}},{"type":"function",]] ..
    [["function":{"name":"w","arguments":"{\"c\": 2}"}}]},{"role":"tool","content":"ok"},{"role":"system","content":"late"},]] ..
    [[{"role":"assistant","content":"","reasoning_content":"r2"}],"tools":[{"type":"function","function":{"name":"w",]] ..
    [["parameters":{"type":"object","$defs":{"s":{"type":"string"}},"properties":{"c":{"type":"array"},]] ..
    [["s":{"$ref":"#/$defs/s"},"b":{"anyOf":[{"type":"boolean"},{"enum":["x"]}]}},"required":["c"]}}}],]] ..
    [["chat_template_kwargs":{"reasoning_effort":"low","preserve_thinking":false}}]])

-- chat: first byte sets piece size; high bit starts in think.
put("chat", "tool", "\3<think>\nplan</think>\n\ntext <tool_call>\n{\"name\": \"f\", " ..
    "\"arguments\": {\"a\": 1}}\n</tool_call>")
put("chat", "think", "\200reasoning here</think>answer")
put("chat", "bad", "\1<tool_call>\n{\"name\": \"nope\", \"arguments\": [}\n</tool_call>")
put("chat", "qwen35", "\131plan\n</think>\n\ntext\n<tool_call>\n<function=get_weather>\n" ..
    "<parameter=city>\nZ\"ü\n</parameter\n\n</parameter>\n<parameter=days>\n3\n</parameter>\n" ..
    "<parameter=opts>\n{\"a\": [1, null]}\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n" ..
    "<function=f>\n<parameter=x>\n[1]\n</parameter>\n<parameter=y>\nword\n</parameter>\n</function>\n</tool_call>")
put("chat", "qwen35-bad", "\4<think>x</think><tool_call>\n<function=get_weather>\n<parameter=days>\n3 \n" ..
    "</parameter>\n</function>\n</tool_call><tool_call>\n<function=nope>\n")
put("chat", "gemma4", "\5<|channel>thought\nplan<channel|>text <|tool_call>call:f{a:<|\"|>x<|\"|>," ..
    "b:[1,{c:true}]}<tool_call|>")

-- http
put("http", "get", "GET /v1/models?x=1&y=%41+b HTTP/1.1\r\nHost: h\r\nConnection: close\r\n\r\n")
put("http", "post", "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n" ..
    "Content-Length: 2\r\n\r\n{}GET / HTTP/1.0\r\n\r\n")
local body = "--XX\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper\r\n" ..
    "--XX\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n" ..
    "Content-Type: audio/wav\r\n\r\nRIFF\r\n--XX--\r\n"
put("http", "multipart", "POST /v1/audio/transcriptions HTTP/1.1\r\n" ..
    "Content-Type: multipart/form-data; boundary=\"XX\"\r\nContent-Length: " .. #body ..
    "\r\n\r\n" .. body)

-- safetensors
local function st(entries, meta)
	local parts, data, off = {}, {}, 0
	if meta then
		parts[#parts + 1] = '"__metadata__":' .. meta
	end
	for _, e in ipairs(entries) do
		local name, dtype, shape, bytes = e[1], e[2], e[3], e[4]
		parts[#parts + 1] = ('"%s":{"dtype":"%s","shape":[%s],"data_offsets":[%d,%d]}'):format(
		    name, dtype, table.concat(shape, ","), off, off + #bytes)
		data[#data + 1] = bytes
		off = off + #bytes
	end
	local h = "{" .. table.concat(parts, ",") .. "}"
	h = h .. string.rep(" ", -#h % 8)
	return string.pack("<I8", #h) .. h .. table.concat(data)
end
put("safetensors", "two", st({
	{ "a", "F32", { 2 }, string.pack("<ff", 1, 2) },
	{ "b.w", "F16", { 2, 1 }, string.pack("<I2I2", 0x3c00, 0x4000) },
}, '{"format":"pt"}'))
put("safetensors", "types", st({
	{ "x", "BF16", { 1 }, string.pack("<I2", 0x3f80) },
	{ "y", "I32", { 1, 1, 1, 1 }, string.pack("<i4", 7) },
	{ "z", "F64", { 1 }, string.pack("<d", 1) },
	{ "i8", "I8", { 3 }, "abc" },
}))
put("safetensors", "empty", string.pack("<I8", 2) .. "{}")

-- vocab: vocab.json, NUL, added_tokens.json
put("vocab", "bytes", '{"a":0,"b":1,"\\u0120c":2,"\\u00c4\\u00a2":3}')
put("vocab", "added", '{"he":0,"llo":1}\0{"<|endoftext|>":2}')

-- wav
local function wav(fmt, ch, rate, bits, samples)
	local block = ch * bits // 8
	local fmtc = string.pack("<I2I2I4I4I2I2", fmt, ch, rate, rate * block, block, bits)
	local chunk = function(id, s)
		return id .. string.pack("<I4", #s) .. s .. (#s % 2 == 1 and "\0" or "")
	end
	local body = "WAVE" .. chunk("fmt ", fmtc) .. chunk("LIST", "INFOx") .. chunk("data", samples)
	return "RIFF" .. string.pack("<I4", #body) .. body
end
put("wav", "s16", wav(1, 1, 16000, 16, string.pack("<i2i2i2i2", 0, 1000, -1000, 32767)))
put("wav", "stereo24", wav(1, 2, 44100, 24, string.rep("\1\2\3", 8)))
put("wav", "f32", wav(3, 1, 24000, 32, string.pack("<ffff", 0, 0.5, -0.5, 1)))
put("wav", "u8", wav(1, 1, 8000, 8, "\128\255\0\64"))

-- tokenize
put("tokenize", "chat", "<|im_start|>user\nhello world 123 xyz<|im_end|>\n<think>")
put("tokenize", "mixed", "Việt 我想 😶‍🌫️ \r\n\t  1234567 it's don't I'LL")
put("tokenize", "bad", "a\xffb\xc3")
put("tokenize", "gemma4", "<|turn>user\nhello world\n\n\n\n<turn|>\u{2581}Q \u{e9}\u{6211}<|\"|>")

-- image: first byte is the mode (see fuzz/image.lua)
package.path = (debug.getinfo(1, "S").source:match("^@(.*/)") or "./") .. "../test/?.lua;" .. package.path
local png = require("png")
local samples = require("image_samples")
local function ramp(x, y, k)
	return (x * 37 + y * 11 + k * 5) % 256
end
put("image", "gray8", "\0" .. png.encode({ w = 7, h = 5, depth = 8, ctype = 0, sample = ramp }))
put("image", "rgba16-adam7", "\0" .. png.encode({ w = 9, h = 6, depth = 16, ctype = 6, interlace = true,
	sample = function(x, y, k) return ramp(x, y, k) * 257 end }))
put("image", "gray1-adam7", "\0" .. png.encode({ w = 11, h = 9, depth = 1, ctype = 0, interlace = true,
	sample = function(x, y) return (x + y) % 2 end }))
put("image", "pal2-trns", "\1" .. png.encode({ w = 5, h = 4, depth = 2, ctype = 3,
	palette = "\255\0\0\0\255\0\0\0\255\9\9\9", trns = "\0\128",
	sample = function(x, y) return (x * y) % 4 end }))
put("image", "ga8", "\0" .. png.encode({ w = 3, h = 3, depth = 8, ctype = 4, sample = ramp }))
put("image", "rgb-key", "\0" .. png.encode({ w = 4, h = 2, depth = 8, ctype = 2, trns = "\0\1\0\2\0\3",
	sample = function(x, _, k) return x == 0 and k + 1 or 200 end }))
put("image", "wrap-rgb", "\2" .. string.char(2, 1, 8, 2, 0) .. png.zlib(string.rep("\1\5\6\7\8\9\10\11\12\13", 2)))
put("image", "wrap-pal", "\3" .. string.char(2, 2, 8, 3, 0) .. png.zlib(string.rep("\0\1\2\3", 3)))
put("image", "dynamic", "\0" .. samples.DYNAMIC_PNG)
put("image", "cmyk.jpg", "\0" .. samples.CMYK_JPEG)
put("image", "progressive.jpg", "\0" .. samples.PROGRESSIVE_JPEG)
put("image", "restart.jpg", "\0" .. samples.RESTART_JPEG)

-- grammar: GBNF, NUL, matcher input; or a JSON schema.
put("grammar", "json", 'root ::= "{" ws (pair ("," ws pair)*)? "}"\npair ::= str ":" ws val ws\n' ..
    'val ::= str | [0-9]+ | "true" | "null" | root\nstr ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""\n' ..
    'ws ::= [ \\n]{0,3}\0\1{\2"a\1"\1:\0001\1}')
put("grammar", "tokens", "root ::= <[26]> (!<[31]>)* <s> x\nx ::= [\\u00e4\\x30-\\x39]+ | \"😀\"?\0\3<s>ä\0010")
put("grammar", "repeat", "root ::= (\"ab\" | [^a-c]{2,4} . ){1,3} # c\r\n\0\5ab x yab")
put("grammar", "think", "root ::= lgml-think-0\nlgml-think-0 ::= \"\\x3C\" lgml-think-1 | [^\\x3C] lgml-think-0\n" ..
    "lgml-think-1 ::= \"\\x3E\" \"{}\" | \"\\x3C\" lgml-think-1 | [^\\x3C\\x3E] lgml-think-0\0\4ab<<>{}")
put("grammar", "schema", '{"type":"object","properties":{"a":{"type":"integer","minimum":-5,"maximum":300},' ..
    '"b":{"type":"array","items":{"$ref":"#/$defs/s"},"minItems":1,"maxItems":3},"c":{"anyOf":[{"const":1},' ..
    '{"type":"string","pattern":"^[a-z]{2}(x|y)?$"}]},"d":{"type":"string","format":"date"}},' ..
    '"required":["a"],"additionalProperties":{"type":"number"},"$defs":{"s":{"type":"string","maxLength":4}}}')
