-- Image roundtrip: text to image, then a vision model answers questions
-- about each image.
-- usage: lua5.4 scripts/roundtrip.lua gen dir [url]   make dir/NN.png
--        lua5.4 scripts/roundtrip.lua ask dir [url]   ask, write report
-- gen needs lgml-server with an image engine; ask needs one with a vision
-- model. The default url is http://127.0.0.1:8080. Needs curl.
local json = require("lgml.json")
local base64 = require("lgml.base64")

-- Each case has a prompt and questions. A question passes when the
-- answer holds one of the accepted words, ignoring case.
local CASES = {
	{ prompt = "three red apples on a blue plate, studio photo, plain white background",
	  q = { { "How many apples are in the image? Answer with one word.", { "three", "3" } },
	        { "What color is the plate? Answer with one word.", { "blue" } } } },
	{ prompt = "a cat wearing a yellow hat, watercolor painting",
	  q = { { "What animal is in the image? Answer with one word.", { "cat", "kitten" } },
	        { "What color is the hat? Answer with one word.", { "yellow" } } } },
	{ prompt = "a red stop sign on a street corner, daytime photo",
	  q = { { "What word is written on the sign? Answer with one word.", { "stop" } },
	        { "What color is the sign? Answer with one word.", { "red" } } } },
	{ prompt = "a green bicycle leaning against a brick wall",
	  q = { { "What vehicle is in the image? Answer with one word.", { "bicycle", "bike" } },
	        { "What color is the vehicle? Answer with one word.", { "green" } } } },
	{ prompt = "two dogs playing in snow, photo",
	  q = { { "How many dogs are in the image? Answer with one word.", { "two", "2" } },
	        { "What covers the ground? Answer with one word.", { "snow" } } } },
	{ prompt = "a lighthouse on a rocky coast at sunset, oil painting",
	  q = { { "What building is in the image? Answer with one word.", { "lighthouse" } },
	        { "Is this a photo or a painting? Answer with one word.", { "painting" } } } },
	{ prompt = "a bowl of ramen with a boiled egg and chopsticks, top-down photo",
	  q = { { "What dish is shown? Answer with one word.", { "ramen", "noodle" } },
	        { "What utensil is shown? Answer with one word.", { "chopstick" } } } },
	{ prompt = "a wooden sign that says OPEN hanging on a shop door",
	  q = { { "What word is written on the sign? Answer with one word.", { "open" } },
	        { "What material is the sign? Answer with one word.", { "wood" } } } },
	{ prompt = "an orange sports car on a race track, motion blur",
	  q = { { "What vehicle is in the image? Answer with one word.", { "car" } },
	        { "What color is the vehicle? Answer with one word.", { "orange" } } } },
	{ prompt = "four yellow rubber ducks in a bathtub",
	  q = { { "How many ducks are in the image? Answer with one word.", { "four", "4" } },
	        { "What color are the ducks? Answer with one word.", { "yellow" } } } },
}
local SEED = 42
local SIZE = "512x512"
local DESCRIBE = "Describe this image in one sentence."

local function quote(s)
	return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function readfile(path)
	local f = assert(io.open(path, "rb"))
	local s = f:read("a")
	f:close()
	return s
end

local function writefile(path, s)
	local f = assert(io.open(path, "wb"))
	f:write(s)
	f:close()
end

-- POST body to url and decode the JSON reply. The body goes through a
-- file, because an image in base64 is too long for a command line.
local function post(url, body, tmp)
	writefile(tmp, json.encode(body))
	local p = assert(io.popen(("curl -sS -H 'content-type: application/json' --data-binary @%s %s")
	    :format(quote(tmp), quote(url))))
	local out = p:read("a")
	p:close()
	local ok, r = pcall(json.decode, out)
	if not ok or type(r) ~= "table" then
		error("bad reply from " .. url .. ": " .. out:sub(1, 200))
	end
	if r.error then
		error(url .. ": " .. json.encode(r.error))
	end
	return r
end

local function gen(dir, url)
	os.execute("mkdir -p " .. quote(dir))
	for i, c in ipairs(CASES) do
		local t0 = os.time()
		local r = post(url .. "/v1/images/generations", {
			prompt = c.prompt, size = SIZE, seed = SEED, n = 1, response_format = "b64_json",
		}, dir .. "/req.json")
		writefile(("%s/%02d.png"):format(dir, i), assert(base64.decode(r.data[1].b64_json)))
		print(("%02d  %ds  %s"):format(i, os.time() - t0, c.prompt))
	end
	os.remove(dir .. "/req.json")
end

-- One chat turn with the image and a question. Greedy, short answer.
local function ask1(url, png, question, tmp)
	local r = post(url .. "/v1/chat/completions", {
		temperature = 0, max_tokens = 2048,
		messages = { { role = "user", content = {
			{ type = "image_url", image_url = { url = "data:image/png;base64," .. png } },
			{ type = "text", text = question },
		} } },
	}, tmp)
	return (r.choices[1].message.content or ""):gsub("^%s+", ""):gsub("%s+$", "")
end

local function ask(dir, url)
	local tmp = dir .. "/req.json"
	local report, pass, total = {}, 0, 0
	for i, c in ipairs(CASES) do
		local png = base64.encode(readfile(("%s/%02d.png"):format(dir, i)))
		local desc = ask1(url, png, DESCRIBE, tmp)
		local rows = {}
		print(("%02d  %s\n    %s"):format(i, c.prompt, desc))
		for _, q in ipairs(c.q) do
			local a = ask1(url, png, q[1], tmp)
			local low, good = a:lower(), false
			for _, w in ipairs(q[2]) do
				good = good or low:find(w, 1, true) ~= nil
			end
			total = total + 1
			pass = pass + (good and 1 or 0)
			rows[#rows + 1] = { question = q[1], answer = a, want = q[2], pass = good }
			print(("    %s  %s -> %s"):format(good and "ok  " or "fail", q[1], a))
		end
		report[i] = { image = ("%02d.png"):format(i), prompt = c.prompt, description = desc, questions = rows }
	end
	os.remove(tmp)
	writefile(dir .. "/report.json", json.encode({ pass = pass, total = total, cases = report }))
	print(("%d/%d questions pass"):format(pass, total))
end

local cmd, dir, url = arg[1], arg[2], arg[3] or "http://127.0.0.1:8080"
if not dir or (cmd ~= "gen" and cmd ~= "ask") then
	io.stderr:write("usage: roundtrip.lua gen|ask dir [url]\n")
	os.exit(2)
end
if cmd == "gen" then
	gen(dir, url)
else
	ask(dir, url)
end
