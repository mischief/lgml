-- Time chat requests through a running lgml-server with lgml-client:
-- one fixed long prompt per model, temperature 0, reasoning off. Prints
-- the server timings (prompt and generation tokens/s) of the median of
-- reps runs; a first run warms the model up.
-- usage: lua serve_bench.lua URL MODEL... [-n reps] [-t max_tokens]
local client = require("lgml.client")

local url, models, reps, max = arg[1], {}, 3, 256
local i = 2
while arg[i] do
	if arg[i] == "-n" then
		reps, i = tonumber(arg[i + 1]), i + 2
	elseif arg[i] == "-t" then
		max, i = tonumber(arg[i + 1]), i + 2
	else
		models[#models + 1], i = arg[i], i + 1
	end
end
assert(url and #models > 0, "usage: serve_bench.lua URL MODEL... [-n reps] [-t max_tokens]")

local para = "The river town kept its old stone bridge, though the mill below it had "
	.. "closed long ago. Each morning the baker opened at six, the ferry crossed at "
	.. "seven, and the school bell rang at eight. Visitors came for the market on "
	.. "Saturdays, when farmers sold apples, cheese, honey and wool from the hills. "
local doc = {}
for k = 1, 12 do
	doc[k] = ("Section %d. "):format(k) .. para
end
local prompt = table.concat(doc, "\n") .. "\nWrite a long, detailed story set in this town."

local c = assert(client.new({ url = url, timeout = 900 }))
local function median(t)
	table.sort(t)
	return t[(#t + 1) // 2]
end
for _, m in ipairs(models) do
	local pp, tg, pn, tn = {}, {}, 0, 0
	for rep = 0, reps do
		local r = assert(c:call("POST", "/v1/chat/completions", {
			model = m, max_tokens = max, temperature = 0, reasoning_effort = "none",
			-- the run number leads, so no run reuses the cached prompt of another
			messages = { { role = "user", content = ("Run %d.\n"):format(rep) .. prompt } },
		}))
		local t = r.timings
		if rep > 0 then
			pp[#pp + 1], tg[#tg + 1] = t.prompt_per_second, t.predicted_per_second
			pn, tn = t.prompt_n, t.predicted_n
		end
	end
	print(("%-16s pp%d %.1f t/s  tg%d %.2f t/s"):format(m, pn, median(pp), tn, median(tg)))
end
c:close()
