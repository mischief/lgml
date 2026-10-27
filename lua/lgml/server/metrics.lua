--- Server metrics.
-- Each finished request adds to per-model counters in the front
-- process. `render` gives them as Prometheus text for GET /metrics, and
-- `headers` turns one request's numbers into `x-lgml-*` headers.
-- @module lgml.server.metrics
local M = {}

-- name -> { help, type, values = { [model] = number } }
local families = {}
local order = {}

local function family(name, help, kind)
	local f = families[name]
	if not f then
		f = { help = help, kind = kind, values = {} }
		families[name] = f
		order[#order + 1] = name
	end
	return f
end

--- Add v to a counter.
-- @tparam string name the metric name, without the `lgml_` prefix
-- @tparam string help one line of help text
-- @tparam string model the model id label
-- @tparam number v the amount
function M.add(name, help, model, v)
	local f = family(name, help, "counter")
	f.values[model] = (f.values[model] or 0) + (v or 0)
end

--- Set a gauge.
-- @tparam string name the metric name, without the `lgml_` prefix
-- @tparam string help one line of help text
-- @tparam string model the model id label
-- @tparam number v the value
function M.set(name, help, model, v)
	local f = family(name, help, "gauge")
	f.values[model] = v
end

--- Count one request.
-- @tparam string model the model id
-- @tparam string kind "chat", "transcription", "speech", "decide" or "image"
-- @tparam boolean ok false when the request failed
function M.request(model, kind, ok)
	M.add(kind .. "_requests_total", kind .. " requests", model, 1)
	if not ok then
		M.add(kind .. "_errors_total", kind .. " requests that failed", model, 1)
	end
end

--- Count the timings of a finished chat request.
-- `repeat_stop` counts the replies that the repeat guard ended.
-- @tparam string model the model id
-- @tparam table t llama.cpp-style timings from the engine
function M.chat(model, t)
	M.add("prompt_tokens_total", "prompt tokens processed", model, t.prompt_n)
	M.add("prompt_seconds_total", "time spent on prompt processing", model, (t.prompt_ms or 0) / 1000)
	M.add("predicted_tokens_total", "tokens generated", model, t.predicted_n)
	M.add("predicted_seconds_total", "time spent generating", model, (t.predicted_ms or 0) / 1000)
	M.add("cache_tokens_total", "prompt tokens reused from the cache", model, t.cache_n)
	M.add("restored_tokens_total", "prompt tokens restored from host memory", model, t.restored_n)
	M.add("repeat_stops_total", "replies that ended on one token repeated", model, t.repeat_stop)
	if t.draft_n then
		M.add("draft_tokens_total", "speculative draft tokens verified", model, t.draft_n)
		M.add("draft_accepted_tokens_total", "speculative draft tokens kept", model, t.draft_accepted)
	end
end

-- Header value: integers as they are, other numbers with 3 decimals.
local function fmt(v)
	if math.type(v) == "integer" or (type(v) == "number" and v == math.floor(v) and
	    math.abs(v) < 2^53) then
		return ("%d"):format(v)
	elseif type(v) == "number" then
		return ("%.3f"):format(v)
	end
	return tostring(v)
end

--- Headers for one request: `x-lgml-` plus each key, with `_` as `-`.
-- @tparam table fields name -> number or string
-- @tparam[opt] table into a header table to add to
-- @treturn table the headers
function M.headers(fields, into)
	into = into or {}
	for k, v in pairs(fields or {}) do
		if type(v) == "number" or type(v) == "string" then
			into["x-lgml-" .. k:gsub("_", "-")] = fmt(v)
		end
	end
	return into
end

-- Label value with Prometheus escapes.
local function label(s)
	return (tostring(s):gsub("[\\\"\n]", { ["\\"] = "\\\\", ['"'] = '\\"', ["\n"] = "\\n" }))
end

-- name -> function, and the names in order of first use
local collectors, corder = {}, {}

--- Set a function that gives gauges at each render, such as the state
-- of the slots now. A second call with the same key replaces it.
-- @tparam string key the name of the collector
-- @tparam function fn returns a list of samples: `{ name, help, model,
-- labels, value }`; name has no `lgml_` prefix, and labels is an
-- optional table of more label names and values
function M.collect(key, fn)
	if not collectors[key] then
		corder[#corder + 1] = key
	end
	collectors[key] = fn
end

-- The labels of a sample: model first, then the others by name.
local function labels(model, more)
	local t = { ('model="%s"'):format(label(model)) }
	local keys = {}
	for k in pairs(more or {}) do
		keys[#keys + 1] = k
	end
	table.sort(keys)
	for _, k in ipairs(keys) do
		t[#t + 1] = ('%s="%s"'):format(k, label(more[k]))
	end
	return table.concat(t, ",")
end

-- The samples of the collectors, by family in order of first use.
local function collected(out)
	local names, by = {}, {}
	for _, key in ipairs(corder) do
		for _, x in ipairs(collectors[key]()) do
			if not by[x.name] then
				names[#names + 1] = x.name
				by[x.name] = { help = x.help, rows = {} }
			end
			local rows = by[x.name].rows
			rows[#rows + 1] = ("lgml_%s{%s} %s\n"):format(x.name, labels(x.model, x.labels), fmt(x.value))
		end
	end
	for _, name in ipairs(names) do
		out[#out + 1] = ("# HELP lgml_%s %s\n# TYPE lgml_%s gauge\n"):format(name, by[name].help, name)
		table.move(by[name].rows, 1, #by[name].rows, #out + 1, out)
	end
end

--- The metrics in the Prometheus text format.
-- @treturn string the text
function M.render()
	local out = {}
	for _, name in ipairs(order) do
		local f = families[name]
		local full = "lgml_" .. name
		out[#out + 1] = ("# HELP %s %s\n# TYPE %s %s\n"):format(full, f.help, full, f.kind)
		local models = {}
		for m in pairs(f.values) do
			models[#models + 1] = m
		end
		table.sort(models)
		for _, m in ipairs(models) do
			out[#out + 1] = ('%s{model="%s"} %s\n'):format(full, label(m), fmt(f.values[m]))
		end
	end
	collected(out)
	return table.concat(out)
end

--- Forget every counter and collector (for tests).
function M.reset()
	families, order, collectors, corder = {}, {}, {}, {}
end

return M
