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
-- @tparam string model the model id
-- @tparam table t llama.cpp-style timings from the engine
function M.chat(model, t)
	M.add("prompt_tokens_total", "prompt tokens processed", model, t.prompt_n)
	M.add("prompt_seconds_total", "time spent on prompt processing", model, (t.prompt_ms or 0) / 1000)
	M.add("predicted_tokens_total", "tokens generated", model, t.predicted_n)
	M.add("predicted_seconds_total", "time spent generating", model, (t.predicted_ms or 0) / 1000)
	M.add("cache_tokens_total", "prompt tokens reused from the cache", model, t.cache_n)
	M.add("restored_tokens_total", "prompt tokens restored from host memory", model, t.restored_n)
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
	return table.concat(out)
end

--- Forget every counter (for tests).
function M.reset()
	families, order = {}, {}
end

return M
