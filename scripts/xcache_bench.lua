-- Decode speed and expert cache hits of a MoE model through the engine.
-- Usage: lua5.4 scripts/xcache_bench.lua model.gguf mode [tokens]
-- mode: cpu, static:N (experts of N layers on the GPU), cache:MiB; the
-- rest of the model is on the CPU. gpu-static:N and gpu-cache:MiB put
-- the rest on the GPU. Environment: BACKEND, PROMPTS (a, b or all),
-- PROFILE (profile to start from), SAVE (profile out), HALFLIFE, SWAP.
local engine = require("lgml.engine")
local uv = require("luv")

local path, mode = arg[1], arg[2] or "cpu"
local max_tokens = tonumber(arg[3]) or 200

-- Two prompt sets: a profile from one, a measure on the other.
local SETS = {
	a = {
		"Write a Python function that parses a CSV file with quoted fields and returns a list of dicts. Explain each step.",
		"Tell a short story about a lighthouse keeper who finds a message in a bottle.",
		"Explain how a hash table handles collisions, with examples in C.",
	},
	b = {
		"Write a Lua module that keeps a least recently used cache with a fixed capacity, with tests.",
		"Describe the water cycle for a ten year old, then give five quiz questions about it.",
		"Compare TCP and UDP. When should a game use each one? Give concrete examples.",
	},
}

local opts = { model = path, n_ctx = 4096, slots = 1, cache_ram = 0, kv_initial = 4096 }
local kind, n = mode:match("^([%a-]+):?([%d.]*)$")
n = tonumber(n)
if kind == "cpu" then
	opts.backend = "CPU"
else
	opts.backend = os.getenv("BACKEND") or "Vulkan0"
	local all_cpu = not kind:match("^gpu%-")
	if all_cpu then
		opts.n_gpu_layers = 0
	end
	if kind:match("static$") then
		local list = {}
		for il = 0, n - 1 do
			list[#list + 1] = ("blk%%.%d%%.ffn_[%%a_]+exps=GPU"):format(il)
		end
		list[#list + 1] = "exps=CPU"
		opts.override = list
	elseif kind:match("cache$") then
		opts.moe_cache = n
		opts.moe_cache_profile = os.getenv("PROFILE")
		opts.moe_cache_halflife = tonumber(os.getenv("HALFLIFE"))
		opts.moe_cache_swap = tonumber(os.getenv("SWAP"))
	else
		error("bad mode " .. mode)
	end
end

-- NOPIN=1 keeps the host weights out of pinned device memory.
if os.getenv("NOPIN") then
	local transformer = require("lgml.transformer")
	local load = transformer.load
	transformer.load = function(be, file, o)
		if o then
			o.pinned = false
		end
		return load(be, file, o)
	end
end
local t0 = uv.hrtime()
local e = engine.new(opts)
local load_s = (uv.hrtime() - t0) / 1e9
local xc = e.model.xcache
local mem = require("lgml").memory()
local vram = 0
for name, m in pairs(mem) do
	if name ~= "CPU" then
		vram = vram + m.weights + m.tensors
	end
end
print(("mode %s: load %.1f s, device weights and tensors %d MiB%s"):format(mode, load_s,
    vram >> 20, xc and (", cache %d slots of %d, %d MiB"):format(xc.n, xc.ne, xc.bytes >> 20) or ""))

local function chat(text)
	local r, done, first = {}, false, nil
	local sink = {}
	function sink.content()
		if not first then
			first = uv.hrtime()
			if xc then
				r.prefill_hits = { xc:stats() }
				xc:reset()
			end
		end
		return true
	end
	sink.reasoning, sink.tool_arguments = sink.content, sink.content
	function sink.tool_call()
		return 0
	end
	function sink.finish(_, reason, usage, timings)
		r.reason, r.usage, r.timings = reason, usage, timings
	end
	function sink.cancelled()
		return false
	end
	coroutine.wrap(function()
		r.ok, r.err = e:chat({ messages = { { role = "user", content = text } },
		    max_tokens = max_tokens, top_k = 1, chat_template_kwargs = { enable_thinking = false } },
		    sink)
		done = true
	end)()
	while not done do
		uv.run("once")
	end
	return r
end

local want = os.getenv("PROMPTS") or "b"
local list = {}
for _, k in ipairs({ "a", "b" }) do
	if want == "all" or want == k then
		table.move(SETS[k], 1, #SETS[k], #list + 1, list)
	end
end
local toks, secs, hits, misses = 0, 0, 0, 0
for i, text in ipairs(list) do
	if xc then
		xc:reset()
	end
	local r = chat(text)
	local t = r.timings or {}
	local nd = (r.usage and r.usage.completion_tokens or 0) - 1
	local ds = (t.predicted_ms or 0) / 1e3
	local h, m, rate = 0, 0, 0
	if xc then
		h, m, rate = xc:stats()
	end
	print(("prompt %d: %d tokens, prefill %d tok at %.1f tok/s, decode %.2f tok/s, hit rate %.3f")
	    :format(i, nd + 1, r.usage and r.usage.prompt_tokens or 0, t.prompt_per_second or 0,
	    t.predicted_per_second or 0, rate))
	toks, secs, hits, misses = toks + nd, secs + ds, hits + h, misses + m
end
print(("total: decode %.2f tok/s over %d tokens%s"):format(secs > 0 and toks / secs or 0, toks,
    xc and (", hit rate %.3f, %d MiB moved"):format(hits / math.max(1, hits + misses),
    xc.moved >> 20) or ""))
for name, m in pairs(require("lgml").memory()) do
	print(("memory %s: weights %d MiB, tensors %d MiB, peak compute %d MiB, peak %d MiB"):format(
	    name, m.weights >> 20, m.tensors >> 20, m.peak_compute >> 20, m.peak >> 20))
end
if xc and os.getenv("SAVE") then
	xc:save(os.getenv("SAVE"))
	print("profile saved to " .. os.getenv("SAVE"))
end
