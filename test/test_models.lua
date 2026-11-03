-- Tests for lgml.server.models: spec checks, lgml-worker arguments,
-- config files and the managed pool, with a stub worker. TAP output.
local here = debug.getinfo(1, "S").source:match("^@(.*/)") or "./"
package.path = here .. "?.lua;" .. package.path
local tiny = require("tiny")
local models = require("lgml.server.models")

local n, failed = 0, 0
local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local MiB = 1048576
local path = tiny.tmp(".gguf")
tiny.write(path, "qwen3")

-- A stub worker hook: it records the arguments of each start.
local function hooks(gpu_free)
	local o = { starts = {}, names = {} }
	function o.start(spec, args, modes, name)
		o.starts[#o.starts + 1] = { spec = spec, args = args, modes = modes, name = name }
		return { id = spec.id, args = args }
	end
	function o.devices()
		return gpu_free and { { type = "gpu", free = gpu_free, total = gpu_free } } or {}
	end
	return o
end

-- The value after flag in args, or true for a flag with no value.
local function flag(args, f)
	for i, a in ipairs(args) do
		if a == f then
			local v = args[i + 1]
			return (v == nil or v:sub(1, 2) == "--") and true or v
		end
	end
end

local function err_of(spec)
	local good, err = pcall(models.engine, spec, hooks())
	return not good and tostring(err) or nil
end

-- Bad values are errors that name the key.
for _, c in ipairs({
	{ { ctx = "abc" }, "bad context size abc" },
	{ { speculative = 65 }, "bad speculative setting 65" },
	{ { speculative = 4, spec_min_p = 2 }, "bad spec_min_p 2" },
	{ { slots = 0 }, "bad slot count 0" },
	{ { ubatch = "big" }, "bad ubatch big" },
	{ { threads = -1 }, "bad thread count -1" },
	{ { lora = { { 1, 2 } } }, "bad lora entry 1" },
	{ { image_max_side = 64 }, "image_max_side needs engine lgml.image.engine" },
}) do
	local spec = c[1]
	spec.model = path
	local err = err_of(spec)
	ok(err and err:find(c[2], 1, true), "error: " .. c[2] .. " (" .. tostring(err) .. ")")
end

-- Worker arguments carry the spec.
do
	local o = hooks()
	local e = models.engine({ model = path, id = "a", ctx = 512, speculative = 4, spec_draft = 3,
		spec_min_p = 0.7, slots = 2, flash_attn = false, lora = "x.gguf:0.5",
		reasoning_budget_tokens = 64, n_cpu_moe = 2 }, o)
	local args = e.args
	ok(flag(args, "-m") == path and flag(args, "--alias") == "a" and flag(args, "-c") == "512",
	    "args: model, alias and context")
	ok(flag(args, "--spec") == "4" and flag(args, "--spec-draft") == "3" and flag(args, "--spec-min-p") == "0.7",
	    "args: speculation, draft length and min p")
	ok(flag(args, "--slots") == "2" and flag(args, "--no-fa") == true and flag(args, "--lora") == "x.gguf:0.5",
	    "args: slots, no flash attention and lora")
	ok(flag(args, "--reasoning-budget") == "64" and flag(args, "--n-cpu-moe") == "2",
	    "args: reasoning budget and expert layers on the CPU")
	ok(o.names["pc-a"] and o.starts[1].name == "pc-a", "the prompt store name goes to names and start")
	e = models.engine({ model = path, speculative = true }, hooks())
	ok(flag(e.args, "--spec") == "16", "speculative = true is 16 draft tokens")
	e = models.engine({ model = path, speculative = 0, spec_draft = 3 }, hooks())
	ok(not flag(e.args, "--spec") and not flag(e.args, "--spec-draft"), "speculative = 0 sends no spec flags")
	e = models.engine({ model = path, moe_cache = true, moe_cache_zerocopy = true }, hooks())
	ok(flag(e.args, "--moe-cache") == "auto" and flag(e.args, "--moe-cache-zerocopy") == true,
	    "moe_cache = true is auto")
	e = models.engine({ model = path, reasoning_budget_tokens = 0 }, hooks())
	ok(not flag(e.args, "--reasoning-budget"), "a reasoning budget of 0 sends none")
end

-- Config files.
do
	local cfg = tiny.tmp(".lua")
	local function write(s)
		local f = assert(io.open(cfg, "w"))
		f:write(s)
		f:close()
	end
	write("return { models = {} }")
	local good, err = pcall(models.load, cfg)
	ok(not good and err:find("must return { models", 1, true), "a config with no models is an error")
	write("return { models = { { model = 'x' } } } +")
	good, err = pcall(models.load, cfg)
	ok(not good and tostring(err):find(cfg, 1, true), "a config that does not parse is an error")
	write(("return { threads = 3, models = { { model = %q, id = 'one' }, { model = %q, id = 'two', ctx = 128 } } }")
	    :format(path, path))
	local c = models.load(cfg)
	ok(#c.models == 2, "load gives the models")
	os.remove(cfg)

	-- The managed pool: one pool model per spec, under the GPU budget.
	local o = hooks(100 * MiB + models.VRAM_RESERVE)
	local router = models.make({ C = cfg }, c, o)
	local list = router:models()
	ok(#list == 2 and list[1].id == "one" and list[2].id == "two", "a config gives a router of its models")
	ok(list[1].context_length == 256 and list[2].context_length == 128,
	    "a model lists the trained context, or its ctx")
	ok(list[1].status == "unloaded" and #o.starts == 0, "no worker starts before a request")
	ok(c.models[1].threads == 3 and c.models[2].threads == 3, "the threads key of the config goes to each spec")
	ok(o.names["pc-one"] and o.names["pc-two"], "each model has a prompt store name")

	c = { models = { { model = path, id = "x" }, { model = path, id = "x" } } }
	good, err = pcall(models.make, { C = "cfg" }, c, hooks())
	ok(not good and err:find("two models have the id x", 1, true), "two models with one id are an error")
	c = { models = { { model = 5 } } }
	good, err = pcall(models.make, { C = "cfg" }, c, hooks())
	ok(not good and err:find("models[1].model must be a path", 1, true), "a spec needs a model path")
	c = { vram = "lots", models = { { model = path } } }
	good, err = pcall(models.make, { C = "cfg" }, c, hooks())
	ok(not good and err:find("bad vram lots", 1, true), "a bad budget is an error")
end

-- Options: -m makes one worker engine; no engine option is an error.
do
	local o = hooks()
	local e = models.make({ m = path, alias = "solo", c = "64" }, {}, o)
	ok(e.id == "solo" and flag(e.args, "-c") == "64", "-m starts one worker with the flags")
	local good, err = pcall(models.make, {}, {}, hooks())
	ok(not good and err:find("no engine", 1, true), "no -m, -C or --fake is an error")
	ok(models.make({ fake = true }, {}, hooks()).models ~= nil, "--fake gives the fake engine")
end

os.remove(path)
print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
