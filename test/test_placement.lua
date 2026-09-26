-- Weight placement rules: which tensors go to the CPU. Output is TAP.
local placement = require("lgml.placement")

local n = 0
local failed = false
local function ok(cond, name)
	n = n + 1
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
	if not cond then
		failed = true
	end
end

-- The tensors of a small MoE model: 4 layers, 100 bytes of experts and
-- 10 of other weights per layer, 50 for the embeddings, 40 for output.
local function tensors(tied)
	local ts = { { name = "token_embd.weight", size = 50 }, { name = "output_norm.weight", size = 1 } }
	if not tied then
		ts[#ts + 1] = { name = "output.weight", size = 40 }
	end
	for i = 0, 3 do
		local p = "blk." .. i .. "."
		for _, s in ipairs({ "attn_norm.weight", "attn_q.weight", "ffn_gate_inp.weight" }) do
			ts[#ts + 1] = { name = p .. s, size = s == "attn_norm.weight" and 2 or 4 }
		end
		ts[#ts + 1] = { name = p .. "ffn_gate_up_exps.weight", size = 60 }
		ts[#ts + 1] = { name = p .. "ffn_down_exps.weight", size = 40 }
		ts[#ts + 1] = { name = p .. "ffn_down_exps.scale", size = 1 }
		ts[#ts + 1] = { name = p .. "ffn_down_shexp.weight", size = 3 }
	end
	return ts
end

local function names(set)
	local list = {}
	for k in pairs(set or {}) do
		list[#list + 1] = k
	end
	table.sort(list)
	return table.concat(list, " ")
end

ok(placement.rules({}) == nil and placement.rules({ n_cpu_moe = 0 }) == nil,
    "no rules when nothing moves")

local r = placement.rules({ cpu_moe = true })
local set, bytes = placement.plan(r, tensors(), 4)
ok(bytes == 400 and set["blk.3.ffn_gate_up_exps.weight"] and set["blk.0.ffn_down_exps.weight"] and
    not set["blk.0.ffn_down_exps.scale"] and not set["blk.0.ffn_down_shexp.weight"] and
    not set["blk.0.ffn_gate_inp.weight"], "cpu_moe: the expert weights of every layer")

set, bytes = placement.plan(placement.rules({ n_cpu_moe = 2 }), tensors(), 4)
ok(bytes == 200 and set["blk.1.ffn_down_exps.weight"] and not set["blk.2.ffn_down_exps.weight"],
    "n_cpu_moe 2: the experts of layers 0 and 1")

local layers
set, bytes, layers = placement.plan(placement.rules({ n_gpu_layers = 3 }), tensors(), 4)
ok(names(set) == "blk.0.attn_norm.weight blk.0.attn_q.weight blk.0.ffn_down_exps.scale " ..
    "blk.0.ffn_down_exps.weight blk.0.ffn_down_shexp.weight blk.0.ffn_gate_inp.weight " ..
    "blk.0.ffn_gate_up_exps.weight output.weight output_norm.weight token_embd.weight",
    "n_gpu_layers 3: layer 0, the embeddings and the output on the CPU")
ok(layers[0] and not layers[1], "n_gpu_layers 3: layer 0 keeps its cache on the CPU")

set = placement.plan(placement.rules({ n_gpu_layers = 5 }), tensors(), 4)
ok(set == nil, "n_gpu_layers 5: all on the backend")
set = placement.plan(placement.rules({ n_gpu_layers = 4 }), tensors(), 4)
ok(names(set) == "output.weight output_norm.weight", "n_gpu_layers 4: only the output on the CPU")
set = placement.plan(placement.rules({ n_gpu_layers = 3 }), tensors(true), 4)
ok(set["token_embd.weight"], "tied embeddings follow the output")
set = placement.plan(placement.rules({ n_gpu_layers = 5, override = "blk%.0%.=CPU" }), tensors(true), 4)
ok(not set["token_embd.weight"] and set["blk.0.attn_q.weight"], "tied embeddings stay with the output")

-- Overrides: first match wins, before the other rules.
r = placement.rules({ override = { "down_exps=GPU", "exps=CPU" }, cpu_moe = true })
set = placement.plan(r, tensors(), 4)
ok(not set["blk.0.ffn_down_exps.weight"] and set["blk.0.ffn_gate_up_exps.weight"] and
    set["blk.0.ffn_down_shexp.weight"] == nil and set["blk.2.ffn_gate_up_exps.weight"],
    "overrides in order, before cpu_moe")
r = placement.rules({ override = "blk%.[01]%.ffn_gate_up=CPU,attn_q=cpu" })
set = placement.plan(r, tensors(), 4)
ok(names(set) == "blk.0.attn_q.weight blk.0.ffn_gate_up_exps.weight blk.1.attn_q.weight " ..
    "blk.1.ffn_gate_up_exps.weight blk.2.attn_q.weight blk.3.attn_q.weight",
    "an override string with two entries")
r = placement.rules({ override = { ["attn_q"] = "CPU" } })
ok(#r.override == 1 and r.override[1].cpu and r.override[1].pattern == "attn_q", "an override map")
ok(not pcall(placement.rules, { override = "exps" }), "an override without a backend is an error")
ok(not pcall(placement.rules, { override = "[x=CPU" }), "a bad pattern is an error")
ok(not pcall(placement.rules, { n_cpu_moe = -1 }), "a negative count is an error")

-- auto: the fewest expert layers on the CPU so the rest fits.
r = placement.rules({ n_cpu_moe = "auto" })
ok(r.auto and not r.n_cpu_moe, "n_cpu_moe auto")
local total = 0
for _, t in ipairs(tensors()) do
	total = total + t.size
end
local nn, dev = placement.auto(r, tensors(), 4, total)
ok(nn == 0 and dev == total, "auto: 0 when all fits")
nn, dev = placement.auto(r, tensors(), 4, total - 150)
ok(nn == 2 and dev == total - 200, "auto: 2 layers for 150 bytes less")
nn = placement.auto(r, tensors(), 4, 10)
ok(nn == 4, "auto: all layers when nothing fits")

print("1.." .. n)
if failed then
	os.exit(1)
end
