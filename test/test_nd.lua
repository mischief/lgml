-- Tests for lgml.nd on every device. TAP output.
local lgml = require("lgml")
local nd = require("lgml.nd")

local n, failed = 0, 0

local function ok(cond, name)
	n = n + 1
	if not cond then
		failed = failed + 1
	end
	print((cond and "ok " or "not ok ") .. n .. " - " .. name)
end

local function near(t, want)
	local got = nd.is_tensor(t) and t:get() or t
	if #got ~= #want then
		return false
	end
	for i = 1, #want do
		if math.abs(got[i] - want[i]) > 1e-3 then
			return false
		end
	end
	return true
end

local function same(a, b)
	return table.concat(a, ",") == table.concat(b, ",")
end

for _, dev in ipairs(lgml.devices()) do
	nd.device(dev.name)
	local tag = dev.name .. ": "
	local a = nd.tensor{ { 1, 2 }, { 3, 4 } }
	local b = nd.tensor{ { 5, 6 }, { 7, 8 } }
	ok(same(a:shape(), { 2, 2 }) and a:dtype() == "f32", tag .. "tensor from nested table")
	ok(near(a + b, { 6, 8, 10, 12 }), tag .. "add")
	ok(near(a - b, { -4, -4, -4, -4 }) and near(1 - a, { 0, -1, -2, -3 }), tag .. "sub")
	ok(near(a * b, { 5, 12, 21, 32 }) and near(a * 2, { 2, 4, 6, 8 }), tag .. "mul")
	ok(near(a / 2, { 0.5, 1, 1.5, 2 }) and near(2 / a, { 2, 1, 2 / 3, 0.5 }), tag .. "div")
	ok(near(-a, { -1, -2, -3, -4 }), tag .. "neg")
	ok(near(a + nd.tensor{ 10, 20 }, { 11, 22, 13, 24 }), tag .. "broadcast row")
	ok(near(a:mm(b), { 19, 22, 43, 50 }), tag .. "mm")
	ok(near(a:t(), { 1, 3, 2, 4 }), tag .. "transpose")
	ok(near(a[2], { 3, 4 }) and near(a:select(2, 1), { 1, 3 }), tag .. "select")
	ok(near(nd.arange(6):reshape(2, 3):narrow(2, 2, 2), { 1, 2, 4, 5 }), tag .. "narrow")
	ok(near(nd.arange(6):reshape(3, -1), { 0, 1, 2, 3, 4, 5 }) and
	    same(nd.arange(6):reshape(3, -1):shape(), { 3, 2 }), tag .. "reshape -1")
	ok(near(a:sum(), { 10 }) and near(a:sum(-1), { 3, 7 }) and near(a:mean(-1), { 1.5, 3.5 }),
	    tag .. "sum and mean")
	local s = nd.tensor{ { 0, 0 }, { 0, math.log(3) } }:softmax()
	ok(near(s, { 0.5, 0.5, 0.25, 0.75 }), tag .. "softmax")
	ok(near(a:cat(b, 1), { 1, 2, 3, 4, 5, 6, 7, 8 }) and near(a:cat(b, 2), { 1, 2, 5, 6, 3, 4, 7, 8 }),
	    tag .. "cat")
	ok(near(nd.zeros{ 2 } + 1, { 1, 1 }) and near(nd.full({ 2 }, 7), { 7, 7 }), tag .. "zeros and full")
	nd.manual_seed(7)
	local r1 = nd.randn{ 4 }:get()
	nd.manual_seed(7)
	ok(near(nd.randn{ 4 }, r1), tag .. "seeded randn repeats")
	local big = nd.randn{ 2000 }
	local m = big:mean():item()
	ok(math.abs(m) < 0.1, tag .. "randn mean near 0")
	local lin = nd.nn.Linear.new(nd.tensor{ { 1, 0 }, { 0, 1 }, { 1, 1 } }, nd.tensor{ 0, 0, 1 })
	ok(near(lin(nd.tensor{ { 2, 3 } }), { 2, 3, 6 }), tag .. "nn.Linear")
	ok(near(a:tolist()[2], { 3, 4 }), tag .. "tolist")
	-- Many ops before one read: the graph flushes on its own.
	local x = nd.zeros{ 4 }
	for _ = 1, 1000 do
		x = x + 1
	end
	ok(near(x, { 1000, 1000, 1000, 1000 }), tag .. "long chain flushes")
	ok(tostring(a):find("tensor%(%[%[1%.0000, 2%.0000%]") ~= nil, tag .. "print")
end

print("1.." .. n)
os.exit(failed == 0 and 0 or 1)
