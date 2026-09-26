--- Acoustic (sigma-VAE) decoder: latents (T, vae) to audio (T * 3200).
-- Activations are (T, C). ConvNeXt blocks use a causal depthwise conv
-- and a GELU feed-forward. Transposed convs upsample between stages.
-- @module lgml.vibevoice.decoder
local M = {}

local EPS = 1e-5

local function block_weights(w, p)
	local function get(name)
		return assert(w:get(p .. name), p .. name)
	end
	return {
		norm = get("norm.weight"), dw = get("dw_conv.weight"),
		dw_b = get("dw_conv.bias"), gamma = get("gamma"),
		ffn_ln = get("ffn_ln.weight"), up = get("ffn.up.weight"),
		up_b = get("ffn.up.bias"), down = get("ffn.down.weight"),
		down_b = get("ffn.down.bias"), ffn_gamma = get("ffn_gamma"),
	}
end

--- Collect the decoder weights.
-- The decoder runs the depths in reverse.
-- @tparam lgml.weights w loaded gguf weights
-- @tparam {int,...} depths encoder depths
-- @tparam {int,...} ratios encoder ratios
-- @treturn table decoder weights; pass them to prepare, then forward
-- @raise when a tensor is missing
function M.load(w, depths, ratios)
	local d = { stages = {}, ratios = ratios }
	d.stem_w = assert(w:get("at_dec.us.0.0.conv.weight"))
	d.stem_b = assert(w:get("at_dec.us.0.0.conv.bias"))
	d.head_w = assert(w:get("at_dec.head.weight"))
	d.head_b = assert(w:get("at_dec.head.bias"))
	for s = 0, #ratios do
		local st = { blocks = {} }
		for b = 0, depths[#depths - s] - 1 do
			st.blocks[b + 1] = block_weights(w, ("at_dec.s.%d.%d."):format(s, b))
		end
		if s > 0 then
			st.up_w = assert(w:get(("at_dec.us.%d.0.convtr.weight"):format(s)))
			st.up_b = assert(w:get(("at_dec.us.%d.0.convtr.bias"):format(s)))
		end
		d.stages[s + 1] = st
	end
	return d
end

--- Make device-side weight layouts in d.
-- Depthwise taps become (K, C) f32 rows. Transposed conv weights
-- become (OC * K, IC).
-- @tparam lgml.backend be backend
-- @tparam table d decoder weights from load
function M.prepare(be, d)
	local g <close> = be:graph()
	local outs = {}
	for _, st in ipairs(d.stages) do
		for _, b in ipairs(st.blocks) do
			local sh = b.dw:shape()
			local c, k = sh[1], sh[3]
			b.dw_t = be:tensor("f32", { k, c })
			local t = g:cont(g:transpose(g:reshape(g:cast(b.dw, "f32"), { c, k })))
			outs[#outs + 1] = g:cpy(t, b.dw_t)
		end
		if st.up_w then
			local sh = st.up_w:shape()
			local ic, oc, k = sh[1], sh[2], sh[3]
			st.k = k
			st.up_wp = be:tensor(st.up_w:type(), { oc * k, ic })
			local t = g:reshape(g:cont(g:permute(st.up_w, { 2, 3, 1 })), { oc * k, ic })
			outs[#outs + 1] = g:cpy(t, st.up_wp)
		end
	end
	g:compute(table.unpack(outs))
end

local function block(g, b, x)
	local h = g:mul(g:rms_norm(x, EPS), b.norm)
	h = g:vibevoice_dwconv(h, b.dw_t, b.dw_b)
	x = g:add(x, g:mul(h, b.gamma))
	h = g:mul(g:rms_norm(x, EPS), b.ffn_ln)
	h = g:gelu_erf(g:add(g:mul_mat(b.up, h), b.up_b))
	h = g:add(g:mul_mat(b.down, h), b.down_b)
	return g:add(x, g:mul(h, b.ffn_gamma))
end

--- Decoder graph.
-- @tparam lgml.graph g graph
-- @tparam table d prepared decoder weights
-- @tparam lgml.tensor z f32 latents (T, vae)
-- @treturn lgml.tensor audio (T * prod(ratios), 1)
function M.forward(g, d, z)
	local x = g:vibevoice_conv(z, d.stem_w, d.stem_b)
	for s, st in ipairs(d.stages) do
		if st.up_wp then
			x = g:vibevoice_convtr(x, st.up_wp, st.up_b, d.ratios[s - 1])
		end
		for _, b in ipairs(st.blocks) do
			x = block(g, b, x)
		end
	end
	return g:vibevoice_conv(x, d.head_w, d.head_b)
end

return M
