--- The Qwen3.8-Flash-Next text model (gguf arch qwen4exp).
-- The trunk keeps hc residual streams (hyper-connections). Each layer
-- mixes the streams into one input for a gated delta net or a QSA
-- layer, and again for routed experts with a shared expert. QSA is
-- attention over the cells of the best scored blocks of compress_ratio
-- cells, and the incomplete block at the end. One delta net layer
-- first adds n-gram hash embeddings (PLE). `lgml.transformer` calls
-- this module for the arch; its caches and passes work as for qwen35.
-- @module lgml.qwen4exp
local lgml = require("lgml")

local M = {}

-- The transformer module, loaded by the time a model exists.
local function T()
	return package.loaded["lgml.transformer"]
end

--- Read the qwen4exp parts of the config.
-- It sets hp.hc (stream count), hp.hc_rank, hp.gdn_gate, hp.idx (the
-- QSA indexer: heads, dim, top_k, ratio; nil without one) and hp.ple
-- (the n-gram embedding: layer from 1, n, per, heads, dim, kern, hist,
-- eos, image, mult, off, vocab; nil without one). A QSA layer gets
-- `idx` (the indexer key size), the PLE layer `ple`.
-- @tparam table hp the config so far, with layers
-- @tparam function k reads a key without the arch prefix, with a default
-- @raise on metadata that the model cannot use
function M.config(hp, k)
	local a = hp.arch
	hp.hc = k("hyper_connection.count")
	hp.hc_rank = k("hyper_connection.low_rank")
	if not hp.hc or hp.hc <= 1 or not hp.hc_rank or hp.hc_rank < 1 then
		error(a .. ": need hyper_connection.count above 1 and hyper_connection.low_rank")
	end
	hp.gdn_gate = "sigmoid"
	local ratios = k("attention.compress_ratios", 0)
	local idx = { heads = k("attention.indexer.head_count"), dim = k("attention.indexer.key_length"),
		top_k = k("attention.indexer.top_k"), ratio = 0 }
	for i, d in ipairs(hp.layers) do
		local r = type(ratios) == "table" and (ratios[i] or 0) or ratios
		if r > 0 and not d.recr then
			if idx.ratio ~= 0 and r ~= idx.ratio then
				error(("%s: QSA layers must share one compress ratio, got %d and %d"):format(a, idx.ratio, r))
			end
			idx.ratio = r
		end
	end
	if idx.ratio > 0 then
		if not idx.heads or not idx.dim or not idx.top_k or idx.ratio == 1 or idx.top_k % idx.ratio ~= 0 then
			error(a .. ": QSA needs indexer sizes and a compress ratio above 1 that divides top_k")
		end
		hp.idx = idx
		for i, d in ipairs(hp.layers) do
			local r = type(ratios) == "table" and (ratios[i] or 0) or ratios
			if r > 0 and not d.recr then
				d.idx = idx.dim
			end
		end
	end
	local pl = k("ple.layers")
	if type(pl) ~= "table" or #pl == 0 then
		return
	end
	if #pl ~= 1 then
		error(a .. ": only one PLE layer is supported")
	end
	local p = {
		layer = pl[1] + 1, n = k("ple.ngram_size"), per = k("ple.heads_per_ngram"),
		kern = k("ple.conv_kernel"), eos = k("ple.eos_token_id"), dim = k("embedding_length_per_layer_input"),
		mult = k("ple.layer_multipliers"), off = k("ple.head_offsets"), vocab = k("ple.head_vocab_sizes"),
	}
	local img = k("ple.image_token_id", 0)
	p.image = img ~= 0 and img or p.eos
	if not p.n or p.n < 2 or p.n > 8 or not p.per or not p.kern or p.kern < 1 or not p.dim then
		error(a .. ": bad PLE sizes")
	end
	p.heads = (p.n - 1) * p.per
	p.hist = (p.kern - 1) * p.n
	if p.heads * p.dim ~= hp.n_embd then
		error(a .. ": PLE heads times head size must be the embedding length")
	end
	if #p.mult < p.n or #p.off < p.heads or #p.vocab < p.heads then
		error(a .. ": PLE hash arrays are too short")
	end
	local d = hp.layers[p.layer]
	if not d or not d.recr then
		error(a .. ": the PLE layer must be a delta net layer")
	end
	d.ple = true
	hp.ple = p
end

--- Load the weights of one layer.
-- @param w the weights (anything with get)
-- @tparam table hp the config
-- @tparam integer i the layer index, from 1
-- @treturn table the layer
-- @raise when a required weight is missing
function M.load_layer(w, hp, i)
	local p = "blk." .. (i - 1) .. "."
	local function get(name)
		return assert(w:get(p .. name), p .. name)
	end
	local d = hp.layers[i]
	local L = { recr = d.recr, D = d.D, Hkv = d.Hkv, n_rot = d.n_rot, base = d.base, idx = d.idx, ple = d.ple }
	for _, part in ipairs({ "attn", "ffn" }) do
		for _, f in ipairs({ "norm", "down", "up", "inject" }) do
			L["hc_" .. part .. "_" .. f] = get("hc_" .. part .. "_" .. f .. ".weight")
		end
	end
	if d.recr then
		L.qkv = get("attn_qkv.weight")
		L.qkv_gate = get("attn_gate.weight")
		L.beta = get("ssm_beta.weight")
		L.alpha = get("ssm_alpha.weight")
		L.conv = get("ssm_conv1d.weight")
		L.dt = get("ssm_dt.bias")
		L.a = get("ssm_a")
		L.ssm_norm = get("ssm_norm.weight")
		L.ssm_out = get("ssm_out.weight")
	else
		L.wq = get("attn_q.weight")
		L.wk = get("attn_k.weight")
		L.wv = get("attn_v.weight")
		L.wo = get("attn_output.weight")
		L.q_norm = get("attn_q_norm.weight")
		L.k_norm = get("attn_k_norm.weight")
		if d.idx then
			L.idx_q = get("indexer.q_proj.weight")
			L.idx_k = get("indexer.k_proj.weight")
			L.idx_q_norm = get("indexer.q_norm.weight")
			L.idx_k_norm = get("indexer.k_norm.weight")
		end
	end
	if d.ple then
		for _, f in ipairs({ "key", "value", "norm_key", "norm_query", "norm_conv", "conv1d" }) do
			L["ple_" .. f] = get("ple_" .. f .. ".weight")
		end
	end
	L.gate_inp = get("ffn_gate_inp.weight")
	L.gate_up_exps = w:get(p .. "ffn_gate_up_exps.weight")
	if not L.gate_up_exps then
		L.gate_exps = get("ffn_gate_exps.weight")
		L.up_exps = get("ffn_up_exps.weight")
	end
	L.down_exps = get("ffn_down_exps.weight")
	L.shexp_gate = w:get(p .. "ffn_gate_inp_shexp.weight")
	if L.shexp_gate then
		L.gate_shexp = get("ffn_gate_shexp.weight")
		L.up_shexp = get("ffn_up_shexp.weight")
		L.down_shexp = get("ffn_down_shexp.weight")
	end
	return L
end

-- Weights from several stores: get looks in each in turn.
local Multi = {}
Multi.__index = Multi

function Multi:get(name)
	for _, s in ipairs(self) do
		local t = s:get(name)
		if t then
			return t
		end
	end
end

function Multi:size()
	local dev, host = 0, 0
	for _, s in ipairs(self) do
		local d, h = s:size()
		dev, host = dev + d, host + (h or 0)
	end
	return dev, host
end

function Multi:close()
	for _, s in ipairs(self) do
		s:close()
	end
end

-- The files of a split gguf, from the path of any of them.
local function shards(path, meta)
	local n = meta["split.count"] or 1
	if n <= 1 then
		return { path }
	end
	local head, tail = path:match("^(.*%-)%d%d%d%d%d%-of%-%d%d%d%d%d(%.gguf)$")
	if not head then
		error(path .. ": a split gguf needs the -NNNNN-of-NNNNN.gguf name")
	end
	local list = {}
	for i = 1, n do
		list[i] = ("%s%05d-of-%05d%s"):format(head, i, n, tail)
	end
	return list
end

-- The name of the n-gram table.
local TABLE = "per_layer_token_embd.weight"

--- Load a model of this arch.
-- It reads every file of a split gguf. The n-gram table stays in its
-- file: an `lgml.ngram` reader in model.ngram reads the rows that a
-- pass needs. With opts.ngram false, a map of the table serves
-- `get_rows` on the CPU instead, or the table loads when the backend
-- has no CPU. MTP layers past n_layer do not load.
-- @param be the backend
-- @tparam string path a gguf file of the model
-- @tparam table opts the options of `lgml.transformer.load`, and
-- opts.ngram (false for get_rows on the table) and opts.mmap (true to
-- map the weights on the CPU backend in place of a load)
-- @tparam table info the gguf info of path
-- @tparam table hp the config
-- @treturn Model the model, with the metatable of `lgml.transformer`
function M.load(be, path, opts, info, hp)
	local placement = require("lgml.placement")
	local cpu = opts.placement and placement.plan(opts.placement, info.tensors, hp.n_layer)
	if cpu and not be:cpu() then
		error(path .. ": the placement moves weights to the CPU; open the backend with cpu = true")
	end
	local w = setmetatable({}, Multi)
	local files = shards(path, info.meta)
	local mapped, reader = false, nil
	for _, file in ipairs(files) do
		local finfo = file == path and info or assert(lgml.gguf(file))
		local skip = {}
		for _, t in ipairs(finfo.tensors) do
			local il = tonumber(t.name:match("^blk%.(%d+)%."))
			if il and il >= hp.n_layer then
				skip[t.name] = true
			end
		end
		local has_table = false
		for _, t in ipairs(finfo.tensors) do
			has_table = has_table or t.name == TABLE
		end
		if has_table and hp.ple and opts.ngram ~= false then
			local err
			reader, err = lgml.ngram.open(file, { meta = files[1] })
			if not reader then
				error(err)
			end
			skip[TABLE] = true
		elseif has_table and hp.ple then
			local ok, s = pcall(be.map, be, file, { TABLE }, { random = true })
			if ok then
				w[#w + 1] = s
				skip[TABLE] = true
				mapped = true
			end
		end
		if opts.mmap then
			local names = {}
			for _, t in ipairs(finfo.tensors) do
				if not skip[t.name] then
					names[#names + 1] = t.name
				end
			end
			w[#w + 1] = be:map(file, names)
		else
			w[#w + 1] = assert(be:load(file, { cpu = cpu, pinned = opts.pinned, skip = skip }))
		end
	end
	local function get(name)
		return assert(w:get(name), name)
	end
	local m = setmetatable({ be = be, w = w, hp = hp, meta = info.meta }, T())
	m.tok_embd = get("token_embd.weight")
	m.output = w:get("output.weight") or m.tok_embd
	m.hc_head_norm = get("output_hc_norm.weight")
	m.hc_head_down = get("output_hc_down.weight")
	m.hc_head_up = get("output_hc_up.weight")
	hp.n_vocab = hp.n_vocab or m.output:shape()[1]
	if reader then
		m.ngram = reader
	elseif hp.ple then
		m.ple_table = get(TABLE)
		m.ple_mapped = mapped
	end
	m.layers = {}
	for i = 1, hp.n_layer do
		local L = M.load_layer(w, hp, i)
		m.recurrent = m.recurrent or L.recr or nil
		L.cpu = L.hc_attn_norm:backend() ~= be or nil
		m.layers[i] = L
	end
	if opts.replay == false then
		m.replay_key = false
	end
	return m
end

-- The CPU backend that holds the token ids of the cells.
local function host(self)
	if not self.host then
		self.host = self.be:cpu() or (self.be:name():upper():match("^CPU") and self.be) or
		    assert(lgml.backend("CPU"))
	end
	return self.host
end

--- Add the qwen4exp parts of a cache: indexer keys of the QSA layers
-- and token ids (with global true), and the PLE conv state.
-- Extra row tensors go in c.x: x[il] holds the f16 indexer keys of
-- layer il, x[0] the i32 token ids of the cells (on the CPU).
-- @tparam Model self the model
-- @tparam table c the cache
-- @tparam integer rows the rows of the global layers
-- @tparam boolean global true when the cache holds the global rows
function M.add_cache(self, c, rows, global)
	local hp = self.hp
	c.x = c.x or {}
	if c.conv then
		c.pconv = c.pconv or {}
	end
	for il, L in ipairs(self.layers) do
		local lbe = L.cpu and self.be:cpu() or self.be
		if L.ple and c.pconv then
			c.pconv[il] = lbe:tensor("f32", { 1, hp.hc * hp.n_embd, hp.ple.hist })
			c.pconv[il]:zero()
		end
		if L.idx and global then
			c.x[il] = lbe:tensor("f16", { rows, L.idx })
			c.x[il]:zero()
		end
	end
	if hp.ple and global then
		c.x[0] = host(self):tensor("i32", { rows, 1 })
		c.x[0]:zero()
	end
end

-- x mod m, with x read as an unsigned 64-bit integer.
local function umod(x, m)
	if x >= 0 then
		return x % m
	end
	return ((x >> 1) % m * 2 + (x & 1)) % m
end

--- Write the n-gram table rows of the tokens of a pass, as llama.cpp
-- hashes them. A token's context is it and the n - 1 tokens before
-- it; a missing token, or an EOS before the token, makes it and all
-- older tokens EOS. Head h of an m-gram reads row mixed % vocab[h] +
-- off[h], where mixed xors each context token times its multiplier.
-- @tparam table p hp.ple
-- @tparam {integer,...} prev the tokens before ids[1], oldest first
-- (at most n - 1; fewer at the sequence start)
-- @tparam {integer,...} ids the tokens of the pass
-- @tparam table out gets heads rows per token, token-major
-- @tparam[opt=0] integer off the entries of out before the first row
function M.ngram_rows(p, prev, ids, out, off)
	off = off or 0
	local n, H, per, eos, np = p.n, p.heads, p.per, p.eos, #prev
	local ctx = {}
	for i = 1, #ids do
		ctx[1] = ids[i]
		local cut = false
		for s = 1, n - 1 do
			local t
			if not cut then
				if i > s then
					t = ids[i - s]
				else
					t = prev[np + i - s]
				end
			end
			cut = cut or t == nil or t < 0 or t == eos
			ctx[s + 1] = cut and eos or t
		end
		local base = off + (i - 1) * H
		for m = 2, n do
			local mixed = ctx[1] * p.mult[1]
			for j = 2, m do
				mixed = mixed ~ (ctx[j] * p.mult[j])
			end
			for h = (m - 2) * per + 1, (m - 1) * per do
				out[base + h] = umod(mixed, p.vocab[h]) + p.off[h]
			end
		end
	end
	return out
end

--- Make the qwen4exp input leaves of a pass of n tokens.
-- The n-gram input is ple_rows (i32, n * heads rows of the table) or,
-- with a reader in model.ngram, ple_embd (f32, n by n_embd). The
-- reader's `embed(ids, prev, out, off)` writes the table rows of ids
-- into out at byte offset off, one row of n_embd per token; prev lists
-- the tokens before ids[1], oldest first (n - 1 of them, fewer at the
-- start of a sequence); see `lgml.ngram`. qsa_w holds the indexer head
-- weights.
-- @tparam Model self the model
-- @tparam table inp the inputs
-- @tparam integer n the tokens of the pass
function M.leaves(self, inp, n)
	local hp, be = self.hp, self.be
	if hp.ple then
		if self.ngram then
			inp.ple_embd = be:tensor("f32", { n, hp.ple.heads * hp.ple.dim })
		else
			inp.ple_rows = be:tensor("i32", { n * hp.ple.heads })
		end
	end
	local I = hp.idx
	if I then
		local w = {}
		for i = 1, n * I.heads do
			w[i] = 1 / math.sqrt(I.dim)
		end
		inp.qsa_w = be:tensor("f32", { n, I.heads }, w)
	end
end

--- Make the QSA block positions leaf of one sequence, qsa_pos: the
-- position of the first cell of each block of the attention span. A
-- span with at most top_k / ratio blocks needs none.
-- @tparam Model self the model
-- @tparam table b the inputs or a batch entry
-- @tparam integer span the cells that attention reads
function M.seq_leaves(self, b, span)
	local I = self.hp.idx
	if I and span // I.ratio > I.top_k // I.ratio then
		b.qsa_pos = self.be:tensor("i32", { span // I.ratio })
	end
end

-- Pack integers as i32 bytes.
local function pack_i32(list, i, j)
	local parts = {}
	for k = i, j do
		parts[#parts + 1] = string.pack("<i4", list[k])
	end
	return table.concat(parts)
end

--- Fill the qwen4exp inputs of one sequence and store its token ids.
-- Call `finish` after the last sequence of a pass.
-- @tparam Model self the model
-- @tparam table inp the inputs, from `leaves`
-- @tparam table b the inputs or batch entry, from `seq_leaves`
-- @tparam table c the cache
-- @tparam {integer,...} ids the tokens of the sequence
-- @tparam integer n_past the position of ids[1]
-- @tparam integer lo the first cell of the attention span
-- @tparam integer off the tokens of the pass before ids[1]
function M.fill(self, inp, b, c, ids, n_past, lo, off)
	local hp = self.hp
	local p = hp.ple
	local n = #ids
	if p then
		if inp.embd then
			local img = {}
			for i = 1, n do
				img[i] = p.image
			end
			ids = img
		end
		local tok = (c.pool or c).x[0]
		local np = math.min(n_past, p.n - 1)
		local cells = T().cells(c, n_past - np, n + np)
		local buf = lgml.hostbuf(4 * #cells)
		local prev = {}
		for i = 1, np do
			tok:to_host(buf, 4 * (i - 1), 4 * cells[i], 4)
		end
		if np > 0 then
			local s = buf:bytes(0, 4 * np)
			for i = 1, np do
				prev[i] = string.unpack("<i4", s, 4 * i - 3)
			end
		end
		-- Store the tokens, in runs of consecutive cells.
		buf:set(0, pack_i32(ids, 1, n))
		local i = 1
		while i <= n do
			local j = i
			while j < n and cells[np + j + 1] == cells[np + j] + 1 do
				j = j + 1
			end
			tok:from_host(buf, 4 * (i - 1), 4 * cells[np + i], 4 * (j - i + 1))
			i = j + 1
		end
		buf:close()
		if inp.ple_embd then
			self.ngram:embed(ids, prev, inp.ple_embd, off * p.heads * lgml.row_size("f32", p.dim))
		else
			inp.ple_list = inp.ple_list or {}
			M.ngram_rows(p, prev, ids, inp.ple_list, off * p.heads)
		end
	end
	if b.qsa_pos then
		local R = hp.idx.ratio
		local P = b.qsa_pos:shape()[1]
		local pos = {}
		if c.pool then
			local B = c.pool.block
			local at = {}
			for i, blk in ipairs(c.blocks) do
				at[blk] = i - 1
			end
			for j = 1, P do
				local cell = lo + (j - 1) * R
				local i = at[cell // B]
				pos[j] = i and i * B + cell % B or 0
			end
		else
			for j = 1, P do
				pos[j] = (j - 1) * R
			end
		end
		b.qsa_pos:set(pos)
	end
end

-- Load the n-gram rows that `fill` gathered into their leaf.
local function finish(inp)
	if inp.ple_list then
		inp.ple_rows:set(inp.ple_list)
		inp.ple_list = nil
	end
end

--- Fill the qwen4exp inputs of a pass of one sequence at cache.n_past.
-- @tparam Model self the model
-- @tparam table inp the inputs
-- @tparam table cache the cache
-- @tparam {integer,...} ids the tokens
function M.pass(self, inp, cache, ids)
	M.fill(self, inp, inp, cache, ids, cache.n_past, inp.lo, 0)
	finish(inp)
end

--- Fill the qwen4exp inputs of a batch pass: sequence i runs ids[i]
-- at its cache.n_past.
-- @tparam Model self the model
-- @tparam table inp the batch inputs
-- @tparam {table,...} ids a list of token ids per sequence
function M.batch(self, inp, ids)
	for i, b in ipairs(inp.batch) do
		M.fill(self, inp, b, b.cache, ids[i], b.cache.n_past, b.lo, b.off)
	end
	finish(inp)
end

-- Mix the hc streams x (n, hc, E) into one (n, E), and give the
-- (n, hc) inject weights when wi is set.
local function hc_mix(g, hp, x, wn, wd, wu, wi)
	local n, E, HC = x:shape()[1], hp.n_embd, hp.hc
	local xn = g:mul(g:rms_norm(x, hp.eps), g:reshape(wn, { HC, E }))
	xn = g:reshape(xn, { n, HC * E })
	local lo = g:silu(g:scale(g:mul_mat(wd, xn), 1 / HC))
	local gate = g:mul_mat(wu, lo)
	local gated = g:reshape(g:mul(xn, g:sigmoid(gate)), { n, HC, E })
	local st = gated:strides()
	local mixed = g:cont(g:view(gated, { n, E }, { st[1] }))
	for s = 1, HC - 1 do
		mixed = g:add(mixed, g:view(gated, { n, E }, { st[1] }, s * st[2]))
	end
	mixed = g:scale(mixed, 1 / HC)
	return mixed, wi and g:mul_mat(wi, xn)
end

-- Add the block output o (n, E) to each stream of res (n, hc, E),
-- scaled by 2 * sigmoid(inject / hc).
local function hc_add(g, hp, res, o, inj)
	local n, E, HC = res:shape()[1], hp.n_embd, hp.hc
	local w = g:scale(g:sigmoid(g:scale(inj, 1 / HC)), 2)
	local b = g:tile(g:reshape(o, { n, 1, E }), res)
	return g:add(res, g:mul(b, g:reshape(w, { n, HC, 1 })))
end

-- The PLE causal conv of one sequence: x (n, C) after state (1, C,
-- hist), taps dilated by the n-gram size. The graph writes the new
-- state back.
local function ple_conv(g, hp, L, state, x)
	local p = hp.ple
	local n, C = x:shape()[1], x:shape()[2]
	local f4 = lgml.row_size("f32", 1)
	local xin = g:concat(state, g:transpose(x), 3)
	local s = xin:strides()
	g:expand(g:cpy(g:view(xin, { 1, C, p.hist }, { s[1], s[2] }, n * f4), state))
	local wel = lgml.row_size(L.ple_conv1d:type(), 1)
	local out
	for k = 0, p.kern - 1 do
		local sh = g:cont(g:transpose(g:view(xin, { C, n }, { s[2] }, k * p.n * f4)))
		local wk = g:cont(g:view(L.ple_conv1d, { C, 1 }, { p.kern * wel }, k * wel))
		wk = g:reshape(wk, { C })
		if wk:type() ~= "f32" then
			wk = g:cast(wk, "f32")
		end
		local t = g:mul(sh, wk)
		out = out and g:add(out, t) or t
	end
	return g:silu(out)
end

-- Rows off+1 .. off+n of an (N, ...) node.
local function rows_of(g, t, off, n)
	local sh, st = t:shape(), t:strides()
	local shape, strides = { n }, {}
	for i = 2, #sh do
		shape[i] = sh[i]
	end
	for i = 1, #sh - 1 do
		strides[i] = st[i]
	end
	return g:view(t, shape, strides, off * st[1])
end

-- The PLE block: gated n-gram values and their conv, added to every
-- stream of res.
local function ple_block(self, g, inp, L, res)
	local hp = self.hp
	local p, E, HC = hp.ple, hp.n_embd, hp.hc
	local n = res:shape()[1]
	local emb = inp.ple_embd
	if not emb then
		-- Reference path: rows of the table in this graph. The reader
		-- of model.ngram replaces it with ple_embd.
		emb = g:reshape(g:get_rows(self.ple_table, inp.ple_rows), { n, p.heads * p.dim })
	end
	local function gn(x, w)
		return g:mul(g:rms_norm(g:reshape(x, { n, HC, E }), hp.eps), g:reshape(w, { HC, E }))
	end
	local key = gn(g:mul_mat(L.ple_key, emb), L.ple_norm_key)
	local value = g:mul_mat(L.ple_value, emb)
	local query = gn(res, L.ple_norm_query)
	local s = g:scale(g:sum_rows(g:mul(key, query)), 1 / math.sqrt(E))
	local mag = g:sqrt(g:clamp(g:abs(s), 1e-6, math.huge))
	local gate = g:sigmoid(g:mul(g:sgn(s), mag))
	local gated = g:mul(g:tile(g:reshape(value, { n, 1, E }), res), gate)
	local normed = g:reshape(gn(gated, L.ple_norm_conv), { n, HC * E })
	local il = L.il
	local conv
	if inp.batch then
		for _, b in ipairs(inp.batch) do
			local o = ple_conv(g, hp, L, b.cache.pconv[il], g:cont(rows_of(g, normed, b.off, b.n)))
			conv = conv and g:concat_qwen3tts(conv, o, 1) or o
		end
	else
		conv = ple_conv(g, hp, L, inp.cache.pconv[il], normed)
	end
	return g:add(res, g:add(gated, g:reshape(conv, { n, HC, E })))
end

-- The QSA mask of one sequence in layer L: the base mask with -1e30
-- on the visible complete blocks that the indexer does not pick. cur
-- holds the layer inputs of the sequence, ikw the indexer keys after
-- this pass wrote them. With at most top_k / ratio blocks in the span
-- it is the base mask.
local function qsa_mask(g, hp, L, cur, pos, ikw, lo, span, mask, qpos, qw)
	local I = hp.idx
	local R, P, K = I.ratio, span // I.ratio, I.top_k // I.ratio
	if not qpos or P <= K then
		return mask
	end
	local n = cur:shape()[1]
	local rope = { mode = hp.rope_mode, base = L.base, n_dims = L.n_rot, n_ctx_orig = hp.rope_ctx }
	-- Each block key is the mean of its raw keys, normed, at the
	-- position of its first cell.
	local f2 = lgml.row_size("f16", I.dim)
	local win = g:cast(g:view(ikw, { P, R, I.dim }, { R * f2, f2 }, lo * f2), "f32")
	local st = win:strides()
	local pk = g:cont(g:view(win, { P, I.dim }, { st[1] }))
	for r = 1, R - 1 do
		pk = g:add(pk, g:view(win, { P, I.dim }, { st[1] }, r * st[2]))
	end
	pk = g:mul(g:rms_norm(g:scale(pk, 1 / R), hp.eps), L.idx_k_norm)
	pk = g:cast(g:rope(g:reshape(pk, { P, 1, I.dim }), qpos, rope), "f16")
	local q = g:reshape(g:mul_mat(L.idx_q, cur), { n, I.heads, I.dim })
	q = g:rope(g:mul(g:rms_norm(q, hp.eps), L.idx_q_norm), pos, rope)
	-- A block is visible to a token when its last cell is.
	local mt = mask:type()
	local el = lgml.row_size(mt, 1)
	local vis = g:reshape(g:cont(g:view(mask, { n, P, 1 }, { mask:strides()[1], R * el }, (R - 1) * el)), { n, P })
	local score = g:lightning_indexer(q, pk, qw, mt == "f16" and vis or g:cast(vis, "f16"))
	-- Drop the visible blocks that top_k does not pick, as llama.cpp.
	-- A pick past the visible blocks is masked anyway.
	local top = g:top_k(score, K)
	local zero = g:scale(g:step(g:get_rows(g:reshape(score, { n, P, 1 }), top)), 0)
	local drop = g:exp(mt == "f32" and vis or g:cast(vis, "f32"))
	drop = g:set_rows(g:reshape(drop, { n, P, 1 }), zero, top)
	-- The add keeps the mask type: f16 turns -1e30 into -inf.
	local cut = g:scale(drop, -1e30)
	return g:reshape(g:add(g:reshape(mask, { n, P, R }), cut), { n, span })
end

-- A QSA layer on its input cur: indexer, attention, output gate and
-- projection. It writes the indexer keys of the pass.
local function qsa_layer(self, g, inp, L, il, cur, shared)
	local hp = self.hp
	local tr = T()
	local q, k, v, gate = tr.qkv(g, hp, L, cur, inp.pos)
	local a = inp
	if L.idx then
		local kraw = g:mul_mat(L.idx_k, cur)
		if inp.batch then
			local list = {}
			for i, b in ipairs(inp.batch) do
				local c = b.cache
				local ikw = g:set_rows(c.pool.x[il], g:cont(rows_of(g, kraw, b.off, b.n)), b.rows)
				local ci = g:cont(rows_of(g, cur, b.off, b.n))
				local pi = g:cont(rows_of(g, inp.pos, b.off, b.n))
				local wi = g:cont(rows_of(g, inp.qsa_w, b.off, b.n))
				local m = qsa_mask(g, hp, L, ci, pi, ikw, b.lo, b.n_attn, b.mask, b.qsa_pos, wi)
				list[i] = setmetatable({ mask = m }, { __index = b })
			end
			a = setmetatable({ batch = list }, { __index = inp })
		else
			local c = inp.cache
			local ikw = g:set_rows((c.pool or c).x[il], kraw, inp.rows)
			local m = qsa_mask(g, hp, L, cur, inp.pos, ikw, inp.lo, inp.n_attn, inp.mask, inp.qsa_pos, inp.qsa_w)
			a = setmetatable({ mask = m }, { __index = inp })
		end
	end
	local o = tr.attention(hp, g, a, il, L, q, k, v, shared)
	local n = o:shape()[1]
	o = g:mul(o, g:sigmoid(g:reshape(g:cont(gate), { n, hp.n_head * L.D })))
	return g:mul_mat(L.wo, o)
end

-- Routed experts plus the gated shared expert, on the mixed input x.
local function moe(g, hp, L, x, taps, sfx)
	local out = T().moe(g, hp, L, x, x, taps, sfx)
	if L.shexp_gate then
		local sh = g:swiglu_split(g:mul_mat(L.gate_shexp, x), g:mul_mat(L.up_shexp, x))
		sh = g:mul_mat(L.down_shexp, sh)
		out = g:add(out, g:mul(sh, g:sigmoid(g:mul_mat(L.shexp_gate, x))))
	end
	return out
end

--- Build the forward graph, as `lgml.transformer` forward does.
-- With inp.hidden the result is the streams (n, hc, n_embd) after that
-- many layers; the normed hidden state is the final stream mix.
-- @tparam Model self the model
-- @param g the graph
-- @tparam table inp the inputs
-- @return the logits or hidden state, and the named nodes
function M.forward(self, g, inp)
	local hp, n = self.hp, inp.n
	local E, HC = hp.n_embd, hp.hc
	local tr = T()
	local taps, shared = {}, {}
	local x = inp.embd or g:get_rows(self.tok_embd, inp.ids)
	taps.embd = x
	-- The streams start as copies of the embedding.
	local one = g:reshape(x, { n, 1, E })
	local res = one
	for _ = 2, HC do
		res = g:concat(res, one, 2)
	end
	for il, L in ipairs(self.layers) do
		local sfx = "-" .. (il - 1)
		L.il = il
		if L.ple then
			res = ple_block(self, g, inp, L, res)
		end
		local cur, inj = hc_mix(g, hp, res, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject)
		local o
		if L.recr then
			local la = inp.batch and tr.linear_attention_batch or tr.linear_attention
			o = la(hp, g, inp, L, il, cur, taps, sfx)
		else
			o = qsa_layer(self, g, inp, L, il, cur, shared)
		end
		res = hc_add(g, hp, res, o, inj)
		cur, inj = hc_mix(g, hp, res, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject)
		o = moe(g, hp, L, cur, taps, sfx)
		taps["ffn_out" .. sfx] = o
		res = hc_add(g, hp, res, o, inj)
		taps["l_out" .. sfx] = res
		if il == inp.hidden then
			return res, taps
		end
	end
	local function pick(t, rows)
		local m = rows:shape()[1]
		return g:reshape(g:get_rows(g:reshape(res, { n, HC * E }), rows), { m, HC, E })
	end
	if inp.logits == "rows" then
		res = pick(res, inp.logit_rows)
		n = res:shape()[1]
	elseif (inp.logits == "last" or inp.pool == "last") and n > 1 then
		local st = res:strides()
		res = g:cont(g:view(res, { 1, HC, E }, { st[1], st[2] }, (n - 1) * st[1]))
		n = 1
	end
	if inp.pool == "rows" then
		res = pick(res, inp.pool_rows)
	end
	local h = hc_mix(g, hp, res, self.hc_head_norm, self.hc_head_down, self.hc_head_up)
	taps.result_norm = h
	if inp.pool then
		return h, taps
	end
	return g:mul_mat(self.output, h), taps
end

return M
