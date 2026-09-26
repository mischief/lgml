--- VibeVoice realtime 0.5B text to speech.
-- A 4-layer base LM reads the text. A 20-layer TTS LM interleaves text
-- windows and speech frames. A diffusion head makes one acoustic latent
-- per frame (7.5 Hz). The acoustic decoder turns the latents into
-- 24 kHz audio.
--
-- A loaded model has the tokenizer in tk and the hyperparameters in hp.
-- The usual sequence is voice, tk:encode, generate, decode, finish.
-- @classmod lgml.vibevoice.model
local lgml = require("lgml")
local lm = require("lgml.vibevoice.lm")
local diffusion = require("lgml.vibevoice.diffusion")
local decoder = require("lgml.vibevoice.decoder")
local tokenizer = require("lgml.vibevoice.tokenizer")
local rng = require("lgml.vibevoice.rng")

local M = {}
M.__index = M

--- Output sample rate in Hz.
M.RATE = 24000
--- Token id of the negative path prompt.
M.IMAGE_PAD = 151655
--- Token id of a newline.
M.NEWLINE = 198
--- Text tokens per text window.
M.TEXT_WINDOW = 5
--- Speech frames per speech window.
M.SPEECH_WINDOW = 6
--- Maximum frames per generate call.
M.MAX_FRAMES = 300
--- Cache length bucket of the speech step graph.
M.BUCKET = 64
--- Samples per peak block in decode.
M.PEAK_BLOCK = 200

local function hparams(meta)
	local function k(s)
		return assert(meta["vibevoice." .. s], "missing vibevoice." .. s)
	end
	return {
		d = k("d_lm"), n_head = k("n_heads"), n_kv = k("n_kv_heads"),
		head_dim = k("head_dim"), rope_theta = k("rope_theta"),
		base_layers = k("n_lm_layers"), tts_layers = k("tts_n_layers"),
		vae_dim = k("vae_dim_acoustic"), depths = k("encoder_depths"),
		ratios = k("encoder_ratios"), hop = k("total_downsample"),
		steps = meta["vibevoice.ddpm_inference_steps"] or 20,
		train_steps = meta["vibevoice.ddpm_num_steps"] or 1000,
		eps = 1e-6, head_eps = 1e-5,
	}
end

local function zeros(be, shape)
	local n = 1
	for _, v in ipairs(shape) do
		n = n * v
	end
	return be:tensor("f32", shape, string.rep("\0", 4 * n))
end

local function count_pred_layers(w)
	local n = 0
	while w:get("pred.layers." .. n .. ".norm.weight") do
		n = n + 1
	end
	return n
end

-- Row j of a (rows, d) weight as an f32 leaf.
local function row_leaf(be, g, t, j, d)
	local out = be:tensor("f32", { 1, d })
	local row = lgml.row_size(t:type(), d)
	return out, g:cpy(g:cast(g:view(t, { 1, d }, { row }, j * row), "f32"), out)
end

--- Load a VibeVoice realtime TTS gguf.
-- @tparam lgml.backend be backend
-- @string path gguf path
-- @treturn lgml.vibevoice.model the model
-- @raise when the file is not a VibeVoice realtime TTS gguf
-- @static
function M.load(be, path)
	local info = assert(lgml.gguf(path))
	local meta = info.meta
	if meta["general.architecture"] ~= "vibevoice-tts" or (meta["vibevoice.tts_n_layers"] or 0) == 0 then
		error(path .. ": not a VibeVoice realtime TTS gguf")
	end
	local w = assert(be:load(path))
	local hp = hparams(meta)
	local function get(name)
		return assert(w:get(name), name)
	end
	local m = setmetatable({ be = be, w = w, hp = hp, path = path, voices = {} }, M)
	m.tk = tokenizer.new(meta["tokenizer.ggml.tokens"])
	m.base = lm.layers(w, "lm", hp.base_layers)
	m.tts = lm.layers(w, "tts_lm", hp.tts_layers)
	m.tok_emb = get("lm.tok_emb.weight")
	m.tts_norm = get("tts_lm.norm.weight")
	m.conn = {
		fc1 = get("at_conn.fc1.weight"), fc1_b = get("at_conn.fc1.bias"),
		norm = get("at_conn.norm.weight"),
		fc2 = get("at_conn.fc2.weight"), fc2_b = get("at_conn.fc2.bias"),
	}
	m.eos = {
		fc1 = get("tts_eos.fc1.weight"), fc1_b = get("tts_eos.fc1.bias"),
		fc2 = get("tts_eos.fc2.weight"), fc2_b = get("tts_eos.fc2.bias"),
	}
	m.pred = diffusion.load(w, count_pred_layers(w))
	m.dec = decoder.load(w, hp.depths, hp.ratios)
	decoder.prepare(be, m.dec)
	m.sf = get("speech_scaling_factor"):get()[1]
	m.bias = get("speech_bias_factor")
	m:alloc()
	m:constants()
	return m
end

--- Allocate the caches and per-frame leaves. load calls it.
function M:alloc()
	local be, hp = self.be, self.hp
	local kvw = hp.n_kv * hp.head_dim
	self.base_cache = lm.cache(be, hp.base_layers, 512, kvw)
	self.pos_cache = lm.cache(be, hp.tts_layers, 1280, kvw)
	self.neg_cache = lm.cache(be, hp.tts_layers, M.MAX_FRAMES + M.BUCKET, kvw)
	self.io = {
		cond = zeros(be, { 2, hp.d }),
		noise = zeros(be, { 1, hp.vae_dim }),
		fid = be:tensor("i32", { 1 }, { 0 }),
		latents = zeros(be, { M.MAX_FRAMES, hp.vae_dim }),
		ones = be:tensor("f32", { hp.d }, string.rep(string.pack("<f", 1), hp.d)),
	}
	self.graphs = {}
end

--- The cond row for the positive (1) or negative (2) path.
-- @tparam lgml.graph g graph
-- @int i path, 1 or 2
-- @treturn lgml.tensor view (1, d) of the cond leaf
function M:cond_row(g, i)
	local row = lgml.row_size("f32", self.hp.d)
	return g:view(self.io.cond, { 1, self.hp.d }, { row }, (i - 1) * row)
end

--- Compute the type rows and the negative path prefill. load calls it.
-- The prefill sends one IMAGE_PAD token through the base LM, then adds
-- the text type and sends it through the TTS LM.
function M:constants()
	local be, hp, d = self.be, self.hp, self.hp.d
	local g <close> = be:graph()
	local c1, c2
	self.speech_type, c1 = row_leaf(be, g, self.w:get("tts_types.weight"), 0, d)
	self.text_type, c2 = row_leaf(be, g, self.w:get("tts_types.weight"), 1, d)
	g:compute(c1, c2)

	local g2 <close> = be:graph()
	local ids <close> = be:tensor("i32", { 1 }, { M.IMAGE_PAD })
	local inp = lm.inputs(be, 1, 1)
	lm.fill(inp, 0)
	local x = lm.stack(g2, hp, self.base, g2:get_rows(self.tok_emb, ids), inp, self.base_cache)
	x = lm.stack(g2, hp, self.tts, g2:add(x, self.text_type), inp, self.neg_cache)
	self.neg0 = be:tensor("f32", { 1, d })
	g2:compute(g2:cpy(g2:mul(g2:rms_norm(x, hp.eps), self.tts_norm), self.neg0))
	lm.close_inputs(inp)
end

--- Leaves that depend on the step count, cached per step count.
-- @int steps diffusion steps
-- @treturn table sched (schedule), temb (timestep embeddings) and zeros
function M:step_consts(steps)
	local key = "steps" .. steps
	if self.graphs[key] then
		return self.graphs[key]
	end
	local be, hp, p = self.be, self.hp, self.pred
	local sched = diffusion.schedule(steps, hp.train_steps)
	local tab = diffusion.timestep_table(sched, p.t0:shape()[2])
	local g <close> = be:graph()
	local tsin <close> = be:tensor("f32", { steps, #tab // steps }, tab)
	local temb = be:tensor("f32", { steps, hp.d })
	local e = g:mul_mat(p.t2, g:silu(g:mul_mat(p.t0, tsin)))
	g:compute(g:cpy(e, temb))
	local c = { sched = sched, temb = temb, zeros = zeros(be, { steps, 2, hp.d }) }
	self.graphs[key] = c
	return c
end

--- The diffusion graph for (steps, cfg).
-- Inputs are model leaves, so one graph serves every frame. The model
-- keeps one diffusion graph and closes the old one on a change.
-- @int steps diffusion steps
-- @number cfg classifier-free guidance scale
-- @treturn lgml.graph the graph
-- @treturn lgml.tensor node that writes the latent
-- @treturn {tensor,...} the guided v of each step
function M:diffusion_graph(steps, cfg)
	local key = ("diff%d:%g"):format(steps, cfg)
	if self.graphs[key] then
		return table.unpack(self.graphs[key])
	end
	for k, v in pairs(self.graphs) do
		if k:match("^diff") then
			v[1]:close()
			self.graphs[k] = nil
		end
	end
	local c = self:step_consts(steps)
	local io = self.io
	local g = self.be:graph()
	local out, vs = diffusion.graph(g, self.pred, {
		noise = io.noise, cond = io.cond, temb = c.temb, zeros = c.zeros,
		ones = io.ones, latents = io.latents, fid = io.fid,
	}, c.sched, cfg, self.hp)
	self.graphs[key] = { g, out, vs }
	return g, out, vs
end

--- Load a voice preset, or return it from the cache.
-- A preset holds KV caches for the base and TTS LMs.
-- @string path voice gguf path
-- @treturn table voice: path, w, lm_len and tts_len
-- @raise when the file is not a voice gguf or the prompt is too long
function M:voice(path)
	local v = self.voices[path]
	if v then
		return v
	end
	local meta = assert(lgml.gguf(path)).meta
	if meta["general.architecture"] ~= "vibevoice-voice" then
		error(path .. ": not a VibeVoice voice gguf")
	end
	v = {
		path = path, w = assert(self.be:load(path)),
		lm_len = meta["voice.lm.seq_len"], tts_len = meta["voice.tts_lm.seq_len"],
	}
	if v.lm_len + 128 > self.base_cache.n_ctx or v.tts_len + 128 > self.pos_cache.n_ctx then
		v.w:close()
		error(path .. ": voice prompt too long")
	end
	self.voices[path] = v
	return v
end

-- Copy voice KV (kv_heads, seq, head_dim) f16 into cache rows 0..seq-1.
local function copy_kv(g, src, dst, seq, width)
	local t = g:reshape(g:cont(g:permute(src, { 2, 1, 3 })), { seq, width })
	local row = lgml.row_size("f32", width)
	return g:cpy(t, g:view(dst, { seq, width }, { row }, 0))
end

--- Copy the voice KV into the base and positive caches.
-- The call does nothing when v is the current voice.
-- @tparam table v voice from voice
function M:load_voice(v)
	if self.cur_voice == v.path then
		return
	end
	local g <close> = self.be:graph()
	local outs, kvw = {}, self.hp.n_kv * self.hp.head_dim
	local function copy(prefix, cache, n, seq)
		for i = 0, n - 1 do
			for _, kv in ipairs({ "k", "v" }) do
				local src = assert(v.w:get(("voice.%s.%d.%s"):format(prefix, i, kv)))
				outs[#outs + 1] = copy_kv(g, src, cache[kv][i + 1], seq, kvw)
			end
		end
	end
	copy("lm", self.base_cache, self.hp.base_layers, v.lm_len)
	copy("tts_lm", self.pos_cache, self.hp.tts_layers, v.tts_len)
	g:compute(table.unpack(outs))
	self.cur_voice = v.path
end

--- Base LM over the text after the voice prompt.
-- Also resets the negative condition.
-- @tparam {int,...} ids text token ids
-- @tparam table v current voice
-- @treturn lgml.tensor new f32 leaf (n, d)
function M:base_hidden(ids, v)
	local be, hp, n = self.be, self.hp, #ids
	local g <close> = be:graph()
	local t <close> = be:tensor("i32", { n }, ids)
	local inp = lm.inputs(be, n, v.lm_len + n)
	lm.fill(inp, v.lm_len)
	local x = lm.stack(g, hp, self.base, g:get_rows(self.tok_emb, t), inp, self.base_cache)
	local out = be:tensor("f32", { n, hp.d })
	g:compute(g:cpy(x, out), g:cpy(self.neg0, self:cond_row(g, 2)))
	lm.close_inputs(inp)
	return out
end

--- Run a text window through the TTS LM.
-- The window is rows [at, at + n) of the base hidden state, plus the
-- text type. The last hidden state is the new positive condition.
-- @tparam lgml.tensor base base hidden state from base_hidden
-- @int at first row (0-based)
-- @int n row count
-- @int n_past positive cache position
function M:text_window(base, at, n, n_past)
	local be, hp, d = self.be, self.hp, self.hp.d
	local g <close> = be:graph()
	local row = lgml.row_size("f32", d)
	local x = g:add(g:view(base, { n, d }, { row }, at * row), self.text_type)
	local inp = lm.inputs(be, n, n_past + n)
	lm.fill(inp, n_past)
	x = lm.last(g, lm.stack(g, hp, self.tts, x, inp, self.pos_cache), n, d)
	g:compute(g:cpy(g:mul(g:rms_norm(x, hp.eps), self.tts_norm), self:cond_row(g, 1)))
	lm.close_inputs(inp)
end

local function bucket(n, cap)
	return math.min(cap, ((n + M.BUCKET - 1) // M.BUCKET) * M.BUCKET)
end

--- Speech step graph.
-- The latent at fid goes through the connector, then one step of the
-- positive and negative TTS LM. The graph also computes the EOS logit.
-- The model keeps one graph per pair of cache length buckets.
-- @int n_pos positive cache position
-- @int n_neg negative cache position
-- @treturn table g (graph), ip and in_ (inputs), eos (logit) and outs
function M:speech_graph(n_pos, n_neg)
	local lp = bucket(n_pos + 1, self.pos_cache.n_ctx)
	local ln = bucket(n_neg + 1, self.neg_cache.n_ctx)
	local sg = self.graphs.speech
	if sg and sg.lp == lp and sg.ln == ln then
		return sg
	end
	if sg then
		sg.g:close()
		lm.close_inputs(sg.ip)
		lm.close_inputs(sg.in_)
	end
	local be, hp, c, e = self.be, self.hp, self.conn, self.eos
	local g = be:graph()
	sg = { g = g, lp = lp, ln = ln, ip = lm.inputs(be, 1, lp), in_ = lm.inputs(be, 1, ln) }
	local z = g:get_rows(self.io.latents, self.io.fid)
	local x = g:add(g:mul_mat(c.fc1, z), c.fc1_b)
	x = g:mul(g:rms_norm(x, hp.eps), c.norm)
	x = g:add(g:add(g:mul_mat(c.fc2, x), c.fc2_b), self.speech_type)
	local hpos = lm.stack(g, hp, self.tts, x, sg.ip, self.pos_cache)
	local hneg = lm.stack(g, hp, self.tts, x, sg.in_, self.neg_cache)
	hpos = g:mul(g:rms_norm(hpos, hp.eps), self.tts_norm)
	hneg = g:mul(g:rms_norm(hneg, hp.eps), self.tts_norm)
	local l = g:relu(g:add(g:mul_mat(e.fc1, hpos), e.fc1_b))
	sg.eos = g:add(g:mul_mat(e.fc2, l), e.fc2_b)
	sg.outs = { g:cpy(hpos, self:cond_row(g, 1)), g:cpy(hneg, self:cond_row(g, 2)), sg.eos }
	self.graphs.speech = sg
	return sg
end

--- Run one speech step.
-- @int n_pos positive cache position
-- @int n_neg negative cache position
-- @treturn bool true when the EOS classifier fires
function M:speech_step(n_pos, n_neg)
	local sg = self:speech_graph(n_pos, n_neg)
	lm.fill(sg.ip, n_pos)
	lm.fill(sg.in_, n_neg)
	sg.g:compute(table.unpack(sg.outs))
	return sg.eos:get()[1] > 0
end

--- Generate latents for token ids.
-- @tparam {int,...} ids text token ids
-- @tparam table v voice from voice
-- @tparam table opts options
-- opts.seed: noise seed. The default is 42.
-- opts.steps: diffusion steps. The default is hp.steps.
-- opts.cfg: guidance scale. The default is 1.5.
-- opts.tick: function that runs between frames; return false to stop.
-- @treturn int frame count, or nil when tick stopped generation
-- @raise when the text is too long for one pass
function M:generate(ids, v, opts)
	local n_text = #ids
	local n_frames = math.min(math.max(12, n_text * 4), M.MAX_FRAMES)
	if v.tts_len + n_text + n_frames + 1 > self.pos_cache.n_ctx or
	    v.lm_len + n_text > self.base_cache.n_ctx then
		error("text too long for one pass")
	end
	self:load_voice(v)
	local base <close> = self:base_hidden(ids, v)
	local dg, dout = self:diffusion_graph(opts.steps or self.hp.steps, opts.cfg or 1.5)
	local r = rng.new(opts.seed or 42)
	local n_past, n_neg, cursor, fi = v.tts_len, 1, 0, 0
	local finished = false
	while not finished and fi < n_frames do
		if cursor < n_text then
			local win = math.min(M.TEXT_WINDOW, n_text - cursor)
			self:text_window(base, cursor, win, n_past)
			cursor, n_past = cursor + win, n_past + win
		end
		for _ = 1, math.min(M.SPEECH_WINDOW, n_frames - fi) do
			local z = r:normal(self.hp.vae_dim)
			self.io.noise:set(string.pack("<" .. string.rep("f", #z), table.unpack(z)))
			self.io.fid:set({ fi })
			dg:compute(dout)
			fi = fi + 1
			if fi < n_frames then
				finished = self:speech_step(n_past, n_neg)
				n_past, n_neg = n_past + 1, n_neg + 1
			end
			if opts.tick and opts.tick() == false then
				return nil
			end
			if finished then
				break
			end
		end
	end
	return fi
end

--- Decode the first n latents.
-- @int n frame count from generate
-- @treturn lgml.tensor new f32 leaf (n * hop) with the audio
-- @treturn {number,...} peak of each PEAK_BLOCK samples
function M:decode(n)
	local be, hp = self.be, self.hp
	local g <close> = be:graph()
	local row = lgml.row_size("f32", hp.vae_dim)
	local z = g:view(self.io.latents, { n, hp.vae_dim }, { row }, 0)
	z = g:sub(g:scale(z, 1 / self.sf), self.bias)
	local a = g:reshape(decoder.forward(g, self.dec, z), { n * hp.hop })
	local out = be:tensor("f32", { n * hp.hop })
	local peaks = g:vibevoice_peak(a, M.PEAK_BLOCK)
	g:compute(g:cpy(a, out), peaks)
	return out, peaks:get()
end

-- Trim bounds from block peaks: keep 800 samples before the first loud
-- block and 1200 after the last one.
local function trim_bounds(peaks, n)
	local b = M.PEAK_BLOCK
	local first, last
	for i, p in ipairs(peaks) do
		if first == nil and p > 0.005 then
			first = i
		end
		if p > 0.01 then
			last = i
		end
	end
	if first == nil or last == nil or last < first then
		return 0, n
	end
	return math.max(0, (first - 1) * b - 800), math.min(n, last * b + 1200)
end

local function ramp(be, n, f)
	local t = {}
	for i = 0, n - 1 do
		t[i + 1] = f(i, n)
	end
	return be:tensor("f32", { n }, t)
end

--- Trim the audio and add a 20 ms fade-in and a 50 ms fade-out.
-- The trim keeps 800 samples before the first loud block and 1200
-- after the last one.
-- @tparam lgml.tensor a audio leaf from decode
-- @tparam {number,...} peaks block peaks from decode
-- @tparam[opt] lgml.audio dst lgml audio to append to
-- @treturn lgml.audio dst, or a new lgml audio when dst is nil
function M:finish(a, peaks, dst)
	local be = self.be
	local s, e = trim_bounds(peaks, a:shape()[1])
	local fi, fo = 480, 1200
	self.fade_in = self.fade_in or ramp(be, fi, function(i, n) return (i / n) ^ 2 end)
	self.fade_out = self.fade_out or ramp(be, fo, function(i, n) return (n - i) / n end)
	local g <close> = be:graph()
	local function part(at, n)
		return g:view(a, { n }, nil, at * 4)
	end
	local parts
	if e - s < 2 * (fi + fo) then
		parts = { part(s, e - s) }
	else
		parts = { g:mul(part(s, fi), self.fade_in), part(s + fi, e - s - fi - fo),
		    g:mul(part(e - fo, fo), self.fade_out) }
	end
	g:compute(table.unpack(parts))
	for _, p in ipairs(parts) do
		if dst then
			dst:append(p)
		else
			dst = lgml.audio_from(p, M.RATE)
		end
	end
	return dst
end

return M
