/*** Tensor graphs on ggml backends.
 * Shapes and dimension numbers use torch order, outermost first, from 1.
 * @module lgml
 */

/*
 * Lua binding for audio.c. An audio object holds mono float samples in
 * C memory. Samples never become Lua values; a:mel writes the
 * spectrogram straight into a tensor.
 */

#include <stdlib.h>
#include <string.h>

#include <ggml.h>
#include <ggml-backend.h>

#include <lua.h>
#include <lauxlib.h>

#include "audio.h"
#include "lgml.h"

#define AUDIO_MT	"lgml_audio"

static struct pcm *
check_audio(lua_State *L, int idx)
{
	struct pcm *p = luaL_checkudata(L, idx, AUDIO_MT);

	if (p->x == NULL)
		luaL_error(L, "audio closed");
	return p;
}

/* Push an empty audio object. Fill it before any call that can fail. */
static struct pcm *
new_audio(lua_State *L)
{
	struct pcm *p = lua_newuserdatauv(L, sizeof(*p), 0);

	p->x = NULL;
	p->n = 0;
	p->rate = 0;
	luaL_setmetatable(L, AUDIO_MT);
	return p;
}

/*** Decode a WAV file.
 * Channels mix down to mono.
 * @tparam string bytes contents of the WAV file
 * @treturn[1] audio the audio
 * @return[2] nil
 * @treturn[2] string error message
 * @function audio
 */
static int
l_audio(lua_State *L)
{
	size_t len;
	const char *s = luaL_checklstring(L, 1, &len);
	struct pcm tmp, *p;
	const char *err;

	p = new_audio(L);
	err = wav_decode(s, len, &tmp);
	if (err != NULL) {
		luaL_pushfail(L);
		lua_pushstring(L, err);
		return 2;
	}
	*p = tmp;
	return 1;
}

/* Read the samples of an f32 tensor with storage. Returns a malloc copy. */
static float *
tensor_samples(lua_State *L, int idx, size_t *n)
{
	struct tensor *t = lgml_check_tensor(L, idx);
	const struct ggml_tensor *gt = t->t;
	float *x;

	if (gt->buffer == NULL || gt->type != GGML_TYPE_F32 || !ggml_is_contiguous(gt))
		luaL_error(L, "audio needs a computed, contiguous f32 tensor");
	*n = (size_t)ggml_nelements(gt);
	x = malloc(*n * sizeof(*x) + 1);
	if (x == NULL)
		luaL_error(L, "out of memory");
	ggml_backend_tensor_get(gt, x, 0, *n * sizeof(*x));
	return x;
}

/*** Make audio from the samples of a tensor.
 * @tparam tensor t computed, contiguous f32 tensor; every element is a sample
 * @tparam int rate sample rate in Hz, 1000 to 384000
 * @treturn audio the audio
 * @raise on a bad tensor or rate
 * @function audio_from
 */
static int
l_audio_from(lua_State *L)
{
	lua_Integer rate = luaL_checkinteger(L, 2);
	struct pcm *p;
	size_t n;
	float *x;

	luaL_argcheck(L, rate >= 1000 && rate <= 384000, 2, "rate out of range");
	p = new_audio(L);
	x = tensor_samples(L, 1, &n);
	p->x = x;
	p->n = n;
	p->rate = (int)rate;
	return 1;
}

/*** Mono float samples in C memory.
 * Samples never become Lua values. `audio:mel` writes the spectrogram
 * straight into a tensor.
 * @type audio
 */

/*** Add the samples of a tensor to the end.
 * @tparam tensor t computed, contiguous f32 tensor
 * @raise on a bad tensor, or if the audio gets too long
 * @function audio:append
 */
static int
audio_append(lua_State *L)
{
	struct pcm *p = check_audio(L, 1);
	size_t n;
	float *x = tensor_samples(L, 2, &n), *y;

	if (n > ((size_t)1 << 31) - p->n) {
		free(x);
		return luaL_error(L, "audio too long");
	}
	y = realloc(p->x, (p->n + n) * sizeof(*y) + 1);
	if (y == NULL) {
		free(x);
		return luaL_error(L, "out of memory");
	}
	memcpy(y + p->n, x, n * sizeof(*y));
	free(x);
	p->x = y;
	p->n += n;
	return 0;
}

/*** Free the samples.
 * The `__gc` and `__close` metamethods call this function.
 * @function audio:close
 */
static int
audio_close(lua_State *L)
{
	struct pcm *p = luaL_checkudata(L, 1, AUDIO_MT);

	pcm_free(p);
	return 0;
}

/*** Get the sample rate.
 * @treturn int samples per second
 * @function audio:rate
 */
static int
audio_rate(lua_State *L)
{
	lua_pushinteger(L, check_audio(L, 1)->rate);
	return 1;
}

/*** Get the length.
 * @treturn int number of samples
 * @function audio:len
 */
static int
audio_len(lua_State *L)
{
	lua_pushinteger(L, (lua_Integer)check_audio(L, 1)->n);
	return 1;
}

/*** Get the duration.
 * @treturn number seconds
 * @function audio:duration
 */
static int
audio_duration(lua_State *L)
{
	struct pcm *p = check_audio(L, 1);

	lua_pushnumber(L, (lua_Number)p->n / p->rate);
	return 1;
}

/*** Resample to a new rate.
 * The resampler uses a Kaiser-windowed sinc low-pass filter.
 * @tparam int rate new rate in Hz, 1000 to 768000
 * @treturn audio new audio
 * @raise if the rate is out of range or resampling fails
 * @function audio:resample
 */
static int
audio_resample(lua_State *L)
{
	struct pcm *p = check_audio(L, 1);
	lua_Integer rate = luaL_checkinteger(L, 2);
	struct pcm tmp, *q;

	luaL_argcheck(L, rate >= AUDIO_MIN_RATE && rate <= AUDIO_MAX_RATE, 2,
	    "rate out of range");
	q = new_audio(L);
	if (pcm_resample(p, (int)rate, &tmp) != 0)
		return luaL_error(L, "resample failed");
	*q = tmp;
	return 1;
}

/*** Join this audio and more audio into new audio.
 * All sample rates must match.
 * @tparam audio ... audio to add after this audio
 * @treturn audio new audio
 * @raise if the rates differ or the result is too long
 * @function audio:concat
 */
static int
audio_concat(lua_State *L)
{
	int i, n = lua_gettop(L);
	struct pcm *p, *q;
	size_t total = 0, off = 0;

	for (i = 1; i <= n; i++) {
		p = check_audio(L, i);
		if (p->rate != check_audio(L, 1)->rate)
			return luaL_error(L, "concat: sample rates differ");
		if (p->n > ((size_t)1 << 31) - total)
			return luaL_error(L, "concat: too long");
		total += p->n;
	}
	q = new_audio(L);
	if ((q->x = malloc(total * sizeof(float))) == NULL)
		return luaL_error(L, "out of memory");
	q->n = total;
	q->rate = check_audio(L, 1)->rate;
	for (i = 1; i <= n; i++) {
		p = lua_touserdata(L, i);
		memcpy(q->x + off, p->x, p->n * sizeof(float));
		off += p->n;
	}
	return 1;
}

/*** Encode as a WAV file.
 * Each channel is a copy of the mono signal.
 * @tparam[opt=1] int channels 1 to 32
 * @tparam[opt="s16"] string format "s16", "s24", "s32" or "f32"
 * @treturn string WAV bytes
 * @raise if memory runs out or the audio is too long
 * @function audio:wav
 */
static int
audio_wav(lua_State *L)
{
	static const char *const names[] = { "s16", "s24", "s32", "f32", NULL };
	struct pcm *p = check_audio(L, 1);
	lua_Integer ch = luaL_optinteger(L, 2, 1);
	int fmt = luaL_checkoption(L, 3, "s16", names);
	size_t len;
	void *buf;

	luaL_argcheck(L, ch >= 1 && ch <= 32, 2, "channels out of range");
	buf = wav_encode(p, (int)ch, (enum wav_format)fmt, &len);
	if (buf == NULL)
		return luaL_error(L, "wav: out of memory or too long");
	lua_pushlstring(L, buf, len);
	free(buf);
	return 1;
}

/*** Write the Whisper log-mel spectrogram into a tensor.
 * The window starts at sample start. The audio must be 16 kHz.
 * @tparam tensor t f32 leaf with storage, of shape (`n_mels`, `n_frames`);
 * `n_mels` up to 512
 * @tparam[opt=0] int start first sample, 0 to the length
 * @raise on a bad rate, tensor or start
 * @function audio:mel
 */
static int
audio_mel(lua_State *L)
{
	struct pcm *p = check_audio(L, 1);
	struct tensor *t = lgml_check_tensor(L, 2);
	lua_Integer start = luaL_optinteger(L, 3, 0);
	const struct ggml_tensor *gt = t->t;
	float *filt, *out;
	int r;

	if (p->rate != MEL_RATE)
		return luaL_error(L, "mel needs %d Hz audio", MEL_RATE);
	luaL_argcheck(L, start >= 0 && (size_t)start <= p->n, 3, "start out of range");
	if (t->s == NULL || gt->buffer == NULL || gt->type != GGML_TYPE_F32 ||
	    !ggml_is_contiguous(gt) || gt->ne[2] != 1 || gt->ne[3] != 1)
		return luaL_error(L, "mel needs an f32 leaf of shape (n_mels, n_frames)");
	if (gt->ne[1] > MEL_MAX_BINS || gt->ne[0] > 1 << 20)
		return luaL_error(L, "mel: too many bins or frames");
	filt = mel_filters((int)gt->ne[1], MEL_N_FFT, MEL_RATE);
	out = malloc((size_t)ggml_nelements(gt) * sizeof(float));
	if (filt == NULL || out == NULL) {
		free(filt);
		free(out);
		return luaL_error(L, "out of memory");
	}
	r = log_mel(p, (size_t)start, filt, (int)gt->ne[1], (int)gt->ne[0], out);
	if (r == 0)
		ggml_backend_tensor_set(t->t, out, 0, ggml_nbytes(gt));
	free(filt);
	free(out);
	if (r != 0)
		return luaL_error(L, "mel failed");
	return 0;
}

static const luaL_Reg audio_meta[] = {
	{ "__gc", audio_close },
	{ "__close", audio_close },
	{ NULL, NULL }
};

static const luaL_Reg audio_methods[] = {
	{ "close", audio_close },
	{ "rate", audio_rate },
	{ "len", audio_len },
	{ "duration", audio_duration },
	{ "resample", audio_resample },
	{ "concat", audio_concat },
	{ "append", audio_append },
	{ "wav", audio_wav },
	{ "mel", audio_mel },
	{ NULL, NULL }
};

void
lgml_open_audio(lua_State *L)
{
	luaL_newmetatable(L, AUDIO_MT);
	luaL_setfuncs(L, audio_meta, 0);
	lua_newtable(L);
	luaL_setfuncs(L, audio_methods, 0);
	lua_setfield(L, -2, "__index");
	lua_pop(L, 1);
	lua_pushcfunction(L, l_audio);
	lua_setfield(L, -2, "audio");
	lua_pushcfunction(L, l_audio_from);
	lua_setfield(L, -2, "audio_from");
}
