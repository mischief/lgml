/*
 * Audio in C. WAV decoding treats the input as untrusted. The mel
 * spectrogram follows Whisper: periodic Hann window, centered frames
 * with reflect padding, Slaney mel filters, log10 and dynamic range
 * compression.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "audio.h"

#ifndef M_PI
#define M_PI	3.14159265358979323846
#endif

/* Windowed-sinc resampler: zero crossings per side, table steps. */
#define RS_ZC		48
#define RS_OS		512
#define RS_BETA		8.0
#define RS_ROLLOFF	0.95

#define MAX_SAMPLES	((size_t)1 << 31)

void
pcm_free(struct pcm *p)
{
	free(p->x);
	p->x = NULL;
	p->n = 0;
}

static uint32_t
le16(const unsigned char *p)
{
	return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}

static uint32_t
le32(const unsigned char *p)
{
	return le16(p) | le16(p + 2) << 16;
}

struct wav_fmt {
	uint32_t	 tag;
	uint32_t	 channels;
	uint32_t	 rate;
	uint32_t	 block;
	uint32_t	 bits;
};

static const char *
parse_fmt(const unsigned char *p, uint32_t sz, struct wav_fmt *f)
{
	if (sz < 16)
		return "fmt chunk too short";
	f->tag = le16(p);
	f->channels = le16(p + 2);
	f->rate = le32(p + 4);
	f->block = le16(p + 12);
	f->bits = le16(p + 14);
	if (f->tag == 0xfffe) {
		if (sz < 40)
			return "fmt chunk too short";
		f->tag = le16(p + 24);
	}
	if (f->tag != 1 && f->tag != 3)
		return "WAV encoding not PCM or float";
	if (f->channels < 1 || f->channels > 32)
		return "WAV has 0 or more than 32 channels";
	if (f->rate < AUDIO_MIN_RATE || f->rate > AUDIO_MAX_RATE)
		return "WAV sample rate out of range";
	if (f->tag == 1 && f->bits != 8 && f->bits != 16 && f->bits != 24 && f->bits != 32)
		return "PCM WAV needs 8, 16, 24 or 32 bits";
	if (f->tag == 3 && f->bits != 32 && f->bits != 64)
		return "float WAV needs 32 or 64 bits";
	if (f->block != f->channels * f->bits / 8)
		return "WAV block size does not match format";
	return NULL;
}

static float
sample(const unsigned char *p, const struct wav_fmt *f)
{
	uint32_t u;
	uint64_t w;
	float x;
	double d;

	switch (f->bits) {
	case 8:
		return ((float)p[0] - 128.0f) / 128.0f;
	case 16:
		return (float)(int16_t)le16(p) / 32768.0f;
	case 24:
		u = (uint32_t)p[0] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 24;
		return (float)((int32_t)u >> 8) / 8388608.0f;
	case 32:
		u = le32(p);
		if (f->tag == 1)
			return (float)((double)(int32_t)u / 2147483648.0);
		memcpy(&x, &u, sizeof(x));
		return isfinite(x) ? x : 0.0f;
	default:
		w = (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32;
		memcpy(&d, &w, sizeof(d));
		return isfinite(d) ? (float)d : 0.0f;
	}
}

/* Find the fmt and data chunks. */
static const char *
find_chunks(const unsigned char *p, size_t len, struct wav_fmt *f,
    const unsigned char **data, size_t *dlen)
{
	const char *err;
	size_t off = 12, avail;
	uint32_t sz;
	int have_fmt = 0;

	*data = NULL;
	while (len - off >= 8 && !(have_fmt && *data != NULL)) {
		sz = le32(p + off + 4);
		avail = len - off - 8;
		if (memcmp(p + off, "fmt ", 4) == 0) {
			if (sz > avail)
				return "fmt chunk past end of file";
			if ((err = parse_fmt(p + off + 8, sz, f)) != NULL)
				return err;
			have_fmt = 1;
		} else if (memcmp(p + off, "data", 4) == 0) {
			/* Streamed files can give a size past the end. */
			*data = p + off + 8;
			*dlen = sz < avail ? sz : avail;
		}
		if (sz >= avail)
			break;
		off += 8 + (size_t)sz + (sz & 1);
		if (off > len)
			break;
	}
	if (!have_fmt)
		return "WAV has no fmt chunk";
	if (*data == NULL)
		return "WAV has no data chunk";
	return NULL;
}

/* Decode a WAV file. Channels mix down to mono. Returns NULL or a message. */
const char *
wav_decode(const void *buf, size_t len, struct pcm *out)
{
	const unsigned char *p = buf, *data, *fp;
	struct wav_fmt f = { 0, 0, 0, 0, 0 };
	const char *err;
	size_t dlen = 0, frames, i, bps;
	uint32_t c;
	float acc;

	out->x = NULL;
	out->n = 0;
	if (len < 12 || memcmp(p, "RIFF", 4) != 0 || memcmp(p + 8, "WAVE", 4) != 0)
		return "not a RIFF WAVE file";
	if ((err = find_chunks(p, len, &f, &data, &dlen)) != NULL)
		return err;
	frames = dlen / f.block;
	if (frames == 0)
		return "WAV has no samples";
	if (frames > MAX_SAMPLES)
		return "WAV too long";
	out->x = malloc(frames * sizeof(float));
	if (out->x == NULL)
		return "out of memory";
	bps = f.bits / 8;
	for (i = 0; i < frames; i++) {
		fp = data + i * f.block;
		acc = 0.0f;
		for (c = 0; c < f.channels; c++)
			acc += sample(fp + c * bps, &f);
		out->x[i] = acc / (float)f.channels;
	}
	out->n = frames;
	out->rate = (int)f.rate;
	return NULL;
}

static void
put16(unsigned char *p, uint32_t v)
{
	p[0] = (unsigned char)v;
	p[1] = (unsigned char)(v >> 8);
}

static void
put32(unsigned char *p, uint32_t v)
{
	put16(p, v & 0xffff);
	put16(p + 2, v >> 16);
}

static int32_t
quantize(float x, double scale)
{
	double v = (double)x;

	if (v > 1.0)
		v = 1.0;
	else if (v < -1.0)
		v = -1.0;
	v = floor(v * scale + 0.5);
	return v > scale ? (int32_t)scale : (int32_t)v;
}

static void
put_sample(unsigned char *p, float x, enum wav_format fmt)
{
	uint32_t u;

	switch (fmt) {
	case WAV_S16:
		put16(p, (uint32_t)quantize(x, 32767.0) & 0xffff);
		break;
	case WAV_S24:
		u = (uint32_t)quantize(x, 8388607.0);
		p[0] = (unsigned char)u;
		p[1] = (unsigned char)(u >> 8);
		p[2] = (unsigned char)(u >> 16);
		break;
	case WAV_S32:
		put32(p, (uint32_t)quantize(x, 2147483647.0));
		break;
	case WAV_F32:
		memcpy(&u, &x, sizeof(u));
		put32(p, u);
		break;
	}
}

/* Encode as WAV with each channel a copy of the mono signal. */
void *
wav_encode(const struct pcm *in, int channels, enum wav_format fmt, size_t *lenp)
{
	static const unsigned bits[] = { 16, 24, 32, 32 };
	unsigned char *p, *q;
	size_t i, dlen, bps = bits[fmt] / 8;
	int c;

	if (channels < 1 || channels > 32 || in->n > 0x7fffffff / (bps * (size_t)channels))
		return NULL;
	dlen = in->n * bps * (size_t)channels;
	if ((p = malloc(44 + dlen)) == NULL)
		return NULL;
	memcpy(p, "RIFF", 4);
	put32(p + 4, (uint32_t)(36 + dlen));
	memcpy(p + 8, "WAVEfmt ", 8);
	put32(p + 16, 16);
	put16(p + 20, fmt == WAV_F32 ? 3 : 1);
	put16(p + 22, (uint32_t)channels);
	put32(p + 24, (uint32_t)in->rate);
	put32(p + 28, (uint32_t)((size_t)in->rate * bps * (size_t)channels));
	put16(p + 32, (uint32_t)(bps * (size_t)channels));
	put16(p + 34, bits[fmt]);
	memcpy(p + 36, "data", 4);
	put32(p + 40, (uint32_t)dlen);
	q = p + 44;
	for (i = 0; i < in->n; i++)
		for (c = 0; c < channels; c++, q += bps)
			put_sample(q, in->x[i], fmt);
	*lenp = 44 + dlen;
	return p;
}

/* Modified Bessel function of the first kind, order 0. */
static double
bessel_i0(double x)
{
	double sum = 1.0, term = 1.0, h = x / 2.0;
	int k;

	for (k = 1; k < 64; k++) {
		term *= (h / k) * (h / k);
		sum += term;
		if (term < sum * 1e-17)
			break;
	}
	return sum;
}

/* Kaiser-windowed sinc table on [0, RS_ZC] zero crossings. */
static const float *
sinc_table(void)
{
	static float tab[RS_ZC * RS_OS + 2];
	static int ready;
	double u, r, k0;
	int i;

	if (ready)
		return tab;
	k0 = bessel_i0(RS_BETA);
	for (i = 0; i <= RS_ZC * RS_OS; i++) {
		u = (double)i / RS_OS;
		r = u / RS_ZC;
		tab[i] = (float)((i == 0 ? 1.0 : sin(M_PI * u) / (M_PI * u)) *
		    bessel_i0(RS_BETA * sqrt(1.0 - r * r)) / k0);
	}
	tab[RS_ZC * RS_OS + 1] = 0.0f;
	ready = 1;
	return tab;
}

/* Filter value at distance x input samples, for cutoff fc. */
static double
kernel(const float *tab, double x, double fc)
{
	double u = fabs(x) * 2.0 * fc * RS_OS, f;
	size_t idx = (size_t)u;

	if (idx >= RS_ZC * RS_OS)
		return 0.0;
	f = u - (double)idx;
	return 2.0 * fc * (tab[idx] + f * (tab[idx + 1] - tab[idx]));
}

static uint64_t
gcd(uint64_t a, uint64_t b)
{
	uint64_t t;

	while (b != 0) {
		t = a % b;
		a = b;
		b = t;
	}
	return a;
}

/* One output sample at input time t, with bounds checks. */
static float
resample_at(const struct pcm *in, const float *tab, double t, double fc, double w)
{
	int64_t j, lo = (int64_t)ceil(t - w), hi = (int64_t)floor(t + w);
	double acc = 0.0;

	if (lo < 0)
		lo = 0;
	if (hi > (int64_t)in->n - 1)
		hi = (int64_t)in->n - 1;
	for (j = lo; j <= hi; j++)
		acc += in->x[j] * kernel(tab, t - (double)j, fc);
	return (float)acc;
}

/*
 * Polyphase filter bank for a rational ratio: nph phases of ntap
 * taps. Tap k of phase p weighs input base - half + 1 + k.
 */
static float *
make_bank(const float *tab, size_t nph, size_t ntap, double fc)
{
	float *bank = malloc(nph * ntap * sizeof(float));
	size_t p, k, half = ntap / 2;

	if (bank == NULL)
		return NULL;
	for (p = 0; p < nph; p++)
		for (k = 0; k < ntap; k++)
			bank[p * ntap + k] = (float)kernel(tab, (double)p / nph -
			    ((double)k - (double)half + 1.0), fc);
	return bank;
}

static void
resample_bank(const struct pcm *in, struct pcm *out, const float *tab, double fc,
    double w, uint64_t up, uint64_t down)
{
	size_t ntap = 2 * ((size_t)w + 1), half = ntap / 2, i, k;
	const float *c, *x;
	uint64_t pos, base;
	float *bank;
	double acc;

	bank = up * ntap <= ((size_t)1 << 22) ? make_bank(tab, (size_t)up, ntap, fc) : NULL;
	for (i = 0; i < out->n; i++) {
		pos = (uint64_t)i * down;
		base = pos / up;
		if (bank == NULL || base + 1 < half || base + half >= in->n) {
			out->x[i] = resample_at(in, tab, (double)pos / (double)up, fc, w);
			continue;
		}
		c = bank + (pos % up) * ntap;
		x = in->x + base + 1 - half;
		acc = 0.0;
		for (k = 0; k < ntap; k++)
			acc += c[k] * x[k];
		out->x[i] = (float)acc;
	}
	free(bank);
}

/* Resample to rate with a Kaiser-windowed sinc low-pass filter. Returns 0 or -1. */
int
pcm_resample(const struct pcm *in, int rate, struct pcm *out)
{
	const float *tab = sinc_table();
	uint64_t g;
	double fc;
	size_t n;

	out->x = NULL;
	out->n = 0;
	if (rate < AUDIO_MIN_RATE || rate > AUDIO_MAX_RATE || in->n == 0)
		return -1;
	n = (size_t)((uint64_t)in->n * (uint64_t)rate / (uint64_t)in->rate);
	if (n == 0 || n > MAX_SAMPLES)
		return -1;
	if ((out->x = malloc(n * sizeof(float))) == NULL)
		return -1;
	out->n = n;
	out->rate = rate;
	if (rate == in->rate) {
		memcpy(out->x, in->x, n * sizeof(float));
		return 0;
	}
	/* Cutoff in cycles per input sample. */
	fc = 0.5 * RS_ROLLOFF * (rate < in->rate ? (double)rate / in->rate : 1.0);
	g = gcd((uint64_t)rate, (uint64_t)in->rate);
	resample_bank(in, out, tab, fc, RS_ZC / (2.0 * fc), (uint64_t)rate / g,
	    (uint64_t)in->rate / g);
	return 0;
}

static double
hz_to_mel(double f)
{
	const double f_sp = 200.0 / 3.0, logstep = log(6.4) / 27.0;

	if (f < 1000.0)
		return f / f_sp;
	return 1000.0 / f_sp + log(f / 1000.0) / logstep;
}

static double
mel_to_hz(double m)
{
	const double f_sp = 200.0 / 3.0, logstep = log(6.4) / 27.0;

	if (m < 1000.0 / f_sp)
		return m * f_sp;
	return 1000.0 * exp(logstep * (m - 1000.0 / f_sp));
}

/*
 * Slaney mel filter bank, as librosa.filters.mel(sr, n_fft, n_mels)
 * with fmin 0 and fmax sr / 2. Returns n_mels rows of n_fft / 2 + 1.
 */
float *
mel_filters(int n_mels, int n_fft, int sr)
{
	int i, k, nb = n_fft / 2 + 1;
	double *hz, lo, hi, v, fk, enorm, mmax;
	float *w;

	if (n_mels < 1 || n_mels > MEL_MAX_BINS || n_fft < 2 || sr < 1)
		return NULL;
	w = calloc((size_t)n_mels * (size_t)nb, sizeof(*w));
	hz = malloc((size_t)(n_mels + 2) * sizeof(*hz));
	if (w == NULL || hz == NULL) {
		free(w);
		free(hz);
		return NULL;
	}
	mmax = hz_to_mel(sr / 2.0);
	for (i = 0; i < n_mels + 2; i++)
		hz[i] = mel_to_hz(mmax * i / (n_mels + 1));
	for (i = 0; i < n_mels; i++) {
		enorm = 2.0 / (hz[i + 2] - hz[i]);
		for (k = 0; k < nb; k++) {
			fk = (double)k * sr / n_fft;
			lo = (fk - hz[i]) / (hz[i + 1] - hz[i]);
			hi = (hz[i + 2] - fk) / (hz[i + 2] - hz[i + 1]);
			v = lo < hi ? lo : hi;
			w[(size_t)i * nb + k] = v > 0.0 ? (float)(v * enorm) : 0.0f;
		}
	}
	free(hz);
	return w;
}

static double fft_cos[MEL_N_FFT], fft_sin[MEL_N_FFT], hann[MEL_N_FFT];

static void
fft_init(void)
{
	static int ready;
	int i;

	if (ready)
		return;
	for (i = 0; i < MEL_N_FFT; i++) {
		fft_cos[i] = cos(2.0 * M_PI * i / MEL_N_FFT);
		fft_sin[i] = -sin(2.0 * M_PI * i / MEL_N_FFT);
		hann[i] = 0.5 - 0.5 * cos(2.0 * M_PI * i / MEL_N_FFT);
	}
	ready = 1;
}

/*
 * Mixed-radix FFT of n complex values (interleaved) read with stride.
 * Even sizes split in two; odd sizes use a direct DFT. ts is
 * MEL_N_FFT / n, the step in the twiddle tables.
 */
static void
fft(const double *in, size_t stride, double *out, size_t n, size_t ts)
{
	double re, im, c, s, xr, xi, tr, ti;
	size_t j, k, h, t;

	if (n % 2 == 1) {
		for (k = 0; k < n; k++) {
			re = im = 0.0;
			for (j = 0, t = 0; j < n; j++) {
				xr = in[2 * j * stride];
				xi = in[2 * j * stride + 1];
				re += xr * fft_cos[t * ts] - xi * fft_sin[t * ts];
				im += xr * fft_sin[t * ts] + xi * fft_cos[t * ts];
				if ((t += k) >= n)
					t -= n;
			}
			out[2 * k] = re;
			out[2 * k + 1] = im;
		}
		return;
	}
	h = n / 2;
	fft(in, 2 * stride, out, h, 2 * ts);
	fft(in + 2 * stride, 2 * stride, out + 2 * h, h, 2 * ts);
	for (k = 0; k < h; k++) {
		c = fft_cos[k * ts];
		s = fft_sin[k * ts];
		tr = out[2 * (k + h)] * c - out[2 * (k + h) + 1] * s;
		ti = out[2 * (k + h)] * s + out[2 * (k + h) + 1] * c;
		out[2 * (k + h)] = out[2 * k] - tr;
		out[2 * (k + h) + 1] = out[2 * k + 1] - ti;
		out[2 * k] += tr;
		out[2 * k + 1] += ti;
	}
}

/* Sample j of the signal: audio, then n_pad zeros, reflected at both ends. */
static double
padded(const struct pcm *p, int64_t j, int64_t total)
{
	if (j < 0)
		j = -j;
	if (j >= total)
		j = 2 * (total - 1) - j;
	if (j < 0 || j >= (int64_t)p->n)
		return 0.0;
	return p->x[j];
}

/* Nonzero range of each filter row. */
static void
filter_ranges(const float *filt, int n_mels, int nb, int *lo, int *hi)
{
	int m, k;

	for (m = 0; m < n_mels; m++) {
		lo[m] = nb;
		hi[m] = 0;
		for (k = 0; k < nb; k++) {
			if (filt[(size_t)m * nb + k] != 0.0f) {
				if (lo[m] == nb)
					lo[m] = k;
				hi[m] = k + 1;
			}
		}
	}
}

static void
frame_power(const struct pcm *p, int64_t center, int64_t total, double *buf, double *spec,
    double *power)
{
	int i;

	for (i = 0; i < MEL_N_FFT; i++) {
		buf[2 * i] = padded(p, center - MEL_N_FFT / 2 + i, total) * hann[i];
		buf[2 * i + 1] = 0.0;
	}
	fft(buf, 1, spec, MEL_N_FFT, 1);
	for (i = 0; i <= MEL_N_FFT / 2; i++)
		power[i] = spec[2 * i] * spec[2 * i] + spec[2 * i + 1] * spec[2 * i + 1];
}

/*
 * Log-mel spectrogram of n_frames frames from sample start, into out
 * (n_mels rows of n_frames). The signal is p followed by 30 s of
 * zeros, as in Whisper. The clamp to max - 8 uses the max of this
 * window. Returns 0, or -1 for bad arguments.
 */
int
log_mel(const struct pcm *p, size_t start, const float *filt, int n_mels, int n_frames,
    float *out)
{
	const int nb = MEL_N_FFT / 2 + 1;
	double buf[2 * MEL_N_FFT], spec[2 * MEL_N_FFT], power[MEL_N_FFT / 2 + 1];
	int64_t total = (int64_t)p->n + 30 * MEL_RATE;
	int lo[MEL_MAX_BINS], hi[MEL_MAX_BINS], m, k, t;
	double acc, v, vmax = -1e30;
	size_t nout = (size_t)n_mels * (size_t)n_frames, i;

	if (n_mels < 1 || n_mels > MEL_MAX_BINS || n_frames < 1)
		return -1;
	fft_init();
	filter_ranges(filt, n_mels, nb, lo, hi);
	for (t = 0; t < n_frames; t++) {
		frame_power(p, (int64_t)start + (int64_t)t * MEL_HOP, total, buf, spec, power);
		for (m = 0; m < n_mels; m++) {
			acc = 0.0;
			for (k = lo[m]; k < hi[m]; k++)
				acc += filt[(size_t)m * nb + k] * power[k];
			v = log10(acc > 1e-10 ? acc : 1e-10);
			out[(size_t)m * n_frames + t] = (float)v;
			if (v > vmax)
				vmax = v;
		}
	}
	for (i = 0; i < nout; i++) {
		v = out[i] < vmax - 8.0 ? vmax - 8.0 : out[i];
		out[i] = (float)((v + 4.0) / 4.0);
	}
	return 0;
}
