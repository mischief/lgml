/* WAV reader, resampler and mel tests. They need no model. Output is TAP. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio.h"

#ifndef M_PI
#define M_PI	3.14159265358979323846
#endif

static int ntest, nfail;

static void
ok(int cond, const char *name)
{
	ntest++;
	if (!cond)
		nfail++;
	printf("%s %d - %s\n", cond ? "ok" : "not ok", ntest, name);
}

static struct pcm
sine(int rate, double hz, double amp, double secs)
{
	struct pcm p;
	size_t i;

	p.rate = rate;
	p.n = (size_t)(rate * secs);
	p.x = malloc(p.n * sizeof(float));
	if (p.x == NULL)
		abort();
	for (i = 0; i < p.n; i++)
		p.x[i] = (float)(amp * sin(2.0 * M_PI * hz * (double)i / rate));
	return p;
}

/* Amplitude of the hz component, and the RMS of the rest, over the middle half. */
static void
fit(const struct pcm *p, double hz, double *amp, double *resid)
{
	size_t i, lo = p->n / 4, hi = 3 * p->n / 4;
	double s = 0, c = 0, w, e = 0, a, b, y;

	for (i = lo; i < hi; i++) {
		w = 2.0 * M_PI * hz * (double)i / p->rate;
		s += p->x[i] * sin(w);
		c += p->x[i] * cos(w);
	}
	a = 2.0 * s / (double)(hi - lo);
	b = 2.0 * c / (double)(hi - lo);
	for (i = lo; i < hi; i++) {
		w = 2.0 * M_PI * hz * (double)i / p->rate;
		y = p->x[i] - a * sin(w) - b * cos(w);
		e += y * y;
	}
	*amp = sqrt(a * a + b * b);
	*resid = sqrt(e / (double)(hi - lo));
}

static double
max_diff(const struct pcm *a, const struct pcm *b)
{
	double d, m = 0;
	size_t i;

	if (a->n != b->n)
		return 1e9;
	for (i = 0; i < a->n; i++) {
		d = fabs((double)a->x[i] - b->x[i]);
		if (d > m)
			m = d;
	}
	return m;
}

static void
round_trip(enum wav_format fmt, int channels, double tol, const char *name)
{
	struct pcm src = sine(44100, 1000, 0.5, 0.25), dst;
	const char *err;
	size_t len;
	void *buf;

	buf = wav_encode(&src, channels, fmt, &len);
	err = buf != NULL ? wav_decode(buf, len, &dst) : "encode failed";
	ok(err == NULL && dst.rate == 44100 && max_diff(&src, &dst) < tol, name);
	if (err == NULL)
		pcm_free(&dst);
	free(buf);
	pcm_free(&src);
}

static void
test_formats(void)
{
	round_trip(WAV_S16, 1, 1.0 / 32000, "s16 mono round trip");
	round_trip(WAV_S16, 2, 1.0 / 32000, "s16 stereo round trip");
	round_trip(WAV_S24, 1, 1.0 / 8000000, "s24 round trip");
	round_trip(WAV_S32, 2, 1e-7, "s32 stereo round trip");
	round_trip(WAV_F32, 1, 1e-9, "f32 round trip");
}

static void
test_mixdown(void)
{
	struct pcm one = { NULL, 1, 16000 }, dst;
	float x = 0;
	unsigned char *buf;
	size_t len;

	one.x = &x;
	buf = wav_encode(&one, 2, WAV_S16, &len);
	/* Left 0.5, right -0.25: the mix is 0.125. */
	buf[44] = 0x00;
	buf[45] = 0x40;
	buf[46] = 0x00;
	buf[47] = 0xe0;
	ok(wav_decode(buf, len, &dst) == NULL && dst.n == 1 &&
	    fabs(dst.x[0] - 0.125) < 1e-6, "stereo mixes down to mono");
	pcm_free(&dst);
	free(buf);
}

static void
test_bad(void)
{
	struct pcm src = sine(16000, 440, 0.5, 0.01), dst;
	unsigned char *buf, *cp;
	size_t len, i, k;
	uint32_t seed = 1;

	buf = wav_encode(&src, 1, WAV_S16, &len);
	ok(wav_decode(buf, 11, &dst) != NULL, "short file rejected");
	ok(wav_decode(buf, 44, &dst) != NULL, "file with no samples rejected");
	/* A data size past the end clamps to the bytes present. */
	buf[40] = buf[41] = buf[42] = buf[43] = 0xff;
	ok(wav_decode(buf, len - 10, &dst) == NULL && dst.n == (len - 10 - 44) / 2,
	    "oversized data chunk clamps");
	pcm_free(&dst);
	buf[32] = 3;
	ok(wav_decode(buf, len, &dst) != NULL, "block size mismatch rejected");
	buf[32] = 2;
	buf[22] = 0;
	ok(wav_decode(buf, len, &dst) != NULL, "zero channels rejected");
	buf[22] = 1;
	buf[24] = buf[25] = buf[26] = buf[27] = 0;
	ok(wav_decode(buf, len, &dst) != NULL, "zero rate rejected");
	buf[24] = 0x80;
	buf[25] = 0x3e;
	buf[16] = 0xff;
	ok(wav_decode(buf, len, &dst) != NULL, "fmt size past end rejected");
	buf[16] = 16;

	/* Random byte flips in the header area must not crash or leak. */
	cp = malloc(len);
	for (i = 0; i < 20000 && cp != NULL; i++) {
		memcpy(cp, buf, len);
		for (k = 0; k < 4; k++) {
			seed = seed * 1103515245 + 12345;
			cp[(seed >> 8) % 64] = (unsigned char)(seed >> 20);
		}
		if (wav_decode(cp, (seed >> 4) % (len + 1), &dst) == NULL)
			pcm_free(&dst);
	}
	ok(1, "random header corruption survives");
	free(cp);
	free(buf);
	pcm_free(&src);
}

static void
check_resample(int from, int to, double hz, const char *name)
{
	struct pcm src = sine(from, hz, 0.5, 1.0), dst;
	double amp = 0, resid = 1;
	char msg[128];

	if (pcm_resample(&src, to, &dst) == 0) {
		fit(&dst, hz, &amp, &resid);
		pcm_free(&dst);
	}
	snprintf(msg, sizeof(msg), "%s (amp %.4f, residual %.2g)", name, amp, resid);
	ok(fabs(amp - 0.5) < 0.005 && resid < 1e-3, msg);
	pcm_free(&src);
}

static void
test_resample(void)
{
	struct pcm src = sine(44100, 12000, 0.5, 1.0), dst;
	size_t i;
	double e = 0;

	check_resample(44100, 16000, 1000, "44.1 kHz to 16 kHz keeps 1 kHz");
	check_resample(48000, 16000, 7000, "48 kHz to 16 kHz keeps 7 kHz");
	check_resample(8000, 16000, 1000, "8 kHz to 16 kHz keeps 1 kHz");
	check_resample(22050, 16000, 3000, "22.05 kHz to 16 kHz keeps 3 kHz");

	/* 12 kHz is above the new Nyquist rate and must not alias. */
	if (pcm_resample(&src, 16000, &dst) == 0) {
		for (i = dst.n / 4; i < 3 * dst.n / 4; i++)
			e += (double)dst.x[i] * dst.x[i];
		e = sqrt(e / (double)(dst.n / 2));
		printf("# 12 kHz residual rms %.3g\n", e);
		ok(dst.n == 16000 && e < 0.5e-3, "12 kHz is removed at 16 kHz");
		pcm_free(&dst);
	} else
		ok(0, "12 kHz is removed at 16 kHz");
	pcm_free(&src);
	src = sine(16000, 440, 0.5, 0.1);
	ok(pcm_resample(&src, 16000, &dst) == 0 && max_diff(&src, &dst) == 0,
	    "same rate copies");
	pcm_free(&dst);
	ok(pcm_resample(&src, 10, &dst) != 0, "bad rate rejected");
	pcm_free(&src);
}

static void
test_mel(void)
{
	struct pcm quiet = { NULL, 0, 16000 }, tone;
	float *filt, *out;
	double sum, best = -1e9;
	int m, k, peak = -1, nb = MEL_N_FFT / 2 + 1;

	filt = mel_filters(128, MEL_N_FFT, MEL_RATE);
	out = malloc(128 * 3000 * sizeof(float));
	/* Slaney filters integrate to about 2 / bandwidth; check coverage. */
	for (sum = 0, k = 0; k < nb; k++)
		sum += filt[64 * nb + k];
	ok(filt != NULL && sum > 0, "mel filters built");
	quiet.x = calloc(16000, sizeof(float));
	quiet.n = 16000;
	ok(log_mel(&quiet, 0, filt, 128, 3000, out) == 0 && out[0] == -1.5f &&
	    out[128 * 3000 - 1] == -1.5f, "silence gives -1.5 everywhere");
	tone = sine(16000, 1000, 0.5, 30.0);
	log_mel(&tone, 0, filt, 128, 3000, out);
	for (m = 0; m < 128; m++) {
		if (out[m * 3000 + 1500] > best) {
			best = out[m * 3000 + 1500];
			peak = m;
		}
	}
	/* 1 kHz is mel 15; bin centers step by about 0.356 mel over 128 bins. */
	ok(peak >= 40 && peak <= 43, "1 kHz tone peaks near mel bin 41");
	pcm_free(&tone);
	free(quiet.x);
	free(filt);
	free(out);
}

int
main(void)
{
	test_formats();
	test_mixdown();
	test_bad();
	test_resample();
	test_mel();
	printf("1..%d\n", ntest);
	return nfail == 0 ? 0 : 1;
}
