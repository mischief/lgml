/* Audio in C: WAV decoding, resampling and the Whisper log-mel spectrogram. */
#ifndef AUDIO_H
#define AUDIO_H

#include <stddef.h>

#define AUDIO_MIN_RATE	1000
#define AUDIO_MAX_RATE	768000

#define MEL_N_FFT	400
#define MEL_HOP		160
#define MEL_RATE	16000
#define MEL_MAX_BINS	512

/* Mono float samples. */
struct pcm {
	float	*x;
	size_t	 n;
	int	 rate;
};

enum wav_format {
	WAV_S16,
	WAV_S24,
	WAV_S32,
	WAV_F32
};

void		 pcm_free(struct pcm *);
const char	*wav_decode(const void *, size_t, struct pcm *);
void		*wav_encode(const struct pcm *, int, enum wav_format, size_t *);
int		 pcm_resample(const struct pcm *, int, struct pcm *);
float		*mel_filters(int, int, int);
int		 log_mel(const struct pcm *, size_t, const float *, int, int, float *);

#endif
