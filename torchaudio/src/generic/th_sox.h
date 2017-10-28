#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/th_sox.h"
#else

void libthsox_(read_audio_file)(const char *file_name, THTensor* tensor, int* sample_rate, unsigned long *total_frames, size_t offset, long nframes);
void libthsox_(write_audio_file)(const char *file_name, THTensor* src, const char *extension, int sample_rate);
#endif
