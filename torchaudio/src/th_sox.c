
#include <TH/TH.h>

#include <sox.h>

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define libthsox_(NAME) TH_CONCAT_4(libthsox_, Real, _, NAME)

#include "generic/th_sox.c"
#include "THGenerateAllTypes.h"

void libthsox_get_info(const char *file_name, int *bits_per_sample, unsigned long *length,
                       unsigned int *sample_rate, unsigned int *nchannels)
{
  // Create sox objects and read into int32_t buffer
  sox_format_t *fd;
  fd = sox_open_read(file_name, NULL, NULL, NULL);
  if (fd == NULL)
    THError("[read_audio_file] Failure to read file");
  *bits_per_sample = fd->signal.precision;
  *length = fd->signal.length;
  *sample_rate = fd->signal.rate;
  *nchannels = fd->signal.channels;
  sox_close(fd);
}
