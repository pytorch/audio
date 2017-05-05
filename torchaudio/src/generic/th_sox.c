#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/th_sox.c"
#else

void libthsox_(read_audio)(sox_format_t *fd, THTensor* tensor,
                         int* sample_rate, size_t nsamples)
{
  int nchannels = fd->signal.channels;
  long buffer_size = fd->signal.length;
  if (buffer_size == 0) {
    if (nsamples != -1) {
      buffer_size = nsamples;
    } else {
      THError("[read_audio] Unknown length");
    }
  }
  *sample_rate = (int) fd->signal.rate;
  int32_t *buffer = (int32_t *)malloc(sizeof(int32_t) * buffer_size);
  size_t samples_read = sox_read(fd, buffer, buffer_size);
  if (samples_read == 0)
    THError("[read_audio] Empty file or read failed in sox_read");
  // alloc tensor
  THTensor_(resize2d)(tensor, samples_read / nchannels, nchannels );
  real *tensor_data = THTensor_(data)(tensor);
  // convert audio to dest tensor
  int x,k;
  for (x=0; x<samples_read/nchannels; x++) {
    for (k=0; k<nchannels; k++) {
      *tensor_data++ = (real)buffer[x*nchannels+k];
    }
  }
  // free buffer and sox structures
  free(buffer);
}

void libthsox_(read_audio_file)(const char *file_name, THTensor* tensor, int* sample_rate)
{
  // Create sox objects and read into int32_t buffer
  sox_format_t *fd;
  fd = sox_open_read(file_name, NULL, NULL, NULL);
  if (fd == NULL)
    THError("[read_audio_file] Failure to read file");
  libthsox_(read_audio)(fd, tensor, sample_rate, -1);
  sox_close(fd);
}

#endif
