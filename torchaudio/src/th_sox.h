/* #include <TH/TH.h> */

/* #define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME) */
/* #define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor) */
/* #define libthsox_(NAME) TH_CONCAT_4(libthsox_, Real, _, NAME) */

/* #include "generic/th_sox.h" */
/* #include "THGenerateAllTypes.h" */

/* gcc -E th_sox.h -I /home/soumith/code/pytorch/torch/lib/include/TH -I /home/soumith/code/pytorch/torch/lib/include/ -I .|grep libthsox */
void libthsox_Float_read_audio_file(const char *file_name, THFloatTensor *tensor, int *sample_rate,
                                    unsigned long *total_frames, size_t offset, long nframes);
void libthsox_Double_read_audio_file(const char *file_name, THDoubleTensor *tensor, int *sample_rate,
                                     unsigned long *total_frames, size_t offset, long nframes);
void libthsox_Byte_read_audio_file(const char *file_name, THByteTensor *tensor, int *sample_rate,
                                   unsigned long *total_frames, size_t offset, long nframes);
void libthsox_Char_read_audio_file(const char *file_name, THCharTensor* tensor, int* sample_rate,
                                   unsigned long *total_frames, size_t offset, long nframes);
void libthsox_Short_read_audio_file(const char *file_name, THShortTensor* tensor, int* sample_rate,
                                   unsigned long *total_frames, size_t offset, long nframes);
void libthsox_Int_read_audio_file(const char *file_name, THIntTensor* tensor, int* sample_rate,
                                  unsigned long *total_frames, size_t offset, long nframes);
void libthsox_Long_read_audio_file(const char *file_name, THLongTensor* tensor, int* sample_rate,
                                   unsigned long *total_frames, size_t offset, long nframes);

void libthsox_Float_write_audio_file(const char *file_name, THFloatTensor* tensor, const char *extension,
                                        int sample_rate);
void libthsox_Double_write_audio_file(const char *file_name, THDoubleTensor* tensor, const char *extension,
                                        int sample_rate);
void libthsox_Byte_write_audio_file(const char *file_name, THByteTensor* tensor, const char *extension,
                                        int sample_rate);
void libthsox_Char_write_audio_file(const char *file_name, THCharTensor* tensor, const char *extension,
                                        int sample_rate);
void libthsox_Short_write_audio_file(const char *file_name, THShortTensor* tensor, const char *extension,
                                        int sample_rate);
void libthsox_Int_write_audio_file(const char *file_name, THIntTensor* tensor, const char *extension,
                                        int sample_rate);
void libthsox_Long_write_audio_file(const char *file_name, THLongTensor* tensor, const char *extension,
                                        int sample_rate);

void libthsox_get_info(const char *file_name, int *bits_per_sample, unsigned long *length,
                       unsigned int *sample_rate, unsigned int *nchannels);
