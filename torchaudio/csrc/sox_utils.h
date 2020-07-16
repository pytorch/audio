#ifndef TORCHAUDIO_SOX_UTILS_H
#define TORCHAUDIO_SOX_UTILS_H

#include <sox.h>
#include <torch/script.h>

namespace torchaudio {
namespace sox_utils {

////////////////////////////////////////////////////////////////////////////////
// APIs for Python interaction
////////////////////////////////////////////////////////////////////////////////

/// Set sox global options
void set_seed(const int64_t seed);

void set_verbosity(const int64_t verbosity);

void set_use_threads(const bool use_threads);

void set_buffer_size(const int64_t buffer_size);

std::vector<std::vector<std::string>> list_effects();

std::vector<std::string> list_formats();

/// Class for exchanging signal infomation (tensor + meta data) between
/// C++ and Python for read/write operation.
struct TensorSignal : torch::CustomClassHolder {
  torch::Tensor tensor;
  int64_t sample_rate;
  bool channels_first;

  TensorSignal(
      torch::Tensor tensor_,
      int64_t sample_rate_,
      bool channels_first_);

  torch::Tensor getTensor() const;
  int64_t getSampleRate() const;
  bool getChannelsFirst() const;
};

////////////////////////////////////////////////////////////////////////////////
// Utilities for sox_io / sox_effects implementations
////////////////////////////////////////////////////////////////////////////////

const std::unordered_set<std::string> UNSUPPORTED_EFFECTS =
    {"input", "output", "spectrogram", "noiseprof", "noisered", "splice"};

/// helper class to automatically close sox_format_t*
struct SoxFormat {
  explicit SoxFormat(sox_format_t* fd) noexcept;
  SoxFormat(const SoxFormat& other) = delete;
  SoxFormat(SoxFormat&& other) = delete;
  SoxFormat& operator=(const SoxFormat& other) = delete;
  SoxFormat& operator=(SoxFormat&& other) = delete;
  ~SoxFormat();
  sox_format_t* operator->() const noexcept;
  operator sox_format_t*() const noexcept;

 private:
  sox_format_t* fd_;
};

///
/// Verify that input file is found, has known encoding, and not empty
void validate_input_file(const SoxFormat& sf);

///
/// Verify that input Tensor is 2D, CPU and either uin8, int16, int32 or float32
void validate_input_tensor(const torch::Tensor);

///
/// Get target dtype for the given encoding and precision.
caffe2::TypeMeta get_dtype(
    const sox_encoding_t encoding,
    const unsigned precision);

///
/// Convert sox_sample_t buffer to uint8/int16/int32/float32 Tensor
/// NOTE: This function might modify the values in the input buffer to
/// reduce the number of memory copy.
/// @param buffer Pointer to buffer that contains audio data.
/// @param num_samples The number of samples to read.
/// @param num_channels The number of channels. Used to reshape the resulting
/// Tensor.
/// @param dtype Target dtype. Determines the output dtype and value range in
/// conjunction with normalization.
/// @param noramlize Perform normalization. Only effective when dtype is not
/// kFloat32. When effective, the output tensor is kFloat32 type and value range
/// is [-1.0, 1.0]
/// @param channels_first When True, output Tensor has shape of [num_channels,
/// num_frames].
torch::Tensor convert_to_tensor(
    sox_sample_t* buffer,
    const int32_t num_samples,
    const int32_t num_channels,
    const caffe2::TypeMeta dtype,
    const bool normalize,
    const bool channels_first);

///
/// Convert float32/int32/int16/uint8 Tensor to int32 for Torch -> Sox
/// conversion.
torch::Tensor unnormalize_wav(const torch::Tensor);

/// Extract extension from file path
const std::string get_filetype(const std::string path);

/// Get sox_signalinfo_t for passing a torch::Tensor object.
sox_signalinfo_t get_signalinfo(
    const TensorSignal* signal,
    const std::string filetype);

/// Get sox_encofinginfo_t for saving audoi file
sox_encodinginfo_t get_encodinginfo(
    const std::string filetype,
    const caffe2::TypeMeta dtype,
    const double compression);

} // namespace sox_utils
} // namespace torchaudio
#endif
