#ifndef TORCHAUDIO_SOX_UTILS_H
#define TORCHAUDIO_SOX_UTILS_H

#include <sox.h>
#include <torch/script.h>

namespace torchaudio {
namespace sox_utils {

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

/// helper class to automatically close sox_format_t*
struct SoxFormat {
  explicit SoxFormat(sox_format_t* fd) noexcept;
  SoxFormat(const SoxFormat& other) = delete;
  SoxFormat(SoxFormat&& other) = delete;
  SoxFormat& operator=(const SoxFormat& other) = delete;
  SoxFormat& operator=(SoxFormat&& other) = delete;
  ~SoxFormat();
  sox_format_t* operator->() const noexcept;
  sox_format_t* get() const noexcept;

 private:
  sox_format_t* fd_;
};

///
/// Verify that input file is found, has known encoding, and not empty
void validate_input_file(const SoxFormat& sf);

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

} // namespace sox_utils
} // namespace torchaudio
#endif
