#ifndef TORCHAUDIO_SOX_UTILS_H
#define TORCHAUDIO_SOX_UTILS_H

#include <sox.h>
#include <torch/script.h>

#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#endif // TORCH_API_INCLUDE_EXTENSION_H

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

std::vector<std::string> list_read_formats();

std::vector<std::string> list_write_formats();

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

  void close();

 private:
  sox_format_t* fd_;
};

///
/// Verify that input file is found, has known encoding, and not empty
void validate_input_file(const SoxFormat& sf, bool check_length = true);

///
/// Verify that input Tensor is 2D, CPU and either uin8, int16, int32 or float32
void validate_input_tensor(const torch::Tensor);

///
/// Get target dtype for the given encoding and precision.
caffe2::TypeMeta get_dtype(
    const sox_encoding_t encoding,
    const unsigned precision);

caffe2::TypeMeta get_dtype_from_str(const std::string dtype);

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

/// Extract extension from file path
const std::string get_filetype(const std::string path);

/// Get sox_signalinfo_t for passing a torch::Tensor object.
sox_signalinfo_t get_signalinfo(
    const torch::Tensor* waveform,
    const int64_t sample_rate,
    const std::string filetype,
    const bool channels_first);

/// Get sox_encodinginfo_t for Tensor I/O
sox_encodinginfo_t get_tensor_encodinginfo(const caffe2::TypeMeta dtype);

/// Get sox_encodinginfo_t for saving to file/file object
sox_encodinginfo_t get_encodinginfo_for_save(
    const std::string& format,
    const caffe2::TypeMeta dtype,
    const c10::optional<double> compression,
    const c10::optional<std::string> encoding,
    const c10::optional<int64_t> bits_per_sample);

#ifdef TORCH_API_INCLUDE_EXTENSION_H

uint64_t read_fileobj(py::object* fileobj, uint64_t size, char* buffer);

#endif // TORCH_API_INCLUDE_EXTENSION_H

} // namespace sox_utils
} // namespace torchaudio
#endif
