#include <torchaudio/csrc/ffmpeg/stream_writer/audio_converter.h>

namespace torchaudio::io {

namespace {

void validate_audio_input(AVFrame* buffer, const torch::Tensor& t) {
  auto dtype = t.dtype().toScalarType();
  switch (static_cast<AVSampleFormat>(buffer->format)) {
    case AV_SAMPLE_FMT_U8:
      TORCH_CHECK(
          dtype == c10::ScalarType::Byte, "Expected Tensor of uint8 type.");
      break;
    case AV_SAMPLE_FMT_S16:
      TORCH_CHECK(
          dtype == c10::ScalarType::Short, "Expected Tensor of int16 type.");
      break;
    case AV_SAMPLE_FMT_S32:
      TORCH_CHECK(
          dtype == c10::ScalarType::Int, "Expected Tensor of int32 type.");
      break;
    case AV_SAMPLE_FMT_S64:
      TORCH_CHECK(
          dtype == c10::ScalarType::Long, "Expected Tensor of int64 type.");
      break;
    case AV_SAMPLE_FMT_FLT:
      TORCH_CHECK(
          dtype == c10::ScalarType::Float, "Expected Tensor of float32 type.");
      break;
    case AV_SAMPLE_FMT_DBL:
      TORCH_CHECK(
          dtype == c10::ScalarType::Double, "Expected Tensor of float64 type.");
      break;
    default:
      TORCH_CHECK(
          false,
          "Internal error: Audio encoding stream is not properly configured.");
  }
  TORCH_CHECK(t.device().is_cpu(), "Input tensor has to be on CPU.");
  TORCH_CHECK(t.dim() == 2, "Input Tensor has to be 2D.");
  const auto num_channels = t.size(1);
  TORCH_CHECK(
      num_channels == buffer->channels,
      "Expected waveform with ",
      buffer->channels,
      " channels. Found ",
      num_channels);
}

// 2D (time, channel) and contiguous.
void convert_func_(const torch::Tensor& chunk, AVFrame* buffer) {
  auto num_frames = chunk.size(0);
  auto byte_size = chunk.numel() * chunk.element_size();
  // TODO: make writable
  // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00334
  TORCH_CHECK(av_frame_is_writable(buffer), "frame is not writable.");

  memcpy(buffer->data[0], chunk.data_ptr(), byte_size);
  buffer->nb_samples = static_cast<int>(num_frames);
}

} // namespace

AudioTensorConverter::AudioTensorConverter(
    AVFrame* buffer_,
    const int64_t buffer_size_)
    : buffer(buffer_), buffer_size(buffer_size_), convert_func(convert_func_) {}

SlicingTensorConverter AudioTensorConverter::convert(
    const torch::Tensor& frames) {
  validate_audio_input(buffer, frames);
  return SlicingTensorConverter{
      frames.contiguous(),
      buffer,
      convert_func,
      buffer_size,
  };
}

} // namespace torchaudio::io
