#include <torchaudio/csrc/ffmpeg/stream_writer/audio_converter.h>

namespace torchaudio::io {

namespace {

AVFramePtr get_audio_frame(
    AVSampleFormat src_fmt,
    AVCodecContext* codec_ctx,
    int default_frame_size) {
  AVFramePtr frame{};
  frame->pts = 0;
  frame->format = src_fmt;
  frame->channel_layout = codec_ctx->channel_layout;
  frame->sample_rate = codec_ctx->sample_rate;
  frame->nb_samples =
      codec_ctx->frame_size ? codec_ctx->frame_size : default_frame_size;
  if (frame->nb_samples) {
    int ret = av_frame_get_buffer(frame, 0);
    TORCH_CHECK(
        ret >= 0,
        "Error allocating an audio buffer (",
        av_err2string(ret),
        ").");
  }
  return frame;
}

void validate_audio_input(
    enum AVSampleFormat fmt,
    AVCodecContext* ctx,
    const torch::Tensor& t) {
  auto dtype = t.dtype().toScalarType();
  switch (fmt) {
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
      num_channels == ctx->channels,
      "Expected waveform with ",
      ctx->channels,
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
    enum AVSampleFormat src_fmt_,
    AVCodecContext* codec_ctx_,
    int default_frame_size)
    : src_fmt(src_fmt_),
      codec_ctx(codec_ctx_),
      buffer(get_audio_frame(src_fmt_, codec_ctx_, default_frame_size)),
      buffer_size(buffer->nb_samples),
      convert_func(convert_func_) {}

SlicingTensorConverter AudioTensorConverter::convert(
    const torch::Tensor& frames) {
  validate_audio_input(src_fmt, codec_ctx, frames);
  return SlicingTensorConverter{
      frames.contiguous(),
      buffer,
      convert_func,
      buffer_size,
  };
}

} // namespace torchaudio::io
