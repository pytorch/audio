#include <torchaudio/csrc/ffmpeg/stream_writer/audio_output_stream.h>

namespace torchaudio::io {

AudioOutputStream::AudioOutputStream(
    AVFormatContext* format_ctx,
    AVCodecContextPtr&& codec_ctx,
    std::unique_ptr<FilterGraph>&& filter,
    AVFramePtr&& src_frame,
    int64_t frame_capacity_)
    : OutputStream(
          format_ctx,
          std::move(codec_ctx),
          std::move(filter),
          std::move(src_frame)),
      frame_capacity(frame_capacity_) {}

namespace {

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

} // namespace

void AudioOutputStream::write_chunk(const torch::Tensor& waveform) {
  validate_audio_input(
      static_cast<AVSampleFormat>(src_frame->format), codec_ctx, waveform);

  AVRational time_base{1, codec_ctx->sample_rate};

  using namespace torch::indexing;
  AT_DISPATCH_ALL_TYPES(waveform.scalar_type(), "write_audio_frames", [&] {
    for (int64_t i = 0; i < waveform.size(0); i += frame_capacity) {
      auto chunk = waveform.index({Slice(i, i + frame_capacity), Slice()});
      auto num_valid_frames = chunk.size(0);
      auto byte_size = chunk.numel() * chunk.element_size();
      chunk = chunk.reshape({-1}).contiguous();

      // TODO: make writable
      // https://ffmpeg.org/doxygen/4.1/muxing_8c_source.html#l00334
      TORCH_CHECK(
          av_frame_is_writable(src_frame),
          "Internal Error: frame is not writable.");

      memcpy(
          src_frame->data[0],
          static_cast<void*>(chunk.data_ptr<scalar_t>()),
          byte_size);
      src_frame->pts =
          av_rescale_q(num_frames, time_base, codec_ctx->time_base);
      src_frame->nb_samples = num_valid_frames;
      num_frames += num_valid_frames;

      process_frame(src_frame);
    }
  });
}

} // namespace torchaudio::io
