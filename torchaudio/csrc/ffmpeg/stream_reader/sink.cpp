#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/chunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/unchunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/sink.h>
#include <stdexcept>

namespace torchaudio {
namespace ffmpeg {

namespace {
std::unique_ptr<Buffer> get_buffer(
    AVMediaType type,
    int frames_per_chunk,
    int num_chunks,
    const torch::Device& device) {
  TORCH_CHECK(
      frames_per_chunk > 0 || frames_per_chunk == -1,
      "`frames_per_chunk` must be positive or -1. Found: ",
      frames_per_chunk);

  TORCH_CHECK(
      num_chunks > 0 || num_chunks == -1,
      "`num_chunks` must be positive or -1. Found: ",
      num_chunks);

  TORCH_INTERNAL_ASSERT(
      type == AVMEDIA_TYPE_AUDIO || type == AVMEDIA_TYPE_VIDEO,
      "Unsupported media type: ",
      av_get_media_type_string(type),
      ". Only video or audio is supported ");

  // Chunked Mode
  if (frames_per_chunk > 0) {
    if (type == AVMEDIA_TYPE_AUDIO) {
      return std::unique_ptr<Buffer>(
          new detail::ChunkedAudioBuffer(frames_per_chunk, num_chunks));
    } else {
      return std::unique_ptr<Buffer>(
          new detail::ChunkedVideoBuffer(frames_per_chunk, num_chunks, device));
    }
  } else { // unchunked mode
    if (type == AVMEDIA_TYPE_AUDIO) {
      return std::unique_ptr<Buffer>(new detail::UnchunkedAudioBuffer());
    } else {
      return std::unique_ptr<Buffer>(new detail::UnchunkedVideoBuffer(device));
    }
  }
}

std::unique_ptr<FilterGraph> get_filter_graph(
    AVRational input_time_base,
    AVCodecParameters* codecpar,
    const std::string& filter_description) {
  auto p = std::make_unique<FilterGraph>(codecpar->codec_type);

  switch (codecpar->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
      p->add_audio_src(
          static_cast<AVSampleFormat>(codecpar->format),
          input_time_base,
          codecpar->sample_rate,
          codecpar->channel_layout);
      break;
    case AVMEDIA_TYPE_VIDEO:
      p->add_video_src(
          static_cast<AVPixelFormat>(codecpar->format),
          input_time_base,
          codecpar->width,
          codecpar->height,
          codecpar->sample_aspect_ratio);
      break;
    default:
      TORCH_CHECK(false, "Only audio/video are supported.");
  }
  p->add_sink();
  p->add_process(filter_description);
  p->create_filter();
  return p;
}

} // namespace

Sink::Sink(
    AVRational input_time_base_,
    AVCodecParameters* codecpar_,
    int frames_per_chunk,
    int num_chunks,
    const c10::optional<std::string>& filter_description_,
    const torch::Device& device)
    : input_time_base(input_time_base_),
      codecpar(codecpar_),
      filter_description(filter_description_.value_or(
          codecpar->codec_type == AVMEDIA_TYPE_AUDIO ? "anull" : "null")),
      filter(get_filter_graph(input_time_base_, codecpar_, filter_description)),
      buffer(get_buffer(
          codecpar_->codec_type,
          frames_per_chunk,
          num_chunks,
          device)) {}

// 0: some kind of success
// <0: Some error happened
int Sink::process_frame(AVFrame* pFrame) {
  int ret = filter->add_frame(pFrame);
  while (ret >= 0) {
    ret = filter->get_frame(frame);
    //  AVERROR(EAGAIN) means that new input data is required to return new
    //  output.
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      return 0;
    }
    if (ret >= 0) {
      buffer->push_frame(frame);
    }
    av_frame_unref(frame);
  }
  return ret;
}

std::string Sink::get_filter_description() const {
  return filter_description;
}

void Sink::flush() {
  filter = get_filter_graph(input_time_base, codecpar, filter_description);
  buffer->flush();
}

} // namespace ffmpeg
} // namespace torchaudio
