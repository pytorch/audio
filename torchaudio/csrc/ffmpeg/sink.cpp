#include <torchaudio/csrc/ffmpeg/sink.h>

namespace torchaudio {
namespace ffmpeg {

namespace {
std::unique_ptr<Buffer> get_buffer(
    AVMediaType type,
    int frames_per_chunk,
    int num_chunks) {
  switch (type) {
    case AVMEDIA_TYPE_AUDIO:
      return std::unique_ptr<Buffer>(
          new AudioBuffer(frames_per_chunk, num_chunks));
    case AVMEDIA_TYPE_VIDEO:
      return std::unique_ptr<Buffer>(
          new VideoBuffer(frames_per_chunk, num_chunks));
    default:
      throw std::runtime_error(
          std::string("Unsupported media type: ") +
          av_get_media_type_string(type));
  }
}
} // namespace

Sink::Sink(
    AVRational input_time_base,
    AVCodecParameters* codecpar,
    int frames_per_chunk,
    int num_chunks,
    double output_time_base,
    std::string filter_description)
    : filter(input_time_base, codecpar, filter_description),
      buffer(get_buffer(codecpar->codec_type, frames_per_chunk, num_chunks)),
      time_base(output_time_base) {}

// 0: some kind of success
// <0: Some error happened
int Sink::process_frame(AVFrame* pFrame) {
  int ret = filter.add_frame(pFrame);
  while (ret >= 0) {
    ret = filter.get_frame(frame);
    //  AVERROR(EAGAIN) means that new input data is required to return new
    //  output.
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return 0;
    if (ret >= 0)
      buffer->push_frame(frame);
    av_frame_unref(frame);
  }
  return ret;
}

bool Sink::is_buffer_ready() const {
  return buffer->is_ready();
}
} // namespace ffmpeg
} // namespace torchaudio
