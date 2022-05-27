#include <torchaudio/csrc/ffmpeg/stream_reader_wrapper.h>

namespace torchaudio {
namespace ffmpeg {
namespace {

SrcInfo convert(SrcStreamInfo ssi) {
  return SrcInfo(std::forward_as_tuple(
      av_get_media_type_string(ssi.media_type),
      ssi.codec_name,
      ssi.codec_long_name,
      ssi.fmt_name,
      ssi.bit_rate,
      ssi.sample_rate,
      ssi.num_channels,
      ssi.width,
      ssi.height,
      ssi.frame_rate));
}

OutInfo convert(OutputStreamInfo osi) {
  return OutInfo(
      std::forward_as_tuple(osi.source_index, osi.filter_description));
}
} // namespace

StreamReaderBinding::StreamReaderBinding(AVFormatContextPtr&& p)
    : StreamReader(std::move(p)) {}

SrcInfo StreamReaderBinding::get_src_stream_info(int64_t i) {
  return convert(StreamReader::get_src_stream_info(static_cast<int>(i)));
}

OutInfo StreamReaderBinding::get_out_stream_info(int64_t i) {
  return convert(StreamReader::get_out_stream_info(static_cast<int>(i)));
}

int64_t StreamReaderBinding::process_packet(
    const c10::optional<double>& timeout,
    const double backoff) {
  int64_t code = [&]() {
    if (timeout.has_value()) {
      return StreamReader::process_packet_block(timeout.value(), backoff);
    }
    return StreamReader::process_packet();
  }();
  if (code < 0) {
    throw std::runtime_error(
        "Failed to process a packet. (" + av_err2string(code) + "). ");
  }
  return code;
}

void StreamReaderBinding::process_all_packets() {
  int64_t ret = 0;
  do {
    ret = process_packet();
  } while (!ret);
}

} // namespace ffmpeg
} // namespace torchaudio
