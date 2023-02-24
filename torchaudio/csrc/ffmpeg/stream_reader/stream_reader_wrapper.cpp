#include <torchaudio/csrc/ffmpeg/binding_utils.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_reader_wrapper.h>

namespace torchaudio {
namespace io {
namespace {

SrcInfo convert(SrcStreamInfo ssi) {
  return SrcInfo(std::forward_as_tuple(
      av_get_media_type_string(ssi.media_type),
      ssi.codec_name,
      ssi.codec_long_name,
      ssi.fmt_name,
      ssi.bit_rate,
      ssi.num_frames,
      ssi.bits_per_sample,
      to_c10(ssi.metadata),
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
  TORCH_CHECK(
      code >= 0, "Failed to process a packet. (" + av_err2string(code) + "). ");
  return code;
}

void StreamReaderBinding::process_all_packets() {
  int64_t ret = 0;
  do {
    ret = process_packet();
  } while (!ret);
}

int64_t StreamReaderBinding::fill_buffer(
    const c10::optional<double>& timeout,
    const double backoff) {
  while (!is_buffer_ready()) {
    int code = process_packet(timeout, backoff);
    if (code != 0) {
      return code;
    }
  }
  return 0;
}

std::vector<c10::optional<ChunkData>> StreamReaderBinding::pop_chunks() {
  std::vector<c10::optional<ChunkData>> ret;
  ret.reserve(static_cast<size_t>(num_out_streams()));
  for (auto& c : StreamReader::pop_chunks()) {
    if (c) {
      ret.emplace_back(std::forward_as_tuple(c->frames, c->pts));
    } else {
      ret.emplace_back();
    }
  }
  return ret;
}

} // namespace io
} // namespace torchaudio
