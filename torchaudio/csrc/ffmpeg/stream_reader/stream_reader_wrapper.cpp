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
