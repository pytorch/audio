#include <torchaudio/csrc/ffmpeg/stream_reader_wrapper.h>

namespace torchaudio {
namespace ffmpeg {
namespace {

// TODO:
// merge the implementation with the one from stream_reader_binding.cpp
std::map<std::string, std::string> convert_map(
    const c10::Dict<std::string, std::string>& src) {
  std::map<std::string, std::string> ret;
  for (const auto& it : src) {
    ret.insert({it.key(), it.value()});
  }
  return ret;
}

SrcInfo convert(SrcStreamInfo ssi) {
  return SrcInfo(std::forward_as_tuple(
      av_get_media_type_string(ssi.media_type),
      ssi.codec_name,
      ssi.codec_long_name,
      ssi.fmt_name,
      ssi.bit_rate,
      ssi.num_frames,
      ssi.bits_per_sample,
      ssi.metadata,
      ssi.sample_rate,
      ssi.num_channels,
      ssi.width,
      ssi.height,
      ssi.frame_rate));
}

SrcInfoPyBind convert_pybind(SrcStreamInfo ssi) {
  return SrcInfoPyBind(std::forward_as_tuple(
      av_get_media_type_string(ssi.media_type),
      ssi.codec_name,
      ssi.codec_long_name,
      ssi.fmt_name,
      ssi.bit_rate,
      ssi.num_frames,
      ssi.bits_per_sample,
      convert_map(ssi.metadata),
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

AVFormatInputContextPtr get_input_format_context(
    const std::string& src,
    const c10::optional<std::string>& device,
    const OptionDict& option,
    AVIOContext* io_ctx) {
  AVFormatContext* pFormat = avformat_alloc_context();
  if (!pFormat) {
    throw std::runtime_error("Failed to allocate AVFormatContext.");
  }
  if (io_ctx) {
    pFormat->pb = io_ctx;
  }

  auto* pInput = [&]() -> AVFORMAT_CONST AVInputFormat* {
    if (device.has_value()) {
      std::string device_str = device.value();
      AVFORMAT_CONST AVInputFormat* p =
          av_find_input_format(device_str.c_str());
      if (!p) {
        std::ostringstream msg;
        msg << "Unsupported device/format: \"" << device_str << "\"";
        throw std::runtime_error(msg.str());
      }
      return p;
    }
    return nullptr;
  }();

  AVDictionary* opt = get_option_dict(option);
  int ret = avformat_open_input(&pFormat, src.c_str(), pInput, &opt);
  clean_up_dict(opt);

  if (ret < 0)
    throw std::runtime_error(
        "Failed to open the input \"" + src + "\" (" + av_err2string(ret) +
        ").");
  return AVFormatInputContextPtr(pFormat);
}

StreamReaderBinding::StreamReaderBinding(AVFormatInputContextPtr&& p)
    : StreamReader(std::move(p)) {}

SrcInfo StreamReaderBinding::get_src_stream_info(int64_t i) {
  return convert(StreamReader::get_src_stream_info(static_cast<int>(i)));
}

SrcInfoPyBind StreamReaderBinding::get_src_stream_info_pybind(int64_t i) {
  return convert_pybind(StreamReader::get_src_stream_info(static_cast<int>(i)));
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
