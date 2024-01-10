#pragma once
#include <libtorio/ffmpeg/ffmpeg.h>
#include <libtorio/ffmpeg/filter_graph.h>
#include <libtorio/ffmpeg/stream_writer/encoder.h>
#include <libtorio/ffmpeg/stream_writer/tensor_converter.h>
#include <libtorio/ffmpeg/stream_writer/types.h>
#include <torch/types.h>

namespace torio::io {

class EncodeProcess {
  TensorConverter converter;
  AVFramePtr src_frame;
  FilterGraph filter;
  AVFramePtr dst_frame{alloc_avframe()};
  Encoder encoder;
  AVCodecContextPtr codec_ctx;

 public:
  EncodeProcess(
      TensorConverter&& converter,
      AVFramePtr&& frame,
      FilterGraph&& filter_graph,
      Encoder&& encoder,
      AVCodecContextPtr&& codec_ctx) noexcept;

  EncodeProcess(EncodeProcess&&) noexcept = default;

  void process(const torch::Tensor& tensor, const c10::optional<double>& pts);

  void process_frame(AVFrame* src);

  void flush();
};

EncodeProcess get_audio_encode_process(
    AVFormatContext* format_ctx,
    int sample_rate,
    int num_channels,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<int>& encoder_sample_rate,
    const c10::optional<int>& encoder_num_channels,
    const c10::optional<CodecConfig>& codec_config,
    const c10::optional<std::string>& filter_desc,
    bool disable_converter = false);

EncodeProcess get_video_encode_process(
    AVFormatContext* format_ctx,
    double frame_rate,
    int width,
    int height,
    const std::string& format,
    const c10::optional<std::string>& encoder,
    const c10::optional<OptionDict>& encoder_option,
    const c10::optional<std::string>& encoder_format,
    const c10::optional<double>& encoder_frame_rate,
    const c10::optional<int>& encoder_width,
    const c10::optional<int>& encoder_height,
    const c10::optional<std::string>& hw_accel,
    const c10::optional<CodecConfig>& codec_config,
    const c10::optional<std::string>& filter_desc,
    bool disable_converter = false);

}; // namespace torio::io
