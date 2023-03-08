#pragma once
#include <torch/types.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/filter_graph.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/encoder.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/tensor_converter.h>

namespace torchaudio::io {

class EncodeProcess {
  // In the reverse order of the process
  AVCodecContextPtr codec_ctx;
  Encoder encoder;
  AVFramePtr dst_frame{};
  FilterGraph filter;
  AVFramePtr src_frame;
  TensorConverter converter;

 public:
  // Constructor for audio
  EncodeProcess(
      AVFormatContext* format_ctx,
      int sample_rate,
      int num_channels,
      const enum AVSampleFormat format,
      const c10::optional<std::string>& encoder,
      const c10::optional<OptionDict>& encoder_option,
      const c10::optional<std::string>& encoder_format);

  // constructor for video
  EncodeProcess(
      AVFormatContext* format_ctx,
      double frame_rate,
      int width,
      int height,
      const enum AVPixelFormat format,
      const c10::optional<std::string>& encoder,
      const c10::optional<OptionDict>& encoder_option,
      const c10::optional<std::string>& encoder_format,
      const c10::optional<std::string>& hw_accel);

  void process(AVMediaType type, const torch::Tensor& tensor);

  void process_frame(AVFrame* src);

  void flush();
};

}; // namespace torchaudio::io
