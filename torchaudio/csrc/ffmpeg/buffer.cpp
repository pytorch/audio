#include <torchaudio/csrc/ffmpeg/buffer.h>
#include <stdexcept>
#include <vector>

namespace torchaudio {
namespace ffmpeg {

Buffer::Buffer(AVMediaType type) : media_type(type) {}

namespace {
torch::Tensor convert_audio_tensor(AVFrame* pFrame) {
  // ref: https://ffmpeg.org/doxygen/4.1/filter__audio_8c_source.html#l00215
  AVSampleFormat format = static_cast<AVSampleFormat>(pFrame->format);
  int num_channels = pFrame->channels;
  int bps = av_get_bytes_per_sample(format);

  // Note
  // FFMpeg's `nb_samples` represnts the number of samples par channel.
  // This corresponds to `num_frames` in torchaudio's notation.
  // Also torchaudio uses `num_samples` as the number of samples
  // across channels.
  int num_frames = pFrame->nb_samples;

  int is_planar = av_sample_fmt_is_planar(format);
  int num_planes = is_planar ? num_channels : 1;
  int plane_size = bps * num_frames * (is_planar ? 1 : num_channels);
  std::vector<int64_t> shape = is_planar
      ? std::vector<int64_t>{num_channels, num_frames}
      : std::vector<int64_t>{num_frames, num_channels};

  torch::Tensor t;
  uint8_t* ptr = NULL;
  switch (format) {
    case AV_SAMPLE_FMT_U8:
    case AV_SAMPLE_FMT_U8P: {
      t = torch::empty(shape, torch::kUInt8);
      ptr = t.data_ptr<uint8_t>();
      break;
    }
    case AV_SAMPLE_FMT_S16:
    case AV_SAMPLE_FMT_S16P: {
      t = torch::empty(shape, torch::kInt16);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<int16_t>());
      break;
    }
    case AV_SAMPLE_FMT_S32:
    case AV_SAMPLE_FMT_S32P: {
      t = torch::empty(shape, torch::kInt32);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<int32_t>());
      break;
    }
    case AV_SAMPLE_FMT_S64:
    case AV_SAMPLE_FMT_S64P: {
      t = torch::empty(shape, torch::kInt64);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<int64_t>());
      break;
    }
    case AV_SAMPLE_FMT_FLT:
    case AV_SAMPLE_FMT_FLTP: {
      t = torch::empty(shape, torch::kFloat32);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<float>());
      break;
    }
    case AV_SAMPLE_FMT_DBL:
    case AV_SAMPLE_FMT_DBLP: {
      t = torch::empty(shape, torch::kFloat64);
      ptr = reinterpret_cast<uint8_t*>(t.data_ptr<double>());
      break;
    }
    default:
      throw std::runtime_error(
          "Unsupported audio format: " +
          std::string(av_get_sample_fmt_name(format)));
  }
  for (int i = 0; i < num_planes; ++i) {
    memcpy(ptr, pFrame->extended_data[i], plane_size);
    ptr += plane_size;
  }
  if (is_planar)
    t = t.t();
  return t;
}
} // namespace

void Buffer::push_audio_frame(AVFrame* pFrame) {
  chunks.push_back(convert_audio_tensor(pFrame));
}

namespace {
torch::Tensor convert_image_tensor(AVFrame* pFrame) {
  // ref:
  // https://ffmpeg.org/doxygen/4.1/filtering__video_8c_source.html#l00179
  // https://ffmpeg.org/doxygen/4.1/decode__video_8c_source.html#l00038
  AVPixelFormat format = static_cast<AVPixelFormat>(pFrame->format);
  int width = pFrame->width;
  int height = pFrame->height;
  uint8_t* buf = pFrame->data[0];
  int linesize = pFrame->linesize[0];

  int channel;
  switch (format) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
      channel = 3;
      break;
    case AV_PIX_FMT_ARGB:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_ABGR:
    case AV_PIX_FMT_BGRA:
      channel = 4;
      break;
    case AV_PIX_FMT_GRAY8:
      channel = 1;
      break;
    default:
      throw std::runtime_error(
          "Unexpected format: " + std::string(av_get_pix_fmt_name(format)));
  }

  torch::Tensor t;
  t = torch::empty({1, height, width, channel}, torch::kUInt8);
  auto ptr = t.data_ptr<uint8_t>();
  int stride = width * channel;
  for (int i = 0; i < height; ++i) {
    memcpy(ptr, buf, stride);
    buf += linesize;
    ptr += stride;
  }
  return t.permute({0, 3, 1, 2});
}
} // namespace

void Buffer::push_video_frame(AVFrame* pFrame) {
  chunks.push_back(convert_image_tensor(pFrame));
}

torch::Tensor Buffer::pop_all() {
  if (!chunks.size())
    return torch::empty({});

  std::vector<torch::Tensor> tmp;
  while (chunks.size()) {
    tmp.push_back(chunks.front());
    chunks.pop_front();
  }
  return torch::cat(tmp, 0);
}

void Buffer::push_frame(AVFrame* frame) {
  switch (media_type) {
    case AVMEDIA_TYPE_AUDIO:
      push_audio_frame(frame);
      break;
    case AVMEDIA_TYPE_VIDEO:
      push_video_frame(frame);
      break;
    default:
      throw std::runtime_error(
          "Unexpected media type. Only audio/video is supported.");
  }
}
} // namespace ffmpeg
} // namespace torchaudio
