#include <torchaudio/csrc/ffmpeg/filter_graph.h>
#include <stdexcept>

namespace torchaudio {
namespace ffmpeg {

FilterGraph::FilterGraph(AVMediaType media_type) : media_type(media_type) {
  switch (media_type) {
    case AVMEDIA_TYPE_AUDIO:
    case AVMEDIA_TYPE_VIDEO:
      break;
    default:
      throw std::runtime_error("Only audio and video type is supported.");
  }
}

////////////////////////////////////////////////////////////////////////////////
// Configuration methods
////////////////////////////////////////////////////////////////////////////////
namespace {
std::string get_audio_src_args(
    AVSampleFormat format,
    AVRational time_base,
    int sample_rate,
    uint64_t channel_layout) {
  char args[512];
  std::snprintf(
      args,
      sizeof(args),
      "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=0x%" PRIx64,
      time_base.num,
      time_base.den,
      sample_rate,
      av_get_sample_fmt_name(format),
      channel_layout);
  return std::string(args);
}

std::string get_video_src_args(
    AVPixelFormat format,
    AVRational time_base,
    int width,
    int height,
    AVRational sample_aspect_ratio) {
  char args[512];
  std::snprintf(
      args,
      sizeof(args),
      "video_size=%dx%d:pix_fmt=%s:time_base=%d/%d:pixel_aspect=%d/%d",
      width,
      height,
      av_get_pix_fmt_name(format),
      time_base.num,
      time_base.den,
      sample_aspect_ratio.num,
      sample_aspect_ratio.den);
  return std::string(args);
}

} // namespace

void FilterGraph::add_audio_src(
    AVSampleFormat format,
    AVRational time_base,
    int sample_rate,
    uint64_t channel_layout) {
  TORCH_CHECK(
      media_type == AVMEDIA_TYPE_AUDIO, "The filter graph is not audio type.");
  std::string args =
      get_audio_src_args(format, time_base, sample_rate, channel_layout);
  add_src(args);
}

void FilterGraph::add_video_src(
    AVPixelFormat format,
    AVRational time_base,
    int width,
    int height,
    AVRational sample_aspect_ratio) {
  TORCH_CHECK(
      media_type == AVMEDIA_TYPE_VIDEO, "The filter graph is not video type.");
  std::string args =
      get_video_src_args(format, time_base, width, height, sample_aspect_ratio);
  add_src(args);
}

void FilterGraph::add_src(const std::string& args) {
  const AVFilter* buffersrc = avfilter_get_by_name(
      media_type == AVMEDIA_TYPE_AUDIO ? "abuffer" : "buffer");
  int ret = avfilter_graph_create_filter(
      &buffersrc_ctx, buffersrc, "in", args.c_str(), NULL, pFilterGraph);
  if (ret < 0) {
    throw std::runtime_error(
        "Failed to create input filter: \"" + args + "\" (" +
        av_err2string(ret) + ")");
  }
}

void FilterGraph::add_sink() {
  if (buffersink_ctx) {
    throw std::runtime_error("Sink buffer is already allocated.");
  }
  const AVFilter* buffersink = avfilter_get_by_name(
      media_type == AVMEDIA_TYPE_AUDIO ? "abuffersink" : "buffersink");
  // Note
  // Originally, the code here followed the example
  // https://ffmpeg.org/doxygen/4.1/filtering_audio_8c-example.html
  // which sets option for `abuffersink`, which caused an issue where the
  // `abuffersink` parameters set for the first time survive across multiple
  // fitler generations.
  // According to the other example
  // https://ffmpeg.org/doxygen/4.1/filter_audio_8c-example.html
  // `abuffersink` should not take options, and this resolved issue.
  int ret = avfilter_graph_create_filter(
      &buffersink_ctx, buffersink, "out", nullptr, nullptr, pFilterGraph);
  if (ret < 0) {
    throw std::runtime_error("Failed to create output filter.");
  }
}

namespace {

// Encapsulating AVFilterInOut* with handy methods since
// we need to deal with multiple of them at the same time.
class InOuts {
  AVFilterInOut* p = nullptr;
  // Disable copy constructor/assignment just in case.
  InOuts(const InOuts&) = delete;
  InOuts& operator=(const InOuts&) = delete;

 public:
  InOuts(const char* name, AVFilterContext* pCtx) {
    p = avfilter_inout_alloc();
    if (!p) {
      throw std::runtime_error("Failed to allocate AVFilterInOut.");
    }
    p->name = av_strdup(name);
    p->filter_ctx = pCtx;
    p->pad_idx = 0;
    p->next = nullptr;
  }
  ~InOuts() {
    avfilter_inout_free(&p);
  }
  operator AVFilterInOut**() {
    return &p;
  }
};

} // namespace

void FilterGraph::add_process(const std::string& filter_description) {
  // Note
  // The official example and other derived codes out there use
  // https://ffmpeg.org/doxygen/4.1/filtering_audio_8c-example.html#_a37
  // variable name `in` for "out"/buffersink, and `out` for "in"/buffersrc.
  // If you are debugging this part of the code, you might get confused.
  InOuts in{"in", buffersrc_ctx}, out{"out", buffersink_ctx};

  int ret = avfilter_graph_parse_ptr(
      pFilterGraph, filter_description.c_str(), out, in, nullptr);

  if (ret < 0) {
    throw std::runtime_error(
        "Failed to create the filter from \"" + filter_description + "\" (" +
        av_err2string(ret) + ".)");
  }
}

void FilterGraph::create_filter() {
  int ret = avfilter_graph_config(pFilterGraph, nullptr);
  if (ret < 0) {
    throw std::runtime_error(
        "Failed to configure the graph: " + av_err2string(ret));
  }
  // char* desc = avfilter_graph_dump(pFilterGraph.get(), NULL);
  // std::cerr << "Filter created:\n" << desc << std::endl;
  // av_free(static_cast<void*>(desc));
}

////////////////////////////////////////////////////////////////////////////////
// Streaming process
//////////////////////////////////////////////////////////////////////////////
int FilterGraph::add_frame(AVFrame* pInputFrame) {
  return av_buffersrc_add_frame_flags(
      buffersrc_ctx, pInputFrame, AV_BUFFERSRC_FLAG_KEEP_REF);
}

int FilterGraph::get_frame(AVFrame* pOutputFrame) {
  return av_buffersink_get_frame(buffersink_ctx, pOutputFrame);
}

} // namespace ffmpeg
} // namespace torchaudio
