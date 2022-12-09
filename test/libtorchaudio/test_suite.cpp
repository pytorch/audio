#include <gtest/gtest.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer_wrapper.h>

using namespace ::testing;

TEST(test_suite, AVFormatContextTestMemoryLeak) {
  auto c = torchaudio::ffmpeg::get_output_format_context("foobar.mp4", {});
}
