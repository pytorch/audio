#pragma once
#include <libtorio/ffmpeg/ffmpeg.h>

namespace torio {
namespace io {
class PacketBuffer {
 public:
  void push_packet(AVPacket* packet);
  std::vector<AVPacketPtr> pop_packets();
  bool has_packets();

 private:
  std::deque<AVPacketPtr> packets;
};
} // namespace io
} // namespace torio
