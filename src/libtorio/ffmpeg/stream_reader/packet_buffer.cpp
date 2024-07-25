#include <libtorio/ffmpeg/stream_reader/packet_buffer.h>

namespace torio::io {
void PacketBuffer::push_packet(AVPacket* packet) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(packet, "Packet is null.");
  AVPacket* p = av_packet_clone(packet);
  TORCH_INTERNAL_ASSERT(p, "Failed to clone packet.");
  packets.emplace_back(p);
}
std::vector<AVPacketPtr> PacketBuffer::pop_packets() {
  std::vector<AVPacketPtr> ret{
      std::make_move_iterator(packets.begin()),
      std::make_move_iterator(packets.end())};
  packets.clear();
  return ret;
}
bool PacketBuffer::has_packets() {
  return packets.size() > 0;
}
} // namespace torio::io
