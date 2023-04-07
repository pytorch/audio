#include <torchaudio/csrc/ffmpeg/stream_reader/packet_buffer.h>

namespace torchaudio {
namespace io {
void PacketBuffer::push_packet(AVPacket* packet) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(packet, "Packet is null.");
  AVPacketPtr pPacket;
  av_packet_ref(pPacket, packet);
  packets.push_back(std::move(pPacket));
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
} // namespace io
} // namespace torchaudio
