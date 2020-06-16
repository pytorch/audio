#include <torch/script.h>
#include <torchaudio/csrc/typedefs.h>

namespace torchaudio {
namespace sox_io {

c10::intrusive_ptr<::torchaudio::SignalInfo> get_info(
    const std::string& file_name);

} // namespace sox_io
} // namespace torchaudio
