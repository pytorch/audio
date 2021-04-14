#include <torchaudio/csrc/transducer/types.h>

namespace torchaudio {
namespace transducer {

const char* toString(status_t status) {
  switch (status) {
    case SUCCESS:
      return "success";
    case FAILURE:
      return "failure";
    default:
      return "unknown";
  }
}

const char* toString(device_t device) {
  switch (device) {
    case UNDEFINED:
      return "undefined";
    case CPU:
      return "cpu";
    default:
      return "unknown";
  }
}

} // namespace transducer
} // namespace torchaudio
