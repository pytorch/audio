#include <torchaudio/csrc/rnnt/macros.h>

#ifdef USE_GLOG
#else

const char* ToString(level_t level) {
  switch (level) {
    case INFO:
      return "INFO";
    case WARNING:
      return "WARNING";
    case ERROR:
      return "ERROR";
    case FATAL:
      return "FATAL";
    default:
      return "UNKNOWN";
  }
}

#endif // USE_GLOG
