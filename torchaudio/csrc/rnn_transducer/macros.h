#pragma once

#define HOST_AND_DEVICE
#define FORCE_INLINE inline

#ifdef USE_GLOG
#include <glog/logging.h>
#else
#include <cstring>
#include <iostream>

typedef enum { INFO = 0, WARNING = 1, ERROR = 2, FATAL = 3 } level_t;

const char* ToString(level_t level);

struct LOG {
  LOG(const level_t& level) {
    ::std::cerr << "LOG(" << ToString(level) << "): ";
  }
  ~LOG() {
    ::std::cerr << ::std::endl;
  }
};

template <typename T>
LOG&& operator<<(LOG&& log, const T& object) {
  ::std::cerr << object;
  return ::std::move(log);
}

#define DCHECK(x)
#define DCHECK_EQ(x, y)
#define DCHECK_NE(x, y)
#define CHECK(x)
#define CHECK_EQ(x, y)
#define CHECK_NE(x, y)
#endif // USE_GLOG
