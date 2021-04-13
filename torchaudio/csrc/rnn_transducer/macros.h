#pragma once

#define HOST_AND_DEVICE
#define FORCE_INLINE inline

#include <cstring>
#include <iostream>

typedef enum { INFO = 0, WARNING = 1, ERROR = 2, FATAL = 3 } level_t;

const char* ToString(level_t level);
