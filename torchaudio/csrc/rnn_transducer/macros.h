#pragma once

#define HOST_AND_DEVICE
#define FORCE_INLINE inline

#include <cstring>
#include <iostream>

typedef enum { INFO = 0, WARNING = 1, ERROR = 2, FATAL = 3 } level_t;

const char* ToString(level_t level);

#define DCHECK(x)
#define DCHECK_EQ(x, y)
#define DCHECK_NE(x, y)
#define CHECK(x)
#define CHECK_EQ(x, y)
#define CHECK_NE(x, y)
