#pragma once

extern "C" {
#include <libavformat/version.h>
}

#ifndef LIBAVFORMAT_VERSION_MAJOR
#error LIBAVFORMAT_VERSION_MAJOR is not defined.
#endif

#if LIBAVFORMAT_VERSION_MAJOR >= 59
#define AVFORMAT_CONST const
#else
#define AVFORMAT_CONST
#endif
