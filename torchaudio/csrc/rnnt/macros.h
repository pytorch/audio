#pragma once

#ifdef USE_CUDA
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define REDUCE_THREADS 256
#define HOST_AND_DEVICE __host__ __device__
#define FORCE_INLINE __forceinline__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#elif USE_ROCM
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024
#define REDUCE_THREADS 256
#define HOST_AND_DEVICE __host__ __device__
#define FORCE_INLINE __forceinline__
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#define HOST_AND_DEVICE
#define FORCE_INLINE inline
#endif // USE_CUDA

#include <cstring>
#include <iostream>

typedef enum { INFO = 0, WARNING = 1, ERROR = 2, FATAL = 3 } level_t;

const char* ToString(level_t level);
