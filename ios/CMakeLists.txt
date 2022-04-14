cmake_minimum_required(VERSION 3.4.1)
set(TARGET torchaudio)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 14)
set(LIBTORCH_HEADER_ROOT ${LIBTORCH_HEADER_ROOT})
set(LIBRARY_OUTPUT_PATH ../lib)

set(AUDIO_SRCS
    ../torchaudio/csrc/lfilter.cpp
    ../torchaudio/csrc/overdrive.cpp
    ../torchaudio/csrc/utils.cpp)

add_library(${TARGET} STATIC
  ${AUDIO_SRCS}
)

set(PYTORCH_HEADERS "${LIBTORCH_HEADER_ROOT}")
set(PYTORCH_HEADERS_CSRC "${LIBTORCH_HEADER_ROOT}/torch/csrc/api/include")
target_include_directories(${TARGET} PRIVATE
  ${PYTORCH_HEADERS}
  ${PYTORCH_HEADERS_CSRC}
)
