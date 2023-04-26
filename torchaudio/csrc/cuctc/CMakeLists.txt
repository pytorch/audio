# Custom CMakeLists for building cuda ctc decoder

set(CMAKE_CXX_VISIBILITY_PRESET default)

# the following line is added in order to export symbols when building on Windows
# this approach has some limitations as documented in https://github.com/pytorch/pytorch/pull/3650
if (MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

set(
  libctc_prefix_decoder_src
  src/ctc_prefix_decoder.cpp
  src/ctc_prefix_decoder_kernel_v2.cu
  )

set(
  additional_libs
  )

list(
  APPEND
  additional_libs
  cuda_deps
  )

torchaudio_library(
  libctc_prefix_decoder
  "${libctc_prefix_decoder_src}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${additional_libs}"
  ""
  )

if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  torchaudio_extension(
    pybind11_prefixctc
    src/python_binding.cpp
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "libctc_prefix_decoder"
    ""
    )
endif()
