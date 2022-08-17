# the following line is added in order to export symbols when building on Windows
# this approach has some limitations as documented in https://github.com/pytorch/pytorch/pull/3650
if (MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

################################################################################
# libtorchaudio
################################################################################
set(
  LIBTORCHAUDIO_SOURCES
  lfilter.cpp
  overdrive.cpp
  utils.cpp
  )

set(
  LIBTORCHAUDIO_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  )

set(
  LIBTORCHAUDIO_LINK_LIBRARIES
  torch
  )

set(
  LIBTORCHAUDIO_COMPILE_DEFINITIONS)

#------------------------------------------------------------------------------#
# START OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#
if(BUILD_RNNT)
  list(
    APPEND
    LIBTORCHAUDIO_SOURCES
    rnnt/cpu/compute_alphas.cpp
    rnnt/cpu/compute_betas.cpp
    rnnt/cpu/compute.cpp
    rnnt/compute_alphas.cpp
    rnnt/compute_betas.cpp
    rnnt/compute.cpp
    rnnt/autograd.cpp
    )
  if (USE_CUDA)
    list(
      APPEND
      LIBTORCHAUDIO_SOURCES
      rnnt/gpu/compute_alphas.cu
      rnnt/gpu/compute_betas.cu
      rnnt/gpu/compute.cu
      )
  endif()
endif()

if(USE_CUDA)
  list(
    APPEND
    LIBTORCHAUDIO_INCLUDE_DIRS
    ${CUDA_TOOLKIT_INCLUDE}
    )
  list(
    APPEND
    LIBTORCHAUDIO_LINK_LIBRARIES
    ${C10_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    )
  list(
    APPEND
    LIBTORCHAUDIO_COMPILE_DEFINITIONS
    USE_CUDA
  )
endif()

if(BUILD_KALDI)
  list(APPEND LIBTORCHAUDIO_LINK_LIBRARIES kaldi)
  list(APPEND LIBTORCHAUDIO_SOURCES kaldi.cpp)
  list(APPEND LIBTORCHAUDIO_COMPILE_DEFINITIONS INCLUDE_KALDI)
endif()

if(BUILD_SOX)
  list(
    APPEND
    LIBTORCHAUDIO_LINK_LIBRARIES
    libsox
    )
  list(
    APPEND
    LIBTORCHAUDIO_SOURCES
    sox/io.cpp
    sox/utils.cpp
    sox/effects.cpp
    sox/effects_chain.cpp
    sox/types.cpp
    )
  list(
    APPEND
    LIBTORCHAUDIO_COMPILE_DEFINITIONS
    INCLUDE_SOX
    )
endif()

if(OpenMP_CXX_FOUND)
  list(
    APPEND
    LIBTORCHAUDIO_LINK_LIBRARIES
    OpenMP::OpenMP_CXX
    )
endif()

if(USE_FFMPEG)
  list(
    APPEND
    LIBTORCHAUDIO_COMPILE_DEFINITIONS
    USE_FFMPEG
    )
endif()

#------------------------------------------------------------------------------#
# END OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#

torchaudio_library(
  libtorchaudio
  "${LIBTORCHAUDIO_SOURCES}"
  "${LIBTORCHAUDIO_INCLUDE_DIRS}"
  "${LIBTORCHAUDIO_LINK_LIBRARIES}"
  "${LIBTORCHAUDIO_COMPILE_DEFINITIONS}"
  )

if (APPLE)
  set(TORCHAUDIO_LIBRARY libtorchaudio CACHE INTERNAL "")
else()
  set(TORCHAUDIO_LIBRARY -Wl,--no-as-needed libtorchaudio -Wl,--as-needed CACHE INTERNAL "")
endif()

################################################################################
# libtorchaudio_ffmpeg
################################################################################
if(USE_FFMPEG)
  set(
    LIBTORCHAUDIO_FFMPEG_SOURCES
    ffmpeg/ffmpeg.cpp
    ffmpeg/filter_graph.cpp
    ffmpeg/stream_reader/buffer.cpp
    ffmpeg/stream_reader/decoder.cpp
    ffmpeg/stream_reader/sink.cpp
    ffmpeg/stream_reader/stream_processor.cpp
    ffmpeg/stream_reader/stream_reader.cpp
    ffmpeg/stream_reader/stream_reader_wrapper.cpp
    ffmpeg/stream_reader/stream_reader_binding.cpp
    ffmpeg/stream_writer/stream_writer.cpp
    ffmpeg/stream_writer/stream_writer_wrapper.cpp
    ffmpeg/stream_writer/stream_writer_binding.cpp
    ffmpeg/utils.cpp
    )
  message(STATUS "FFMPEG_ROOT=$ENV{FFMPEG_ROOT}")
  find_package(FFMPEG 4.1 REQUIRED COMPONENTS avdevice avfilter avformat avcodec avutil)
  torchaudio_library(
    libtorchaudio_ffmpeg
    "${LIBTORCHAUDIO_FFMPEG_SOURCES}"
    "${LIBTORCHAUDIO_INCLUDE_DIRS};${FFMPEG_INCLUDE_DIRS}"
    "torch;${FFMPEG_LIBRARIES}"
    "${LIBTORCHAUDIO_COMPILE_DEFINITIONS}"
  )
endif()

################################################################################
# TODO: Rename this to _torchaudio_sox.so
# _torchaudio.so
################################################################################
if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  set(
    EXTENSION_SOURCES
    sox/pybind/pybind.cpp
    )
  #----------------------------------------------------------------------------#
  # START OF CUSTOMIZATION LOGICS
  #----------------------------------------------------------------------------#
  if(BUILD_SOX)
    list(
      APPEND
      EXTENSION_SOURCES
      sox/pybind/effects.cpp
      sox/pybind/effects_chain.cpp
      sox/pybind/io.cpp
      sox/pybind/utils.cpp
      )
  endif()
  #----------------------------------------------------------------------------#
  # END OF CUSTOMIZATION LOGICS
  #----------------------------------------------------------------------------#
  torchaudio_extension(
    _torchaudio
    "${EXTENSION_SOURCES}"
    ""
    libtorchaudio
    "${LIBTORCHAUDIO_COMPILE_DEFINITIONS}"
    )
  if(USE_FFMPEG)
    set(
      FFMPEG_EXTENSION_SOURCES
      ffmpeg/pybind/typedefs.cpp
      ffmpeg/pybind/pybind.cpp
      ffmpeg/pybind/stream_reader.cpp
      ffmpeg/pybind/stream_writer.cpp
      )
    torchaudio_extension(
      _torchaudio_ffmpeg
      "${FFMPEG_EXTENSION_SOURCES}"
      "${FFMPEG_INCLUDE_DIRS}"
      "libtorchaudio_ffmpeg"
      ""
      )
  endif()
endif()
