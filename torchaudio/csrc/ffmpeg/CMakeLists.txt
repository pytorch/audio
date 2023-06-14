message(STATUS "FFMPEG_ROOT=$ENV{FFMPEG_ROOT}")
find_package(FFMPEG 4.1 REQUIRED COMPONENTS avdevice avfilter avformat avcodec avutil)
add_library(ffmpeg INTERFACE)
target_include_directories(ffmpeg INTERFACE "${FFMPEG_INCLUDE_DIRS}")
if (NOT DLOPEN_FFMPEG)
target_link_libraries(ffmpeg INTERFACE "${FFMPEG_LIBRARIES}")
endif()

set(
  sources
  stub.cpp
  ffmpeg.cpp
  filter_graph.cpp
  hw_context.cpp
  stream_reader/buffer/chunked_buffer.cpp
  stream_reader/buffer/unchunked_buffer.cpp
  stream_reader/conversion.cpp
  stream_reader/packet_buffer.cpp
  stream_reader/post_process.cpp
  stream_reader/stream_processor.cpp
  stream_reader/stream_reader.cpp
  stream_writer/encode_process.cpp
  stream_writer/encoder.cpp
  stream_writer/packet_writer.cpp
  stream_writer/stream_writer.cpp
  stream_writer/tensor_converter.cpp
  compat.cpp
  )

if (USE_CUDA)
  set(
    additional_lib
    cuda_deps)
endif()

if (DLOPEN_FFMPEG)
  set(compile_definitions DLOPEN_FFMPEG)
endif()

torchaudio_library(
  libtorchaudio_ffmpeg
  "${sources}"
  ""
  "torch;ffmpeg;${additional_lib}"
  "${compile_definitions}"
  )

if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  torchaudio_extension(
    _torchaudio_ffmpeg
    pybind/pybind.cpp
    ""
    "libtorchaudio_ffmpeg"
    "${compile_definitions}"
    )
endif ()
