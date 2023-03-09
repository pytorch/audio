message(STATUS "FFMPEG_ROOT=$ENV{FFMPEG_ROOT}")
find_package(FFMPEG 4.1 REQUIRED COMPONENTS avdevice avfilter avformat avcodec avutil)
add_library(ffmpeg INTERFACE)
target_include_directories(ffmpeg INTERFACE "${FFMPEG_INCLUDE_DIRS}")
target_link_libraries(ffmpeg INTERFACE "${FFMPEG_LIBRARIES}")


set(
  sources
  ffmpeg.cpp
  filter_graph.cpp
  stream_reader/buffer/common.cpp
  stream_reader/buffer/chunked_buffer.cpp
  stream_reader/buffer/unchunked_buffer.cpp
  stream_reader/sink.cpp
  stream_reader/stream_processor.cpp
  stream_reader/stream_reader.cpp
  stream_writer/encode_process.cpp
  stream_writer/encoder.cpp
  stream_writer/stream_writer.cpp
  stream_writer/tensor_converter.cpp
  compat.cpp
  utils.cpp
  )

if (USE_CUDA)
  set(
    additional_lib
    cuda_deps)
endif()

torchaudio_library(
  libtorchaudio_ffmpeg
  "${sources}"
  ""
  "torch;ffmpeg;${additional_lib}"
  ""
  )

if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  set(
    ext_sources
    pybind/fileobj.cpp
    pybind/pybind.cpp
    )
  torchaudio_extension(
    _torchaudio_ffmpeg
    "${ext_sources}"
    ""
    "libtorchaudio_ffmpeg"
    ""
    )
endif ()
