# CMake file for searching existing FFmpeg installation and defining ffmpeg TARGET

message(STATUS "Searching existing FFmpeg installation")
message(STATUS FFMPEG_ROOT=$ENV{FFMPEG_ROOT})
if (NOT DEFINED ENV{FFMPEG_ROOT})
  message(FATAL_ERROR "Environment variable FFMPEG_ROOT is not set.")
endif()

set(_root $ENV{FFMPEG_ROOT})
set(lib_dirs "${_root}/lib" "${_root}/bin")
set(include_dir "${_root}/include")

add_library(ffmpeg INTERFACE)
target_include_directories(ffmpeg INTERFACE "${include_dir}")

function (_find_ffmpeg_lib component)
  find_path("${component}_header"
    NAMES "lib${component}/${component}.h"
    PATHS "${include_dir}"
    DOC "The include directory for ${component}"
    REQUIRED
    NO_DEFAULT_PATH)
  find_library("lib${component}"
    NAMES "${component}"
    PATHS ${lib_dirs}
    DOC "${component} library"
    REQUIRED
    NO_DEFAULT_PATH)
  message(STATUS "Found ${component}: ${lib${component}}")
  target_link_libraries(
    ffmpeg
    INTERFACE
    ${lib${component}})
endfunction ()

_find_ffmpeg_lib(avutil)
_find_ffmpeg_lib(avcodec)
_find_ffmpeg_lib(avformat)
_find_ffmpeg_lib(avdevice)
_find_ffmpeg_lib(avfilter)
