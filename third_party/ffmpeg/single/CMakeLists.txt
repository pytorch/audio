# CMake file for searching existing FFmpeg installation and defining ffmpeg TARGET

message(STATUS "Searching existing FFmpeg installation")
message(STATUS FFMPEG_ROOT=$ENV{FFMPEG_ROOT})
if (NOT DEFINED ENV{FFMPEG_ROOT})
  message(STATUS "FFMPEG_ROOT not set, will search system paths")
endif()

# Maintain FFMPEG_ROOT approach and add common system paths
set(_root $ENV{FFMPEG_ROOT})
set(lib_dirs "${_root}/lib" "${_root}/bin" "/usr/lib64" "/usr/lib" "/usr/local/lib64" "/usr/local/lib")
set(include_dir "${_root}/include" "/usr/include/ffmpeg" "/usr/include" "/usr/local/include")

# Add multiarch paths for Debian/Ubuntu
execute_process(COMMAND gcc -print-multiarch OUTPUT_VARIABLE MULTIARCH_TRIPLET
               OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(MULTIARCH_TRIPLET)
  list(APPEND lib_dirs "/usr/lib/${MULTIARCH_TRIPLET}")
  list(APPEND include_dir "/usr/include/${MULTIARCH_TRIPLET}")
endif()

add_library(ffmpeg INTERFACE)
target_include_directories(ffmpeg INTERFACE "${include_dir}")

function (_find_ffmpeg_lib component)
  find_path("${component}_header"
    NAMES "lib${component}/${component}.h"
    PATHS "${include_dir}"
    DOC "The include directory for ${component}"
    REQUIRED)
  find_library("lib${component}"
    NAMES "${component}"
    PATHS ${lib_dirs}
    DOC "${component} library"
    REQUIRED)
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
