################################################################################
# This file defines the following FFmpeg libraries using pre-built binaries.

add_library(ffmpeg4 INTERFACE)
add_library(ffmpeg ALIAS ffmpeg4)

################################################################################

include(FetchContent)

set(base_url https://pytorch.s3.amazonaws.com/torchaudio/ffmpeg)

if (APPLE)
  if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    FetchContent_Declare(
      f4
      URL ${base_url}/2023-07-06/macos_arm64/4.1.8.tar.gz
      URL_HASH SHA256=a44b8152b7f204ce5050fc7f6fd2bbbafe7ae4e45f03e135f3b45dd9a08f404e
      )
  elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    FetchContent_Declare(
      f4
      URL ${base_url}/2023-07-06/macos_x86_64/4.1.8.tar.gz
      URL_HASH SHA256=392d5af0b24535bfc69d6244e7595e5f07117b93d94505d0a4b34c82ae479f48
      )
  else ()
    message(
      FATAL_ERROR
      "CPU architecture ${CMAKE_SYSTEM_PROCESSOR} is not currently supported. If you do not need FFmpeg integration, then setting USE_FFMPEG=0 will bypass the issue.")
  endif()
elseif (UNIX)
  if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    FetchContent_Declare(
      f4
      URL ${base_url}/2023-07-06/linux_aarch64/4.1.8.tar.gz
      URL_HASH SHA256=aae0b713040e30ceebe0d0bc82353d3d9054055c7af8a4f4abc1766015ab7681
    )
  elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    FetchContent_Declare(
      f4
      URL ${base_url}/2023-07-06/linux_x86_64/4.1.8.tar.gz
      URL_HASH SHA256=52e53b8857739bdd54f9d8541e22569b57f6c6f16504ee83963c2ed3e7061a23
      )
  else ()
    # Possible case ppc64le (though it's not officially supported.)
    message(
      FATAL_ERROR
      "CPU architecture ${CMAKE_SYSTEM_PROCESSOR} is not currently supported. If you do not need FFmpeg integration, then setting USE_FFMPEG=0 will bypass the issue.")
  endif()
elseif(MSVC)
  FetchContent_Declare(
    f4
    URL ${base_url}/2023-07-06/windows/4.1.8.tar.gz
    URL_HASH SHA256=c45cd36e0575490f97ace07365bb67c5e1cbe9f3e6a4272d035c19348df96790
    )
endif()

FetchContent_MakeAvailable(f4)
target_include_directories(ffmpeg4 INTERFACE ${f4_SOURCE_DIR}/include)

if(APPLE)
  target_link_libraries(
    ffmpeg4
    INTERFACE
    ${f4_SOURCE_DIR}/lib/libavutil.56.dylib
    ${f4_SOURCE_DIR}/lib/libavcodec.58.dylib
    ${f4_SOURCE_DIR}/lib/libavformat.58.dylib
    ${f4_SOURCE_DIR}/lib/libavdevice.58.dylib
    ${f4_SOURCE_DIR}/lib/libavfilter.7.dylib
    )
elseif (UNIX)
  target_link_libraries(
    ffmpeg4
    INTERFACE
    ${f4_SOURCE_DIR}/lib/libavutil.so.56
    ${f4_SOURCE_DIR}/lib/libavcodec.so.58
    ${f4_SOURCE_DIR}/lib/libavformat.so.58
    ${f4_SOURCE_DIR}/lib/libavdevice.so.58
    ${f4_SOURCE_DIR}/lib/libavfilter.so.7
    )
elseif(MSVC)
  target_link_libraries(
    ffmpeg4
    INTERFACE
    ${f4_SOURCE_DIR}/bin/avutil.lib
    ${f4_SOURCE_DIR}/bin/avcodec.lib
    ${f4_SOURCE_DIR}/bin/avformat.lib
    ${f4_SOURCE_DIR}/bin/avdevice.lib
    ${f4_SOURCE_DIR}/bin/avfilter.lib
    )
endif()
