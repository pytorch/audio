set(TORCHAUDIO_THIRD_PARTIES "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

################################################################################
# sox
################################################################################
add_library(libsox INTERFACE)
if (BUILD_SOX)
  add_subdirectory(sox)
  target_include_directories(libsox INTERFACE ${SOX_INCLUDE_DIR})
  target_link_libraries(libsox INTERFACE ${SOX_LIBRARIES})
  list(APPEND TORCHAUDIO_THIRD_PARTIES libsox)
endif()

################################################################################
# ffmpeg
################################################################################
if (BUILD_FFMPEG)
  if(WIN32)
    message(FATAL_ERROR "FFmpeg integration is not supported on Windows.")
    # TODO
    # Since pkg-config is not well-supported on Windows (for example, ffmpeg
    # installed from conda-forge does not include `.pc` medata file),
    # to support windows, we need to fill interface manually.
    #
    # Something like this might work
    # https://github.com/microsoft/vcpkg/issues/1379#issuecomment-312483740
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    # requires ffmpeg>=4.1
    libavdevice>=58
    libavfilter>=7
    libavformat>=58
    libavcodec>=58
    libswresample>=3
    libswscale>=3
    libavutil>=56
    )
  add_library(ffmpeg INTERFACE)
  target_include_directories(ffmpeg INTERFACE ${LIBAV_INCLUDE_DIRS})
  target_link_libraries(ffmpeg INTERFACE ${LIBAV_LINK_LIBRARIES})
endif()

################################################################################
# kaldi
################################################################################
if (BUILD_KALDI)
  add_subdirectory(kaldi)
  list(APPEND TORCHAUDIO_THIRD_PARTIES kaldi)
endif()

set_property(GLOBAL PROPERTY TORCHAUDIO_THIRD_PARTIES "${TORCHAUDIO_THIRD_PARTIES}")

################################################################################
# KenLM
################################################################################
if (BUILD_CTC_DECODER)
  add_subdirectory(zlib)
  add_subdirectory(bzip2)
  add_subdirectory(lzma)
  add_subdirectory(boost)
  add_subdirectory(kenlm)
endif()
