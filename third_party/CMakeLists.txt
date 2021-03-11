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
# kaldi
################################################################################
if (BUILD_KALDI)
  add_subdirectory(kaldi)
  list(APPEND TORCHAUDIO_THIRD_PARTIES kaldi)
endif()

################################################################################
# transducer
################################################################################
if (BUILD_TRANSDUCER)
  add_subdirectory(transducer)
  list(APPEND TORCHAUDIO_THIRD_PARTIES warprnnt)
endif()

set_property(GLOBAL PROPERTY TORCHAUDIO_THIRD_PARTIES "${TORCHAUDIO_THIRD_PARTIES}")
