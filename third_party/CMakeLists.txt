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
else()
  # If not building and linking libsox statically, then we expect that
  # sox library and header are found in search path
  target_link_libraries(libsox INTERFACE -lsox)
endif()
list(APPEND TORCHAUDIO_THIRD_PARTIES libsox)

################################################################################
# kaldi
################################################################################
if (BUILD_KALDI)
  add_subdirectory(kaldi)
  list(APPEND TORCHAUDIO_THIRD_PARTIES kaldi)

  add_library(kaldifeat_core STATIC
    kaldifeat/kaldifeat/csrc/feature-fbank.cc
    kaldifeat/kaldifeat/csrc/feature-window.cc
    kaldifeat/kaldifeat/csrc/mel-computations.cc
  )
  target_link_libraries(kaldifeat_core PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(kaldifeat_core PUBLIC kaldifeat)
  list(APPEND TORCHAUDIO_THIRD_PARTIES kaldifeat_core)
endif()

################################################################################
# transducer
################################################################################
if (BUILD_TRANSDUCER)
  add_subdirectory(transducer)
  list(APPEND TORCHAUDIO_THIRD_PARTIES warprnnt)
endif()

set_property(GLOBAL PROPERTY TORCHAUDIO_THIRD_PARTIES "${TORCHAUDIO_THIRD_PARTIES}")
