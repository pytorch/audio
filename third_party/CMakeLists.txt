set(TORCHAUDIO_THIRD_PARTIES "")

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
# transducer
################################################################################
if (BUILD_TRANSDUCER)
  add_subdirectory(transducer)
  list(APPEND TORCHAUDIO_THIRD_PARTIES warprnnt)
endif()

set_property(GLOBAL PROPERTY TORCHAUDIO_THIRD_PARTIES "${TORCHAUDIO_THIRD_PARTIES}")
