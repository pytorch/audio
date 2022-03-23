set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

################################################################################
# sox
################################################################################
add_library(libsox INTERFACE)
if (BUILD_SOX)
  add_subdirectory(sox)
  target_include_directories(libsox INTERFACE ${SOX_INCLUDE_DIR})
  target_link_libraries(libsox INTERFACE ${SOX_LIBRARIES})
endif()

################################################################################
# kaldi
################################################################################
if (BUILD_KALDI)
  add_subdirectory(kaldi)
endif()

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
