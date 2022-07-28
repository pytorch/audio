set(CMAKE_CXX_VISIBILITY_PRESET hidden)

################################################################################
# sox
################################################################################
if (BUILD_SOX)
  add_subdirectory(sox)
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
  add_subdirectory(kenlm)
  add_subdirectory(flashlight-text)
endif()
