set(CMAKE_CXX_VISIBILITY_PRESET hidden)

file(MAKE_DIRECTORY install/include)
file(MAKE_DIRECTORY install/lib)

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
# Decoder
################################################################################
if (BUILD_CTC_DECODER)
  add_subdirectory(zlib)
  add_subdirectory(bzip2)
  if (NOT MSVC)
    add_subdirectory(lzma)
  endif()
  add_subdirectory(kenlm)
  add_subdirectory(flashlight-text)
endif()

if (BUILD_CUDA_CTC_DECODER)
  add_subdirectory(cuctc)
endif()