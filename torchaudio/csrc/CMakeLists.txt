################################################################################
# libtorchaudio
################################################################################
set(
  sources
  lfilter.cpp
  overdrive.cpp
  utils.cpp
  )

set(
  additional_libs
  )

set(
  compile_definitions)

#------------------------------------------------------------------------------#
# START OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#
if(BUILD_RNNT)
  list(
    APPEND
    sources
    rnnt/cpu/compute_alphas.cpp
    rnnt/cpu/compute_betas.cpp
    rnnt/cpu/compute.cpp
    rnnt/compute_alphas.cpp
    rnnt/compute_betas.cpp
    rnnt/compute.cpp
    rnnt/autograd.cpp
    )
  if (USE_CUDA)
    list(
      APPEND
      sources
      rnnt/gpu/compute_alphas.cu
      rnnt/gpu/compute_betas.cu
      rnnt/gpu/compute.cu
      )
  endif()
endif()

if(BUILD_RIR)
  list(
    APPEND
    LIBTORCHAUDIO_SOURCES
    rir.cpp
    )
  list(
    APPEND
    LIBTORCHAUDIO_COMPILE_DEFINITIONS
    INCLUDE_RIR
    )
endif()

if(USE_CUDA)
  list(
    APPEND
    sources
    iir_cuda.cu
  )
  list(
    APPEND
    additional_libs
    cuda_deps
    )
endif()

if(BUILD_KALDI)
  list(APPEND additional_libs kaldi)
  list(APPEND sources kaldi.cpp)
  list(APPEND compile_definitions INCLUDE_KALDI)
endif()

if(OpenMP_CXX_FOUND)
  list(
    APPEND
    additional_libs
    OpenMP::OpenMP_CXX
    )
endif()

#------------------------------------------------------------------------------#
# END OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#

torchaudio_library(
  libtorchaudio
  "${sources}"
  ""
  "torch;${additional_libs}"
  "${compile_definitions}"
  )

if (APPLE)
  set(TORCHAUDIO_LIBRARY libtorchaudio CACHE INTERNAL "")
else()
  set(TORCHAUDIO_LIBRARY -Wl,--no-as-needed libtorchaudio -Wl,--as-needed CACHE INTERNAL "")
endif()

################################################################################
# Python extensions
################################################################################
if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  set(
    extension_sources
    pybind/pybind.cpp
    )
  torchaudio_extension(
    _torchaudio
    "${extension_sources}"
    ""
    "libtorchaudio"
    ""
    )
endif()
