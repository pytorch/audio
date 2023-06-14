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
  if (USE_ROCM)
    hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/torchaudio/csrc/rnnt/gpu HIP_SOURCE_DIR ${PROJECT_SOURCE_DIR}/torchaudio/csrc/rnnt/hip)
    list(
      APPEND
      sources
      rnnt/hip/compute_alphas.hip
      rnnt/hip/compute_betas.hip
      rnnt/hip/compute.hip
      )
  endif()
endif()

if(BUILD_RIR)
  list(APPEND sources rir.cpp)
  list(APPEND compile_definitions INCLUDE_RIR)
endif()

if(BUILD_ALIGN)
  list(
    APPEND
    sources
    forced_align/compute.cpp
    forced_align/cpu/compute.cpp
  )
  list(APPEND compile_definitions INCLUDE_ALIGN)
  if (USE_CUDA)
    list(
      APPEND
      sources
      forced_align/gpu/compute.cu
    )
  endif()
  if (USE_ROCM)
    hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/torchaudio/csrc/forced_align/gpu HIP_SOURCE_DIR ${PROJECT_SOURCE_DIR}/torchaudio/csrc/forced_align/hip)
    list(
      APPEND
      sources
      forced_align/hip/compute.hip
    )
  endif()
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

if(USE_ROCM)
  hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/torchaudio/csrc HIP_SOURCE_DIR ${PROJECT_SOURCE_DIR}/torchaudio/csrc)
  list(
    APPEND
    sources
    iir_hip.hip
  )
  list(
    APPEND
    additional_libs
    hip::host
    hip::device
    )
  list(
    APPEND
    compile_definitions
    USE_ROCM
  )
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
