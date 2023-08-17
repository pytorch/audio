################################################################################
# libtorchaudio
################################################################################

if(USE_ROCM)
  list (APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
  FIND_PACKAGE(HIP REQUIRED)
  MESSAGE(STATUS "hip found ${ROCM_FOUND}")

  list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/third_party/hipify_torch/cmake")
  include(Hipify)

  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  list( APPEND CMAKE_INSTALL_RPATH "/opt/rocm/llvm/lib" )
  set(OPENMP_LIBRARIES "/opt/rocm/llvm/lib/")
  set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
  set(OpenMP_CXX_FLAGS "-fopenmp=libomp")
  #set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY omp)
  find_package(OpenMP REQUIRED)

endif()


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
    if ( NOT HIP_ADD_LIBRARY_FOUND )
      list(APPEND CMAKE_MODULE_PATH /opt/rocm/hip/cmake)
      find_package(HIP REQUIRED)
    endif()

    list(
      APPEND
      sources 
      rnnt/hip/compute_alphas.hip
      rnnt/hip/compute_betas.hip
      rnnt/hip/compute.hip
      )
  endif()
endif()

if(USE_ROCM)
  list(
    APPEND
    additional_libs 
    hip::host
    hip::device
    /opt/rocm/llvm/lib/libomp.so
    )
  list(
    APPEND
    compile_definitions 
    USE_ROCM
  )
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
