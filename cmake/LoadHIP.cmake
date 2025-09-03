##############################################################################
# Get the HIP arch flags specified by PYTORCH_ROCM_ARCH.
# Usage:
#   torch_hip_get_arch_list(variable_to_store_flags)
#
macro(torch_hip_get_arch_list store_var)
  if(DEFINED ENV{PYTORCH_ROCM_ARCH})
    set(_TMP $ENV{PYTORCH_ROCM_ARCH})
  else()
    # Use arch of installed GPUs as default
    execute_process(COMMAND "rocm_agent_enumerator" COMMAND bash "-c" "grep -v gfx000 | sort -u | xargs | tr -d '\n'"
                    RESULT_VARIABLE ROCM_AGENT_ENUMERATOR_RESULT
                    OUTPUT_VARIABLE ROCM_ARCH_INSTALLED)
    if(NOT ROCM_AGENT_ENUMERATOR_RESULT EQUAL 0)
      message(FATAL_ERROR " Could not detect ROCm arch for GPUs on machine. Result: '${ROCM_AGENT_ENUMERATOR_RESULT}'")
    endif()
    set(_TMP ${ROCM_ARCH_INSTALLED})
  endif()
  string(REPLACE " " ";" ${store_var} "${_TMP}")
endmacro()

macro(pytorch_load_hip)
  find_package(hip REQUIRED CONFIG)
  message(STATUS "hip version: ${hip_VERSION}")
  find_package(amd_comgr REQUIRED)
  message(STATUS "amd_comgr version: ${amd_comgr_VERSION}")
  find_package(rocrand REQUIRED)
  message(STATUS "rocrand version: ${rocrand_VERSION}")
  find_package(hiprand REQUIRED)
  message(STATUS "hiprand version: ${hiprand_VERSION}")
  find_package(rocblas REQUIRED)
  message(STATUS "rocblas version: ${rocblas_VERSION}")
  find_package(hipblas REQUIRED)
  message(STATUS "hipblas_VERSION: ${hipblas_VERSION}")
  find_package(miopen REQUIRED)
  message(STATUS "miopen version: ${miopen_VERSION}")
  find_package(hipfft REQUIRED)
  message(STATUS "hipfft version: ${hipfft_VERSION}")
  find_package(hipsparse REQUIRED)
  message(STATUS "hipsparse version: ${hipsparse_VERSION}")
  find_package(rocprim REQUIRED)
  message(STATUS "rocprim version: ${rocprim_VERSION}")
  find_package(hipcub REQUIRED)
  message(STATUS "hipcub version: ${hipcub_VERSION}")
  find_package(rocthrust REQUIRED)
  message(STATUS "rocthrust version: ${rocthrust_VERSION}")
  find_package(hipsolver REQUIRED)
  message(STATUS "hipsolver versio: ${hipsolver_VERSION}")

  if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
    message(WARNING "Work around hiprtc cmake failure for cmake >= 4")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    find_package(hiprtc REQUIRED)
    unset(CMAKE_POLICY_VERSION_MINIMUM)
  else()
    find_package(hiprtc REQUIRED)
  endif()
  message(STATUS "hiprtc version: ${hiprtc_VERSION}")

  # Original version made these UNIX-only.
  if(NOT WIN32)
    find_package(rccl REQUIRED)
    message(STATUS "rccl version: ${rccl_VERSION}")
    find_package(hsa-runtime64 REQUIRED)
    message(STATUS "hsa-runtime64 version: ${hsa-runtime64_VERSION}")
  endif()
  find_package(hipblaslt REQUIRED)
  message(STATUS "hipblaslt version: ${hipblaslt_VERSION}")

  # Extract ROCM version parts from the hip package version.
  string(REPLACE "." ";" ROCM_VERSION_PARTS "${hip_VERSION}")
  list(GET ROCM_VERSION_PARTS 0 ROCM_VERSION_DEV_MAJOR)
  list(GET ROCM_VERSION_PARTS 1 ROCM_VERSION_DEV_MINOR)
  list(GET ROCM_VERSION_PARTS 2 ROCM_VERSION_DEV_PATCH)
  set(ROCM_VERSION "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")

  message(STATUS "\n***** ROCm version: ****\n")
  message(STATUS "  ROCM_VERSION: ${ROCM_VERSION}")
  message(STATUS "  ROCM_VERSION_DEV_MAJOR: ${ROCM_VERSION_DEV_MAJOR}")
  message(STATUS "  ROCM_VERSION_DEV_MINOR: ${ROCM_VERSION_DEV_MINOR}")
  message(STATUS "  ROCM_VERSION_DEV_PATCH: ${ROCM_VERSION_DEV_PATCH}")
  message(STATUS "  HIP_VERSION_MAJOR: ${ROCM_VERSION_DEV_MAJOR}")
  message(STATUS "  HIP_VERSION_MINOR: ${ROCM_VERSION_DEV_MINOR}")

  # Create ROCM_VERSION_DEV_INT which is later used as a preprocessor macros
  set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
  math(EXPR ROCM_VERSION_DEV_INT "(${ROCM_VERSION_DEV_MAJOR}*10000) + (${ROCM_VERSION_DEV_MINOR}*100) + ${ROCM_VERSION_DEV_PATCH}")
  math(EXPR TORCH_HIP_VERSION "(${ROCM_VERSION_DEV_MAJOR} * 100) + ${ROCM_VERSION_DEV_MINOR}")

  message(STATUS "  ROCM_VERSION_DEV_INT:   ${ROCM_VERSION_DEV_INT}")
  message(STATUS "  TORCH_HIP_VERSION: ${TORCH_HIP_VERSION}")

  # Locate the ROCM_ROCTX_LIB that kineto depends on. This is either part of
  # roctracer (deprecated) and located with find_library(roctx64) or it is
  # part of rocprofiler-sdk (aka. rocprofiler v3) as the rocprofiler-sdk-tx
  # library.
  # TODO: This isn't quite right and needs to mate up with whether kineto
  # depends on roctracer or rocprofiler-sdk. The coupling here is fragile and
  # needs to be reworked.
  if(NOT WIN32)
    find_package(rocprofiler-sdk-roctx)
    if(rocprofiler-sdk-roctx_FOUND)
      message(STATUS "rocprofiler-sdk-roctx version: ${rocprofiler-sdk-roctx_VERSION} found (will use instead of roctracer)")
      set(ROCM_ROCTX_LIB rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library)
    else()
      find_library(ROCM_ROCTX_LIB roctx64)
      if(NOT ROCM_ROCTX_LIB)
        cmake(WARNING "Neither rocprofiler-sdk nor libroctx64.so was found: This may result in errors if components rely on it")
      endif()
    endif()
  endif()

  # PyTorch makes some use of hip_add_library and friends, which are only
  # available in the legacy FindHIP.cmake finder module. This is bundled in the
  # same CMAKE_PREFIX_PATH as is used for the regular packages, but is put in
  # a different place on Linux vs Windows for reasons that are lost to time:
  #   Linux: lib/cmake/hip/FindHIP.cmake
  #   Windows: lib/cmake/FindHIP.cmake
  # While we could ask the user to provide an explicit CMAKE_MODULE_PATH, we
  # do some path munging in an attempt to make this legacy hiccup transparent
  # to most. If this mechanism ever breaks, the fix is to configure explicitly
  # with CMAKE_MODULE_PATH pointing at the directory in the ROCM SDK that
  # contains FindHIP.cmake.
  
  function(find_rocm_sdk_module_path)
    set(hip_lib_dir "${hip_LIB_INSTALL_DIR}")
    foreach(candidate_path "${hip_lib_dir}/cmake" "${hip_lib_dir}/cmake/hip" "${hip_lib_dir}/../cmake")
      if(EXISTS "${candidate_path}/FindHIP.cmake")
        list(PREPEND CMAKE_MODULE_PATH "${candidate_path}")
        message(STATUS "Legacy FindHIP.cmake module found in ${candidate_path}")
        set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
        return()
      endif()
    endforeach()

    message(STATUS "Could not locate legacy FindHIP.cmake: You may need to set CMAKE_MODULE_PATH explicitly to its location")
  endfunction()
  find_rocm_sdk_module_path()
  find_package(HIP MODULE REQUIRED)

  set(HIP_NEW_TYPE_ENUMS ON)
  set(PYTORCH_FOUND_HIP ON)
endmacro()

message(STATUS "___ROCM")
set(PYTORCH_FOUND_HIP FALSE)
set(HIP_PLATFORM "amd")
find_package(hip CONFIG)
if(hip_FOUND)
  # Apparently, aotriton compilation breaks if PYTORCH_ROCM_ARCH isn't converted to a list here.
  torch_hip_get_arch_list(PYTORCH_ROCM_ARCH)
  if(PYTORCH_ROCM_ARCH STREQUAL "")
    message(FATAL_ERROR "No GPU arch specified for ROCm build. Please use PYTORCH_ROCM_ARCH environment variable to specify GPU archs to build for.")
  endif()
  message("Building PyTorch for GPU arch: ${PYTORCH_ROCM_ARCH}")
  pytorch_load_hip()
endif()
