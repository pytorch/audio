cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Most of the configurations are taken from PyTorch
# https://github.com/pytorch/pytorch/blob/0c9fb4aff0d60eaadb04e4d5d099fb1e1d5701a9/CMakeLists.txt

# Use compiler ID "AppleClang" instead of "Clang" for XCode.
# Not setting this sometimes makes XCode C compiler gets detected as "Clang",
# even when the C++ one is detected as "AppleClang".
cmake_policy(SET CMP0010 NEW)
cmake_policy(SET CMP0025 NEW)

# Suppress warning flags in default MSVC configuration.  It's not
# mandatory that we do this (and we don't if cmake is old), but it's
# nice when it's possible, and it's possible on our Windows configs.
if(NOT CMAKE_VERSION VERSION_LESS 3.15.0)
  cmake_policy(SET CMP0092 NEW)
endif()

# Suppress warning about ExternalProject_Add timestamp
if(NOT CMAKE_VERSION VERSION_LESS 3.24.0)
  cmake_policy(SET CMP0135 OLD)
endif()

project(torchaudio)


# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++17. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_C_STANDARD   11 CACHE STRING "The C standard whose features are requested to build this target.")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Apple specific
if(APPLE)
  # Get clang version on macOS
  execute_process( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
  string(REGEX REPLACE "Apple LLVM version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
  message( STATUS "CLANG_VERSION_STRING:         " ${CLANG_VERSION_STRING} )

  # RPATH stuff
  set(CMAKE_MACOSX_RPATH ON)

  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()


# Options
option(BUILD_SOX "Build libsox statically" ON)
option(BUILD_RIR "Enable RIR simulation" ON)
option(BUILD_RNNT "Enable RNN transducer" ON)
option(BUILD_ALIGN "Enable forced alignment" ON)
option(BUILD_CUDA_CTC_DECODER "Build CUCTC decoder" OFF)
option(BUILD_TORCHAUDIO_PYTHON_EXTENSION "Build Python extension" OFF)
option(USE_FFMPEG "Enable ffmpeg-based features" OFF)
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_ROCM "Enable ROCM support" OFF)
option(USE_OPENMP "Enable OpenMP support" OFF)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# check that USE_CUDA and USE_ROCM are not set at the same time
if(USE_CUDA AND USE_ROCM)
  message(FATAL "CUDA and ROCm are mutually exclusive")
endif()

if(USE_ROCM)
  # Find the HIP package, set the HIP paths, load the HIP CMake.
  include(cmake/LoadHIP.cmake)
  if(NOT PYTORCH_FOUND_HIP)
    set(USE_ROCM OFF)
  endif()
endif()

if(USE_CUDA)
  enable_language(CUDA)
  set(
    CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} \
    -DCUDA_HAS_FP16=1 \
    -D__CUDA_NO_HALF_OPERATORS__ \
    -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__"
  )
endif()

include(cmake/TorchAudioHelper.cmake)

# https://github.com/pytorch/pytorch/issues/54174
function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
    endif()
endfunction()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
  if(USE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819")
    foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration
      set_but_not_used field_without_dll_interface
      base_class_has_different_dll_interface
      dll_interface_conflict_none_assumed
      dll_interface_conflict_dllexport_assumed
      implicit_return_from_non_void_function
      unsigned_compare_with_zero
      declared_but_not_referenced
      bad_friend_decl)
      string(
        APPEND
        CMAKE_CUDA_FLAGS
        " -Xcudafe \
        --diag_suppress=${diag} \
        -DCUDA_HAS_FP16=1 \
        -D__CUDA_NO_HALF_OPERATORS__ \
        -D__CUDA_NO_HALF_CONVERSIONS__ \
        -D__CUDA_NO_HALF2_OPERATORS__"
      )
    endforeach()
    CUDA_CONVERT_FLAGS(torch_cpu)
    if(TARGET torch_cuda)
      CUDA_CONVERT_FLAGS(torch_cuda)
    endif()
    if(TARGET torch_cuda_cu)
      CUDA_CONVERT_FLAGS(torch_cuda_cu)
    endif()
    if(TARGET torch_cuda_cpp)
      CUDA_CONVERT_FLAGS(torch_cuda_cpp)
    endif()
  endif()
endif()

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# TORCH_CXX_FLAGS contains the same -D_GLIBCXX_USE_CXX11_ABI value as PyTorch
if (MSVC)
  set(warning_flags /W4)
  # the following line is added in order to export symbols when building on Windows
  # this approach has some limitations as documented in https://github.com/pytorch/pytorch/pull/3650
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  set(warning_flags -Wall)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${warning_flags} ${TORCH_CXX_FLAGS}")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found ccache")
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
  message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
endif()

add_subdirectory(torchaudio/csrc)
if (BUILD_SOX)
  add_subdirectory(third_party/sox)
  add_subdirectory(torchaudio/csrc/sox)
endif()
if (USE_FFMPEG)
  if (DEFINED ENV{FFMPEG_ROOT})
    add_subdirectory(third_party/ffmpeg/single)
  else()
    message(STATUS "Building FFmpeg integration with multi version support")
    add_subdirectory(third_party/ffmpeg/multi)
  endif()
  add_subdirectory(torchaudio/csrc/ffmpeg)
endif()
if (BUILD_CUDA_CTC_DECODER)
  if (NOT USE_CUDA)
    message(FATAL "BUILD_CUDA_CTC_DECODER=1 but USE_CUDA=0.")
  endif()
  add_subdirectory(torchaudio/csrc/cuctc)
endif()
