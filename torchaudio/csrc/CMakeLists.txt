get_property(TORCHAUDIO_THIRD_PARTIES GLOBAL PROPERTY TORCHAUDIO_THIRD_PARTIES)

################################################################################
# libtorchaudio
################################################################################
set(
  LIBTORCHAUDIO_SOURCES
  lfilter.cpp
  overdrive.cpp
  utils.cpp
  )

set(
  LIBTORCHAUDIO_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  )

set(
  LIBTORCHAUDIO_LINK_LIBRARIES
  torch
  ${TORCHAUDIO_THIRD_PARTIES}
  )

set(
  LIBTORCHAUDIO_COMPILE_DEFINITIONS)

#------------------------------------------------------------------------------#
# START OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#
if(BUILD_RNNT)
  list(
    APPEND
    LIBTORCHAUDIO_SOURCES
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
      LIBTORCHAUDIO_SOURCES
      rnnt/gpu/compute_alphas.cu
      rnnt/gpu/compute_betas.cu
      rnnt/gpu/compute.cu
      )
  endif()
endif()

if(USE_CUDA)
  list(
    APPEND
    LIBTORCHAUDIO_INCLUDE_DIRS
    ${CUDA_TOOLKIT_INCLUDE}
    )
  list(
    APPEND
    LIBTORCHAUDIO_LINK_LIBRARIES
    ${C10_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    )
  list(
    APPEND
    LIBTORCHAUDIO_COMPILE_DEFINITIONS
    USE_CUDA
  )
endif()

if(BUILD_KALDI)
  list(APPEND LIBTORCHAUDIO_SOURCES kaldi.cpp)
  list(APPEND LIBTORCHAUDIO_COMPILE_DEFINITIONS INCLUDE_KALDI)
endif()

if(BUILD_SOX)
  list(
    APPEND
    LIBTORCHAUDIO_SOURCES
    sox/io.cpp
    sox/utils.cpp
    sox/effects.cpp
    sox/effects_chain.cpp
    sox/types.cpp
    )
  list(
    APPEND
    LIBTORCHAUDIO_COMPILE_DEFINITIONS
    INCLUDE_SOX
    )
endif()

if(OpenMP_CXX_FOUND)
  list(
    APPEND
    LIBTORCHAUDIO_LINK_LIBRARIES
    OpenMP::OpenMP_CXX
    )
endif()

#------------------------------------------------------------------------------#
# END OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#
add_library(
  libtorchaudio
  SHARED
  ${LIBTORCHAUDIO_SOURCES}
  )
set_target_properties(libtorchaudio PROPERTIES PREFIX "")

target_include_directories(
  libtorchaudio
  PRIVATE
  ${LIBTORCHAUDIO_INCLUDE_DIRS}
  )
target_link_libraries(
  libtorchaudio
  ${LIBTORCHAUDIO_LINK_LIBRARIES}
  )
target_compile_definitions(
  libtorchaudio
  PRIVATE ${LIBTORCHAUDIO_COMPILE_DEFINITIONS}
  )

if (MSVC)
    set_target_properties(libtorchaudio PROPERTIES SUFFIX ".pyd")
endif(MSVC)

install(
  TARGETS libtorchaudio
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib  # For Windows
  )

if (APPLE)
  set(TORCHAUDIO_LIBRARY libtorchaudio CACHE INTERNAL "")
else()
  set(TORCHAUDIO_LIBRARY -Wl,--no-as-needed libtorchaudio -Wl,--as-needed CACHE INTERNAL "")
endif()

################################################################################
# _torchaudio.so
################################################################################
if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  set(
    EXTENSION_SOURCES
    pybind/pybind.cpp
    )
  if(BUILD_SOX)
    list(
      APPEND
      EXTENSION_SOURCES
      pybind/sox/effects.cpp
      pybind/sox/effects_chain.cpp
      pybind/sox/io.cpp
      pybind/sox/utils.cpp
      )
  endif()
  add_library(
    _torchaudio
    SHARED
    ${EXTENSION_SOURCES}
    )

  target_compile_definitions(
    _torchaudio
    PRIVATE ${LIBTORCHAUDIO_COMPILE_DEFINITIONS}
    )

  set_target_properties(_torchaudio PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(_torchaudio PROPERTIES SUFFIX ".pyd")
  endif(MSVC)

  if (APPLE)
    # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
    # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
    set_target_properties(_torchaudio PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()

  target_include_directories(
    _torchaudio
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${Python_INCLUDE_DIR}
    )

  # See https://github.com/pytorch/pytorch/issues/38122
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

  if (WIN32)
    find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)
    set(ADDITIONAL_ITEMS Python3::Python)
  endif()

  target_link_libraries(
    _torchaudio
    libtorchaudio
    ${TORCH_PYTHON_LIBRARY}
    ${ADDITIONAL_ITEMS}
    )

  install(
    TARGETS _torchaudio
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .  # For Windows
    )
endif()
