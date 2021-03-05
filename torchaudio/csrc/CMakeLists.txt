get_property(TORCHAUDIO_THIRD_PARTIES GLOBAL PROPERTY TORCHAUDIO_THIRD_PARTIES)

################################################################################
# Stuff common to libtorchaudio and _torchaudio.so
################################################################################
set(
  LIBTORCHAUDIO_SOURCES
  lfilter.cpp
  overdrive.cpp
  utils.cpp
  )

if(BUILD_TRANSDUCER)
  list(APPEND LIBTORCHAUDIO_SOURCES transducer.cpp)
endif()

if(BUILD_KALDI)
  list(APPEND LIBTORCHAUDIO_SOURCES kaldi.cpp)
endif()

if(BUILD_SOX)
  set(
    SOX_SOURCES
    sox/io.cpp
    sox/utils.cpp
    sox/effects.cpp
    sox/effects_chain.cpp
    sox/types.cpp
  )
  list(APPEND LIBTORCHAUDIO_SOURCES ${SOX_SOURCES})
endif()

################################################################################
# libtorchaudio.so
################################################################################
if(BUILD_LIBTORCHAUDIO)
  add_library(
    libtorchaudio
    SHARED
    ${LIBTORCHAUDIO_SOURCES}
    )
  set_target_properties(libtorchaudio PROPERTIES PREFIX "")

  target_include_directories(
    libtorchaudio
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    )

  target_link_libraries(
    libtorchaudio
    ${TORCHAUDIO_THIRD_PARTIES}
    )

  install(
    TARGETS
    libtorchaudio
    LIBRARY DESTINATION lib
    )

  set(TORCHAUDIO_LIBRARY -Wl,--no-as-needed libtorchaudio -Wl,--as-needed CACHE INTERNAL "")
endif()

################################################################################
# _torchaudio.so
################################################################################
if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  add_library(
    _torchaudio
    SHARED
    pybind.cpp
    ${LIBTORCHAUDIO_SOURCES}
    )

  set_target_properties(_torchaudio PROPERTIES PREFIX "")

  if (APPLE)
    # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
    # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
    set_target_properties(_torchaudio PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()

  target_compile_definitions(
    _torchaudio PRIVATE TORCH_API_INCLUDE_EXTENSION_H
  )

  if (BUILD_SOX)
    target_compile_definitions(_torchaudio PRIVATE INCLUDE_SOX)
  endif()

  if (BUILD_KALDI)
    target_compile_definitions(_torchaudio PRIVATE INCLUDE_KALDI)
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
    set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION} 3.9 3.8 3.7 3.6)
    find_package(Python3 COMPONENTS Development)
    set(ADDITIONAL_ITEMS Python3::Python)
  endif()

  target_link_libraries(
    _torchaudio
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    ${TORCHAUDIO_THIRD_PARTIES}
    ${ADDITIONAL_ITEMS}
    )

  install(TARGETS _torchaudio LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()
