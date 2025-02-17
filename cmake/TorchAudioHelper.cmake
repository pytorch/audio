find_package(Torch REQUIRED)

# Remove stray mkl dependency found in Intel mac.
#
# For Intel mac, torch_cpu has caffe2::mkl, which adds link flags like
# -lmkl_intel_ilp64, -lmkl_core and -lmkl_intel_thread.
# Even though TorchAudio does not call any of MKL functions directly,
# Apple's linker does not drop them, instead it bakes these dependencies
# Therefore, we remove it.
# See https://github.com/pytorch/audio/pull/3307
get_target_property(dep torch_cpu INTERFACE_LINK_LIBRARIES)
if ("caffe2::mkl" IN_LIST dep)
  list(REMOVE_ITEM dep "caffe2::mkl")
  set_target_properties(torch_cpu PROPERTIES INTERFACE_LINK_LIBRARIES "${dep}")
endif()

function (_library destination name source include_dirs link_libraries compile_defs)
  add_library(${name} SHARED ${source})
  target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}/src;${include_dirs}")
  target_link_libraries(${name} ${link_libraries})
  target_compile_definitions(${name} PRIVATE ${compile_defs})
  set_target_properties(${name} PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
  endif(MSVC)
  install(
    TARGETS ${name}
    LIBRARY DESTINATION "${destination}"
    RUNTIME DESTINATION "${destination}"  # For Windows
    )
endfunction()

function(torchaudio_library name source include_dirs link_libraries compile_defs)
  _library(
    torchaudio/lib
    "${name}"
    "${source}"
    "${include_dirs}"
    "${link_libraries}"
    "${compile_defs}"
    )
endfunction()

function(torio_library name source include_dirs link_libraries compile_defs)
  _library(
    torio/lib
    "${name}"
    "${source}"
    "${include_dirs}"
    "${link_libraries}"
    "${compile_defs}"
    )
endfunction()

if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  # See https://github.com/pytorch/pytorch/issues/38122
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

  if (WIN32)
    message(PYTHON_VERSION="${PYTHON_VERSION}")
    set(CMAKE_FIND_DEBUG_MODE TRUE)
    find_package(Python 3.13.2 COMPONENTS Development)
    set(CMAKE_FIND_DEBUG_MODE FALSE)
    set(ADDITIONAL_ITEMS Python::Python)
  endif()
  function(_extension destination name sources include_dirs libraries definitions)
    add_library(${name} SHARED ${sources})
    target_compile_definitions(${name} PRIVATE "${definitions}")
    target_include_directories(
      ${name}
      PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${Python_INCLUDE_DIR}
      "${TORCH_INSTALL_PREFIX}/include"
      ${include_dirs})
    target_link_libraries(
      ${name}
      ${libraries}
      ${TORCH_PYTHON_LIBRARY}
      ${ADDITIONAL_ITEMS}
      )
    set_target_properties(${name} PROPERTIES PREFIX "")
    if (MSVC)
      set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
    endif(MSVC)
    if (APPLE)
      # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
      # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
      set_target_properties(${name} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    install(
      TARGETS ${name}
      LIBRARY DESTINATION "${destination}"
      RUNTIME DESTINATION "${destination}"  # For Windows
      )
  endfunction()

  function(torchaudio_extension name sources include_dirs libraries definitions)
    _extension(
      torchaudio/lib
      "${name}"
      "${sources}"
      "${include_dirs}"
      "${libraries}"
      "${definitions}"
      )
  endfunction()
  function(torio_extension name sources include_dirs libraries definitions)
    _extension(
      torio/lib
      "${name}"
      "${sources}"
      "${include_dirs}"
      "${libraries}"
      "${definitions}"
      )
  endfunction()
endif()


if (USE_CUDA)
  add_library(cuda_deps INTERFACE)
  target_include_directories(cuda_deps INTERFACE ${CUDA_TOOLKIT_INCLUDE})
  target_compile_definitions(cuda_deps INTERFACE USE_CUDA)
  target_link_libraries(
    cuda_deps
    INTERFACE
    ${C10_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    )
endif()
