find_package(Torch REQUIRED)

function (torchaudio_library name source include_dirs link_libraries compile_defs)
  add_library(${name} SHARED ${source})
  target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR};${include_dirs}")
  target_link_libraries(${name} ${link_libraries})
  target_compile_definitions(${name} PRIVATE ${compile_defs})
  set_target_properties(${name} PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
  endif(MSVC)
  install(
    TARGETS ${name}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib  # For Windows
    )
endfunction()


if (BUILD_TORCHAUDIO_PYTHON_EXTENSION)
  # See https://github.com/pytorch/pytorch/issues/38122
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

  if (WIN32)
    find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)
    set(ADDITIONAL_ITEMS Python3::Python)
  endif()
  function(torchaudio_extension name sources include_dirs libraries definitions)
    add_library(${name} SHARED ${sources})
    target_compile_definitions(${name} PRIVATE "${definitions}")
    target_include_directories(
      ${name}
      PRIVATE
      ${PROJECT_SOURCE_DIR}
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
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION lib  # For Windows
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
