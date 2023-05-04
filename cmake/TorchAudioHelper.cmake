find_package(Torch REQUIRED)

message(STATUS TORCH_LIBRARIES="${TORCH_LIBRARIES}")

if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    
    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
endif()
    
function(print_properties)
    message("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction()
    
function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()

print_target_properties(torch)
print_target_properties(torch_cpu_library)
print_target_properties(torch_cpu)
print_target_properties(c10)
print_target_properties(caffe2::mkl)
get_target_property(dep torch_cpu INTERFACE_LINK_LIBRARIES)
if ("caffe2::mkl" IN_LIST dep)
  list(REMOVE_ITEM dep "caffe2::mkl")
  set_target_properties(torch_cpu PROPERTIES INTERFACE_LINK_LIBRARIES "${dep}")
endif()
print_target_properties(torch_cpu)

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
