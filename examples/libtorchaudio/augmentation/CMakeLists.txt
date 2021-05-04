add_executable(augment main.cpp)
target_link_libraries(augment "${TORCH_LIBRARIES}" "${TORCHAUDIO_LIBRARY}")
set_property(TARGET augment PROPERTY CXX_STANDARD 14)
