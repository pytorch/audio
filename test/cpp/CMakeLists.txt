include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  wall_collision
  rir/wall_collision.cpp
)
target_link_libraries(
  wall_collision
  torch
  GTest::gtest_main
)
target_include_directories(
  wall_collision
  PRIVATE
  "${PROJECT_SOURCE_DIR}"
)
add_test(NAME wall_collision_test COMMAND wall_collision)
