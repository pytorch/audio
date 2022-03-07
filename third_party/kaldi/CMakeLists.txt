set(KALDI_REPO ${CMAKE_CURRENT_SOURCE_DIR}/submodule)

if (NOT EXISTS ${KALDI_REPO}/src/base/version.h)
# Apply custom patch
execute_process(
  WORKING_DIRECTORY ${KALDI_REPO}
  COMMAND "git" "checkout" "."
  )
execute_process(
  WORKING_DIRECTORY ${KALDI_REPO}
  COMMAND git apply ../kaldi.patch
  )
# Update the version string
execute_process(
  WORKING_DIRECTORY ${KALDI_REPO}/src/base
  COMMAND bash get_version.sh
  )
endif()

set(KALDI_SOURCES
  src/matrix/kaldi-vector.cc
  src/matrix/kaldi-matrix.cc
  submodule/src/base/kaldi-error.cc
  submodule/src/base/kaldi-math.cc
  submodule/src/feat/feature-functions.cc
  submodule/src/feat/pitch-functions.cc
  submodule/src/feat/resample.cc
  )

add_library(kaldi STATIC ${KALDI_SOURCES})
target_include_directories(kaldi PUBLIC src submodule/src)
target_include_directories(kaldi PRIVATE ${TORCH_INCLUDE_DIRS})
