set(KALDI_REPO ${CMAKE_CURRENT_SOURCE_DIR}/submodule)

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
  COMMAND sh get_version.sh
  )

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
target_link_libraries(kaldi ${TORCH_LIBRARIES})
