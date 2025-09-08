#!/bin/bash

set -ex

# We need to install pybind11 because we need its CMake helpers in order to
# compile correctly on Mac. Pybind11 is actually a C++ header-only library,
# and PyTorch actually has it included. PyTorch, however, does not have the
# CMake helpers.
conda install -y pybind11 -c conda-forge
