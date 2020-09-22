#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

# shellcheck source=../../../../tools/conda_envs/utils.sh
. "$(git rev-parse --show-toplevel)/tools/conda_envs/utils.sh"

init_conda
activate_env master "${PYTHON_VERSION}"
build_master "${CUDA_VERSION:-}"
