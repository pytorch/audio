#!/usr/bin/env bash

set -e

# shellcheck source=../../../../tools/conda_envs/utils.sh
. "$(git rev-parse --show-toplevel)/tools/conda_envs/utils.sh"

init_conda
activate_env master "${PYTHON_VERSION}"

codecov
