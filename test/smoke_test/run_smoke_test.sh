#!/usr/bin/env bash

set -eux

# This script is used by CI to perform smoke tests on installed binaries.

# When `import torchaudio` is executed from the root directory of the repo,
# the source `torchaudio` directory will shadow the actual installation.
# Changing to this directory to avoid that.
cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"

python smoke_test.py
