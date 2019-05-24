#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

python --version

run_tests() {
    # -v it's a short --verbose
    # -s means 'disable all capturing'
    # --forked is to run each test in separate process which is important for C/C++ extensions
    # -cov is for code coverage
    # --durations determines how many of the slowest test times to display
    py.test -s -v --forked --cov=torchaudio --durations=20
}

if [[ "$RUN_FLAKE8" == "true" ]]; then
    flake8
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi
