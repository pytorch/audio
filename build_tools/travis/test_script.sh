#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
set -e

python --version

run_tests() {
  # find all the test files that match "test*.py"
  TEST_FILES="$(find test -type f -name "test*.py" | sort)"
  echo "Test files are:"
  echo $TEST_FILES

  echo "Executing tests:"
  EXIT_STATUS=0
  for FILE in $TEST_FILES; do
    # run each file on a separate process. if one fails, just keep going and
    # return the final exit status.
    python -m unittest -v $FILE
    STATUS=$?
    EXIT_STATUS="$(($EXIT_STATUS+STATUS))"
  done

  echo "Done, exit status: $EXIT_STATUS"
  exit $EXIT_STATUS
}

if [[ "$RUN_FLAKE8" == "true" ]]; then
  flake8
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
  run_tests
fi
