#!/usr/bin/env bash

set -eux

root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

eval "$("${conda_dir}/bin/conda" shell.bash hook)"
conda activate "${env_dir}"

# 1. Install tools
conda install flake8
printf "Installed flake8: "
flake8 --version

clangformat_path="${root_dir}/clang-format"
curl https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/clang-format-linux64 -o "${clangformat_path}"
chmod +x "${clangformat_path}"
printf "Installed clang-fortmat"
"${clangformat_path}" --version

# 2. Run style checks
# We want to run all the style checks even if one of them fail.

set +e

exit_status=0

printf "\x1b[34mRunning flake8:\x1b[0m\n"
flake8 torchaudio test build_tools/setup_helpers
status=$?
exit_status="$((exit_status+status))"
if [ "${status}" -ne 0 ]; then
    printf "\x1b[31mflake8 failed. Check the format of Python files.\x1b[0m\n"
fi

printf "\x1b[34mRunning clang-format:\x1b[0m\n"
"${this_dir}"/run_clang_format.py \
  -r torchaudio/csrc \
  --clang-format-executable "${clangformat_path}" \
    && git diff --exit-code
status=$?
exit_status="$((exit_status+status))"
if [ "${status}" -ne 0 ]; then
    printf "\x1b[31mC++ files are not formatted. Please use clang-format to format CPP files.\x1b[0m\n"
fi
exit $exit_status
