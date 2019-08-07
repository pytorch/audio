#!/bin/bash

set -ex
export OUT_DIR="/remote/cpu"

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd /opt/python
DESIRED_PYTHON=(*/)
for desired_py in "${DESIRED_PYTHON[@]}"; do
    python_installations+=("/opt/python/$desired_py")
done

. "$script_dir/../setup_workdir"

OLD_PATH=$PATH
for PYDIR in "${python_installations[@]}"; do
    export PATH="$PYDIR/bin:$OLD_PATH"
    pip install --upgrade pip
    "$script_dir/build_wheel.sh"
done
