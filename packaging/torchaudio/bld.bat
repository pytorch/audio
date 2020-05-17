@echo off

sh packaging/download_deps.sh "$(pwd)"

set IS_CONDA=1

python setup.py install --single-version-externally-managed --record=record.txt
