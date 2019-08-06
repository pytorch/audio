#!/usr/bin/env bash
set -ex

build_tools/packaging/build_from_source.sh "$(pwd)"

IS_CONDA=1 python setup.py install --single-version-externally-managed --record=record.txt
