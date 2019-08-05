#!/usr/bin/env bash
set -ex

PREFIX="$(pwd)"
. build_tools/packaging/wheel/build_from_source.sh

IS_CONDA=1 python setup.py install --single-version-externally-managed --record=record.txt
