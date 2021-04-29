#!/usr/bin/env bash
set -ex

BUILD_TRANSDUCER=1 BUILD_SOX=1 python setup.py install --single-version-externally-managed --record=record.txt
