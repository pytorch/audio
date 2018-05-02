#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages

import build

this_file = os.path.dirname(__file__)

setup(
    name="torchaudio",
    version="0.1",
    description="An audio package for PyTorch",
    url="https://github.com/pytorch/audio",
    author="Soumith Chintala, David Pollack, Sean Naren",
    author_email="soumith@pytorch.org",
    # Require cffi.
    install_requires=["cffi>=1.0.0", "torch>=0.4"],
    setup_requires=["cffi>=1.0.0", "torch>=0.4"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
)
