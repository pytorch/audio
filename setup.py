#!/usr/bin/env python

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="torchaudio",
    version="0.1",
    description="An audio package for PyTorch",
    url="https://github.com/pytorch/audio",
    author="Soumith Chintala, David Pollack, Sean Naren, Peter Goldsborough",
    author_email="soumith@pytorch.org",
    install_requires=["torch>=0.4"],
    setup_requires=["torch>=0.4"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    ext_modules=[
        CppExtension(
            '_torch_sox', ['torchaudio/torch_sox.cpp'], libraries=['sox']),
    ],
    cmdclass={'build_ext': BuildExtension})
