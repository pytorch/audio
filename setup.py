#!/usr/bin/env python
import os
import platform

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


def check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in set(['ON', '1', 'YES', 'TRUE', 'Y'])

DEBUG = check_env_flag('DEBUG')
eca = []
ela = []
if DEBUG:
    if platform.system() == 'Windows':
        ela += ['/DEBUG:FULL']
    else:
        eca += ['-O0', '-g']
        ela += ['-O0', '-g']

setup(
    name="torchaudio",
    version="0.2",
    description="An audio package for PyTorch",
    url="https://github.com/pytorch/audio",
    author="Soumith Chintala, David Pollack, Sean Naren, Peter Goldsborough",
    author_email="soumith@pytorch.org",
    classifiers=[
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: C++",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    ext_modules=[
        CppExtension(
            '_torch_sox',
            ['torchaudio/torch_sox.cpp'],
            libraries=['sox'],
            extra_compile_args=eca,
            extra_link_args=ela),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch']
)
