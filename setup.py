#!/usr/bin/env python
import os
import platform
import sys
import subprocess

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


def check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in set(['ON', '1', 'YES', 'TRUE', 'Y'])

DEBUG = check_env_flag('DEBUG')
IS_WHEEL = check_env_flag('IS_WHEEL')
IS_CONDA = check_env_flag('IS_CONDA')

print('DEBUG:', DEBUG, 'IS_WHEEL:', IS_WHEEL, 'IS_CONDA:', IS_CONDA)

eca = []
ela = []
if DEBUG:
    if platform.system() == 'Windows':
        ela += ['/DEBUG:FULL']
    else:
        eca += ['-O0', '-g']
        ela += ['-O0', '-g']


libraries = []
include_dirs = []
extra_objects = []

# Hypothetically, the conda distribution could rely on an external sox,
# but the library is pretty small and it is not available on the default
# anaconda channel.  So we statically link it in, just as we do with wheels.
if IS_WHEEL or IS_CONDA:
    audio_path = os.path.dirname(os.path.abspath(__file__))

    include_dirs += [os.path.join(audio_path, 'third_party/flac/include')]
    include_dirs += [os.path.join(audio_path, 'third_party/lame/include')]
    include_dirs += [os.path.join(audio_path, 'third_party/sox/include')]
    include_dirs += [os.path.join(audio_path, 'third_party/mad/include')]

    # proper link order (sox, mad, flac, lame)
    # (the most important thing is that dependencies come after a libraryl
    # e.g., sox comes first)
    extra_objects += [os.path.join(audio_path, 'third_party/sox/lib/libsox.a')]
    extra_objects += [os.path.join(audio_path, 'third_party/mad/lib/libmad.a')]
    extra_objects += [os.path.join(audio_path, 'third_party/flac/lib/libFLAC.a')]
    extra_objects += [os.path.join(audio_path, 'third_party/lame/lib/libmp3lame.a')]
else:
    libraries += ['sox']


# Creating the version file
cwd = os.path.dirname(os.path.abspath(__file__))
version = '0.4.0a0'
sha = 'Unknown'

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]
print('-- Building version ' + version)

version_path = os.path.join(cwd, 'torchaudio', 'version.py')
with open(version_path, 'w') as f:
    f.write("__version__ = '{}'\n".format(version))
    f.write("git_version = {}\n".format(repr(sha)))

pytorch_package_version = os.getenv('PYTORCH_VERSION')

pytorch_package_dep = 'torch'
if pytorch_package_version is not None:
    pytorch_package_dep += "==" + pytorch_package_version

if platform.system() == 'Windows':
    ext_modules = None
else:
    ext_modules=[
        CppExtension(
            '_torch_sox',
            ['torchaudio/torch_sox.cpp'],
            libraries=libraries,
            include_dirs=include_dirs,
            extra_compile_args=eca,
            extra_objects=extra_objects,
            extra_link_args=ela),
    ]

setup(
    name="torchaudio",
    version=version,
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
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=[pytorch_package_dep]
)
