#!/usr/bin/env python
import os
import shutil
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
import distutils.command.clean

from tools import setup_helpers

ROOT_DIR = Path(__file__).parent.resolve()


# Creating the version file
version = '0.7.0a0'
sha = 'Unknown'

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT_DIR).decode('ascii').strip()
except Exception:
    pass

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    version += '+' + sha[:7]
print('-- Building version ' + version)

version_path = ROOT_DIR / 'torchaudio' / 'version.py'
with open(version_path, 'w') as f:
    f.write("__version__ = '{}'\n".format(version))
    f.write("git_version = {}\n".format(repr(sha)))

pytorch_package_version = os.getenv('PYTORCH_VERSION')

pytorch_package_dep = 'torch'
if pytorch_package_version is not None:
    pytorch_package_dep += "==" + pytorch_package_version


class clean(distutils.command.clean.clean):
    def run(self):
        # Run default behavior first
        distutils.command.clean.clean.run(self)

        # Remove torchaudio extension
        for path in (ROOT_DIR / 'torchaudio').glob('**/*.so'):
            print(f'removing \'{path}\'')
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / 'build',
            ROOT_DIR / 'third_party' / 'build',
        ]
        for path in build_dirs:
            if path.exists():
                print(f'removing \'{path}\' (and everything under it)')
                shutil.rmtree(str(path), ignore_errors=True)


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
    packages=find_packages(exclude=["build*", "test*", "torchaudio.csrc*", "third_party*", "build_tools*"]),
    ext_modules=setup_helpers.get_ext_modules(),
    cmdclass={
        'build_ext': setup_helpers.BuildExtension.with_options(no_python_abi_suffix=True)
    },
    install_requires=[pytorch_package_dep],
    zip_safe=False,
)
