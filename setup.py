#!/usr/bin/env python
import os
import re
import shutil
import subprocess
from pathlib import Path
from setuptools import setup, find_packages
import distutils.command.clean

from build_tools import setup_helpers

ROOT_DIR = Path(__file__).parent.resolve()


def _run_cmd(cmd, default):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR).decode('ascii').strip()
    except Exception:
        return default


# Creating the version file
version = '0.10.0a0'
sha = _run_cmd(['git', 'rev-parse', 'HEAD'], default='Unknown')

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
        ]
        for path in build_dirs:
            if path.exists():
                print(f'removing \'{path}\' (and everything under it)')
                shutil.rmtree(str(path), ignore_errors=True)


def _get_packages():
    exclude = [
        "build*",
        "test*",
        "torchaudio.csrc*",
        "third_party*",
        "build_tools*",
    ]
    exclude_prototype = False
    branch_name = _run_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], default=None)
    is_on_tag = _run_cmd(['git', 'describe', '--tags', '--exact-match', '@'], default=None)

    if branch_name is not None and branch_name.startswith('release/'):
        print('On release branch')
        exclude_prototype = True
    if is_on_tag is not None and re.match(r'v[\d.]+(-rc\d+)?', is_on_tag):
        print('On release tag')
        exclude_prototype = True
    if exclude_prototype:
        print('Excluding torchaudio.prototype from the package.')
        exclude.append("torchaudio.prototype")
    return find_packages(exclude=exclude)


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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    packages=_get_packages(),
    ext_modules=setup_helpers.get_ext_modules(),
    cmdclass={
        'build_ext': setup_helpers.CMakeBuild,
        'clean': clean,
    },
    install_requires=[pytorch_package_dep],
    zip_safe=False,
)
