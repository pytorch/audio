import os
import sys
from sysconfig import get_config_var
import platform
import subprocess
from pathlib import Path
from setuptools import Extension
from setuptools.command.build_ext import build_ext

import torch

__all__ = [
    'get_ext_modules',
    'CMakeBuild',
]

_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()
_TORCHAUDIO_DIR = _ROOT_DIR / 'torchaudio'


def _get_build_sox():
    val = os.environ.get('BUILD_SOX', '0')
    trues = ['1', 'true', 'TRUE', 'on', 'ON', 'yes', 'YES']
    falses = ['0', 'false', 'FALSE', 'off', 'OFF', 'no', 'NO']
    if val in trues:
        return True
    if val not in falses:
        print(
            f'WARNING: Unexpected environment variable value `BUILD_SOX={val}`. '
            f'Expected one of {trues + falses}')
    return False


def get_ext_modules():
    if platform.system() == 'Windows':
        return None
    return [Extension(name='torchaudio._torchaudio', sources=[])]


# Based off of
# https://github.com/pybind/cmake_example/blob/580c5fd29d4651db99d8874714b07c0c49a53f8a/setup.py
class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake is not available.")
        super().run()

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # library_path = os.path.join(get_config_var('LIBDIR'), get_config_var('LDLIBRARY'))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython3_LIBRARY={get_config_var('LIBDIR')}",
            f"-DPython3_INCLUDE_DIR={get_config_var('INCLUDEPY')}",
            f"-DBUILD_SOX:BOOL={_get_build_sox()}",
            "-DBUILD_PYTHON_EXTENSION:BOOL=ON",
            "-DBUILD_LIBTORCHAUDIO:BOOL=OFF",
        ]
        build_args = [
            "--verbose",
        ]

        if 'CMAKE_CXX_FLAGS' in os.environ:
            cmake_args += [f"-DCMAKE_CXX_FLAGS={os.environ['CMAKE_CXX_FLAGS']}"]

        # Default to Ninja
        if 'CMAKE_GENERATOR' not in os.environ:
            cmake_args += ["-GNinja"]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split('.')
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = '.'.join(without_abi)
        return ext_filename
