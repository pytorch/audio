import os
import platform
import subprocess
from pathlib import Path
import distutils.sysconfig

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


def _get_build(var, default=False):
    if var not in os.environ:
        return default

    val = os.environ.get(var, '0')
    trues = ['1', 'true', 'TRUE', 'on', 'ON', 'yes', 'YES']
    falses = ['0', 'false', 'FALSE', 'off', 'OFF', 'no', 'NO']
    if val in trues:
        return True
    if val not in falses:
        print(
            f'WARNING: Unexpected environment variable value `{var}={val}`. '
            f'Expected one of {trues + falses}')
    return False


_BUILD_SOX = False if platform.system() == 'Windows' else _get_build("BUILD_SOX")
_BUILD_KALDI = False if platform.system() == 'Windows' else _get_build("BUILD_KALDI", True)
_BUILD_TRANSDUCER = _get_build("BUILD_TRANSDUCER")
_USE_ROCM = _get_build("USE_ROCM")
_USE_CUDA = torch.cuda.is_available()


def get_ext_modules():
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

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            '-DCMAKE_VERBOSE_MAKEFILE=ON',
            f"-DPython_INCLUDE_DIR={distutils.sysconfig.get_python_inc()}",
            f"-DBUILD_SOX:BOOL={'ON' if _BUILD_SOX else 'OFF'}",
            f"-DBUILD_KALDI:BOOL={'ON' if _BUILD_KALDI else 'OFF'}",
            f"-DBUILD_TRANSDUCER:BOOL={'ON' if _BUILD_TRANSDUCER else 'OFF'}",
            "-DBUILD_TORCHAUDIO_PYTHON_EXTENSION:BOOL=ON",
            "-DBUILD_LIBTORCHAUDIO:BOOL=OFF",
            f"-DUSE_ROCM:BOOL={'ON' if _USE_ROCM else 'OFF'}",
            f"-DUSE_CUDA:BOOL={'ON' if _USE_CUDA else 'OFF'}",
        ]
        build_args = [
            '--target', 'install'
        ]

        # Default to Ninja
        if 'CMAKE_GENERATOR' not in os.environ or platform.system() == 'Windows':
            cmake_args += ["-GNinja"]
        if platform.system() == 'Windows':
            import sys
            python_version = sys.version_info
            cmake_args += [
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
                f"-DPYTHON_VERSION={python_version.major}.{python_version.minor}",
            ]

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
            ["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp)

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split('.')
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = '.'.join(without_abi)
        return ext_filename
