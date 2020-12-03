import os
import sys
import platform
import subprocess
import sysconfig
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


def _get_cxx11_abi():
    try:
        return int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except ImportError:
        return 0


def get_ext_modules():
    if platform.system() == 'Windows':
        return None
    return [Extension(name='torchaudio._torchaudio', sources=[])]


def _get_python_include_dir():
    # https://github.com/pytorch/pytorch/blob/7f869dca70606c42994d822ba11362a353411a1c/cmake/Dependencies.cmake#L904-L940
    dir_ = distutils.sysconfig.get_python_inc()
    if os.path.exists(dir_):
        return dir_
    dir_ = sysconfig.get_paths()['include']
    if os.path.exists(dir_):
        return dir_
    raise RuntimeError('Cannot find Python development include directory.')


def _get_python_library():
    lib = sysconfig.get_paths()['stdlib']
    if os.path.exists(lib):
        return lib
    raise RuntimeError(f'Cannot find Python library. {lib}')


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
            f"-DBUILD_SOX:BOOL={_get_build_sox()}",
            f"-DPYTHON_INCLUDE_DIR={_get_python_include_dir()}",
            f"-DPYTHON_LIBRARY={_get_python_library()}",
            f"-DPYTHON_VERSION={sys.version_info[0]}.{sys.version_info[1]}",
            f"-D_GLIBCXX_USE_CXX11_ABI={_get_cxx11_abi()}",
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
