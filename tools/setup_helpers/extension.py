import distutils.sysconfig
import os
import platform
import subprocess
from pathlib import Path

import torch
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import CUDA_HOME

__all__ = [
    "get_ext_modules",
    "CMakeBuild",
]

_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()
_TORCHAUDIO_DIR = _ROOT_DIR / "torchaudio"


def _get_build(var, default=False):
    if var not in os.environ:
        return default

    val = os.environ.get(var, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        print(f"WARNING: Unexpected environment variable value `{var}={val}`. " f"Expected one of {trues + falses}")
    return False


_BUILD_CPP_TEST = _get_build("BUILD_CPP_TEST", False)
_BUILD_SOX = False if platform.system() == "Windows" else _get_build("BUILD_SOX", True)
_BUILD_RIR = _get_build("BUILD_RIR", True)
_BUILD_RNNT = _get_build("BUILD_RNNT", True)
_USE_FFMPEG = _get_build("USE_FFMPEG", True)
_USE_ROCM = _get_build("USE_ROCM", torch.backends.cuda.is_built() and torch.version.hip is not None)
_USE_CUDA = _get_build("USE_CUDA", torch.backends.cuda.is_built() and torch.version.hip is None)
_BUILD_ALIGN = _get_build("BUILD_ALIGN", True)
_BUILD_CUDA_CTC_DECODER = _get_build("BUILD_CUDA_CTC_DECODER", _USE_CUDA)
_USE_OPENMP = _get_build("USE_OPENMP", True) and "ATen parallel backend: OpenMP" in torch.__config__.parallel_info()
_TORCH_CUDA_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST", None)


def get_ext_modules():
    modules = [
        Extension(name="torchaudio.lib.libtorchaudio", sources=[]),
        Extension(name="torchaudio.lib._torchaudio", sources=[]),
    ]
    if _BUILD_SOX:
        modules.extend(
            [
                Extension(name="torchaudio.lib.libtorchaudio_sox", sources=[]),
                Extension(name="torchaudio.lib._torchaudio_sox", sources=[]),
            ]
        )
    if _BUILD_CUDA_CTC_DECODER:
        modules.extend(
            [
                Extension(name="torchaudio.lib.libctc_prefix_decoder", sources=[]),
                Extension(name="torchaudio.lib.pybind11_prefixctc", sources=[]),
            ]
        )
    if _USE_FFMPEG:
        if "FFMPEG_ROOT" in os.environ:
            # single version ffmpeg mode
            modules.extend(
                [
                    Extension(name="torio.lib.libtorio_ffmpeg", sources=[]),
                    Extension(name="torio.lib._torio_ffmpeg", sources=[]),
                ]
            )
        else:
            modules.extend(
                [
                    Extension(name="torio.lib.libtorio_ffmpeg4", sources=[]),
                    Extension(name="torio.lib._torio_ffmpeg4", sources=[]),
                    Extension(name="torio.lib.libtorio_ffmpeg5", sources=[]),
                    Extension(name="torio.lib._torio_ffmpeg5", sources=[]),
                    Extension(name="torio.lib.libtorio_ffmpeg6", sources=[]),
                    Extension(name="torio.lib._torio_ffmpeg6", sources=[]),
                ]
            )
    return modules


# Based off of
# https://github.com/pybind/cmake_example/blob/580c5fd29d4651db99d8874714b07c0c49a53f8a/setup.py
class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.") from None
        super().run()

    def build_extension(self, ext):
        # Since two library files (libtorchaudio and _torchaudio) need to be
        # recognized by setuptools, we instantiate `Extension` twice. (see `get_ext_modules`)
        # This leads to the situation where this `build_extension` method is called twice.
        # However, the following `cmake` command will build all of them at the same time,
        # so, we do not need to perform `cmake` twice.
        # Therefore we call `cmake` only for `torchaudio._torchaudio`.
        if ext.name != "torchaudio.lib.libtorchaudio":
            return

        # Note:
        # the last part "lib" does not really matter. We want to get the full path of
        # the root build directory. Passing "torchaudio" will be interpreted as
        # `torchaudio.[so|dylib|pyd]`, so we need something `torchaudio.foo`, that is
        # interpreted as `torchaudio/foo.so` then use dirname to get the `torchaudio`
        # directory.
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath("foo")))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            f"-DPython_INCLUDE_DIR={distutils.sysconfig.get_python_inc()}",
            f"-DBUILD_CPP_TEST={'ON' if _BUILD_CPP_TEST else 'OFF'}",
            f"-DBUILD_SOX:BOOL={'ON' if _BUILD_SOX else 'OFF'}",
            f"-DBUILD_RIR:BOOL={'ON' if _BUILD_RIR else 'OFF'}",
            f"-DBUILD_RNNT:BOOL={'ON' if _BUILD_RNNT else 'OFF'}",
            f"-DBUILD_ALIGN:BOOL={'ON' if _BUILD_ALIGN else 'OFF'}",
            f"-DBUILD_CUDA_CTC_DECODER:BOOL={'ON' if _BUILD_CUDA_CTC_DECODER else 'OFF'}",
            "-DBUILD_TORCHAUDIO_PYTHON_EXTENSION:BOOL=ON",
            "-DBUILD_TORIO_PYTHON_EXTENSION:BOOL=ON",
            f"-DUSE_ROCM:BOOL={'ON' if _USE_ROCM else 'OFF'}",
            f"-DUSE_CUDA:BOOL={'ON' if _USE_CUDA else 'OFF'}",
            f"-DUSE_OPENMP:BOOL={'ON' if _USE_OPENMP else 'OFF'}",
            f"-DUSE_FFMPEG:BOOL={'ON' if _USE_FFMPEG else 'OFF'}",
        ]
        build_args = ["--target", "install"]
        # Pass CUDA architecture to cmake
        if _TORCH_CUDA_ARCH_LIST is not None:
            # Convert MAJOR.MINOR[+PTX] list to new style one
            # defined at https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
            _arches = _TORCH_CUDA_ARCH_LIST.replace(".", "").replace(" ", ";").split(";")
            _arches = [arch[:-4] if arch.endswith("+PTX") else f"{arch}-real" for arch in _arches]
            cmake_args += [f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(_arches)}"]

        if platform.system() != "Windows" and CUDA_HOME is not None:
            cmake_args += [f"-DCMAKE_CUDA_COMPILER='{CUDA_HOME}/bin/nvcc'"]
            cmake_args += [f"-DCUDA_TOOLKIT_ROOT_DIR='{CUDA_HOME}'"]

        # Default to Ninja
        if "CMAKE_GENERATOR" not in os.environ or platform.system() == "Windows":
            cmake_args += ["-GNinja"]
        if platform.system() == "Windows":
            import sys

            python_version = sys.version_info

            cxx_compiler = os.environ.get('CXX', 'cl')
            c_compiler = os.environ.get('CC', 'cl')

            cmake_args += [
                f"-DCMAKE_C_COMPILER={c_compiler}",
                f"-DCMAKE_CXX_COMPILER={cxx_compiler}",
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

        subprocess.check_call(["cmake", str(_ROOT_DIR)] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

    def get_ext_filename(self, fullname):
        ext_filename = super().get_ext_filename(fullname)
        ext_filename_parts = ext_filename.split(".")
        without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
        ext_filename = ".".join(without_abi)
        return ext_filename
