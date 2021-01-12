import os
import platform
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import (
    CppExtension,
    BuildExtension as TorchBuildExtension
)

__all__ = [
    'get_ext_modules',
    'BuildExtension',
]

_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()
_CSRC_DIR = _ROOT_DIR / 'torchaudio' / 'csrc'
_TP_BASE_DIR = _ROOT_DIR / 'third_party'
_TP_INSTALL_DIR = _TP_BASE_DIR / 'install'


def _get_build(var):
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


_BUILD_SOX = _get_build("BUILD_SOX")
_BUILD_TRANSDUCER = _get_build("BUILD_TRANSDUCER")


def _get_eca(debug):
    eca = []
    if debug:
        eca += ["-O0", "-g"]
    else:
        eca += ["-O3"]
    if _BUILD_TRANSDUCER:
        eca += ['-DBUILD_TRANSDUCER']
    return eca


def _get_ela(debug):
    ela = []
    if debug:
        if platform.system() == "Windows":
            ela += ["/DEBUG:FULL"]
        else:
            ela += ["-O0", "-g"]
    else:
        ela += ["-O3"]
    return ela


def _get_srcs():
    srcs = [_CSRC_DIR / 'pybind.cpp']
    srcs += list(_CSRC_DIR.glob('sox/**/*.cpp'))
    if _BUILD_TRANSDUCER:
        srcs += [_CSRC_DIR / 'transducer.cpp']
    return [str(path) for path in srcs]


def _get_include_dirs():
    dirs = [
        str(_ROOT_DIR),
    ]
    if _BUILD_SOX or _BUILD_TRANSDUCER:
        dirs.append(str(_TP_INSTALL_DIR / 'include'))
    return dirs


def _get_extra_objects():
    libs = []
    if _BUILD_SOX:
        # NOTE: The order of the library listed bellow matters.
        #
        # (the most important thing is that dependencies come after a library
        # e.g., sox comes first, flac/vorbis comes before ogg, and
        # vorbisenc/vorbisfile comes before vorbis
        libs += [
            'libsox.a',
            'libmad.a',
            'libFLAC.a',
            'libmp3lame.a',
            'libopusfile.a',
            'libopus.a',
            'libvorbisenc.a',
            'libvorbisfile.a',
            'libvorbis.a',
            'libogg.a',
            'libopencore-amrnb.a',
            'libopencore-amrwb.a',
        ]
    if _BUILD_TRANSDUCER:
        libs += ['libwarprnnt.a']

    return [str(_TP_INSTALL_DIR / 'lib' / lib) for lib in libs]


def _get_libraries():
    return [] if _BUILD_SOX else ['sox']


def _get_cxx11_abi():
    try:
        value = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except ImportError:
        value = 0
    return f'-D_GLIBCXX_USE_CXX11_ABI={value}'


def _build_third_party(base_build_dir):
    build_dir = os.path.join(base_build_dir, 'third_party')
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(
        args=[
            'cmake',
            f"-DCMAKE_CXX_FLAGS='{_get_cxx11_abi()}'",
            '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON',
            f'-DCMAKE_INSTALL_PREFIX={_TP_INSTALL_DIR}',
            f'-DBUILD_SOX={"ON" if _BUILD_SOX else "OFF"}',
            f'-DBUILD_TRANSDUCER={"ON" if _BUILD_TRANSDUCER else "OFF"}',
            f'{_TP_BASE_DIR}'],
        cwd=build_dir,
        check=True,
    )
    command = ['cmake', '--build', '.']
    if _BUILD_TRANSDUCER:
        command += ['--target', 'install']
    subprocess.run(
        args=command,
        cwd=build_dir,
        check=True,
    )


_EXT_NAME = 'torchaudio._torchaudio'


def get_ext_modules(debug=False):
    if platform.system() == 'Windows':
        return None
    return [
        CppExtension(
            _EXT_NAME,
            _get_srcs(),
            libraries=_get_libraries(),
            include_dirs=_get_include_dirs(),
            extra_compile_args=_get_eca(debug),
            extra_objects=_get_extra_objects(),
            extra_link_args=_get_ela(debug),
        ),
    ]


class BuildExtension(TorchBuildExtension):
    def build_extension(self, ext):
        if ext.name == _EXT_NAME and _BUILD_SOX:
            _build_third_party(self.build_temp)
        super().build_extension(ext)
