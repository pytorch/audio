import os
import platform
import subprocess
from pathlib import Path

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


_BUILD_SOX = _get_build_sox()


def _get_eca(debug):
    eca = []
    if debug:
        eca += ["-O0", "-g"]
    else:
        eca += ["-O3"]
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
    return [str(p) for p in _CSRC_DIR.glob('**/*.cpp')]


def _get_include_dirs():
    dirs = [
        str(_ROOT_DIR),
    ]
    if _BUILD_SOX:
        dirs.append(str(_TP_INSTALL_DIR / 'include'))
    return dirs


def _get_extra_objects():
    objs = []
    if _BUILD_SOX:
        # NOTE: The order of the library listed bellow matters.
        #
        # (the most important thing is that dependencies come after a library
        # e.g., sox comes first, flac/vorbis comes before ogg, and
        # vorbisenc/vorbisfile comes before vorbis
        libs = [
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
        for lib in libs:
            objs.append(str(_TP_INSTALL_DIR / 'lib' / lib))
    return objs


def _get_libraries():
    return [] if _BUILD_SOX else ['sox']


def _build_third_party():
    build_dir = str(_TP_BASE_DIR / 'build')
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(
        args=['cmake', '..'],
        cwd=build_dir,
        check=True,
    )
    subprocess.run(
        args=['cmake', '--build', '.'],
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
        _get_transducer_module(),
    ]


class BuildExtension(TorchBuildExtension):
    def build_extension(self, ext):
        if ext.name == _EXT_NAME and _BUILD_SOX:
            _build_third_party()
        if ext.name == _TRANSDUCER_NAME:
            _build_transducer()
        super().build_extension(ext)


_TRANSDUCER_NAME = '_warp_transducer'
_TP_TRANSDUCER_BASE_DIR = _ROOT_DIR / 'third_party' / 'warp_transducer'


def _build_transducer():
    build_dir = str(_TP_TRANSDUCER_BASE_DIR / 'submodule' / 'build')
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(
        args=['cmake', str(_TP_TRANSDUCER_BASE_DIR), '-DWITH_OMP=OFF'],
        cwd=build_dir,
        check=True,
    )
    subprocess.run(
        args=['cmake', '--build', '.'],
        cwd=build_dir,
        check=True,
    )


def _get_transducer_module():
    extra_compile_args = [
        '-fPIC',
        '-std=c++14',
    ]

    librairies = ['warprnnt']

    source_paths = [
        _TP_TRANSDUCER_BASE_DIR / 'binding.cpp',
        _TP_TRANSDUCER_BASE_DIR / 'submodule' / 'pytorch_binding' / 'src' / 'binding.cpp',
    ]
    build_path = _TP_TRANSDUCER_BASE_DIR / 'submodule' / 'build'
    include_path = _TP_TRANSDUCER_BASE_DIR / 'submodule' / 'include'

    return CppExtension(
        name=_TRANSDUCER_NAME,
        sources=[os.path.realpath(path) for path in source_paths],
        libraries=librairies,
        include_dirs=[os.path.realpath(include_path)],
        library_dirs=[os.path.realpath(build_path)],
        extra_compile_args=extra_compile_args,
        extra_objects=[str(build_path / f'lib{lib}.a') for lib in librairies],
        extra_link_args=['-Wl,-rpath,' + os.path.realpath(build_path)],
    )
