import os
import platform
import subprocess
import torch

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
_TP_TRANSDUCER_BASE_DIR = _ROOT_DIR / 'third_party' / 'warp_transducer'
_TP_TRANSDUCER_MODULE_DIR = _ROOT_DIR / 'third_party' / 'warp_transducer' / 'submodule'
_TP_INSTALL_DIR = _TP_BASE_DIR / 'install'


def _get_build_option(var):
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


_BUILD_SOX = _get_build_option("BUILD_SOX")
_BUILD_CUDA_WARP_TRANSDUCER = _get_build_option("BUILD_CUDA_WT")


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


def _build_third_party(base_dir, target=['..'], options=[]):
    print(f"Building third party library in {base_dir}...")
    build_dir = str(base_dir / 'build')
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(
        args=['cmake'] + target + options,
        cwd=build_dir,
        check=True,
    )
    subprocess.run(
        args=['cmake', '--build', '.'],
        cwd=build_dir,
        check=True,
    )


def _get_ext(debug):
    return CppExtension(
        _EXT_NAME,
        _get_srcs(),
        libraries=_get_libraries(),
        include_dirs=_get_include_dirs(),
        extra_compile_args=_get_eca(debug),
        extra_objects=_get_extra_objects(),
        extra_link_args=_get_ela(debug),
    )


def _get_ext_transducer(debug):
    extra_compile_args = [
        '-fPIC',
        '-std=c++14',
    ]

    if _BUILD_CUDA_WARP_TRANSDUCER and torch.cuda.is_available():
        print("Building GPU extensions for warp_transudcer.")
        if "CUDA_HOME" not in os.environ:
            raise RuntimeError("Please specify the environment variable CUDA_HOME.")
        extra_compile_args += ['-DWARPRNNT_ENABLE_GPU']
    else:
        print("Not building GPU extensions for warp_transudcer.")

    librairies = ['warprnnt']
    build_path = _TP_TRANSDUCER_MODULE_DIR / 'build'
    include_path = _TP_TRANSDUCER_MODULE_DIR / 'include'
    source_path = _TP_TRANSDUCER_BASE_DIR / 'binding.cpp'

    return CppExtension(
        name='_warp_transducer',
        sources=[os.path.realpath(source_path)],
        libraries=librairies,
        include_dirs=[os.path.realpath(include_path)],
        library_dirs=[os.path.realpath(build_path)],
        extra_compile_args=extra_compile_args,
        extra_objects=[str(build_path / f'lib{l}.a') for l in librairies],
        extra_link_args=['-Wl,-rpath,' + os.path.realpath(build_path)],
    )


_EXT_NAME = 'torchaudio._torchaudio'


def get_ext_modules(debug=False):
    if platform.system() == 'Windows':
        return None
    return [
        _get_ext(debug),
        _get_ext_transducer(debug),
    ]


class BuildExtension(TorchBuildExtension):
    def build_extension(self, ext):
        if ext.name == _EXT_NAME and _BUILD_SOX:
            _build_third_party(_TP_BASE_DIR)
        if ext.name == "_warp_transducer":
            # TODO Support OMP on MacOS
            _build_third_party(
                _TP_TRANSDUCER_MODULE_DIR,
                target=[str(_TP_TRANSDUCER_BASE_DIR)],
                options=[
                    "-DWITH_OMP=OFF",
                    "-DWITH_GPU=ON" if _BUILD_CUDA_WARP_TRANSDUCER else "-DWITH_GPU=OFF",
                ]
            )
        super().build_extension(ext)
