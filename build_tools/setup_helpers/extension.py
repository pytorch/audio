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
_TP_TRANSDUCER_BASE_DIR = _ROOT_DIR / 'third_party' / 'warp_transducer'
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


def _build_third_party(base_dir):
    build_dir = str(base_dir / 'build')
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


def _get_ext_rnnt(debug):
    import torch
    # from torch.utils.cpp_extension import CppExtension

    extra_compile_args = ['-fPIC']
    extra_compile_args += ['-std=c++14']
    base_path = _TP_TRANSDUCER_BASE_DIR
    default_warp_rnnt_path = base_path / "build"

    if torch.cuda.is_available():

        if "CUDA_HOME" not in os.environ:
            raise RuntimeError("Please specify the environment variable CUDA_HOME")

        enable_gpu = True

    else:
        print("Torch was not built with CUDA support, not building GPU extensions.")
        enable_gpu = False

    if enable_gpu:
        extra_compile_args += ['-DWARPRNNT_ENABLE_GPU']

    if "WARP_RNNT_PATH" in os.environ:
        warp_rnnt_path = os.environ["WARP_RNNT_PATH"]
    else:
        warp_rnnt_path = default_warp_rnnt_path
    include_dirs = [os.path.realpath(os.path.join(base_path, 'include'))]

    return CppExtension(
        name='_warp_transducer',
        sources=[os.path.realpath(base_path / 'pytorch_binding' / 'src' / 'binding.cpp')],
        include_dirs=include_dirs,
        library_dirs=[os.path.realpath(warp_rnnt_path)],
        libraries=['warprnnt'],
        extra_link_args=['-Wl,-rpath,' + os.path.realpath(warp_rnnt_path)],
        extra_compile_args=extra_compile_args
    )


_EXT_NAME = 'torchaudio._torchaudio'


def get_ext_modules(debug=False):
    if platform.system() == 'Windows':
        return None
    return [
        _get_ext(debug),
        _get_ext_rnnt(debug),
    ]


class BuildExtension(TorchBuildExtension):
    def build_extension(self, ext):
        if ext.name == _EXT_NAME and _BUILD_SOX:
            _build_third_party(_TP_BASE_DIR)
        if ext.name == "_warp_transducer":
            _build_third_party(_TP_TRANSDUCER_BASE_DIR)
        super().build_extension(ext)
