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
_TP_INSTALL_DIR = _TP_BASE_DIR / 'build'


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
    return [
        str(_ROOT_DIR),
        str(_TP_INSTALL_DIR / 'include'),
    ]


def _get_library_dirs():
    return [
        str(_TP_INSTALL_DIR / 'lib'),
    ]


def _get_libraries():
    # NOTE: The order of the library listed bellow matters.
    #
    # (the most important thing is that dependencies come after a library
    # e.g., sox comes first)
    return [
        'sox',
        'mad',
        'FLAC',
        'mp3lame',
    ]


def _build_codecs():
    subprocess.run(
        args=[str(_THIS_DIR / 'build_third_party.sh')],
        check=True,
    )


def _configure_third_party():
    _build_codecs()


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
            library_dirs=_get_library_dirs(),
            extra_compile_args=_get_eca(debug),
            extra_link_args=_get_ela(debug),
        ),
    ]


class BuildExtension(TorchBuildExtension):
    def build_extension(self, ext):
        if ext.name == _EXT_NAME:
            _configure_third_party()
        super().build_extension(ext)
