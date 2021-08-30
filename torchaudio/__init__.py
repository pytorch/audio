from torchaudio._internal import module_utils as _mod_utils  # noqa: F401

if _mod_utils.is_module_available('torchaudio._torchaudio'):
    # Note this import has two purposes
    # 1. Make _torchaudio accessible by the other modules (regular import)
    # 2. Register torchaudio's custom ops bound via TorchScript
    #
    # For 2, normally function calls `torch.ops.load_library` and `torch.classes.load_library`
    # are used. However, in our cases, this is inconvenient and unnecessary.
    #
    # - Why inconvenient?
    # When torchaudio is deployed with `pex` format, all the files are deployed as a single zip
    # file, and the extension module is not present as a file with full path. Therefore it is not
    # possible to pass the path to library to `torch.[ops|classes].load_library` functions.
    #
    # - Why unnecessary?
    # When torchaudio extension module (C++ module) is available, it is assumed that
    # the extension contains both TorchScript-based binding and PyBind11-based binding.*
    # Under this assumption, simply performing `from torchaudio import _torchaudio` will load the
    # library which contains TorchScript-based binding as well, and the functions/classes bound
    # via TorchScript become accessible under `torch.ops` and `torch.classes`.
    #
    # *Note that this holds true even when these two bindings are split into two library files and
    # the library that contains PyBind11-based binding (`_torchaudio.so` in the following diagram)
    # depends on the other one (`libtorchaudio.so`), because when the process tries to load
    # `_torchaudio.so` it detects undefined symbols from `libtorchaudio.so` and will automatically
    # loads `libtorchaudio.so`. (given that the library is found in a search path)
    #
    # [libtorchaudio.so] <- [_torchaudio.so]
    #
    #
    from torchaudio import _torchaudio  # noqa
else:
    import warnings
    warnings.warn('torchaudio C++ extension is not available.')

from torchaudio import (
    compliance,
    datasets,
    functional,
    models,
    kaldi_io,
    utils,
    sox_effects,
    transforms,
)

from torchaudio.backend import (
    list_audio_backends,
    get_audio_backend,
    set_audio_backend,
)

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass

__all__ = [
    'compliance',
    'datasets',
    'functional',
    'models',
    'kaldi_io',
    'utils',
    'sox_effects',
    'transforms',
    'list_audio_backends',
    'get_audio_backend',
    'set_audio_backend',
]
