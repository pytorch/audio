import os
import warnings
import importlib

import torch


def _init_extension():
    ext = 'torchaudio._torchaudio'
    lib_dir = os.path.join(os.path.dirname(__file__), '..')
    for item in os.listdir(lib_dir):
        print(item, flush=True)

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    spec = extfinder.find_spec("_torchaudio")
    if spec is None:
        warnings.warn('torchaudio C++ extension is not available.')
    else:
        path = spec.origin
        torch.classes.load_library(path)
        torch.ops.load_library(path)
        import torchaudio._torchaudio  # noqa
