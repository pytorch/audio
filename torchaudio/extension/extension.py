import importlib

import torch


def _init_extension():
    _init_script_module('torchaudio._torchaudio')
    import torchaudio._torchaudio  # noqa


def _init_script_module(module):
    path = importlib.util.find_spec(module).origin
    torch.classes.load_library(path)
    torch.ops.load_library(path)
