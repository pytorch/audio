import warnings
import importlib
from collections import namedtuple

import torch
from torchaudio._internal import module_utils as _mod_utils


def _init_extension():
    ext = 'torchaudio._torchaudio'
    if _mod_utils.is_module_available(ext):
        _init_script_module(ext)
    else:
        warnings.warn('torchaudio C++ extension is not available.')
        _init_dummy_module()


def _init_script_module(module):
    path = importlib.util.find_spec(module).origin
    torch.classes.load_library(path)
    torch.ops.load_library(path)


def _init_dummy_module():
    class SignalInfo:
        """Data class for audio format information

        Used when torchaudio C++ extension is not available for annotating
        sox_io backend functions so that torchaudio is still importable
        without extension.
        This class has to implement the same interface as C++ equivalent.
        """
        def __init__(self, sample_rate: int, num_channels: int, num_samples: int):
            self.sample_rate = sample_rate
            self.num_channels = num_channels
            self.num_samples = num_samples

        def get_sample_rate(self):
            return self.sample_rate

        def get_num_channels(self):
            return self.num_channels

        def get_num_samples(self):
            return self.num_samples

    DummyModule = namedtuple('torchaudio', ['SignalInfo'])
    module = DummyModule(SignalInfo)
    setattr(torch.classes, 'torchaudio', module)
