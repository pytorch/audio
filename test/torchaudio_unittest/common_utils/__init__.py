from .autograd_utils import use_deterministic_algorithms
from .case_utils import (
    disabledInCI,
    HttpServerMixin,
    PytorchTestCase,
    skipIfCudaSmallMemory,
    skipIfNoAudioDevice,
    skipIfNoCtcDecoder,
    skipIfNoCuCtcDecoder,
    skipIfNoCuda,
    skipIfNoExec,
    skipIfNoFFmpeg,
    skipIfNoHWAccel,
    skipIfNoMacOS,
    skipIfNoModule,
    skipIfNoQengine,
    skipIfNoRIR,
    skipIfPy310,
    skipIfRocm,
    TempDirMixin,
    TestBaseMixin,
    TorchaudioTestCase,
    zip_equal,
)
from .data_utils import get_asset_path, get_sinusoid, get_spectrogram, get_whitenoise
from .func_utils import torch_script
from .parameterized_utils import load_params, nested_params
from .wav_utils import get_wav_data, load_wav, normalize_wav, save_wav
import pytest

class RequestMixin:
    """
    Adds the `self.request` attribute to a test instance, which uniquely identifies the test.
    It looks like, e.g.:
    test/torchaudio_unittest/functional/librosa_compatibility_cpu_test.py__TestFunctionalCPU__test_create_mel_fb_13
    """

    @pytest.fixture(autouse=True)
    def inject_request(self, request):
        self.request = request.node.nodeid.replace(":", "_").replace("_cpu_", "_").replace("_cuda_", "_")

__all__ = [
    "get_asset_path",
    "get_whitenoise",
    "get_sinusoid",
    "get_spectrogram",
    "TempDirMixin",
    "HttpServerMixin",
    "TestBaseMixin",
    "PytorchTestCase",
    "RequestMixin",
    "TorchaudioTestCase",
    "skipIfNoAudioDevice",
    "skipIfNoCtcDecoder",
    "skipIfNoCuCtcDecoder",
    "skipIfNoCuda",
    "skipIfCudaSmallMemory",
    "skipIfNoExec",
    "skipIfNoMacOS",
    "skipIfNoModule",
    "skipIfNoRIR",
    "skipIfRocm",
    "skipIfNoQengine",
    "skipIfNoFFmpeg",
    "skipIfNoHWAccel",
    "skipIfPy310",
    "disabledInCI",
    "get_wav_data",
    "normalize_wav",
    "load_wav",
    "save_wav",
    "load_params",
    "nested_params",
    "torch_script",
    "use_deterministic_algorithms",
    "zip_equal",
]
