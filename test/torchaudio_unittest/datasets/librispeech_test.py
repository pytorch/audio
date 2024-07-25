from torchaudio.datasets import librispeech
from torchaudio_unittest.common_utils import TorchaudioTestCase
from torchaudio_unittest.datasets.librispeech_test_impl import LibriSpeechTestMixin


class TestLibriSpeech(LibriSpeechTestMixin, TorchaudioTestCase):
    librispeech_cls = librispeech.LIBRISPEECH
