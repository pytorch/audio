import itertools
from typing import List

import torch
from parameterized import parameterized
from torchaudio.models._hdemucs import _HDecLayer, _HEncLayer, HDemucs, hdemucs_high, hdemucs_low
from torchaudio_unittest.common_utils import skipIfNoModule, TestBaseMixin, TorchaudioTestCase


def _get_hdemucs_model(sources: List[str], n_fft: int = 4096, depth: int = 6):
    return HDemucs(sources, nfft=n_fft, depth=depth)


def _get_inputs(sample_rate: int, device: torch.device, batch_size: int = 1, duration: int = 10, channels: int = 2):
    sample = torch.rand(batch_size, channels, duration * sample_rate, dtype=torch.float32, device=device)
    return sample


SOURCE_OPTIONS = [
    (["bass", "drums", "other", "vocals"],),
    (["bass", "drums", "other"],),
    (["bass", "vocals"],),
    (["vocals"],),
]

SOURCES_OUTPUT_CONFIG = parameterized.expand(SOURCE_OPTIONS)


class HDemucsTests(TestBaseMixin):
    @parameterized.expand(list(itertools.product(SOURCE_OPTIONS, [(1024, 5), (2048, 6), (4096, 6)])))
    def test_hdemucs_output_shape(self, sources, nfft_bundle):
        r"""Feed tensors with specific shape to HDemucs and validate
        that it outputs with a tensor with expected shape.
        """
        duration = 10
        channels = 2
        batch_size = 1
        sample_rate = 44100
        nfft = nfft_bundle[0]
        depth = nfft_bundle[1]

        model = _get_hdemucs_model(sources, nfft, depth).to(self.device).eval()
        inputs = _get_inputs(sample_rate, self.device, batch_size, duration, channels)

        split_sample = model(inputs)

        assert split_sample.shape == (batch_size, len(sources), channels, duration * sample_rate)

    def test_encoder_output_shape_frequency(self):
        r"""Feed tensors with specific shape to HDemucs Decoder and validate
        that it outputs with a tensor with expected shape for frequency domain.
        """
        batch_size = 1
        chin, chout = 4, 48
        f_bins = 2048
        t = 800
        stride = 4

        model = _HEncLayer(chin, chout).to(self.device).eval()

        x = torch.rand(batch_size, chin, f_bins, t, device=self.device, dtype=self.dtype)
        out = model(x)

        assert out.size() == (batch_size, chout, f_bins / stride, t)

    def test_decoder_output_shape_frequency(self):
        r"""Feed tensors with specific shape to HDemucs Decoder and validate
        that it outputs with a tensor with expected shape for frequency domain.
        """
        batch_size = 1
        chin, chout = 96, 48
        f_bins = 128
        t = 800
        stride = 4

        model = _HDecLayer(chin, chout).to(self.device).eval()

        x = torch.rand(batch_size, chin, f_bins, t, device=self.device, dtype=self.dtype)
        skip = torch.rand(batch_size, chin, f_bins, t, device=self.device, dtype=self.dtype)
        z, y = model(x, skip, t)

        assert z.size() == (batch_size, chout, f_bins * stride, t)
        assert y.size() == (batch_size, chin, f_bins, t)

    def test_encoder_output_shape_time(self):
        r"""Feed tensors with specific shape to HDemucs Decoder and validate
        that it outputs with a tensor with expected shape for time domain.
        """
        batch_size = 1
        chin, chout = 4, 48
        t = 800
        stride = 4

        model = _HEncLayer(chin, chout, freq=False).to(self.device).eval()

        x = torch.rand(batch_size, chin, t, device=self.device, dtype=self.dtype)
        out = model(x)

        assert out.size() == (batch_size, chout, t / stride)

    def test_decoder_output_shape_time(self):
        r"""Feed tensors with specific shape to HDemucs Decoder and validate
        that it outputs with a tensor with expected shape for time domain.
        """
        batch_size = 1
        chin, chout = 96, 48
        t = 800
        stride = 4

        model = _HDecLayer(chin, chout, freq=False).to(self.device).eval()

        x = torch.rand(batch_size, chin, t, device=self.device, dtype=self.dtype)
        skip = torch.rand(batch_size, chin, t, device=self.device, dtype=self.dtype)
        z, y = model(x, skip, t * stride)

        assert z.size() == (batch_size, chout, t * stride)
        assert y.size() == (batch_size, chin, t)


@skipIfNoModule("demucs")
class CompareHDemucsOriginal(TorchaudioTestCase):
    """Test the process of importing the models from demucs.

    Test methods in this test suite will check to assure correctness in factory functions,
    comparing with original hybrid demucs
    """

    def _get_original_model(self, sources: List[str], nfft: int, depth: int):
        from demucs import hdemucs as original

        original = original.HDemucs(sources, nfft=nfft, depth=depth)
        return original

    def _assert_equal_models(self, factory_hdemucs, depth, nfft, sample_rate, sources):
        torch.random.manual_seed(0)
        original_hdemucs = self._get_original_model(sources, nfft, depth).to(self.device).eval()
        inputs = _get_inputs(sample_rate=sample_rate, device=self.device)
        factory_output = factory_hdemucs(inputs)
        original_output = original_hdemucs(inputs)
        self.assertEqual(original_output, factory_output)

    @SOURCES_OUTPUT_CONFIG
    def test_import_recreate_low_model(self, sources):
        sample_rate = 8000
        nfft = 1024
        depth = 5

        torch.random.manual_seed(0)
        factory_hdemucs = hdemucs_low(sources).to(self.device).eval()
        self._assert_equal_models(factory_hdemucs, depth, nfft, sample_rate, sources)

    @SOURCES_OUTPUT_CONFIG
    def test_import_recreate_high_model(self, sources):
        sample_rate = 44100
        nfft = 4096
        depth = 6

        torch.random.manual_seed(0)
        factory_hdemucs = hdemucs_high(sources).to(self.device).eval()
        self._assert_equal_models(factory_hdemucs, depth, nfft, sample_rate, sources)
