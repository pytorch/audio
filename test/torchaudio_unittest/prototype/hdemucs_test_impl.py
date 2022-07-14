from typing import List

import torch
from parameterized import parameterized
from torchaudio.prototype.models.hdemucs import (
    _HDecLayer,
    _HEncLayer,
    HDemucs,
    hdemucs_high,
    hdemucs_low,
    hdemucs_medium,
)
from torchaudio_unittest.common_utils import skipIfNoModule, TestBaseMixin, TorchaudioTestCase


def _get_hdemucs_model(sources: List[str], n_fft: int = 4096, depth: int = 6, sample_rate: int = 44100):
    return HDemucs(sources, nfft=n_fft, depth=depth, sample_rate=sample_rate)


def _get_inputs(duration: int, channels: int, batch_size: int, sample_rate: int):
    sample = torch.rand(batch_size, channels, duration * sample_rate, dtype=torch.float32)
    return sample


class HDemucsTests(TestBaseMixin):
    @parameterized.expand(
        [
            (["bass", "drums", "other", "vocals"],),
            (["bass", "drums", "other"],),
            (["bass", "vocals"],),
            (["vocals"],),
        ]
    )
    def test_hdemucs_output_shape(self, sources):
        r"""Feed tensors with specific shape to HDemucs and validate
        that it outputs with a tensor with expected shape.
        """
        batch_size = 1
        duration = 10
        channels = 2
        sample_rate = 44100

        model = _get_hdemucs_model(sources).to(self.device).eval()
        inputs = _get_inputs(duration, channels, batch_size, sample_rate)

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

    Test methods in this test suite will check to
    1. Assure correctness in factory functions, comparing with original hybrid demucs
    """

    def _get_original_model(self, sources: List[str], nfft: int, depth: int):
        from demucs import hdemucs as original

        original = original.HDemucs(sources, nfft=nfft, depth=depth)
        return original

    @parameterized.expand(
        [
            (["bass", "drums", "other", "vocals"],),
            (["bass", "drums", "other"],),
            (["bass", "vocals"],),
            (["vocals"],),
        ]
    )
    def test_import_recreate_low_model_test(self, sources):
        sample_rate = 8000
        nfft = 1024
        depth = 5
        duration = 10
        channels = 2
        batch_size = 1

        torch.random.manual_seed(0)
        factory_hdemucs = hdemucs_low(sources, sample_rate=sample_rate).eval()
        self._assert_equal_models(batch_size, channels, depth, duration, factory_hdemucs, nfft, sample_rate, sources)

    @parameterized.expand(
        [
            (["bass", "drums", "other", "vocals"],),
            (["bass", "drums", "other"],),
            (["bass", "vocals"],),
            (["vocals"],),
        ]
    )
    def test_import_recreate_medium_model_test(self, sources):
        sample_rate = 16000
        nfft = 2048
        depth = 6
        duration = 10
        channels = 2
        batch_size = 1

        torch.random.manual_seed(0)
        factory_hdemucs = hdemucs_medium(sources, sample_rate=sample_rate).eval()
        self._assert_equal_models(batch_size, channels, depth, duration, factory_hdemucs, nfft, sample_rate, sources)

    @parameterized.expand(
        [
            (["bass", "drums", "other", "vocals"],),
            (["bass", "drums", "other"],),
            (["bass", "vocals"],),
            (["vocals"],),
        ]
    )
    def test_import_recreate_high_model_test(self, sources):
        sample_rate = 44100
        nfft = 4096
        depth = 6
        duration = 10
        channels = 2
        batch_size = 1

        torch.random.manual_seed(0)
        factory_hdemucs = hdemucs_high(sources, sample_rate=sample_rate).eval()
        self._assert_equal_models(batch_size, channels, depth, duration, factory_hdemucs, nfft, sample_rate, sources)

    def _assert_equal_models(self, batch_size, channels, depth, duration, factory_hdemucs, nfft, sample_rate, sources):
        torch.random.manual_seed(0)
        original_hdemucs = self._get_original_model(sources, nfft, depth).eval()
        inputs = _get_inputs(duration, channels=channels, batch_size=batch_size, sample_rate=sample_rate)
        factory_output = factory_hdemucs(inputs)
        original_output = original_hdemucs(inputs)
        self.assertEqual(original_output, factory_output)
