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
class TestDemucsIntegration(TorchaudioTestCase):
    """Test the process of importing the models from demucs.

    Test methods in this test suite check the following things
    1. Models loaded with demucs can be imported.
    2. The same model can be recreated without demucs.
    """

    def _get_original_model(self, sources: List[str], nfft: int, depth: int):
        from demucs import hdemucs as original

        old_model = original.HDemucs(sources, nfft=nfft, depth=depth)
        return old_model

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
        torch.random.manual_seed(0)
        model_hdemucs = HDemucs(sources, nfft=nfft, depth=depth, sample_rate=sample_rate).eval()
        torch.random.manual_seed(0)
        old_hdemucs = self._get_original_model(sources, nfft, depth).eval()

        inputs = _get_inputs(duration, channels=channels, batch_size=batch_size, sample_rate=sample_rate)

        factory_output = factory_hdemucs(inputs)
        model_output = model_hdemucs(inputs)
        old_output = old_hdemucs(inputs)

        self.assertEqual(model_output, factory_output)
        self.assertEqual(model_output, old_output)

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
        torch.random.manual_seed(0)
        model_hdemucs = HDemucs(sources, nfft=nfft, depth=depth, sample_rate=sample_rate).eval()
        torch.random.manual_seed(0)
        old_hdemucs = self._get_original_model(sources, nfft, depth).eval()

        inputs = _get_inputs(duration, channels=channels, batch_size=batch_size, sample_rate=sample_rate)

        factory_output = factory_hdemucs(inputs)
        model_output = model_hdemucs(inputs)
        old_output = old_hdemucs(inputs)

        self.assertEqual(model_output, factory_output)
        self.assertEqual(model_output, old_output)

    @parameterized.expand(
        [
            (["bass", "drums", "other", "vocals"],),
            (["bass", "drums", "other"],),
            (["bass", "vocals"],),
            (["vocals"],),
        ]
    )
    def test_import_recreate_medium_model_test(self, sources):
        sample_rate = 44100
        nfft = 4096
        depth = 6
        duration = 10
        channels = 2
        batch_size = 1

        torch.random.manual_seed(0)
        factory_hdemucs = hdemucs_high(sources, sample_rate=sample_rate).eval()
        torch.random.manual_seed(0)
        model_hdemucs = HDemucs(sources, nfft=nfft, depth=depth, sample_rate=sample_rate).eval()
        torch.random.manual_seed(0)
        old_hdemucs = self._get_original_model(sources, nfft, depth).eval()

        inputs = _get_inputs(duration, channels=channels, batch_size=batch_size, sample_rate=sample_rate)

        factory_output = factory_hdemucs(inputs)
        model_output = model_hdemucs(inputs)
        old_output = old_hdemucs(inputs)

        self.assertEqual(model_output, factory_output)
        self.assertEqual(model_output, old_output)
