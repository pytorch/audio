import torch
from parameterized import parameterized
from torchaudio.prototype.models.hdemucs import HDemucs, _HEncLayer, _HDecLayer
from torchaudio_unittest.common_utils import TestBaseMixin


def _get_hdemucs_model(sources):
    return HDemucs(sources)


class HDemucsTests(TestBaseMixin):
    def _get_inputs(self, duration: int, channels: int, batch_size: int, sample_rate: int):
        sample = torch.rand(batch_size, channels, duration * sample_rate, dtype=torch.float32, device=self.device)
        return sample

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
        inputs = self._get_inputs(duration, channels, batch_size, sample_rate)

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
