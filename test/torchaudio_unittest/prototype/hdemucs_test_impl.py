import torch
from parameterized import parameterized
from torchaudio.prototype.models.hdemucs import HDemucs, HEncLayer, HDecLayer
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script

_SOURCE_SETS = [
    (["bass", "drums", "other", "vocals"],),
    (["bass", "drums", "other"],),
    (["bass", "vocals"],),
    (["vocals"],),
]


def _get_hdemucs_model(sources):
    return HDemucs(sources)


class TorchscriptConsistencyMixin(TestBaseMixin):
    r"""Mixin to provide easy access assert torchscript consistency"""

    def _assert_torchscript_consistency(self, model, tensors):
        ts_func = torch_script(model)

        torch.random.manual_seed(40)
        output = model(*tensors)

        torch.random.manual_seed(40)
        ts_output = ts_func(*tensors)

        self.assertEqual(ts_output, output)


class HDemucsEncoderTests(TorchscriptConsistencyMixin):
    def test_hdemucs_encoder_torchscript_consistency(self):
        r"""Validate the torchscript consistency of a Encoder."""
        chin, chout = 48, 96
        model = HEncLayer(chin, chout).to(self.device).eval()

        x = torch.rand(chin, chout, 189013, device=self.device, dtype=self.dtype)

        self._assert_torchscript_consistency(model, x)

    def test_encoder_output_shape(self):
        r"""Feed tensors with specific shape to HDemucs Decoder and validate
        that it outputs with a tensor with expected shape.
        """
        chin, chout = 4, 48
        model = HEncLayer(chin, chout).to(self.device).eval()

        x = torch.rand(1, chin, 2048, 739, device=self.device, dtype=self.dtype)
        out = model(x)

        assert out.size() == (1, chout, 512, 739)


class HDemucsDecoderTests(TorchscriptConsistencyMixin):
    def test_hdemucs_decoder_torchscript_consistency(self):
        r"""Validate the torchscript consistency of a Decoder."""
        chin, chout = 48, 96
        model = HDecLayer(chin, chout).to(self.device).eval()

        x = torch.rand(chin, chout, 189013, device=self.device, dtype=self.dtype)

        self._assert_torchscript_consistency(model, x)

    def test_decoder_output_shape(self):
        r"""Feed tensors with specific shape to HDemucs Decoder and validate
        that it outputs with a tensor with expected shape.
        """
        chin, chout = 96, 48
        model = HDecLayer(chin, chout).to(self.device).eval()

        x = torch.rand(1, chin, 128, 739, device=self.device, dtype=self.dtype)
        skip = torch.rand(1, chin, 128, 739, device=self.device, dtype=self.dtype)
        z, y = model(x, skip, 739)

        assert z.size() == (1, chout, 512, 739)
        assert y.size() == (1, chin, 128, 739)


class HDemucsTests(TorchscriptConsistencyMixin):
    def _get_inputs(self, length: int):
        sample = torch.rand(1, 2, length * 44100, dtype=torch.float32, device=self.device)
        return sample

    @parameterized.expand(_SOURCE_SETS)
    def test_hdemucs_torchscript_consistency(self, sources):
        r"""Validate the torchscript consistency of a HDemucs."""
        length = 10

        model = _get_hdemucs_model(sources).to(self.device).eval()
        inputs = self._get_inputs(length)

        self._assert_torchscript_consistency(model, inputs)

    @parameterized.expand(_SOURCE_SETS)
    def test_hdemucs_output_shape(self, sources):
        r"""Feed tensors with specific shape to HDemucs and validate
        that it outputs with a tensor with expected shape.
        """
        length = 10

        model = _get_hdemucs_model(sources).to(self.device).eval()
        inputs = self._get_inputs(length)

        split_sample = model(inputs)

        assert split_sample.shape == (1, len(sources), 2, length * 44100)
