import torch
from parameterized import parameterized
from torchaudio.prototype.models import hifigan_vocoder, hifigan_vocoder_v1, hifigan_vocoder_v2, hifigan_vocoder_v3
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script

from .original.env import AttrDict
from .original.meldataset import mel_spectrogram as ref_mel_spectrogram
from .original.models import Generator


class HiFiGANTestImpl(TestBaseMixin):
    def _get_model_config(self):
        return {
            "upsample_rates": (8, 8, 4),
            "upsample_kernel_sizes": (16, 16, 8),
            "upsample_initial_channel": 256,
            "resblock_kernel_sizes": (3, 5, 7),
            "resblock_dilation_sizes": ((1, 2), (2, 6), (3, 12)),
            "resblock_type": 2,
            "in_channels": 80,
            "lrelu_slope": 0.1,
        }

    def _get_input_config(self):
        model_config = self._get_model_config()
        return {
            "batch_size": 7,
            "in_channels": model_config["in_channels"],
            "time_length": 10,
        }

    def _get_model(self):
        return hifigan_vocoder(**self._get_model_config()).to(device=self.device, dtype=self.dtype).eval()

    def _get_inputs(self):
        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        time_length = input_config["time_length"]
        in_channels = input_config["in_channels"]

        input = torch.rand(batch_size, in_channels, time_length).to(device=self.device, dtype=self.dtype)
        return input

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(31)

    @parameterized.expand([(hifigan_vocoder_v1,), (hifigan_vocoder_v2,), (hifigan_vocoder_v3,)])
    def test_smoke(self, factory_func):
        r"""Verify that model architectures V1, V2, V3 can be constructed and applied on inputs"""
        model = factory_func().to(device=self.device, dtype=self.dtype)
        input = self._get_inputs()
        model(input)

    def test_torchscript_consistency_forward(self):
        r"""Verify that scripting the model does not change the behavior of method `forward`."""
        inputs = self._get_inputs()

        original_model = self._get_model()
        scripted_model = torch_script(original_model).eval()

        for _ in range(2):
            ref_out = original_model(inputs)
            scripted_out = scripted_model(inputs)
            self.assertEqual(ref_out, scripted_out)

    def test_output_shape_forward(self):
        r"""Check that method `forward` produces correctly-shaped outputs."""
        input_config = self._get_input_config()
        model_config = self._get_model_config()

        batch_size = input_config["batch_size"]
        time_length = input_config["time_length"]

        inputs = self._get_inputs()
        model = self._get_model()

        total_upsample_rate = 1  # Use loop instead of math.prod for compatibility with Python 3.7
        for upsample_rate in model_config["upsample_rates"]:
            total_upsample_rate *= upsample_rate

        for _ in range(2):
            out = model(inputs)
            self.assertEqual(
                (batch_size, 1, total_upsample_rate * time_length),
                out.shape,
            )

    def test_original_implementation_match(self):
        r"""Check that output of our implementation matches the original one."""
        model_config = self._get_model_config()
        model_config = AttrDict(model_config)
        model_config.resblock = "1" if model_config.resblock_type == 1 else "2"
        model_ref = Generator(model_config).to(device=self.device, dtype=self.dtype)
        model_ref.remove_weight_norm()

        inputs = self._get_inputs()
        model = self._get_model()
        model.load_state_dict(model_ref.state_dict())

        ref_output = model_ref(inputs)
        output = model(inputs)
        self.assertEqual(ref_output, output)

    def test_mel_transform(self):
        """Check that HIFIGAN_VOCODER_V3_LJSPEECH.get_mel_transform generates the same mel spectrogram as the original
        HiFiGAN implementation when applied on a synthetic waveform.
        There seems to be no way to change dtype in the original implmentation, so we feed in the waveform with the
        default dtype and cast the output before comparison.
        """
        synth_waveform = torch.rand(1, 1000).to(device=self.device)

        # Get HiFiGAN-compatible transformation from waveform to mel spectrogram
        self.mel_spectrogram = HIFIGAN_VOCODER_V3_LJSPEECH.get_mel_transform().to(dtype=self.dtype, device=self.device)
        mel_spec = self.mel_spectrogram(synth_waveform.to(dtype=self.dtype))
        # Generate mel spectrogram with original implementation
        ref_mel_spec = ref_mel_spectrogram(
            synth_waveform,
            n_fft=self.mel_spectrogram.n_fft,
            num_mels=self.mel_spectrogram.n_mels,
            sampling_rate=self.mel_spectrogram.sample_rate,
            hop_size=self.mel_spectrogram.hop_size,
            win_size=self.mel_spectrogram.win_length,
            fmin=self.mel_spectrogram.f_min,
            fmax=self.mel_spectrogram.f_max,
        )
        self.assertEqual(ref_mel_spec.to(dtype=self.dtype), mel_spec, atol=1e-5, rtol=1e-5)
