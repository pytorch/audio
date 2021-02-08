"""Test suites for jit-ability and its numerical compatibility"""

import torch
import torchaudio.transforms as T

from torchaudio_unittest import common_utils


class Transforms(common_utils.TestBaseMixin):
    """Implements test for Transforms that are performed for different devices"""
    def _assert_consistency(self, transform, tensor):
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        transform = transform.to(device=self.device, dtype=self.dtype)

        ts_transform = torch.jit.script(transform)
        output = transform(tensor)
        ts_output = ts_transform(tensor)
        self.assertEqual(ts_output, output)

    def test_Spectrogram(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.Spectrogram(), tensor)

    def test_GriffinLim(self):
        tensor = torch.rand((1, 201, 6))
        self._assert_consistency(T.GriffinLim(length=1000, rand_init=False), tensor)

    def test_AmplitudeToDB(self):
        spec = torch.rand((6, 201))
        self._assert_consistency(T.AmplitudeToDB(), spec)

    def test_MelScale(self):
        spec_f = torch.rand((1, 6, 201))
        self._assert_consistency(T.MelScale(), spec_f)

    def test_MelSpectrogram(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.MelSpectrogram(), tensor)

    def test_MFCC(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.MFCC(), tensor)

    def test_Resample(self):
        sr1, sr2 = 16000, 8000
        tensor = common_utils.get_whitenoise(sample_rate=sr1)
        self._assert_consistency(T.Resample(float(sr1), float(sr2)), tensor)

    def test_ComplexNorm(self):
        tensor = torch.rand((1, 2, 201, 2))
        self._assert_consistency(T.ComplexNorm(), tensor)

    def test_MuLawEncoding(self):
        tensor = common_utils.get_whitenoise()
        self._assert_consistency(T.MuLawEncoding(), tensor)

    def test_MuLawDecoding(self):
        tensor = torch.rand((1, 10))
        self._assert_consistency(T.MuLawDecoding(), tensor)

    def test_TimeStretch(self):
        n_freq = 400
        hop_length = 512
        fixed_rate = 1.3
        tensor = torch.rand((10, 2, n_freq, 10, 2))
        self._assert_consistency(
            T.TimeStretch(n_freq=n_freq, hop_length=hop_length, fixed_rate=fixed_rate),
            tensor,
        )

    def test_Fade(self):
        waveform = common_utils.get_whitenoise()
        fade_in_len = 3000
        fade_out_len = 3000
        self._assert_consistency(T.Fade(fade_in_len, fade_out_len), waveform)

    def test_FrequencyMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        self._assert_consistency(T.FrequencyMasking(freq_mask_param=60, iid_masks=False), tensor)

    def test_TimeMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        self._assert_consistency(T.TimeMasking(time_mask_param=30, iid_masks=False), tensor)

    def test_Vol(self):
        waveform = common_utils.get_whitenoise()
        self._assert_consistency(T.Vol(1.1), waveform)

    def test_SlidingWindowCmn(self):
        tensor = torch.rand((1000, 10))
        self._assert_consistency(T.SlidingWindowCmn(), tensor)

    def test_Vad(self):
        filepath = common_utils.get_asset_path("vad-go-mono-32000.wav")
        waveform, sample_rate = common_utils.load_wav(filepath)
        self._assert_consistency(T.Vad(sample_rate=sample_rate), waveform)

    def test_SpectralCentroid(self):
        sample_rate = 44100
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate)
        self._assert_consistency(T.SpectralCentroid(sample_rate=sample_rate), waveform)
