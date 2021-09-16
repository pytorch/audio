"""Test numerical consistency among single input and batched input."""
import torch
import torchaudio
from parameterized import parameterized

from torchaudio_unittest import common_utils


class TestTransforms(common_utils.TorchaudioTestCase):
    backend = 'default'

    """Test suite for classes defined in `transforms` module"""
    def test_batch_AmplitudeToDB(self):
        spec = torch.rand((3, 2, 6, 201))

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.AmplitudeToDB()(spec[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.AmplitudeToDB()(spec)

        self.assertEqual(computed, expected)

    def test_batch_Resample(self):
        waveform = torch.randn(3, 2, 2786)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.Resample()(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.Resample()(waveform)

        self.assertEqual(computed, expected)

    def test_batch_MelScale(self):
        specgram = torch.randn(3, 2, 201, 256)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.MelScale()(specgram[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.MelScale()(specgram)

        # shape = (3, 2, 128, 256)
        self.assertEqual(computed, expected)

    def test_batch_InverseMelScale(self):
        n_mels = 32
        n_stft = 5
        mel_spec = torch.randn(3, 2, n_mels, 32) ** 2

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.InverseMelScale(n_stft, n_mels)(mel_spec[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.InverseMelScale(n_stft, n_mels)(mel_spec)

        # shape = (3, 2, n_mels, 32)

        # Because InverseMelScale runs SGD on randomly initialized values so they do not yield
        # exactly same result. For this reason, tolerance is very relaxed here.
        self.assertEqual(computed, expected, atol=1.0, rtol=1e-5)

    def test_batch_compute_deltas(self):
        specgram = torch.randn(3, 2, 31, 2786)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.ComputeDeltas()(specgram[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.ComputeDeltas()(specgram)

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

    def test_batch_mulaw(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.MuLawEncoding()(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.MuLawEncoding()(waveform)

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

        # Single then transform then batch
        expected_decoded = []
        for i in range(3):
            expected_decoded.append(torchaudio.transforms.MuLawDecoding()(expected[i]))
        expected_decoded = torch.stack(expected_decoded)

        # Batch then transform
        computed_decoded = torchaudio.transforms.MuLawDecoding()(computed)

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed_decoded, expected_decoded)

    def test_batch_spectrogram(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.Spectrogram()(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.Spectrogram()(waveform)
        self.assertEqual(computed, expected)

    def test_batch_inverse_spectrogram(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        transform = torchaudio.transforms.Spectrogram(power=None)(waveform)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.InverseSpectrogram()(transform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.InverseSpectrogram()(transform)
        self.assertEqual(computed, expected)

    def test_batch_melspectrogram(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.MelSpectrogram()(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.MelSpectrogram()(waveform)
        self.assertEqual(computed, expected)

    def test_batch_mfcc(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.MFCC()(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.MFCC()(waveform)
        self.assertEqual(computed, expected, atol=1e-4, rtol=1e-5)

    def test_batch_lfcc(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.LFCC()(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.LFCC()(waveform)
        self.assertEqual(computed, expected, atol=1e-4, rtol=1e-5)

    @parameterized.expand([(True, ), (False, )])
    def test_batch_TimeStretch(self, test_pseudo_complex):
        rate = 2
        num_freq = 1025
        num_frames = 400
        batch = 3

        spec = torch.randn(batch, num_freq, num_frames, dtype=torch.complex64)
        if test_pseudo_complex:
            spec = torch.view_as_real(spec)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.TimeStretch(
                fixed_rate=rate,
                n_freq=num_freq,
                hop_length=512,)(spec[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.TimeStretch(
            fixed_rate=rate,
            n_freq=num_freq,
            hop_length=512,
        )(spec)

        self.assertEqual(computed, expected, atol=1e-5, rtol=1e-5)

    def test_batch_Fade(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        fade_in_len = 3000
        fade_out_len = 3000

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.Fade(fade_in_len, fade_out_len)(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.Fade(fade_in_len, fade_out_len)(waveform)
        self.assertEqual(computed, expected)

    def test_batch_Vol(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.Vol(gain=1.1)(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.Vol(gain=1.1)(waveform)
        self.assertEqual(computed, expected)

    def test_batch_spectral_centroid(self):
        sample_rate = 44100
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.SpectralCentroid(sample_rate)(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.SpectralCentroid(sample_rate)(waveform)
        self.assertEqual(computed, expected)

    def test_batch_pitch_shift(self):
        sample_rate = 8000
        n_steps = -2
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)

        # Single then transform then batch
        expected = []
        for i in range(3):
            expected.append(torchaudio.transforms.PitchShift(sample_rate, n_steps, n_fft=400)(waveform[i]))
        expected = torch.stack(expected)

        # Batch then transform
        computed = torchaudio.transforms.PitchShift(sample_rate, n_steps, n_fft=400)(waveform)
        self.assertEqual(computed, expected)
