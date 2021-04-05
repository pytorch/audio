"""Test numerical consistency among single input and batched input."""
import torch
import torchaudio

from torchaudio_unittest import common_utils


class TestTransforms(common_utils.TorchaudioTestCase):
    backend = 'default'

    """Test suite for classes defined in `transforms` module"""
    def test_batch_AmplitudeToDB(self):
        spec = torch.rand((2, 6, 201))

        # Single then transform then batch
        expected = torchaudio.transforms.AmplitudeToDB()(spec).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.AmplitudeToDB()(spec.repeat(3, 1, 1))

        self.assertEqual(computed, expected)

    def test_batch_Resample(self):
        waveform = torch.randn(2, 2786)

        # Single then transform then batch
        expected = torchaudio.transforms.Resample()(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Resample()(waveform.repeat(3, 1, 1))

        self.assertEqual(computed, expected)

    def test_batch_MelScale(self):
        specgram = torch.randn(2, 31, 2786)

        # Single then transform then batch
        expected = torchaudio.transforms.MelScale()(specgram).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MelScale()(specgram.repeat(3, 1, 1, 1))

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

    def test_batch_InverseMelScale(self):
        n_mels = 32
        n_stft = 5
        mel_spec = torch.randn(2, n_mels, 32) ** 2

        # Single then transform then batch
        expected = torchaudio.transforms.InverseMelScale(n_stft, n_mels)(mel_spec).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.InverseMelScale(n_stft, n_mels)(mel_spec.repeat(3, 1, 1, 1))

        # shape = (3, 2, n_mels, 32)

        # Because InverseMelScale runs SGD on randomly initialized values so they do not yield
        # exactly same result. For this reason, tolerance is very relaxed here.
        self.assertEqual(computed, expected, atol=1.0, rtol=1e-5)

    def test_batch_compute_deltas(self):
        specgram = torch.randn(2, 31, 2786)

        # Single then transform then batch
        expected = torchaudio.transforms.ComputeDeltas()(specgram).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.ComputeDeltas()(specgram.repeat(3, 1, 1, 1))

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

    def test_batch_mulaw(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        waveform_encoded = torchaudio.transforms.MuLawEncoding()(waveform)
        expected = waveform_encoded.unsqueeze(0).repeat(3, 1, 1)

        # Batch then transform
        waveform_batched = waveform.unsqueeze(0).repeat(3, 1, 1)
        computed = torchaudio.transforms.MuLawEncoding()(waveform_batched)

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

        # Single then transform then batch
        waveform_decoded = torchaudio.transforms.MuLawDecoding()(waveform_encoded)
        expected = waveform_decoded.unsqueeze(0).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MuLawDecoding()(computed)

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

    def test_batch_spectrogram(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        expected = torchaudio.transforms.Spectrogram()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Spectrogram()(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected)

    def test_batch_melspectrogram(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        expected = torchaudio.transforms.MelSpectrogram()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MelSpectrogram()(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected)

    def test_batch_mfcc(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)

        # Single then transform then batch
        expected = torchaudio.transforms.MFCC()(waveform).repeat(3, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.MFCC()(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected, atol=1e-4, rtol=1e-5)

    def test_batch_TimeStretch(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        rate = 2

        complex_specgrams = torch.view_as_real(
            torch.stft(
                input=waveform,
                n_fft=2048,
                hop_length=512,
                win_length=2048,
                window=torch.hann_window(2048),
                center=True,
                pad_mode='reflect',
                normalized=True,
                onesided=True,
                return_complex=True,
            )
        )

        # Single then transform then batch
        expected = torchaudio.transforms.TimeStretch(
            fixed_rate=rate,
            n_freq=1025,
            hop_length=512,
        )(complex_specgrams).repeat(3, 1, 1, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.TimeStretch(
            fixed_rate=rate,
            n_freq=1025,
            hop_length=512,
        )(complex_specgrams.repeat(3, 1, 1, 1, 1))

        self.assertEqual(computed, expected, atol=1e-5, rtol=1e-5)

    def test_batch_Fade(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100
        fade_in_len = 3000
        fade_out_len = 3000

        # Single then transform then batch
        expected = torchaudio.transforms.Fade(fade_in_len, fade_out_len)(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Fade(fade_in_len, fade_out_len)(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected)

    def test_batch_Vol(self):
        test_filepath = common_utils.get_asset_path('steam-train-whistle-daniel_simon.wav')
        waveform, _ = torchaudio.load(test_filepath)  # (2, 278756), 44100

        # Single then transform then batch
        expected = torchaudio.transforms.Vol(gain=1.1)(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.Vol(gain=1.1)(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected)

    def test_batch_spectral_centroid(self):
        sample_rate = 44100
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate)

        # Single then transform then batch
        expected = torchaudio.transforms.SpectralCentroid(sample_rate)(waveform).repeat(3, 1, 1)

        # Batch then transform
        computed = torchaudio.transforms.SpectralCentroid(sample_rate)(waveform.repeat(3, 1, 1))
        self.assertEqual(computed, expected)
