"""Test numerical consistency among single input and batched input."""
import os

import torch
from parameterized import parameterized
from torchaudio import transforms as T
from torchaudio_unittest import common_utils


class TestTransforms(common_utils.TorchaudioTestCase):
    """Test suite for classes defined in `transforms` module"""

    backend = "default"

    def assert_batch_consistency(self, transform, batch, *args, atol=1e-8, rtol=1e-5, seed=42, **kwargs):
        n = batch.size(0)

        # Compute items separately, then batch the result
        torch.random.manual_seed(seed)
        items_input = batch.clone()
        items_result = torch.stack([transform(items_input[i], *args, **kwargs) for i in range(n)])

        # Batch the input and run
        torch.random.manual_seed(seed)
        batch_input = batch.clone()
        batch_result = transform(batch_input, *args, **kwargs)

        self.assertEqual(items_input, batch_input, rtol=rtol, atol=atol)
        self.assertEqual(items_result, batch_result, rtol=rtol, atol=atol)

    def test_batch_AmplitudeToDB(self):
        spec = torch.rand((3, 2, 6, 201))
        transform = T.AmplitudeToDB()

        self.assert_batch_consistency(transform, spec)

    def test_batch_Resample(self):
        waveform = torch.randn(3, 2, 2786)
        transform = T.Resample()

        self.assert_batch_consistency(transform, waveform)

    def test_batch_MelScale(self):
        specgram = torch.randn(3, 2, 201, 256)

        atol = 1e-6 if os.name == "nt" else 1e-8
        transform = T.MelScale()

        self.assert_batch_consistency(transform, specgram, atol=atol)

    def test_batch_InverseMelScale(self):
        n_mels = 32
        n_stft = 5
        mel_spec = torch.randn(3, 2, n_mels, 32) ** 2
        transform = T.InverseMelScale(n_stft, n_mels)

        # Because InverseMelScale runs SGD on randomly initialized values so they do not yield
        # exactly same result. For this reason, tolerance is very relaxed here.
        self.assert_batch_consistency(transform, mel_spec, atol=1.0, rtol=1e-5)

    def test_batch_BarkScale(self):
        specgram = torch.randn(3, 2, 201, 256)

        atol = 1e-4 if os.name == "nt" else 1e-6
        transform = T.BarkScale()

        self.assert_batch_consistency(transform, specgram, atol=atol)

    def test_batch_InverseBarkScale(self):
        n_barks = 32
        n_stft = 5
        bark_spec = torch.randn(3, 2, n_barks, 32) ** 2
        transform = T.InverseMelScale(n_stft, n_barks)

        # Because InverseBarkScale runs SGD on randomly initialized values so they do not yield
        # exactly same result. For this reason, tolerance is very relaxed here.
        self.assert_batch_consistency(transform, bark_spec, atol=1.0, rtol=1e-5)

    def test_batch_compute_deltas(self):
        specgram = torch.randn(3, 2, 31, 2786)
        transform = T.ComputeDeltas()

        self.assert_batch_consistency(transform, specgram)

    def test_batch_mulaw(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)

        # Single then transform then batch
        expected = [T.MuLawEncoding()(waveform[i]) for i in range(3)]
        expected = torch.stack(expected)

        # Batch then transform
        computed = T.MuLawEncoding()(waveform)

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed, expected)

        # Single then transform then batch
        expected_decoded = [T.MuLawDecoding()(expected[i]) for i in range(3)]
        expected_decoded = torch.stack(expected_decoded)

        # Batch then transform
        computed_decoded = T.MuLawDecoding()(computed)

        # shape = (3, 2, 201, 1394)
        self.assertEqual(computed_decoded, expected_decoded)

    def test_batch_spectrogram(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        transform = T.Spectrogram()

        self.assert_batch_consistency(transform, waveform)

    def test_batch_inverse_spectrogram(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        specgram = common_utils.get_spectrogram(waveform, n_fft=400)
        specgram = specgram.reshape(3, 2, specgram.shape[-2], specgram.shape[-1])
        transform = T.InverseSpectrogram(n_fft=400)

        self.assert_batch_consistency(transform, specgram)

    def test_batch_melspectrogram(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        transform = T.MelSpectrogram()

        self.assert_batch_consistency(transform, waveform)

    def test_batch_mfcc(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        transform = T.MFCC()

        self.assert_batch_consistency(transform, waveform, atol=1e-4, rtol=1e-5)

    def test_batch_lfcc(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        transform = T.LFCC()

        self.assert_batch_consistency(transform, waveform, atol=1e-4, rtol=1e-5)

    def test_batch_TimeStretch(self):
        rate = 2
        num_freq = 1025
        batch = 3

        tensor = common_utils.get_whitenoise(sample_rate=8000, n_channels=batch)
        spec = common_utils.get_spectrogram(tensor, n_fft=num_freq)
        transform = T.TimeStretch(fixed_rate=rate, n_freq=num_freq // 2 + 1, hop_length=512)

        self.assert_batch_consistency(transform, spec, atol=1e-5, rtol=1e-5)

    def test_batch_Fade(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        fade_in_len = 3000
        fade_out_len = 3000
        transform = T.Fade(fade_in_len, fade_out_len)

        self.assert_batch_consistency(transform, waveform)

    def test_batch_Vol(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        transform = T.Vol(gain=1.1)

        self.assert_batch_consistency(transform, waveform)

    def test_batch_spectral_centroid(self):
        sample_rate = 44100
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        transform = T.SpectralCentroid(sample_rate)

        self.assert_batch_consistency(transform, waveform)

    def test_batch_pitch_shift(self):
        sample_rate = 8000
        n_steps = -2
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=6)
        waveform = waveform.reshape(3, 2, -1)
        transform = T.PitchShift(sample_rate, n_steps, n_fft=400)

        self.assert_batch_consistency(transform, waveform)

    def test_batch_PSD(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        specgram = common_utils.get_spectrogram(waveform, n_fft=400)
        specgram = specgram.reshape(3, 2, specgram.shape[-2], specgram.shape[-1])
        transform = T.PSD()

        self.assert_batch_consistency(transform, specgram)

    def test_batch_PSD_with_mask(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        waveform = waveform.to(torch.double)
        specgram = common_utils.get_spectrogram(waveform, n_fft=400)
        specgram = specgram.reshape(3, 2, specgram.shape[-2], specgram.shape[-1])
        mask = torch.rand((3, specgram.shape[-2], specgram.shape[-1]))
        transform = T.PSD()

        # Single then transform then batch
        expected = [transform(specgram[i], mask[i]) for i in range(3)]
        expected = torch.stack(expected)

        # Batch then transform
        computed = transform(specgram, mask)

        self.assertEqual(computed, expected)

    @parameterized.expand(
        [
            [True],
            [False],
        ]
    )
    def test_MVDR(self, multi_mask):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        specgram = common_utils.get_spectrogram(waveform, n_fft=400)
        specgram = specgram.reshape(3, 2, specgram.shape[-2], specgram.shape[-1])
        if multi_mask:
            mask_s = torch.rand((3, 2, specgram.shape[-2], specgram.shape[-1]))
            mask_n = torch.rand((3, 2, specgram.shape[-2], specgram.shape[-1]))
        else:
            mask_s = torch.rand((3, specgram.shape[-2], specgram.shape[-1]))
            mask_n = torch.rand((3, specgram.shape[-2], specgram.shape[-1]))
        transform = T.MVDR(multi_mask=multi_mask)

        # Single then transform then batch
        expected = [transform(specgram[i], mask_s[i], mask_n[i]) for i in range(3)]
        expected = torch.stack(expected)

        # Batch then transform
        computed = transform(specgram, mask_s, mask_n)

        self.assertEqual(computed, expected)

    def test_rtf_mvdr(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        specgram = common_utils.get_spectrogram(waveform, n_fft=400)
        batch_size, channel, freq, time = 3, 2, specgram.shape[-2], specgram.shape[-1]
        specgram = specgram.reshape(batch_size, channel, freq, time)
        rtf = torch.rand(batch_size, freq, channel, dtype=torch.cfloat)
        psd_n = torch.rand(batch_size, freq, channel, channel, dtype=torch.cfloat)
        reference_channel = 0
        transform = T.RTFMVDR()

        # Single then transform then batch
        expected = [transform(specgram[i], rtf[i], psd_n[i], reference_channel) for i in range(batch_size)]
        expected = torch.stack(expected)

        # Batch then transform
        computed = transform(specgram, rtf, psd_n, reference_channel)

        self.assertEqual(computed, expected)

    def test_souden_mvdr(self):
        waveform = common_utils.get_whitenoise(sample_rate=8000, duration=1, n_channels=6)
        specgram = common_utils.get_spectrogram(waveform, n_fft=400)
        batch_size, channel, freq, time = 3, 2, specgram.shape[-2], specgram.shape[-1]
        specgram = specgram.reshape(batch_size, channel, freq, time)
        psd_s = torch.rand(batch_size, freq, channel, channel, dtype=torch.cfloat)
        psd_n = torch.rand(batch_size, freq, channel, channel, dtype=torch.cfloat)
        reference_channel = 0
        transform = T.SoudenMVDR()

        # Single then transform then batch
        expected = [transform(specgram[i], psd_s[i], psd_n[i], reference_channel) for i in range(batch_size)]
        expected = torch.stack(expected)

        # Batch then transform
        computed = transform(specgram, psd_s, psd_n, reference_channel)

        self.assertEqual(computed, expected)
