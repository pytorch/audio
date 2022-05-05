import math

import torch
from dataset.hubert_dataset import _crop_audio_label
from parameterized import parameterized
from torchaudio_unittest.common_utils import get_whitenoise, TorchaudioTestCase


class TestCropAudioLabel(TorchaudioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        torch.random.manual_seed(31)

    @parameterized.expand(
        [
            (400,),
            (800,),
        ]
    )
    def test_zero_offset(self, num_frames):
        sample_rate = 16000
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05)
        length = waveform.shape[1]
        label = torch.rand(50)
        waveform_out, label_out, length = _crop_audio_label(waveform, label, length, num_frames, rand_crop=False)
        self.assertEqual(waveform_out.shape[0], num_frames)
        self.assertEqual(waveform_out, waveform[0, :num_frames])
        self.assertEqual(length, waveform_out.shape[0])
        self.assertEqual(label_out.shape[0], math.floor((num_frames - 25 * 16) / (20 * 16)) + 1)

    @parameterized.expand(
        [
            (400,),
            (800,),
        ]
    )
    def test_rand_crop(self, num_frames):
        sample_rate = 16000
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05)
        length = waveform.shape[1]
        label = torch.rand(50)
        waveform_out, label_out, length = _crop_audio_label(waveform, label, length, num_frames, rand_crop=True)
        self.assertEqual(waveform_out.shape[0], num_frames)
        self.assertEqual(length, waveform_out.shape[0])
        self.assertEqual(label_out.shape[0], math.floor((num_frames - 25 * 16) / (20 * 16)) + 1)
