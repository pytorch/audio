import torch
from dataset.hubert_dataset import _crop_audio_label
from parameterized import parameterized
from torchaudio.models import hubert_base
from torchaudio_unittest.common_utils import get_whitenoise, TorchaudioTestCase


class TestCropAudioLabel(TorchaudioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    @parameterized.expand(
        [
            (400,),
            (800,),
        ]
    )
    def test_zero_offset(self, num_frames):
        """Test _crop_audio_label method with zero frame offset.
        Given the ``num_frames`` argument, the method returns the first ``num_frames`` samples in the waveform,
        the corresponding labels, and the length of the cropped waveform.
        The cropped waveform should be identical to the first ``num_frames`` samples of original waveform.
        The length of the cropped waveform should be identical to ``num_frames``.
        The dimension of the labels should be identical to HuBERT transformer layer output frame dimension.
        """
        sample_rate = 16000
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05)
        length = waveform.shape[1]
        label = torch.rand(50)
        model = hubert_base()
        waveform_out, label_out, length = _crop_audio_label(waveform, label, length, num_frames, rand_crop=False)
        hubert_feat = model.extract_features(waveform_out.unsqueeze(0), num_layers=1)[0][0]
        self.assertEqual(waveform_out.shape[0], num_frames, length)
        self.assertEqual(waveform_out, waveform[0, :num_frames])
        self.assertEqual(label_out.shape[0], hubert_feat.shape[1])

    @parameterized.expand(
        [
            (400,),
            (800,),
        ]
    )
    def test_rand_crop(self, num_frames):
        """Test _crop_audio_label method with random frame offset.
        Given the ``num_frames`` argument, the method returns ``num_frames`` samples in the waveform
        starting with random offset, the corresponding labels, and the length of the cropped waveform.
        The length of the cropped waveform should be identical to ``num_frames``.
        The dimension of the labels should be identical to HuBERT transformer layer output frame dimension.
        """
        sample_rate = 16000
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05)
        length = waveform.shape[1]
        label = torch.rand(50)
        model = hubert_base()
        waveform_out, label_out, length = _crop_audio_label(waveform, label, length, num_frames, rand_crop=False)
        hubert_feat = model.extract_features(waveform_out.unsqueeze(0), num_layers=1)[0][0]
        self.assertEqual(waveform_out.shape[0], num_frames, length)
        self.assertEqual(label_out.shape[0], hubert_feat.shape[1])
