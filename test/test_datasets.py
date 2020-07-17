import os
import unittest

from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchaudio.datasets.utils import diskcache_iterator, bg_iterator
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.yesno import YESNO
from torchaudio.datasets.ljspeech import LJSPEECH
from torchaudio.datasets.gtzan import GTZAN
from torchaudio.datasets.cmuarctic import CMUARCTIC
from torchaudio.datasets.libritts import LIBRITTS

from .common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_asset_path,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


class TestDatasets(TorchaudioTestCase):
    backend = 'default'
    path = get_asset_path()

    def test_vctk(self):
        data = VCTK(self.path)
        data[0]

    def test_librispeech(self):
        data = LIBRISPEECH(self.path, "dev-clean")
        data[0]

    def test_ljspeech(self):
        data = LJSPEECH(self.path)
        data[0]

    def test_speechcommands(self):
        data = SPEECHCOMMANDS(self.path)
        data[0]

    def test_gtzan(self):
        data = GTZAN(self.path)
        data[0]

    def test_cmuarctic(self):
        data = CMUARCTIC(self.path)
        data[0]


class TestCommonVoice(TorchaudioTestCase):
    backend = 'default'
    path = get_asset_path()

    def test_commonvoice(self):
        data = COMMONVOICE(self.path, url="tatar")
        data[0]

    def test_commonvoice_diskcache(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = diskcache_iterator(data)
        # Save
        data[0]
        # Load
        data[0]

    def test_commonvoice_bg(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = bg_iterator(data, 5)
        for _ in data:
            pass


class TestYesNo(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    data = []
    labels = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        base_dir = os.path.join(cls.root_dir, 'waves_yesno')
        os.makedirs(base_dir, exist_ok=True)
        for label in cls.labels:
            filename = f'{"_".join(str(l) for l in label)}.wav'
            path = os.path.join(base_dir, filename)

            data = get_whitenoise(sample_rate=8000, duration=6, n_channels=1, dtype='int16')
            save_wav(path, data, 8000)
            cls.data.append(normalize_wav(data))

    def test_yesno(self):
        dataset = YESNO(self.root_dir)
        samples = list(dataset)
        samples.sort(key=lambda s: s[2])
        for i, (waveform, sample_rate, label) in enumerate(samples):
            expected_label = self.labels[i]
            expected_data = self.data[i]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == 8000
            assert label == expected_label


class TestLibriTTS(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    data = []
    utterance_ids = [
        [19, 198, '000000', '000000'],
        [26, 495, '000004', '000000'],
    ]
    original_text = 'this is the test text.'
    normalized_text = 'this is the normalized text.'

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        base_dir = os.path.join(cls.root_dir, 'LibriTTS', 'train-clean-100')
        for utterance_id in cls.utterance_ids:
            filename = f'{"_".join(str(u) for u in utterance_id)}.wav'
            file_dir = os.path.join(base_dir, str(utterance_id[0]), str(utterance_id[1]))
            os.makedirs(file_dir, exist_ok=True)
            path = os.path.join(file_dir, filename)

            data = get_whitenoise(sample_rate=8000, duration=6, n_channels=1, dtype='int16')
            save_wav(path, data, 8000)
            cls.data.append(normalize_wav(data))

            original_text_filename = f'{"_".join(str(u) for u in utterance_id)}.original.txt'
            path_original = os.path.join(file_dir, original_text_filename)
            f = open(path_original, 'w')
            f.write(cls.original_text)
            f.close()

            normalized_text_filename = f'{"_".join(str(u) for u in utterance_id)}.normalized.txt'
            path_normalized = os.path.join(file_dir, normalized_text_filename)
            f = open(path_normalized, 'w')
            f.write(cls.normalized_text)
            f.close()

    def test_libritts(self):
        dataset = LIBRITTS(self.root_dir)
        samples = list(dataset)
        samples.sort(key=lambda s: s[2])

        for i, (waveform, sample_rate, original_utterance, normalized_utterance, speaker_id, chapter_id, utterance_id) in enumerate(samples):
            expected_ids = self.utterance_ids[i]
            expected_data = self.data[i]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == 8000
            assert speaker_id == expected_ids[0]
            assert chapter_id == expected_ids[1]
            assert original_utterance == self.original_text
            assert normalized_utterance == self.normalized_text
            assert utterance_id == f'{"_".join(str(u) for u in expected_ids[-4:])}'


if __name__ == "__main__":
    unittest.main()
