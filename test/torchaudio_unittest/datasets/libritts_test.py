import os
from pathlib import Path

from torchaudio.datasets.libritts import LIBRITTS

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


class TestLibriTTS(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    data = []
    utterance_ids = [
        [19, 198, '000000', '000000'],
        [26, 495, '000004', '000000'],
    ]
    original_text = 'this is the original text.'
    normalized_text = 'this is the normalized text.'

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        base_dir = os.path.join(cls.root_dir, 'LibriTTS', 'train-clean-100')
        for i, utterance_id in enumerate(cls.utterance_ids):
            filename = f'{"_".join(str(u) for u in utterance_id)}.wav'
            file_dir = os.path.join(base_dir, str(utterance_id[0]), str(utterance_id[1]))
            os.makedirs(file_dir, exist_ok=True)
            path = os.path.join(file_dir, filename)

            data = get_whitenoise(sample_rate=24000, duration=2, n_channels=1, dtype='int16', seed=i)
            save_wav(path, data, 24000)
            cls.data.append(normalize_wav(data))

            original_text_filename = f'{"_".join(str(u) for u in utterance_id)}.original.txt'
            path_original = os.path.join(file_dir, original_text_filename)
            with open(path_original, 'w') as file_:
                file_.write(cls.original_text)

            normalized_text_filename = f'{"_".join(str(u) for u in utterance_id)}.normalized.txt'
            path_normalized = os.path.join(file_dir, normalized_text_filename)
            with open(path_normalized, 'w') as file_:
                file_.write(cls.normalized_text)

    def _test_libritts(self, dataset):
        n_ites = 0
        for i, (waveform,
                sample_rate,
                original_text,
                normalized_text,
                speaker_id,
                chapter_id,
                utterance_id) in enumerate(dataset):

            expected_ids = self.utterance_ids[i]
            expected_data = self.data[i]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == 24000
            assert speaker_id == expected_ids[0]
            assert chapter_id == expected_ids[1]
            assert original_text == self.original_text
            assert normalized_text == self.normalized_text
            assert utterance_id == f'{"_".join(str(u) for u in expected_ids[-4:])}'
            n_ites += 1
        assert n_ites == len(self.utterance_ids)

    def test_libritts_str(self):
        dataset = LIBRITTS(self.root_dir)
        self._test_libritts(dataset)

    def test_libritts_path(self):
        dataset = LIBRITTS(Path(self.root_dir))
        self._test_libritts(dataset)
