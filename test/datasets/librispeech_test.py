import os

from torchaudio.datasets import librispeech

from ..common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)

# Used to generate a unique utterance for each dummy audio file
NUMBERS = [
    'ZERO',
    'ONE',
    'TWO',
    'THREE',
    'FOUR',
    'FIVE',
    'SIX',
    'SEVEN',
    'EIGHT',
    'NINE'
]


class TestLibriSpeech(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        dataset_dir = os.path.join(
            cls.root_dir, librispeech.FOLDER_IN_ARCHIVE, librispeech.URL
        )
        os.makedirs(dataset_dir, exist_ok=True)
        sample_rate = 16000  # 16kHz
        seed = 0

        for speaker_id in range(5):
            speaker_path = os.path.join(dataset_dir, str(speaker_id))
            os.makedirs(speaker_path, exist_ok=True)

            for chapter_id in range(3):
                chapter_path = os.path.join(speaker_path, str(chapter_id))
                os.makedirs(chapter_path, exist_ok=True)
                trans_content = []

                for utterance_id in range(20):
                    filename = f'{speaker_id}-{chapter_id}-{utterance_id:04d}.wav'
                    path = os.path.join(chapter_path, filename)

                    utterance = ' '.join(
                        [NUMBERS[int(x)] for x in list(
                            str(speaker_id) + str(chapter_id) + str(utterance_id)
                        )]
                    )
                    trans_content.append(
                        f'{speaker_id}-{chapter_id}-{utterance_id:04d} {utterance}'
                    )

                    data = get_whitenoise(
                        sample_rate=sample_rate,
                        duration=0.01,
                        n_channels=1,
                        dtype='float32',
                        seed=seed
                    )
                    save_wav(path, data, sample_rate)
                    sample = (
                        normalize_wav(data),
                        sample_rate,
                        utterance,
                        speaker_id,
                        chapter_id,
                        utterance_id
                    )
                    cls.samples.append(sample)

                    seed += 1

                trans_filename = f'{speaker_id}-{chapter_id}.trans.txt'
                trans_path = os.path.join(chapter_path, trans_filename)
                with open(trans_path, 'w') as f:
                    f.write('\n'.join(trans_content))

    def test_librispeech(self):
        dataset = librispeech.LIBRISPEECH(self.root_dir, ext_audio='.wav')
        print(dataset._path)

        num_samples = 0
        for i, (
            data, sample_rate, utterance, speaker_id, chapter_id, utterance_id
        ) in enumerate(dataset):
            self.assertEqual(data, self.samples[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.samples[i][1]
            assert utterance == self.samples[i][2]
            assert speaker_id == self.samples[i][3]
            assert chapter_id == self.samples[i][4]
            assert utterance_id == self.samples[i][5]
            num_samples += 1

        assert num_samples == len(self.samples)
