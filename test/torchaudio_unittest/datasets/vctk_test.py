import os
from pathlib import Path

from torchaudio.datasets import vctk

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)

# Used to generate a unique utterance for each dummy audio file
UTTERANCE = [
    'Please call Stella',
    'Ask her to bring these things',
    'with her from the store',
    'Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob',
    'We also need a small plastic snake and a big toy frog for the kids',
    'She can scoop these things into three red bags, and we will go meet her Wednesday at the train station',
    'When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow',
    'The rainbow is a division of white light into many beautiful colors',
    'These take the shape of a long round arch, with its path high above, and its two ends \
        apparently beyond the horizon',
    'There is, according to legend, a boiling pot of gold at one end'
]


class TestVCTK(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        dataset_dir = os.path.join(cls.root_dir, 'VCTK-Corpus-0.92')
        os.makedirs(dataset_dir, exist_ok=True)
        sample_rate = 48000
        seed = 0

        for speaker in range(225, 230):
            speaker_id = 'p' + str(speaker)
            audio_dir = os.path.join(dataset_dir, 'wav48_silence_trimmed', speaker_id)
            os.makedirs(audio_dir, exist_ok=True)

            file_dir = os.path.join(dataset_dir, 'txt', speaker_id)
            os.makedirs(file_dir, exist_ok=True)

            for utterance_id in range(1, 11):
                filename = f'{speaker_id}_{utterance_id:03d}_mic2'
                audio_file_path = os.path.join(audio_dir, filename + '.wav')

                data = get_whitenoise(
                    sample_rate=sample_rate,
                    duration=0.01,
                    n_channels=1,
                    dtype='float32',
                    seed=seed
                )
                save_wav(audio_file_path, data, sample_rate)

                txt_file_path = os.path.join(file_dir, filename[:-5] + '.txt')
                utterance = UTTERANCE[utterance_id - 1]
                with open(txt_file_path, 'w') as f:
                    f.write(utterance)

                sample = (
                    normalize_wav(data),
                    sample_rate,
                    utterance,
                    speaker_id,
                    utterance_id
                )
                cls.samples.append(sample)

                seed += 1

    def _test_vctk(self, dataset):
        num_samples = 0
        for i, (data, sample_rate, utterance, speaker_id, utterance_id) in enumerate(dataset):
            self.assertEqual(data, self.samples[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.samples[i][1]
            assert utterance == self.samples[i][2]
            assert speaker_id == self.samples[i][3]
            assert int(utterance_id) == self.samples[i][4]
            num_samples += 1

        assert num_samples == len(self.samples)

    def test_vctk_str(self):
        dataset = vctk.VCTK_092(self.root_dir, audio_ext=".wav")
        self._test_vctk(dataset)

    def test_vctk_path(self):
        dataset = vctk.VCTK_092(Path(self.root_dir), audio_ext=".wav")
        self._test_vctk(dataset)
