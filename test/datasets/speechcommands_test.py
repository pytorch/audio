import os

from torchaudio.datasets import speechcommands

from ..common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    normalize_wav,
    save_wav,
)


class TestSpeechCommands(TempDirMixin, TorchaudioTestCase):
    backend = "default"

    root_dir = None
    samples = []

    @classmethod
    def setUp(cls):
        cls.root_dir = cls.get_base_temp_dir()
        dataset_dir = os.path.join(
            cls.root_dir, speechcommands.FOLDER_IN_ARCHIVE, speechcommands.URL
        )
        os.makedirs(dataset_dir, exist_ok=True)
        sample_rate = 16000  # 16kHz sample rate
        for i in range(30):
            label = "label_{:02d}".format(i)
            path = os.path.join(dataset_dir, label)
            os.makedirs(path, exist_ok=True)
            for j in range(100):
                # generate hash ID for speaker
                speaker = "{:08x}".format(j)

                for utterance in range(3):
                    filename = f"{speaker}{speechcommands.HASH_DIVIDER}{utterance}.wav"
                    file_path = os.path.join(path, filename)
                    data = get_whitenoise(
                        sample_rate=sample_rate,
                        duration=0.01,
                        n_channels=1,
                        dtype="int16",
                        seed=0,
                    )
                    save_wav(file_path, data, sample_rate)
                    sample = (
                        normalize_wav(data),
                        sample_rate,
                        label,
                        speaker,
                        utterance,
                    )
                    cls.samples.append(sample)

    def testSpeechCommands(self):
        dataset = speechcommands.SPEECHCOMMANDS(self.root_dir)

        num_samples = 0
        for i, (data, sample_rate, label, speaker_id, utterance_number) in enumerate(
            dataset
        ):
            self.assertEqual(data, self.samples[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.samples[i][1]
            assert label == self.samples[i][2]
            assert speaker_id == self.samples[i][3]
            assert utterance_number == self.samples[i][4]
            num_samples += 1

        assert num_samples == len(self.samples)
