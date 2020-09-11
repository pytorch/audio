import os

from torchaudio.datasets import tedlium

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)

# Used to generate a unique utterance for each dummy audio file
UTTERANCES = [
    "AaronHuey_2010X 1 AaronHuey_2010X 0.0 2.0 <o,f0,female> script1\n",
    "AaronHuey_2010X 1 AaronHuey_2010X 2.0 4.0 <o,f0,female> script2\n",
    "AaronHuey_2010X 1 AaronHuey_2010X 4.0 6.0 <o,f0,female> script3\n",
    "AaronHuey_2010X 1 AaronHuey_2010X 6.0 8.0 <o,f0,female> script4\n",
    "AaronHuey_2010X 1 AaronHuey_2010X 8.0 10.0 <o,f0,female> script5\n",
]


class TestTedlium(TempDirMixin, TorchaudioTestCase):
    backend = "default"

    root_dir = None
    samples = {}

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.root_dir = dataset_dir = os.path.join(cls.root_dir, "tedlium")
        os.makedirs(dataset_dir, exist_ok=True)
        sample_rate = 16000  # 16kHz
        seed = 0

        for release in ["release1", "release2", "release3"]:
            data = get_whitenoise(sample_rate=sample_rate, duration=10.00, n_channels=1, dtype="float32", seed=seed)
            if release in ["release1", "release2"]:
                release_dir = os.path.join(
                    dataset_dir,
                    tedlium._RELEASE_CONFIGS[release]["folder_in_archive"],
                    tedlium._RELEASE_CONFIGS[release]["subset"],
                )
            else:
                release_dir = os.path.join(
                    dataset_dir,
                    tedlium._RELEASE_CONFIGS[release]["folder_in_archive"],
                    tedlium._RELEASE_CONFIGS[release]["data_path"],
                )
            os.makedirs(release_dir, exist_ok=True)
            os.makedirs(os.path.join(release_dir, "stm"), exist_ok=True)  # Subfolder for transcripts
            os.makedirs(os.path.join(release_dir, "sph"), exist_ok=True)  # Subfolder for audio files
            filename = f"{release}.sph"
            path = os.path.join(os.path.join(release_dir, "sph"), filename)
            save_wav(path, data, sample_rate)

            trans_filename = f"{release}.stm"
            trans_path = os.path.join(os.path.join(release_dir, "stm"), trans_filename)
            with open(trans_path, "w") as f:
                f.write("".join(UTTERANCES))

            # Create a samples list to compare with
            cls.samples[release] = []
            for utterance in UTTERANCES:
                talk_id, _, speaker_id, start_time, end_time, identifier, transcript = utterance.split(" ", 6)
                start_time = int(float(start_time)) * sample_rate
                end_time = int(float(end_time)) * sample_rate
                sample = (
                    data[:, start_time:end_time],
                    sample_rate,
                    transcript,
                    talk_id,
                    speaker_id,
                    identifier,
                )
                cls.samples[release].append(sample)
            seed += 1

    def test_tedlium_release1(self):
        release = "release1"
        dataset = tedlium.TEDLIUM(self.root_dir, release=release)
        num_samples = 0
        for i, (data, sample_rate, transcript, talk_id, speaker_id, identifier) in enumerate(dataset):
            self.assertEqual(data, self.samples[release][i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.samples[release][i][1]
            assert transcript == self.samples[release][i][2]
            assert talk_id == self.samples[release][i][3]
            assert speaker_id == self.samples[release][i][4]
            assert identifier == self.samples[release][i][5]
            num_samples += 1

        assert num_samples == len(self.samples[release])

    def test_tedlium_release2(self):
        release = "release2"
        dataset = tedlium.TEDLIUM(self.root_dir, release=release)
        num_samples = 0
        for i, (data, sample_rate, transcript, talk_id, speaker_id, identifier) in enumerate(dataset):
            self.assertEqual(data, self.samples[release][i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.samples[release][i][1]
            assert transcript == self.samples[release][i][2]
            assert talk_id == self.samples[release][i][3]
            assert speaker_id == self.samples[release][i][4]
            assert identifier == self.samples[release][i][5]
            num_samples += 1

        assert num_samples == len(self.samples[release])

    def test_tedlium_release3(self):
        release = "release3"
        dataset = tedlium.TEDLIUM(self.root_dir, release=release)
        num_samples = 0
        for i, (data, sample_rate, transcript, talk_id, speaker_id, identifier) in enumerate(dataset):
            self.assertEqual(data, self.samples[release][i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.samples[release][i][1]
            assert transcript == self.samples[release][i][2]
            assert talk_id == self.samples[release][i][3]
            assert speaker_id == self.samples[release][i][4]
            assert identifier == self.samples[release][i][5]
            num_samples += 1

        assert num_samples == len(self.samples[release])

