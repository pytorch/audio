import os

from torchaudio.datasets import cmuarctic

from ..common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


class TestCMUARCTIC(TempDirMixin, TorchaudioTestCase):
    backend = "default"

    root_dir = None
    URL = "aew"  # default url in CMUARCTIC
    samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        sample_rate = 16000
        seed = 42
        utterance = "This is a test utterance."

        base_dir = os.path.join(cls.root_dir, "ARCTIC", "cmu_us_" + cls.URL + "_arctic")
        # Contains utterance ID & sentence prompts
        txt_dir = os.path.join(base_dir, "etc")
        os.makedirs(txt_dir, exist_ok=True)
        txt_file = os.path.join(txt_dir, "txt.done.data")
        # Contains the audio files
        audio_dir = os.path.join(base_dir, "wav")
        os.makedirs(audio_dir, exist_ok=True)

        with open(txt_file, "w") as txt:
            for i in range(10):
                # Write audio file
                utterance_id = f"arctic_a{i:04d}"
                path = os.path.join(audio_dir, f"{utterance_id}.wav")
                data = get_whitenoise(
                    sample_rate=sample_rate,
                    duration=3,
                    n_channels=1,
                    dtype="int16",
                    seed=seed,
                )
                save_wav(path, data, sample_rate)
                sample = (
                    normalize_wav(data),
                    sample_rate,
                    utterance,
                    utterance_id.split("_")[1],
                )
                cls.samples.append(sample)
                # Write sentence prompt
                txt.write(f'( {utterance_id} "{utterance}" )\n')

    def test_cmuarctic(self):
        dataset = cmuarctic.CMUARCTIC(self.root_dir)
        for i, (waveform, sample_rate, utterance, utterance_id) in enumerate(dataset):
            expected_sample = self.samples[i]
            assert sample_rate == expected_sample[1]
            assert utterance == expected_sample[2]
            assert utterance_id == expected_sample[3]
            self.assertEqual(expected_sample[0], waveform, atol=5e-5, rtol=1e-8)
        assert (i + 1) == len(self.data)
