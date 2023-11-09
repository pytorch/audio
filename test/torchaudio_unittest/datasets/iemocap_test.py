import os
import random

from torchaudio.datasets import iemocap
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase

LABELS = ["neu", "hap", "ang", "sad", "exc", "fru", "xxx"]
SAMPLE_RATE = 16000


def _save_wav(filepath: str, seed: int):
    wav = get_whitenoise(
        sample_rate=SAMPLE_RATE,
        duration=0.01,
        n_channels=1,
        seed=seed,
    )
    save_wav(filepath, wav, SAMPLE_RATE)
    return wav


def _save_label(label_folder: str, filename: str, wav_stem: str):
    label = random.choice(LABELS)
    line = f"[xxx]\t{wav_stem}\t{label}\t[yyy]"
    filepath = os.path.join(label_folder, filename)

    with open(filepath, "a") as f:
        f.write(line + "\n")
    return label


def _get_samples(dataset_dir: str, session: int):
    session_folder = os.path.join(dataset_dir, f"Session{session}")
    os.makedirs(session_folder, exist_ok=True)

    wav_folder = os.path.join(session_folder, "sentences", "wav")
    label_folder = os.path.join(session_folder, "dialog", "EmoEvaluation")
    os.makedirs(wav_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    wav_stems = []
    for i in range(5):
        for g in ["F", "M"]:
            for utt in ["impro", "script"]:
                speaker = f"Ses0{session}{g}"
                subfolder = f"{speaker}_{utt}0{i}"
                subfolder_path = os.path.join(wav_folder, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)

                for j in range(5):
                    wav_stem = f"{subfolder}_F00{j}"
                    wav_stems.append(wav_stem)

    all_samples = []
    impro_samples = []
    script_samples = []
    wav_stems = sorted(wav_stems)
    for wav_stem in wav_stems:
        subfolder = wav_stem[:-5]
        speaker = subfolder.split("_")[0]

        wav_file = os.path.join(wav_folder, subfolder, wav_stem + ".wav")
        wav = _save_wav(wav_file, seed=0)
        label = _save_label(label_folder, subfolder + ".txt", wav_stem)
        if label == "xxx":
            continue
        sample = (wav, SAMPLE_RATE, wav_stem, label, speaker)
        all_samples.append(sample)

        if "impro" in subfolder:
            impro_samples.append(sample)
        else:
            script_samples.append(sample)

    return all_samples, script_samples, impro_samples


def get_mock_dataset(dataset_dir: str):
    os.makedirs(dataset_dir, exist_ok=True)

    all_samples = []
    script_samples = []
    impro_samples = []
    for session in range(1, 4):
        samples = _get_samples(dataset_dir, session)
        all_samples += samples[0]
        script_samples += samples[1]
        impro_samples += samples[2]
    return all_samples, script_samples, impro_samples


class TestIemocap(TempDirMixin, TorchaudioTestCase):
    root_dir = None

    all_samples = []
    script_samples = []
    impro_samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        dataset_dir = os.path.join(cls.root_dir, "IEMOCAP")
        cls.all_samples, cls.script_samples, cls.impro_samples = get_mock_dataset(dataset_dir)

    def _testIEMOCAP(self, dataset, samples):
        num_samples = 0
        for i, data in enumerate(dataset):
            self.assertEqual(data, samples[i])
            num_samples += 1

        assert num_samples == len(samples)

    def testIEMOCAPFullDataset(self):
        dataset = iemocap.IEMOCAP(self.root_dir)
        self._testIEMOCAP(dataset, self.all_samples)

    def testIEMOCAPScriptedDataset(self):
        dataset = iemocap.IEMOCAP(self.root_dir, utterance_type="scripted")
        self._testIEMOCAP(dataset, self.script_samples)

    def testIEMOCAPImprovisedDataset(self):
        dataset = iemocap.IEMOCAP(self.root_dir, utterance_type="improvised")
        self._testIEMOCAP(dataset, self.impro_samples)
