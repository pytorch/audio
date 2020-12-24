import os
import csv
from pathlib import Path

from torchaudio.datasets import COMMONVOICE
from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


class TestCommonVoice(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    data = []
    _headers = [u"client_ids", u"path", u"sentence", u"up_votes", u"down_votes", u"age", u"gender", u"accent"]
    # Note: extension is changed to wav for the sake of test
    # Note: the first content is missing values for `age`, `gender` and `accent` as in the original data.
    _train_csv_contents = [
        ["9d16c5d980247861130e0480e2719f448be73d86a496c36d01a477cbdecd8cfd1399403d7a77bf458d211a70711b2da0845c",
            "common_voice_en_18885784",
            "He was accorded a State funeral, and was buried in Drayton and Toowoomba Cemetery.", "2", "0", "", "", ""],
        ["c82eb9291328620f06025a1f8112b909099e447e485e99236cb87df008650250e79fea5ca772061fb6a370830847b9c44d20",
            "common_voice_en_556542", "Once more into the breach", "2", "0", "thirties", "male", "us"],
        ["f74d880c5ad4c5917f314a604d3fc4805159d255796fb9f8defca35333ecc002bdf53dc463503c12674ea840b21b4a507b7c",
            "common_voice_en_18607573",
            "Caddy, show Miss Clare and Miss Summerson their rooms.", "2", "0", "twenties", "male", "canada"],
    ]
    sample_rate = 48000

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        # Tsv file name difference does not mean different subset, testing as a whole dataset here
        tsv_filename = os.path.join(cls.root_dir, "train.tsv")
        audio_base_path = os.path.join(cls.root_dir, "clips")
        os.makedirs(audio_base_path, exist_ok=True)
        with open(tsv_filename, "w", newline='') as tsv:
            writer = csv.writer(tsv, delimiter='\t')
            writer.writerow(cls._headers)
            for i, content in enumerate(cls._train_csv_contents):
                writer.writerow(content)

                # Generate and store audio
                audio_path = os.path.join(audio_base_path, content[1] + COMMONVOICE._ext_audio)
                data = get_whitenoise(sample_rate=cls.sample_rate, duration=1, n_channels=1, seed=i, dtype='float32')
                save_wav(audio_path, data, cls.sample_rate)

                # Append data entry
                cls.data.append((normalize_wav(data), cls.sample_rate, dict(zip(cls._headers, content))))

    def _test_commonvoice(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, dictionary) in enumerate(dataset):
            expected_dictionary = self.data[i][2]
            expected_data = self.data[i][0]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == TestCommonVoice.sample_rate
            assert dictionary == expected_dictionary
            n_ite += 1
        assert n_ite == len(self.data)

    def test_commonvoice_str(self):
        dataset = COMMONVOICE(self.root_dir)
        self._test_commonvoice(dataset)

    def test_commonvoice_path(self):
        dataset = COMMONVOICE(Path(self.root_dir))
        self._test_commonvoice(dataset)
