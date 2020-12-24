import csv
import os
from pathlib import Path

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)

from torchaudio.datasets import COMMONVOICE


class TestCommonVoice(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    data = []
    en_data = []
    fr_data = []
    _headers = [u"client_ids", u"path", u"sentence", u"up_votes", u"down_votes", u"age", u"gender", u"accent"]
    # Note: extension is changed to wav for the sake of test
    # Note: the first content is missing values for `age`, `gender` and `accent` as in the original data.
    _en_train_csv_contents = [
        ["9d16c5d980247861130e0480e2719f448be73d86a496c36d01a477cbdecd8cfd1399403d7a77bf458d211a70711b2da0845c",
         "common_voice_en_18885784.wav",
         "He was accorded a State funeral, and was buried in Drayton and Toowoomba Cemetery.", "2", "0", "", "", ""],
        ["c82eb9291328620f06025a1f8112b909099e447e485e99236cb87df008650250e79fea5ca772061fb6a370830847b9c44d20",
         "common_voice_en_556542.wav", "Once more into the breach", "2", "0", "thirties", "male", "us"],
        ["f74d880c5ad4c5917f314a604d3fc4805159d255796fb9f8defca35333ecc002bdf53dc463503c12674ea840b21b4a507b7c",
         "common_voice_en_18607573.wav",
         "Caddy, show Miss Clare and Miss Summerson their rooms.", "2", "0", "twenties", "male", "canada"],
    ]
    _fr_train_csv_contents = [
        [
            "a2e8e1e1cc74d08c92a53d7b9ff84e077eb90410edd85b8882f16fd037cecfcb6a19413c6c63ce6458cfea9579878fa91cef"
            "18343441c601cae0597a4b0d3144",
            "89e67e7682b36786a0b4b4022c4d42090c86edd96c78c12d30088e62522b8fe466ea4912e6a1055dfb91b296a0743e0a2bbe"
            "16cebac98ee5349e3e8262cb9329",
            "Or sur ce point nous n’avons aucune réponse de votre part.", "2", "0", "twenties", "male", "france"],
        [
            "a2e8e1e1cc74d08c92a53d7b9ff84e077eb90410edd85b8882f16fd037cecfcb6a19413c6c63ce6458cfea9579878fa91cef18"
            "343441c601cae0597a4b0d3144",
            "87d71819a26179e93acfee149d0b21b7bf5e926e367d80b2b3792d45f46e04853a514945783ff764c1fc237b4eb0ee2b0a7a7"
            "cbd395acbdfcfa9d76a6e199bbd",
            "Monsieur de La Verpillière, laissez parler le ministre", "2", "0", "twenties", "male", "france"],

    ]

    sample_rate = 48000

    def fill_data(cls, train_csv_contents):
        cls.root_dir = cls.get_base_temp_dir()
        # Tsv file name difference does not mean different subset, testing as a whole dataset here
        tsv_filename = os.path.join(cls.root_dir, "train.tsv")
        audio_base_path = os.path.join(cls.root_dir, "clips")
        os.makedirs(audio_base_path, exist_ok=True)
        with open(tsv_filename, "w", newline='') as tsv:
            writer = csv.writer(tsv, delimiter='\t')
            writer.writerow(cls._headers)
            for i, content in enumerate(train_csv_contents):
                writer.writerow(content)
                # Generate and store audio
                if content[1].endswith(".wav"):
                    audio_path = os.path.join(audio_base_path, content[1])
                else:
                    audio_path = os.path.join(audio_base_path, content[1] + COMMONVOICE._ext_audio)
                data = get_whitenoise(sample_rate=cls.sample_rate, duration=1, n_channels=1, seed=i, dtype='float32')
                save_wav(audio_path, data, cls.sample_rate)
                # Append data entry
                cls.data.append((normalize_wav(data), cls.sample_rate, dict(zip(cls._headers, content))))
        return cls.data

    @classmethod
    def setUpClass(cls):
        cls.en_data = cls.fill_data(train_csv_contents=cls._en_train_csv_contents)
        cls.fr_data = cls.fill_data(train_csv_contents=cls._fr_train_csv_contents)

    def _en_test_commonvoice(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, dictionary) in enumerate(dataset):
            expected_dictionary = self.en_data[i][2]
            expected_data = self.en_data[i][0]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == TestCommonVoice.sample_rate
            assert dictionary == expected_dictionary
            n_ite += 1
        assert n_ite == len(self.en_data)

    def _fr_test_commonvoice(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, dictionary) in enumerate(dataset):
            expected_dictionary = self.fr_data[i][2]
            expected_data = self.fr_data[i][0]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == TestCommonVoice.sample_rate
            assert dictionary == expected_dictionary
            n_ite += 1
        assert n_ite == len(self.fr_data)

    def test_en_commonvoice_str(self):
        dataset = COMMONVOICE(self.root_dir)
        self._en_test_commonvoice(dataset)

    def test_en_commonvoice_path(self):
        dataset = COMMONVOICE(Path(self.root_dir))
        self._en_test_commonvoice(dataset)

    def test_fr_commonvoice_str(self):
        dataset = COMMONVOICE(self.root_dir)
        self._fr_test_commonvoice(dataset)

    def test_fr_commonvoice_path(self):
        dataset = COMMONVOICE(Path(self.root_dir))
        self._fr_test_commonvoice(dataset)
