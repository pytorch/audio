import os
import csv
import random

from torchaudio.datasets import commonvoice
from ..common_utils import (
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
    _client_ids = [u"0", u"1"]
    _sentence = [u"test", u"pytorch"]
    _up_votes = [u"0", u"1"]
    _down_votes = [u"0", u"1"]
    _age = [u"0", u"1"]
    _gender = [u"M", u"W", u"U"]
    _accent = [u"Y", u"N"]
    _folder_audio = "clips"
    sample_rate = 22050

    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.join(cls.get_base_temp_dir(), 'waves_commonvoice')
        # The path convention commonvoice uses
        base_dir = os.path.join(cls.root_dir, commonvoice.FOLDER_IN_ARCHIVE, commonvoice.VERSION, "en")
        os.makedirs(base_dir, exist_ok=True)

        # Tsv file name difference does not mean different subset, testing as a whole dataset here
        tsv_filename = os.path.join(base_dir, commonvoice.DEFAULT_TSV)
        with open(tsv_filename, "w", newline='') as tsv:
            writer = csv.writer(tsv, delimiter='\t')
            writer.writerow(cls._headers)
            for i in range(100):
                audio_filename = f'voice_{i:05d}.wav'

                # Generate and store text data
                text_data = [
                    random.choice(cls._client_ids),
                    audio_filename,
                    random.choice(cls._sentence),
                    random.choice(cls._client_ids),
                    random.choice(cls._up_votes),
                    random.choice(cls._down_votes),
                    random.choice(cls._age),
                    random.choice(cls._gender),
                    random.choice(cls._accent)
                ]
                writer.writerow(text_data)

                # Generate and store audio
                audio_base_path = os.path.join(base_dir, cls._folder_audio)
                os.makedirs(audio_base_path, exist_ok=True)
                audio_path = os.path.join(audio_base_path, audio_filename)
                data = get_whitenoise(sample_rate=cls.sample_rate, duration=6, n_channels=1, dtype='float32')
                save_wav(audio_path, data, cls.sample_rate)

                # Append data entry
                cls.data.append((normalize_wav(data), cls.sample_rate, dict(zip(cls._headers, text_data))))

    def test_commonvoice(self):
        dataset = commonvoice.COMMONVOICE(self.root_dir)
        n_ite = 0
        for i, (waveform, sample_rate, dictionary) in enumerate(dataset):
            expected_dictionary = self.data[i][2]
            expected_data = self.data[i][0]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == TestCommonVoice.sample_rate
            assert dictionary == expected_dictionary
            n_ite += 1
        assert n_ite == len(self.data)
