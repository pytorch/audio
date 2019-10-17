import unittest
from pprint import pprint

from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.yesno import YESNO


class TestDatasets(unittest.TestCase):
    def test_yesno(self):
        data = YESNO("./yesnotest")
        pprint("YESNO")
        for d in data:
            pprint(d)
            break

    def test_vctk(self):
        data = VCTK("./vctktest/")
        pprint("VCTK")
        for d in data:
            pprint(d)
            break

    def test_librispeech(self):
        data = LIBRISPEECH("./librispeechtest/", "dev-clean")
        pprint("LIBRISPEECH")
        for d in data:
            pprint(d)
            break

    def test_commonvoice(self):
        data = COMMONVOICE("./commonvoicetest/", "tatar", "train.tsv")
        pprint("COMMONVOICE")
        for d in data:
            pprint(d)
            break


if __name__ == "__main__":
    unittest.main()
