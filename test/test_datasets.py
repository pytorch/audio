import unittest
from pprint import pprint

from torchaudio.datasets.commonvoice import COMMONVOICE, COMMONVOICE2
from torchaudio.datasets.librispeech import LIBRISPEECH, LIBRISPEECH2
from torchaudio.datasets.vctk import VCTK, VCTK2
from torchaudio.datasets.yesno import YESNO, YESNO2


class TestDatasets(unittest.TestCase):
    def test_yesno(self):
        data = YESNO("./yesnotest")
        pprint("YESNO")
        pprint(next(data))

    def test_vctk(self):
        data = VCTK("./vctktest/")
        pprint("VCTK")
        pprint(next(data))

    def test_librispeech(self):
        data = LIBRISPEECH("./librispeechtest/", "dev-clean")
        pprint("LIBRISPEECH")
        pprint(next(data))

    def test_commonvoice(self):
        data = COMMONVOICE("./commonvoicetest/", "tatar", "train.tsv")
        pprint("COMMONVOICE")
        pprint(next(data))

    def test_yesno2(self):
        data = YESNO2("./yesnotest")
        pprint("YESNO2")
        for d in data:
            pprint(d)
            break

    def test_vctk2(self):
        data = VCTK2("./vctktest/")
        pprint("VCTK2")
        for d in data:
            pprint(d)
            break

    def test_librispeech2(self):
        data = LIBRISPEECH2("./librispeechtest/", "dev-clean")
        pprint("LIBRISPEECH2")
        for d in data:
            pprint(d)
            break

    def test_commonvoice2(self):
        data = COMMONVOICE2("./commonvoicetest/", "tatar", "train.tsv")
        pprint("COMMONVOICE2")
        for d in data:
            pprint(d)
            break


if __name__ == "__main__":
    unittest.main()
