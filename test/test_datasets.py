import os
import unittest

from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.utils import diskcache_iterator, bg_iterator
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.yesno import YESNO

import common_utils


class TestDatasets(unittest.TestCase):
    test_dirpath, test_dir = common_utils.create_temp_assets_dir()
    path = os.path.join(test_dirpath, "assets")

    def test_yesno(self):
        data = YESNO(self.path, return_dict=True)
        data[0]

    def test_vctk(self):
        data = VCTK(self.path, return_dict=True)
        data[0]

    def test_librispeech(self):
        data = LIBRISPEECH(self.path, "dev-clean")
        data[0]

    def test_commonvoice(self):
        path = os.path.join(self.path, "commonvoice")
        data = COMMONVOICE(path, "train.tsv", "tatar")
        data[0]

    def test_commonvoice_diskcache(self):
        path = os.path.join(self.path, "commonvoice")
        data = COMMONVOICE(path, "train.tsv", "tatar")
        data = diskcache_iterator(data)
        # Save
        data[0]
        # Load
        data[0]

    def test_commonvoice_bg(self):
        path = os.path.join(self.path, "commonvoice")
        data = COMMONVOICE(path, "train.tsv", "tatar")
        data = bg_iterator(data, 5)
        for d in data:
            pass


if __name__ == "__main__":
    unittest.main()
