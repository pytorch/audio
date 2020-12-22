import os
from pathlib import Path

from torchaudio.datasets import utils as dataset_utils
from torchaudio.datasets.commonvoice import COMMONVOICE

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_asset_path,
)


class TestWalkFiles(TempDirMixin, TorchaudioTestCase):
    root = None
    expected = None

    def _add_file(self, *parts):
        path = self.get_temp_path(*parts)
        self.expected.append(path)
        Path(path).touch()

    def setUp(self):
        self.root = self.get_temp_path()
        self.expected = []

        # level 1
        for filename in ['a.txt', 'b.txt', 'c.txt']:
            self._add_file(filename)

        # level 2
        for dir1 in ['d1', 'd2', 'd3']:
            for filename in ['d.txt', 'e.txt', 'f.txt']:
                self._add_file(dir1, filename)
            # level 3
            for dir2 in ['d1', 'd2', 'd3']:
                for filename in ['g.txt', 'h.txt', 'i.txt']:
                    self._add_file(dir1, dir2, filename)

        print('\n'.join(self.expected))

    def test_walk_files(self):
        """walk_files should traverse files in alphabetical order"""
        n_ites = 0
        for i, path in enumerate(dataset_utils.walk_files(self.root, '.txt', prefix=True)):
            found = os.path.join(self.root, path)
            assert found == self.expected[i]
            n_ites += 1
        assert n_ites == len(self.expected)


class TestIterator(TorchaudioTestCase):
    backend = 'default'
    path = get_asset_path('CommonVoice', 'cv-corpus-4-2019-12-10', 'tt')

    def test_disckcache_iterator(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = dataset_utils.diskcache_iterator(data)
        # Save
        data[0]
        # Load
        data[0]

    def test_bg_iterator(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = dataset_utils.bg_iterator(data, 5)
        for _ in data:
            pass
