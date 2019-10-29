import os

import torch
from torch.utils.data import Dataset
from torchaudio.datasets.utils import makedir_exist_ok


class DiskCache(Dataset):
    """
    Wrap a dataset so that, whenever a new item is returned, it is saved to disk.
    """

    def __init__(self, dataset, location=".cached"):
        self.dataset = dataset
        self.location = location

        self._id = id(self)
        self._cache = [None] * len(dataset)

    def __getitem__(self, n):
        if self._cache[n]:
            f = self._cache[n]
            return torch.load(f)

        f = str(self._id) + "-" + str(n)
        f = os.path.join(self.location, f)
        item = self.dataset[n]

        self._cache[n] = f
        makedir_exist_ok(self.location)
        torch.save(item, f)

        return item

    def __len__(self):
        return len(self.dataset)
