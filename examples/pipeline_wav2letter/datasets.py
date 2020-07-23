import itertools

import torch
from torchaudio.datasets import LIBRISPEECH


class IterableMemoryCache:
    def __init__(self, iterable):
        self.iterable = iterable
        self._iter = iter(iterable)
        self._done = False
        self._values = []

    def __iter__(self):
        if self._done:
            return iter(self._values)
        return itertools.chain(self._values, self._gen_iter())

    def _gen_iter(self):
        for new_value in self._iter:
            self._values.append(new_value)
            yield new_value
        self._done = True

    def __len__(self):
        return len(self._iterable)


class MapMemoryCache(torch.utils.data.Dataset):
    """
    Wrap a dataset so that, whenever a new item is returned, it is saved to memory.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._cache = [None] * len(dataset)

    def __getitem__(self, n):
        if self._cache[n] is not None:
            return self._cache[n]

        item = self.dataset[n]
        self._cache[n] = item

        return item

    def __len__(self):
        return len(self.dataset)


class ProcessedLIBRISPEECH(LIBRISPEECH):
    def __init__(self, transforms, encode, *args, **kwargs):
        self.transforms = transforms
        self.encode = encode
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        return self.process_datapoint(item)

    def __next__(self):
        item = super().__next__()
        return self.process_datapoint(item)

    def process_datapoint(self, item):
        transformed = item[0]  # .to(device)
        target = item[2].lower()

        transformed = self.transforms(transformed)
        transformed = transformed[0, ...].transpose(0, -1)

        # target = " " + target + " "
        target = self.encode(target)
        target = torch.tensor(target, dtype=torch.long, device=transformed.device)

        # transformed = transformed.to("cpu")
        # target = target.to("cpu")
        return transformed, target


def datasets_librispeech(
    transforms,
    language_model,
    root="/datasets01/",
    folder_in_archive="librispeech/062419/",
):
    def create(tag):

        if isinstance(tag, str):
            tag = [tag]

        data = torch.utils.data.ConcatDataset(
            [
                ProcessedLIBRISPEECH(
                    transforms,
                    language_model.encode,
                    root,
                    t,
                    folder_in_archive=folder_in_archive,
                    download=False,
                )
                for t in tag
            ]
        )

        # data = diskcache_iterator(data)
        data = MapMemoryCache(data)
        return data

    return create("train-clean-100"), create("dev-clean"), None
    # return create(["train-clean-100", "train-clean-360", "train-other-500"]), create(["dev-clean", "dev-other"]), None


def collate_factory(model_length_function):
    def collate_fn(batch):

        tensors = [b[0] for b in batch if b]

        tensors_lengths = torch.tensor(
            [model_length_function(t) for t in tensors],
            dtype=torch.long,
            device=tensors[0].device,
        )

        tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        tensors = tensors.transpose(1, -1)

        targets = [b[1] for b in batch if b]
        target_lengths = torch.tensor(
            [target.shape[0] for target in targets],
            dtype=torch.long,
            device=tensors.device,
        )
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

        return tensors, targets, tensors_lengths, target_lengths

    return collate_fn
