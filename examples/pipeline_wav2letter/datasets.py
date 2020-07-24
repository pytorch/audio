import torch
from torchaudio.datasets import LIBRISPEECH


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


class Processed(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms, encode):
        self.dataset = dataset
        self.transforms = transforms
        self.encode = encode

    def __getitem__(self, key):
        item = self.dataset[key]
        return self.process_datapoint(item)

    def __len__(self):
        return len(self.dataset)

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


def split_process_librispeech(
    datasets, transforms, language_model, root, folder_in_archive,
):
    def create(tags, cache=True):

        if isinstance(tags, str):
            tags = [tags]
        if isinstance(transforms, list):
            transform_list = transforms
        else:
            transform_list = [transforms]

        data = torch.utils.data.ConcatDataset(
            [
                Processed(
                    LIBRISPEECH(
                        root, tag, folder_in_archive=folder_in_archive, download=False,
                    ),
                    transform,
                    language_model.encode,
                )
                for tag, transform in zip(tags, transform_list)
            ]
        )

        # data = diskcache_iterator(data)
        data = MapMemoryCache(data)
        return data

    # FIXME For performance, we cache all datasets
    # Do not cache first dataset
    # return tuple(create(dataset, cache=i > 0) for i, dataset in enumerate(datasets))
    return tuple(create(dataset) for dataset in datasets)


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
