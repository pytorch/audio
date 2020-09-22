import itertools
from typing import List

import torch
from torch import Tensor
from torchaudio.datasets import LIBRISPEECH

from speechcommands import SPEECHCOMMANDS


def pad_sequence(sequences, padding_value=0.0):
    # type: (List[Tensor], float) -> Tensor
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. If the input is list of
    sequences with size ``* x L`` then the output is and ``B x * x T``.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(300, 25)
        >>> b = torch.ones(300, 22)
        >>> c = torch.ones(300, 15)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([300, 3, 25])

    Note:
        This function returns a Tensor of size ``B x * x T``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``B x * x T``
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[:-1]
    max_len = max([s.size(-1) for s in sequences])
    out_dims = (len(sequences),) + trailing_dims + (max_len,)

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(-1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, ..., :length] = tensor

    return out_tensor


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
        if self._cache[n] is None:
            self._cache[n] = self.dataset[n]
        return self._cache[n]

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
        """
        Consume a LibriSpeech data point tuple:
        (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id).
        - Transforms are applied to waveform. Output tensor shape (freq, time).
        - target gets transformed into lower case, and encoded into a one dimensional long tensor.
        """
        transformed = item[0]
        target = item[2].lower()

        transformed = self.transforms(transformed)

        target = self.encode(target)
        target = torch.tensor(target, dtype=torch.long, device=transformed.device)

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

        data = MapMemoryCache(data)
        return data

    # For performance, we cache all datasets
    return tuple(create(dataset) for dataset in datasets)


def split_process_speechcommands(
    datasets, transforms, language_model, root,
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
                    SPEECHCOMMANDS(
                        root, split=tag, download=False,
                    ),
                    transform,
                    language_model.encode,
                )
                for tag, transform in zip(tags, transform_list)
            ]
        )

        data = MapMemoryCache(data)
        return data

    # For performance, we cache all datasets
    return tuple(create(dataset) for dataset in datasets)


def collate_factory(model_length_function, transforms=None):

    if transforms is None:
        transforms = torch.nn.Sequential()

    def collate_fn(batch):

        tensors = [transforms(b[0]) for b in batch]  # apply transforms to waveforms

        tensors_lengths = torch.tensor(
            [model_length_function(t) for t in tensors],
            dtype=torch.long,
            device=tensors[0].device,
        )

        # tensors = [b.transpose(1, -1) for b in batch]
        # tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # tensors = tensors.transpose(1, -1)
        tensors = pad_sequence(tensors)

        targets = [b[1] for b in batch]  # extract target utterance
        target_lengths = torch.tensor(
            [target.shape[0] for target in targets],
            dtype=torch.long,
            device=tensors.device,
        )
        # targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
        targets = pad_sequence(targets)

        return tensors, targets, tensors_lengths, target_lengths

    return collate_fn
