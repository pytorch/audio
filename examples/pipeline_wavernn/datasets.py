import os
import random

import torch
import torchaudio
from torch.utils.data.dataset import random_split
from torchaudio.datasets import LJSPEECH
from torchaudio.transforms import MuLawEncoding

from functional import label_to_waveform, waveform_to_label


class MapMemoryCache(torch.utils.data.Dataset):
    r"""Wrap a dataset so that, whenever a new item is returned, it is saved to memory.
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
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, key):
        item = self.dataset[key]
        return self.process_datapoint(item)

    def __len__(self):
        return len(self.dataset)

    def process_datapoint(self, waveform):
        specgram = self.transforms(waveform[0])
        return waveform[0].squeeze(0), specgram


def split_process_ljspeech(args, transforms):
    data = LJSPEECH(root=args.file_path, download=False)

    val_length = int(len(data) * args.val_ratio)
    lengths = [len(data) - val_length, val_length]
    train_dataset, val_dataset = random_split(data, lengths)

    train_dataset = Processed(train_dataset, transforms)
    val_dataset = Processed(val_dataset, transforms)

    train_dataset = MapMemoryCache(train_dataset)
    val_dataset = MapMemoryCache(val_dataset)

    return train_dataset, val_dataset


def collate_factory(args):
    def raw_collate(batch):

        pad = (args.kernel_size - 1) // 2

        # input waveform length
        wave_length = args.hop_length * args.seq_len_factor
        # input spectrogram length
        spec_length = args.seq_len_factor + pad * 2

        # max start postion in spectrogram
        max_offsets = [x[1].shape[-1] - (spec_length + pad * 2) for x in batch]

        # random start postion in spectrogram
        spec_offsets = [random.randint(0, offset) for offset in max_offsets]
        # random start postion in waveform
        wave_offsets = [(offset + pad) * args.hop_length for offset in spec_offsets]

        waveform_combine = [
            x[0][wave_offsets[i]: wave_offsets[i] + wave_length + 1]
            for i, x in enumerate(batch)
        ]
        specgram = [
            x[1][:, spec_offsets[i]: spec_offsets[i] + spec_length]
            for i, x in enumerate(batch)
        ]

        specgram = torch.stack(specgram)
        waveform_combine = torch.stack(waveform_combine)

        waveform = waveform_combine[:, :wave_length]
        target = waveform_combine[:, 1:]

        # waveform: [-1, 1], target: [0, 2**bits-1] if mode = 'waveform'
        if args.mode == "waveform":

            if args.mulaw:
                mulaw_encode = MuLawEncoding(2 ** args.n_bits)
                waveform = mulaw_encode(waveform)
                target = mulaw_encode(waveform)

                waveform = label_to_waveform(waveform, args.n_bits)

            else:
                target = waveform_to_label(target, args.n_bits)

        return waveform.unsqueeze(1), specgram.unsqueeze(1), target.unsqueeze(1)

    return raw_collate
