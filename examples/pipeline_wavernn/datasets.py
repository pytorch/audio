import os
import random

import torch
import torchaudio
from torchaudio.datasets import LJSPEECH

from transform import linear_to_mel
from functional import (
    label_to_waveform,
    mulaw_encode,
    specgram_normalize,
    waveform_to_label,
)


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


class ProcessedLJSPEECH(LJSPEECH):
    def __init__(self, dataset, transforms, args):
        self.dataset = dataset
        self.transforms = transforms
        self.args = args

    def __getitem__(self, index):
        filename = self.dataset[index][0]
        folder = "LJSpeech-1.1/wavs/"
        file = os.path.join(self.args.file_path, folder, filename + ".wav")

        return self.process_datapoint(file)

    def __len__(self):
        return len(self.dataset)

    def process_datapoint(self, file):
        args = self.args
        n_fft = 2048
        waveform, sample_rate = torchaudio.load(file)
        specgram = self.transforms(waveform)
        # TODO Replace by torchaudio, once https://github.com/pytorch/audio/pull/593 is resolved
        specgram = linear_to_mel(specgram, sample_rate, n_fft, args.n_freq, args.f_min)
        specgram = specgram_normalize(specgram, args.min_level_db)
        waveform = waveform.squeeze(0)

        if args.mode == "waveform":
            waveform = (
                mulaw_encode(waveform, 2 ** args.n_bits)
                if args.mulaw
                else waveform_to_label(waveform, args.n_bits)
            )

        return waveform, specgram


def split_data(data, val_ratio, seed):
    dataset = data._walker

    random.seed(seed)
    random.shuffle(dataset)

    train_dataset = dataset[: -int(val_ratio * len(dataset))]
    val_dataset = dataset[-int(val_ratio * len(dataset)):]

    return train_dataset, val_dataset


def gen_datasets_ljspeech(
    args, transforms,
):
    data = LJSPEECH(root=args.file_path, download=False)

    train_dataset, val_dataset = split_data(data, args.val_ratio, args.seed)

    train_dataset = ProcessedLJSPEECH(train_dataset, transforms, args)
    val_dataset = ProcessedLJSPEECH(val_dataset, transforms, args)

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
        # waveform: [-1, 1], target: [-1, 1] if mode = 'mol'
        if args.mode == "waveform":
            waveform = label_to_waveform(waveform.float(), args.n_bits)

        return waveform.unsqueeze(1), specgram.unsqueeze(1), target.unsqueeze(1)

    return raw_collate
