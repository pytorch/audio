import os
import random

import torch
import torchaudio
from torchaudio.datasets import LJSPEECH

from transform import linear_to_mel
from utils import (label_to_waveform, mulaw_encode, specgram_normalize,
                   waveform_to_label)


class ProcessedLJSPEECH(LJSPEECH):
    def __init__(self, files, transforms, args):

        self.transforms = transforms
        self.files = files
        self.args = args

    def __getitem__(self, index):

        file = self.files[index]
        args = self.args
        n_fft = 2048
        waveform, sample_rate = torchaudio.load(file)
        specgram = self.transforms(waveform)
        specgram = linear_to_mel(specgram, sample_rate, n_fft, args.n_freq, args.f_min)
        specgram = specgram_normalize(specgram, args.min_level_db)
        waveform = waveform.squeeze(0)

        if args.mode == "waveform":
            waveform = (
                mulaw_encode(waveform, 2 ** args.n_bits)
                if args.mulaw
                else waveform_to_label(waveform, args.n_bits)
            )

        elif args.mode == "mol":
            waveform = waveform_to_label(waveform, 16)

        return waveform, specgram

    def __len__(self):
        return len(self.files)


def datasets_ljspeech(args, transforms):

    root = args.file_path
    wavefiles = [os.path.join(root, file) for file in os.listdir(root)]

    random.seed(args.seed)
    random.shuffle(wavefiles)

    train_files = wavefiles[: -args.test_samples]
    test_files = wavefiles[-args.test_samples :]

    train_dataset = ProcessedLJSPEECH(train_files, transforms, args)
    test_dataset = ProcessedLJSPEECH(test_files, transforms, args)

    return train_dataset, test_dataset


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
            x[0][wave_offsets[i] : wave_offsets[i] + wave_length + 1]
            for i, x in enumerate(batch)
        ]
        specgram = [
            x[1][:, spec_offsets[i] : spec_offsets[i] + spec_length]
            for i, x in enumerate(batch)
        ]

        specgram = torch.stack(specgram)
        waveform_combine = torch.stack(waveform_combine)

        waveform = waveform_combine[:, :wave_length]
        target = waveform_combine[:, 1:]

        # waveform: [-1, 1], target: [0, 2**bits-1] if mode = 'waveform'
        # waveform: [-1, 1], target: [-1, 1] if mode = 'mol'
        bits = 16 if args.mode == "mol" else args.n_bits

        waveform = label_to_waveform(waveform.float(), bits)

        if args.mode == "mol":
            target = label_to_waveform(target.float(), bits)

        return waveform.unsqueeze(1), specgram.unsqueeze(1), target.unsqueeze(1)

    return raw_collate
