import os
import random
import torch
import torchaudio
from torchaudio.datasets import LJSPEECH


class ProcessedLJSPEECH(LJSPEECH):

    def __init__(self,
                 files,
                 transforms,
                 mode,
                 n_bits):

        self.transforms = transforms
        self.files = files
        self.mode = mode
        self.n_bits = n_bits

    def __getitem__(self, index):

        file = self.files[index]
        x, sample_rate = torchaudio.load(file)
        mel = self.transforms(x)

        bits = 16 if self.mode == 'MOL' else self.n_bits

        x = (x + 1.) * (2 ** bits - 1) / 2
        x = torch.clamp(x, min=0, max=2 ** bits - 1)

        return mel.squeeze(0), x.int().squeeze(0)

    def __len__(self):
        return len(self.files)


def datasets_ljspeech(args, transforms):

    root = args.file_path
    wavefiles = [os.path.join(root, file) for file in os.listdir(root)]

    random.seed(args.seed)
    random.shuffle(wavefiles)

    train_files = wavefiles[:-args.test_samples]
    test_files = wavefiles[-args.test_samples:]

    train_dataset = ProcessedLJSPEECH(train_files, transforms, args.mode, args.n_bits)
    test_dataset = ProcessedLJSPEECH(test_files, transforms, args.mode, args.n_bits)

    return train_dataset, test_dataset


def collate_factory(args):

    def raw_collate(batch):

        pad = (args.kernel_size - 1) // 2
        seq_len = args.hop_length * args.seq_len_factor
        mel_win = args.seq_len_factor + 2 * pad

        max_offsets = [x[0].shape[-1] - (mel_win + 2 * pad) for x in batch]
        mel_offsets = [random.randint(0, offset) for offset in max_offsets]
        wav_offsets = [(offset + pad) * args.hop_length for offset in mel_offsets]

        mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]
        waves = [x[1][wav_offsets[i]:wav_offsets[i] + seq_len + 1] for i, x in enumerate(batch)]

        mels = torch.stack(mels)
        waves = torch.stack(waves).long()

        x_input = waves[:, :seq_len]
        y_coarse = waves[:, 1:]

        return x_input, mels, y_coarse

    return raw_collate
