import os
import random
import torch
import torchaudio
from torchaudio.datasets import LJSPEECH


class ProcessedLJSPEECH(LJSPEECH):

    def __init__(self, files, transforms, mode, mulaw, n_bits):

        self.transforms = transforms
        self.files = files
        self.mulaw = mulaw
        self.n_bits = n_bits
        self.mode = mode

    def __getitem__(self, index):

        file = self.files[index]

    # use torchaudio transform to get waveform and specgram
        # waveform, sample_rate = torchaudio.load(file)
        # specgram = self.transforms(x)
        # return waveform.squeeze(0), mel.squeeze(0)

    # use librosa transform to get waveform and specgram
        waveform = self.transforms.load(file)
        specgram = self.transforms.melspectrogram(waveform)

        # waveform and spectrogram: [0, 2**bits-1].
        if self.mode == 'waveform':
            waveform = self.transforms.mulaw_encode(waveform, 2**self.n_bits) if self.mulaw \
                else float_2_int(waveform, self.n_bits)

        elif self.mode == 'mol':
            waveform = float_2_int(waveform, 16)

        return waveform, specgram

    def __len__(self):
        return len(self.files)


# From float waveform [-1, 1] to integer label [0, 2 ** bits - 1]
def float_2_int(waveform, bits):
    assert abs(waveform).max() <= 1.0
    waveform = (waveform + 1.) * (2**bits - 1) / 2
    return torch.clamp(waveform, 0, 2**bits - 1).int()


# From integer label [0, 2 ** bits - 1] to float waveform [-1, 1]
def int_2_float(waveform, bits):
    return 2 * waveform / (2**bits - 1.) - 1.


def datasets_ljspeech(args, transforms):

    root = args.file_path
    wavefiles = [os.path.join(root, file) for file in os.listdir(root)]

    random.seed(args.seed)
    random.shuffle(wavefiles)

    train_files = wavefiles[:-args.test_samples]
    test_files = wavefiles[-args.test_samples:]

    train_dataset = ProcessedLJSPEECH(train_files, transforms, args.mode, args.mulaw, args.n_bits)
    test_dataset = ProcessedLJSPEECH(test_files, transforms, args.mode, args.mulaw, args.n_bits)

    return train_dataset, test_dataset


def collate_factory(args):

    def raw_collate(batch):

        pad = (args.kernel_size - 1) // 2
        # input sequence length, increase seq_len_factor to increase it.
        wave_length = args.hop_length * args.seq_len_factor
        # input spectrogram length
        spec_length = args.seq_len_factor + pad * 2

        # max start postion in spectrogram
        max_offsets = [x[1].shape[-1] - (spec_length + pad * 2) for x in batch]

        # random start postion in spectrogram
        spec_offsets = [random.randint(0, offset) for offset in max_offsets]

        # random start postion in waveform
        wave_offsets = [(offset + pad) * args.hop_length for offset in spec_offsets]

        waveform_combine = [x[0][wave_offsets[i]:wave_offsets[i] + wave_length + 1] for i, x in enumerate(batch)]
        specgram = [x[1][:, spec_offsets[i]:spec_offsets[i] + spec_length] for i, x in enumerate(batch)]

        # stack batch
        specgram = torch.stack(specgram)
        waveform_combine = torch.stack(waveform_combine)

        waveform = waveform_combine[:, :wave_length]
        target = waveform_combine[:, 1:]

        # waveform: [-1, 1], target: [0, 2**bits-1] if mode = 'waveform'
        # waveform: [-1, 1], target: [-1, 1] if mode = 'mol'
        bits = 16 if args.mode == 'mol' else args.n_bits

        waveform = int_2_float(waveform.float(), bits)

        if args.mode == 'mol':
            target = int_2_float(target.float(), bits)

        return waveform, specgram, target

    return raw_collate
