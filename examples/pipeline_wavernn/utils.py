import logging
import os
import shutil
from collections import defaultdict, deque

import torch


class MetricLogger:
    r"""Logger for model metrics
    """

    def __init__(self, group, print_freq=1):
        self.print_freq = print_freq
        self._iter = 0
        self.data = defaultdict(lambda: deque(maxlen=self.print_freq))
        self.data["group"].append(group)

    def __call__(self, key, value):
        self.data[key].append(value)

    def _get_last(self):
        return {k: v[-1] for k, v in self.data.items()}

    def __str__(self):
        return str(self._get_last())

    def print(self):
        self._iter = (self._iter + 1) % self.print_freq
        if not self._iter:
            print(self, flush=True)


def save_checkpoint(state, is_best, filename):
    r"""Save the model to a temporary file first,
    then copy it to filename, in case the signal interrupts
    the torch.save() process.
    """

    if filename == "":
        return

    tempfile = filename + ".temp"

    # Remove tempfile in case interuption during the copying from tempfile to filename
    if os.path.isfile(tempfile):
        os.remove(tempfile)

    torch.save(state, tempfile)
    if os.path.isfile(tempfile):
        os.rename(tempfile, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")
    logging.info("Checkpoint: saved")


def count_parameters(model):
    r"""Count the total number of parameters in the model
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def specgram_normalize(specgram, min_level_db):
    r"""Normalize the spectrogram with a minimum db value
    """

    specgram = 20 * torch.log10(torch.clamp(specgram, min=1e-5))
    return torch.clamp((min_level_db - specgram) / min_level_db, min=0, max=1)


def mulaw_encode(waveform, mu):
    r"""Waveform mulaw encoding
    """

    mu = mu - 1
    fx = (
        torch.sign(waveform)
        * torch.log(1 + mu * torch.abs(waveform))
        / torch.log(torch.as_tensor(1.0 + mu))
    )
    return torch.floor((fx + 1) / 2 * mu + 0.5).int()


def waveform_to_label(waveform, bits):
    r"""Transform waveform [-1, 1] to label [0, 2 ** bits - 1]
    """

    assert abs(waveform).max() <= 1.0
    waveform = (waveform + 1.0) * (2 ** bits - 1) / 2
    return torch.clamp(waveform, 0, 2 ** bits - 1).int()


def label_to_waveform(label, bits):
    r"""Transform label [0, 2 ** bits - 1] to waveform [-1, 1]
    """

    return 2 * label / (2 ** bits - 1.0) - 1.0
