import logging
import os
import shutil
from collections import defaultdict, deque

import torch


class MetricLogger:
    def __init__(self, group, print_freq=1):
        self.print_freq = print_freq
        self._iter = 0
        self.data = defaultdict(lambda: deque(maxlen=self.print_freq))
        self.data["group"].append(group)

    def __setitem__(self, key, value):
        self.data[key].append(value)

    def _get_last(self):
        return {k: v[-1] for k, v in self.data.items()}

    def __str__(self):
        return str(self._get_last())

    def __call__(self):
        self._iter = (self._iter + 1) % self.print_freq
        if not self._iter:
            print(self, flush=True)


def save_checkpoint(state, is_best, filename, rank):
    """
    Save the model to a temporary file first,
    then copy it to filename, in case the signal interrupts
    the torch.save() process.
    """

    if rank != 0:
        return

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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
