import os
import shutil
import sys
from collections import defaultdict, deque

import torch


class MetricLogger:
    def __init__(self, group, print_freq=1):
        self.print_freq = print_freq
        self.data = defaultdict(lambda: deque(maxlen=self.print_freq))
        self.data["group"].append(group)
        self._iter = 0

    def __call__(self, key, value):
        self.data[key].append(value)

    def print(self):
        self._iter += 1
        if self._iter % self.print_freq:
            # d = {k: statistics.mean(v) for k, v in self.data.items()}
            d = {k: v[-1] for k, v in self.data.items()}
            print(d, flush=True)


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
    print("Checkpoint: saved", file=sys.stderr, flush=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
