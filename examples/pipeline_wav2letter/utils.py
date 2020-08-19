import logging
import os
import shutil
from collections import defaultdict, deque

import torch


class MetricLogger(defaultdict):
    def __init__(self, name, print_freq=1):
        self.print_freq = print_freq
        self._iter = 0
        super().__init__(lambda: 0.)
        self["name"] = name

    def __str__(self):
        return str(dict(self.data))

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
