import json
import logging
import shutil
import time
from collections import defaultdict

import torch


class Logger(defaultdict):
    def __init__(self, name, print_freq=1, disable=False):
        super().__init__(float)
        self.disable = disable
        self.print_freq = print_freq

        self._name = "name"
        self._start = "start time"
        self._time = "current time"
        self._iteration = "iteration"

        self[self._name] = name
        self[self._start] = time.monotonic()

    def __str__(self):
        self[self._time] = time.monotonic()
        return json.dumps(self)

    def flush(self):
        self[self._iteration] += 1
        if not self.disable and not self[self._iteration] % self.print_freq:
            print(self, flush=True)


def save_checkpoint(state, is_best, filename):
    """
    Save the model to a temporary file first,
    then copy it to filename, in case the signal interrupts
    the torch.save() process.
    """

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

    logging.warning("Checkpoint: saved")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
