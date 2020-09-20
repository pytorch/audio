import json
import logging
import shutil
import time
from collections import defaultdict

import torch


class Logger(defaultdict):
    def __init__(self, name, print_freq=1, disable=False, filename=None):
        super().__init__(float)

        self.disable = disable
        self.print_freq = print_freq
        self.filename = filename

        self._name = "name"
        self._time = "elapsed time"

        self[self._name] = name
        self[self._time] = time.monotonic()
        self._iteration = 0

    def __str__(self):
        self[self._time] = time.monotonic() - self[self._time]
        return json.dumps(self)

    def flush(self):
        self._iteration += 1
        if not self.disable and not self._iteration % self.print_freq:
            if self.filename:
                self._append_to_file()
            else:
                print(self, flush=True)

    def _append_to_file(self):
        with open(self._filename, "a") as f:
            f.write(self + "\n")


def save_checkpoint(state, is_best, filename):
    """
    Save the model to a temporary file first,
    then copy it to filename, in case the signal interrupts
    the torch.save() process.
    """

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, "best_" + filename)

    logging.warning("Checkpoint: saved")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
