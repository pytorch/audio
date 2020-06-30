import csv
import json
import os
import shutil
from collections import defaultdict

import torch
from tabulate import tabulate


class MetricLogger:
    def __init__(self, log=None, disable=False):
        self.disable = disable
        self.log = defaultdict(lambda: defaultdict(list)) if log is None else log

    def record(self, group, metric, value, msg=None):
        if not self.disable:

            self.log[group][metric].append(value)
            if msg is not None:
                print(msg, "{: >10}".format(round(value, 5)), flush=True)

        return value

    def print_last_row(self, group=None):
        if self.disable:
            return
        for group in group or self.log:
            # print({k: v[-1] for k, v in group.items()})
            print(
                tabulate({k: v[-1] for k, v in group.items()}, headers="keys"),
                flush=True,
            )

    def print_all_row(self, group=None):
        if self.disable:
            return
        for group in group or self.log:
            # print({k: v[-1] for k, v in group.items()})
            print(tabulate(self.log[group], flush=True))

    def write_csv(self, prefix=""):
        if self.disable:
            return
        for group in self.log:
            filename = prefix + group + ".csv"
            content = tabulate(self.log[group])
            with open(filename, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(content)

    def write_json(self, filename):
        if self.disable:
            return
        with open(filename, "w") as outfile:
            json.dump(self.log, outfile)


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
    print("Checkpoint: saved", flush=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
