import os

import torch


def average_checkpoints(last):
    avg = None
    for path in last:
        states = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= len(last)
            else:
                avg[k] //= len(last)
    return avg


def ensemble(args):
    last = [
        os.path.join(args.exp_dir, args.experiment_name, f"epoch={n}.ckpt")
        for n in range(args.epochs - 10, args.epochs)
    ]
    model_path = os.path.join(args.exp_dir, args.experiment_name, f"model_avg_10.pth")
    torch.save({"state_dict": average_checkpoints(last)}, model_path)
