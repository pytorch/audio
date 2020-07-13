import argparse
import logging
import os
import signal
from collections import defaultdict
from datetime import datetime
from time import time
from typing import List

import torch
import torchaudio
from torch import nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchaudio.datasets.utils import bg_iterator
from torchaudio.models._wavernn import _WaveRNN

from datasets import collate_factory, datasets_ljspeech
from losses import MoLLoss
from utils import MetricLogger, count_parameters, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint",
    )
    parser.add_argument(
        "--epochs",
        default=8000,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="manual epoch number"
    )
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency in epochs",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--learning-rate", default=1e-4, type=float, metavar="LR", help="learning rate",
    )
    parser.add_argument("--clip-grad", metavar="NORM", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument(
        "--mulaw",
        default=True,
        action="store_true",
        help="if used, waveform is mulaw encoded",
    )
    parser.add_argument(
        "--jit", default=False, action="store_true", help="if used, model is jitted"
    )
    parser.add_argument(
        "--upsample-scales",
        default=[5, 5, 11],
        type=List[int],
        help="the list of upsample scales",
    )
    parser.add_argument(
        "--n-bits", default=8, type=int, help="the bits of output waveform",
    )
    parser.add_argument(
        "--sample-rate",
        default=22050,
        type=int,
        help="the rate of audio dimensions (samples per second)",
    )
    parser.add_argument(
        "--hop-length",
        default=275,
        type=int,
        help="the number of samples between the starts of consecutive frames",
    )
    parser.add_argument(
        "--win-length",
        default=1100,
        type=int,
        help="the number of samples between the starts of consecutive frames",
    )
    parser.add_argument(
        "--f-min",
        default=40.0,
        type=float,
        help="the number of samples between the starts of consecutive frames",
    )
    parser.add_argument(
        "--min-level-db",
        default=-100,
        type=float,
        help="the min db value for spectrogam normalization",
    )
    parser.add_argument(
        "--n-res-block", default=10, type=int, help="the number of ResBlock in stack",
    )
    parser.add_argument(
        "--n-rnn", default=512, type=int, help="the dimension of RNN layer",
    )
    parser.add_argument(
        "--n-fc", default=512, type=int, help="the dimension of fully connected layer",
    )
    parser.add_argument(
        "--kernel-size",
        default=5,
        type=int,
        help="the number of kernel size in the first Conv1d layer",
    )
    parser.add_argument(
        "--n-freq", default=80, type=int, help="the number of bins in a spectrogram",
    )
    parser.add_argument(
        "--n-hidden", default=128, type=int, help="the number of hidden dimensions",
    )
    parser.add_argument(
        "--n-output", default=128, type=int, help="the number of output dimensions",
    )
    parser.add_argument(
        "--mode",
        default="waveform",
        choices=["waveform", "mol"],
        type=str,
        help="the mode of waveform",
    )
    parser.add_argument(
        "--seq-len-factor",
        default=5,
        type=int,
        help="seq_length = hop_length * seq_len_factor",
    )
    parser.add_argument(
        "--test-samples",
        default=50,
        type=float,
        help="the number of waveforms for testing",
    )
    parser.add_argument(
        "--file-path",
        default="",
        type=str,
        help="the path of audio files",
    )

    args = parser.parse_args()
    return args


def train_one_epoch(model, mode, criterion, optimizer, data_loader, device, epoch):

    model.train()

    sums = defaultdict(lambda: 0.0)
    start1 = time()

    metric = MetricLogger("train_iteration")
    metric("epoch", epoch)

    for waveform, specgram, target in bg_iterator(data_loader, maxsize=2):

        start2 = time()

        waveform = waveform.to(device)
        specgram = specgram.to(device)
        target = target.to(device)

        output = model(waveform, specgram)
        output, target = output.squeeze(1), target.squeeze(1)

        if mode == "waveform":
            output = output.transpose(1, 2)
            target = target.long()

        else:
            # use mol mode
            target = target.unsqueeze(-1)

        loss = criterion(output, target)
        loss_item = loss.item()
        sums["loss"] += loss_item
        metric("loss", loss_item)

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad > 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_grad
            )
            sums["gradient"] += gradient
            metric("gradient", gradient.item())

        optimizer.step()

        metric("iteration", sums["iteration"])
        metric("time", time() - start2)
        metric.print()
        sums["iteration"] += 1

    avg_loss = sums["loss"] / len(data_loader)

    metric = MetricLogger("train_epoch")
    metric("epoch", epoch)
    metric("loss", avg_loss)
    metric("gradient", sums["gradient"] / len(data_loader))
    metric("time", time() - start1)
    metric.print()


def evaluate(model, mode, criterion, data_loader, device, epoch):

    with torch.no_grad():

        model.eval()
        sums = defaultdict(lambda: 0.0)
        start = time()

        for waveform, specgram, target in bg_iterator(data_loader, maxsize=2):

            waveform = waveform.to(device)
            specgram = specgram.to(device)
            target = target.to(device)

            output = model(waveform, specgram)

            if mode == "waveform":
                output = output.transpose(1, 2)
                target = target.long()

            else:
                # use mol mode
                target = target.unsqueeze(-1)

            loss = criterion(output, target)
            sums["loss"] += loss.item()

        avg_loss = sums["loss"] / len(data_loader)

        metric = MetricLogger("validation")
        metric("epoch", epoch)
        metric("loss", avg_loss)
        metric("time", time() - start)
        metric.print()

        return avg_loss


def main(args):

    devices = ["cuda" if torch.cuda.is_available() else "cpu"]

    logging.info("Start time: {}".format(str(datetime.now())))

    # Empty CUDA cache
    torch.cuda.empty_cache()

    melkwargs = {
        "n_fft": 2048,
        "power": 1,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
    }

    transforms = torch.nn.Sequential(torchaudio.transforms.Spectrogram(**melkwargs))

    train_dataset, test_dataset = datasets_ljspeech(args, transforms)

    loader_training_params = {
        "num_workers": args.workers,
        "pin_memory": False,
        "shuffle": True,
        "drop_last": False,
    }
    loader_validation_params = loader_training_params.copy()
    loader_validation_params["shuffle"] = False

    collate_fn = collate_factory(args)

    loader_training = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_training_params,
    )
    loader_test = DataLoader(
        test_dataset, batch_size=1, collate_fn=collate_fn, **loader_validation_params,
    )

    model = _WaveRNN(
        upsample_scales=args.upsample_scales,
        n_bits=args.n_bits,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_res_block=args.n_res_block,
        n_rnn=args.n_rnn,
        n_fc=args.n_fc,
        kernel_size=args.kernel_size,
        n_freq=args.n_freq,
        n_hidden=args.n_hidden,
        n_output=args.n_output,
        mode=args.mode,
    )

    if args.jit:
        model = torch.jit.script(model)

    model = torch.nn.DataParallel(model)
    model = model.to(devices[0], non_blocking=True)

    n = count_parameters(model)
    logging.info(f"Number of parameters: {n}")

    # Optimizer
    optimizer_params = {
        "lr": args.learning_rate,
    }

    optimizer = Adam(model.parameters(), **optimizer_params)

    criterion = nn.CrossEntropyLoss() if args.mode == "waveform" else MoLLoss

    best_loss = 10.0

    load_checkpoint = args.checkpoint and os.path.isfile(args.checkpoint)

    if load_checkpoint:
        logging.info(f"Checkpoint: loading '{args.checkpoint}'")
        checkpoint = torch.load(args.checkpoint)

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        logging.info(
            f"Checkpoint: loaded '{args.checkpoint}' at epoch {checkpoint['epoch']}"
        )
    else:
        logging.info("Checkpoint: not found")

        save_checkpoint(
            {
                "epoch": args.start_epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            },
            False,
            args.checkpoint,
        )

    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(
            model, args.mode, criterion, optimizer, loader_training, devices[0], epoch,
        )

        if not (epoch + 1) % args.print_freq or epoch == args.epochs - 1:

            sum_loss = evaluate(
                model, args.mode, criterion, loader_test, devices[0], epoch,
            )

            is_best = sum_loss < best_loss
            best_loss = min(sum_loss, best_loss)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.checkpoint,
            )

    logging.info(f"End time: {datetime.now()}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
