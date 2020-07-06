import argparse
import logging
import os
import signal
from collections import defaultdict
from datetime import datetime
from time import time

import torch
import torch.nn as nn
import torchaudio
from transform import Transform
from datasets import datasets_ljspeech, collate_factory
from typing import List
from torchaudio.models import _WaveRNN
from torch.utils.data import DataLoader
from torchaudio.datasets.utils import bg_iterator
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.optim import Adam
from mol_loss import Mol_Loss
from utils import MetricLogger, count_parameters, save_checkpoint

SIGNAL_RECEIVED = False


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
        default="checkpoint.pth.tar",
        type=str,
        metavar="FILE",
        help="filename to latest checkpoint",
    )
    parser.add_argument(
        "--epochs",
        default=3000,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number"
    )
    parser.add_argument(
        "--print-freq",
        default=100,
        type=int,
        metavar="N",
        help="print frequency in epochs",
    )
    parser.add_argument(
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size"
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay"
    )
    parser.add_argument(
        "--adam-beta1",
        default=0.9,
        type=float,
        metavar="AD1",
        help="adam_beta1"
    )
    parser.add_argument(
        "--adam-beta2",
        default=0.999,
        type=float,
        metavar="AD2",
        help="adam_beta2"
    )
    parser.add_argument(
        "--eps",
        metavar="EPS",
        type=float,
        default=1e-8
    )
    parser.add_argument(
        "--clip-norm",
        metavar="NORM",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--scheduler",
        metavar="S",
        default="exponential",
        choices=["exponential", "reduceonplateau"],
        help="optimizer to use",
    )
    parser.add_argument(
        "--gamma",
        default=0.999,
        type=float,
        metavar="GAMMA",
        help="learning rate exponential decay constant",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="random seed"
    )
    parser.add_argument(
        "--progress-bar",
        default=False,
        action="store_true",
        help="use progress bar while training"
    )
    parser.add_argument(
        "--mulaw",
        default=True,
        action="store_true",
        help="if used, waveform is mulaw encoded"
    )
    parser.add_argument(
        "--jit",
        default=False,
        action="store_true",
        help="if used, model is jitted"
    )
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH',
        help='path to latest checkpoint'
    )
    # the product of `upsample_scales` must equal `hop_length`
    parser.add_argument(
        "--upsample-scales",
        default=[5, 5, 11],
        type=List[int],
        help="the list of upsample scales",
    )
    parser.add_argument(
        "--n-bits",
        default=9,
        type=int,
        help="the bits of output waveform",
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
        default=40.,
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
        "--n-res-block",
        default=10,
        type=int,
        help="the number of ResBlock in stack",
    )
    parser.add_argument(
        "--n-rnn",
        default=512,
        type=int,
        help="the dimension of RNN layer",
    )
    parser.add_argument(
        "--n-fc",
        default=512,
        type=int,
        help="the dimension of fully connected layer",
    )
    parser.add_argument(
        "--kernel-size",
        default=5,
        type=int,
        help="the number of kernel size in the first Conv1d layer",
    )
    parser.add_argument(
        "--n-freq",
        default=80,
        type=int,
        help="the number of bins in a spectrogram",
    )
    parser.add_argument(
        "--n-hidden",
        default=128,
        type=int,
        help="the number of hidden dimensions",
    )
    parser.add_argument(
        "--n-output",
        default=128,
        type=int,
        help="the number of output dimensions",
    )
    parser.add_argument(
        "--mode",
        default="waveform",
        choices=["waveform", "mol"],
        type=str,
        help="the type of waveform",
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
        default="/private/home/jimchen90/datasets/LJSpeech-1.1/wavs/",
        type=str,
        help="the path of audio files",
    )

    args = parser.parse_args()
    return args


def signal_handler(a, b):
    global SIGNAL_RECEIVED
    print("Signal received", a, datetime.now().strftime("%y%m%d.%H%M%S"), flush=True)
    SIGNAL_RECEIVED = True


def train_one_epoch(
    model, mode, criterion, optimizer, scheduler, data_loader, device, epoch
):

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

        if mode == 'waveform':
            output = output.transpose(1, 2)
            target = target.long()

        elif mode == 'mol':
            target = target.unsqueeze(-1)

        else:
            raise ValueError(
                f"Expected mode: `waveform` or `mol`, but found {mode}"
            )

        loss = criterion(output, target)
        loss_item = loss.item()
        sums["loss"] += loss_item
        metric("loss", loss_item)

        optimizer.zero_grad()
        loss.backward()

        if args.clip_norm > 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_norm
            )
            sums["gradient"] += gradient
            metric("gradient", gradient.item())

        optimizer.step()

        metric("iteration", sums["iteration"])
        metric("time", time() - start2)
        metric.print()
        sums["iteration"] += 1

        if SIGNAL_RECEIVED:
            return

    avg_loss = sums["loss"] / len(data_loader)

    metric = MetricLogger("train_epoch")
    metric("epoch", epoch)
    metric("loss", avg_loss)
    if "gradient" in sums:
        metric("gradient", sums["gradient"] / len(data_loader))
    metric("lr", scheduler.get_last_lr()[0])
    metric("time", time() - start1)
    metric.print()

    scheduler.step()


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

            if mode == 'waveform':
                output = output.transpose(1, 2)
                target = target.long()

            elif mode == 'mol':
                target = target.unsqueeze(-1)

            else:
                raise ValueError(
                    f"Expected mode: `waveform` or `mol`, but found {mode}"
                )

            loss = criterion(output, target)
            sums["loss"] += loss.item()

            if SIGNAL_RECEIVED:
                break

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

    # Install signal handler
    signal.signal(signal.SIGUSR1, lambda a, b: signal_handler(a, b))

    # use torchaudio transform to get waveform and specgram

#    melkwargs = {
#        "n_fft": 2048,
#        "n_mels": args.n_freq,
#        "hop_length": args.hop_length,
#        "f_min": args.f_min,
#        "win_length": args.win_length
#    }

#    transforms = torch.nn.Sequential(
#        torchaudio.transforms.MelSpectrogram(
#            sample_rate=args.sample_rate, **melkwargs
#        ),
#    )

    # use librosa transform to get waveform and specgram

    melkwargs = {
        "n_fft": 2048,
        "num_mels": args.n_freq,
        "hop_length": args.hop_length,
        "fmin": args.f_min,
        "win_length": args.win_length,
        "sample_rate": args.sample_rate,
        "min_level_db": args.min_level_db
    }
    transforms = Transform(**melkwargs)

    # dataset
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
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_validation_params,
    )

    # model
    model = _WaveRNN(upsample_scales=args.upsample_scales,
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
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    }

    optimizer = Adam(model.parameters(), **optimizer_params)

    if args.scheduler == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "reduceonplateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=10, threshold=1e-3)

    criterion = nn.CrossEntropyLoss() if args.mode == 'waveform' else Mol_Loss

    best_loss = 10.

    load_checkpoint = args.checkpoint and os.path.isfile(args.checkpoint)

    if load_checkpoint:
        logging.info(f"Checkpoint: loading '{args.checkpoint}'")
        checkpoint = torch.load(args.checkpoint)

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

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
                "scheduler": scheduler.state_dict(),
            },
            False,
            args.checkpoint,
        )

    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(
            model, args.mode, criterion, optimizer, scheduler, loader_training, devices[0], epoch,
        )

        if SIGNAL_RECEIVED:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, False, args.checkpoint)

        if not (epoch + 1) % args.print_freq or epoch == args.epochs - 1:

            sum_loss = evaluate(
                model,
                args.mode,
                criterion,
                loader_test,
                devices[0],
                epoch,
            )

            is_best = sum_loss < best_loss
            best_loss = min(sum_loss, best_loss)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                is_best,
                args.checkpoint,
            )

    logging.info(f"End time: {datetime.now()}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    main(args)
