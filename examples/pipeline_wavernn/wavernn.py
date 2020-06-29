import argparse
import os
import shutil
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torchaudio
from transform import Transform
from datasets import datasets_ljspeech, collate_factory
from typing import List
from torchaudio.models import _WaveRNN
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from loss_mol import LossFn_Mol


def parse_args():
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint.pth.par",
        type=str,
        metavar="FILE",
        help="filename to latest checkpoint",
    )
    parser.add_argument(
        "--epochs",
        default=2000,
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
        default=32,
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
        metavar="BETA1",
        help="adam_beta1"
    )
    parser.add_argument(
        "--adam-beta2",
        default=0.999,
        type=float,
        metavar="BETA2",
        help="adam_beta2"
    )
    parser.add_argument(
        "--eps",
        default=1e-8,
        type=float,
        metavar="EPS",
        help="eps")
    parser.add_argument(
        "--clip-norm",
        metavar="NORM",
        type=float,
        default=4.0,
        help="clip norm value")

    parser.add_argument("--seed", type=int, default=1000, help="random seed")
    parser.add_argument("--progress-bar", default=False, action="store_true", help="use progress bar while training")
    parser.add_argument("--mulaw", default=True, action="store_true", help="if used, waveform is mulaw encoded")
    parser.add_argument("--jit", default=False, action="store_true", help="if used, model is jitted")
    # parser.add_argument("--distributed", default=False, action="store_true", help="enable DistributedDataParallel")

    # model parameters

    # the product of upsample_scales must equal hop_length
    parser.add_argument(
        "--upsample-scales",
        default=[5, 5, 11],
        type=List[int],
        help="the list of upsample scales",
    )
    # output waveform bits
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
        help="the dimension of fully connected layer ",
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
    # mode = ['waveform', 'mol']
    parser.add_argument(
        "--mode",
        default="mol",
        type=str,
        help="the mode of waveform",
    )
    # the length of input waveform and spectrogram
    parser.add_argument(
        "--seq-len-factor",
        default=5,
        type=int,
        help="seq_length = hop_length * seq_len_factor",
    )
    # the number of waveforms for testing
    parser.add_argument(
        "--test-samples",
        default=50,
        type=float,
        help="the number of test waveforms",
    )
    # the path to store audio files
    parser.add_argument(
        "--file-path",
        default="/private/home/jimchen90/datasets/LJSpeech-1.1/wavs/",
        type=str,
        help="the path of audio files",
    )

    args = parser.parse_args()
    return args


# From wav2letter pipeline:
# https://github.com/vincentqb/audio/blob/wav2letter/examples/pipeline/wav2letter.py
def save_checkpoint(state, is_best, filename):

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


# count parameter numbers in model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# train one epoch
def train_one_epoch(model, mode, bits, mulaw, criterion, optimizer, data_loader, device, pbar=None):
    model.train()

    sums = defaultdict(lambda: 0.0)

    for i, (waveform, specgram, target) in enumerate(data_loader, 1):
        waveform = waveform.to(device, non_blocking=True)
        specgram = specgram.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(waveform, specgram)

        if mode == 'waveform':
            # (n_batch, 2 ** n_bits, n_time)
            output = output.transpose(1, 2)
            target = target.long()

        elif mode == 'mol':
            # (n_batch, n_time, 1)
            target = target.unsqueeze(-1)

        else:
            raise ValueError(f"Expected mode: `waveform` or `mol`, but found {mode}")

        loss = criterion(output, target)
        sums["loss"] += loss.item()

        optimizer.zero_grad()
        loss.backward()

        if args.clip_norm > 0:
            sums["gradient"] += torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_norm
            )

        optimizer.step()

        if pbar is not None:
            pbar.update(1 / len(data_loader))

    avg_loss = sums["loss"] / len(data_loader)
    print(f"Training loss: {avg_loss:4.5f}", flush=True)

    if "gradient" in sums:
        avg_gradient = sums["gradient"] / len(data_loader)
        print(f"Average gradient norm: {avg_gradient:4.8f}", flush=True)


def evaluate(model, mode, bits, mulaw, criterion, data_loader, device):

    with torch.no_grad():

        model.eval()

        sums = defaultdict(lambda: 0.0)

        for i, (waveform, specgram, target) in enumerate(data_loader, 1):
            waveform = waveform.to(device, non_blocking=True)
            specgram = specgram.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(waveform, specgram)

            if mode == 'waveform':
                # (batch, 2 ** bits, seq_len)
                output = output.transpose(1, 2)
                target = target.long()

            elif mode == 'mol':
                # (batch, seq_len, 1)
                target = target.unsqueeze(-1)

            else:
                raise ValueError(f"Expected mode: `waveform` or `mol`, but found {mode}")

            loss = criterion(output, target)
            sums["loss"] += loss.item()

        avg_loss = sums["loss"] / len(data_loader)
        print(f"Validation loss: {avg_loss:.8f}", flush=True)

        return avg_loss


def main(args):

    devices = ["cuda" if torch.cuda.is_available() else "cpu"]

    print("Start time: {}".format(str(datetime.now())), flush=True)

    # Empty CUDA cache
    torch.cuda.empty_cache()

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
#        torchaudio.transforms.MuLawEncoding(2**args.n_bits)
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
        "pin_memory": True,
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
                     mode=args.mode)

#    if args.jit:
#        model = torch.jit.script(model)

#    if not args.distributed:
#        model = torch.nn.DataParallel(model)
#    else:
#        model.cuda()
#        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=devices)
    model = torch.nn.DataParallel(model)
    model = model.to(devices[0], non_blocking=True)

    n = count_parameters(model)
    print(f"Number of parameters: {n}", flush=True)

    # Optimizer

    optimizer_params = {
        "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    }

    optimizer = Adam(model.parameters(), **optimizer_params)

    criterion = nn.CrossEntropyLoss() if args.mode == 'waveform' else LossFn_Mol

    best_loss = 1.0

    load_checkpoint = args.checkpoint and os.path.isfile(args.checkpoint)

    if load_checkpoint:
        print("Checkpoint: loading '{}'".format(args.checkpoint), flush=True)
        checkpoint = torch.load(args.checkpoint)

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])

        print("Checkpoint: loaded '{}' at epoch {}".format(args.checkpoint, checkpoint["epoch"]), flush=True,)

    else:
        print("Checkpoint: not found", flush=True)

        save_checkpoint(
            {
                "epoch": args.start_epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                # "scheduler": scheduler.state_dict(),
            },
            False,
            args.checkpoint,
        )

    with tqdm(total=args.epochs, unit_scale=1, disable=not args.progress_bar) as pbar:

        for epoch in range(args.start_epoch, args.epochs):

            train_one_epoch(
                model,
                args.mode,
                args.n_bits,
                args.mulaw,
                criterion,
                optimizer,
                loader_training,
                devices[0],
                pbar=pbar,
            )

            if not (epoch + 1) % args.print_freq or epoch + 1 == args.epochs:

                sum_loss = evaluate(
                    model,
                    args.mode,
                    args.n_bits,
                    args.mulaw,
                    criterion,
                    loader_test,
                    devices[0],
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
