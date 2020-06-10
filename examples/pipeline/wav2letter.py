import argparse
import os
import shutil
import signal
import string
from collections import defaultdict
from datetime import datetime

import torch
import torchaudio
from torch.optim import SGD, Adadelta, Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchaudio.datasets.utils import bg_iterator, diskcache_iterator
from torchaudio.models.wav2letter import Wav2Letter
from torchaudio.transforms import MFCC, Resample
from tqdm import tqdm

from ctc_decoders import GreedyDecoder, ViterbiDecoder
from datasets import collate_factory, datasets_librispeech
from languagemodels import LanguageModel
from metrics import levenshtein_distance


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        default=0,
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
        default=200,
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
        "--progress-bar", action="store_true", help="use progress bar while training"
    )

    parser.add_argument(
        "--decoder",
        metavar="D",
        default="greedy",
        choices=["greedy", "viterbi"],
        help="decoder to use",
    )

    parser.add_argument(
        "--batch-size", default=64, type=int, metavar="N", help="mini-batch size"
    )

    parser.add_argument(
        "--n-bins",
        default=13,
        type=int,
        metavar="N",
        help="number of bins in transforms",
    )
    parser.add_argument(
        "--learning-rate",
        default=1.0,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--gamma",
        default=0.96,
        type=float,
        metavar="GAMMA",
        help="learning rate exponential decay constant",
    )
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument(
        "--weight-decay", default=1e-5, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument("--eps", metavar="EPS", type=float, default=1e-8)
    parser.add_argument("--rho", metavar="RHO", type=float, default=0.95)
    parser.add_argument("--clip-norm", metavar="NORM", type=float, default=0.0)

    parser.add_argument(
        "--dataset",
        default="librispeech",
        type=str,
        help="select dataset to train with",
    )
    parser.add_argument(
        "--distributed", action="store_true", help="enable DistributedDataParallel"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--world-size", type=int, default=8, help="the world size to initiate DPP"
    )

    parser.add_argument("--jit", action="store_true", help="if used, model is jitted")

    args = parser.parse_args()

    return args


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


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


def model_length_function(tensor):
    return int(tensor.shape[0]) // 2 + 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model, criterion, optimizer, scheduler, data_loader, device, pbar=None,
):

    model.train()

    sums = defaultdict(lambda: 0.0)

    for inputs, targets, tensors_lengths, target_lengths in bg_iterator(
        data_loader, maxsize=2
    ):

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # keep batch first for data parallel
        outputs = model(inputs).transpose(-1, -2).transpose(0, 1)

        # CTC
        # outputs: input length, batch size, number of classes (including blank)
        # targets: batch size, max target length
        # input_lengths: batch size
        # target_lengths: batch size

        loss = criterion(outputs, targets, tensors_lengths, target_lengths)
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

    # Average
    for k in sums.keys():
        sums[k] /= len(data_loader)

    print(f"Training loss: {sums['loss']:4.5f}", flush=True)
    if "gradient" in sums:
        print(f"Average gradient norm: {sums['gradient']:4.5f}", flush=True)

    scheduler.step()


def evaluate(model, criterion, data_loader, decoder, language_model, device):

    with torch.no_grad():

        model.eval()

        sums = defaultdict(lambda: 0.0)

        for inputs, targets, tensors_lengths, target_lengths in bg_iterator(
            data_loader, maxsize=2
        ):

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # keep batch first for data parallel
            outputs = model(inputs).transpose(-1, -2).transpose(0, 1)

            # CTC
            # outputs: input length, batch size, number of classes (including blank)
            # targets: batch size, max target length
            # input_lengths: batch size
            # target_lengths: batch size

            sums["loss"] += criterion(
                outputs, targets, tensors_lengths, target_lengths
            ).item()

            output = outputs.transpose(0, 1).to("cpu")
            output = decoder(output)

            output = language_model.decode(output.tolist())
            target = language_model.decode(targets.tolist())

            print_length = 20
            for i in range(2):
                output_print = output[i].ljust(print_length)[:print_length]
                target_print = target[i].ljust(print_length)[:print_length]
                print(f"Target: {target_print}   Output: {output_print}", flush=True)

            cers = [levenshtein_distance(a, b) for a, b in zip(target, output)]
            # cers_normalized = [d / len(a) for a, d in zip(target, cers)]
            cers = sum(cers)
            sums["cer"] += cers

            output = [o.split(language_model.char_space) for o in output]
            target = [o.split(language_model.char_space) for o in target]

            wers = [levenshtein_distance(a, b) for a, b in zip(target, output)]
            # wers_normalized = [d / len(a) for a, d in zip(target, wers)]
            wers = sum(wers)
            sums["wer"] += wers

        # Average
        for k in sums.keys():
            sums[k] /= len(data_loader)

        print(f"Validation loss: {sums['loss']:.5f}", flush=True)
        print(f"CER: {sums['cer']}  WER: {sums['wer']}", flush=True)

        return sums["loss"]


def main(args, rank=0):

    if args.distributed:
        setup(rank, args.world_size)

    print("Start time: {}".format(str(datetime.now())), flush=True)
    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(args.seed)

    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Change backend
    torchaudio.set_audio_backend("soundfile")

    if args.distributed:
        n = torch.cuda.device_count() // args.world_size
        devices = list(range(rank * n, (rank + 1) * n))
    else:
        devices = ["cuda" if torch.cuda.is_available() else "cpu"]

    loader_training_params = {
        "num_workers": args.workers,
        "pin_memory": True,
        "shuffle": True,
        "drop_last": True,
    }
    loader_validation_params = loader_training_params.copy()
    loader_validation_params["shuffle"] = False

    # audio

    melkwargs = {
        "n_fft": 512,
        "n_mels": args.n_bins,  # 13, 20, 128
        "hop_length": 80,  # (160, 80)
    }

    sample_rate_original = 16000

    transforms = torch.nn.Sequential(
        # torchaudio.transforms.Resample(sample_rate_original, sample_rate_original//2),
        # torchaudio.transforms.MFCC(sample_rate=sample_rate_original, n_mfcc=args.n_bins, melkwargs=melkwargs),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate_original, **melkwargs
        ),
        # torchaudio.transforms.FrequencyMasking(freq_mask_param=args.n_bins),
        # torchaudio.transforms.TimeMasking(time_mask_param=35)
    )

    # Text preprocessing

    char_blank = "*"
    char_space = " "
    char_apostrophe = "'"
    labels = char_blank + char_space + char_apostrophe + string.ascii_lowercase
    language_model = LanguageModel(labels, char_blank, char_space)

    training, validation, _ = datasets_librispeech(transforms, language_model)

    if args.decoder == "greedy":
        decoder = GreedyDecoder()
    elif args.decoder == "viterbi":
        decoder = ViterbiDecoder(
            training, len(language_model), progress_bar=args.progress_bar
        )

    model = Wav2Letter(
        num_classes=language_model.length, input_type="mfcc", num_features=args.n_bins
    )

    if args.jit:
        model = torch.jit.script(model)

    if not args.distributed:
        model = torch.nn.DataParallel(model)
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=devices)
        # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    model = model.to(devices[0], non_blocking=True)

    n = count_parameters(model)
    print(f"Number of parameters: {n}", flush=True)

    # Optimizer

    optimizer_params = {
        "lr": args.learning_rate,
        "eps": args.eps,
        "rho": args.rho,
        "weight_decay": args.weight_decay,
    }

    optimizer = Adadelta(model.parameters(), **optimizer_params)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    # scheduler = ReduceLROnPlateau(optimizer, patience=2, threshold=1e-3)

    criterion = torch.nn.CTCLoss(
        blank=language_model.mapping[char_blank], zero_infinity=False
    )
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.NLLLoss()

    torch.autograd.set_detect_anomaly(False)

    collate_fn = collate_factory(model_length_function)

    loader_training = DataLoader(
        training,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_training_params,
    )
    loader_validation = DataLoader(
        validation,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_validation_params,
    )

    best_loss = 1.0

    load_checkpoint = args.checkpoint and os.path.isfile(args.checkpoint)

    if args.distributed:
        torch.distributed.barrier()

    if load_checkpoint:
        print("Checkpoint: loading '{}'".format(args.checkpoint), flush=True)
        checkpoint = torch.load(args.checkpoint)

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        print(
            "Checkpoint: loaded '{}' at epoch {}".format(
                args.checkpoint, checkpoint["epoch"]
            ),
            flush=True,
        )
    else:
        print("Checkpoint: not found", flush=True)

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
            rank,
        )

    if args.distributed:
        torch.distributed.barrier()

    with tqdm(total=args.epochs, unit_scale=1, disable=not args.progress_bar) as pbar:

        for epoch in range(args.start_epoch, args.epochs):

            train_one_epoch(
                model,
                criterion,
                optimizer,
                scheduler,
                loader_training,
                devices[0],
                pbar=pbar,
            )

            if not epoch % args.print_freq or epoch == args.epochs - 1:

                sum_loss = evaluate(
                    model,
                    criterion,
                    loader_validation,
                    decoder,
                    language_model,
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
                        "scheduler": scheduler.state_dict(),
                    },
                    is_best,
                    args.checkpoint,
                    rank,
                )

    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    args = parse_args()
    if args.distributed:
        torch.multiprocessing.spawn(lambda x: main(args, x), nprocs=args.world_size, join=True)
    else:
        main(args)
