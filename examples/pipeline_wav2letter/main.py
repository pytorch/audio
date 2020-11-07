import argparse
import logging
import os
import string
from datetime import datetime
from time import time

import torch
import torchaudio
from torch.optim import SGD, Adadelta, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchaudio.datasets.utils import bg_iterator
from torchaudio.models.wav2letter import Wav2Letter

from ctc_decoders import GreedyDecoder
from datasets import collate_factory, split_process_librispeech
from languagemodels import LanguageModel
from metrics import levenshtein_distance
from transforms import Normalize, UnsqueezeFirst
from utils import MetricLogger, count_parameters, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        metavar="T",
        default="mfcc",
        choices=["waveform", "mfcc"],
        help="input type for model",
    )
    parser.add_argument(
        "--freq-mask",
        default=0,
        type=int,
        metavar="N",
        help="maximal width of frequency mask",
    )
    parser.add_argument(
        "--win-length",
        default=400,
        type=int,
        metavar="N",
        help="width of spectrogram window",
    )
    parser.add_argument(
        "--hop-length",
        default=160,
        type=int,
        metavar="N",
        help="width of spectrogram window",
    )
    parser.add_argument(
        "--time-mask",
        default=0,
        type=int,
        metavar="N",
        help="maximal width of time mask",
    )
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
        "--reduce-lr-valid",
        action="store_true",
        help="reduce learning rate based on validation loss",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="normalize model input"
    )
    parser.add_argument(
        "--progress-bar", action="store_true", help="use progress bar while training"
    )
    parser.add_argument(
        "--decoder",
        metavar="D",
        default="greedy",
        choices=["greedy"],
        help="decoder to use",
    )
    parser.add_argument(
        "--batch-size", default=128, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--n-bins",
        default=13,
        type=int,
        metavar="N",
        help="number of bins in transforms",
    )
    parser.add_argument(
        "--optimizer",
        metavar="OPT",
        default="adadelta",
        choices=["sgd", "adadelta", "adam", "adamw"],
        help="optimizer to use",
    )
    parser.add_argument(
        "--scheduler",
        metavar="S",
        default="reduceonplateau",
        choices=["exponential", "reduceonplateau"],
        help="optimizer to use",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.6,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--gamma",
        default=0.99,
        type=float,
        metavar="GAMMA",
        help="learning rate exponential decay constant",
    )
    parser.add_argument(
        "--momentum", default=0.8, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay", default=1e-5, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument("--eps", metavar="EPS", type=float, default=1e-8)
    parser.add_argument("--rho", metavar="RHO", type=float, default=0.95)
    parser.add_argument("--clip-grad", metavar="NORM", type=float, default=0.0)
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="specify dataset root folder",
    )
    parser.add_argument(
        "--dataset-folder-in-archive",
        type=str,
        help="specify dataset folder in archive",
    )
    parser.add_argument(
        "--dataset-train",
        default=["train-clean-100"],
        nargs="+",
        type=str,
        help="select which part of librispeech to train with",
    )
    parser.add_argument(
        "--dataset-valid",
        default=["dev-clean"],
        nargs="+",
        type=str,
        help="select which part of librispeech to validate with",
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
    logging.info(args)
    return args


def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def model_length_function(tensor):
    if tensor.shape[1] == 1:
        # waveform mode
        return int(tensor.shape[0]) // 160 // 2 + 1
    return int(tensor.shape[0]) // 2 + 1


def compute_error_rates(outputs, targets, decoder, language_model, metric):
    output = outputs.transpose(0, 1).to("cpu")
    output = decoder(output)

    # Compute CER

    output = language_model.decode(output.tolist())
    target = language_model.decode(targets.tolist())

    print_length = 20
    for i in range(2):
        # Print a few examples
        output_print = output[i].ljust(print_length)[:print_length]
        target_print = target[i].ljust(print_length)[:print_length]
        logging.info("Target: %s    Output: %s", target_print, output_print)

    cers = [levenshtein_distance(t, o) for t, o in zip(target, output)]
    cers = sum(cers)
    n = sum(len(t) for t in target)
    metric["cer over target length"] = cers / n
    metric["cumulative cer"] += cers
    metric["total chars"] += n
    metric["cumulative cer over target length"] = metric["cer"] / metric["total chars"]

    # Compute WER

    output = [o.split(language_model.char_space) for o in output]
    target = [t.split(language_model.char_space) for t in target]

    wers = [levenshtein_distance(t, o) for t, o in zip(target, output)]
    wers = sum(wers)
    n = sum(len(t) for t in target)
    metric["wer over target length"] = wers / n
    metric["cumulative wer"] += wers
    metric["total words"] += n
    metric["cumulative wer over target length"] = metric["wer"] / metric["total words"]


def train_one_epoch(
    model,
    criterion,
    optimizer,
    scheduler,
    data_loader,
    decoder,
    language_model,
    device,
    epoch,
    clip_grad,
    disable_logger=False,
    reduce_lr_on_plateau=False,
):

    model.train()

    metric = MetricLogger("train", disable=disable_logger)
    metric["epoch"] = epoch

    for inputs, targets, tensors_lengths, target_lengths in bg_iterator(
        data_loader, maxsize=2
    ):

        start = time()
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

        optimizer.zero_grad()
        loss.backward()

        if clip_grad > 0:
            metric["gradient"] = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad
            )

        optimizer.step()

        compute_error_rates(outputs, targets, decoder, language_model, metric)

        try:
            metric["lr"] = scheduler.get_last_lr()[0]
        except AttributeError:
            metric["lr"] = optimizer.param_groups[0]["lr"]

        metric["batch size"] = len(inputs)
        metric["n_channel"] = inputs.shape[1]
        metric["n_time"] = inputs.shape[-1]
        metric["dataset length"] += metric["batch size"]
        metric["iteration"] += 1
        metric["loss"] = loss.item()
        metric["cumulative loss"] += metric["loss"]
        metric["average loss"] = metric["cumulative loss"] / metric["iteration"]
        metric["iteration time"] = time() - start
        metric["epoch time"] += metric["iteration time"]
        metric()

    if reduce_lr_on_plateau and isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(metric["average loss"])
    elif not isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step()


def evaluate(
    model,
    criterion,
    data_loader,
    decoder,
    language_model,
    device,
    epoch,
    disable_logger=False,
):

    with torch.no_grad():

        model.eval()
        start = time()
        metric = MetricLogger("validation", disable=disable_logger)
        metric["epoch"] = epoch

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

            metric["cumulative loss"] += criterion(
                outputs, targets, tensors_lengths, target_lengths
            ).item()

            metric["dataset length"] += len(inputs)
            metric["iteration"] += 1

            compute_error_rates(outputs, targets, decoder, language_model, metric)

        metric["average loss"] = metric["cumulative loss"] / metric["iteration"]
        metric["validation time"] = time() - start
        metric()

        return metric["average loss"]


def main(rank, args):

    # Distributed setup

    if args.distributed:
        setup_distributed(rank, args.world_size)

    not_main_rank = args.distributed and rank != 0

    logging.info("Start time: %s", datetime.now())

    # Explicitly set seed to make sure models created in separate processes
    # start from same random weights and biases
    torch.manual_seed(args.seed)

    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Change backend for flac files
    torchaudio.set_audio_backend("soundfile")

    # Transforms

    melkwargs = {
        "n_fft": args.win_length,
        "n_mels": args.n_bins,
        "hop_length": args.hop_length,
    }

    sample_rate_original = 16000

    if args.type == "mfcc":
        transforms = torch.nn.Sequential(
            torchaudio.transforms.MFCC(
                sample_rate=sample_rate_original,
                n_mfcc=args.n_bins,
                melkwargs=melkwargs,
            ),
        )
        num_features = args.n_bins
    elif args.type == "waveform":
        transforms = torch.nn.Sequential(UnsqueezeFirst())
        num_features = 1
    else:
        raise ValueError("Model type not supported")

    if args.normalize:
        transforms = torch.nn.Sequential(transforms, Normalize())

    augmentations = torch.nn.Sequential()
    if args.freq_mask:
        augmentations = torch.nn.Sequential(
            augmentations,
            torchaudio.transforms.FrequencyMasking(freq_mask_param=args.freq_mask),
        )
    if args.time_mask:
        augmentations = torch.nn.Sequential(
            augmentations,
            torchaudio.transforms.TimeMasking(time_mask_param=args.time_mask),
        )

    # Text preprocessing

    char_blank = "*"
    char_space = " "
    char_apostrophe = "'"
    labels = char_blank + char_space + char_apostrophe + string.ascii_lowercase
    language_model = LanguageModel(labels, char_blank, char_space)

    # Dataset

    training, validation = split_process_librispeech(
        [args.dataset_train, args.dataset_valid],
        [transforms, transforms],
        language_model,
        root=args.dataset_root,
        folder_in_archive=args.dataset_folder_in_archive,
    )

    # Decoder

    if args.decoder == "greedy":
        decoder = GreedyDecoder()
    else:
        raise ValueError("Selected decoder not supported")

    # Model

    model = Wav2Letter(
        num_classes=language_model.length,
        input_type=args.type,
        num_features=num_features,
    )

    if args.jit:
        model = torch.jit.script(model)

    if args.distributed:
        n = torch.cuda.device_count() // args.world_size
        devices = list(range(rank * n, (rank + 1) * n))
        model = model.to(devices[0])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=devices)
    else:
        devices = ["cuda" if torch.cuda.is_available() else "cpu"]
        model = model.to(devices[0], non_blocking=True)
        model = torch.nn.DataParallel(model)

    n = count_parameters(model)
    logging.info("Number of parameters: %s", n)

    # Optimizer

    if args.optimizer == "adadelta":
        optimizer = Adadelta(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.eps,
            rho=args.rho,
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = Adam(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError("Selected optimizer not supported")

    if args.scheduler == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "reduceonplateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=10, threshold=1e-3)
    else:
        raise ValueError("Selected scheduler not supported")

    criterion = torch.nn.CTCLoss(
        blank=language_model.mapping[char_blank], zero_infinity=False
    )

    # Data Loader

    collate_fn_train = collate_factory(model_length_function, augmentations)
    collate_fn_valid = collate_factory(model_length_function)

    loader_training_params = {
        "num_workers": args.workers,
        "pin_memory": True,
        "shuffle": True,
        "drop_last": True,
    }
    loader_validation_params = loader_training_params.copy()
    loader_validation_params["shuffle"] = False

    loader_training = DataLoader(
        training,
        batch_size=args.batch_size,
        collate_fn=collate_fn_train,
        **loader_training_params,
    )
    loader_validation = DataLoader(
        validation,
        batch_size=args.batch_size,
        collate_fn=collate_fn_valid,
        **loader_validation_params,
    )

    # Setup checkpoint

    best_loss = 1.0

    load_checkpoint = args.checkpoint and os.path.isfile(args.checkpoint)

    if args.distributed:
        torch.distributed.barrier()

    if load_checkpoint:
        logging.info("Checkpoint: loading %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        logging.info(
            "Checkpoint: loaded '%s' at epoch %s", args.checkpoint, checkpoint["epoch"]
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
            not_main_rank,
        )

    if args.distributed:
        torch.distributed.barrier()

    torch.autograd.set_detect_anomaly(False)

    for epoch in range(args.start_epoch, args.epochs):

        logging.info("Epoch: %s", epoch)

        train_one_epoch(
            model,
            criterion,
            optimizer,
            scheduler,
            loader_training,
            decoder,
            language_model,
            devices[0],
            epoch,
            args.clip_grad,
            not_main_rank,
            not args.reduce_lr_valid,
        )

        loss = evaluate(
            model,
            criterion,
            loader_validation,
            decoder,
            language_model,
            devices[0],
            epoch,
            not_main_rank,
        )

        if args.reduce_lr_valid and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
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
            not_main_rank,
        )

    logging.info("End time: %s", datetime.now())

    if args.distributed:
        torch.distributed.destroy_process_group()


def spawn_main(main, args):
    if args.distributed:
        torch.multiprocessing.spawn(
            main, args=(args,), nprocs=args.world_size, join=True
        )
    else:
        main(0, args)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    spawn_main(main, args)
