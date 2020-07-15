import argparse
import logging
import os
import signal
import string
from collections import defaultdict
from datetime import datetime
from time import time

import torch
import torchaudio
from torch.optim import SGD, Adadelta, Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchaudio.datasets.utils import bg_iterator, diskcache_iterator
from torchaudio.models.wav2letter import Wav2Letter
from torchaudio.transforms import MFCC, Resample

from ctc_decoders import GreedyDecoder, ViterbiDecoder
from datasets import collate_factory, datasets_librispeech
from languagemodels import LanguageModel
from metrics import levenshtein_distance
from utils import MetricLogger, count_parameters, save_checkpoint

# TODO Remove before merge pull request
MAIN_PID = os.getpid()
SIGNAL_RECEIVED = False


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
        "--optimizer",
        metavar="OPT",
        default="sgd",
        choices=["sgd", "adadelta", "adam"],
        help="optimizer to use",
    )
    parser.add_argument(
        "--scheduler",
        metavar="S",
        default="exponential",
        choices=["exponential", "reduceonplateau"],
        help="optimizer to use",
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
    parser.add_argument(
        "--momentum", default=0.0, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay", default=1e-5, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument("--eps", metavar="EPS", type=float, default=1e-8)
    parser.add_argument("--rho", metavar="RHO", type=float, default=0.95)
    parser.add_argument("--clip-grad", metavar="NORM", type=float, default=0.0)
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
    logging.info(args)
    return args


# TODO Remove before merge pull request
def signal_handler(a, b):
    global SIGNAL_RECEIVED
    logging.info(f"Signal received on {datetime.now()}")
    SIGNAL_RECEIVED = True


# TODO Remove before merge pull request
def trigger_job_requeue():
    # Submit a new job to resume from checkpoint.
    if os.environ["SLURM_PROCID"] == "0" and os.getpid() == MAIN_PID:
        logging.info(f"PID: {os.getpid()}. PPID: {os.getppid()}.")
        logging.info("Resubmitting job")
        command = "scontrol requeue " + os.environ["SLURM_JOB_ID"]
        logging.info(command)
        if os.system(command):
            raise RuntimeError("Fail to resubmit")
        logging.info("New job submitted to the queue")
    exit(0)


def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def model_length_function(tensor):
    return int(tensor.shape[0]) // 2 + 1


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
):

    model.train()

    sums = defaultdict(lambda: 0.0)
    start1 = time()

    metric = MetricLogger("train_iteration")
    metric["epoch"] = epoch

    for inputs, targets, tensors_lengths, target_lengths in bg_iterator(
        data_loader, maxsize=2
    ):

        start2 = time()
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
        loss_item = loss.item()
        sums["loss"] += loss_item
        metric["loss"] = loss_item

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad > 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_grad
            )
            sums["gradient"] += gradient
            metric["gradient"] = gradient

        optimizer.step()

        sums["length_dataset"] += len(inputs)

        compute_error_rates(outputs, targets, decoder, language_model, sums)
        record_error_rates(sums, metric)

        metric["iteration"] = sums["iteration"]
        metric["time"] = time() - start2
        metric["epoch"] = epoch
        metric()

        sums["iteration"] += 1

        # TODO Remove before merge pull request
        if SIGNAL_RECEIVED:
            return loss_item

    avg_loss = sums["loss"] / len(data_loader)

    metric = MetricLogger("train_epoch")
    record_error_rates(sums, metric)
    metric["epoch"] = epoch
    metric["loss"] = avg_loss
    metric["gradient"] = sums["gradient"] / len(data_loader)
    try:
        metric["lr"] = scheduler.get_last_lr()[0]
    except AttributeError:
        pass
    metric["time"] = time() - start1
    metric()

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_loss)
    else:
        scheduler.step()


def record_error_rates(sums, metric):

    metric["cer"] = sums["cer"]
    metric["wer"] = sums["wer"]
    metric["cer over dataset length"] = sums["cer"] / sums["length_dataset"]
    metric["wer over dataset length"] = sums["wer"] / sums["length_dataset"]
    metric["cer over target length"] = sums["cer"] / sums["total_chars"]
    metric["wer over target length"] = sums["wer"] / sums["total_words"]
    metric["target length"] = sums["total_chars"]
    metric["target length"] = sums["total_words"]
    metric["dataset length"] = sums["length_dataset"]


def compute_error_rates(outputs, targets, decoder, language_model, sums):
    output = outputs.transpose(0, 1).to("cpu")
    output = decoder(output)

    output = language_model.decode(output.tolist())
    target = language_model.decode(targets.tolist())

    print_length = 20
    for i in range(2):
        output_print = output[i].ljust(print_length)[:print_length]
        target_print = target[i].ljust(print_length)[:print_length]
        logging.info(f"Target: {target_print}   Output: {output_print}")

    cers = [levenshtein_distance(a, b) for a, b in zip(target, output)]
    # cers_normalized = [d / len(a) for a, d in zip(target, cers)]
    cers = sum(cers)
    n = sum(len(t) for t in target)
    sums["cer"] += cers
    # sums["cer_relative"] += cers / n
    sums["total_chars"] += n

    output = [o.split(language_model.char_space) for o in output]
    target = [o.split(language_model.char_space) for o in target]

    wers = [levenshtein_distance(a, b) for a, b in zip(target, output)]
    # wers_normalized = [d / len(a) for a, d in zip(target, wers)]
    wers = sum(wers)
    n = len(target)
    sums["wer"] += wers
    # sums["wer_relative"] += wers / n
    sums["total_words"] += n


def evaluate(
    model, criterion, data_loader, decoder, language_model, device, epoch,
):

    with torch.no_grad():

        model.eval()
        sums = defaultdict(lambda: 0.0)
        start = time()
        metric = MetricLogger("validation")

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

            sums["length_dataset"] += len(inputs)

            compute_error_rates(outputs, targets, decoder, language_model, sums)

            # TODO Remove before merge pull request
            if SIGNAL_RECEIVED:
                return sums["loss"] / len(data_loader)

        avg_loss = sums["loss"] / len(data_loader)

        metric["epoch"] = epoch
        metric["loss"] = avg_loss
        metric["time"] = time() - start
        record_error_rates(sums, metric)
        metric()

        return avg_loss


def main(args, rank=0):

    if args.distributed:
        setup_distributed(rank, args.world_size)

    logging.info("Start time: {}".format(str(datetime.now())))

    # Install signal handler
    signal.signal(signal.SIGUSR1, signal_handler)

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

    if args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=devices)
        # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)

    model = model.to(devices[0], non_blocking=True)

    n = count_parameters(model)
    logging.info(f"Number of parameters: {n}")

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

    if args.scheduler == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "reduceonplateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=10, threshold=1e-3)

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
            rank,
        )

    if args.distributed:
        torch.distributed.barrier()

    for epoch in range(args.start_epoch, args.epochs):

        logging.info(f"Epoch: {epoch}")

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
        )

        if not (epoch + 1) % args.print_freq or epoch == args.epochs - 1:

            loss = evaluate(
                model,
                criterion,
                loader_validation,
                decoder,
                language_model,
                devices[0],
                epoch,
            )

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
                rank,
            )

        # TODO Remove before merge pull request
        if SIGNAL_RECEIVED:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                False,
                args.checkpoint,
                rank,
            )
            trigger_job_requeue()

    logging.info(f"End time: {datetime.now()}")

    if args.distributed:
        torch.distributed.destroy_process_group()


def spawn_main(args, main):
    if args.distributed:
        torch.multiprocessing.spawn(
            lambda x: main(args, x), nprocs=args.world_size, join=True
        )
    else:
        main(args)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    spawn_main(args, main)
