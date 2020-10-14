import logging
import os
import string

import torch
import torchaudio
from torch.optim import SGD, Adadelta, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchaudio.datasets.utils import bg_iterator
from torchaudio.transforms import MFCC
from torchaudio.models.wav2letter import Wav2Letter

from ctc_decoders import GreedyDecoder
from datasets import collate_factory, split_process_librispeech
from languagemodels import LanguageModel
from metrics import levenshtein_distance
from transforms import Normalize, ToMono, UnsqueezeFirst
from utils import Logger, count_parameters, save_checkpoint


def setup_distributed(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    # See documentation for choice of backend
    # https://pytorch.org/docs/stable/distributed.html
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # initialize the process group
    torch.distributed.init_process_group(
        backend, rank=rank, world_size=world_size, init_method="env://"
    )


def model_length_function_constructor(model_input_type):
    if model_input_type == "waveform":
        return lambda tensor: int(tensor.shape[-1]) // 160 // 2 + 1
    elif model_input_type == "mfcc":
        return lambda tensor: int(tensor.shape[-1]) // 2 + 1
    raise NotImplementedError(
        f"Selected model input type {model_input_type} not supported"
    )


def record_losses(outputs, targets, decoder, language_model, loss_value, metric):

    # outputs: input length, batch size, number of classes (including blank)
    metric["batch size"] = outputs.shape[1]
    metric["cumulative batch size"] += metric["batch size"]

    # Record loss

    metric["cumulative loss"] += loss_value
    metric["epoch loss"] = metric["cumulative loss"] / metric["cumulative batch size"]
    metric["batch loss"] = loss_value / metric["batch size"]

    # Decode output

    output = outputs.transpose(0, 1).to("cpu")
    output = decoder(output)

    # Compute CER

    output = language_model.decode(output.tolist())
    target = language_model.decode(targets.tolist())

    cers = [levenshtein_distance(t, o) for t, o in zip(target, output)]
    cers = sum(cers)
    n = sum(len(t) for t in target)

    metric["total chars"] += n
    metric["cumulative char errors"] += cers
    metric["batch cer"] = cers / n
    metric["epoch cer"] = metric["cumulative char errors"] / metric["total chars"]

    # Print a few output/target pairs

    print_length = 20
    for i in range(2):
        # Print a few examples
        output_print = output[i].ljust(print_length)[:print_length]
        target_print = target[i].ljust(print_length)[:print_length]
        logging.info("Target: %s    | Output: %s", target_print, output_print)

    # Compute WER

    output = [o.split(language_model.char_space) for o in output]
    target = [t.split(language_model.char_space) for t in target]

    wers = [levenshtein_distance(t, o) for t, o in zip(target, output)]
    wers = sum(wers)
    n = sum(len(t) for t in target)

    metric["total words"] += n
    metric["cumulative word errors"] += wers
    metric["batch wer"] = wers / n
    metric["epoch wer"] = metric["cumulative word errors"] / metric["total words"]

    return metric["epoch loss"]


def _get_optimizer(args, model):
    if args.optimizer == "adadelta":
        return Adadelta(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.eps,
            rho=args.rho,
        )
    elif args.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        return Adam(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adamw":
        return AdamW(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    raise NotImplementedError(f"Selected optimizer {args.optimizer} not supported")


def _get_scheduler(args, optimizer):
    if args.scheduler == "exponential":
        return ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "reduceonplateau":
        return ReduceLROnPlateau(optimizer, patience=10, threshold=1e-3)

    raise NotImplementedError(f"Selected scheduler {args.scheduler} not supported")


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

    metric = Logger("train", disable=disable_logger)
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

        loss = criterion(outputs, targets, tensors_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()

        if clip_grad is not None:
            metric["gradient"] = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad
            )

        optimizer.step()

        # FIXME reduced summed loss value in distributed case?
        avg_loss = record_losses(
            outputs, targets, decoder, language_model, loss.item(), metric
        )

        metric["lr"] = optimizer.param_groups[0]["lr"]
        metric["channel size"] = inputs.shape[1]
        metric["time size"] = inputs.shape[-1]
        metric.flush()

    if reduce_lr_on_plateau and isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_loss)
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
        metric = Logger("validation", disable=disable_logger)
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

            loss = criterion(outputs, targets, tensors_lengths, target_lengths)

            avg_loss = record_losses(
                outputs, targets, decoder, language_model, loss.item(), metric
            )

        metric.flush()

        return avg_loss


def main(rank, args):

    # Distributed setup

    if args.distributed:
        setup_distributed(rank, args.world_size, args.distributed_master_addr, args.distributed_master_port)

    main_rank = rank == 0

    logging.info("Start")

    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Change backend for flac files
    torchaudio.set_audio_backend("soundfile")

    # Transforms

    melkwargs = {
        "n_fft": args.win_length,
        "n_mels": args.bins,
        "hop_length": args.hop_length,
    }

    sample_rate_original = 16000

    transforms = torch.nn.Sequential(ToMono())

    if args.model_input_type == "mfcc":
        transforms = torch.nn.Sequential(
            transforms,
            MFCC(
                sample_rate=sample_rate_original, n_mfcc=args.bins, melkwargs=melkwargs,
            ),
        )
    elif args.model_input_type == "waveform":
        transforms = torch.nn.Sequential(transforms, UnsqueezeFirst())
        assert args.bins == 1, "waveform model input type only supports bins == 1"
    else:
        raise NotImplementedError(
            f"Selected model input type {args.model_input_type} not supported"
        )

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
        num_classes=len(language_model),
        input_type=args.model_input_type,
        num_features=args.bins,
    )

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
    optimizer = _get_optimizer(args, model)
    scheduler = _get_scheduler(args, optimizer)

    # Loss
    encoded_char_blank = language_model.encode(char_blank)[0]
    criterion = torch.nn.CTCLoss(
        blank=encoded_char_blank, zero_infinity=False, reduction=args.reduction
    )

    # Data Loader

    model_length_function = model_length_function_constructor(args.model_input_type)
    collate_fn_train = collate_factory(model_length_function, augmentations)
    collate_fn_valid = collate_factory(model_length_function)

    loader_training = DataLoader(
        training,
        batch_size=args.batch_size,
        collate_fn=collate_fn_train,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    loader_validation = DataLoader(
        validation,
        batch_size=args.batch_size,
        collate_fn=collate_fn_valid,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # Setup checkpoint

    best_loss = 1.0

    checkpoint_exists = os.path.isfile(args.checkpoint)

    if args.distributed:
        torch.distributed.barrier()

    if args.checkpoint and checkpoint_exists and args.resume:
        logging.info("Checkpoint loading %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        logging.info(
            "Checkpoint loaded '%s' at epoch %s", args.checkpoint, checkpoint["epoch"]
        )
    elif args.checkpoint and checkpoint_exists:
        raise RuntimeError(
            "Checkpoint already exists. Add --resume to resume, or manually delete existing file."
        )
    elif args.checkpoint and args.resume:
        raise RuntimeError("Checkpoint not found")
    elif args.checkpoint and main_rank and args.checkpoint:
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
    elif not args.checkpoint and args.resume:
        raise RuntimeError("Checkpoint not provided. Use --checkpoint to specify.")

    if args.distributed:
        torch.distributed.barrier()

    torch.autograd.set_detect_anomaly(False)

    for epoch in range(args.start_epoch, args.max_epoch):

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
            not main_rank,
            not args.reduce_lr_valid,
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
                not main_rank,
            )

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if main_rank and args.checkpoint:
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

        if args.reduce_lr_valid and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if main_rank and args.checkpoint:
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

    logging.info("End")

    if args.distributed:
        torch.distributed.destroy_process_group()


def spawn_main(args):
    if args.distributed:
        torch.multiprocessing.spawn(
            main, args=(args,), nprocs=args.world_size, join=True
        )
    else:
        main(0, args)
