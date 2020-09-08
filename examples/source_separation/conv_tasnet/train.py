#!/usr/bin/env python3
"""Train Conv-TasNet"""
import time
import pathlib
import argparse

import torch
import torchaudio
import torchaudio.models

import conv_tasnet
from utils import dist_utils
from utils.dataset import utils as dataset_utils

_LG = dist_utils.getLogger(__name__)


def _parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__,)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug behavior. Each epoch will end with just one batch.")
    group = parser.add_argument_group("Model Options")
    group.add_argument(
        "--num-speakers", required=True, type=int, help="The number of speakers."
    )
    group = parser.add_argument_group("Dataset Options")
    group.add_argument(
        "--sample-rate",
        required=True,
        type=int,
        help="Sample rate of audio files in the given dataset.",
    )
    group.add_argument(
        "--dataset",
        default="wsj0mix",
        choices=["wsj0mix"],
        help='Dataset type. (default: "wsj0mix")',
    )
    group.add_argument(
        "--dataset-dir",
        required=True,
        type=pathlib.Path,
        help=(
            "Directory where dataset is found. "
            'If the dataset type is "wsj9mix", then this is the directory where '
            '"cv", "tt" and "tr" subdirectories are found.'
        ),
    )
    group = parser.add_argument_group("Save Options")
    group.add_argument(
        "--save-dir",
        required=True,
        type=pathlib.Path,
        help=(
            "Directory where the checkpoints and logs are saved. "
            "Though, only the worker 0 saves checkpoint data, "
            "all the worker processes must have access to the directory."
        ),
    )
    group = parser.add_argument_group("Dataloader Options")
    group.add_argument(
        "--batch-size",
        type=int,
        help=f"Batch size. (default: 16 // world_size)",
    )
    group = parser.add_argument_group("Training Options")
    group.add_argument(
        "--epochs",
        metavar="NUM_EPOCHS",
        default=100,
        type=int,
        help="The number of epochs to train. (default: 100)",
    )
    group.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Initial learning rate. (default: 1e-3)",
    )
    group.add_argument(
        "--grad-clip",
        metavar="CLIP_VALUE",
        default=5.0,
        type=float,
        help="Gradient clip value (l2 norm). (default: 5.0)",
    )
    group.add_argument(
        "--resume",
        metavar="CHECKPOINT_PATH",
        help="Previous checkpoint file from which the training is resumed.",
    )

    args = parser.parse_args(args)

    # Delaing the default value initialization until parse_args is done because
    # if `--help` is given, distributed training is not enabled.
    if args.batch_size is None:
        args.batch_size = 16 // torch.distributed.get_world_size()

    return args


def _get_model(
    num_sources,
    enc_kernel_size=16,
    enc_num_feats=512,
    msk_kernel_size=3,
    msk_num_feats=128,
    msk_num_hidden_feats=512,
    msk_num_layers=8,
    msk_num_stacks=3,
):
    model = torchaudio.models.ConvTasNet(
        num_sources=num_sources,
        enc_kernel_size=enc_kernel_size,
        enc_num_feats=enc_num_feats,
        msk_kernel_size=msk_kernel_size,
        msk_num_feats=msk_num_feats,
        msk_num_hidden_feats=msk_num_hidden_feats,
        msk_num_layers=msk_num_layers,
        msk_num_stacks=msk_num_stacks,
    )
    _LG.info_on_master("Model Configuration:")
    _LG.info_on_master(" - N: %d", enc_num_feats)
    _LG.info_on_master(" - L: %d", enc_kernel_size)
    _LG.info_on_master(" - B: %d", msk_num_feats)
    _LG.info_on_master(" - H: %d", msk_num_hidden_feats)
    _LG.info_on_master(" - Sc: %d", msk_num_feats)
    _LG.info_on_master(" - P: %d", msk_kernel_size)
    _LG.info_on_master(" - X: %d", msk_num_layers)
    _LG.info_on_master(" - R: %d", msk_num_stacks)
    _LG.info_on_master(
        " - Receptive Field: %s [samples]", model.mask_generator.receptive_field,
    )
    return model


def _get_dataloader(dataset_type, dataset_dir, num_speakers, sample_rate, batch_size):
    train_dataset, valid_dataset, eval_dataset = dataset_utils.get_dataset(
        dataset_type, dataset_dir, num_speakers, sample_rate,
    )
    train_collate_fn = dataset_utils.get_collate_fn(
        dataset_type, mode='train', sample_rate=sample_rate, duration=4
    )

    test_collate_fn = dataset_utils.get_collate_fn(dataset_type, mode='test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(train_dataset),
        collate_fn=train_collate_fn,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(valid_dataset),
        collate_fn=test_collate_fn,
        pin_memory=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(eval_dataset),
        collate_fn=test_collate_fn,
        pin_memory=True,
    )
    return train_loader, valid_loader, eval_loader


def _write_header(log_path, args):
    rows = [
        [f"# torch: {torch.__version__}", ],
        [f"# torchaudio: {torchaudio.__version__}", ]
    ]
    rows.append(["# arguments"])
    for key, item in vars(args).items():
        rows.append([f"#   {key}: {item}"])

    dist_utils.write_csv_on_master(log_path, *rows)


def train(args):
    args = _parse_args(args)
    _LG.info("%s", args)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    if "sox_io" in torchaudio.list_audio_backends():
        torchaudio.set_audio_backend("sox_io")

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.sample_rate != checkpoint["sample_rate"]:
            raise ValueError(
                "The provided sample rate ({args.sample_rate}) does not match "
                "the sample rate from the check point ({checkpoint['sample_rate']})."
            )
        if args.num_speakers != checkpoint["num_speakers"]:
            raise ValueError(
                "The provided #of speakers ({args.num_speakers}) does not match "
                "the #of speakers from the check point ({checkpoint['num_speakers']}.)"
            )
        start_epoch = checkpoint["epoch"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LG.info("Using: %s", device)

    model = _get_model(num_sources=args.num_speakers)
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device] if torch.cuda.is_available() else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume:
        _LG.info("Loading parameters from the checkpoint...")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        dist_utils.synchronize_params(
            str(args.save_dir / f"tmp.pt"), device, model, optimizer
        )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    train_loader, valid_loader, eval_loader = _get_dataloader(
        args.dataset,
        args.dataset_dir,
        args.num_speakers,
        args.sample_rate,
        args.batch_size,
    )

    num_train_samples = len(train_loader.dataset)
    num_valid_samples = len(valid_loader.dataset)
    num_eval_samples = len(eval_loader.dataset)

    _LG.info_on_master("Datasets:")
    _LG.info_on_master(" - Train: %s", num_train_samples)
    _LG.info_on_master(" - Valid: %s", num_valid_samples)
    _LG.info_on_master(" -  Eval: %s", num_eval_samples)

    trainer = conv_tasnet.trainer.Trainer(
        model,
        optimizer,
        train_loader,
        valid_loader,
        eval_loader,
        args.grad_clip,
        device,
        debug=args.debug,
    )

    log_path = args.save_dir / f"log.csv"
    _write_header(log_path, args)
    dist_utils.write_csv_on_master(
        log_path,
        [
            "epoch",
            "learning_rate",
            "valid_si_snri",
            "valid_sdri",
            "eval_si_snri",
            "eval_sdri",
        ],
    )

    _LG.info_on_master("Running %s epochs", args.epochs)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        _LG.info_on_master("=" * 70)
        _LG.info_on_master("Epoch: %s", epoch)
        _LG.info_on_master("Learning rate: %s", optimizer.param_groups[0]["lr"])
        _LG.info_on_master("=" * 70)

        t0 = time.monotonic()
        trainer.train_one_epoch()
        train_sps = num_train_samples / (time.monotonic() - t0)

        _LG.info_on_master("-" * 70)

        t0 = time.monotonic()
        valid_metric = trainer.validate()
        valid_sps = num_valid_samples / (time.monotonic() - t0)
        _LG.info_on_master("Valid: %s", valid_metric)

        _LG.info_on_master("-" * 70)

        t0 = time.monotonic()
        eval_metric = trainer.evaluate()
        eval_sps = num_eval_samples / (time.monotonic() - t0)
        _LG.info_on_master(" Eval: %s", eval_metric)

        _LG.info_on_master("-" * 70)

        _LG.info_on_master("Train: Speed: %6.2f [samples/sec]", train_sps)
        _LG.info_on_master("Valid: Speed: %6.2f [samples/sec]", valid_sps)
        _LG.info_on_master(" Eval: Speed: %6.2f [samples/sec]", eval_sps)

        _LG.info_on_master("-" * 70)

        dist_utils.write_csv_on_master(
            log_path,
            [
                epoch,
                optimizer.param_groups[0]["lr"],
                valid_metric.si_snri,
                valid_metric.sdri,
                eval_metric.si_snri,
                eval_metric.sdri,
            ],
        )

        lr_scheduler.step(valid_metric.si_snri)

        save_path = args.save_dir / f"epoch_{epoch}.pt"
        dist_utils.save_on_master(
            save_path,
            {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_speakers": args.num_speakers,
                "sample_rate": args.sample_rate,
                "epoch": epoch,
            },
        )
