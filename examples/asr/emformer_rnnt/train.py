#!/usr/bin/env python3
import logging
import pathlib
from argparse import ArgumentParser

from common import MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_MUSTC, MODEL_TYPE_TEDLIUM3
from librispeech.lightning import LibriSpeechRNNTModule
from mustc.lightning import MuSTCRNNTModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tedlium3.lightning import TEDLIUM3RNNTModule


def get_trainer(args):
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=True,
        verbose=True,
    )
    callbacks = [
        checkpoint,
        train_checkpoint,
    ]
    return Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
    )


def get_lightning_module(args):
    if args.model_type == MODEL_TYPE_LIBRISPEECH:
        return LibriSpeechRNNTModule(
            librispeech_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    elif args.model_type == MODEL_TYPE_TEDLIUM3:
        return TEDLIUM3RNNTModule(
            tedlium_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    elif args.model_type == MODEL_TYPE_MUSTC:
        return MuSTCRNNTModule(
            mustc_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    else:
        raise ValueError(f"Encountered unsupported model type {args.model_type}.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-type", type=str, choices=[MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_TEDLIUM3, MODEL_TYPE_MUSTC], required=True
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
        required=True,
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        help="Path to datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--num-nodes",
        default=4,
        type=int,
        help="Number of nodes to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--epochs",
        default=120,
        type=int,
        help="Number of epochs to train for. (Default: 120)",
    )
    parser.add_argument(
        "--gradient-clip-val", default=10.0, type=float, help="Value to clip gradient values to. (Default: 10.0)"
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    model = get_lightning_module(args)
    trainer = get_trainer(args)
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()
