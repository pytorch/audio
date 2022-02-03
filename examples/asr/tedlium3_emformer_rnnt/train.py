import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

from lightning import RNNTModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def run_train(args):
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
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        gradient_clip_val=5.0,
        callbacks=callbacks,
    )

    model = RNNTModule(
        tedlium_path=str(args.tedlium_path),
        sp_model_path=str(args.sp_model_path),
        global_stats_path=str(args.global_stats_path),
        reduction=args.reduction,
    )
    trainer.fit(model)


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--tedlium-path",
        type=pathlib.Path,
        required=True,
        help="Path to TED-LIUM release 3 dataset.",
    )
    parser.add_argument(
        "--reduction",
        default="mean",
        type=str,
        help="Reduction option for RNN Transducer loss function." "(Default: ``mean``)",
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--num-nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
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
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = _parse_args()
    _init_logger(args.debug)
    run_train(args)


if __name__ == "__main__":
    cli_main()
