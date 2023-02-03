import os
import pathlib
from argparse import ArgumentParser

from lightning import ConformerRNNTModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from transforms import get_data_module


def run_train(args):
    seed_everything(1)
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint,
        train_checkpoint,
        lr_monitor,
    ]
    if os.path.exists(args.resume) and args.resume != '':
        trainer = Trainer(
            default_root_dir=args.exp_dir,
            max_epochs=args.epochs,
            num_nodes=args.nodes,
            gpus=args.gpus,
            accelerator="gpu",
            strategy=DDPStrategy(find_unused_parameters=False),
            callbacks=callbacks,
            reload_dataloaders_every_n_epochs=1,
            resume_from_checkpoint=args.resume
        )
    else:
        trainer = Trainer(
            default_root_dir=args.exp_dir,
            max_epochs=args.epochs,
            num_nodes=args.nodes,
            gpus=args.gpus,
            accelerator="gpu",
            strategy=DDPStrategy(find_unused_parameters=False),
            callbacks=callbacks,
            reload_dataloaders_every_n_epochs=1,
        )

    model = ConformerRNNTModule(str(args.sp_model_path), args.biasing)
    data_module = get_data_module(str(args.librispeech_path), str(args.global_stats_path), str(args.sp_model_path),
                                  subset=args.subset, biasinglist=args.biasing_list, droprate=args.droprate,
                                  maxsize=args.maxsize)
    trainer.fit(model, data_module, ckpt_path=args.checkpoint_path)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats_100.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=1,
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
        "--subset",
        default='train-clean-100',
        type=str,
        help="Train on subset of librispeech.",
    )
    parser.add_argument(
        "--biasing",
        action="store_true",
        help="Use biasing",
    )
    parser.add_argument(
        "--biasing-list",
        type=pathlib.Path,
        help="Path to the biasing list.",
        required=True,
    )
    parser.add_argument(
        "--maxsize",
        default=1000,
        type=int,
        help="Size of biasing lists"
    )
    parser.add_argument(
        "--droprate",
        default=0.0,
        type=float,
        help="Biasing component regularisation drop rate"
    )
    parser.add_argument(
        "--resume",
        default='',
        type=str,
        help="Path to resume model.",
    )
    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    cli_main()
