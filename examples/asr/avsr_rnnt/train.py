import os
import pathlib
import logging
import sentencepiece as spm
from argparse import ArgumentParser
from average_checkpoints import ensemble
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from transforms import get_data_module


def get_trainer(args):
    seed_everything(1)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.experiment_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=True,
        filename=f'{{epoch}}',
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint,
        lr_monitor,
    ]
    return Trainer(
        sync_batchnorm=True,
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


def get_lightning_module(args):
    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))
    if args.md == "av":
        from lightning_av import AVConformerRNNTModule
        model = AVConformerRNNTModule(args, sp_model)
    else:
        from lightning import ConformerRNNTModule
        model = ConformerRNNTModule(args, sp_model)
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--md",
        type=str,
        help="Modality",
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Perform online or offline recognition.",
        required=True,
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory to LRS3 audio-visual datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=str,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to Pretraned model.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--num-nodes",
        default=8,
        type=int,
        help="Number of nodes to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 8)",
    )
    parser.add_argument(
        "--epochs",
        default=55,
        type=int,
        help="Number of epochs to train for. (Default: 55)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        type=str,
        help="Path to the checkpoint to resume from"
    )
    parser.add_argument(
        "--debug",
        action="store_true", 
        help="whether to use debug level for logging"
    )
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    model = get_lightning_module(args)
    data_module = get_data_module(args, str(args.sp_model_path))
    trainer = get_trainer(args)
    trainer.fit(model, data_module)

    ensemble(args)

if __name__ == "__main__":
    cli_main()
