#!/usr/bin/env python3
"""Fine-tune the HuBERTPretrainModel on 10 hours of LibriLightLimited dataset.
Example:
python finetune.py --dataset-path ./root/datasets/ --exp-dir ./exp_finetune \
  --checkpoint /exp_iter2/checkpoints_librispeech_hubert_pretrain_base/epoch=361-step=399999.ckpt \
  --gpus 1 --debug --warmup-updates 2000 --hold-updates 8000 --decay-updates 10000 \
  --max-updates 20000 --learning-rate 5e-5
"""

import logging
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter
from typing import Tuple

from lightning import HuBERTFineTuneModule

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything


logger = logging.getLogger(__name__)


class _Formatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    # To use ArgumentDefaultsHelpFormatter as the formatter_class and
    # RawDescriptionHelpFormatter to add custom formatting to description or epilog.
    # Check: https://stackoverflow.com/a/18462760
    pass


def run_train(args):
    seed_everything(1337)
    checkpoint_dir = args.exp_dir / f"checkpoints_{args.model_name}"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    callbacks = [
        checkpoint,
        train_checkpoint,
    ]
    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_steps=args.max_updates,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator="gpu",
        strategy="ddp",
        replace_sampler_ddp=False,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        val_check_interval=500,
        check_val_every_n_epoch=None,
    )

    model = HuBERTFineTuneModule(
        model_name=args.model_name,
        encoder_projection_dropout=args.encoder_projection_dropout,
        encoder_attention_dropout=args.encoder_attention_dropout,
        encoder_ff_interm_dropout=args.encoder_ff_interm_dropout,
        encoder_dropout=args.encoder_dropout,
        encoder_layer_drop=args.encoder_layer_drop,
        mask_prob=args.mask_prob,
        mask_channel_prob=args.mask_channel_prob,
        mask_channel_length=args.mask_channel_length,
        num_classes=args.num_classes,
        aux_num_out=args.aux_num_out,
        checkpoint=args.checkpoint,
        dataset_path=args.dataset_path,
        seconds_per_batch=args.seconds_per_batch,
        subset=args.subset,
        learning_rate=args.learning_rate,
        betas=args.betas,
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
        freeze_encoder_updates=args.freeze_encoder_updates,
        warmup_updates=args.warmup_updates,
        hold_updates=args.hold_updates,
        decay_updates=args.decay_updates,
    )
    trainer.fit(model, ckpt_path=args.resume_checkpoint)


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=_Formatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        required=True,
        help="Path to the LibriSpeech and LibriLightLimited datasets.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pre-trained HuBERTPretrainModel checpoint as the initialization.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        type=str,
        help="The path to the checkpoint to resume the fine-tuning if training fails in the middle.",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp_finetune"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp_finetune')",
    )
    parser.add_argument(
        "--model-name",
        default="hubert_pretrain_base",
        choices=["hubert_pretrain_base", "hubert_pretrain_large", "hubert_pretrain_xlarge"],
        type=str,
        help="The HuBERTPretrainModel to fine-tune. (Default: 'hubert_pretrain_base')",
    )
    parser.add_argument(
        "--encoder-projection-dropout",
        default=0.0,
        type=float,
        help="The dropout probability applied after the input feature "
        "is projected to ``encoder_embed_dim``. (Default: 0.0)",
    )
    parser.add_argument(
        "--encoder-attention-dropout",
        default=0.0,
        type=float,
        help="The dropout probability applied after softmax in self-attention layer." "(Default: 0.0)",
    )
    parser.add_argument(
        "--encoder-ff-interm-dropout",
        default=0.1,
        type=float,
        help="The dropout probability applied in feedforward layer." "(Default: 0.1)",
    )
    parser.add_argument(
        "--encoder-dropout",
        default=0.0,
        type=float,
        help="The dropout probability applied at the end of feed forward layer." "(Default: 0.0)",
    )
    parser.add_argument(
        "--encoder-layer-drop",
        default=0.05,
        type=float,
        help="Probability to drop each encoder layer during training. (Default: 0.1)",
    )
    parser.add_argument(
        "--mask-prob",
        default=0.65,
        type=float,
        help="Probability to mask the frames of the convolutional layer feature." "(Default: 0.75)",
    )
    parser.add_argument(
        "--mask-channel-prob",
        default=0.5,
        type=float,
        help="Probability to mask the feature dimension of the convolutional layer feature." "(Default: 0.5)",
    )
    parser.add_argument(
        "--mask-channel-length",
        default=64,
        type=int,
        help="Minimum space between spans (if no overlap is enabled) for channel masking." "(Default: 64)",
    )
    parser.add_argument(
        "--num-classes",
        choices=[100, 500],
        type=int,
        default=500,
        help="The ``num_class`` in the pre-trained checkpoint. (Default: 500",
    )
    parser.add_argument(
        "--aux-num-out",
        default=29,
        type=int,
        help="The dimension of linear layer for CTC training. (Default: 29)",
    )
    parser.add_argument(
        "--learning-rate", default=5e-5, type=float, help="The learning rate of Adam optimizer. (Default: 5e-5)"
    )
    parser.add_argument(
        "--betas",
        default=(0.9, 0.98),
        type=Tuple,
        help="The coefficients for computing running averages of gradient and its square (Default: (0.9, 0.98))",
    )
    parser.add_argument(
        "--adam-eps",
        default=1e-8,
        type=float,
        help="Epsilon value in Adam optimizer. (Default: 1e-8)",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0,
        type=float,
        help="Weight decay (L2 penalty) (Default: 0.0)",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use for fine-tuning. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs per node to use for fine-tuning. (Default: 1)",
    )
    parser.add_argument(
        "--freeze-encoder-updates",
        default=10000,
        type=int,
        help="Number of steps to freeze the transformer encoder in HuBERT. (Default: 10000)",
    )
    parser.add_argument(
        "--warmup-updates",
        default=2000,
        type=int,
        help="Number of steps for warm up the learning rate. (Default: 8000)",
    )
    parser.add_argument(
        "--hold-updates",
        default=8000,
        type=int,
        help="Number of steps for keeping the peak learning rate. (Default: 0)",
    )
    parser.add_argument(
        "--decay-updates",
        default=10000,
        type=int,
        help="Number of steps for decreasing the learning rate. (Default: 72000)",
    )
    parser.add_argument(
        "--max-updates",
        default=20000,
        type=int,
        help="Total number of training steps. (Default: 250000)",
    )
    parser.add_argument(
        "--seconds-per-batch",
        default=200,
        type=float,
        help="Number of seconds of audio in a mini-batch. (Default: 200)",
    )
    parser.add_argument(
        "--subset",
        default="10h",
        type=str,
        choices=["10min", "1h", "10h"],
        help="The subset of LibriLightLimited dataset for fine-tuning. (Default: '10h')",
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
