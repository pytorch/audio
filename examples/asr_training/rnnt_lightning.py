from argparse import ArgumentParser
from collections import namedtuple
import json
import logging
import math
import os
import pathlib
from typing import List, Tuple

from fairseq.data import Dictionary
import sentencepiece as spm

import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.prototype.rnnt import emformer_rnnt_base
from torchaudio.prototype.rnnt_decoder import Hypothesis, RNNTBeamSearch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


logger = logging.getLogger()

Batch = namedtuple(
    "Batch", ["features", "feature_lengths", "targets", "target_lengths"]
)


_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)

_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=400, n_mels=80, hop_length=160
)


def _batch_by_token_count(idx_target_lengths, token_limit):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > token_limit:
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


class CustomDataset(torch.utils.data.Dataset):
    r"""Sort samples by target length and batch to max token count."""

    def __init__(self, base_dataset, max_token_limit):
        super().__init__()
        self.base_dataset = base_dataset

        fileid_to_target_length = {}
        idx_target_lengths = [
            (idx, self._target_length(fileid, fileid_to_target_length))
            for idx, fileid in enumerate(self.base_dataset._walker)
        ]

        assert len(idx_target_lengths) > 0

        idx_target_lengths = sorted(
            idx_target_lengths, key=lambda x: x[1], reverse=True
        )

        assert max_token_limit >= idx_target_lengths[0][1]

        self.batches = _batch_by_token_count(idx_target_lengths, max_token_limit)

    def _target_length(self, fileid, fileid_to_target_length):
        if fileid not in fileid_to_target_length:
            speaker_id, chapter_id, _ = fileid.split("-")

            file_text = speaker_id + "-" + chapter_id + self.base_dataset._ext_txt
            file_text = os.path.join(
                self.base_dataset._path, speaker_id, chapter_id, file_text
            )

            with open(file_text) as ft:
                for line in ft:
                    fileid_text, transcript = line.strip().split(" ", 1)
                    fileid_to_target_length[fileid_text] = len(transcript)

        return fileid_to_target_length[fileid]

    def __getitem__(self, idx):
        return [self.base_dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class TimeMasking(torchaudio.transforms._AxisMasking):
    def __init__(
        self, time_mask_param: int, min_mask_p: float, iid_masks: bool = False
    ) -> None:
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks)
        self.min_mask_p = min_mask_p

    def forward(self, specgram: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
        if self.iid_masks and specgram.dim() == 4:
            mask_param = min(
                self.mask_param, self.min_mask_p * specgram.shape[self.axis + 1]
            )
            return F.mask_along_axis_iid(
                specgram, mask_param, mask_value, self.axis + 1
            )
        else:
            mask_param = min(
                self.mask_param, self.min_mask_p * specgram.shape[self.axis]
            )
            return F.mask_along_axis(specgram, mask_param, mask_value, self.axis)


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class GlobalStatsNormalization(torch.nn.Module):
    def __init__(self, global_stats_path):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob["mean"])
        self.invstddev = torch.tensor(blob["invstddev"])

    def forward(self, input):
        return (input - self.mean) * self.invstddev


def _piecewise_linear_log(x):
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_updates, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        return [
            (min(1.0, self._step_count / self.warmup_updates)) * base_lr
            for base_lr in self.base_lrs
        ]


class RNNTModule(LightningModule):
    def __init__(
        self,
        *,
        librispeech_path: str,
        sp_model_path: str,
        tgt_dict_path: str,
        global_stats_path: str,
    ):
        super().__init__()

        self.model = emformer_rnnt_base()
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum", clamp=1.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.96, patience=0
        )
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 10000)

        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(lambda x: x.transpose(1, 2)),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.FrequencyMasking(27),
            TimeMasking(100, 0.2),
            TimeMasking(100, 0.2),
            FunctionalModule(lambda x: torch.nn.functional.pad(x, (0, 4))),
            FunctionalModule(lambda x: x.transpose(1, 2)),
        )
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(lambda x: x.transpose(1, 2)),
            FunctionalModule(lambda x: torch.nn.functional.pad(x, (0, 4))),
            FunctionalModule(lambda x: x.transpose(1, 2)),
        )

        self.librispeech_path = librispeech_path

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sp_model_path)

        self.tgt_dict = Dictionary.load(tgt_dict_path)
        self.tgt_dict.add_symbol("<blank>")
        self.blank_idx = len(self.tgt_dict) - 1

    def _extract_labels(self, samples: List):
        targets = [self.sp_model.EncodeAsIds(sample[2].lower()) for sample in samples]
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True,
            padding_value=1.0,
        ).to(dtype=torch.int32)
        return targets, lengths

    def _train_extract_features(self, samples: List):
        mel_features = [
            _spectrogram_transform(sample[0].squeeze()).transpose(1, 0)
            for sample in samples
        ]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.train_data_pipeline(features)
        lengths = torch.tensor(
            [elem.shape[0] for elem in mel_features], dtype=torch.int32
        )
        return features, lengths

    def _valid_extract_features(self, samples: List):
        mel_features = [
            _spectrogram_transform(sample[0].squeeze()).transpose(1, 0)
            for sample in samples
        ]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.valid_data_pipeline(features)
        lengths = torch.tensor(
            [elem.shape[0] for elem in mel_features], dtype=torch.int32
        )
        return features, lengths

    def _train_collate_fn(self, samples: List):
        features, feature_lengths = self._train_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _valid_collate_fn(self, samples: List):
        features, feature_lengths = self._valid_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _test_collate_fn(self, samples: List):
        return self._valid_collate_fn(samples), samples

    def _step(self, batch, batch_idx, step_type):
        if batch is None:
            return None

        prepended_targets = batch.targets.new_empty(
            [batch.targets.size(0), batch.targets.size(1) + 1]
        )
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1
        output, src_lengths, _, _ = self.model(
            batch.features,
            batch.feature_lengths,
            prepended_targets,
            prepended_target_lengths,
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [
                {
                    "scheduler": self.lr_scheduler,
                    "monitor": "Losses/val_loss",
                    "interval": "epoch",
                },
                {"scheduler": self.warmup_lr_scheduler, "interval": "step"},
            ],
        )

    def forward(self, batch: Batch):
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        hypotheses = decoder(
            batch.features.to(self.device), batch.feature_lengths.to(self.device), 20
        )
        return post_process_hypos(hypotheses, self.sp_model, self.tgt_dict)[0][0]

    def training_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def train_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
            [
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(
                        self.librispeech_path, url="train-clean-360"
                    ),
                    1000,
                ),
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(
                        self.librispeech_path, url="train-clean-100"
                    ),
                    1000,
                ),
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(
                        self.librispeech_path, url="train-other-500"
                    ),
                    1000,
                ),
            ]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._train_collate_fn,
            num_workers=10,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
            [
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(
                        self.librispeech_path, url="dev-clean"
                    ),
                    1000,
                ),
                CustomDataset(
                    torchaudio.datasets.LIBRISPEECH(
                        self.librispeech_path, url="dev-other"
                    ),
                    1000,
                ),
            ]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, collate_fn=self._valid_collate_fn, num_workers=10,
        )
        return dataloader

    def test_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(
            self.librispeech_path, url="test-clean"
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, collate_fn=self._test_collate_fn
        )
        return dataloader


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(
        seq1.lower().split(), seq2.lower().split()
    )


def post_process_hypos(
    hypos: List[Hypothesis], sp_model: spm.SentencePieceProcessor, tgt_dict: Dictionary
) -> List[Tuple[str, float, List[int], List[int]]]:
    post_process_remove_list = [
        sp_model.unk_id(),
        sp_model.eos_id(),
        sp_model.pad_id(),
    ]
    hypos_str = [
        tgt_dict.string(
            [
                token_index
                for token_index in h.tokens[1:]
                if token_index not in post_process_remove_list
            ]
        )
        for h in hypos
    ]
    hypos_str = [sp_model.DecodePieces(s.split()) for s in hypos_str]
    hypos_ali = [h.alignment[1:] for h in hypos]
    hypos_ids = [h.tokens[1:] for h in hypos]
    hypos_score = [[math.exp(h.score)] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ali, hypos_ids))

    return nbest_batch


def run_eval(args):
    model = (
        RNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            librispeech_path=args.librispeech_path,
            sp_model_path=args.sp_model_path,
            tgt_dict_path=args.tgt_dict_path,
            global_stats_path=args.global_stats_path,
        )
        .eval()
        .to(device="cuda")
    )
    total_edit_distance = 0
    total_length = 0
    dataloader = model.test_dataloader()
    with torch.no_grad():
        for idx, (batch, sample) in enumerate(dataloader):
            actual = sample[0][2]
            predicted = model(batch)
            total_edit_distance += compute_word_level_distance(actual, predicted)
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.info(
                    f"Processed elem {idx}; WER: {total_edit_distance / total_length}"
                )
    logger.info(f"Final WER: {total_edit_distance / total_length}")


def run_train(args):
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=10,
        save_weights_only=True,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=10,
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
        gradient_clip_val=10.0,
        callbacks=callbacks,
    )

    model = RNNTModule(
        librispeech_path=args.librispeech_path,
        sp_model_path=args.sp_model_path,
        tgt_dict_path=args.tgt_dict_path,
        global_stats_path=args.global_stats_path,
    )
    trainer.fit(model)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--eval", action="store_true", default=False, help="Run in eval mode.",
    )
    parser.add_argument(
        "--exp_dir",
        default=pathlib.Path("./exp"),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--global_stats_path",
        type=str,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--librispeech_path", type=str, help="Path to LibriSpeech datasets.",
    )
    parser.add_argument(
        "--sp_model_path", type=str, help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--tgt_dict_path", type=str, help="Path to fairseq token dictionary.",
    )
    parser.add_argument(
        "--num_nodes",
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
    args = parser.parse_args()

    if args.eval:
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    cli_main()
