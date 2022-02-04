import json
import math
import os
from collections import namedtuple
from typing import List, Tuple

import sentencepiece as spm
import torch
import torchaudio
from pytorch_lightning import LightningModule
from torchaudio.models import Hypothesis, RNNTBeamSearch, emformer_rnnt_base
from torchaudio.transforms import TimeMasking
from utils import GAIN, piecewise_linear_log, spectrogram_transform

Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths"])


def _batch_by_token_count(idx_target_lengths, token_limit):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if target_length == -1:
            continue
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
    r"""Sort samples by target length and batch to max durations."""

    def __init__(self, base_dataset, max_token_limit):
        super().__init__()
        self.base_dataset = base_dataset

        idx_target_lengths = [
            (idx, self._target_length(fileid, line)) for idx, (fileid, line) in enumerate(self.base_dataset._filelist)
        ]

        assert len(idx_target_lengths) > 0

        idx_target_lengths = sorted(idx_target_lengths, key=lambda x: x[1])

        assert max_token_limit >= idx_target_lengths[-1][1]

        self.batches = _batch_by_token_count(idx_target_lengths, max_token_limit)

    def _target_length(self, fileid, line):
        transcript_path = os.path.join(self.base_dataset._path, "stm", fileid)
        with open(transcript_path + ".stm") as f:
            transcript = f.readlines()[line]
            talk_id, _, speaker_id, start_time, end_time, identifier, transcript = transcript.split(" ", 6)
            if transcript.lower() == "ignore_time_segment_in_scoring\n":
                return -1
            else:
                return float(end_time) - float(start_time)

    def __getitem__(self, idx):
        return [self.base_dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


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


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_updates, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        return [(min(1.0, self._step_count / self.warmup_updates)) * base_lr for base_lr in self.base_lrs]


def post_process_hypos(
    hypos: List[Hypothesis], sp_model: spm.SentencePieceProcessor
) -> List[Tuple[str, float, List[int], List[int]]]:
    post_process_remove_list = [
        sp_model.unk_id(),
        sp_model.eos_id(),
        sp_model.pad_id(),
    ]
    filtered_hypo_tokens = [
        [token_index for token_index in h.tokens[1:] if token_index not in post_process_remove_list] for h in hypos
    ]
    hypos_str = [sp_model.decode(s) for s in filtered_hypo_tokens]
    hypos_ali = [h.alignment[1:] for h in hypos]
    hypos_ids = [h.tokens[1:] for h in hypos]
    hypos_score = [[math.exp(h.score)] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ali, hypos_ids))

    return nbest_batch


class RNNTModule(LightningModule):
    def __init__(
        self,
        *,
        tedlium_path: str,
        sp_model_path: str,
        global_stats_path: str,
        reduction: str,
    ):
        super().__init__()

        self.model = emformer_rnnt_base(num_symbols=501)
        self.loss = torchaudio.transforms.RNNTLoss(reduction=reduction, clamp=1.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 10000)

        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: piecewise_linear_log(x * GAIN)),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(lambda x: x.transpose(1, 2)),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.FrequencyMasking(27),
            TimeMasking(100, p=0.2),
            TimeMasking(100, p=0.2),
            FunctionalModule(lambda x: torch.nn.functional.pad(x, (0, 4))),
            FunctionalModule(lambda x: x.transpose(1, 2)),
        )
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: piecewise_linear_log(x * GAIN)),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(lambda x: x.transpose(1, 2)),
            FunctionalModule(lambda x: torch.nn.functional.pad(x, (0, 4))),
            FunctionalModule(lambda x: x.transpose(1, 2)),
        )

        self.tedlium_path = tedlium_path

        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.blank_idx = self.sp_model.get_piece_size()

    def _extract_labels(self, samples: List):
        """Convert text transcript into int labels.

        Note:
            There are ``<unk>`` tokens in the training set that are regarded as normal tokens
            by the SentencePiece model. This will impact RNNT decoding since the decoding result
            of ``<unk>`` will be ``?? unk ??`` and will not be excluded from the final prediction.
            To address it, here we replace ``<unk>`` with ``<garbage>`` and set
            ``user_defined_symbols=["<garbage>"]`` in the SentencePiece model training.
            Then we map the index of ``<garbage>`` to the real ``unknown`` index.
        """
        targets = [
            self.sp_model.encode(sample[2].lower().replace("<unk>", "<garbage>").replace("\n", ""))
            for sample in samples
        ]
        targets = [
            [ele if ele != 4 else self.sp_model.unk_id() for ele in target] for target in targets
        ]  # map id of <unk> token to unk_id
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True,
            padding_value=1.0,
        ).to(dtype=torch.int32)
        return targets, lengths

    def _train_extract_features(self, samples: List):
        mel_features = [spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.train_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths

    def _valid_extract_features(self, samples: List):
        mel_features = [spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.valid_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
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

        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
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
                {"scheduler": self.warmup_lr_scheduler, "interval": "step"},
            ],
        )

    def forward(self, batch: Batch):
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        hypotheses = decoder(batch.features.to(self.device), batch.feature_lengths.to(self.device), 20)
        return post_process_hypos(hypotheses, self.sp_model)[0][0]

    def training_step(self, batch: Batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def train_dataloader(self):
        dataset = CustomDataset(torchaudio.datasets.TEDLIUM(self.tedlium_path, release="release3", subset="train"), 100)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._train_collate_fn,
            num_workers=10,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataset = CustomDataset(torchaudio.datasets.TEDLIUM(self.tedlium_path, release="release3", subset="dev"), 100)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._valid_collate_fn,
            num_workers=10,
        )
        return dataloader

    def test_dataloader(self):
        dataset = torchaudio.datasets.TEDLIUM(self.tedlium_path, release="release3", subset="test")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader

    def dev_dataloader(self):
        dataset = torchaudio.datasets.TEDLIUM(self.tedlium_path, release="release3", subset="dev")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader
