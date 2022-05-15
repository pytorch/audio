import os
from functools import partial
from typing import List

import sentencepiece as spm
import torch
import torchaudio
from common import (
    Batch,
    batch_by_token_count,
    FunctionalModule,
    GlobalStatsNormalization,
    piecewise_linear_log,
    post_process_hypos,
    spectrogram_transform,
    WarmupLR,
)
from pytorch_lightning import LightningModule
from torchaudio.models import emformer_rnnt_base, RNNTBeamSearch


class CustomDataset(torch.utils.data.Dataset):
    r"""Sort TEDLIUM3 samples by target length and batch to max durations."""

    def __init__(self, base_dataset, max_token_limit):
        super().__init__()
        self.base_dataset = base_dataset

        idx_target_lengths = [
            (idx, self._target_length(fileid, line)) for idx, (fileid, line) in enumerate(self.base_dataset._filelist)
        ]
        idx_target_lengths = [(idx, length) for idx, length in idx_target_lengths if length != -1]

        assert len(idx_target_lengths) > 0

        idx_target_lengths = sorted(idx_target_lengths, key=lambda x: x[1])

        assert max_token_limit >= idx_target_lengths[-1][1]

        self.batches = batch_by_token_count(idx_target_lengths, max_token_limit)

    def _target_length(self, fileid, line):
        transcript_path = os.path.join(self.base_dataset._path, "stm", fileid)
        with open(transcript_path + ".stm") as f:
            transcript = f.readlines()[line]
            _, _, _, start_time, end_time, _, transcript = transcript.split(" ", 6)
            if transcript.lower() == "ignore_time_segment_in_scoring\n":
                return -1
            else:
                return float(end_time) - float(start_time)

    def __getitem__(self, idx):
        return [self.base_dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class EvalDataset(torch.utils.data.IterableDataset):
    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset

    def __iter__(self):
        for sample in iter(self.base_dataset):
            actual = sample[2].replace("\n", "")
            if actual == "ignore_time_segment_in_scoring":
                continue
            yield sample


class TEDLIUM3RNNTModule(LightningModule):
    def __init__(
        self,
        *,
        tedlium_path: str,
        sp_model_path: str,
        global_stats_path: str,
    ):
        super().__init__()

        self.model = emformer_rnnt_base(num_symbols=501)
        self.loss = torchaudio.transforms.RNNTLoss(reduction="mean", clamp=1.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 10000)

        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            FunctionalModule(partial(torch.nn.functional.pad, pad=(0, 4))),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            FunctionalModule(partial(torch.nn.functional.pad, pad=(0, 4))),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
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
        return self._valid_collate_fn(samples), [sample[2] for sample in samples]

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

    def test_step(self, batch_tuple, batch_idx):
        return self._step(batch_tuple[0], batch_idx, "test")

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
        dataset = EvalDataset(torchaudio.datasets.TEDLIUM(self.tedlium_path, release="release3", subset="test"))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader

    def dev_dataloader(self):
        dataset = EvalDataset(torchaudio.datasets.TEDLIUM(self.tedlium_path, release="release3", subset="dev"))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader
