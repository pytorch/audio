from collections import namedtuple
import json
import math
import os
import random
from typing import List, Optional, Tuple

import sentencepiece as spm

import torch
import torchaudio
from torchaudio.models.rnnt import _CustomLSTM, _TimeReduction
from torchaudio.models import Conformer, Hypothesis, RNNT, RNNTBeamSearch
from pytorch_lightning import LightningModule, seed_everything

import logging

logger = logging.getLogger()

seed_everything(1)


class Transcriber(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_reduction = _TimeReduction(4)
        self.input_linear = torch.nn.Linear(320, 256)
        self.conformer = Conformer(
            num_layers=16,
            input_dim=256,
            ffn_dim=1024,
            num_attention_heads=4,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
        )
        self.output_linear = torch.nn.Linear(256, 1024)
        self.layer_norm = torch.nn.LayerNorm(1024)

    def forward(self, input, lengths):
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, lengths = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, lengths


class Predictor(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    Args:
        num_symbols (int): size of target token lexicon.
        output_dim (int): feature dimension of each output sequence element.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_layer_norm (bool, optional): if ``True``, enables layer normalization
            for LSTM layers. (Default: ``False``)
        lstm_layer_norm_epsilon (float, optional): value of epsilon to use in
            LSTM layer normalization layers. (Default: 1e-5)
        lstm_dropout (float, optional): LSTM dropout probability. (Default: 0.0)

    """

    def __init__(
        self,
        num_symbols: int,
        output_dim: int,
        symbol_embedding_dim: int,
        num_lstm_layers: int,
        lstm_hidden_dim: int,
        lstm_layer_norm: bool = False,
        lstm_layer_norm_epsilon: float = 1e-5,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_symbols, symbol_embedding_dim)
        self.input_layer_norm = torch.nn.LayerNorm(symbol_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                _CustomLSTM(
                    symbol_embedding_dim if idx == 0 else lstm_hidden_dim,
                    lstm_hidden_dim,
                    layer_norm=lstm_layer_norm,
                    layer_norm_epsilon=lstm_layer_norm_epsilon,
                )
                for idx in range(num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=lstm_dropout)
        self.linear = torch.nn.Linear(lstm_hidden_dim, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.lstm_dropout = lstm_dropout

    def forward(
        self, input: torch.Tensor, lengths: torch.Tensor, state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(lstm_out, None if state is None else state[layer_idx])
            lstm_out = self.dropout(lstm_out)
            state_out.append(lstm_state_out)

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out.permute(1, 0, 2), lengths, state_out


class Joiner(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.tanh = torch.nn.Tanh()

    def forward(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, D)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        tanh_out = self.tanh(joint_encodings)
        output = self.linear(tanh_out)
        return output, source_lengths, target_lengths


encoder = Transcriber()
decoder = Predictor(
    num_symbols=1024,
    output_dim=1024,
    symbol_embedding_dim=256,
    num_lstm_layers=2,
    lstm_hidden_dim=512,
    lstm_layer_norm=True,
    lstm_layer_norm_epsilon=1e-5,
    lstm_dropout=0.3,
)
joiner = Joiner(1024, 1024)
rnnt = RNNT(encoder, decoder, joiner)

Batch = namedtuple("Batch", ["features", "feature_lengths", "targets", "target_lengths"])


_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)

_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)


def _piecewise_linear_log(x):
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


def _batch_by_token_count(idx_target_lengths, token_limit, sample_limit=None):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > token_limit or (sample_limit and len(current_batch) == sample_limit):
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


def get_sample_lengths(librispeech_dataset):
    fileid_to_target_length = {}

    def _target_length(fileid):
        if fileid not in fileid_to_target_length:
            speaker_id, chapter_id, _ = fileid.split("-")

            file_text = speaker_id + "-" + chapter_id + librispeech_dataset._ext_txt
            file_text = os.path.join(librispeech_dataset._path, speaker_id, chapter_id, file_text)

            with open(file_text) as ft:
                for line in ft:
                    fileid_text, transcript = line.strip().split(" ", 1)
                    fileid_to_target_length[fileid_text] = len(transcript)

        return fileid_to_target_length[fileid]

    return [_target_length(fileid) for fileid in librispeech_dataset._walker]


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, lengths, max_token_limit, num_buckets, shuffle=False):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_token_limit >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [(idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)]
        if shuffle:
            idx_length_buckets = random.sample(idx_length_buckets, len(idx_length_buckets))
        else:
            idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[1], reverse=True)

        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets], max_token_limit
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

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
    def __init__(self, optimizer, warmup_updates, force_anneal_epoch, anneal_factor, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.force_anneal_epoch = force_anneal_epoch
        self.anneal_factor = anneal_factor
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count < self.force_anneal_epoch:
            return [(min(1.0, self._step_count / self.warmup_updates)) * base_lr for base_lr in self.base_lrs]
        else:
            scaling_factor = self.anneal_factor ** (self._step_count - self.force_anneal_epoch)
            return [scaling_factor * base_lr for base_lr in self.base_lrs]


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
        self, *, librispeech_path: str, sp_model_path: str, global_stats_path: str,
    ):
        super().__init__()

        self.model = rnnt
        self.loss = torchaudio.transforms.RNNTLoss(reduction="mean")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=8e-4, betas=(0.9, 0.98), eps=1e-9)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 40, 120, 0.96)

        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(lambda x: x.transpose(1, 2)),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            FunctionalModule(lambda x: x.transpose(1, 2)),
        )
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)), GlobalStatsNormalization(global_stats_path),
        )

        self.librispeech_path = librispeech_path

        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.blank_idx = self.sp_model.get_piece_size()
        self.train_dataset_lengths = None
        self.val_dataset_lengths = None

    def _extract_labels(self, samples: List):
        targets = [self.sp_model.encode(sample[2].lower()) for sample in samples]
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets], batch_first=True, padding_value=1.0,
        ).to(dtype=torch.int32)
        return targets, lengths

    def _train_extract_features(self, samples: List):
        mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.train_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths

    def _valid_extract_features(self, samples: List):
        mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
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

    def _step(self, batch, _, step_type):
        if batch is None:
            return None

        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1
        output, src_lengths, _, _ = self.model(
            batch.features, batch.feature_lengths, prepended_targets, prepended_target_lengths,
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [{"scheduler": self.warmup_lr_scheduler, "interval": "epoch"}],
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
        datasets = [
            torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="train-clean-360"),
            torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="train-clean-100"),
            torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="train-other-500"),
        ]

        if not self.train_dataset_lengths:
            self.train_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(dataset, lengths, 700, 50, shuffle=True)
                for dataset, lengths in zip(datasets, self.train_dataset_lengths)
            ]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, collate_fn=self._train_collate_fn, num_workers=10, batch_size=None,
        )
        return dataloader

    def val_dataloader(self):
        datasets = [
            torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="dev-clean"),
            torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="dev-other"),
        ]

        if not self.val_dataset_lengths:
            self.val_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(dataset, lengths, 700, 1)
                for dataset, lengths in zip(datasets, self.val_dataset_lengths)
            ]
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, collate_fn=self._valid_collate_fn, num_workers=10,
        )
        return dataloader

    def test_dataloader(self):
        dataset = torchaudio.datasets.LIBRISPEECH(self.librispeech_path, url="test-clean")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=self._test_collate_fn)
        return dataloader
