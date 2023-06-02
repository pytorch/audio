import json
import math
from functools import partial
from typing import List

import sentencepiece as spm
import torch
import torchaudio
from data_module import LibriSpeechDataModule
from lightning import Batch

import torchaudio.transforms as T


_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)

_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)


def _piecewise_linear_log(x):
    x = x * _gain
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


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


def _extract_labels(sp_model, samples: List):
    targets = [sp_model.encode(sample[2].lower()) for sample in samples]
    lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
    targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(elem) for elem in targets],
        batch_first=True,
        padding_value=1.0,
    ).to(dtype=torch.int32)
    return targets, lengths


def _extract_features(data_pipeline, samples: List):
    mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
    features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
    features = data_pipeline(features)
    lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
    return features, lengths


class TrainTransform:
    def __init__(self, global_stats_path: str, sp_model_path: str):
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)

        spec_aug_transform = T.SpecAugment(
            n_time_masks=10,
            time_mask_param=30,
            p=0.2,
            n_freq_masks=2,
            freq_mask_param=27,
            iid_masks=True,
            zero_masking=True,
        )

        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(_piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            # torchaudio.transforms.FrequencyMasking(27),
            # torchaudio.transforms.FrequencyMasking(27),
            # torchaudio.transforms.TimeMasking(100, p=0.2),
            # torchaudio.transforms.TimeMasking(100, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            # torchaudio.transforms.TimeMasking(30, p=0.2),
            spec_aug_transform,
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.train_data_pipeline, samples)
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return Batch(features, feature_lengths, targets, target_lengths)


class ValTransform:
    def __init__(self, global_stats_path: str, sp_model_path: str):
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(_piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
        )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.valid_data_pipeline, samples)
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return Batch(features, feature_lengths, targets, target_lengths)


class TestTransform:
    def __init__(self, global_stats_path: str, sp_model_path: str):
        self.val_transforms = ValTransform(global_stats_path, sp_model_path)

    def __call__(self, sample):
        return self.val_transforms([sample]), [sample]


def get_data_module(librispeech_path, global_stats_path, sp_model_path):
    train_transform = TrainTransform(global_stats_path=global_stats_path, sp_model_path=sp_model_path)
    val_transform = ValTransform(global_stats_path=global_stats_path, sp_model_path=sp_model_path)
    test_transform = TestTransform(global_stats_path=global_stats_path, sp_model_path=sp_model_path)
    return LibriSpeechDataModule(
        librispeech_path=librispeech_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=None,
        max_tokens=1500,
        train_num_buckets=50,
    )
