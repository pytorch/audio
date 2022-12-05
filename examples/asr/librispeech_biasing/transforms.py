import json
import math
import random
from functools import partial
from typing import List

import sentencepiece as spm
import torch
import torchaudio
from data_module import LibriSpeechDataModule
from lightning import Batch


_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)

_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)

random.seed(999)


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
    biasingwords = []
    for sample in samples:
        for word in sample[6]:
            if word not in biasingwords:
                biasingwords.append(word)
    lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
    targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(elem) for elem in targets],
        batch_first=True,
        padding_value=1.0,
    ).to(dtype=torch.int32)
    return targets, lengths, biasingwords


def _extract_features(data_pipeline, samples: List):
    mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
    features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
    features = data_pipeline(features)
    lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
    return features, lengths


def _extract_tries(sp_model, biasingwords, blist, droprate, maxsize):
    if len(biasingwords) > 0 and droprate > 0:
        biasingwords = random.sample(biasingwords, k=int(len(biasingwords) * (1 - droprate)))
    if len(biasingwords) < maxsize:
        distractors = random.sample(blist, k=max(0, maxsize - len(biasingwords)))
        for word in distractors:
            if word not in biasingwords:
                biasingwords.append(word)
    biasingwords = [sp_model.encode(word.lower()) for word in biasingwords]
    biasingwords = sorted(biasingwords)
    worddict = {tuple(word): i + 1 for i, word in enumerate(biasingwords)}
    lextree = make_lexical_tree(worddict, -1)
    return lextree, biasingwords


def make_lexical_tree(word_dict, word_unk):
    """Make a prefix tree"""
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:
            succ = root[0]
            for i, cid in enumerate(w):
                if cid not in succ:
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w) - 1:
                    succ[cid][1] = wid
                succ = succ[cid][0]
    return root


class TrainTransform:
    def __init__(self, global_stats_path: str, sp_model_path: str, blist: list,
                 droprate: float, maxsize: int):
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(_piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )
        self.blist = blist
        self.droprate = droprate
        self.maxsize = maxsize

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.train_data_pipeline, samples)
        targets, target_lengths, biasingwords = _extract_labels(self.sp_model, samples)
        tries, biasingwords = _extract_tries(self.sp_model, biasingwords, self.blist, self.droprate, self.maxsize)
        return Batch(features, feature_lengths, targets, target_lengths, tries)


class ValTransform:
    def __init__(self, global_stats_path: str, sp_model_path: str, blist: list,
                 droprate: float, maxsize: int):
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(_piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
        )
        self.blist = blist
        self.droprate = droprate
        self.maxsize = maxsize

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.valid_data_pipeline, samples)
        targets, target_lengths, biasingwords = _extract_labels(self.sp_model, samples)
        if self.blist != []:
            tries, biasingwords = _extract_tries(self.sp_model, biasingwords, self.blist, self.droprate, self.maxsize)
        else:
            tries = []
        return Batch(features, feature_lengths, targets, target_lengths, tries)


class TestTransform:
    def __init__(self, global_stats_path: str, sp_model_path: str, blist: list,
                 droprate: float, maxsize: int):
        self.val_transforms = ValTransform(global_stats_path, sp_model_path, blist, droprate, maxsize)

    def __call__(self, sample):
        return self.val_transforms([sample]), [sample]


def get_data_module(librispeech_path, global_stats_path, sp_model_path, subset="", biasinglist="",
                    droprate=0.0, maxsize=1000):
    fullbiasinglist = []
    if biasinglist != "":
        with open(biasinglist) as fin:
            fullbiasinglist = [line.strip() for line in fin]
    train_transform = TrainTransform(global_stats_path=global_stats_path, sp_model_path=sp_model_path,
                                     blist=fullbiasinglist, droprate=droprate, maxsize=maxsize)
    val_transform = ValTransform(global_stats_path=global_stats_path, sp_model_path=sp_model_path,
                                 blist=fullbiasinglist, droprate=droprate, maxsize=maxsize)
    test_transform = TestTransform(global_stats_path=global_stats_path, sp_model_path=sp_model_path,
                                   blist=fullbiasinglist, droprate=droprate, maxsize=maxsize)
    return LibriSpeechDataModule(
        librispeech_path=librispeech_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        subset=subset,
        fullbiasinglist=fullbiasinglist
    )
