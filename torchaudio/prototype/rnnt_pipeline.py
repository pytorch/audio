from typing import Callable, List, Tuple
from dataclasses import dataclass
import json
import math
import os
import sys
import torch

import sentencepiece as spm
import torchaudio
from torchaudio._internal import download_url_to_file, load_state_dict_from_url
from torchaudio.prototype import RNNT, RNNTBeamSearch, emformer_rnnt_base


__all__ = []


_BASE_MODELS_URL = "https://download.pytorch.org/torchaudio/models"
_BASE_PIPELINES_URL = "https://download.pytorch.org/torchaudio/pipeline-assets"


def _download_asset(asset_path: str):
    dst_path = f"{os.getcwd()}/_assets/{asset_path}"
    if not os.path.exists(dst_path):
        os.makedirs(f"{os.getcwd()}/_assets", exist_ok=True)
        download_url_to_file(f"{_BASE_PIPELINES_URL}/{asset_path}", dst_path)
    else:
        sys.stderr.write(f"{asset_path} found at {dst_path}; skipping download.\n")
        sys.stderr.flush()
    return dst_path


_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)


def _piecewise_linear_log(x):
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class _FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class _GlobalStatsNormalization(torch.nn.Module):
    def __init__(self, global_stats_path):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob["mean"])
        self.invstddev = torch.tensor(blob["invstddev"])

    def forward(self, input):
        return (input - self.mean) * self.invstddev


class FeatureExtractor(torch.nn.Module):
    def __init__(self, pipeline: torch.nn.Module) -> None:
        super().__init__()
        self.pipeline = pipeline

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.pipeline(input)
        lengths = torch.tensor([features.shape[0]])
        return features, lengths


class SentencePieceTokenProcessor:
    def __init__(self, sp_model_path: str) -> None:
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.post_process_remove_list = {
            self.sp_model.unk_id(),
            self.sp_model.eos_id(),
            self.sp_model.pad_id(),
        }

    def __call__(self, tokens: List[str]) -> str:
        filtered_hypo_tokens = [
            token_index for token_index in tokens[1:] if token_index not in self.post_process_remove_list
        ]
        return self.sp_model.decode(filtered_hypo_tokens)


@dataclass
class RNNTBundle:
    _rnnt_path: str
    _rnnt_factory_func: Callable[[], RNNT]
    _global_stats_path: str
    _sp_model_path: str
    _right_padding: int
    _blank: int

    def _get_model(self) -> RNNT:
        model = self._rnnt_factory_func()
        url = f"{_BASE_MODELS_URL}/{self._rnnt_path}"
        state_dict = load_state_dict_from_url(url)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_decoder(self) -> RNNTBeamSearch:
        model = self._get_model()
        return RNNTBeamSearch(model, self._blank)

    def get_feature_extractor(self) -> FeatureExtractor:
        local_path = _download_asset(self._global_stats_path)
        return FeatureExtractor(
            torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160),
                _FunctionalModule(lambda x: x.transpose(1, 0)),
                _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
                _GlobalStatsNormalization(local_path),
                _FunctionalModule(lambda x: torch.nn.functional.pad(x, (0, 0, 0, self._right_padding))),
            )
        )

    def get_streaming_feature_extractor(self) -> FeatureExtractor:
        local_path = _download_asset(self._global_stats_path)
        return FeatureExtractor(
            torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160),
                _FunctionalModule(lambda x: x.transpose(1, 0)),
                _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
                _GlobalStatsNormalization(local_path),
            )
        )

    def get_token_processor(self) -> SentencePieceTokenProcessor:
        local_path = _download_asset(self._sp_model_path)
        return SentencePieceTokenProcessor(local_path)


EMFORMER_RNNT_BASE_LIBRISPEECH = RNNTBundle(
    _rnnt_path="emformer_rnnt_base_librispeech.pt",
    _rnnt_factory_func=emformer_rnnt_base,
    _global_stats_path="global_stats_rnnt_librispeech.json",
    _sp_model_path="spm_bpe_4096_librispeech.model",
    _right_padding=4,
    _blank=4096,
)
