from ._conformer_wav2vec2 import (
    conformer_wav2vec2_base as _conformer_wav2vec2_base,
    conformer_wav2vec2_model as _conformer_wav2vec2_model,
    conformer_wav2vec2_pretrain_base as _conformer_wav2vec2_pretrain_base,
    conformer_wav2vec2_pretrain_large as _conformer_wav2vec2_pretrain_large,
    conformer_wav2vec2_pretrain_model as _conformer_wav2vec2_pretrain_model,
    ConformerWav2Vec2PretrainModel as _ConformerWav2Vec2PretrainModel,
)
from ._emformer_hubert import (
    emformer_hubert_base as _emformer_hubert_base,
    emformer_hubert_model as _emformer_hubert_model
)
from .conv_emformer import ConvEmformer as _ConvEmformer
from .hifi_gan import (
    hifigan_vocoder as _hifigan_vocoder,
    hifigan_vocoder_v1 as _hifigan_vocoder_v1,
    hifigan_vocoder_v2 as _hifigan_vocoder_v2,
    hifigan_vocoder_v3 as _hifigan_vocoder_v3,
    HiFiGANVocoder as _HiFiGANVocoder
)
from .rnnt import (
    conformer_rnnt_base as _conformer_rnnt_base,
    conformer_rnnt_biasing as _conformer_rnnt_biasing,
    conformer_rnnt_biasing_base as _conformer_rnnt_biasing_base,
    conformer_rnnt_model as _conformer_rnnt_model
)
from .rnnt_decoder import Hypothesis as _Hypothesis, RNNTBeamSearchBiasing as _RNNTBeamSearchBiasing

from torchaudio._internal.module_utils import dropping_support, dropping_const_support, dropping_class_support


conformer_rnnt_base = dropping_support(_conformer_rnnt_base)
conformer_rnnt_model = dropping_support(_conformer_rnnt_model)
conformer_rnnt_biasing = dropping_support(_conformer_rnnt_biasing)
conformer_rnnt_biasing_base = dropping_support(_conformer_rnnt_biasing_base)
conformer_wav2vec2_model = dropping_support(_conformer_wav2vec2_model)
conformer_wav2vec2_base = dropping_support(_conformer_wav2vec2_base)
conformer_rnnt_biasing_base = dropping_support(_conformer_rnnt_biasing_base)
conformer_wav2vec2_pretrain_model = dropping_support(_conformer_wav2vec2_pretrain_model)
conformer_wav2vec2_pretrain_base = dropping_support(_conformer_wav2vec2_pretrain_base)
conformer_wav2vec2_pretrain_large = dropping_support(_conformer_wav2vec2_pretrain_large)
emformer_hubert_base = dropping_support(_emformer_hubert_base)
emformer_hubert_model = dropping_support(_emformer_hubert_model)
hifigan_vocoder = dropping_support(_hifigan_vocoder)
hifigan_vocoder_v1 = dropping_support(_hifigan_vocoder_v1)
hifigan_vocoder_v2 = dropping_support(_hifigan_vocoder_v2)
hifigan_vocoder_v3 = dropping_support(_hifigan_vocoder_v3)
ConvEmformer = dropping_class_support(_ConvEmformer)
ConformerWav2Vec2PretrainModel = dropping_class_support(_ConformerWav2Vec2PretrainModel)
RNNTBeamSearchBiasing = dropping_class_support(_RNNTBeamSearchBiasing)
HiFiGANVocoder = dropping_class_support(_HiFiGANVocoder)
Hypothesis = dropping_const_support(_Hypothesis, name="Hypothesis")



__all__ = [
    "conformer_rnnt_base",
    "conformer_rnnt_model",
    "conformer_rnnt_biasing",
    "conformer_rnnt_biasing_base",
    "ConvEmformer",
    "conformer_wav2vec2_model",
    "conformer_wav2vec2_base",
    "conformer_wav2vec2_pretrain_model",
    "conformer_wav2vec2_pretrain_base",
    "conformer_wav2vec2_pretrain_large",
    "ConformerWav2Vec2PretrainModel",
    "emformer_hubert_base",
    "emformer_hubert_model",
    "Hypothesis",
    "RNNTBeamSearchBiasing",
    "HiFiGANVocoder",
    "hifigan_vocoder_v1",
    "hifigan_vocoder_v2",
    "hifigan_vocoder_v3",
    "hifigan_vocoder",
]
