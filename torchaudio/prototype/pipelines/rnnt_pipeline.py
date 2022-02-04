from functools import partial

from torchaudio.models import emformer_rnnt_base
from torchaudio.pipelines import RNNTBundle


EMFORMER_RNNT_BASE_TEDLIUM3 = RNNTBundle(
    _rnnt_path="emformer_rnnt_base_tedlium3.pt",
    _rnnt_factory_func=partial(emformer_rnnt_base, num_symbols=501),
    _global_stats_path="global_stats_rnnt_tedlium3.json",
    _sp_model_path="spm_bpe_500_tedlium3.model",
    _right_padding=4,
    _blank=500,
    _sample_rate=16000,
    _n_fft=400,
    _n_mels=80,
    _hop_length=160,
    _segment_length=16,
    _right_context_length=4,
)
EMFORMER_RNNT_BASE_TEDLIUM3.__doc__ = """Pre-trained Emformer-RNNT-based ASR pipeline capable of performing both streaming and non-streaming inference.

    The underlying model is constructed by :py:func:`torchaudio.models.emformer_rnnt_base`
    and utilizes weights trained on TED-LIUM Release 3 dataset using training script ``train.py``
    `here <https://github.com/pytorch/audio/tree/main/examples/asr/tedlium3_emformer_rnnt>`__ with ``num_symbols=501``.

    Please refer to :py:class:`torchaudio.pipelines.RNNTBundle` for usage instructions.
    """
