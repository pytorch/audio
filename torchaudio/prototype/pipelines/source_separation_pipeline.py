from functools import partial

from torchaudio.models import hdemucs_high

from torchaudio.pipelines import SourceSeparationBundle


HDEMUCS_HIGH_MUSDB_PLUS = SourceSeparationBundle(
    _model_path="models/hdemucs_high_trained.pt",
    _model_factory_func=partial(hdemucs_high, sources=["drums", "bass", "other", "vocals"]),
    _sample_rate=44100,
)
HDEMUCS_HIGH_MUSDB_PLUS.__doc__ = """Pre-trained *Hybrid Demucs* [:footcite:`defossez2021hybrid`] pipeline for music
    source separation. The underlying model is constructed by
    :py:func:`torchaudio.prototype.models.hdemucs_high` and utilizes weights trained on MUSDB-HQ [:footcite:`MUSDB18HQ`]
    and internal extra training data, all at the same sample rate of 44.1 kHZ. The model separates mixture music into
    “drums”, “base”, “vocals”, and “other” sources. Training was performed in the original HDemucs repository
    `here <https://github.com/facebookresearch/demucs/>`__.
    """


HDEMUCS_HIGH_MUSDB = SourceSeparationBundle(
    _model_path="models/hdemucs_high_musdbhq_only.pt",
    _model_factory_func=partial(hdemucs_high, sources=["drums", "bass", "other", "vocals"]),
    _sample_rate=44100,
)
HDEMUCS_HIGH_MUSDB.__doc__ = """Pre-trained *Hybrid Demucs* [:footcite:`defossez2021hybrid`] pipeline for music
    source separation. The underlying model is constructed by
    :py:func:`torchaudio.prototype.models.hdemucs_high` and utilizes weights trained on only
    MUSDB-HQ [:footcite:`MUSDB18HQ`] at the same sample rate of 44.1 kHZ. The model separates mixture music into
    “drums”, “base”, “vocals”, and “other” sources. Training was performed in the original HDemucs repository
    `here <https://github.com/facebookresearch/demucs/>`__.
    """
