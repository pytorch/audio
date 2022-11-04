from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torchaudio

from torchaudio.models import conv_tasnet_base, hdemucs_high


@dataclass
class SourceSeparationBundle:
    """Dataclass that bundles components for performing source separation.

    Example
        >>> import torchaudio
        >>> from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX
        >>> import torch
        >>>
        >>> # Build the separation model.
        >>> model = CONVTASNET_BASE_LIBRI2MIX.get_model()
        >>> 100%|███████████████████████████████|19.1M/19.1M [00:04<00:00, 4.93MB/s]
        >>>
        >>> # Instantiate the test set of Libri2Mix dataset.
        >>> dataset = torchaudio.datasets.LibriMix("/home/datasets/", subset="test")
        >>>
        >>> # Apply source separation on mixture audio.
        >>> for i, data in enumerate(dataset):
        >>>     sample_rate, mixture, clean_sources = data
        >>>     # Make sure the shape of input suits the model requirement.
        >>>     mixture = mixture.reshape(1, 1, -1)
        >>>     estimated_sources = model(mixture)
        >>>     score = si_snr_pit(estimated_sources, clean_sources) # for demonstration
        >>>     print(f"Si-SNR score is : {score}.)
        >>>     break
        >>> Si-SNR score is : 16.24.
        >>>
    """

    _model_path: str
    _model_factory_func: Callable[[], torch.nn.Module]
    _sample_rate: int

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio that the model is trained on.

        :type: int
        """
        return self._sample_rate

    def get_model(self) -> torch.nn.Module:
        """Construct the model and load the pretrained weight."""
        model = self._model_factory_func()
        path = torchaudio.utils.download_asset(self._model_path)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.eval()
        return model


CONVTASNET_BASE_LIBRI2MIX = SourceSeparationBundle(
    _model_path="models/conv_tasnet_base_libri2mix.pt",
    _model_factory_func=partial(conv_tasnet_base, num_sources=2),
    _sample_rate=8000,
)
CONVTASNET_BASE_LIBRI2MIX.__doc__ = """Pre-trained Source Separation pipeline with *ConvTasNet*
:cite:`Luo_2019` trained on *Libri2Mix dataset* :cite:`cosentino2020librimix`.

The source separation model is constructed by :func:`~torchaudio.models.conv_tasnet_base`
and is trained using the training script ``lightning_train.py``
`here <https://github.com/pytorch/audio/tree/release/0.12/examples/source_separation/>`__
with default arguments.

Please refer to :class:`SourceSeparationBundle` for usage instructions.
"""


HDEMUCS_HIGH_MUSDB_PLUS = SourceSeparationBundle(
    _model_path="models/hdemucs_high_trained.pt",
    _model_factory_func=partial(hdemucs_high, sources=["drums", "bass", "other", "vocals"]),
    _sample_rate=44100,
)
HDEMUCS_HIGH_MUSDB_PLUS.__doc__ = """Pre-trained music source separation pipeline with
*Hybrid Demucs* :cite:`defossez2021hybrid` trained on both training and test sets of
MUSDB-HQ :cite:`MUSDB18HQ` and an additional 150 extra songs from an internal database
that was specifically produced for Meta.

The model is constructed by :func:`~torchaudio.models.hdemucs_high`.

Training was performed in the original HDemucs repository `here <https://github.com/facebookresearch/demucs/>`__.

Please refer to :class:`SourceSeparationBundle` for usage instructions.
"""


HDEMUCS_HIGH_MUSDB = SourceSeparationBundle(
    _model_path="models/hdemucs_high_musdbhq_only.pt",
    _model_factory_func=partial(hdemucs_high, sources=["drums", "bass", "other", "vocals"]),
    _sample_rate=44100,
)
HDEMUCS_HIGH_MUSDB.__doc__ = """Pre-trained music source separation pipeline with
*Hybrid Demucs* :cite:`defossez2021hybrid` trained on the training set of MUSDB-HQ :cite:`MUSDB18HQ`.

The model is constructed by :func:`~torchaudio.models.hdemucs_high`.
Training was performed in the original HDemucs repository `here <https://github.com/facebookresearch/demucs/>`__.

Please refer to :class:`SourceSeparationBundle` for usage instructions.
"""
