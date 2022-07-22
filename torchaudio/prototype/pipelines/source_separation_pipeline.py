from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torchaudio

from torchaudio.prototype.models import conv_tasnet_base


@dataclass
class SourceSeparationBundle:
    """torchaudio.prototype.pipelines.SourceSeparationBundle()

    Dataclass that bundles components for performing source separation.

    Example
        >>> import torchaudio
        >>> from torchaudio.prototype.pipelines import CONVTASNET_BASE_LIBRI2MIX
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
CONVTASNET_BASE_LIBRI2MIX.__doc__ = """Pre-trained *ConvTasNet* [:footcite:`Luo_2019`] pipeline for source separation.

    The underlying model is constructed by :py:func:`torchaudio.prototyoe.models.conv_tasnet_base`
    and utilizes weights trained on *Libri2Mix dataset* [:footcite:`cosentino2020librimix`] using training script
    ``lightning_train.py`` `here <https://github.com/pytorch/audio/tree/release/0.12/examples/source_separation/>`__
    with default arguments.

    Please refer to :py:class:`SourceSeparationBundle` for usage instructions.
    """
