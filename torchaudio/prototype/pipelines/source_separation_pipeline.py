from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch
import torchaudio

from torchaudio.prototype.models import conv_tasnet_base


class _FeatureEncoder(torch.nn.Module):
    """Feature encoder for source separation.
    If the separator is an end-to-end model (waveform to waveform), the
    feature encoder can be torch.nn.Identity(). If the separator is
    time-frequency masking based model, the encoder can be torchaudio.transforms.Spectrogram().
    """

    def __init__(self, feature_encoder: torch.nn.Module):
        super().__init__()
        self.feature_encoder = feature_encoder

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Generates features for source separation given the input Tensor.

        Args:
            input (torch.Tensor): input tensor.

        Returns:
            (torch.Tensor): Features. The dimensions depend on the type of separator.
        """
        feature = self.feature_encoder(input)
        return feature


class _FeatureDecoder(torch.nn.Module):
    """Feature decoder for source separation.
    If the separator is an end-to-end model (waveform to waveform), the
    feature decoder can be torch.nn.Identity(). If the separator is
    time-frequency masking based model, the encoder can be torchaudio.transforms.InverseSpectrogram().
    """

    def __init__(self, feature_decoder: torch.nn.Module):
        super().__init__()
        self.feature_decoder = feature_decoder

    def forward(self, separated_sources: torch.Tensor, length: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decodes the separator output to waveforms.

        Args:
            separated_sources (torch.Tensor): separated sources.
            length (torch.Tensor or None, optional): The expected lengths of the waveform.

        Returns:
            (torch.Tensor): The separated waveforms.
        """
        if length is None:
            return self.feature_decoder(separated_sources)
        else:
            return self.feature_decoder(separated_sources, length)


class _Separator(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Separates the input mixture speech into different sources.

        Args:
            input (torch.Tensor): input tensor with dimensions `[batch, time]`.

        Returns:
            (torch.Tensor): Separated sources with dimensions `[batch, n_source, time]`.
        """
        output = self.model(feature)
        return output


@dataclass
class SourceSeparationBundle:
    """torchaudio.prototype.pipelines.SourceSeparationBundle()

    Dataclass that bundles components for performing source separation.

    Example
        >>> import torchaudio
        >>> from torchaudio.prototype.pipelines import CONVTASNET_BASE_LIBRI2MIX
        >>> import torch
        >>>
        >>> # Build feature encoder, feature decoder, and separator model.
        >>> feature_encoder = CONVTASNET_BASE_LIBRI2MIX.get_feature_encoder()
        >>> feature_decoder = CONVTASNET_BASE_LIBRI2MIX.get_feature_decoder()
        >>> separator = CONVTASNET_BASE_LIBRI2MIX.get_separator()
        >>> 100%|███████████████████████████████|19.1M/19.1M [00:04<00:00, 4.93MB/s]
        >>>
        >>> # Instantiate the test set of Libri2Mix dataset.
        >>> dataset = torchaudio.datasets.LibriMix("/home/datasets/", subset="test")
        >>>
        >>> # Apply source separation on mixture audio.
        >>> for i, data in enumerate(dataset):
        >>>     sample_rate, mixture, clean_sources = data
        >>>     feature = feature_encoder(mixture)
        >>>     output = separator(feature)
        >>>     estimated_sources = feature_decoder(output)
        >>>     score = si_snr_pit(estimated_sources, clean_sources) # for demonstration
        >>>     print(f"Si-SNR score is : {score}.)
        >>>     break
        >>> 16.24
        >>>
    """

    class FeatureEncoder(_FeatureEncoder):
        pass

    class FeatureDecoder(_FeatureDecoder):
        pass

    class Separator(_Separator):
        pass

    _feature_encoder: torch.nn.Module
    _feature_decoder: torch.nn.Module
    _model_path: str
    _separator_factory_func: Callable[[], torch.nn.Module]
    _sample_rate: int

    @property
    def sample_rate(self) -> int:
        """Sample rate (in cycles per second) of input waveforms.
        :type: int
        """
        return self._sample_rate

    def get_feature_encoder(self) -> FeatureEncoder:
        """Constructs feature encoder.
        Returns:
            FeatureEncoder
        """
        return _FeatureEncoder(self._feature_encoder)

    def get_feature_decoder(self) -> FeatureDecoder:
        """Constructs feature decoder.
        Returns:
            FeatureDecoder
        """
        return _FeatureDecoder(self._feature_decoder)

    def get_separator(self) -> Separator:
        model = self._separator_factory_func()
        path = torchaudio.utils.download_asset(self._model_path)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.eval()
        return _Separator(model)


class View(torch.nn.Module):
    """View module to reshape the input waveforms for ConvTasNet.
    The final shape of the input Tensor is `(batch, 1, time)`.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        if input.ndim == 1:
            input = input.reshape(1, 1, -1)
        elif input.ndim == 2:
            input = input.unsqueeze(dim=1)
        else:
            input = input
        return input


CONVTASNET_BASE_LIBRI2MIX = SourceSeparationBundle(
    _model_path="models/conv_tasnet_base_libri2mix.pt",
    _feature_encoder=View(),
    _feature_decoder=View(),
    _separator_factory_func=partial(conv_tasnet_base, num_sources=2),
    _sample_rate=8000,
)
CONVTASNET_BASE_LIBRI2MIX.__doc__ = """Pre-trained ConvTasNet pipeline for source separation.
    The underlying model is constructed by :py:func:`torchaudio.prototyoe.models.conv_tasnet_base`
    and utilizes weights trained on Libri2Mix using training script ``lightning_train.py``
    `here <https://github.com/pytorch/audio/tree/main/examples/source_separation/>`__ with default arguments.
    Please refer to :py:class:`SourceSeparationBundle` for usage instructions.
    """
