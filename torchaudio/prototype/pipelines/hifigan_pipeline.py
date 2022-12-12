from dataclasses import dataclass
from typing import Any, Dict

from torchaudio._internal import load_state_dict_from_url

from torchaudio.prototype.models.hifi_gan import hifigan_generator, HiFiGANGenerator


@dataclass
class HiFiGANGeneratorBundle:
    """Data class that bundles associated information to use pretrained
    :py:class:`~torchaudio.prototype.models.HiFiGANGenerator`.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    Please see below for the usage example.

    Example: Synthetic spectrogram to audio.
        >>> import torchaudio
        >>> import torch
        >>>
        >>> bundle = torchaudio.prototypes.pipelines.HIFIGAN_GENERATOR_LJSPEECH_V3
        >>> vocoder = bundle.get_vocoder()
        >>> specgram = torch.sin(0.5 * torch.arange(start=0, end=100)).expand(bundle._params.in_channels, 100)
        >>> waveforms = vocoder(specgram)
        >>> torchaudio.save('sample.wav', waveforms, vocoder.sample_rate)


    Example: Usage together with Tactron2, text to audio.
        >>> import torchaudio
        >>> import torch
        >>>
        >>> text = "A quick brown fox jumped over a lazy dog"
        >>> bundle_tactron2 = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        >>> processor = bundle_tactron2.get_text_processor()
        >>> tacotron2 = bundle_tactron2.get_tacotron2()
        >>>
        >>> input, lengths = processor(text)
        >>>
        >>> specgram, lengths, _ = tacotron2.infer(input, lengths)
        >>>
        >>> bundle_hifigan = torchaudio.prototypes.pipelines.HIFIGAN_GENERATOR_LJSPEECH_V3
        >>>
        >>> vocoder = bundle_hifigan.get_vocoder()
        >>> waveforms = vocoder(specgram)
        >>> torchaudio.save('sample.wav', waveforms, vocoder.sample_rate)
    """  # noqa: E501

    _path: str
    _params: Dict[str, Any]
    _sample_rate: float = 22000

    def _get_state_dict(self, dl_kwargs):
        url = f"https://download.pytorch.org/torchaudio/models/{self._path}"
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        return state_dict

    def get_vocoder(self, *, dl_kwargs=None) -> HiFiGANGenerator:
        """Construct the HiFiGAN Generator model and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.prototype.models.HiFiGANGenerator`.
        """
        model = hifigan_generator(**self._params)
        model.load_state_dict(self._get_state_dict(dl_kwargs))
        model.eval()
        return model

    @property
    def sample_rate(self):
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate


HIFIGAN_GENERATOR_LJSPEECH_V3 = HiFiGANGeneratorBundle(
    "hifigan_generator_ljspeech_v3.pth",
    _params={
        "upsample_rates": (8, 8, 4),
        "upsample_kernel_sizes": (16, 16, 8),
        "upsample_initial_channel": 256,
        "resblock_kernel_sizes": (3, 5, 7),
        "resblock_dilation_sizes": ((1, 2), (2, 6), (3, 12)),
        "resblock_type": 2,
        "in_channels": 80,
        "lrelu_slope": 0.1,
    },
)
HIFIGAN_GENERATOR_LJSPEECH_V3.__doc__ = """Pre-trained HiFiGAN Generator pipeline, transforming Mel spectrograms into
    waveforms. The underlying model is constructed by :py:func:`torchaudio.prototype.models.hifigan_generator`
    and utilizes weights trained on *The LJ Speech Dataset* :cite:`ljspeech17`. The weights are converted from the ones
    published with the original paper :cite:`NEURIPS2020_c5d73680` (See links to pre-trained models on
    `GitHub <https://github.com/jik876/hifi-gan#pretrained-model>`__).

    Please refer to :py:class:`torchaudio.prototype.pipelines.HiFiGANGeneratorBundle` for usage instructions.
    """
