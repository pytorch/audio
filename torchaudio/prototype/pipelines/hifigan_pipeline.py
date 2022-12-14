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
        >>> import torch
        >>> import torchaudio
        >>> # Since HiFiGAN bundle is in prototypes, it needs to be exported explicitly
        >>> from torchaudio.prototype.pipelines import HIFIGAN_GENERATOR_V3_LJSPEECH as bundle
        >>>
        >>> # Load the HiFiGAN bundle
        >>> vocoder = bundle.get_vocoder()
        Downloading: "https://download.pytorch.org/torchaudio/models/hifigan_generator_v3_ljspeech.pth"
        100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.59M/5.59M [00:00<00:00, 18.7MB/s]
        >>>
        >>> # Generate synthetic mel spectrogram
        >>> specgram = torch.sin(0.5 * torch.arange(start=0, end=100)).expand(bundle._params["in_channels"], 100)
        >>>
        >>> # Trasform mel spectrogram into audio
        >>> waveforms = vocoder(specgram)
        >>> torchaudio.save('sample.wav', waveforms, bundle.sample_rate)


    Example: Usage together with Tactron2, text to audio.
        >>> import torch
        >>> import torchaudio
        >>> # Since HiFiGAN bundle is in prototypes, it needs to be exported explicitly
        >>> from torchaudio.prototype.pipelines import HIFIGAN_GENERATOR_V3_LJSPEECH as bundle_hifigan
        >>>
        >>> # Load Tactron2 bundle
        >>> bundle_tactron2 = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        >>> processor = bundle_tactron2.get_text_processor()
        >>> tacotron2 = bundle_tactron2.get_tacotron2()
        >>>
        >>> # Use Tactron2 to convert text to mel spectrogram
        >>> text = "A quick brown fox jumped over a lazy dog"
        >>> input, lengths = processor(text)
        >>> specgram, lengths, _ = tacotron2.infer(input, lengths)
        >>>
        >>> # Load HiFiGAN bundle
        >>> vocoder = bundle_hifigan.get_vocoder()
        Downloading: "https://download.pytorch.org/torchaudio/models/hifigan_generator_v3_ljspeech.pth"
        100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.59M/5.59M [00:03<00:00, 1.55MB/s]
        >>>
        >>> # Use HiFiGAN to convert mel spectrogram to audio
        >>> waveforms = vocoder(specgram).squeeze(0)
        >>> torchaudio.save('sample.wav', waveforms, bundle_hifigan.sample_rate)
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


HIFIGAN_GENERATOR_V3_LJSPEECH = HiFiGANGeneratorBundle(
    "hifigan_generator_v3_ljspeech.pth",
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
HIFIGAN_GENERATOR_V3_LJSPEECH.__doc__ = """Pre-trained HiFiGAN Generator pipeline, transforming mspectrograms into
    waveforms. The underlying model is constructed by :py:func:`torchaudio.prototype.models.hifigan_generator`
    and utilizes weights trained on *The LJ Speech Dataset* :cite:`ljspeech17`. The weights are converted from the ones
    published with the original paper :cite:`NEURIPS2020_c5d73680` (See links to pre-trained models on
    `GitHub <https://github.com/jik876/hifi-gan#pretrained-model>`__).

    Please refer to :py:class:`torchaudio.prototype.pipelines.HiFiGANGeneratorBundle` for usage instructions.
    """
