from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torchaudio._internal import load_state_dict_from_url

from torchaudio.prototype.models.hifi_gan import hifigan_vocoder, HiFiGANVocoder
from torchaudio.transforms import MelSpectrogram


@dataclass
class HiFiGANVocoderBundle:
    """Data class that bundles associated information to use pretrained
    :py:class:`~torchaudio.prototype.models.HiFiGANVocoder`.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    This bundle can convert mel spectrorgam to waveforms and vice versa. A typical use case would be a flow like
    `text -> mel spectrogram -> waveform`, where one can use an external component, e.g. Tacotron2,
    to generate mel spectrogram from text. Please see below for the code example.

    Example: Transform synthetic mel spectrogram to audio.
        >>> import torch
        >>> import torchaudio
        >>> # Since HiFiGAN bundle is in prototypes, it needs to be exported explicitly
        >>> from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as bundle
        >>>
        >>> # Load the HiFiGAN bundle
        >>> vocoder = bundle.get_vocoder()
        Downloading: "https://download.pytorch.org/torchaudio/models/hifigan_vocoder_v3_ljspeech.pth"
        100%|████████████| 5.59M/5.59M [00:00<00:00, 18.7MB/s]
        >>>
        >>> # Generate synthetic mel spectrogram
        >>> specgram = torch.sin(0.5 * torch.arange(start=0, end=100)).expand(bundle._vocoder_params["in_channels"], 100)
        >>>
        >>> # Transform mel spectrogram into audio
        >>> waveform = vocoder(specgram)
        >>> torchaudio.save('sample.wav', waveform, bundle.sample_rate)

    Example: Usage together with Tacotron2, text to audio.
        >>> import torch
        >>> import torchaudio
        >>> # Since HiFiGAN bundle is in prototypes, it needs to be exported explicitly
        >>> from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH as bundle_hifigan
        >>>
        >>> # Load Tacotron2 bundle
        >>> bundle_tactron2 = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        >>> processor = bundle_tactron2.get_text_processor()
        >>> tacotron2 = bundle_tactron2.get_tacotron2()
        >>>
        >>> # Use Tacotron2 to convert text to mel spectrogram
        >>> text = "A quick brown fox jumped over a lazy dog"
        >>> input, lengths = processor(text)
        >>> specgram, lengths, _ = tacotron2.infer(input, lengths)
        >>>
        >>> # Load HiFiGAN bundle
        >>> vocoder = bundle_hifigan.get_vocoder()
        Downloading: "https://download.pytorch.org/torchaudio/models/hifigan_vocoder_v3_ljspeech.pth"
        100%|████████████| 5.59M/5.59M [00:03<00:00, 1.55MB/s]
        >>>
        >>> # Use HiFiGAN to convert mel spectrogram to audio
        >>> waveform = vocoder(specgram).squeeze(0)
        >>> torchaudio.save('sample.wav', waveform, bundle_hifigan.sample_rate)
    """  # noqa: E501

    _path: str
    _vocoder_params: Dict[str, Any]  # Vocoder parameters
    _mel_params: Dict[str, Any]  # Mel transformation parameters
    _sample_rate: float

    def _get_state_dict(self, dl_kwargs):
        url = f"https://download.pytorch.org/torchaudio/models/{self._path}"
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        return state_dict

    def get_vocoder(self, *, dl_kwargs=None) -> HiFiGANVocoder:
        """Construct the HiFiGAN Generator model, which can be used a vocoder, and load the pretrained weight.

        The weight file is downloaded from the internet and cached with
        :func:`torch.hub.load_state_dict_from_url`

        Args:
            dl_kwargs (dictionary of keyword arguments): Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Variation of :py:class:`~torchaudio.prototype.models.HiFiGANVocoder`.
        """
        model = hifigan_vocoder(**self._vocoder_params)
        model.load_state_dict(self._get_state_dict(dl_kwargs))
        model.eval()
        return model

    def get_mel_transform(self) -> Module:
        """Construct an object which transforms waveforms into mel spectrograms."""
        return _HiFiGANMelSpectrogram(
            n_mels=self._vocoder_params["in_channels"],
            sample_rate=self._sample_rate,
            **self._mel_params,
        )

    @property
    def sample_rate(self):
        """Sample rate of the audio that the model is trained on.

        :type: float
        """
        return self._sample_rate


class _HiFiGANMelSpectrogram(torch.nn.Module):
    """
    Generate mel spectrogram in a way equivalent to the original HiFiGAN implementation:
    https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/meldataset.py#L49-L72

    This class wraps around :py:class:`torchaudio.transforms.MelSpectrogram`, but performs extra steps to achive
    equivalence with the HiFiGAN implementation.

    Args:
        hop_size (int): Length of hop between STFT windows.
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins.
        win_length (int): Window size.
        f_min (float or None):  Minimum frequency.
        f_max (float or None): Maximum frequency.
        sample_rate (int):  Sample rate of audio signal.
        n_mels (int):  Number of mel filterbanks.
    """

    def __init__(
        self,
        hop_size: int,
        n_fft: int,
        win_length: int,
        f_min: Optional[float],
        f_max: Optional[float],
        sample_rate: float,
        n_mels: int,
    ):
        super(_HiFiGANMelSpectrogram, self).__init__()
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_size,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            normalized=False,
            pad=0,
            mel_scale="slaney",
            norm="slaney",
            center=False,
        )
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.pad_size = int((n_fft - hop_size) / 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Generate mel spectrogram from a waveform. Should have same sample rate as ``self.sample_rate``.

        Args:
            waveform (Tensor): waveform of shape ``(batch_size, time_length)``.
        Returns:
            Tensor of shape ``(batch_size, n_mel, time_length)``
        """
        ref_waveform = F.pad(waveform.unsqueeze(1), (self.pad_size, self.pad_size), mode="reflect")
        ref_waveform = ref_waveform.squeeze(1)

        spectr = (self.mel_transform.spectrogram(ref_waveform) + 1e-9) ** 0.5
        mel_spectrogram = self.mel_transform.mel_scale(spectr)
        mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        return mel_spectrogram


HIFIGAN_VOCODER_V3_LJSPEECH = HiFiGANVocoderBundle(
    "hifigan_vocoder_v3_ljspeech.pth",
    _vocoder_params={
        "upsample_rates": (8, 8, 4),
        "upsample_kernel_sizes": (16, 16, 8),
        "upsample_initial_channel": 256,
        "resblock_kernel_sizes": (3, 5, 7),
        "resblock_dilation_sizes": ((1, 2), (2, 6), (3, 12)),
        "resblock_type": 2,
        "in_channels": 80,
        "lrelu_slope": 0.1,
    },
    _mel_params={
        "hop_size": 256,
        "n_fft": 1024,
        "win_length": 1024,
        "f_min": 0,
        "f_max": 8000,
    },
    _sample_rate=22050,
)
HIFIGAN_VOCODER_V3_LJSPEECH.__doc__ = """HiFiGAN Vocoder pipeline, trained on *The LJ Speech Dataset*
    :cite:`ljspeech17`.

    This pipeine can be used with an external component which generates mel spectrograms from text, for example,
    Tacotron2 - see examples in :py:class:`HiFiGANVocoderBundle`.
    Although this works with the existing Tacotron2 bundles, for the best results one needs to retrain Tacotron2
    using the same data preprocessing pipeline which was used for training HiFiGAN. In particular, the original
    HiFiGAN implementation uses a custom method of generating mel spectrograms from waveforms, different from
    :py:class:`torchaudio.transforms.MelSpectrogram`. We reimplemented this transform as
    :py:meth:`HiFiGANVocoderBundle.get_mel_transform`, making sure it is equivalent to the original HiFiGAN code `here
    <https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/meldataset.py#L49-L72>`_.

    The underlying vocoder is constructed by
    :py:func:`torchaudio.prototype.models.hifigan_vocoder`. The weights are converted from the ones published
    with the original paper :cite:`NEURIPS2020_c5d73680` under `MIT License
    <https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/LICENSE>`__. See links to
    pre-trained models on `GitHub <https://github.com/jik876/hifi-gan#pretrained-model>`__.

    Please refer to :py:class:`HiFiGANVocoderBundle` for usage instructions.
    """
