import re
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, Tuple, List

import torch
from torch import Tensor
from torchaudio._internal import load_state_dict_from_url
from torchaudio.functional import mu_law_decoding
from torchaudio.models import Tacotron2, WaveRNN
from torchaudio.transforms import InverseMelScale, GriffinLim

from . import utils
from .interface import Tacotron2TTSBundle

__all__ = []

_BASE_URL = "https://download.pytorch.org/torchaudio/models"


################################################################################
# Pipeline implementation - Text Processor
################################################################################


class _EnglishCharProcessor(Tacotron2TTSBundle.TextProcessor):
    def __init__(self):
        super().__init__()
        self._tokens = utils._get_chars()
        self._mapping = {s: i for i, s in enumerate(self._tokens)}

    @property
    def tokens(self):
        return self._tokens

    def __call__(self, texts: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        indices = [[self._mapping[c] for c in t.lower() if c in self._mapping] for t in texts]
        return utils._to_tensor(indices)


class _EnglishPhoneProcessor(Tacotron2TTSBundle.TextProcessor):
    def __init__(self, *, dl_kwargs=None):
        super().__init__()
        self._tokens = utils._get_phones()
        self._mapping = {p: i for i, p in enumerate(self._tokens)}
        self._phonemizer = utils._load_phonemizer("en_us_cmudict_forward.pt", dl_kwargs=dl_kwargs)
        self._pattern = r"(\[[A-Z]+?\]|[_!'(),.:;? -])"

    @property
    def tokens(self):
        return self._tokens

    def __call__(self, texts: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
        if isinstance(texts, str):
            texts = [texts]

        indices = []
        for phones in self._phonemizer(texts, lang="en_us"):
            # '[F][UW][B][AA][R]!' -> ['F', 'UW', 'B', 'AA', 'R', '!']
            ret = [re.sub(r"[\[\]]", "", r) for r in re.findall(self._pattern, phones)]
            indices.append([self._mapping[p] for p in ret])
        return utils._to_tensor(indices)


################################################################################
# Pipeline implementation - Vocoder
################################################################################


class _WaveRNNVocoder(torch.nn.Module, Tacotron2TTSBundle.Vocoder):
    def __init__(self, model: WaveRNN, min_level_db: Optional[float] = -100):
        super().__init__()
        self._sample_rate = 22050
        self._model = model
        self._min_level_db = min_level_db

    @property
    def sample_rate(self):
        return self._sample_rate

    def forward(self, mel_spec, lengths=None):
        mel_spec = torch.exp(mel_spec)
        mel_spec = 20 * torch.log10(torch.clamp(mel_spec, min=1e-5))
        if self._min_level_db is not None:
            mel_spec = (self._min_level_db - mel_spec) / self._min_level_db
            mel_spec = torch.clamp(mel_spec, min=0, max=1)
        waveform, lengths = self._model.infer(mel_spec, lengths)
        waveform = utils._unnormalize_waveform(waveform, self._model.n_bits)
        waveform = mu_law_decoding(waveform, self._model.n_classes)
        waveform = waveform.squeeze(1)
        return waveform, lengths


class _GriffinLimVocoder(torch.nn.Module, Tacotron2TTSBundle.Vocoder):
    def __init__(self):
        super().__init__()
        self._sample_rate = 22050
        self._inv_mel = InverseMelScale(
            n_stft=(1024 // 2 + 1),
            n_mels=80,
            sample_rate=self.sample_rate,
            f_min=0.0,
            f_max=8000.0,
            mel_scale="slaney",
            norm="slaney",
        )
        self._griffin_lim = GriffinLim(
            n_fft=1024,
            power=1,
            hop_length=256,
            win_length=1024,
        )

    @property
    def sample_rate(self):
        return self._sample_rate

    def forward(self, mel_spec, lengths=None):
        mel_spec = torch.exp(mel_spec)
        mel_spec = mel_spec.clone().detach().requires_grad_(True)
        spec = self._inv_mel(mel_spec)
        spec = spec.detach().requires_grad_(False)
        waveforms = self._griffin_lim(spec)
        return waveforms, lengths


################################################################################
# Bundle classes mixins
################################################################################


class _CharMixin:
    def get_text_processor(self) -> Tacotron2TTSBundle.TextProcessor:
        return _EnglishCharProcessor()


class _PhoneMixin:
    def get_text_processor(self, *, dl_kwargs=None) -> Tacotron2TTSBundle.TextProcessor:
        return _EnglishPhoneProcessor(dl_kwargs=dl_kwargs)


@dataclass
class _Tacotron2Mixin:
    _tacotron2_path: str
    _tacotron2_params: Dict[str, Any]

    def get_tacotron2(self, *, dl_kwargs=None) -> Tacotron2:
        model = Tacotron2(**self._tacotron2_params)
        url = f"{_BASE_URL}/{self._tacotron2_path}"
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        model.load_state_dict(state_dict)
        model.eval()
        return model


@dataclass
class _WaveRNNMixin:
    _wavernn_path: Optional[str]
    _wavernn_params: Optional[Dict[str, Any]]

    def get_vocoder(self, *, dl_kwargs=None):
        wavernn = self._get_wavernn(dl_kwargs=dl_kwargs)
        return _WaveRNNVocoder(wavernn)

    def _get_wavernn(self, *, dl_kwargs=None):
        model = WaveRNN(**self._wavernn_params)
        url = f"{_BASE_URL}/{self._wavernn_path}"
        dl_kwargs = {} if dl_kwargs is None else dl_kwargs
        state_dict = load_state_dict_from_url(url, **dl_kwargs)
        model.load_state_dict(state_dict)
        model.eval()
        return model


class _GriffinLimMixin:
    def get_vocoder(self, **_):
        return _GriffinLimVocoder()


################################################################################
# Bundle classes
################################################################################


@dataclass
class _Tacotron2WaveRNNCharBundle(_WaveRNNMixin, _Tacotron2Mixin, _CharMixin, Tacotron2TTSBundle):
    pass


@dataclass
class _Tacotron2WaveRNNPhoneBundle(_WaveRNNMixin, _Tacotron2Mixin, _PhoneMixin, Tacotron2TTSBundle):
    pass


@dataclass
class _Tacotron2GriffinLimCharBundle(_GriffinLimMixin, _Tacotron2Mixin, _CharMixin, Tacotron2TTSBundle):
    pass


@dataclass
class _Tacotron2GriffinLimPhoneBundle(_GriffinLimMixin, _Tacotron2Mixin, _PhoneMixin, Tacotron2TTSBundle):
    pass


################################################################################
# Instantiate bundle objects
################################################################################


TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH = _Tacotron2GriffinLimCharBundle(
    _tacotron2_path="tacotron2_english_characters_1500_epochs_ljspeech.pth",
    _tacotron2_params=utils._get_taco_params(n_symbols=38),
)
TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.__doc__ = """Character-based TTS pipeline with :py:class:`torchaudio.models.Tacotron2` and
:py:class:`torchaudio.transforms.GriffinLim`.

The text processor encodes the input texts character-by-character.

Tacotron2 was trained on *LJSpeech* [:footcite:`ljspeech17`] for 1,500 epochs.
You can find the training script `here <https://github.com/pytorch/audio/tree/main/examples/pipeline_tacotron2>`__.
The default parameters were used.

The vocoder is based on :py:class:`torchaudio.transforms.GriffinLim`.

Please refer to :func:`torchaudio.pipelines.Tacotron2TTSBundle` for the usage.

Example - "Hello world! T T S stands for Text to Speech!"

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.png
      :alt: Spectrogram generated by Tacotron2

   .. raw:: html

      <audio controls="controls">
         <source src="https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.wav" type="audio/wav">
         Your browser does not support the <code>audio</code> element.
      </audio>

Example - "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired,"

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH_v2.png
      :alt: Spectrogram generated by Tacotron2

   .. raw:: html

      <audio controls="controls">
         <source src="https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH_v2.wav" type="audio/wav">
         Your browser does not support the <code>audio</code> element.
      </audio>
"""  # noqa: E501

TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH = _Tacotron2GriffinLimPhoneBundle(
    _tacotron2_path="tacotron2_english_phonemes_1500_epochs_ljspeech.pth",
    _tacotron2_params=utils._get_taco_params(n_symbols=96),
)
TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH.__doc__ = """Phoneme-based TTS pipeline with :py:class:`torchaudio.models.Tacotron2` and
:py:class:`torchaudio.transforms.GriffinLim`.

The text processor encodes the input texts based on phoneme.
It uses `DeepPhonemizer <https://github.com/as-ideas/DeepPhonemizer>`__ to convert
graphemes to phonemes.
The model (*en_us_cmudict_forward*) was trained on
`CMUDict <http://www.speech.cs.cmu.edu/cgi-bin/cmudict>`__.

Tacotron2 was trained on *LJSpeech* [:footcite:`ljspeech17`] for 1,500 epochs.
You can find the training script `here <https://github.com/pytorch/audio/tree/main/examples/pipeline_tacotron2>`__.
The text processor is set to the *"english_phonemes"*.

The vocoder is based on :py:class:`torchaudio.transforms.GriffinLim`.

Please refer to :func:`torchaudio.pipelines.Tacotron2TTSBundle` for the usage.

Example - "Hello world! T T S stands for Text to Speech!"

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH.png
      :alt: Spectrogram generated by Tacotron2

   .. raw:: html

      <audio controls="controls">
         <source src="https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH.wav" type="audio/wav">
         Your browser does not support the <code>audio</code> element.
      </audio>

Example - "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired,"

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH_v2.png
      :alt: Spectrogram generated by Tacotron2

   .. raw:: html

      <audio controls="controls">
         <source src="https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH_v2.wav" type="audio/wav">
         Your browser does not support the <code>audio</code> element.
      </audio>

"""  # noqa: E501

TACOTRON2_WAVERNN_CHAR_LJSPEECH = _Tacotron2WaveRNNCharBundle(
    _tacotron2_path="tacotron2_english_characters_1500_epochs_wavernn_ljspeech.pth",
    _tacotron2_params=utils._get_taco_params(n_symbols=38),
    _wavernn_path="wavernn_10k_epochs_8bits_ljspeech.pth",
    _wavernn_params=utils._get_wrnn_params(),
)
TACOTRON2_WAVERNN_CHAR_LJSPEECH.__doc__ = """Character-based TTS pipeline with :py:class:`torchaudio.models.Tacotron2` and
:py:class:`torchaudio.models.WaveRNN`.

The text processor encodes the input texts character-by-character.

Tacotron2 was trained on *LJSpeech* [:footcite:`ljspeech17`] for 1,500 epochs.
You can find the training script `here <https://github.com/pytorch/audio/tree/main/examples/pipeline_tacotron2>`__.
The following parameters were used; ``win_length=1100``, ``hop_length=275``, ``n_fft=2048``,
``mel_fmin=40``, and ``mel_fmax=11025``.

The vocder is based on :py:class:`torchaudio.models.WaveRNN`.
It was trained on 8 bits depth waveform of *LJSpeech* [:footcite:`ljspeech17`] for 10,000 epochs.
You can find the training script `here <https://github.com/pytorch/audio/tree/main/examples/pipeline_wavernn>`__.

Please refer to :func:`torchaudio.pipelines.Tacotron2TTSBundle` for the usage.

Example - "Hello world! T T S stands for Text to Speech!"

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_WAVERNN_CHAR_LJSPEECH.png
      :alt: Spectrogram generated by Tacotron2

   .. raw:: html

      <audio controls="controls">
         <source src="https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_WAVERNN_CHAR_LJSPEECH.wav" type="audio/wav">
         Your browser does not support the <code>audio</code> element.
      </audio>

Example - "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired,"

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_WAVERNN_CHAR_LJSPEECH_v2.png
      :alt: Spectrogram generated by Tacotron2

   .. raw:: html

      <audio controls="controls">
         <source src="https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_WAVERNN_CHAR_LJSPEECH_v2.wav" type="audio/wav">
         Your browser does not support the <code>audio</code> element.
      </audio>
"""  # noqa: E501

TACOTRON2_WAVERNN_PHONE_LJSPEECH = _Tacotron2WaveRNNPhoneBundle(
    _tacotron2_path="tacotron2_english_phonemes_1500_epochs_wavernn_ljspeech.pth",
    _tacotron2_params=utils._get_taco_params(n_symbols=96),
    _wavernn_path="wavernn_10k_epochs_8bits_ljspeech.pth",
    _wavernn_params=utils._get_wrnn_params(),
)
TACOTRON2_WAVERNN_PHONE_LJSPEECH.__doc__ = """Phoneme-based TTS pipeline with :py:class:`torchaudio.models.Tacotron2` and
:py:class:`torchaudio.models.WaveRNN`.

The text processor encodes the input texts based on phoneme.
It uses `DeepPhonemizer <https://github.com/as-ideas/DeepPhonemizer>`__ to convert
graphemes to phonemes.
The model (*en_us_cmudict_forward*) was trained on
`CMUDict <http://www.speech.cs.cmu.edu/cgi-bin/cmudict>`__.

Tacotron2 was trained on *LJSpeech* [:footcite:`ljspeech17`] for 1,500 epochs.
You can find the training script `here <https://github.com/pytorch/audio/tree/main/examples/pipeline_tacotron2>`__.
The following parameters were used; ``win_length=1100``, ``hop_length=275``, ``n_fft=2048``,
``mel_fmin=40``, and ``mel_fmax=11025``.

The vocder is based on :py:class:`torchaudio.models.WaveRNN`.
It was trained on 8 bits depth waveform of *LJSpeech* [:footcite:`ljspeech17`] for 10,000 epochs.
You can find the training script `here <https://github.com/pytorch/audio/tree/main/examples/pipeline_wavernn>`__.

Please refer to :func:`torchaudio.pipelines.Tacotron2TTSBundle` for the usage.

Example - "Hello world! T T S stands for Text to Speech!"

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_WAVERNN_PHONE_LJSPEECH.png
      :alt: Spectrogram generated by Tacotron2

   .. raw:: html

      <audio controls="controls">
         <source src="https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_WAVERNN_PHONE_LJSPEECH.wav" type="audio/wav">
         Your browser does not support the <code>audio</code> element.
      </audio>


Example - "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired,"

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_WAVERNN_PHONE_LJSPEECH_v2.png
      :alt: Spectrogram generated by Tacotron2

   .. raw:: html

      <audio controls="controls">
         <source src="https://download.pytorch.org/torchaudio/doc-assets/TACOTRON2_WAVERNN_PHONE_LJSPEECH_v2.wav" type="audio/wav">
         Your browser does not support the <code>audio</code> element.
      </audio>
"""  # noqa: E501
