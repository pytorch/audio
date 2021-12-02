from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional

from torch import Tensor
from torchaudio.models import Tacotron2


class _TextProcessor(ABC):
    @property
    @abstractmethod
    def tokens(self):
        """The tokens that the each value in the processed tensor represent.

        See :func:`torchaudio.pipelines.Tacotron2TTSBundle.get_text_processor` for the usage.

        :type: List[str]
        """

    @abstractmethod
    def __call__(self, texts: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
        """Encode the given (batch of) texts into numerical tensors

        See :func:`torchaudio.pipelines.Tacotron2TTSBundle.get_text_processor` for the usage.

        Args:
            text (str or list of str): The input texts.

        Returns:
            (Tensor, Tensor):
            Tensor:
                The encoded texts. Shape: `(batch, max length)`
            Tensor:
                The valid length of each sample in the batch. Shape: `(batch, )`.
        """


class _Vocoder(ABC):
    @property
    @abstractmethod
    def sample_rate(self):
        """The sample rate of the resulting waveform

        See :func:`torchaudio.pipelines.Tacotron2TTSBundle.get_vocoder` for the usage.

        :type: float
        """

    @abstractmethod
    def __call__(self, specgrams: Tensor, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Generate waveform from the given input, such as spectrogram

        See :func:`torchaudio.pipelines.Tacotron2TTSBundle.get_vocoder` for the usage.

        Args:
            specgrams (Tensor):
                The input spectrogram. Shape: `(batch, frequency bins, time)`.
                The expected shape depends on the implementation.
            lengths (Tensor, or None, optional):
                The valid length of each sample in the batch. Shape: `(batch, )`.
                (Default: `None`)

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor:
                The generated waveform. Shape: `(batch, max length)`
            Tensor or None:
                The valid length of each sample in the batch. Shape: `(batch, )`.
        """


class Tacotron2TTSBundle(ABC):
    """Data class that bundles associated information to use pretrained Tacotron2 and vocoder.

    This class provides interfaces for instantiating the pretrained model along with
    the information necessary to retrieve pretrained weights and additional data
    to be used with the model.

    Torchaudio library instantiates objects of this class, each of which represents
    a different pretrained model. Client code should access pretrained models via these
    instances.

    Please see below for the usage and the available values.

    Example - Character-based TTS pipeline with Tacotron2 and WaveRNN
        >>> import torchaudio
        >>>
        >>> text = "Hello, T T S !"
        >>> bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        >>>
        >>> # Build processor, Tacotron2 and WaveRNN model
        >>> processor = bundle.get_text_processor()
        >>> tacotron2 = bundle.get_tacotron2()
        Downloading:
        100%|███████████████████████████████| 107M/107M [00:01<00:00, 87.9MB/s]
        >>> vocoder = bundle.get_vocoder()
        Downloading:
        100%|███████████████████████████████| 16.7M/16.7M [00:00<00:00, 78.1MB/s]
        >>>
        >>> # Encode text
        >>> input, lengths = processor(text)
        >>>
        >>> # Generate (mel-scale) spectrogram
        >>> specgram, lengths, _ = tacotron2.infer(input, lengths)
        >>>
        >>> # Convert spectrogram to waveform
        >>> waveforms, lengths = vocoder(specgram, lengths)
        >>>
        >>> torchaudio.save('hello-tts.wav', waveforms, vocoder.sample_rate)

    Example - Phoneme-based TTS pipeline with Tacotron2 and WaveRNN
        >>>
        >>> # Note:
        >>> #     This bundle uses pre-trained DeepPhonemizer as
        >>> #     the text pre-processor.
        >>> #     Please install deep-phonemizer.
        >>> #     See https://github.com/as-ideas/DeepPhonemizer
        >>> #     The pretrained weight is automatically downloaded.
        >>>
        >>> import torchaudio
        >>>
        >>> text = "Hello, TTS!"
        >>> bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        >>>
        >>> # Build processor, Tacotron2 and WaveRNN model
        >>> processor = bundle.get_text_processor()
        Downloading:
        100%|███████████████████████████████| 63.6M/63.6M [00:04<00:00, 15.3MB/s]
        >>> tacotron2 = bundle.get_tacotron2()
        Downloading:
        100%|███████████████████████████████| 107M/107M [00:01<00:00, 87.9MB/s]
        >>> vocoder = bundle.get_vocoder()
        Downloading:
        100%|███████████████████████████████| 16.7M/16.7M [00:00<00:00, 78.1MB/s]
        >>>
        >>> # Encode text
        >>> input, lengths = processor(text)
        >>>
        >>> # Generate (mel-scale) spectrogram
        >>> specgram, lengths, _ = tacotron2.infer(input, lengths)
        >>>
        >>> # Convert spectrogram to waveform
        >>> waveforms, lengths = vocoder(specgram, lengths)
        >>>
        >>> torchaudio.save('hello-tts.wav', waveforms, vocoder.sample_rate)
    """

    # Using the inner class so that these interfaces are not directly exposed on
    # `torchaudio.pipelines`, but still listed in documentation.
    # The thing is, text processing and vocoder are generic and we do not know what kind of
    # new text processing and vocoder will be added in the future, so we want to make these
    # interfaces specific to this Tacotron2TTS pipeline.
    class TextProcessor(_TextProcessor):
        """Interface of the text processing part of Tacotron2TTS pipeline

        See :func:`torchaudio.pipelines.Tacotron2TTSBundle.get_text_processor` for the usage.
        """

    class Vocoder(_Vocoder):
        """Interface of the vocoder part of Tacotron2TTS pipeline

        See :func:`torchaudio.pipelines.Tacotron2TTSBundle.get_vocoder` for the usage.
        """

    @abstractmethod
    def get_text_processor(self, *, dl_kwargs=None) -> TextProcessor:
        # Overriding the signature so that the return type is correct on Sphinx
        """get_text_processor(self, *, dl_kwargs=None) -> torchaudio.pipelines.Tacotron2TTSBundle.TextProcessor

        Create a text processor

        For character-based pipeline, this processor splits the input text by character.
        For phoneme-based pipeline, this processor converts the input text (grapheme) to
        phonemes.

        If a pre-trained weight file is necessary,
        :func:`torch.hub.download_url_to_file` is used to downloaded it.

        Args:
            dl_kwargs (dictionary of keyword arguments,):
                Passed to :func:`torch.hub.download_url_to_file`.

        Returns:
            TTSTextProcessor:
                A callable which takes a string or a list of strings as input and
                returns Tensor of encoded texts and Tensor of valid lengths.
                The object also has ``tokens`` property, which allows to recover the
                tokenized form.

        Example - Character-based
            >>> text = [
            >>>     "Hello World!",
            >>>     "Text-to-speech!",
            >>> ]
            >>> bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
            >>> processor = bundle.get_text_processor()
            >>> input, lengths = processor(text)
            >>>
            >>> print(input)
            tensor([[19, 16, 23, 23, 26, 11, 34, 26, 29, 23, 15,  2,  0,  0,  0],
                    [31, 16, 35, 31,  1, 31, 26,  1, 30, 27, 16, 16, 14, 19,  2]],
                   dtype=torch.int32)
            >>>
            >>> print(lengths)
            tensor([12, 15], dtype=torch.int32)
            >>>
            >>> print([processor.tokens[i] for i in input[0, :lengths[0]]])
            ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!']
            >>> print([processor.tokens[i] for i in input[1, :lengths[1]]])
            ['t', 'e', 'x', 't', '-', 't', 'o', '-', 's', 'p', 'e', 'e', 'c', 'h', '!']

        Example - Phoneme-based
            >>> text = [
            >>>     "Hello, T T S !",
            >>>     "Text-to-speech!",
            >>> ]
            >>> bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
            >>> processor = bundle.get_text_processor()
            Downloading:
            100%|███████████████████████████████| 63.6M/63.6M [00:04<00:00, 15.3MB/s]
            >>> input, lengths = processor(text)
            >>>
            >>> print(input)
            tensor([[54, 20, 65, 69, 11, 92, 44, 65, 38,  2,  0,  0,  0,  0],
                    [81, 40, 64, 79, 81,  1, 81, 20,  1, 79, 77, 59, 37,  2]],
                   dtype=torch.int32)
            >>>
            >>> print(lengths)
            tensor([10, 14], dtype=torch.int32)
            >>>
            >>> print([processor.tokens[i] for i in input[0]])
            ['HH', 'AH', 'L', 'OW', ' ', 'W', 'ER', 'L', 'D', '!', '_', '_', '_', '_']
            >>> print([processor.tokens[i] for i in input[1]])
            ['T', 'EH', 'K', 'S', 'T', '-', 'T', 'AH', '-', 'S', 'P', 'IY', 'CH', '!']
        """

    @abstractmethod
    def get_vocoder(self, *, dl_kwargs=None) -> Vocoder:
        # Overriding the signature so that the return type is correct on Sphinx
        """get_vocoder(self, *, dl_kwargs=None) -> torchaudio.pipelines.Tacotron2TTSBundle.Vocoder

        Create a vocoder module, based off of either WaveRNN or GriffinLim.

        If a pre-trained weight file is necessary,
        :func:`torch.hub.load_state_dict_from_url` is used to downloaded it.

        Args:
            dl_kwargs (dictionary of keyword arguments):
                Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Callable[[Tensor, Optional[Tensor]], Tuple[Tensor, Optional[Tensor]]]:
                A vocoder module, which takes spectrogram Tensor and an optional
                length Tensor, then returns resulting waveform Tensor and an optional
                length Tensor.
        """

    @abstractmethod
    def get_tacotron2(self, *, dl_kwargs=None) -> Tacotron2:
        # Overriding the signature so that the return type is correct on Sphinx
        """get_tacotron2(self, *, dl_kwargs=None) -> torchaudio.models.Tacotron2

        Create a Tacotron2 model with pre-trained weight.

        Args:
            dl_kwargs (dictionary of keyword arguments):
                Passed to :func:`torch.hub.load_state_dict_from_url`.

        Returns:
            Tacotron2:
                The resulting model.
        """
