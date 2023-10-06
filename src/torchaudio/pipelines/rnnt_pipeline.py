import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple

import torch
import torchaudio
from torchaudio._internal import module_utils
from torchaudio.models import emformer_rnnt_base, RNNT, RNNTBeamSearch


__all__ = []

_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)


def _piecewise_linear_log(x):
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class _FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class _GlobalStatsNormalization(torch.nn.Module):
    def __init__(self, global_stats_path):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.register_buffer("mean", torch.tensor(blob["mean"]))
        self.register_buffer("invstddev", torch.tensor(blob["invstddev"]))

    def forward(self, input):
        return (input - self.mean) * self.invstddev


class _FeatureExtractor(ABC):
    @abstractmethod
    def __call__(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates features and length output from the given input tensor.

        Args:
            input (torch.Tensor): input tensor.

        Returns:
            (torch.Tensor, torch.Tensor):
            torch.Tensor:
                Features, with shape `(length, *)`.
            torch.Tensor:
                Length, with shape `(1,)`.
        """


class _TokenProcessor(ABC):
    @abstractmethod
    def __call__(self, tokens: List[int], **kwargs) -> str:
        """Decodes given list of tokens to text sequence.

        Args:
            tokens (List[int]): list of tokens to decode.

        Returns:
            str:
                Decoded text sequence.
        """


class _ModuleFeatureExtractor(torch.nn.Module, _FeatureExtractor):
    """``torch.nn.Module``-based feature extraction pipeline.

    Args:
        pipeline (torch.nn.Module): module that implements feature extraction logic.
    """

    def __init__(self, pipeline: torch.nn.Module) -> None:
        super().__init__()
        self.pipeline = pipeline

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates features and length output from the given input tensor.

        Args:
            input (torch.Tensor): input tensor.

        Returns:
            (torch.Tensor, torch.Tensor):
            torch.Tensor:
                Features, with shape `(length, *)`.
            torch.Tensor:
                Length, with shape `(1,)`.
        """
        features = self.pipeline(input)
        length = torch.tensor([features.shape[0]])
        return features, length


class _SentencePieceTokenProcessor(_TokenProcessor):
    """SentencePiece-model-based token processor.

    Args:
        sp_model_path (str): path to SentencePiece model.
    """

    def __init__(self, sp_model_path: str) -> None:
        if not module_utils.is_module_available("sentencepiece"):
            raise RuntimeError("SentencePiece is not available. Please install it.")

        import sentencepiece as spm

        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.post_process_remove_list = {
            self.sp_model.unk_id(),
            self.sp_model.eos_id(),
            self.sp_model.pad_id(),
        }

    def __call__(self, tokens: List[int], lstrip: bool = True) -> str:
        """Decodes given list of tokens to text sequence.

        Args:
            tokens (List[int]): list of tokens to decode.
            lstrip (bool, optional): if ``True``, returns text sequence with leading whitespace
                removed. (Default: ``True``).

        Returns:
            str:
                Decoded text sequence.
        """
        filtered_hypo_tokens = [
            token_index for token_index in tokens[1:] if token_index not in self.post_process_remove_list
        ]
        output_string = "".join(self.sp_model.id_to_piece(filtered_hypo_tokens)).replace("\u2581", " ")

        if lstrip:
            return output_string.lstrip()
        else:
            return output_string


@dataclass
class RNNTBundle:
    """Dataclass that bundles components for performing automatic speech recognition (ASR, speech-to-text)
    inference with an RNN-T model.

    More specifically, the class provides methods that produce the featurization pipeline,
    decoder wrapping the specified RNN-T model, and output token post-processor that together
    constitute a complete end-to-end ASR inference pipeline that produces a text sequence
    given a raw waveform.

    It can support non-streaming (full-context) inference as well as streaming inference.

    Users should not directly instantiate objects of this class; rather, users should use the
    instances (representing pre-trained models) that exist within the module,
    e.g. :data:`torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH`.

    Example
        >>> import torchaudio
        >>> from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH
        >>> import torch
        >>>
        >>> # Non-streaming inference.
        >>> # Build feature extractor, decoder with RNN-T model, and token processor.
        >>> feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_feature_extractor()
        100%|███████████████████████████████| 3.81k/3.81k [00:00<00:00, 4.22MB/s]
        >>> decoder = EMFORMER_RNNT_BASE_LIBRISPEECH.get_decoder()
        Downloading: "https://download.pytorch.org/torchaudio/models/emformer_rnnt_base_librispeech.pt"
        100%|███████████████████████████████| 293M/293M [00:07<00:00, 42.1MB/s]
        >>> token_processor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_token_processor()
        100%|███████████████████████████████| 295k/295k [00:00<00:00, 25.4MB/s]
        >>>
        >>> # Instantiate LibriSpeech dataset; retrieve waveform for first sample.
        >>> dataset = torchaudio.datasets.LIBRISPEECH("/home/librispeech", url="test-clean")
        >>> waveform = next(iter(dataset))[0].squeeze()
        >>>
        >>> with torch.no_grad():
        >>>     # Produce mel-scale spectrogram features.
        >>>     features, length = feature_extractor(waveform)
        >>>
        >>>     # Generate top-10 hypotheses.
        >>>     hypotheses = decoder(features, length, 10)
        >>>
        >>> # For top hypothesis, convert predicted tokens to text.
        >>> text = token_processor(hypotheses[0][0])
        >>> print(text)
        he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to [...]
        >>>
        >>>
        >>> # Streaming inference.
        >>> hop_length = EMFORMER_RNNT_BASE_LIBRISPEECH.hop_length
        >>> num_samples_segment = EMFORMER_RNNT_BASE_LIBRISPEECH.segment_length * hop_length
        >>> num_samples_segment_right_context = (
        >>>     num_samples_segment + EMFORMER_RNNT_BASE_LIBRISPEECH.right_context_length * hop_length
        >>> )
        >>>
        >>> # Build streaming inference feature extractor.
        >>> streaming_feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_streaming_feature_extractor()
        >>>
        >>> # Process same waveform as before, this time sequentially across overlapping segments
        >>> # to simulate streaming inference. Note the usage of ``streaming_feature_extractor`` and ``decoder.infer``.
        >>> state, hypothesis = None, None
        >>> for idx in range(0, len(waveform), num_samples_segment):
        >>>     segment = waveform[idx: idx + num_samples_segment_right_context]
        >>>     segment = torch.nn.functional.pad(segment, (0, num_samples_segment_right_context - len(segment)))
        >>>     with torch.no_grad():
        >>>         features, length = streaming_feature_extractor(segment)
        >>>         hypotheses, state = decoder.infer(features, length, 10, state=state, hypothesis=hypothesis)
        >>>     hypothesis = hypotheses[0]
        >>>     transcript = token_processor(hypothesis[0])
        >>>     if transcript:
        >>>         print(transcript, end=" ", flush=True)
        he hoped there would be stew for dinner turn ips and car rots and bru 'd oes and fat mut ton pieces to [...]
    """

    class FeatureExtractor(_FeatureExtractor):
        """Interface of the feature extraction part of RNN-T pipeline"""

    class TokenProcessor(_TokenProcessor):
        """Interface of the token processor part of RNN-T pipeline"""

    _rnnt_path: str
    _rnnt_factory_func: Callable[[], RNNT]
    _global_stats_path: str
    _sp_model_path: str
    _right_padding: int
    _blank: int
    _sample_rate: int
    _n_fft: int
    _n_mels: int
    _hop_length: int
    _segment_length: int
    _right_context_length: int

    def _get_model(self) -> RNNT:
        model = self._rnnt_factory_func()
        path = torchaudio.utils.download_asset(self._rnnt_path)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @property
    def sample_rate(self) -> int:
        """Sample rate (in cycles per second) of input waveforms.

        :type: int
        """
        return self._sample_rate

    @property
    def n_fft(self) -> int:
        """Size of FFT window to use.

        :type: int
        """
        return self._n_fft

    @property
    def n_mels(self) -> int:
        """Number of mel spectrogram features to extract from input waveforms.

        :type: int
        """
        return self._n_mels

    @property
    def hop_length(self) -> int:
        """Number of samples between successive frames in input expected by model.

        :type: int
        """
        return self._hop_length

    @property
    def segment_length(self) -> int:
        """Number of frames in segment in input expected by model.

        :type: int
        """
        return self._segment_length

    @property
    def right_context_length(self) -> int:
        """Number of frames in right contextual block in input expected by model.

        :type: int
        """
        return self._right_context_length

    def get_decoder(self) -> RNNTBeamSearch:
        """Constructs RNN-T decoder.

        Returns:
            RNNTBeamSearch
        """
        model = self._get_model()
        return RNNTBeamSearch(model, self._blank)

    def get_feature_extractor(self) -> FeatureExtractor:
        """Constructs feature extractor for non-streaming (full-context) ASR.

        Returns:
            FeatureExtractor
        """
        local_path = torchaudio.utils.download_asset(self._global_stats_path)
        return _ModuleFeatureExtractor(
            torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length
                ),
                _FunctionalModule(lambda x: x.transpose(1, 0)),
                _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
                _GlobalStatsNormalization(local_path),
                _FunctionalModule(lambda x: torch.nn.functional.pad(x, (0, 0, 0, self._right_padding))),
            )
        )

    def get_streaming_feature_extractor(self) -> FeatureExtractor:
        """Constructs feature extractor for streaming (simultaneous) ASR.

        Returns:
            FeatureExtractor
        """
        local_path = torchaudio.utils.download_asset(self._global_stats_path)
        return _ModuleFeatureExtractor(
            torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length
                ),
                _FunctionalModule(lambda x: x.transpose(1, 0)),
                _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)),
                _GlobalStatsNormalization(local_path),
            )
        )

    def get_token_processor(self) -> TokenProcessor:
        """Constructs token processor.

        Returns:
            TokenProcessor
        """
        local_path = torchaudio.utils.download_asset(self._sp_model_path)
        return _SentencePieceTokenProcessor(local_path)


EMFORMER_RNNT_BASE_LIBRISPEECH = RNNTBundle(
    _rnnt_path="models/emformer_rnnt_base_librispeech.pt",
    _rnnt_factory_func=partial(emformer_rnnt_base, num_symbols=4097),
    _global_stats_path="pipeline-assets/global_stats_rnnt_librispeech.json",
    _sp_model_path="pipeline-assets/spm_bpe_4096_librispeech.model",
    _right_padding=4,
    _blank=4096,
    _sample_rate=16000,
    _n_fft=400,
    _n_mels=80,
    _hop_length=160,
    _segment_length=16,
    _right_context_length=4,
)
EMFORMER_RNNT_BASE_LIBRISPEECH.__doc__ = """ASR pipeline based on Emformer-RNNT,
pretrained on *LibriSpeech* dataset :cite:`7178964`,
capable of performing both streaming and non-streaming inference.

The underlying model is constructed by :py:func:`torchaudio.models.emformer_rnnt_base`
and utilizes weights trained on LibriSpeech using training script ``train.py``
`here <https://github.com/pytorch/audio/tree/main/examples/asr/emformer_rnnt>`__ with default arguments.

Please refer to :py:class:`RNNTBundle` for usage instructions.
"""
