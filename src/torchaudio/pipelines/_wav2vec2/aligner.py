from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torchaudio.functional as F
from torch import Tensor
from torchaudio.functional import TokenSpan


class ITokenizer(ABC):
    @abstractmethod
    def __call__(self, transcript: List[str]) -> List[List[str]]:
        """Tokenize the given transcript (list of word)

        .. note::

           The toranscript must be normalized.

        Args:
            transcript (list of str): Transcript (list of word).

        Returns:
            (list of int): List of token sequences
        """


class Tokenizer(ITokenizer):
    def __init__(self, dictionary: Dict[str, int]):
        self.dictionary = dictionary

    def __call__(self, transcript: List[str]) -> List[List[int]]:
        return [[self.dictionary[c] for c in word] for word in transcript]


def _align_emission_and_tokens(emission: Tensor, tokens: List[int], blank: int = 0):
    device = emission.device
    emission = emission.unsqueeze(0)
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)

    aligned_tokens, scores = F.forced_align(emission, targets, blank=blank)

    scores = scores.exp()  # convert back to probability
    aligned_tokens, scores = aligned_tokens[0], scores[0]  # remove batch dimension
    return aligned_tokens, scores


class IAligner(ABC):
    @abstractmethod
    def __call__(self, emission: Tensor, tokens: List[List[int]]) -> List[List[TokenSpan]]:
        """Generate list of time-stamped token sequences

        Args:
            emission (Tensor): Sequence of token probability distributions in log-domain.
                Shape: `(time, tokens)`.
            tokens (list of integer sequence): Tokenized transcript.
                Output from :py:class:`torchaudio.pipelines.Wav2Vec2FABundle.Tokenizer`.

        Returns:
            (list of TokenSpan sequence): Tokens with time stamps and scores.
        """


def _unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def _flatten(nested_list):
    return [item for list_ in nested_list for item in list_]


class Aligner(IAligner):
    def __init__(self, blank):
        self.blank = blank

    def __call__(self, emission: Tensor, tokens: List[List[int]]) -> List[List[TokenSpan]]:
        if emission.ndim != 2:
            raise ValueError(f"The input emission must be 2D. Found: {emission.shape}")

        aligned_tokens, scores = _align_emission_and_tokens(emission, _flatten(tokens), self.blank)
        spans = F.merge_tokens(aligned_tokens, scores)
        return _unflatten(spans, [len(ts) for ts in tokens])
