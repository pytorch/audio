from __future__ import annotations

import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Union

import torch
import torchaudio
from torchaudio.utils import download_asset

try:
    # We prioritize the version from upstream flashlight here.
    # This will allow applications that use the upstream flashlight
    # alongside torchaudio.
    from flashlight.lib.text.decoder import (
        CriterionType as _CriterionType,
        KenLM as _KenLM,
        LexiconDecoder as _LexiconDecoder,
        LexiconDecoderOptions as _LexiconDecoderOptions,
        LexiconFreeDecoder as _LexiconFreeDecoder,
        LexiconFreeDecoderOptions as _LexiconFreeDecoderOptions,
        LM as _LM,
        LMState as _LMState,
        SmearingMode as _SmearingMode,
        Trie as _Trie,
        ZeroLM as _ZeroLM,
    )
    from flashlight.lib.text.dictionary import (
        create_word_dict as _create_word_dict,
        Dictionary as _Dictionary,
        load_words as _load_words,
    )
except Exception:
    torchaudio._extension._load_lib("libflashlight-text")
    from torchaudio.flashlight_lib_text_decoder import (
        CriterionType as _CriterionType,
        KenLM as _KenLM,
        LexiconDecoder as _LexiconDecoder,
        LexiconDecoderOptions as _LexiconDecoderOptions,
        LexiconFreeDecoder as _LexiconFreeDecoder,
        LexiconFreeDecoderOptions as _LexiconFreeDecoderOptions,
        LM as _LM,
        LMState as _LMState,
        SmearingMode as _SmearingMode,
        Trie as _Trie,
        ZeroLM as _ZeroLM,
    )
    from torchaudio.flashlight_lib_text_dictionary import (
        create_word_dict as _create_word_dict,
        Dictionary as _Dictionary,
        load_words as _load_words,
    )


__all__ = [
    "CTCHypothesis",
    "CTCDecoder",
    "CTCDecoderLM",
    "CTCDecoderLMState",
    "ctc_decoder",
    "download_pretrained_files",
]

_PretrainedFiles = namedtuple("PretrainedFiles", ["lexicon", "tokens", "lm"])


def _construct_trie(tokens_dict, word_dict, lexicon, lm, silence):
    vocab_size = tokens_dict.index_size()
    trie = _Trie(vocab_size, silence)
    start_state = lm.start(False)

    for word, spellings in lexicon.items():
        word_idx = word_dict.get_index(word)
        _, score = lm.score(start_state, word_idx)
        for spelling in spellings:
            spelling_idx = [tokens_dict.get_index(token) for token in spelling]
            trie.insert(spelling_idx, word_idx, score)
    trie.smear(_SmearingMode.MAX)
    return trie


def _get_word_dict(lexicon, lm, lm_dict, tokens_dict, unk_word):
    word_dict = None
    if lm_dict is not None:
        word_dict = _Dictionary(lm_dict)

    if lexicon and word_dict is None:
        word_dict = _create_word_dict(lexicon)
    elif not lexicon and word_dict is None and type(lm) == str:
        d = {tokens_dict.get_entry(i): [[tokens_dict.get_entry(i)]] for i in range(tokens_dict.index_size())}
        d[unk_word] = [[unk_word]]
        word_dict = _create_word_dict(d)

    return word_dict


class CTCHypothesis(NamedTuple):
    r"""Represents hypothesis generated by CTC beam search decoder :py:func:`CTCDecoder`.

    Note:
        The ``words`` field is only applicable if a lexicon is provided to the decoder. If
        decoding without a lexicon, it will be blank. Please refer to ``tokens`` and
        :py:func:`idxs_to_tokens <torchaudio.models.decoder.CTCDecoder.idxs_to_tokens>` instead.

    :ivar torch.LongTensor tokens: Predicted sequence of token IDs. Shape `(L, )`, where
        `L` is the length of the output sequence
    :ivar List[str] words: List of predicted words
    :ivar float score: Score corresponding to hypothesis
    :ivar torch.IntTensor timesteps: Timesteps corresponding to the tokens. Shape `(L, )`,
        where `L` is the length of the output sequence
    """
    tokens: torch.LongTensor
    words: List[str]
    score: float
    timesteps: torch.IntTensor


class CTCDecoderLMState(_LMState):
    """Language model state.

    :ivar Dict[int] children: Map of indices to LM states
    """

    def child(self, usr_index: int):
        """Returns child corresponding to usr_index, or creates and returns a new state if input index
        is not found.

        Args:
            usr_index (int): index corresponding to child state

        Returns:
            CTCDecoderLMState: child state corresponding to usr_index
        """
        return super().child(usr_index)

    def compare(self, state: CTCDecoderLMState):
        """Compare two language model states.

        Args:
            state (CTCDecoderLMState): LM state to compare against

        Returns:
            int: 0 if the states are the same, -1 if self is less, +1 if self is greater.
        """
        pass


class CTCDecoderLM(_LM):
    """Language model base class for creating custom language models to use with the decoder."""

    @abstractmethod
    def start(self, start_with_nothing: bool):
        """Initialize or reset the language model.

        Args:
            start_with_nothing (bool): whether or not to start sentence with sil token.

        Returns:
            CTCDecoderLMState: starting state
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, state: CTCDecoderLMState, usr_token_idx: int):
        """Evaluate the language model based on the current LM state and new word.

        Args:
            state (CTCDecoderLMState): current LM state
            usr_token_idx (int): index of the word

        Returns:
            Tuple[CTCDecoderLMState, float]
                CTCDecoderLMState: new LM state
                float: score
        """
        raise NotImplementedError

    @abstractmethod
    def finish(self, state: CTCDecoderLMState):
        """Evaluate end for language model based on current LM state.

        Args:
            state (CTCDecoderLMState): current LM state

        Returns:
            (CTCDecoderLMState, float)
                CTCDecoderLMState:
                    new LM state
                float:
                    score
        """
        raise NotImplementedError


class CTCDecoder:
    """
    .. devices:: CPU

    CTC beam search decoder from *Flashlight* [:footcite:`kahn2022flashlight`].

    Note:
        To build the decoder, please use the factory function :py:func:`ctc_decoder`.

    Args:
        nbest (int): number of best decodings to return
        lexicon (Dict or None): lexicon mapping of words to spellings, or None for lexicon-free decoder
        word_dict (_Dictionary): dictionary of words
        tokens_dict (_Dictionary): dictionary of tokens
        lm (CTCDecoderLM): language model. If using a lexicon, only word level LMs are currently supported
        decoder_options (_LexiconDecoderOptions or _LexiconFreeDecoderOptions): parameters used for beam search decoding
        blank_token (str): token corresopnding to blank
        sil_token (str): token corresponding to silence
        unk_word (str): word corresponding to unknown
    """

    def __init__(
        self,
        nbest: int,
        lexicon: Optional[Dict],
        word_dict: _Dictionary,
        tokens_dict: _Dictionary,
        lm: CTCDecoderLM,
        decoder_options: Union[_LexiconDecoderOptions, _LexiconFreeDecoderOptions],
        blank_token: str,
        sil_token: str,
        unk_word: str,
    ) -> None:
        self.nbest = nbest
        self.word_dict = word_dict
        self.tokens_dict = tokens_dict
        self.blank = self.tokens_dict.get_index(blank_token)
        silence = self.tokens_dict.get_index(sil_token)
        transitions = []

        if lexicon:
            trie = _construct_trie(tokens_dict, word_dict, lexicon, lm, silence)
            unk_word = word_dict.get_index(unk_word)
            token_lm = False  # use word level LM

            self.decoder = _LexiconDecoder(
                decoder_options,
                trie,
                lm,
                silence,
                self.blank,
                unk_word,
                transitions,
                token_lm,
            )
        else:
            self.decoder = _LexiconFreeDecoder(decoder_options, lm, silence, self.blank, transitions)

    def _get_tokens(self, idxs: torch.IntTensor) -> torch.LongTensor:
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def _get_timesteps(self, idxs: torch.IntTensor) -> torch.IntTensor:
        """Returns frame numbers corresponding to non-blank tokens."""

        timesteps = []
        for i, idx in enumerate(idxs):
            if idx == self.blank:
                continue
            if i == 0 or idx != idxs[i - 1]:
                timesteps.append(i)
        return torch.IntTensor(timesteps)

    def __call__(
        self, emissions: torch.FloatTensor, lengths: Optional[torch.Tensor] = None
    ) -> List[List[CTCHypothesis]]:
        """
        Args:
            emissions (torch.FloatTensor): CPU tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model.
            lengths (Tensor or None, optional): CPU tensor of shape `(batch, )` storing the valid length of
                in time axis of the output Tensor in each batch.

        Returns:
            List[List[CTCHypothesis]]:
                List of sorted best hypotheses for each audio sequence in the batch.
        """

        if emissions.dtype != torch.float32:
            raise ValueError("emissions must be float32.")

        if emissions.is_cuda:
            raise RuntimeError("emissions must be a CPU tensor.")

        if lengths is not None and lengths.is_cuda:
            raise RuntimeError("lengths must be a CPU tensor.")

        B, T, N = emissions.size()
        if lengths is None:
            lengths = torch.full((B,), T)

        float_bytes = 4
        hypos = []

        for b in range(B):
            emissions_ptr = emissions.data_ptr() + float_bytes * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, lengths[b], N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    CTCHypothesis(
                        tokens=self._get_tokens(result.tokens),
                        words=[self.word_dict.get_entry(x) for x in result.words if x >= 0],
                        score=result.score,
                        timesteps=self._get_timesteps(result.tokens),
                    )
                    for result in nbest_results
                ]
            )

        return hypos

    def idxs_to_tokens(self, idxs: torch.LongTensor) -> List:
        """
        Map raw token IDs into corresponding tokens

        Args:
            idxs (LongTensor): raw token IDs generated from decoder

        Returns:
            List: tokens corresponding to the input IDs
        """
        return [self.tokens_dict.get_entry(idx.item()) for idx in idxs]


def ctc_decoder(
    lexicon: Optional[str],
    tokens: Union[str, List[str]],
    lm: Union[str, CTCDecoderLM] = None,
    lm_dict: Optional[str] = None,
    nbest: int = 1,
    beam_size: int = 50,
    beam_size_token: Optional[int] = None,
    beam_threshold: float = 50,
    lm_weight: float = 2,
    word_score: float = 0,
    unk_score: float = float("-inf"),
    sil_score: float = 0,
    log_add: bool = False,
    blank_token: str = "-",
    sil_token: str = "|",
    unk_word: str = "<unk>",
) -> CTCDecoder:
    """
    Builds CTC beam search decoder from *Flashlight* [:footcite:`kahn2022flashlight`].

    Args:
        lexicon (str or None): lexicon file containing the possible words and corresponding spellings.
            Each line consists of a word and its space separated spelling. If `None`, uses lexicon-free
            decoding.
        tokens (str or List[str]): file or list containing valid tokens. If using a file, the expected
            format is for tokens mapping to the same index to be on the same line
        lm (str, CTCDecoderLM, or None, optional): either a path containing KenLM language model,
            custom language model of type `CTCDecoderLM`, or `None` if not using a language model
        lm_dict (str or None, optional): file consisting of the dictionary used for the LM, with a word
            per line sorted by LM index. If decoding with a lexicon, entries in lm_dict must also occur
            in the lexicon file. If `None`, dictionary for LM is constructed using the lexicon file.
            (Default: None)
        nbest (int, optional): number of best decodings to return (Default: 1)
        beam_size (int, optional): max number of hypos to hold after each decode step (Default: 50)
        beam_size_token (int, optional): max number of tokens to consider at each decode step.
            If `None`, it is set to the total number of tokens (Default: None)
        beam_threshold (float, optional): threshold for pruning hypothesis (Default: 50)
        lm_weight (float, optional): weight of language model (Default: 2)
        word_score (float, optional): word insertion score (Default: 0)
        unk_score (float, optional): unknown word insertion score (Default: -inf)
        sil_score (float, optional): silence insertion score (Default: 0)
        log_add (bool, optional): whether or not to use logadd when merging hypotheses (Default: False)
        blank_token (str, optional): token corresponding to blank (Default: "-")
        sil_token (str, optional): token corresponding to silence (Default: "|")
        unk_word (str, optional): word corresponding to unknown (Default: "<unk>")

    Returns:
        CTCDecoder: decoder

    Example
        >>> decoder = ctc_decoder(
        >>>     lexicon="lexicon.txt",
        >>>     tokens="tokens.txt",
        >>>     lm="kenlm.bin",
        >>> )
        >>> results = decoder(emissions) # List of shape (B, nbest) of Hypotheses
    """
    if lm_dict is not None and type(lm_dict) is not str:
        raise ValueError("lm_dict must be None or str type.")

    tokens_dict = _Dictionary(tokens)

    # decoder options
    if lexicon:
        lexicon = _load_words(lexicon)
        decoder_options = _LexiconDecoderOptions(
            beam_size=beam_size,
            beam_size_token=beam_size_token or tokens_dict.index_size(),
            beam_threshold=beam_threshold,
            lm_weight=lm_weight,
            word_score=word_score,
            unk_score=unk_score,
            sil_score=sil_score,
            log_add=log_add,
            criterion_type=_CriterionType.CTC,
        )
    else:
        decoder_options = _LexiconFreeDecoderOptions(
            beam_size=beam_size,
            beam_size_token=beam_size_token or tokens_dict.index_size(),
            beam_threshold=beam_threshold,
            lm_weight=lm_weight,
            sil_score=sil_score,
            log_add=log_add,
            criterion_type=_CriterionType.CTC,
        )

    # construct word dict and language model
    word_dict = _get_word_dict(lexicon, lm, lm_dict, tokens_dict, unk_word)

    if type(lm) == str:
        lm = _KenLM(lm, word_dict)
    elif lm is None:
        lm = _ZeroLM()

    return CTCDecoder(
        nbest=nbest,
        lexicon=lexicon,
        word_dict=word_dict,
        tokens_dict=tokens_dict,
        lm=lm,
        decoder_options=decoder_options,
        blank_token=blank_token,
        sil_token=sil_token,
        unk_word=unk_word,
    )


def _get_filenames(model: str) -> _PretrainedFiles:
    if model not in ["librispeech", "librispeech-3-gram", "librispeech-4-gram"]:
        raise ValueError(
            f"{model} not supported. Must be one of ['librispeech-3-gram', 'librispeech-4-gram', 'librispeech']"
        )

    prefix = f"decoder-assets/{model}"
    return _PretrainedFiles(
        lexicon=f"{prefix}/lexicon.txt",
        tokens=f"{prefix}/tokens.txt",
        lm=f"{prefix}/lm.bin" if model != "librispeech" else None,
    )


def download_pretrained_files(model: str) -> _PretrainedFiles:
    """
    Retrieves pretrained data files used for CTC decoder.

    Args:
        model (str): pretrained language model to download.
            Options: ["librispeech-3-gram", "librispeech-4-gram", "librispeech"]

    Returns:
        Object with the following attributes
            lm:
                path corresponding to downloaded language model, or `None` if the model is not associated with an lm
            lexicon:
                path corresponding to downloaded lexicon file
            tokens:
                path corresponding to downloaded tokens file
    """

    files = _get_filenames(model)
    lexicon_file = download_asset(files.lexicon)
    tokens_file = download_asset(files.tokens)
    if files.lm is not None:
        lm_file = download_asset(files.lm)
    else:
        lm_file = None

    return _PretrainedFiles(
        lexicon=lexicon_file,
        tokens=tokens_file,
        lm=lm_file,
    )
