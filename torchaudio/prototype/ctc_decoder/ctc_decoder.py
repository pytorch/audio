import itertools as it
from collections import namedtuple
from typing import Dict, List, Optional, Union, NamedTuple

import torch
from torchaudio._torchaudio_decoder import (
    _CriterionType,
    _LM,
    _KenLM,
    _LexiconDecoder,
    _LexiconDecoderOptions,
    _SmearingMode,
    _Trie,
    _Dictionary,
    _create_word_dict,
    _load_words,
    _ZeroLM,
)
from torchaudio.utils import download_asset


__all__ = ["Hypothesis", "LexiconDecoder", "lexicon_decoder"]


_PretrainedFiles = namedtuple("PretrainedFiles", ["lexicon", "tokens", "lm"])


class Hypothesis(NamedTuple):
    r"""Represents hypothesis generated by CTC beam search decoder :py:func`LexiconDecoder`.

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


class LexiconDecoder:
    """torchaudio.prototype.ctc_decoder.LexiconDecoder()

    Note:
        To build the decoder, please use factory function
        :py:func:`lexicon_decoder`.
    """

    def __init__(
        self,
        nbest: int,
        lexicon: Dict,
        word_dict: _Dictionary,
        tokens_dict: _Dictionary,
        lm: _LM,
        decoder_options: _LexiconDecoderOptions,
        blank_token: str,
        sil_token: str,
        unk_word: str,
    ) -> None:
        """
        CTC Decoder with Lexicon constraint.

        Note:
            To build the decoder, please use the factory function lexicon_decoder.

        Args:
            nbest (int): number of best decodings to return
            lexicon (Dict): lexicon mapping of words to spellings
            word_dict (_Dictionary): dictionary of words
            tokens_dict (_Dictionary): dictionary of tokens
            lm (_LM): language model
            decoder_options (_LexiconDecoderOptions): parameters used for beam search decoding
            blank_token (str): token corresopnding to blank
            sil_token (str): token corresponding to silence
            unk_word (str): word corresponding to unknown
        """

        self.nbest = nbest
        self.word_dict = word_dict
        self.tokens_dict = tokens_dict

        unk_word = word_dict.get_index(unk_word)
        self.blank = self.tokens_dict.get_index(blank_token)
        silence = self.tokens_dict.get_index(sil_token)

        vocab_size = self.tokens_dict.index_size()
        trie = _Trie(vocab_size, silence)
        start_state = lm.start(False)

        for word, spellings in lexicon.items():
            word_idx = self.word_dict.get_index(word)
            _, score = lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idx = [self.tokens_dict.get_index(token) for token in spelling]
                trie.insert(spelling_idx, word_idx, score)
        trie.smear(_SmearingMode.MAX)

        self.decoder = _LexiconDecoder(
            decoder_options,
            trie,
            lm,
            silence,
            self.blank,
            unk_word,
            [],
            False,  # word level LM
        )

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

    def __call__(self, emissions: torch.FloatTensor, lengths: Optional[torch.Tensor] = None) -> List[List[Hypothesis]]:
        # Overriding the signature so that the return type is correct on Sphinx
        """__call__(self, emissions: torch.FloatTensor, lengths: Optional[torch.Tensor] = None) -> \
            List[List[torchaudio.prototype.ctc_decoder.Hypothesis]]

        Args:
            emissions (torch.FloatTensor): tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model
            lengths (Tensor or None, optional): tensor of shape `(batch, )` storing the valid length of
                in time axis of the output Tensor in each batch

        Returns:
            List[List[Hypothesis]]:
                List of sorted best hypotheses for each audio sequence in the batch.
        """
        assert emissions.dtype == torch.float32

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
                    Hypothesis(
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


def lexicon_decoder(
    lexicon: str,
    tokens: Union[str, List[str]],
    lm: Optional[str] = None,
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
) -> LexiconDecoder:
    """
    Builds Ken LM CTC Lexicon Decoder with given parameters

    Args:
        lexicon (str): lexicon file containing the possible words and corresponding spellings.
            Each line consists of a word and its space separated spelling
        tokens (str or List[str]): file or list containing valid tokens. If using a file, the expected
            format is for tokens mapping to the same index to be on the same line
        lm (str or None, optional): file containing language model, or `None` if not using a language model
        nbest (int, optional): number of best decodings to return (Default: 1)
        beam_size (int, optional): max number of hypos to hold after each decode step (Default: 50)
        beam_size_token (int, optional): max number of tokens to consider at each decode step.
            If None, it is set to the total number of tokens (Default: None)
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
        LexiconDecoder: decoder

    Example
        >>> decoder = lexicon_decoder(
        >>>     lexicon="lexicon.txt",
        >>>     tokens="tokens.txt",
        >>>     lm="kenlm.bin",
        >>> )
        >>> results = decoder(emissions) # List of shape (B, nbest) of Hypotheses
    """
    lexicon = _load_words(lexicon)
    word_dict = _create_word_dict(lexicon)
    lm = _KenLM(lm, word_dict) if lm else _ZeroLM()
    tokens_dict = _Dictionary(tokens)

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

    return LexiconDecoder(
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
        model (str): pretrained language model to download
            Options: ["librispeech-3-gram", "librispeech-4-gram", "librispeech"]

    Returns:
        Object with the following attributes:
            lm: path corresponding to downloaded language model, or None if model is not
                associated with an lm
            lexicon: path corresponding to downloaded lexicon file
            tokens: path corresponding to downloaded tokens file
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
