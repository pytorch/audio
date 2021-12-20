import torch
import itertools as it
from typing import List, Optional, Dict
from collections import namedtuple

import torchaudio

try:
    torchaudio._extension._load_lib('libtorchaudio_decoder')
    from torchaudio._torchaudio_decoder import (
        _CriterionType,
        _KenLM,
        _LexiconDecoder,
        _LexiconDecoderOptions,
        _SmearingMode,
        _Trie,
        _Dictionary,
        _create_word_dict,
        _load_words,
    )
except ImportError:
    raise ImportError(
        "flashlight decoder bindings are required to use this functionality. "
        "Please set BUILD_CTC_DECODER=1 when building from source."
    )

__all__ = ["KenLMLexiconDecoder", "kenlm_lexicon_decoder"]


Hypothesis = namedtuple("Hypothesis", ["tokens", "words", "score"])

class KenLMLexiconDecoder:
    def __init__(
        self,
        nbest: int,
        lexicon: dict,
        word_dict: _Dictionary,
        tokens_dict: _Dictionary,
        kenlm: _KenLM,
        decoder_options: _LexiconDecoderOptions,
        blank_token: str,
        sil_token: str,
    ) -> None:
        """
        KenLM CTC Decoder with Lexicon constraint.

        Note:
            To build the decoder, please use the factory function kenlm_lexicon_decoder.

        Args:
            nbest (int): number of best decodings to return
            lexicon (dict): lexicon mapping of words to spellings
            word_dict (_Dictionary): dictionary of words
            tokens_dict (_Dictionary): dictionary of tokens
            kenlm (_KenLM): n-gram KenLM language model
            decoder_options (_LexiconDecoderOptions): parameters used for beam search decoding
            blank_token (str): token corresopnding to blank
            sil_token (str): token corresponding to silence
        """

        self.nbest = nbest
        self.decoder_options = decoder_options
        self.lexicon = lexicon
        self.word_dict = word_dict
        self.lm = kenlm
        self.tokens_dict = tokens_dict

        self.vocab_size = tokens_dict.index_size()

        self.unk_word = word_dict.get_index("<unk>")
        self.blank = self.tokens_dict.get_index(blank_token)
        self.silence = self.tokens_dict.get_index(sil_token)

        self.trie = _Trie(self.vocab_size, self.silence)
        start_state = self.lm.start(False)

        for word, spellings in self.lexicon.items():
            word_idx = self.word_dict.get_index(word)
            _, score = self.lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idx = [self.tokens_dict.get_index(token) for token in spelling]
                self.trie.insert(spelling_idx, word_idx, score)
        self.trie.smear(_SmearingMode.MAX)

        self.decoder = _LexiconDecoder(
            self.decoder_options,
            self.trie,
            self.lm,
            self.silence,
            self.blank,
            self.unk_word,
            [],
            False,  # word level LM
        )

    def _get_tokens(self, idxs: torch.IntTensor) -> torch.LongTensor:
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def decode(
        self,
        emissions: torch.FloatTensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[List[Dict]]:
        """
        Args:
            emissions (FloatTensor): tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model
            lengths (Tensor or None, optional): tensor of shape `(batch, )` storing the valid length of
                in time axis of the output Tensor in each batch

        Returns:
            List[Hypothesis]:
                List of sorted best hypotheses for each audio sequence in the batch.

                Each hypothesis is named tuple with the following fields:
                    tokens: torch.LongTensor of raw token IDs
                    score: hypothesis score
                    words: list of decoded words
        """
        B, T, N = emissions.size()
        if lengths is None:
            lengths = torch.full((B, ), T)

        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)

            results = self.decoder.decode(emissions_ptr, lengths[b], N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    Hypothesis(
                        self._get_tokens(result.tokens),  # token ids
                        list(self.word_dict.get_entry(x) for x in result.words if x >= 0),  # words
                        result.score,  # score
                    )
                    for result in nbest_results
                ]
            )

        return hypos

    def idxs_to_tokens(self, idxs: torch.LongTensor) -> List:
        """
        Map raw token IDs into correponding tokens

        Args:
            idxs (LongTensor): raw token IDs generated from decoder

        Returns:
            List: tokens corresponding to the input IDs
        """
        return [self.tokens_dict.get_entry(idx.item()) for idx in idxs]


def kenlm_lexicon_decoder(
    lexicon_file: str,
    tokens_file: str,
    kenlm_file: str,
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
) -> KenLMLexiconDecoder:
    """
    Builds Ken LM CTC Lexicon Decoder with given parameters

    Args:
        lexicon_file (str): lexicon file containing the possible words
        tokens_file (str): file containing valid tokens
        kenlm_file (str): file containing languge model
        nbest (int): number of best decodings to return (Default: 1)
        beam_size (int): max number of hypos to hold after each decode step (Default: 50)
        beam_size_token (int): max number of tokens to consider at each decode step.
            If None, it is set to the total number of tokens (Default: None)
        beam_threshold (float): threshold for pruning hypothesis (Default: 50)
        lm_weight (float): weight of lm (Default: 2)
        word_score (float): word insertion score (Default: 0)
        unk_score (float): unknown word insertion score (Default: -inf)
        sil_score (float): silence insertion score (Default: 0)
        log_add (bool): whether or not to use logadd when merging hypotheses (Default: False)
        blank_token (str): token corresponding to blank
        sil_token (str): token corresponding to silence

    Returns:
        KenLMLexiconDecoder: decoder
    """
    lexicon = _load_words(lexicon_file)
    word_dict = _create_word_dict(lexicon)
    kenlm = _KenLM(kenlm_file, word_dict)
    tokens_dict = _Dictionary(tokens_file)

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

    return KenLMLexiconDecoder(
        nbest=nbest,
        lexicon=lexicon,
        word_dict=word_dict,
        tokens_dict=tokens_dict,
        kenlm=kenlm,
        decoder_options=decoder_options,
        blank_token=blank_token,
        sil_token=sil_token,
    )
