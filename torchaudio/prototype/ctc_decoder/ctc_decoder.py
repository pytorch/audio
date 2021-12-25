import itertools as it
from collections import namedtuple
from typing import Dict
from typing import List, Optional

import torch
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


__all__ = ["KenLMLexiconDecoder", "kenlm_lexicon_decoder"]


Hypothesis = namedtuple("Hypothesis", ["tokens", "words", "score"])


class KenLMLexiconDecoder:
    def __init__(
        self,
        nbest: int,
        lexicon: Dict,
        word_dict: _Dictionary,
        tokens_dict: _Dictionary,
        kenlm: _KenLM,
        decoder_options: _LexiconDecoderOptions,
        blank_token: str,
        sil_token: str,
        unk_word: str,
    ) -> None:
        """
        KenLM CTC Decoder with Lexicon constraint.

        Note:
            To build the decoder, please use the factory function kenlm_lexicon_decoder.

        Args:
            nbest (int): number of best decodings to return
            lexicon (Dict): lexicon mapping of words to spellings
            word_dict (_Dictionary): dictionary of words
            tokens_dict (_Dictionary): dictionary of tokens
            kenlm (_KenLM): n-gram KenLM language model
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
        start_state = kenlm.start(False)

        for word, spellings in lexicon.items():
            word_idx = self.word_dict.get_index(word)
            _, score = kenlm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idx = [self.tokens_dict.get_index(token) for token in spelling]
                trie.insert(spelling_idx, word_idx, score)
        trie.smear(_SmearingMode.MAX)

        self.decoder = _LexiconDecoder(
            decoder_options,
            trie,
            kenlm,
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

    def __call__(self, emissions: torch.FloatTensor, lengths: Optional[torch.Tensor] = None) -> List[List[Hypothesis]]:
        """
        Args:
            emissions (torch.FloatTensor): tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model
            lengths (Tensor or None, optional): tensor of shape `(batch, )` storing the valid length of
                in time axis of the output Tensor in each batch

        Returns:
            List[List[Hypothesis]]:
                List of sorted best hypotheses for each audio sequence in the batch.

                Each hypothesis is named tuple with the following fields:
                    tokens: torch.LongTensor of raw token IDs
                    score: hypothesis score
                    words: list of decoded words
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
                        self._get_tokens(result.tokens),  # token ids
                        [self.word_dict.get_entry(x) for x in result.words if x >= 0],  # words
                        result.score,  # score
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


def kenlm_lexicon_decoder(
    lexicon: str,
    tokens: str,
    kenlm: str,
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
) -> KenLMLexiconDecoder:
    """
    Builds Ken LM CTC Lexicon Decoder with given parameters

    Args:
        lexicon (str): lexicon file containing the possible words
        tokens (str): file containing valid tokens
        kenlm (str): file containing languge model
        nbest (int, optional): number of best decodings to return (Default: 1)
        beam_size (int, optional): max number of hypos to hold after each decode step (Default: 50)
        beam_size_token (int, optional): max number of tokens to consider at each decode step.
            If None, it is set to the total number of tokens (Default: None)
        beam_threshold (float, optional): threshold for pruning hypothesis (Default: 50)
        lm_weight (float, optional): weight of lm (Default: 2)
        word_score (float, optional): word insertion score (Default: 0)
        unk_score (float, optional): unknown word insertion score (Default: -inf)
        sil_score (float, optional): silence insertion score (Default: 0)
        log_add (bool, optional): whether or not to use logadd when merging hypotheses (Default: False)
        blank_token (str, optional): token corresponding to blank (Default: "-")
        sil_token (str, optional): token corresponding to silence (Default: "|")
        unk_word (str, optional): word corresponding to unknown (Default: "<unk>")

    Returns:
        KenLMLexiconDecoder: decoder
    """
    lexicon = _load_words(lexicon)
    word_dict = _create_word_dict(lexicon)
    kenlm = _KenLM(kenlm, word_dict)
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

    return KenLMLexiconDecoder(
        nbest=nbest,
        lexicon=lexicon,
        word_dict=word_dict,
        tokens_dict=tokens_dict,
        kenlm=kenlm,
        decoder_options=decoder_options,
        blank_token=blank_token,
        sil_token=sil_token,
        unk_word=unk_word,
    )
