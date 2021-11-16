from torchaudio._torchaudio import (
    CriterionType,
    DecodeResult,
    KenLM,
    LexiconDecoder,
    LexiconDecoderOptions,
    SmearingMode,
    Trie,
    Dictionary,
    create_word_dict,
    load_words,
)

import torch
import itertools as it
from typing import List, Optional, Dict

class KenLMCTCLexiconDecoder:
    def __init__(
        self,
        nbest: int,
        lexicon: dict,
        word_dict: Dictionary,
        tokens_dict: Dictionary,
        kenlm: KenLM,
        decoder_options: LexiconDecoderOptions,
        is_token_lm: bool,
    ) -> None:

        """
        Construct a KenLM CTC Lexcion Decoder.

        Args:
            nbest (int): number of best decodings to return
            lexicon (dict): lexicon mapping of words to spellings
            word_dict (Dictionary): dictionary of words
            tokens_dict (Dictionary): dictionary of tokens
            kenlm (KenLM): n-gram KenLM language model
            decoder_options (LexiconDecoderOptions): parameters used for beam search decoding
            is_token_lm (bool): if LM is token-level or word-level
        """

        self.nbest = nbest
        self.is_token_lm = is_token_lm
        self.decoder_options = decoder_options
        self.lexicon = lexicon
        self.word_dict = word_dict
        self.lm = kenlm
        self.tokens_dict = tokens_dict
        
        self.vocab_size = tokens_dict.index_size()
        
        self.unk_word = word_dict.get_index("<unk>")
        self.blank = (
            self.tokens_dict.get_index("-")
            if self.tokens_dict.contains("-")
            else 0
        )
        if self.tokens_dict.contains("<sep>"):
            self.silence = self.tokens_dict.get_index("<sep>")
        elif self.tokens_dict.contains("|"):
            self.silence = self.tokens_dict.get_index("|")
        else:
            raise RuntimeError

        self.trie = Trie(self.vocab_size, self.silence)    
        start_state = self.lm.start(False)

        for word, spellings in self.lexicon.items():
            word_idx = self.word_dict.get_index(word)
            _, score = self.lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idx = [self.tokens_dict.get_index(token) for token in spelling]
                # TODO: assert unk not in spelling idxs
                self.trie.insert(spelling_idx, word_idx, score)
        self.trie.smear(SmearingMode.MAX)

        self.decoder = LexiconDecoder(
            self.decoder_options,
            self.trie,
            self.lm,
            self.silence,
            self.blank,
            self.unk_word,
            [],
            self.is_token_lm,
        )

    def _get_tokens(self, idxs: torch.IntTensor) -> torch.LongTensor:
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def decode(
        self,
        emissions: torch.FloatTensor,
        lengths: Optional[torch.Tensor],
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        """
        # TODO: example/usage

        Args:
            emissions (FloatTensor): tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model
            lenghts (Tensor or None, optional): tensor of shape `(batch, )` storing the valid length of
                each audio sequence in the batch

        Returns:
            List[List[Dict[str, torch.LongTensor]]]: List of sorted best hypotheses for each audio sequence
                in the batch, each a dictionary mapping the hypothesis to the corresponding tokens, score, 
                and words
        """
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self._get_tokens(result.tokens),
                        "score": result.score,
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ]
                    }
                    for result in nbest_results
                ]
            )
        
        return hypos


def kenlm_ctc_lexicon_decoder(
    nbest: int,
    lexicon_file: str,
    tokens_file: str,
    kenlm_file: str,
    beam_size: int,
    beam_size_token: int, 
    beam_threshold: float,
    lm_weight: float,
    word_score: float,
    unk_score: float,
    sil_score: float,
    log_add: bool,
    is_token_lm: bool,
) -> KenLMCTCLexiconDecoder:
    """
    Builds Ken LM CTC Lexicon Decoder with given parameters

    Args:
        nbest (int): number of best decodings to return
        lexicon_file (str): lexicon file containing the possible words
        tokens_file (str): file containing valid tokens
        kenlm_file (str): file containing languge model
        beam_size (int): max number of hypos to hold after each decode step
        beam_size_token (int): max number of tokens to consider at each decode step
        beam_threshold (float): threshold for pruning hypothesis
        lm_weight (float): weight of lm
        word_score (float): word insertion score
        unk_score (float): unknown word insertion score
        sil_score (float): silence insertion score
        log_add (bool): whether or not to use logadd when merging hypotheses
        is_token_lm (bool): if LM is token-level or word-level

    Returns:
        KenLMCTCLexiconDecoder: decoder
    """
    lexicon = load_words(lexicon_file)
    word_dict = create_word_dict(lexicon)
    kenlm = KenLM(kenlm_file, word_dict)
    tokens_dict = Dictionary(tokens_file)
    
    if not tokens_dict.contains("'"): # TODO: ??
        tokens_dict.add_entry("'")

    decoder_options = LexiconDecoderOptions(
        beam_size=beam_size,
        beam_size_token=beam_size_token, # Optional type -> default to num of tokens
        beam_threshold=beam_threshold,
        lm_weight=lm_weight,
        word_score=word_score,
        unk_score=unk_score,
        sil_score=sil_score,
        log_add=log_add,
        criterion_type=CriterionType.CTC,
    )

    return KenLMCTCLexiconDecoder(
        nbest=nbest,
        lexicon=lexicon,
        word_dict=word_dict,
        tokens_dict=tokens_dict,
        kenlm=kenlm,
        decoder_options=decoder_options,
        is_token_lm=is_token_lm,
    )
