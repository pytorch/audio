import itertools

import torch
from parameterized import parameterized
from torchaudio_unittest.common_utils import get_asset_path, skipIfNoCtcDecoder, TempDirMixin, TorchaudioTestCase

NUM_TOKENS = 8


@skipIfNoCtcDecoder
class CTCDecoderTest(TempDirMixin, TorchaudioTestCase):
    def _get_custom_kenlm(self, kenlm_file):
        from .ctc_decoder_utils import CustomKenLM

        dict_file = get_asset_path("decoder/lexicon.txt")
        custom_lm = CustomKenLM(kenlm_file, dict_file)
        return custom_lm

    def _get_biased_nnlm(self, dict_file, keyword):
        from .ctc_decoder_utils import BiasedLM, CustomBiasedLM

        model = BiasedLM(dict_file, keyword)
        biased_lm = CustomBiasedLM(model, dict_file)
        return biased_lm

    def _get_decoder(self, tokens=None, lm=None, use_lexicon=True, **kwargs):
        from torchaudio.models.decoder import ctc_decoder

        lexicon_file = get_asset_path("decoder/lexicon.txt") if use_lexicon else None
        if tokens is None:
            tokens = get_asset_path("decoder/tokens.txt")

        return ctc_decoder(
            lexicon=lexicon_file,
            tokens=tokens,
            lm=lm,
            **kwargs,
        )

    def _get_emissions(self):
        B, T, N = 4, 15, NUM_TOKENS

        emissions = torch.rand(B, T, N)

        return emissions

    @parameterized.expand(
        list(
            itertools.product(
                [get_asset_path("decoder/tokens.txt"), ["-", "|", "f", "o", "b", "a", "r"]],
                [None, get_asset_path("decoder/kenlm.arpa")],
                [True, False],
            )
        ),
    )
    def test_construct_basic_decoder(self, tokens, lm, use_lexicon):
        self._get_decoder(tokens=tokens, lm=lm, use_lexicon=use_lexicon)

    @parameterized.expand(
        [(True,), (False,)],
    )
    def test_shape(self, use_lexicon):
        emissions = self._get_emissions()
        decoder = self._get_decoder(use_lexicon=use_lexicon)

        results = decoder(emissions)
        self.assertEqual(len(results), emissions.shape[0])

    @parameterized.expand(
        [(True,), (False,)],
    )
    def test_timesteps_shape(self, use_lexicon):
        """Each token should correspond with a timestep"""
        emissions = self._get_emissions()
        decoder = self._get_decoder(use_lexicon=use_lexicon)

        results = decoder(emissions)
        for i in range(emissions.shape[0]):
            result = results[i][0]
            self.assertEqual(result.tokens.shape, result.timesteps.shape)

    def test_no_lm_decoder(self):
        """Check that the following produce the same result
        - using no LM (C++ based implementation)
        - using no LM (Custom Python based wrapper)
        - using a (Ken)LM with 0 weight
        """
        from .ctc_decoder_utils import CustomZeroLM

        emissions = self._get_emissions()
        custom_zerolm = CustomZeroLM()
        zerolm_decoder_custom = self._get_decoder(lm=custom_zerolm)
        zerolm_decoder_cpp = self._get_decoder(lm=None)
        kenlm_file = get_asset_path("decoder/kenlm.arpa")
        kenlm_decoder = self._get_decoder(lm=kenlm_file, lm_weight=0)

        zerolm_custom_results = zerolm_decoder_custom(emissions)
        zerolm_cpp_results = zerolm_decoder_cpp(emissions)
        kenlm_results = kenlm_decoder(emissions)

        self.assertEqual(zerolm_cpp_results, zerolm_custom_results)
        self.assertEqual(zerolm_cpp_results, kenlm_results)

    def test_custom_kenlm_decoder(self):
        """Check that creating a custom Python KenLM wrapper produces same results as C++ based KenLM"""
        emissions = self._get_emissions()

        kenlm_file = get_asset_path("decoder/kenlm.arpa")
        custom_kenlm = self._get_custom_kenlm(kenlm_file)
        kenlm_decoder_custom = self._get_decoder(lm=custom_kenlm)
        kenlm_decoder_cpp = self._get_decoder(lm=kenlm_file)

        kenlm_custom_results = kenlm_decoder_custom(emissions)
        kenlm_cpp_results = kenlm_decoder_cpp(emissions)

        self.assertEqual(kenlm_custom_results, kenlm_cpp_results)

    @parameterized.expand(
        [
            (get_asset_path("decoder/nnlm_lex_dict.txt"), "foo", True),
            (get_asset_path("decoder/nnlm_lexfree_dict.txt"), "f", False),
        ]
    )
    def test_custom_nnlm_decoder(self, lm_dict, keyword, use_lexicon):
        """Check that biased NNLM only produces biased words"""
        emissions = self._get_emissions()
        custom_nnlm = self._get_biased_nnlm(lm_dict, keyword)
        nnlm_decoder = self._get_decoder(lm=custom_nnlm, lm_dict=lm_dict, use_lexicon=use_lexicon, lm_weight=10)
        nnlm_results = nnlm_decoder(emissions)

        if use_lexicon:
            output = [result[0].words for result in nnlm_results]
        else:
            tokens = [nnlm_decoder.idxs_to_tokens(result[0].tokens) for result in nnlm_results]
            output = [list(filter(("|").__ne__, t)) for t in tokens]  # filter out silence characters

        lens = [len(out) for out in output]
        expected = [[keyword] * len for len in lens]  # all of output should match the biased keyword
        assert expected == output

    def test_get_timesteps(self):
        unprocessed_tokens = torch.tensor([2, 2, 0, 3, 3, 3, 0, 3])
        decoder = self._get_decoder()
        timesteps = decoder._get_timesteps(unprocessed_tokens)

        expected = [0, 3, 7]
        self.assertEqual(timesteps, expected)

    def test_get_tokens_and_idxs(self):
        unprocessed_tokens = torch.tensor([2, 2, 0, 3, 3, 3, 0, 3])  # ["f", "f", "-", "o", "o", "o", "-", "o"]

        decoder = self._get_decoder()
        token_ids = decoder._get_tokens(unprocessed_tokens)
        tokens = decoder.idxs_to_tokens(token_ids)

        expected_ids = [2, 3, 3]
        self.assertEqual(token_ids, expected_ids)

        expected_tokens = ["f", "o", "o"]
        self.assertEqual(tokens, expected_tokens)

    @parameterized.expand([(get_asset_path("decoder/tokens.txt"),), (["-", "|", "f", "o", "b", "a", "r"],)])
    def test_index_to_tokens(self, tokens):
        # decoder tokens: '-' '|' 'f' 'o' 'b' 'a' 'r'
        decoder = self._get_decoder(tokens)

        idxs = torch.LongTensor((1, 2, 1, 3, 5))
        tokens = decoder.idxs_to_tokens(idxs)

        expected_tokens = ["|", "f", "|", "o", "a"]
        self.assertEqual(tokens, expected_tokens)
