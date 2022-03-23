import itertools

import torch
from parameterized import parameterized
from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_asset_path,
    skipIfNoCtcDecoder,
)


@skipIfNoCtcDecoder
class CTCDecoderTest(TempDirMixin, TorchaudioTestCase):
    def _get_decoder(self, tokens=None, use_lm=True, **kwargs):
        from torchaudio.prototype.ctc_decoder import lexicon_decoder

        lexicon_file = get_asset_path("decoder/lexicon.txt")
        kenlm_file = get_asset_path("decoder/kenlm.arpa") if use_lm else None

        if tokens is None:
            tokens = get_asset_path("decoder/tokens.txt")

        return lexicon_decoder(
            lexicon=lexicon_file,
            tokens=tokens,
            lm=kenlm_file,
            **kwargs,
        )

    def _get_emissions(self):
        B, T, N = 4, 15, 10

        torch.manual_seed(0)
        emissions = torch.rand(B, T, N)

        return emissions

    @parameterized.expand(
        list(
            itertools.product(
                [get_asset_path("decoder/tokens.txt"), ["-", "|", "f", "o", "b", "a", "r"]],
                [True, False],
            )
        ),
    )
    def test_construct_decoder(self, tokens, use_lm):
        self._get_decoder(tokens=tokens, use_lm=use_lm)

    def test_no_lm_decoder(self):
        """Check that using no LM produces the same result as using an LM with 0 lm_weight"""
        kenlm_decoder = self._get_decoder(lm_weight=0)
        zerolm_decoder = self._get_decoder(use_lm=False)

        emissions = self._get_emissions()
        kenlm_results = kenlm_decoder(emissions)
        zerolm_results = zerolm_decoder(emissions)
        self.assertEqual(kenlm_results, zerolm_results)

    def test_shape(self):
        emissions = self._get_emissions()
        decoder = self._get_decoder()

        results = decoder(emissions)
        self.assertEqual(len(results), emissions.shape[0])

    def test_timesteps_shape(self):
        """Each token should correspond with a timestep"""
        emissions = self._get_emissions()
        decoder = self._get_decoder()

        results = decoder(emissions)
        for i in range(emissions.shape[0]):
            result = results[i][0]
            self.assertEqual(result.tokens.shape, result.timesteps.shape)

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
