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
    def _get_decoder(self, tokens=None):
        from torchaudio.prototype.ctc_decoder import kenlm_lexicon_decoder

        lexicon_file = get_asset_path("decoder/lexicon.txt")
        kenlm_file = get_asset_path("decoder/kenlm.arpa")

        if tokens is None:
            tokens = get_asset_path("decoder/tokens.txt")

        return kenlm_lexicon_decoder(
            lexicon=lexicon_file,
            tokens=tokens,
            kenlm=kenlm_file,
        )

    @parameterized.expand([(get_asset_path("decoder/tokens.txt"),), (["-", "|", "f", "o", "b", "a", "r"],)])
    def test_construct_decoder(self, tokens):
        self._get_decoder(tokens)

    def test_shape(self):
        B, T, N = 4, 15, 10

        torch.manual_seed(0)
        emissions = torch.rand(B, T, N)

        decoder = self._get_decoder()
        results = decoder(emissions)

        self.assertEqual(len(results), B)

    @parameterized.expand([(get_asset_path("decoder/tokens.txt"),), (["-", "|", "f", "o", "b", "a", "r"],)])
    def test_index_to_tokens(self, tokens):
        # decoder tokens: '-' '|' 'f' 'o' 'b' 'a' 'r'
        decoder = self._get_decoder(tokens)

        idxs = torch.LongTensor((1, 2, 1, 3, 5))
        tokens = decoder.idxs_to_tokens(idxs)

        expected_tokens = ["|", "f", "|", "o", "a"]
        self.assertEqual(tokens, expected_tokens)
