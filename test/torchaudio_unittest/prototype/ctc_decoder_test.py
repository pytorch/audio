import torch
from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_asset_path,
    skipIfNoCtcDecoder,
)


@skipIfNoCtcDecoder
class CTCDecoderTest(TempDirMixin, TorchaudioTestCase):
    def _get_decoder(self):
        from torchaudio.prototype.ctc_decoder import kenlm_lexicon_decoder

        lexicon_file = get_asset_path("decoder/lexicon.txt")
        tokens_file = get_asset_path("decoder/tokens.txt")
        kenlm_file = get_asset_path("decoder/kenlm.arpa")

        return kenlm_lexicon_decoder(
            lexicon=lexicon_file,
            tokens=tokens_file,
            kenlm=kenlm_file,
        )

    def test_construct_decoder(self):
        self._get_decoder()

    def test_shape(self):
        B, T, N = 4, 15, 10

        torch.manual_seed(0)
        emissions = torch.rand(B, T, N)

        decoder = self._get_decoder()
        results = decoder(emissions)

        self.assertEqual(len(results), B)

    def test_index_to_tokens(self):
        # decoder tokens: '-' '|' 'f' 'o' 'b' 'a' 'r'
        decoder = self._get_decoder()

        idxs = torch.LongTensor((1, 2, 1, 3, 5))
        tokens = decoder.idxs_to_tokens(idxs)

        expected_tokens = ["|", "f", "|", "o", "a"]
        self.assertEqual(tokens, expected_tokens)
