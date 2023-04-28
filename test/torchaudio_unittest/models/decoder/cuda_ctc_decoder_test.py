import torch
from torchaudio_unittest.common_utils import (
    get_asset_path,
    skipIfNoCuCtcDecoder,
    skipIfNoCuda,
    TempDirMixin,
    TorchaudioTestCase,
)

NUM_TOKENS = 7


@skipIfNoCuda
@skipIfNoCuCtcDecoder
class CUCTCDecoderTest(TempDirMixin, TorchaudioTestCase):
    def _get_decoder(self, tokens=None, **kwargs):
        from torchaudio.models.decoder import cuda_ctc_decoder

        if tokens is None:
            tokens = get_asset_path("decoder/tokens.txt")

        return cuda_ctc_decoder(
            tokens=tokens,
            beam_size=5,
            **kwargs,
        )

    def _get_emissions(self):
        B, T, N = 4, 15, NUM_TOKENS

        emissions = torch.rand(B, T, N).cuda()
        emissions = torch.nn.functional.log_softmax(emissions, -1)

        return emissions

    def test_construct_basic_decoder_path(self):
        tokens_path = get_asset_path("decoder/tokens.txt")
        self._get_decoder(tokens=tokens_path)

    def test_construct_basic_decoder_tokens(self):
        tokens = ["-", "|", "f", "o", "b", "a", "r"]
        self._get_decoder(tokens=tokens)

    def test_shape(self):
        log_probs = self._get_emissions()
        encoder_out_lens = torch.tensor([15, 14, 13, 12], dtype=torch.int32).cuda()
        decoder = self._get_decoder()
        results = decoder(log_probs, encoder_out_lens)
        self.assertEqual(len(results), log_probs.shape[0])
