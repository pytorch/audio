import torch
from parameterized import parameterized
from torchaudio_unittest.common_utils import get_asset_path, skipIfNoCuCtcDecoder, skipIfNoCuda, TempDirMixin, TorchaudioTestCase

NUM_TOKENS = 8

@skipIfNoCuda
@skipIfNoCuCtcDecoder
class CUCTCDecoderTest(TempDirMixin, TorchaudioTestCase):
    def _get_decoder(self, tokens=None, **kwargs):
        from torchaudio.models.decoder import cuda_ctc_decoder

        if tokens is None:
            tokens = get_asset_path("decoder/tokens.txt")

        return cuda_ctc_decoder(
            tokens=tokens,
            **kwargs,
        )

    def _get_emissions(self):
        B, T, N = 4, 15, NUM_TOKENS

        emissions = torch.rand(B, T, N).cuda()
        emissions = torch.nn.functional.log_softmax(emissions, -1)

        return emissions

    @parameterized.expand(
        [get_asset_path("decoder/tokens.txt"), ["-", "|", "f", "o", "b", "a", "r"]],
        )
    def test_construct_basic_decoder(self, tokens):
        self._get_decoder(tokens=tokens)

    def test_shape(self):
        log_probs = self._get_emissions()
        encoder_out_lens = torch.tensor([15,14,13,12], dtype=torch.int32).cuda()
        decoder = self._get_decoder()

        results = decoder(log_probs, encoder_out_lens)
        self.assertEqual(len(results), log_probs.shape[0])
