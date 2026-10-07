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
    def _get_decoder(self, tokens=None, beam_size=5, **kwargs):
        from torchaudio.models.decoder import cuda_ctc_decoder

        if tokens is None:
            tokens = get_asset_path("decoder/tokens.txt")

        return cuda_ctc_decoder(
            tokens=tokens,
            beam_size=beam_size,
            **kwargs,
        )

    def _get_emissions(self, batch_size=4, num_frames=15, num_tokens=NUM_TOKENS):
        emissions = torch.rand(batch_size, num_frames, num_tokens).cuda()
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

    def test_large_beam_decode(self):
        torch.manual_seed(0)
        num_tokens = 129
        beam_size = 128
        # Keep the vocabulary larger than beam_size so the CUDA decoder
        # exercises the full top-k shared-memory result buffers.
        tokens = ["-"] + [f"token_{idx}" for idx in range(1, num_tokens)]
        log_probs = self._get_emissions(batch_size=2, num_frames=4, num_tokens=num_tokens)
        encoder_out_lens = torch.full((log_probs.shape[0],), log_probs.shape[1], dtype=torch.int32, device="cuda")

        decoder = self._get_decoder(tokens=tokens, beam_size=beam_size)
        self.assertEqual(decoder.beam_size, beam_size)
        results = decoder(log_probs, encoder_out_lens)
        torch.cuda.synchronize()

        self.assertEqual(len(results), log_probs.shape[0])
