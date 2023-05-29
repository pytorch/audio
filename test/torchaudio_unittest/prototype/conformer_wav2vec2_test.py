import torch
from parameterized import parameterized
from torchaudio.prototype.models import (
    conformer_wav2vec2_base,
    conformer_wav2vec2_pretrain_base,
    conformer_wav2vec2_pretrain_large,
)
from torchaudio_unittest.common_utils import disabledInCI, nested_params, skipIfNoCuda, torch_script, TorchaudioTestCase


class TestConformerWav2Vec2(TorchaudioTestCase):
    def _smoke_test(self, model, device, dtype):
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        batch_size, num_frames, in_features = 3, 1024, 64
        features = torch.randn(batch_size, num_frames, in_features, device=device, dtype=dtype)
        lengths = torch.randint(
            low=0,
            high=num_frames,
            size=[
                batch_size,
            ],
            device=device,
        )

        model(features, lengths)

    @parameterized.expand([(torch.float32,), (torch.float64,)])
    def test_cpu_smoke_test(self, dtype):
        model = conformer_wav2vec2_base()
        self._smoke_test(model, torch.device("cpu"), dtype)

    @parameterized.expand([(torch.float32,), (torch.float64,)])
    @skipIfNoCuda
    # Disabled in CI: https://github.com/pytorch/audio/issues/3376
    @disabledInCI
    def test_cuda_smoke_test(self, dtype):
        model = conformer_wav2vec2_base()
        self._smoke_test(model, torch.device("cuda"), dtype)

    @nested_params(
        [conformer_wav2vec2_pretrain_base, conformer_wav2vec2_pretrain_large],
        [torch.float32, torch.float64],
    )
    def test_pretrain_cpu_smoke_test(self, model, dtype):
        model = model()
        self._smoke_test(model, torch.device("cpu"), dtype)

    @nested_params(
        [conformer_wav2vec2_pretrain_base, conformer_wav2vec2_pretrain_large],
        [torch.float32, torch.float64],
    )
    @skipIfNoCuda
    def test_pretrain_cuda_smoke_test(self, model, dtype):
        model = model()
        self._smoke_test(model, torch.device("cuda"), dtype)

    def test_extract_feature(self):
        model = conformer_wav2vec2_base()
        model.eval()

        batch_size, num_frames, in_features = 3, 1024, 64
        num_layers = len(model.encoder.conformer)

        features = torch.randn(batch_size, num_frames, in_features)
        lengths = torch.randint(
            low=0,
            high=num_frames,
            size=[
                batch_size,
            ],
        )

        all_features, lengths_ = model.extract_features(features, lengths, num_layers=None)
        assert len(all_features) == num_layers
        for feats in all_features:
            assert feats.ndim == 3
            assert feats.shape[0] == batch_size
        assert lengths_.shape == torch.Size([batch_size])

        for l in range(1, num_layers + 1):
            feats, lengths_ = model.extract_features(features, lengths, num_layers=l)
            assert len(feats) == l
            for i in range(l):
                self.assertEqual(all_features[i], feats[i])
            assert lengths_.shape == torch.Size([batch_size])

    def test_zero_length(self):
        model = conformer_wav2vec2_base()
        model.eval()

        batch_size, num_frames, in_features = 3, 1024, 64
        features = torch.randn(batch_size, num_frames, in_features)
        input_lengths = torch.zeros(batch_size)
        _, output_lengths = model(features, input_lengths)
        self.assertEqual(torch.zeros_like(output_lengths), output_lengths)

        _, output_lengths = model.extract_features(features, input_lengths)
        self.assertEqual(torch.zeros_like(output_lengths), output_lengths)

    def test_torchscript_consistency(self):
        model = conformer_wav2vec2_base()
        model.eval()

        batch_size, num_frames, in_features = 3, 1024, 64
        features = torch.randn(batch_size, num_frames, in_features)
        lengths = torch.randint(
            low=0,
            high=num_frames,
            size=[
                batch_size,
            ],
        )

        ref_out, ref_len = model(features, lengths)

        scripted = torch_script(model)
        hyp_out, hyp_len = scripted(features, lengths)

        self.assertEqual(hyp_out, ref_out)
        self.assertEqual(hyp_len, ref_len)
