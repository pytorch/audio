import torch
from parameterized import parameterized
from torchaudio.prototype.models import (
    conformer_wav2vec2_base,
    conformer_wav2vec2_pretrain_base,
    conformer_wav2vec2_pretrain_large,
    emformer_hubert_base,
)
from torchaudio_unittest.common_utils import nested_params, skipIfNoCuda, torch_script, TorchaudioTestCase


def _get_features_lengths(batch_size, num_frames, feature_dim, low):
    torch.manual_seed(0)
    features = torch.randn(batch_size, num_frames, feature_dim)
    lengths = torch.randint(
        low=low,
        high=num_frames,
        size=[
            batch_size,
        ],
    )
    return features, lengths


class TestSSLModel(TorchaudioTestCase):
    def _smoke_test(self, model, feature_dim, low, device, dtype):
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        batch_size, num_frames = 3, 1024
        features, lengths = _get_features_lengths(batch_size, num_frames, feature_dim, low)
        features = features.to(device=device, dtype=dtype)
        lengths = lengths.to(device)

        model(features, lengths)

    # low=27 for conformer_wav2vec2 is set to ensure negative sampling on masked indices is possible.
    # There must be at least 2 masked indices, so we compute low = math.ceil(2 * time_reduction / mask_prob)
    @nested_params(
        [(conformer_wav2vec2_base, 64, 27), (conformer_wav2vec2_pretrain_base, 64, 27), (emformer_hubert_base, 80, 1)],
        [torch.float32, torch.float64],
    )
    def test_cpu_smoke_test(self, model_feature_dim, dtype):
        model, feature_dim, low = model_feature_dim
        model = model()
        self._smoke_test(model, feature_dim, low, torch.device("cpu"), dtype)

    @nested_params(
        [(conformer_wav2vec2_base, 64, 27), (conformer_wav2vec2_pretrain_base, 64, 27), (emformer_hubert_base, 80, 1)],
        [torch.float32, torch.float64],
    )
    @skipIfNoCuda
    def test_cuda_smoke_test(self, model_feature_dim, low, dtype):
        model, feature_dim, low = model_feature_dim
        model = model()
        self._smoke_test(model, feature_dim, low, torch.device("cuda"), dtype)

    @parameterized.expand(
        [
            (conformer_wav2vec2_base, 64, 27, None),
            (emformer_hubert_base, 80, 1, None),
            (emformer_hubert_base, 80, 1, 512),
        ]
    )
    def test_extract_feature(self, model, feature_dim, low, aux_num_out):
        if aux_num_out is not None:
            model = model(aux_num_out=aux_num_out)
        else:
            model = model()
        model.eval()

        batch_size, num_frames = 3, 1024
        if feature_dim == 64:
            num_layers = len(model.encoder.conformer)
        else:
            num_layers = len(model.encoder.emformer.emformer_layers)

        features, lengths = _get_features_lengths(batch_size, num_frames, feature_dim, low)

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

    @parameterized.expand(
        [
            (conformer_wav2vec2_base, 64, None),
            (emformer_hubert_base, 80, None),
            (emformer_hubert_base, 80, 512),
        ]
    )
    def test_zero_length(self, model, feature_dim, aux_num_out):
        if aux_num_out is not None:
            model = model(aux_num_out=aux_num_out)
        else:
            model = model()
        model.eval()

        batch_size, num_frames = 3, 1024
        features = torch.randn(batch_size, num_frames, feature_dim)
        input_lengths = torch.zeros(batch_size)
        _, output_lengths = model(features, input_lengths)
        self.assertEqual(torch.zeros_like(output_lengths), output_lengths)

        _, output_lengths = model.extract_features(features, input_lengths)
        self.assertEqual(torch.zeros_like(output_lengths), output_lengths)

    @parameterized.expand(
        [
            (conformer_wav2vec2_base, 64, 27, None),
            (emformer_hubert_base, 80, 1, None),
            (emformer_hubert_base, 80, 1, 512),
        ]
    )
    def test_torchscript_consistency(self, model, feature_dim, low, aux_num_out):
        if aux_num_out is not None:
            model = model(aux_num_out=aux_num_out)
        else:
            model = model()
        model.eval()

        batch_size, num_frames = 3, 1024
        features, lengths = _get_features_lengths(batch_size, num_frames, feature_dim, low)

        ref_out, ref_len = model(features, lengths)

        scripted = torch_script(model)
        hyp_out, hyp_len = scripted(features, lengths)

        self.assertEqual(hyp_out, ref_out)
        self.assertEqual(hyp_len, ref_len)

    @parameterized.expand(
        [
            (conformer_wav2vec2_pretrain_base, 256),
            (conformer_wav2vec2_pretrain_large, 768),
        ]
    )
    def test_conformerw2v2_pretrain_shapes(self, model, output_dim):
        batch_size, num_frames, feature_dim = 3, 2048, 64
        features, lengths = _get_features_lengths(batch_size, num_frames, feature_dim, low=27)

        model = model()
        model.eval()
        x, lengths, mask_idxs, targets, negs, neg_idxs = model(features, lengths)

        # factory function parameter values
        stride = 4
        num_negatives = 100
        mask_len = torch.max(mask_idxs.sum(dim=1))

        assert x.shape == (batch_size, num_frames // stride, output_dim)
        assert lengths.shape == (batch_size,)
        assert mask_idxs.shape == (batch_size, num_frames // stride)
        assert targets.shape == (batch_size, mask_len, output_dim)
        assert neg_idxs.shape == (batch_size, num_negatives * mask_len)
        assert negs.shape == (num_negatives, batch_size, mask_len, output_dim)
