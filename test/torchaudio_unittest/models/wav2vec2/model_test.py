import os
from typing import Tuple

import torch
import torch.nn.functional as F
from parameterized import parameterized
from torchaudio.models.wav2vec2 import (
    hubert_base,
    hubert_large,
    hubert_xlarge,
    wav2vec2_base,
    wav2vec2_large,
    wav2vec2_large_lv60k,
)
from torchaudio_unittest.common_utils import skipIfNoCuda, skipIfNoQengine, torch_script, TorchaudioTestCase

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 10):
    import torch.ao.quantization as tq
else:
    import torch.quantization as tq


def _name_func(testcase_func, i, param):
    return f"{testcase_func.__name__}_{i}_{param[0][0].__name__}"


factory_funcs = parameterized.expand(
    [
        (wav2vec2_base,),
        (wav2vec2_large,),
        (wav2vec2_large_lv60k,),
        (hubert_base,),
        (hubert_large,),
        (hubert_xlarge,),
    ],
    name_func=_name_func,
)


class TestWav2Vec2Model(TorchaudioTestCase):
    def _smoke_test(self, model, device, dtype):
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        batch_size, num_frames = 3, 1024

        waveforms = torch.randn(batch_size, num_frames, device=device, dtype=dtype)
        lengths = torch.randint(
            low=0,
            high=num_frames,
            size=[
                batch_size,
            ],
            device=device,
        )

        model(waveforms, lengths)

    @parameterized.expand([(torch.float32,), (torch.float64,)])
    def test_cpu_smoke_test(self, dtype):
        model = wav2vec2_base()
        self._smoke_test(model, torch.device("cpu"), dtype)
        model = wav2vec2_base(aux_num_out=32)
        self._smoke_test(model, torch.device("cpu"), dtype)

    @parameterized.expand([(torch.float32,), (torch.float64,)])
    @skipIfNoCuda
    def test_cuda_smoke_test(self, dtype):
        model = wav2vec2_base()
        self._smoke_test(model, torch.device("cuda"), dtype)
        model = wav2vec2_base(aux_num_out=32)
        self._smoke_test(model, torch.device("cuda"), dtype)

    def _feature_extractor_test(self, model):
        batch_size, num_frames = 3, 1024

        model.eval()
        num_layers = len(model.encoder.transformer.layers)

        waveforms = torch.randn(batch_size, num_frames)
        lengths = torch.randint(
            low=0,
            high=num_frames,
            size=[
                batch_size,
            ],
        )

        # Not providing num_layers returns all the intermediate features from
        # tranformer layers
        all_features, lengths_ = model.extract_features(waveforms, lengths, num_layers=None)
        assert len(all_features) == num_layers
        for features in all_features:
            assert features.ndim == 3
            assert features.shape[0] == batch_size
        assert lengths_.shape == torch.Size([batch_size])

        # Limiting the number of layers to `l`.
        for l in range(1, num_layers + 1):
            features, lengths_ = model.extract_features(waveforms, lengths, num_layers=l)
            assert len(features) == l
            for i in range(l):
                self.assertEqual(all_features[i], features[i])
            assert lengths_.shape == torch.Size([batch_size])

    @factory_funcs
    def test_extract_feature(self, factory_func):
        """`extract_features` method does not fail"""
        self._feature_extractor_test(factory_func(aux_num_out=32))

    def _test_batch_consistency(self, model):
        model.eval()
        batch_size, max_frames = 5, 5 * 1024
        waveforms = torch.randn(batch_size, max_frames)
        input_lengths = torch.tensor([i * 3200 for i in range(1, 6)])

        # Batch process with lengths
        batch_logits, output_lengths = model(waveforms, input_lengths)
        for i in range(batch_size):
            # Par-sample process without feeding length
            single_logit, _ = model(waveforms[i : i + 1, : input_lengths[i]], None)
            batch_logit = batch_logits[i : i + 1, : output_lengths[i]]

            # Convert to probability so that it's easier to interpretate the diff
            single_prob = F.softmax(single_logit, dim=2)
            batch_prob = F.softmax(batch_logit, dim=2)
            # We allow max atol=0.005 -> 0.5%
            self.assertEqual(single_prob, batch_prob, atol=0.005, rtol=0)

    @factory_funcs
    def test_pretrain_batch_consistency(self, factory_func):
        """Results from single process and batched process should be reasonably close"""
        self._test_batch_consistency(factory_func())

    @factory_funcs
    def test_finetune_batch_consistency(self, factory_func):
        """Results from single process and batched process should be reasonably close"""
        self._test_batch_consistency(factory_func(aux_num_out=32))

    def _test_zero_length(self, model):
        model.eval()
        batch_size = 3
        waveforms = torch.randn(batch_size, 1024)
        input_lengths = torch.zeros(batch_size)
        _, output_lengths = model(waveforms, input_lengths)
        self.assertEqual(torch.zeros_like(output_lengths), output_lengths)
        _, output_lengths = model.extract_features(waveforms, input_lengths)
        self.assertEqual(torch.zeros_like(output_lengths), output_lengths)

    @factory_funcs
    def test_pretrain_zero_length(self, factory_func):
        """Passing zero length should not fail"""
        self._test_zero_length(factory_func())

    @factory_funcs
    def test_finetune_zero_length(self, factory_func):
        """Passing zero length should not fail"""
        self._test_zero_length(factory_func(aux_num_out=32))

    def _test_torchscript(self, model):
        model.eval()

        batch_size, num_frames = 3, 1024

        waveforms = torch.randn(batch_size, num_frames)
        lengths = torch.randint(
            low=0,
            high=num_frames,
            size=[
                batch_size,
            ],
        )

        ref_out, ref_len = model(waveforms, lengths)

        scripted = torch_script(model)

        hyp_out, hyp_len = scripted(waveforms, lengths)

        self.assertEqual(hyp_out, ref_out)
        self.assertEqual(hyp_len, ref_len)

    @factory_funcs
    def test_pretrain_torchscript(self, factory_func):
        """Wav2Vec2Model should be scriptable"""
        if factory_func is hubert_xlarge and os.environ.get("CI") == "true":
            self.skipTest(
                "hubert_xlarge is known to fail on CI. " "See https://github.com/pytorch/pytorch/issues/65776"
            )
        self._test_torchscript(factory_func())

    @factory_funcs
    def test_finetune_torchscript(self, factory_func):
        """Wav2Vec2Model should be scriptable"""
        if factory_func is hubert_xlarge and os.environ.get("CI") == "true":
            self.skipTest(
                "hubert_xlarge is known to fail on CI. " "See https://github.com/pytorch/pytorch/issues/65776"
            )
        self._test_torchscript(factory_func(aux_num_out=32))

    def _test_quantize_smoke_test(self, model):
        model.eval()
        batch_size, num_frames = 3, 1024

        # Remove the weight normalization forward hook
        model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
        quantized = tq.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

        # A lazy way to check that Modules are different
        assert str(quantized) != str(model), "Dynamic quantization did not modify the module."

        waveforms = torch.randn(batch_size, num_frames)
        lengths = torch.randint(
            low=0,
            high=num_frames,
            size=[
                batch_size,
            ],
        )

        _, _ = quantized(waveforms, lengths)

    @factory_funcs
    @skipIfNoQengine
    def test_quantize(self, factory_func):
        """Wav2Vec2Model should support basic quantization"""
        self._test_quantize_smoke_test(factory_func(aux_num_out=32))

    def _test_quantize_torchscript(self, model):
        model.eval()

        batch_size, num_frames = 3, 1024

        # Remove the weight normalization forward hook
        model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
        quantized = tq.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

        # A lazy way to check that Modules are different
        assert str(quantized) != str(model), "Dynamic quantization did not modify the module."

        waveforms = torch.randn(batch_size, num_frames)
        lengths = torch.randint(
            low=0,
            high=num_frames,
            size=[
                batch_size,
            ],
        )

        ref_out, ref_len = quantized(waveforms, lengths)

        # Script
        scripted = torch_script(quantized)

        hyp_out, hyp_len = scripted(waveforms, lengths)

        self.assertEqual(hyp_out, ref_out)
        self.assertEqual(hyp_len, ref_len)

    @factory_funcs
    @skipIfNoQengine
    def test_quantize_torchscript(self, factory_func):
        """Quantized Wav2Vec2Model should be scriptable"""
        self._test_quantize_torchscript(factory_func(aux_num_out=32))
