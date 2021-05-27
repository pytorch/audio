import io
import torch
import torch.nn.functional as F

from torchaudio.models.wav2vec2 import (
    wav2vec2_base,
    wav2vec2_large,
    wav2vec2_large_lv60k,
)
from torchaudio_unittest.common_utils import (
    TorchaudioTestCase,
    skipIfNoQengine,
    skipIfNoCuda,
)
from parameterized import parameterized

factory_funcs = parameterized.expand([
    (wav2vec2_base, ),
    (wav2vec2_large, ),
    (wav2vec2_large_lv60k, ),
])


class TestWav2Vec2Model(TorchaudioTestCase):
    def _smoke_test(self, device, dtype):
        model = wav2vec2_base(num_out=32)
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        torch.manual_seed(0)
        batch_size, num_frames = 3, 1024

        waveforms = torch.randn(
            batch_size, num_frames, device=device, dtype=dtype)
        lengths = torch.randint(
            low=0, high=num_frames, size=[batch_size, ], device=device)

        model(waveforms, lengths)

    @parameterized.expand([(torch.float32, ), (torch.float64, )])
    def test_cpu_smoke_test(self, dtype):
        self._smoke_test(torch.device('cpu'), dtype)

    @parameterized.expand([(torch.float32, ), (torch.float64, )])
    @skipIfNoCuda
    def test_cuda_smoke_test(self, dtype):
        self._smoke_test(torch.device('cuda'), dtype)

    @factory_funcs
    def test_feature_extractor_smoke_test(self, factory_func):
        """`extract_features` method does not fail"""
        batch_size, num_frames = 3, 1024

        model = factory_func(num_out=32).eval()

        torch.manual_seed(0)
        waveforms = torch.randn(batch_size, num_frames)
        lengths = torch.randint(low=0, high=num_frames, size=[batch_size, ])

        features, lengths = model.extract_features(waveforms, lengths)
        assert features.ndim == 3
        assert features.shape[0] == batch_size
        assert lengths.shape == torch.Size([batch_size])

    @factory_funcs
    def test_batch_consistency(self, factory_func):
        """Results from sigle process and batched process should be reasonably close
        """
        batch_size, max_frames = 5, 5 * 1024

        model = factory_func(num_out=32).eval()

        torch.manual_seed(0)
        waveforms = torch.randn(batch_size, max_frames)
        input_lengths = torch.tensor([i * 3200 for i in range(1, 6)])

        # Batch process with lengths
        batch_logits, output_lengths = model(waveforms, input_lengths)
        for i in range(batch_size):
            # Par-sample process without feeding length
            single_logit, _ = model(waveforms[i:i + 1, :input_lengths[i]], None)
            batch_logit = batch_logits[i:i + 1, :output_lengths[i]]

            # Convert to probability so that it's easier to interpretate the diff
            single_prob = F.softmax(single_logit, dim=2)
            batch_prob = F.softmax(batch_logit, dim=2)
            # We allow max atol=0.005 -> 0.5%
            self.assertEqual(single_prob, batch_prob, atol=0.005, rtol=0)

    @factory_funcs
    def test_zero_length(self, factory_func):
        """Passing zero length should not fail"""
        model = factory_func(num_out=32).eval()

        torch.manual_seed(0)
        batch_size = 3
        waveforms = torch.randn(batch_size, 1024)
        input_lengths = torch.zeros(batch_size)
        _, output_lengths = model(waveforms, input_lengths)
        self.assertEqual(torch.zeros_like(output_lengths), output_lengths)

    @factory_funcs
    def test_torchscript(self, factory_func):
        """Wav2Vec2Model should be scriptable"""
        batch_size, num_frames = 3, 1024

        model = factory_func(num_out=32).eval()

        torch.manual_seed(0)
        waveforms = torch.randn(batch_size, num_frames)
        lengths = torch.randint(low=0, high=num_frames, size=[batch_size, ])

        ref_out, ref_len = model(waveforms, lengths)

        # TODO: put this in a common method of Mixin class.
        # Script
        scripted = torch.jit.script(model)
        buffer_ = io.BytesIO()
        torch.jit.save(scripted, buffer_)
        buffer_.seek(0)
        scripted = torch.jit.load(buffer_)

        hyp_out, hyp_len = scripted(waveforms, lengths)

        self.assertEqual(hyp_out, ref_out)
        self.assertEqual(hyp_len, ref_len)

    @factory_funcs
    @skipIfNoQengine
    def test_quantize(self, factory_func):
        """Wav2Vec2Model should support basic quantization"""
        batch_size, num_frames = 3, 1024

        model = factory_func(num_out=32).eval()

        # Remove the weight normalization forward hook
        model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
        quantized = torch.quantization.quantize_dynamic(
            model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

        # A lazy way to check that Modules are different
        assert str(quantized) != str(model), "Dynamic quantization did not modify the module."

        torch.manual_seed(0)
        waveforms = torch.randn(batch_size, num_frames)
        lengths = torch.randint(low=0, high=num_frames, size=[batch_size, ])

        _, _ = quantized(waveforms, lengths)

    @factory_funcs
    @skipIfNoQengine
    def test_quantize_torchscript(self, factory_func):
        """Quantized Wav2Vec2Model should be scriptable"""
        print(torch.backends.quantized.supported_engines)
        batch_size, num_frames = 3, 1024

        model = factory_func(num_out=32).eval()

        # Remove the weight normalization forward hook
        model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
        quantized = torch.quantization.quantize_dynamic(
            model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

        # A lazy way to check that Modules are different
        assert str(quantized) != str(model), "Dynamic quantization did not modify the module."

        torch.manual_seed(0)
        waveforms = torch.randn(batch_size, num_frames)
        lengths = torch.randint(low=0, high=num_frames, size=[batch_size, ])

        ref_out, ref_len = quantized(waveforms, lengths)

        # Script
        scripted = torch.jit.script(quantized)
        buffer_ = io.BytesIO()
        torch.jit.save(scripted, buffer_)
        buffer_.seek(0)
        scripted = torch.jit.load(buffer_)

        hyp_out, hyp_len = scripted(waveforms, lengths)

        self.assertEqual(hyp_out, ref_out)
        self.assertEqual(hyp_len, ref_len)
