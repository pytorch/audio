import torch

from torchaudio.prototype import RNNTBeamSearch, emformer_rnnt_model
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script


class RNNTBeamSearchTestImpl(TestBaseMixin):
    def _get_model_config(self):
        return {
            "input_dim": 80,
            "encoding_dim": 128,
            "num_symbols": 256,
            "segment_length": 16,
            "right_context_length": 4,
            "time_reduction_input_dim": 128,
            "time_reduction_stride": 4,
            "transformer_num_heads": 4,
            "transformer_ffn_dim": 64,
            "transformer_num_layers": 3,
            "transformer_dropout": 0.0,
            "transformer_activation": "relu",
            "transformer_left_context_length": 30,
            "transformer_max_memory_size": 0,
            "transformer_weight_init_scale_strategy": "depthwise",
            "transformer_tanh_on_mem": True,
            "symbol_embedding_dim": 64,
            "num_lstm_layers": 2,
            "lstm_layer_norm": True,
            "lstm_layer_norm_epsilon": 1e-3,
            "lstm_dropout": 0.0,
        }

    def _get_model(self):
        return (
            emformer_rnnt_model(**self._get_model_config())
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )

    def test_torchscript_consistency(self):
        torch.random.manual_seed(31)

        batch_size = 1
        max_input_length = 50
        right_context_length = 4
        input_dim = 80

        input = torch.rand(
            batch_size, max_input_length + right_context_length, input_dim
        ).to(device=self.device, dtype=self.dtype)
        lengths = torch.randint(1, max_input_length + 1, (batch_size,)).to(
            device=self.device, dtype=torch.int32
        )

        model = self._get_model()
        beam_search = RNNTBeamSearch(model, 255)
        scripted = torch_script(beam_search)

        res = beam_search(input, lengths, 5)
        scripted_res = scripted(input, lengths, 5)

        self.assertEqual(res, scripted_res)
