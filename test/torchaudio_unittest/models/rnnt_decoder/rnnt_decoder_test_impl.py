import torch
from torchaudio.models import emformer_rnnt_model, RNNTBeamSearch
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script


class RNNTBeamSearchTestImpl(TestBaseMixin):
    def _get_input_config(self):
        model_config = self._get_model_config()
        return {
            "batch_size": 1,
            "max_input_length": 61,
            "num_symbols": model_config["num_symbols"],
            "input_dim": model_config["input_dim"],
            "right_context_length": model_config["right_context_length"],
            "segment_length": model_config["segment_length"],
        }

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
        return emformer_rnnt_model(**self._get_model_config()).to(device=self.device, dtype=self.dtype).eval()

    def test_torchscript_consistency_forward(self):
        r"""Verify that scripting RNNTBeamSearch does not change the behavior of method `forward`."""

        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        max_input_length = input_config["max_input_length"]
        right_context_length = input_config["right_context_length"]
        input_dim = input_config["input_dim"]
        num_symbols = input_config["num_symbols"]
        blank_idx = num_symbols - 1
        beam_width = 5

        input = torch.rand(batch_size, max_input_length + right_context_length, input_dim).to(
            device=self.device, dtype=self.dtype
        )
        lengths = torch.randint(1, max_input_length + 1, (batch_size,)).to(device=self.device, dtype=torch.int32)

        model = self._get_model()
        beam_search = RNNTBeamSearch(model, blank_idx)
        scripted = torch_script(beam_search)

        res = beam_search(input, lengths, beam_width)
        scripted_res = scripted(input, lengths, beam_width)

        self.assertEqual(res, scripted_res)

    def test_torchscript_consistency_infer(self):
        r"""Verify that scripting RNNTBeamSearch does not change the behavior of method `infer`."""

        input_config = self._get_input_config()
        segment_length = input_config["segment_length"]
        right_context_length = input_config["right_context_length"]
        input_dim = input_config["input_dim"]
        num_symbols = input_config["num_symbols"]
        blank_idx = num_symbols - 1
        beam_width = 5

        input = torch.rand(segment_length + right_context_length, input_dim).to(device=self.device, dtype=self.dtype)
        lengths = torch.randint(1, segment_length + right_context_length + 1, ()).to(
            device=self.device, dtype=torch.int32
        )

        model = self._get_model()

        state, hypo = None, None
        scripted_state, scripted_hypo = None, None
        for _ in range(2):
            beam_search = RNNTBeamSearch(model, blank_idx)
            scripted = torch_script(beam_search)

            res = beam_search.infer(input, lengths, beam_width, state=state, hypothesis=hypo)
            scripted_res = scripted.infer(input, lengths, beam_width, state=scripted_state, hypothesis=scripted_hypo)

            self.assertEqual(res, scripted_res)

            state = res[1]
            hypo = res[0]

            scripted_state = scripted_res[1]
            scripted_hypo = scripted_res[0]
