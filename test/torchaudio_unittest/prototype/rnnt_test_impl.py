import torch
from torchaudio.prototype.models import conformer_rnnt_model
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script


class ConformerRNNTTestImpl(TestBaseMixin):
    def _get_input_config(self):
        model_config = self._get_model_config()
        max_input_length = 59
        return {
            "batch_size": 7,
            "max_input_length": max_input_length,
            "num_symbols": model_config["num_symbols"],
            "max_target_length": 45,
            "input_dim": model_config["input_dim"],
            "encoding_dim": model_config["encoding_dim"],
            "joiner_max_input_length": max_input_length // model_config["time_reduction_stride"],
            "time_reduction_stride": model_config["time_reduction_stride"],
        }

    def _get_model_config(self):
        return {
            "input_dim": 80,
            "num_symbols": 128,
            "encoding_dim": 64,
            "symbol_embedding_dim": 32,
            "num_lstm_layers": 2,
            "lstm_hidden_dim": 11,
            "lstm_layer_norm": True,
            "lstm_layer_norm_epsilon": 1e-5,
            "lstm_dropout": 0.3,
            "joiner_activation": "tanh",
            "time_reduction_stride": 4,
            "conformer_input_dim": 100,
            "conformer_ffn_dim": 33,
            "conformer_num_layers": 3,
            "conformer_num_heads": 4,
            "conformer_depthwise_conv_kernel_size": 31,
            "conformer_dropout": 0.1,
        }

    def _get_model(self):
        return conformer_rnnt_model(**self._get_model_config()).to(device=self.device, dtype=self.dtype).eval()

    def _get_transcriber_input(self):
        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        max_input_length = input_config["max_input_length"]
        input_dim = input_config["input_dim"]

        input = torch.rand(batch_size, max_input_length, input_dim).to(device=self.device, dtype=self.dtype)
        lengths = torch.full((batch_size,), max_input_length).to(device=self.device, dtype=torch.int32)
        return input, lengths

    def _get_predictor_input(self):
        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        num_symbols = input_config["num_symbols"]
        max_target_length = input_config["max_target_length"]

        input = torch.randint(0, num_symbols, (batch_size, max_target_length)).to(device=self.device, dtype=torch.int32)
        lengths = torch.full((batch_size,), max_target_length).to(device=self.device, dtype=torch.int32)
        return input, lengths

    def _get_joiner_input(self):
        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        joiner_max_input_length = input_config["joiner_max_input_length"]
        max_target_length = input_config["max_target_length"]
        input_dim = input_config["encoding_dim"]

        utterance_encodings = torch.rand(batch_size, joiner_max_input_length, input_dim).to(
            device=self.device, dtype=self.dtype
        )
        utterance_lengths = torch.randint(0, joiner_max_input_length + 1, (batch_size,)).to(
            device=self.device, dtype=torch.int32
        )
        target_encodings = torch.rand(batch_size, max_target_length, input_dim).to(device=self.device, dtype=self.dtype)
        target_lengths = torch.randint(0, max_target_length + 1, (batch_size,)).to(
            device=self.device, dtype=torch.int32
        )

        return utterance_encodings, utterance_lengths, target_encodings, target_lengths

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(31)

    def test_torchscript_consistency_forward(self):
        r"""Verify that scripting RNNT does not change the behavior of method `forward`."""
        inputs, input_lengths = self._get_transcriber_input()
        targets, target_lengths = self._get_predictor_input()

        rnnt = self._get_model()
        scripted = torch_script(rnnt).eval()

        ref_state, scripted_state = None, None
        for _ in range(2):
            ref_out, ref_input_lengths, ref_target_lengths, ref_state = rnnt(
                inputs, input_lengths, targets, target_lengths, ref_state
            )
            (
                scripted_out,
                scripted_input_lengths,
                scripted_target_lengths,
                scripted_state,
            ) = scripted(inputs, input_lengths, targets, target_lengths, scripted_state)

            self.assertEqual(ref_out, scripted_out, atol=1e-4, rtol=1e-5)
            self.assertEqual(ref_input_lengths, scripted_input_lengths, atol=1e-4, rtol=1e-5)
            self.assertEqual(ref_target_lengths, scripted_target_lengths, atol=1e-4, rtol=1e-5)
            self.assertEqual(ref_state, scripted_state, atol=1e-4, rtol=1e-5)

    def test_torchscript_consistency_transcribe(self):
        r"""Verify that scripting RNNT does not change the behavior of method `transcribe`."""
        input, lengths = self._get_transcriber_input()

        rnnt = self._get_model()
        scripted = torch_script(rnnt)

        ref_out, ref_lengths = rnnt.transcribe(input, lengths)
        scripted_out, scripted_lengths = scripted.transcribe(input, lengths)

        self.assertEqual(ref_out, scripted_out)
        self.assertEqual(ref_lengths, scripted_lengths)

    def test_torchscript_consistency_predict(self):
        r"""Verify that scripting RNNT does not change the behavior of method `predict`."""
        input, lengths = self._get_predictor_input()

        rnnt = self._get_model()
        scripted = torch_script(rnnt)

        ref_state, scripted_state = None, None
        for _ in range(2):
            ref_out, ref_lengths, ref_state = rnnt.predict(input, lengths, ref_state)
            scripted_out, scripted_lengths, scripted_state = scripted.predict(input, lengths, scripted_state)
            self.assertEqual(ref_out, scripted_out)
            self.assertEqual(ref_lengths, scripted_lengths)
            self.assertEqual(ref_state, scripted_state)

    def test_torchscript_consistency_join(self):
        r"""Verify that scripting RNNT does not change the behavior of method `join`."""
        (
            utterance_encodings,
            utterance_lengths,
            target_encodings,
            target_lengths,
        ) = self._get_joiner_input()

        rnnt = self._get_model()
        scripted = torch_script(rnnt)

        ref_out, ref_src_lengths, ref_tgt_lengths = rnnt.join(
            utterance_encodings, utterance_lengths, target_encodings, target_lengths
        )
        scripted_out, scripted_src_lengths, scripted_tgt_lengths = scripted.join(
            utterance_encodings, utterance_lengths, target_encodings, target_lengths
        )
        self.assertEqual(ref_out, scripted_out)
        self.assertEqual(ref_src_lengths, scripted_src_lengths)
        self.assertEqual(ref_tgt_lengths, scripted_tgt_lengths)

    def test_output_shape_forward(self):
        r"""Check that method `forward` produces correctly-shaped outputs."""
        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        joiner_max_input_length = input_config["joiner_max_input_length"]
        max_target_length = input_config["max_target_length"]
        num_symbols = input_config["num_symbols"]

        inputs, input_lengths = self._get_transcriber_input()
        targets, target_lengths = self._get_predictor_input()

        rnnt = self._get_model()

        state = None
        for _ in range(2):
            out, out_lengths, target_lengths, state = rnnt(inputs, input_lengths, targets, target_lengths, state)
            self.assertEqual(
                (batch_size, joiner_max_input_length, max_target_length, num_symbols),
                out.shape,
            )
            self.assertEqual((batch_size,), out_lengths.shape)
            self.assertEqual((batch_size,), target_lengths.shape)

    def test_output_shape_transcribe(self):
        r"""Check that method `transcribe` produces correctly-shaped outputs."""
        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        max_input_length = input_config["max_input_length"]

        input, lengths = self._get_transcriber_input()

        model_config = self._get_model_config()
        encoding_dim = model_config["encoding_dim"]
        time_reduction_stride = model_config["time_reduction_stride"]
        rnnt = self._get_model()

        out, out_lengths = rnnt.transcribe(input, lengths)
        self.assertEqual(
            (batch_size, max_input_length // time_reduction_stride, encoding_dim),
            out.shape,
        )
        self.assertEqual((batch_size,), out_lengths.shape)

    def test_output_shape_predict(self):
        r"""Check that method `predict` produces correctly-shaped outputs."""
        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        max_target_length = input_config["max_target_length"]

        model_config = self._get_model_config()
        encoding_dim = model_config["encoding_dim"]
        input, lengths = self._get_predictor_input()

        rnnt = self._get_model()

        state = None
        for _ in range(2):
            out, out_lengths, state = rnnt.predict(input, lengths, state)
            self.assertEqual((batch_size, max_target_length, encoding_dim), out.shape)
            self.assertEqual((batch_size,), out_lengths.shape)

    def test_output_shape_join(self):
        r"""Check that method `join` produces correctly-shaped outputs."""
        input_config = self._get_input_config()
        batch_size = input_config["batch_size"]
        joiner_max_input_length = input_config["joiner_max_input_length"]
        max_target_length = input_config["max_target_length"]
        num_symbols = input_config["num_symbols"]

        (
            utterance_encodings,
            utterance_lengths,
            target_encodings,
            target_lengths,
        ) = self._get_joiner_input()

        rnnt = self._get_model()

        out, src_lengths, tgt_lengths = rnnt.join(
            utterance_encodings, utterance_lengths, target_encodings, target_lengths
        )
        self.assertEqual(
            (batch_size, joiner_max_input_length, max_target_length, num_symbols),
            out.shape,
        )
        self.assertEqual((batch_size,), src_lengths.shape)
        self.assertEqual((batch_size,), tgt_lengths.shape)
