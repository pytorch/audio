import torch
from parameterized import parameterized
from torchaudio.prototype.models import squim_objective_base
from torchaudio_unittest.common_utils import skipIfNoCuda, torch_script, TorchaudioTestCase


class TestSQUIM(TorchaudioTestCase):
    def _smoke_test_objective(self, model, device, dtype):
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        batch_size, num_frames = 3, 16000
        waveforms = torch.randn(batch_size, num_frames, device=device, dtype=dtype)

        model(waveforms)

    @parameterized.expand([(torch.float32,), (torch.float64,)])
    def test_cpu_smoke_test(self, dtype):
        model = squim_objective_base()
        self._smoke_test_objective(model, torch.device("cpu"), dtype)

    @parameterized.expand([(torch.float32,), (torch.float64,)])
    @skipIfNoCuda
    def test_cuda_smoke_test(self, dtype):
        model = squim_objective_base()
        self._smoke_test_objective(model, torch.device("cuda"), dtype)

    def test_torchscript_consistency(self):
        model = squim_objective_base()
        model.eval()

        batch_size, num_frames = 3, 16000
        waveforms = torch.randn(batch_size, num_frames)

        ref_scores = model(waveforms)

        scripted = torch_script(model)
        hyp_scores = scripted(waveforms)

        self.assertEqual(len(hyp_scores), len(ref_scores))
        for i in range(len(ref_scores)):
            self.assertEqual(hyp_scores[i], ref_scores[i])
