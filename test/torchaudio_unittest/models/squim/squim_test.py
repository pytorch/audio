import torch
from parameterized import parameterized
from torchaudio.models import squim_objective_base, squim_subjective_base
from torchaudio_unittest.common_utils import skipIfNoCuda, torch_script, TorchaudioTestCase


class TestSquimObjective(TorchaudioTestCase):
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

    def test_batch_consistency(self):
        model = squim_objective_base()
        model.eval()

        batch_size, num_frames = 3, 16000
        waveforms = torch.randn(batch_size, num_frames)

        ref_scores = model(waveforms)
        hyp_scores = [torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(batch_size)]
        for i in range(batch_size):
            scores = model(waveforms[i : i + 1])
            for j in range(3):
                hyp_scores[j][i] = scores[j]
        self.assertEqual(len(hyp_scores), len(ref_scores))
        for i in range(len(ref_scores)):
            self.assertEqual(hyp_scores[i], ref_scores[i])

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


class TestSquimSubjective(TorchaudioTestCase):
    def _smoke_test_subjective(self, model, device, dtype):
        model = model.to(device=device, dtype=dtype)
        model = model.eval()

        batch_size, num_frames = 3, 16000
        waveforms = torch.randn(batch_size, num_frames, device=device, dtype=dtype)
        reference = torch.randn(batch_size, num_frames, device=device, dtype=dtype)

        model(waveforms, reference)

    @parameterized.expand([(torch.float32,), (torch.float64,)])
    def test_cpu_smoke_test(self, dtype):
        model = squim_subjective_base()
        self._smoke_test_subjective(model, torch.device("cpu"), dtype)

    @parameterized.expand([(torch.float32,), (torch.float64,)])
    @skipIfNoCuda
    def test_cuda_smoke_test(self, dtype):
        model = squim_subjective_base()
        self._smoke_test_subjective(model, torch.device("cuda"), dtype)

    def test_batch_consistency(self):
        model = squim_subjective_base()
        model.eval()

        batch_size, num_frames = 3, 16000
        waveforms = torch.randn(batch_size, num_frames)
        reference = torch.randn(batch_size, num_frames)

        ref_scores = model(waveforms, reference)
        hyp_scores = []
        for i in range(batch_size):
            scores = model(waveforms[i : i + 1], reference[i : i + 1])
            hyp_scores.append(scores)
        hyp_scores = torch.tensor(hyp_scores)
        self.assertEqual(hyp_scores, ref_scores)

    def test_torchscript_consistency(self):
        model = squim_subjective_base()
        model.eval()

        batch_size, num_frames = 3, 16000
        waveforms = torch.randn(batch_size, num_frames)
        reference = torch.randn(batch_size, num_frames)

        ref_scores = model(waveforms, reference)

        scripted = torch_script(model)
        hyp_scores = scripted(waveforms, reference)

        self.assertEqual(hyp_scores, ref_scores)
