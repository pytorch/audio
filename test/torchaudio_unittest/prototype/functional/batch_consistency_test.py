import torch
import torchaudio.prototype.functional as F
from torchaudio_unittest.common_utils import nested_params, TorchaudioTestCase


class BatchConsistencyTest(TorchaudioTestCase):
    @nested_params(
        [F.convolve, F.fftconvolve],
        ["full", "valid", "same"],
    )
    def test_convolve(self, fn, mode):
        leading_dims = (2, 3)
        L_x, L_y = 89, 43
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        actual = fn(x, y, mode)
        expected = torch.stack(
            [
                torch.stack(
                    [fn(x[i, j].unsqueeze(0), y[i, j].unsqueeze(0), mode).squeeze(0) for j in range(leading_dims[1])]
                )
                for i in range(leading_dims[0])
            ]
        )

        self.assertEqual(expected, actual)

    def test_add_noise(self):
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device) * 10

        actual = F.add_noise(waveform, noise, snr, lengths)

        expected = []
        for i in range(leading_dims[0]):
            for j in range(leading_dims[1]):
                for k in range(leading_dims[2]):
                    expected.append(F.add_noise(waveform[i][j][k], noise[i][j][k], snr[i][j][k], lengths[i][j][k]))

        self.assertEqual(torch.stack(expected), actual.reshape(-1, L))

    def test_speed(self):
        B = 5
        orig_freq = 100
        factor = 0.8
        input_lengths = torch.randint(1, 1000, (B,), dtype=torch.int32)

        unbatched_input = [torch.ones((int(length),)) * 1.0 for length in input_lengths]
        batched_input = torch.nn.utils.rnn.pad_sequence(unbatched_input, batch_first=True)

        output, output_lengths = F.speed(batched_input, input_lengths, orig_freq=orig_freq, factor=factor)

        unbatched_output = []
        unbatched_output_lengths = []
        for idx in range(len(unbatched_input)):
            w, l = F.speed(unbatched_input[idx], input_lengths[idx], orig_freq=orig_freq, factor=factor)
            unbatched_output.append(w)
            unbatched_output_lengths.append(l)

        self.assertEqual(output_lengths, torch.stack(unbatched_output_lengths))
        for idx in range(len(unbatched_output)):
            w, l = output[idx], output_lengths[idx]
            self.assertEqual(unbatched_output[idx], w[:l])

    def test_preemphasis(self):
        waveform = torch.rand(3, 2, 100, device=self.device, dtype=self.dtype)
        coeff = 0.9
        actual = F.preemphasis(waveform, coeff=coeff)

        expected = []
        for i in range(waveform.size(0)):
            expected.append(F.preemphasis(waveform[i], coeff=coeff))

        self.assertEqual(torch.stack(expected), actual)

    def test_deemphasis(self):
        waveform = torch.rand(3, 2, 100, device=self.device, dtype=self.dtype)
        coeff = 0.9
        actual = F.deemphasis(waveform, coeff=coeff)

        expected = []
        for i in range(waveform.size(0)):
            expected.append(F.deemphasis(waveform[i], coeff=coeff))

        self.assertEqual(torch.stack(expected), actual)
