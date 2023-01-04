import os

import torch
import torchaudio.prototype.transforms as T
import torchaudio.transforms as transforms
from torchaudio_unittest.common_utils import nested_params, TorchaudioTestCase


class BatchConsistencyTest(TorchaudioTestCase):
    def assert_batch_consistency(self, transform, batch, *args, atol=1e-8, rtol=1e-5, seed=42, **kwargs):
        n = batch.size(0)

        # Compute items separately, then batch the result
        torch.random.manual_seed(seed)
        items_input = batch.clone()
        items_result = torch.stack([transform(items_input[i], *args, **kwargs) for i in range(n)])

        # Batch the input and run
        torch.random.manual_seed(seed)
        batch_input = batch.clone()
        batch_result = transform(batch_input, *args, **kwargs)

        self.assertEqual(items_input, batch_input, rtol=rtol, atol=atol)
        self.assertEqual(items_result, batch_result, rtol=rtol, atol=atol)

    @nested_params(
        [T.Convolve, T.FFTConvolve],
        ["full", "valid", "same"],
    )
    def test_Convolve(self, cls, mode):
        leading_dims = (2, 3)
        L_x, L_y = 89, 43
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        convolve = cls(mode=mode)
        actual = convolve(x, y)
        expected = torch.stack(
            [
                torch.stack(
                    [convolve(x[i, j].unsqueeze(0), y[i, j].unsqueeze(0)).squeeze(0) for j in range(leading_dims[1])]
                )
                for i in range(leading_dims[0])
            ]
        )

        self.assertEqual(expected, actual)

    def test_batch_BarkScale(self):
        specgram = torch.randn(3, 2, 201, 256)

        atol = 1e-6 if os.name == "nt" else 1e-8
        transform = T.BarkScale()

        self.assert_batch_consistency(transform, specgram, atol=atol)

    def test_batch_InverseBarkScale(self):
        n_barks = 32
        n_stft = 5
        bark_spec = torch.randn(3, 2, n_barks, 32) ** 2
        transform = transforms.InverseMelScale(n_stft, n_barks)

        # Because InverseBarkScale runs SGD on randomly initialized values so they do not yield
        # exactly same result. For this reason, tolerance is very relaxed here.
        self.assert_batch_consistency(transform, bark_spec, atol=1.0, rtol=1e-5)

    def test_Speed(self):
        B = 5
        orig_freq = 100
        factor = 0.8
        input_lengths = torch.randint(1, 1000, (B,), dtype=torch.int32)

        speed = T.Speed(orig_freq, factor)

        unbatched_input = [torch.ones((int(length),)) * 1.0 for length in input_lengths]
        batched_input = torch.nn.utils.rnn.pad_sequence(unbatched_input, batch_first=True)

        output, output_lengths = speed(batched_input, input_lengths)

        unbatched_output = []
        unbatched_output_lengths = []
        for idx in range(len(unbatched_input)):
            w, l = speed(unbatched_input[idx], input_lengths[idx])
            unbatched_output.append(w)
            unbatched_output_lengths.append(l)

        self.assertEqual(output_lengths, torch.stack(unbatched_output_lengths))
        for idx in range(len(unbatched_output)):
            w, l = output[idx], output_lengths[idx]
            self.assertEqual(unbatched_output[idx], w[:l])

    def test_SpeedPerturbation(self):
        B = 5
        orig_freq = 100
        factor = 0.8
        input_lengths = torch.randint(1, 1000, (B,), dtype=torch.int32)

        speed = T.SpeedPerturbation(orig_freq, [factor])

        unbatched_input = [torch.ones((int(length),)) * 1.0 for length in input_lengths]
        batched_input = torch.nn.utils.rnn.pad_sequence(unbatched_input, batch_first=True)

        output, output_lengths = speed(batched_input, input_lengths)

        unbatched_output = []
        unbatched_output_lengths = []
        for idx in range(len(unbatched_input)):
            w, l = speed(unbatched_input[idx], input_lengths[idx])
            unbatched_output.append(w)
            unbatched_output_lengths.append(l)

        self.assertEqual(output_lengths, torch.stack(unbatched_output_lengths))
        for idx in range(len(unbatched_output)):
            w, l = output[idx], output_lengths[idx]
            self.assertEqual(unbatched_output[idx], w[:l])

    def test_AddNoise(self):
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device) * 10

        add_noise = T.AddNoise()
        actual = add_noise(waveform, noise, lengths, snr)

        expected = []
        for i in range(leading_dims[0]):
            for j in range(leading_dims[1]):
                for k in range(leading_dims[2]):
                    expected.append(add_noise(waveform[i][j][k], noise[i][j][k], lengths[i][j][k], snr[i][j][k]))

        self.assertEqual(torch.stack(expected), actual.reshape(-1, L))

    def test_Preemphasis(self):
        waveform = torch.rand((3, 5, 2, 100), dtype=self.dtype, device=self.device)
        preemphasis = T.Preemphasis(coeff=0.97)
        actual = preemphasis(waveform)

        expected = []
        for i in range(waveform.size(0)):
            for j in range(waveform.size(1)):
                for k in range(waveform.size(2)):
                    expected.append(preemphasis(waveform[i][j][k]))

        self.assertEqual(torch.stack(expected), actual.reshape(-1, waveform.size(-1)))

    def test_Deemphasis(self):
        waveform = torch.rand((3, 5, 2, 100), dtype=self.dtype, device=self.device)
        deemphasis = T.Deemphasis(coeff=0.97)
        actual = deemphasis(waveform)

        expected = []
        for i in range(waveform.size(0)):
            for j in range(waveform.size(1)):
                for k in range(waveform.size(2)):
                    expected.append(deemphasis(waveform[i][j][k]))

        self.assertEqual(torch.stack(expected), actual.reshape(-1, waveform.size(-1)))
